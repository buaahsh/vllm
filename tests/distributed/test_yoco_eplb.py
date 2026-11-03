# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise YOCO's real MoE block across an EPLB weight migration."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.parallel_state import ensure_model_parallel_initialized
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.yoco_moe import YOCOMoE
from vllm.model_executor.models.yoco import YOCOForCausalLM
from vllm.model_executor.models.yoco_weights import _YocoWeightLoader
from vllm.v1.worker.workspace import init_workspace_manager

from .eplb_utils import distributed_run, set_env_vars_and_device


@torch.no_grad()
def _run_yoco_eplb(env, world_size, use_async):
    set_env_vars_and_device(env)
    torch.set_default_dtype(torch.bfloat16)
    config = VllmConfig()
    parallel = config.parallel_config
    parallel.tensor_parallel_size = world_size
    parallel.enable_expert_parallel = True
    parallel.enable_eplb = True
    parallel.disable_custom_all_reduce = True
    parallel.eplb_config.num_redundant_experts = 2
    parallel.eplb_config.use_async = use_async
    parallel.eplb_config.communicator = "torch_gloo"
    parallel.eplb_config.window_size = 1
    parallel.eplb_config.step_interval = 1000
    config.kernel_config.moe_backend = "triton"
    hf_config = SimpleNamespace(
        hidden_size=128,
        num_hidden_layers=1,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        moe_latent_dim=64,
        moe_latent_norm=True,
        shared_expert_intermediate_size=128,
        swiglu_limit=10.0,
        router_weights_normalized=True,
        rms_norm_eps=1e-6,
    )
    with set_current_vllm_config(config):
        ensure_model_parallel_initialized(world_size, 1)
        device = torch.device("cuda", torch.accelerator.current_device_index())
        init_workspace_manager(device)
        with torch.device("cuda"):
            moe = YOCOMoE(hf_config, None, "model.layers.0.mlp")
        for name, p in moe.named_parameters():
            if "experts.routed_experts" not in name:
                p.normal_(0, 0.02)
        # Route all rows to initially non-redundant experts 2 and 3.
        moe.gate.weight.zero_()
        moe.gate.weight[2].fill_(0.02)
        moe.gate.weight[3].fill_(0.01)
        moe.initialize_router_weight_cache()
        model = YOCOForCausalLM.__new__(YOCOForCausalLM)
        nn.Module.__init__(model)
        model.config = hf_config
        model.model = nn.Module()
        layer = nn.Module()
        layer.mlp = moe
        model.model.layers = nn.ModuleList([layer])
        model._init_moe_metadata()
        loader = _YocoWeightLoader(model)
        w13 = torch.randn(4, 256, 64) * 0.02
        w2 = torch.randn(4, 64, 128) * 0.02
        loader.load_one("model.layers.0.mlp.experts.w13_weight", w13)
        loader.load_one("model.layers.0.mlp.experts.w2_weight", w2)
        routed = moe.experts.routed_experts
        routed.quant_method.process_weights_after_loading(routed)
        model_config = SimpleNamespace(
            model="yoco-eplb-test", compute_hash=lambda: "test"
        )
        state = EplbState(parallel, device)
        state.add_model(model, model_config)
        model_state = state.model_states["test"]
        old_map = model_state.physical_to_logical_map.clone()
        non_expert = {
            name: p.clone()
            for name, p in moe.named_parameters()
            if "experts.routed_experts" not in name
        }
        addresses = [w.data_ptr() for w in model.expert_weights[0]]
        x = torch.rand(16, 128, device="cuda")
        state.should_record_tensor.fill_(True)
        state.prepare_forward(model_config, x.shape[0])

        def forward():
            with set_forward_context(None, config, num_tokens=x.shape[0]):
                return moe(x).clone()

        baseline = forward()
        assert torch.isfinite(baseline).all()
        assert model_state.expert_load_pass.sum().item() == x.shape[0] * 2
        model_state.expert_load_window[0].copy_(model_state.expert_load_pass)
        if use_async:
            state.start_async_loop()
        state.rearrange()
        for _ in range(100):
            result = forward()
            torch.testing.assert_close(result, baseline, rtol=0.02, atol=2e-4)
            state.step(is_dummy=True)
            if not torch.equal(model_state.physical_to_logical_map, old_map):
                break
        else:
            raise AssertionError("EPLB never installed a new expert layout")
        torch.testing.assert_close(forward(), baseline, rtol=0.02, atol=2e-4)
        assert model_state.logical_replica_count[0, 2:].sum().item() > 2
        assert [w.data_ptr() for w in model.expert_weights[0]] == addresses
        for name, p in moe.named_parameters():
            if name in non_expert:
                torch.testing.assert_close(p, non_expert[name], rtol=0, atol=0)
        # Verify the migrated tensors against the original logical checkpoint,
        # including replicas that were not present on this rank initially.
        rank = torch.distributed.get_rank()
        for local in range(3):
            logical = model_state.physical_to_logical_map[0, rank * 3 + local].item()
            torch.testing.assert_close(routed.w13_weight[local], w13[logical].cuda())
            torch.testing.assert_close(routed.w2_weight[local], w2[logical].cuda())
        if use_async:
            state.drain_async()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2, reason="requires two CUDA GPUs"
)
@pytest.mark.parametrize("use_async", [False, True])
def test_yoco_output_survives_eplb_migration(monkeypatch, use_async):
    monkeypatch.setenv("VLLM_YOCO_BF16_ROUTER", "0")
    monkeypatch.setenv("VLLM_YOCO_MEGA_GATE", "0")
    distributed_run(_run_yoco_eplb, 2, use_async)
