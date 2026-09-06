# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from math import prod
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.models import yoco


def configs():
    model = SimpleNamespace(
        hidden_size=3072,
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=3840,
        moe_latent_dim=1024,
        swiglu_limit=10.0,
    )
    runtime = SimpleNamespace(
        additional_config={"yoco_fast_standalone_flashinfer_moe": True},
        kv_transfer_config=None,
        parallel_config=SimpleNamespace(data_parallel_size=1, pipeline_parallel_size=1),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=True),
        kernel_config=SimpleNamespace(
            moe_backend="triton", enable_flashinfer_autotune=False
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=256, max_num_batched_tokens=8192),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=256),
    )
    return model, runtime


def select(model, runtime, **kwargs):
    values = dict(
        execution_mode="fast",
        quant_config=None,
        tp_size=1,
        config=model,
        vllm_config=runtime,
    )
    values.update(kwargs)
    return yoco._select_yoco_fast_moe_backend(**values)


def test_standalone_selection_and_decode_boundary(monkeypatch):
    import vllm.utils.flashinfer as fi

    monkeypatch.setattr(
        yoco,
        "current_platform",
        SimpleNamespace(get_device_capability=lambda: SimpleNamespace(major=10)),
    )
    monkeypatch.setattr(fi, "has_flashinfer_cutlass_fused_moe", lambda: True)
    model, runtime = configs()
    assert select(model, runtime) == "flashinfer_cutlass"
    runtime.additional_config.clear()
    assert select(model, runtime) == "flashinfer_cutlass"
    assert not runtime.kernel_config.enable_flashinfer_autotune
    assert yoco._yoco_standalone_prefill_min_tokens(runtime) == 1024
    runtime.compilation_config.max_cudagraph_capture_size = 2048
    assert yoco._yoco_standalone_prefill_min_tokens(runtime) == 2049
    runtime.scheduler_config.max_num_batched_tokens = 2048
    assert select(model, runtime) is None


@pytest.mark.parametrize(
    "case",
    [
        "align",
        "tp",
        "dp",
        "cp",
        "fp16",
        "quant",
        "optout",
        "tune",
        "no_fast_prefill",
        "other_model",
        "unavailable",
    ],
)
def test_standalone_gates(monkeypatch, case):
    import vllm.utils.flashinfer as fi

    monkeypatch.setattr(
        yoco,
        "current_platform",
        SimpleNamespace(get_device_capability=lambda: SimpleNamespace(major=10)),
    )
    monkeypatch.setattr(
        fi, "has_flashinfer_cutlass_fused_moe", lambda: case != "unavailable"
    )
    model, runtime = configs()
    extra: dict[str, object] = {}
    if case == "align":
        extra["execution_mode"] = "align"
    elif case == "tp":
        extra["tp_size"] = 4
    elif case == "dp":
        runtime.parallel_config.data_parallel_size = 2
    elif case == "cp":
        runtime.parallel_config.prefill_context_parallel_size = 2
    elif case == "fp16":
        runtime.model_config.dtype = torch.float16
    elif case == "quant":
        extra["quant_config"] = object()
    elif case == "optout":
        runtime.additional_config["yoco_fast_standalone_flashinfer_moe"] = False
    elif case == "tune":
        runtime.kernel_config.enable_flashinfer_autotune = True
    elif case == "no_fast_prefill":
        runtime.cache_config.kv_sharing_fast_prefill = False
    elif case == "other_model":
        model.moe_intermediate_size = 1024
    assert select(model, runtime, **extra) is None


def fake_experts(max_tokens):
    return SimpleNamespace(
        yoco_triton_fallback_max_tokens=max_tokens,
        quant_config=SimpleNamespace(weight_quant_dtype=None),
        quant_dtype=None,
        out_dtype=torch.bfloat16,
        adjust_N_for_activation=lambda n, activation: n // 2,
    )


def workspace(experts, rows):
    return FlashInferExperts.workspace_shapes(
        experts, rows, 7680, 1024, 8, 128, 128, None, MoEActivation.SILU
    )


def test_shared_workspace_covers_smaller_graphs():
    experts = fake_experts(1023)
    profile = workspace(experts, 8192)
    for rows in [1, 64, 256, 1023]:
        w13, w2, output = workspace(experts, rows)
        assert output == (rows, 1024)
        assert prod(w13) >= rows * 8 * 3840
        assert prod(w2) >= rows * 8 * 7680
        assert prod(profile[0]) >= prod(w13)
        assert prod(profile[1]) >= prod(w2)
    # Profiling must not reserve a full 8192-row Triton activation buffer.
    assert prod(profile[1]) < 8192 * 8 * 7680
    assert workspace(fake_experts(1), 64) == ((64, 1024), (0,), (64, 1024))
    previous_bytes = 0
    for rows in range(1, 8193):
        shapes = workspace(experts, rows)
        # WorkspaceManager aligns each of its simultaneous buffers to 256 B.
        allocated = sum(((prod(shape) * 2 + 255) // 256) * 256 for shape in shapes[:2])
        assert allocated >= previous_bytes, rows
        previous_bytes = allocated


def test_large_fallback_borrows_workspace_and_preserves_fast_policy(monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts import triton_moe

    calls = []
    fallback = SimpleNamespace(apply=lambda *args: calls.append(args))
    monkeypatch.setattr(triton_moe, "TritonExperts", lambda *args: fallback)
    experts = fake_experts(1023)
    experts.num_experts = 128
    experts.w1_bias = experts.w2_bias = None
    experts.moe_config = object()
    experts._yoco_triton_fallback = None
    experts._yoco_triton_workspace13 = experts._yoco_triton_workspace2 = None
    experts.swiglu_limit = 10.0
    experts.yoco_fast_w13_config = experts.yoco_separate_w2_config = (
        experts.yoco_fast_moe_sum
    ) = True
    output = torch.empty(64, 64, dtype=torch.bfloat16)
    w1 = torch.empty(128, 128, 64, dtype=torch.bfloat16)
    w2 = torch.empty(128, 64, 64, dtype=torch.bfloat16)
    work13 = torch.empty(64 * 8 * 64, dtype=torch.bfloat16)
    work2 = torch.empty(64 * 8 * 128, dtype=torch.bfloat16)
    FlashInferExperts.apply(
        experts,
        output,
        output,
        w1,
        w2,
        torch.empty(64, 8),
        torch.zeros(64, 8, dtype=torch.int32),
        MoEActivation.SILU,
        128,
        None,
        None,
        None,
        work13,
        work2,
        None,
        False,
    )
    assert calls[0][11] is work13 and calls[0][12] is work2
    assert experts._yoco_triton_workspace13 is None
    assert experts._yoco_triton_workspace2 is None
    assert fallback.yoco_swapped_w13
    assert fallback.yoco_fast_w13_config and fallback.yoco_separate_w2_config
    assert fallback.yoco_fast_moe_sum


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [17, 1024])
def test_hybrid_clamp_and_aliased_workspace_against_reference(monkeypatch, rows):
    import vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe as fi
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG

    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires B200")
    monkeypatch.setattr(
        fi,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            compilation_config=SimpleNamespace(max_cudagraph_capture_size=256)
        ),
    )
    torch.manual_seed(20260905)
    experts, hidden, ffn, top_k = 128, 128, 128, 8
    config = make_dummy_moe_config(experts, top_k, hidden, ffn)
    hybrid = FlashInferExperts(config, FUSED_MOE_UNQUANTIZED_CONFIG)
    hybrid.yoco_triton_fallback_max_tokens = 1023
    hybrid.swiglu_limit = 0.5
    hybrid.yoco_fast_w13_config = hybrid.yoco_separate_w2_config = (
        hybrid.yoco_fast_moe_sum
    ) = True
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16) * 3
    w13 = (
        torch.randn(experts, 2 * ffn, hidden, device="cuda", dtype=torch.bfloat16)
        * 0.05
    )
    w2 = torch.randn(experts, hidden, ffn, device="cuda", dtype=torch.bfloat16) * 0.05
    swapped = torch.cat((w13[:, ffn:], w13[:, :ffn]), dim=1).contiguous()
    routing = torch.randn(rows, experts, device="cuda")
    values, ids = routing.topk(top_k, dim=1)
    weights = values.softmax(-1)
    shapes = hybrid.workspace_shapes(
        rows, 2 * ffn, hidden, top_k, experts, experts, None, MoEActivation.SILU
    )
    common = torch.full(
        (max(prod(shapes[0]), rows * hidden),),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )
    work13 = common[: prod(shapes[0])].view(shapes[0])
    output = common[: rows * hidden].view(rows, hidden)
    work2 = torch.empty(shapes[1], device="cuda", dtype=torch.bfloat16)
    hybrid.apply(
        output,
        x,
        swapped,
        w2,
        weights,
        ids.int(),
        MoEActivation.SILU,
        experts,
        None,
        None,
        None,
        work13,
        work2,
        None,
        False,
    )
    assert torch.isfinite(output).all()
    routed = torch.zeros(rows, top_k, hidden, device="cuda", dtype=torch.bfloat16)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for expert in range(experts):
            token, route = torch.where(ids == expert)
            if token.numel() == 0:
                continue
            pre = torch.nn.functional.linear(
                x[token].float(), w13[expert].float()
            ).bfloat16()
            gate, up = pre.float().chunk(2, -1)
            act = (
                torch.nn.functional.silu(gate.clamp(max=0.5)) * up.clamp(-0.5, 0.5)
            ).bfloat16()
            out = torch.nn.functional.linear(act.float(), w2[expert].float())
            routed[token, route] = (out * weights[token, route, None]).bfloat16()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    reference = routed.float().sum(1).bfloat16()
    relative = (
        output.float() - reference.float()
    ).square().mean().sqrt() / reference.float().square().mean().sqrt()
    assert relative < 0.015, relative.item()
