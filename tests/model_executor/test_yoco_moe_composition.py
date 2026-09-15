# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO moe composition regression tests."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.experts.yoco_triton import (
    yoco_weighted_swiglu,
)
from vllm.model_executor.layers.yoco_moe import (
    YOCOCombinedOutputTransform,
    YOCOLatentInputTransform,
    YOCOLatentOutputTransform,
    YOCOMoE,
)
from vllm.model_executor.layers.yoco_ops.norm import RMSNorm


def test_yoco_latent_projections_follow_fast_quantization(monkeypatch) -> None:
    """Fast passes linear precision to both latent GEMMs; Align stays BF16."""

    import vllm.model_executor.layers.yoco_moe as yoco_module

    class FakeLinear(torch.nn.Module):
        def __init__(self, *args, quant_config=None, **kwargs) -> None:
            super().__init__()
            self.quant_config = quant_config

        def set_out_dtype(self, dtype: torch.dtype) -> None:
            self.out_dtype = dtype

    class FakeFusedMoE(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.quant_config = kwargs["quant_config"]
            self.use_tuned_config = kwargs["use_tuned_config"]
            self.apply_router_weight_before_w2 = kwargs["apply_router_weight_before_w2"]
            self.yoco_policy = kwargs["yoco_policy"]
            self.combined_output_transform = kwargs["combined_output_transform"]

    monkeypatch.setattr(yoco_module, "GateLinear", FakeLinear)
    monkeypatch.setattr(yoco_module, "ReplicatedLinear", FakeLinear)
    monkeypatch.setattr(yoco_module, "YOCOSharedExperts", FakeLinear)
    monkeypatch.setattr(yoco_module, "FusedMoEFactory", FakeFusedMoE)
    monkeypatch.setattr(yoco_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        yoco_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(kernel_config=SimpleNamespace(moe_backend="triton")),
    )
    monkeypatch.setattr(
        yoco_module,
        "RMSNorm",
        lambda *args, **kwargs: torch.nn.Identity(),
    )

    config = type(
        "Config",
        (),
        {
            "hidden_size": 3072,
            "num_hidden_layers": 3,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 1024,
            "moe_latent_dim": 1024,
            "moe_latent_norm": True,
            "shared_expert_intermediate_size": 1024,
            "swiglu_limit": 10.0,
            "router_weights_normalized": True,
            "rms_norm_eps": 1e-6,
        },
    )()
    online_quant_config = object()

    module = YOCOMoE(
        config,
        quant_config=online_quant_config,
        prefix="model.layers.0.mlp",
    )

    assert module.fc1_latent_proj.quant_config is online_quant_config
    assert module.fc2_latent_proj.quant_config is online_quant_config
    assert module.shared_gate.quant_config is None
    assert module.shared_experts.quant_config is online_quant_config
    assert module.experts.quant_config is online_quant_config
    assert module.experts.use_tuned_config
    assert module.experts.apply_router_weight_before_w2
    assert not module.experts.yoco_policy.align_weighted_swiglu
    assert not module.experts.yoco_policy.align_deep_gemm_w2
    assert module.experts.yoco_policy.separate_w2_config
    assert module.experts.yoco_policy.fast_w13_config
    assert module.experts.yoco_policy.triton_fallback_max_tokens == 1
    assert not module.experts.yoco_policy.align_moe_sum
    assert module.experts.yoco_policy.fast_moe_sum
    assert module.experts.combined_output_transform.execution_mode == "fast"

    align_module = YOCOMoE(
        config,
        quant_config=online_quant_config,
        prefix="model.layers.1.mlp",
        execution_mode="align",
    )
    assert align_module.fc1_latent_proj.quant_config is None
    assert align_module.fc2_latent_proj.quant_config is None
    assert not align_module.experts.use_tuned_config
    assert align_module.experts.apply_router_weight_before_w2
    assert align_module.experts.yoco_policy.align_weighted_swiglu
    assert not align_module.experts.yoco_policy.align_deep_gemm_w2
    assert not align_module.experts.yoco_policy.separate_w2_config
    assert not align_module.experts.yoco_policy.fast_w13_config
    assert align_module.experts.yoco_policy.triton_fallback_max_tokens == 0
    assert align_module.experts.yoco_policy.align_moe_sum
    assert not align_module.experts.yoco_policy.fast_moe_sum
    assert align_module.experts.combined_output_transform.execution_mode == "align"

    fast_bf16_module = YOCOMoE(
        config,
        quant_config=None,
        prefix="model.layers.2.mlp",
        execution_mode="fast",
    )
    assert fast_bf16_module.fc1_latent_proj.quant_config is None
    assert fast_bf16_module.fc2_latent_proj.quant_config is None
    assert fast_bf16_module.experts.apply_router_weight_before_w2
    assert not fast_bf16_module.experts.yoco_policy.align_weighted_swiglu
    assert not fast_bf16_module.experts.yoco_policy.align_deep_gemm_w2
    assert fast_bf16_module.experts.yoco_policy.separate_w2_config
    assert fast_bf16_module.experts.yoco_policy.fast_w13_config
    assert fast_bf16_module.experts.yoco_policy.triton_fallback_max_tokens == 1
    assert not fast_bf16_module.experts.yoco_policy.align_moe_sum
    assert fast_bf16_module.experts.yoco_policy.fast_moe_sum


def test_yoco_align_shared_expert_restores_separate_gemms(monkeypatch) -> None:
    import vllm.model_executor.layers.yoco_moe as yoco_module

    projection_calls = 0
    observed: dict[str, torch.Tensor | float] = {}

    class FakeMergedLinear(torch.nn.Module):
        def __init__(self, input_size, output_sizes, **kwargs) -> None:
            super().__init__()
            del kwargs
            self.weight = torch.nn.Parameter(
                torch.empty(sum(output_sizes), input_size, dtype=torch.bfloat16)
            )

        def forward(self, x: torch.Tensor):
            nonlocal projection_calls
            projection_calls += 1
            return F.linear(x, self.weight), None

    class FakeRowLinear(torch.nn.Module):
        def __init__(self, input_size, output_size, **kwargs) -> None:
            super().__init__()
            del kwargs
            self.weight = torch.nn.Parameter(
                torch.empty(output_size, input_size, dtype=torch.bfloat16)
            )

        def forward(self, x: torch.Tensor):
            return F.linear(x, self.weight), None

    def record_swiglu(
        up: torch.Tensor,
        gate: torch.Tensor,
        limit: float,
    ) -> torch.Tensor:
        observed["up"] = up.detach().clone()
        observed["gate"] = gate.detach().clone()
        observed["limit"] = limit
        gate_fp32 = gate.float().clamp(max=limit)
        up_fp32 = up.float().clamp(min=-limit, max=limit)
        return (up_fp32 * F.silu(gate_fp32)).to(up.dtype)

    monkeypatch.setattr(yoco_module, "MergedColumnParallelLinear", FakeMergedLinear)
    monkeypatch.setattr(yoco_module, "RowParallelLinear", FakeRowLinear)
    monkeypatch.setattr(yoco_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        yoco_module,
        "SiluAndMulWithClampFP32",
        lambda *args, **kwargs: torch.nn.Identity(),
    )
    monkeypatch.setattr(
        yoco_module,
        "_yoco_align_shared_expert_swiglu",
        record_swiglu,
    )

    module = yoco_module.YOCOSharedExperts(
        hidden_size=8,
        intermediate_size=4,
        quant_config=None,
        reduce_results=False,
        prefix="model.layers.0.mlp.shared_experts",
        swiglu_limit=10.0,
        execution_mode="align",
    )
    gate_weight = torch.arange(32, dtype=torch.bfloat16).view(4, 8) / 32
    up_weight = -torch.arange(32, dtype=torch.bfloat16).view(4, 8) / 48
    module.gate_up_proj.weight.data.copy_(torch.cat((gate_weight, up_weight)))
    module.down_proj.weight.data.copy_(
        torch.arange(32, dtype=torch.bfloat16).view(8, 4) / 64
    )
    hidden = torch.arange(16, dtype=torch.bfloat16).view(2, 8) / 16

    actual = module(hidden)
    expected_up = F.linear(hidden, up_weight)
    expected_gate = F.linear(hidden, gate_weight)
    expected_activated = (
        expected_up.float().clamp(min=-10.0, max=10.0)
        * F.silu(expected_gate.float().clamp(max=10.0))
    ).to(expected_up.dtype)
    expected = F.linear(expected_activated, module.down_proj.weight)

    # Align bypasses MergedColumnParallelLinear.forward and performs the two
    # original training GEMMs over contiguous views of the packed parameter.
    assert projection_calls == 0
    assert torch.equal(observed["up"], expected_up)
    assert torch.equal(observed["gate"], expected_gate)
    assert observed["limit"] == 10.0
    assert torch.equal(actual, expected)


def test_yoco_fast_shared_expert_uses_m1_down_transpose(monkeypatch) -> None:
    import vllm.model_executor.layers.yoco_moe as yoco_module

    gate_up_projection_calls = 0
    down_projection_calls = 0

    class FakeMergedLinear(torch.nn.Module):
        def __init__(self, input_size, output_sizes, **kwargs) -> None:
            super().__init__()
            del kwargs
            self.weight = torch.nn.Parameter(
                torch.empty(sum(output_sizes), input_size, dtype=torch.bfloat16)
            )

        def forward(self, x: torch.Tensor):
            nonlocal gate_up_projection_calls
            gate_up_projection_calls += 1
            return F.linear(x, self.weight), None

    class FakeRowLinear(torch.nn.Module):
        def __init__(self, input_size, output_size, **kwargs) -> None:
            super().__init__()
            del kwargs
            self.weight = torch.nn.Parameter(
                torch.empty(output_size, input_size, dtype=torch.bfloat16)
            )

        def forward(self, x: torch.Tensor):
            nonlocal down_projection_calls
            down_projection_calls += 1
            return F.linear(x, self.weight), None

    class FakeActivation(torch.nn.Module):
        def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
            gate, up = gate_up.chunk(2, dim=-1)
            return gate * up

    monkeypatch.setattr(yoco_module, "MergedColumnParallelLinear", FakeMergedLinear)
    monkeypatch.setattr(yoco_module, "RowParallelLinear", FakeRowLinear)
    monkeypatch.setattr(yoco_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        yoco_module,
        "SiluAndMulWithClampFP32",
        lambda *args, **kwargs: FakeActivation(),
    )

    module = yoco_module.YOCOSharedExperts(
        hidden_size=8,
        intermediate_size=4,
        quant_config=None,
        reduce_results=False,
        prefix="model.layers.0.mlp.shared_experts",
        execution_mode="fast",
    )
    module.gate_up_proj.weight.data.copy_(
        torch.arange(64, dtype=torch.bfloat16).view(8, 8) / 64
    )
    module.down_proj.weight.data.copy_(
        torch.arange(32, dtype=torch.bfloat16).view(8, 4) / 32
    )
    module._fast_down_weight_t = module.down_proj.weight.t().contiguous()

    single_input = torch.arange(8, dtype=torch.bfloat16).view(1, 8) / 8
    single_activated = module.act_fn(F.linear(single_input, module.gate_up_proj.weight))
    expected_single = torch.mm(single_activated, module._fast_down_weight_t)
    actual_single = module(single_input)
    assert gate_up_projection_calls == 1
    assert down_projection_calls == 0
    assert torch.equal(actual_single, expected_single)

    batch64_input = single_input.expand(64, -1).contiguous()
    batch64_activated = module.act_fn(
        F.linear(batch64_input, module.gate_up_proj.weight)
    )
    expected_batch64 = F.linear(batch64_activated, module.down_proj.weight)
    actual_batch64 = module(batch64_input)
    assert gate_up_projection_calls == 2
    assert down_projection_calls == 1
    assert torch.equal(actual_batch64, expected_batch64)

    batch2_input = single_input.expand(2, -1).contiguous()
    module(batch2_input)
    assert gate_up_projection_calls == 3
    assert down_projection_calls == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 7, 128])
def test_yoco_weighted_swiglu_matches_training_formula(num_tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(8100 + num_tokens)
    x = (
        5
        * torch.randn(
            num_tokens,
            2 * 3840,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
    ).contiguous()
    routing_weights = torch.rand(
        num_tokens,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).contiguous()
    actual = torch.empty(num_tokens, 3840, device="cuda", dtype=torch.bfloat16)

    gate, up = x.float().chunk(2, dim=-1)
    gate = gate.clamp(max=10.0)
    up = up.clamp(min=-10.0, max=10.0)
    expected = (torch.nn.functional.silu(gate) * up * routing_weights.unsqueeze(-1)).to(
        torch.bfloat16
    )
    yoco_weighted_swiglu(actual, x, routing_weights, 10.0)

    torch.testing.assert_close(actual, expected, rtol=4e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_weighted_swiglu_is_batch_invariant() -> None:
    generator = torch.Generator(device="cuda").manual_seed(8200)
    row = torch.randn(
        1,
        2 * 3840,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).contiguous()
    routing_weight = torch.tensor([0.125], device="cuda", dtype=torch.float32)
    expected = torch.empty(1, 3840, device="cuda", dtype=torch.bfloat16)
    yoco_weighted_swiglu(expected, row, routing_weight, 10.0)

    for num_tokens in (2, 17, 257):
        x = torch.randn(
            num_tokens,
            2 * 3840,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        x[0].copy_(row[0])
        routing_weights = torch.rand(
            num_tokens,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        routing_weights[0] = routing_weight[0]
        actual = torch.empty(num_tokens, 3840, device="cuda", dtype=torch.bfloat16)
        yoco_weighted_swiglu(actual, x, routing_weights, 10.0)
        assert torch.equal(actual[0], expected[0])


def test_yoco_latent_transforms_preserve_reference_order() -> None:
    class Linear(torch.nn.Module):
        def __init__(self, weight: torch.Tensor) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(weight)

        def forward(self, x: torch.Tensor):
            return torch.nn.functional.linear(x, self.weight), None

    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.bfloat16)
    down = Linear(torch.arange(12, dtype=torch.bfloat16).view(3, 4) / 10)
    up = Linear(torch.arange(12, dtype=torch.bfloat16).view(4, 3) / 20)
    norm = RMSNorm(3, eps=1e-6, dtype=torch.float32)

    input_transform = YOCOLatentInputTransform(down, norm)
    latent = input_transform(x)
    expected_latent = norm(down(x)[0])
    torch.testing.assert_close(latent, expected_latent, rtol=0, atol=0)

    output_transform = YOCOLatentOutputTransform(norm, up)
    output = output_transform(latent)
    expected_output = up(norm(latent))[0]
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)


@pytest.mark.parametrize("execution_mode", ["align", "fast"])
def test_yoco_shared_output_gate_runs_after_reduction(execution_mode: str) -> None:
    gate = torch.nn.Linear(4, 1, bias=False)
    gate.weight.data.copy_(torch.tensor([[0.25, -0.5, 0.75, 1.0]]))
    transform = YOCOCombinedOutputTransform(  # type: ignore[arg-type]
        gate,
        execution_mode=execution_mode,
    )
    hidden_states = torch.tensor([[1.0, 2.0, -1.0, 0.5]])
    reduced_shared = torch.tensor([[2.0, -3.0, 4.0, -5.0]])
    routed = torch.tensor([[-1.0, 1.5, -2.0, 2.5]])

    actual = transform(reduced_shared, routed, hidden_states)
    scale = torch.sigmoid(torch.nn.functional.linear(hidden_states, gate.weight))
    expected = routed + scale * reduced_shared
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 8, 128])
def test_yoco_fast_combined_output_matches_sequential(num_tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(9100 + num_tokens)
    gate = torch.nn.Linear(3072, 1, bias=False, dtype=torch.bfloat16, device="cuda")
    gate.weight.data.normal_(generator=generator)
    transform = YOCOCombinedOutputTransform(  # type: ignore[arg-type]
        gate,
        execution_mode="fast",
    )
    hidden_states = torch.randn(
        num_tokens,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    shared = torch.randn(
        num_tokens,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    routed = torch.randn(
        num_tokens,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    expected_scale = torch.sigmoid(F.linear(hidden_states, gate.weight))
    expected = routed + expected_scale * shared
    actual = transform(shared, routed, hidden_states)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3.2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 8, 128])
def test_yoco_align_combined_output_is_bitwise_exact(num_tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(9150 + num_tokens)
    gate = torch.nn.Linear(3072, 1, bias=False, dtype=torch.bfloat16, device="cuda")
    gate.weight.data.normal_(generator=generator)
    transform = YOCOCombinedOutputTransform(  # type: ignore[arg-type]
        gate,
        execution_mode="align",
    )
    hidden_states = torch.randn(
        num_tokens,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    shared = torch.randn(
        num_tokens,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    routed = torch.randn(
        num_tokens,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    expected_scale = torch.sigmoid(F.linear(hidden_states, gate.weight))
    expected = routed + expected_scale * shared
    actual = transform(shared, routed, hidden_states)

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_fast_combined_output_is_batch_invariant() -> None:
    generator = torch.Generator(device="cuda").manual_seed(9200)
    gate = torch.nn.Linear(3072, 1, bias=False, dtype=torch.bfloat16, device="cuda")
    gate.weight.data.normal_(generator=generator)
    transform = YOCOCombinedOutputTransform(  # type: ignore[arg-type]
        gate,
        execution_mode="fast",
    )
    row_hidden = torch.randn(
        1,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    row_shared = torch.randn(
        1,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    row_routed = torch.randn(
        1,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    expected = transform(row_shared, row_routed, row_hidden)

    for num_tokens in (2, 17, 257):
        hidden_states = torch.randn(
            num_tokens,
            3072,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        shared = torch.randn(
            num_tokens,
            3072,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        routed = torch.randn(
            num_tokens,
            3072,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        hidden_states[0].copy_(row_hidden[0])
        shared[0].copy_(row_shared[0])
        routed[0].copy_(row_routed[0])
        actual = transform(shared, routed, hidden_states)
        assert torch.equal(actual[0], expected[0])


def test_yoco_separate_shared_reduction_keeps_collective_order(monkeypatch) -> None:
    import vllm.model_executor.layers.fused_moe.runner.moe_runner as runner_module
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    class MoEConfig:
        is_sequence_parallel = False
        tp_size = 2
        ep_size = 1

    class QuantMethod:
        moe_kernel = None

    runner = MoERunner.__new__(MoERunner)
    runner.reduce_shared_experts_separately = True
    runner.moe_config = MoEConfig()
    runner.routed_experts = SimpleNamespace(quant_method=QuantMethod())
    runner.routed_output_transform = None
    runner._shared_experts = object()

    calls: list[torch.Tensor] = []

    def fake_all_reduce(x: torch.Tensor) -> torch.Tensor:
        calls.append(x)
        return x + len(calls)

    monkeypatch.setattr(
        runner_module, "tensor_model_parallel_all_reduce", fake_all_reduce
    )
    shared = torch.tensor([10.0])
    routed = torch.tensor([20.0])
    reduced_routed, already_reduced = (
        runner._maybe_reduce_routed_output_before_transform(routed, False)
    )
    reduced_shared = runner._maybe_reduce_shared_expert_output(shared, already_reduced)

    assert calls[0] is routed
    assert calls[1] is shared
    torch.testing.assert_close(reduced_routed, routed + 1)
    torch.testing.assert_close(reduced_shared, shared + 2)


def test_yoco_router_weight_cache_matches_runtime_normalization() -> None:
    module = YOCOMoE.__new__(YOCOMoE)
    torch.nn.Module.__init__(module)
    module.gate = torch.nn.Linear(8, 4, bias=False, dtype=torch.float32)
    module.gate.weight.data.copy_(
        torch.arange(32, dtype=torch.float32).view(4, 8) / 7 - 2
    )
    module.execution_mode = "fast"
    module.router_weights_normalized = False
    module.register_buffer("_normalized_gate_weight", None, persistent=False)

    expected = module.gate.weight / module.gate.weight.norm(
        dim=1, keepdim=True
    ).clamp_min(1e-6)
    module.initialize_router_weight_cache()

    assert module._normalized_gate_weight is not None
    torch.testing.assert_close(module._normalized_gate_weight, expected, rtol=0, atol=0)
    assert "_normalized_gate_weight" not in module.state_dict()


def test_yoco_router_weight_cache_skips_already_normalized_weights() -> None:
    module = YOCOMoE.__new__(YOCOMoE)
    torch.nn.Module.__init__(module)
    module.gate = torch.nn.Linear(8, 4, bias=False, dtype=torch.float32)
    module.execution_mode = "fast"
    module.router_weights_normalized = True
    module.register_buffer(
        "_normalized_gate_weight",
        torch.full_like(module.gate.weight, float("nan")),
        persistent=False,
    )

    module.initialize_router_weight_cache()

    assert module._normalized_gate_weight is None


def test_yoco_align_router_does_not_cache_normalized_weight() -> None:
    module = YOCOMoE.__new__(YOCOMoE)
    torch.nn.Module.__init__(module)
    module.gate = torch.nn.Linear(8, 4, bias=False, dtype=torch.float32)
    module.execution_mode = "align"
    module.router_weights_normalized = False
    module.register_buffer(
        "_normalized_gate_weight",
        torch.full_like(module.gate.weight, float("nan")),
        persistent=False,
    )

    module.initialize_router_weight_cache()

    assert module._normalized_gate_weight is None
