# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO routing regression tests."""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.yoco_ops.routing import (
    _yoco_align_router_linear,
    _yoco_align_topk_routing,
    _yoco_topk_routing,
)


@pytest.mark.parametrize("version", [None, 1])
def test_mega_gate_rejects_incompatible_or_padding_unsafe_backend(monkeypatch, version):
    from types import SimpleNamespace

    from vllm.utils import deep_gemm

    backend = SimpleNamespace(
        bf16_mega_gate=lambda *a, **k: None,
        _C=SimpleNamespace(yoco_mega_gate_version=version),
    )
    monkeypatch.setattr(deep_gemm, "_import_deep_gemm", lambda: backend)
    deep_gemm.get_deep_gemm_yoco_mega_gate.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="yoco_softmax v2 patch"):
            deep_gemm.get_deep_gemm_yoco_mega_gate()
    finally:
        deep_gemm.get_deep_gemm_yoco_mega_gate.cache_clear()


@pytest.mark.parametrize(
    "case,expected",
    [
        ("supported", True),
        ("disabled", False),
        ("align", False),
        ("invariant", False),
        ("fp32", False),
        ("shape", False),
        ("hardware", False),
        ("tp", False),
        ("ep", False),
        ("dbo", False),
        ("capacity", False),
        ("prefill", True),
        ("prefill_below_threshold", False),
        ("prefill_at_threshold", True),
        ("prefill_no_connector", False),
        ("consumer_no_spec", False),
        ("both_no_spec", False),
        ("standalone_no_spec", False),
        ("dump", False),
        ("unpatched", False),
    ],
)
def test_fast_mega_gate_default_preserves_unsupported_router_paths(
    monkeypatch, case, expected
):
    from types import SimpleNamespace

    from vllm.config import KVTransferConfig
    from vllm.model_executor.layers import yoco_moe
    from vllm.utils import deep_gemm

    monkeypatch.delenv("VLLM_YOCO_MEGA_GATE", raising=False)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", str(int(case == "invariant")))
    module = yoco_moe.YOCOMoE.__new__(yoco_moe.YOCOMoE)
    torch.nn.Module.__init__(module)
    module.execution_mode = "align" if case == "align" else "fast"
    module.bf16_router = case != "fp32"
    module.hidden_size = 2048 if case == "shape" else 3072
    module.num_experts, module.top_k = 128, 8
    module.mega_gate_min_tokens = 128
    module._route_dumper = object() if case == "dump" else None
    parallel = SimpleNamespace(
        tensor_parallel_size=2 if case == "tp" else 1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=case == "ep",
        enable_dbo=case == "dbo",
    )
    if case == "disabled":
        monkeypatch.setenv("VLLM_YOCO_MEGA_GATE", "0")
    kv_role = (
        "kv_producer"
        if case.startswith("prefill")
        else {"consumer_no_spec": "kv_consumer", "both_no_spec": "kv_both"}.get(case)
    )
    kv_transfer = (
        KVTransferConfig(
            kv_connector=None
            if case == "prefill_no_connector"
            else "MooncakeConnector",
            kv_role=kv_role,
        )
        if kv_role is not None
        else None
    )
    no_spec = case.startswith("prefill") or case.endswith("no_spec")
    monkeypatch.setattr(
        yoco_moe.current_platform,
        "is_device_capability_family",
        lambda capability: case != "hardware",
    )
    monkeypatch.setattr(
        yoco_moe,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            parallel_config=parallel,
            scheduler_config=SimpleNamespace(
                max_num_seqs=8 if case == "capacity" else 64,
                max_num_batched_tokens={
                    "prefill_below_threshold": 127,
                    "prefill_at_threshold": 128,
                }.get(case, 8192),
            ),
            speculative_config=None
            if no_spec
            else SimpleNamespace(num_speculative_tokens=8),
            kv_transfer_config=kv_transfer,
        ),
    )

    def backend():
        if case == "unpatched":
            raise RuntimeError("YOCO API unavailable")
        return object()

    monkeypatch.setattr(deep_gemm, "get_deep_gemm_yoco_mega_gate", backend)
    assert module._use_mega_gate() is expected


@pytest.mark.parametrize("rows,fused", [(127, False), (128, True), (576, True)])
def test_mega_gate_routes_original_input_after_latent_transform(
    monkeypatch, rows, fused
):
    from vllm.model_executor.layers import yoco_moe

    module = yoco_moe.YOCOMoE.__new__(yoco_moe.YOCOMoE)
    torch.nn.Module.__init__(module)
    module.mega_gate_min_tokens = 128
    module.register_buffer("_bf16_gate_weight", torch.empty(128, 3072).bfloat16())
    original = torch.empty(rows, 3072, dtype=torch.bfloat16)
    latent = torch.empty(rows, 1024, dtype=torch.bfloat16)
    expected = torch.empty(rows, 8), torch.empty(rows, 8, dtype=torch.int32)
    observed = []

    def mega(x, weight):
        assert x is original and weight is module._bf16_gate_weight
        observed.append("fused")
        return expected

    logits = torch.empty(rows, 128, dtype=torch.bfloat16)

    def linear(x, weight):
        assert x is original and weight is module._bf16_gate_weight
        observed.append("linear")
        return logits

    def topk(x, scores, k, renormalize):
        assert x is latent and scores is logits and k == 8 and renormalize
        observed.append("topk")
        return expected

    monkeypatch.setattr(torch.ops.vllm, "yoco_mega_gate", mega)
    monkeypatch.setattr(torch.ops.vllm, "yoco_router_linear_bf16", linear)
    monkeypatch.setattr(yoco_moe, "_yoco_topk_routing", topk)
    weights, ids = module._mega_gate_routing(latent, original, 8, True)
    assert weights is expected[0] and ids is expected[1]
    assert observed == (["fused"] if fused else ["linear", "topk"])


def test_mega_gate_fake_outputs_match_modular_router_contract():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(576, 3072, device="cuda", dtype=torch.bfloat16)
        weight = torch.empty(128, 3072, device="cuda", dtype=torch.bfloat16)
        weights, ids = torch.ops.vllm.yoco_mega_gate(x, weight)
        assert weights.shape == ids.shape == (576, 8)
        assert weights.dtype == torch.float32 and ids.dtype == torch.int32
        assert weights.device == ids.device == x.device


def test_mega_gate_forward_defers_gemm_and_preserves_original_router_rows(monkeypatch):
    from vllm.model_executor.layers.yoco_moe import YOCOMoE

    module = YOCOMoE.__new__(YOCOMoE)
    torch.nn.Module.__init__(module)
    module.mega_gate = True
    module.router_weights_normalized = True
    module._normalized_gate_weight = None
    module.gate = torch.nn.Linear(3072, 128, bias=False)
    original = torch.empty(576, 3072, dtype=torch.bfloat16)

    class Experts(torch.nn.Module):
        def forward(self, hidden_states, router_logits):
            assert hidden_states.data_ptr() == original.data_ptr()
            assert router_logits.data_ptr() == original.data_ptr()
            assert router_logits.shape == (576, 3072)
            return hidden_states

    def unexpected_linear(*args, **kwargs):
        pytest.fail("the fused route must not materialize logits before the runner")

    module.experts = Experts()
    monkeypatch.setattr(F, "linear", unexpected_linear)
    assert module(original).data_ptr() == original.data_ptr()


@torch.compile
def _llm_train_topk_routing_reference(
    logits: torch.Tensor,
    topk: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    gate_scores = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
    scores, top_indices = torch.topk(gate_scores, k=topk, dim=-1)
    probs = scores / scores.sum(dim=-1, keepdim=True)
    routing_probs = torch.zeros_like(logits).scatter(
        1, top_indices, probs.to(logits.dtype)
    )
    routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    return probs, top_indices, routing_probs, routing_map, gate_scores


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 66, 128])
def test_yoco_align_topk_routing_uses_training_expert_order(
    num_tokens: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(9876 + num_tokens)
    logits = torch.randn(
        num_tokens,
        128,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    hidden_states = torch.empty(num_tokens, 1, device="cuda")

    expected_weights, expected_ids, _, _, _ = _llm_train_topk_routing_reference(
        logits, 8
    )
    expert_order = torch.argsort(expected_ids, dim=-1)
    expected_ids = torch.gather(expected_ids, dim=-1, index=expert_order)
    expected_weights = torch.gather(expected_weights, dim=-1, index=expert_order)
    actual_weights, actual_ids = _yoco_align_topk_routing(
        hidden_states,
        logits,
        topk=8,
        renormalize=True,
    )

    torch.testing.assert_close(actual_weights, expected_weights, rtol=2e-6, atol=0)
    assert torch.equal(actual_ids, expected_ids)
    assert torch.all(actual_ids[:, 1:] > actual_ids[:, :-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 3, 66, 110, 256])
def test_yoco_router_fused_topk_matches_reference(num_tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(1234 + num_tokens)
    logits = torch.randn(
        num_tokens,
        128,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    hidden_states = torch.empty(num_tokens, 1, device="cuda")

    actual_weights, actual_ids = _yoco_topk_routing(
        hidden_states,
        logits,
        topk=8,
        renormalize=True,
    )

    reference_scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    reference_weights, reference_ids = torch.topk(reference_scores, k=8, dim=-1)
    reference_weights /= reference_weights.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(actual_weights, reference_weights, rtol=2e-6, atol=0)
    assert actual_ids.dtype == torch.int32
    torch.testing.assert_close(
        actual_ids, reference_ids.to(torch.int32), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_router_fused_topk_handles_ties_deterministically() -> None:
    logits = torch.zeros(4, 128, dtype=torch.float32, device="cuda")
    hidden_states = torch.empty(4, 1, device="cuda")

    actual_weights, actual_ids = _yoco_topk_routing(
        hidden_states,
        logits,
        topk=8,
        renormalize=True,
    )
    # Both modes use a deterministic left-most tie break; torch.topk's
    # historical training ordering is not stable for tied values.
    expected_ids = torch.arange(8, device="cuda", dtype=torch.int32).expand(4, -1)
    expected_weights = torch.full_like(actual_weights, 1 / 8)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    align_weights, align_ids = _yoco_align_topk_routing(
        hidden_states, logits, topk=8, renormalize=True
    )
    assert torch.equal(align_weights, expected_weights)
    assert torch.equal(align_ids, expected_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_router_fused_topk_is_batch_independent() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260831)
    target = torch.randn(
        1,
        128,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    hidden_states = torch.empty(1, 1, device="cuda")
    expected_weights, expected_ids = _yoco_topk_routing(
        hidden_states,
        target,
        topk=8,
        renormalize=True,
    )

    for num_tokens, position in ((3, 1), (66, 37), (110, 109), (1024, 511)):
        logits = torch.randn(
            num_tokens,
            128,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        logits[position].copy_(target[0])
        actual_weights, actual_ids = _yoco_topk_routing(
            torch.empty(num_tokens, 1, device="cuda"),
            logits,
            topk=8,
            renormalize=True,
        )
        assert torch.equal(actual_weights[position], expected_weights[0])
        assert torch.equal(actual_ids[position], expected_ids[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 8, 128])
def test_yoco_cached_router_linear_is_bitwise_exact(num_tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(8128 + num_tokens)
    hidden_states = torch.randn(
        num_tokens,
        3072,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    weight = torch.randn(
        128,
        3072,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    cached_weight = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-6)

    expected = torch.ops.vllm.yoco_router_linear_tf32(hidden_states, weight, True)
    actual = torch.ops.vllm.yoco_router_linear_tf32(hidden_states, cached_weight, False)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 12, 127])
def test_yoco_fast_router_linear_uses_actual_batch_shape(
    num_tokens: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260809)
    hidden_states = torch.randn(
        num_tokens,
        3072,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    weight = torch.randn(
        128,
        3072,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )

    previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_matmul_precision = torch.get_float32_matmul_precision()
    previous_cuda_precision = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    try:
        expected = torch.nn.functional.linear(hidden_states, weight)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
        torch.set_float32_matmul_precision(previous_matmul_precision)
        torch.backends.cuda.matmul.fp32_precision = previous_cuda_precision
    actual = torch.ops.vllm.yoco_router_linear_tf32(hidden_states, weight, False)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 12, 127])
def test_yoco_align_router_linear_is_ieee_and_batch_invariant(
    num_tokens: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260809)
    hidden_states = torch.randn(
        num_tokens,
        3072,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    weight = torch.randn(
        128,
        3072,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )

    expected = F.linear(hidden_states.double(), weight.double()).float()
    actual = _yoco_align_router_linear(hidden_states, weight, False)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=3e-5)
    split = torch.cat(
        [_yoco_align_router_linear(row[None], weight, False) for row in hidden_states]
    )
    assert torch.equal(actual, split)
