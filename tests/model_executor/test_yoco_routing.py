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
