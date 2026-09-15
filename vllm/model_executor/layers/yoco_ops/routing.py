# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO routing operators and their numerical fallbacks."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

if HAS_TRITON:

    @triton.jit
    def _yoco_align_router_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        stride_x: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # One fixed IEEE FP32 reduction per token/expert pair. No padding the
        # token batch to 128 rows and no process-global TF32 mode switching.
        token = tl.program_id(0)
        expert = tl.program_id(1) * 4 + tl.arange(0, 4)
        k = tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + token * stride_x + k, k < K, 0).to(tl.float32)
        weight = tl.load(
            weight_ptr + expert[:, None] * K + k[None, :],
            (expert[:, None] < N) & (k[None, :] < K),
            0,
        ).to(tl.float32)
        logits = tl.sum(weight * x[None, :], axis=1)
        tl.store(out_ptr + token * N + expert, logits, expert < N)

    @triton.jit
    def _yoco_fused_topk_routing_kernel(
        logits_ptr,
        topk_weights_ptr,
        topk_ids_ptr,
        num_rows,
        BLOCK_ROWS: tl.constexpr,
        TOPK_LOGITS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        cols = tl.arange(0, 128)[None, :]
        row_mask = rows < num_rows
        logits = tl.load(
            logits_ptr + rows * 128 + cols,
            mask=row_mask,
            other=float("-inf"),
        ).to(tl.float32)

        if TOPK_LOGITS:
            # Fast only needs probabilities normalized over the selected
            # experts. Keep the full-softmax rounding path for Align.
            scores = logits
        else:
            row_max = tl.max(logits, axis=1)[:, None]
            numerator = tl.extra.cuda.libdevice.exp(logits - row_max)
            scores = numerator / tl.sum(numerator, axis=1)[:, None]

        ranks = tl.arange(0, 8)[None, :]
        selected_scores = tl.full((BLOCK_ROWS, 8), 0.0, tl.float32)
        selected_ids = tl.full((BLOCK_ROWS, 8), 0, tl.int32)
        for rank in tl.static_range(8):
            value, index = tl.max(
                scores,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            selected_scores = tl.where(
                ranks == rank,
                value[:, None],
                selected_scores,
            )
            selected_ids = tl.where(
                ranks == rank,
                index[:, None],
                selected_ids,
            )
            scores = tl.where(cols == index[:, None], float("-inf"), scores)

        if TOPK_LOGITS:
            selected_max = tl.max(selected_scores, axis=1)[:, None]
            selected_scores = tl.extra.cuda.libdevice.exp(
                selected_scores - selected_max
            )
        selected_scores /= tl.sum(selected_scores, axis=1)[:, None]
        output_offsets = rows * 8 + ranks
        tl.store(topk_weights_ptr + output_offsets, selected_scores, mask=row_mask)
        tl.store(topk_ids_ptr + output_offsets, selected_ids, mask=row_mask)


def _yoco_router_linear_tf32_cuda(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    normalize_weight: bool,
) -> torch.Tensor:
    if normalize_weight:
        weight = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
    # The fast serving path intentionally uses the real token-row count.  Shape
    # padding belongs to the explicit alignment policy, not this default op:
    # CUDA Graph removes host launch overhead but cannot remove padded GEMM work.
    assert hidden_states.ndim == 2
    previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_matmul_precision = torch.get_float32_matmul_precision()
    previous_cuda_precision = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    try:
        return F.linear(hidden_states, weight)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
        torch.set_float32_matmul_precision(previous_matmul_precision)
        torch.backends.cuda.matmul.fp32_precision = previous_cuda_precision


def _yoco_router_linear_tf32_fake(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    normalize_weight: bool,
) -> torch.Tensor:
    del normalize_weight
    return hidden_states.new_empty((*hidden_states.shape[:-1], weight.shape[0]))


def _yoco_router_linear_bf16_cuda(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    assert hidden_states.dtype == weight.dtype == torch.bfloat16
    return F.linear(hidden_states, weight)


def _yoco_router_linear_bf16_fake(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return hidden_states.new_empty((*hidden_states.shape[:-1], weight.shape[0]))


def _yoco_topk_routing_impl(
    router_logits: torch.Tensor,
    topk: int,
    *,
    topk_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert router_logits.dtype == torch.float32 or (
        topk_logits and router_logits.dtype == torch.bfloat16
    )
    assert router_logits.shape[-1] == 128
    assert topk == 8
    router_logits = router_logits.contiguous()
    num_rows = router_logits.shape[0]
    output_shape = (num_rows, 8)
    topk_weights = torch.empty(
        output_shape,
        dtype=torch.float32,
        device=router_logits.device,
    )
    topk_ids = torch.empty(
        output_shape,
        # Every YOCO Fast MoE backend consumes INT32 expert ids.  Returning
        # INT64 here made CustomRoutingRouter launch a standalone cast after
        # each of the 40 logical Router calls.
        dtype=torch.int32,
        device=router_logits.device,
    )
    single_token = (
        topk_logits and num_rows == 1 and current_platform.is_device_capability(100)
    )
    block_rows = 1 if single_token else 4
    _yoco_fused_topk_routing_kernel[(triton.cdiv(num_rows, block_rows),)](
        router_logits,
        topk_weights,
        topk_ids,
        num_rows,
        BLOCK_ROWS=block_rows,
        TOPK_LOGITS=topk_logits,
        num_warps=1 if single_token else 4,
        num_stages=1,
    )
    return topk_weights, topk_ids


@torch.compile
def _yoco_align_topk_routing_impl(
    logits: torch.Tensor,
    topk: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """The complete routing expression compiled by llm-train."""
    gate_scores = F.softmax(logits, dim=-1, dtype=torch.float32)
    scores, top_indices = torch.topk(gate_scores, k=topk, dim=-1)
    probs = scores / scores.sum(dim=-1, keepdim=True)
    routing_probs = torch.zeros_like(logits).scatter(
        1, top_indices, probs.to(logits.dtype)
    )
    routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    return probs, top_indices, routing_probs, routing_map, gate_scores


def _yoco_align_topk_routing(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del hidden_states
    assert renormalize
    if HAS_TRITON and gating_output.is_cuda and current_platform.is_cuda():
        if gating_output.shape[-1] != 128 or topk != 8:
            raise ValueError("YOCO invariant routing requires 128 experts and Top-8")
        # Fix both reductions and the tie rule. Inductor's compiled softmax /
        # renormalization is not a stable numerical contract across warmup
        # shapes and enclosing graph-capture contexts.
        topk_weights, topk_ids = _yoco_topk_routing_impl(gating_output, topk)
    else:
        topk_weights, topk_ids, _, _, _ = _yoco_align_topk_routing_impl(
            gating_output, topk
        )
    # TransformerEngine's token unpermute traverses selected experts in
    # expert-id order. Preserve each probability/id pair while matching that
    # order for the subsequent fixed-order FP32 reduction.
    expert_order = torch.argsort(topk_ids, dim=-1)
    topk_ids = torch.gather(topk_ids, dim=-1, index=expert_order)
    topk_weights = torch.gather(topk_weights, dim=-1, index=expert_order)
    return topk_weights, topk_ids


def _yoco_topk_routing(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del hidden_states
    assert renormalize
    if gating_output.shape[-1] != 128 or topk != 8:
        gate_scores = F.softmax(gating_output, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(gate_scores, k=topk, dim=-1)
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights, topk_ids
    return _yoco_topk_routing_impl(gating_output, topk, topk_logits=True)


def _yoco_normalized_router_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    weight = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return F.linear(hidden_states, weight)


def _yoco_align_router_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    normalize_weight: bool,
) -> torch.Tensor:
    """Fixed IEEE FP32 Router reduction, independent of token-row shape."""
    assert hidden_states.ndim == 2
    if hidden_states.is_cuda and current_platform.is_cuda():
        assert hidden_states.dtype == weight.dtype == torch.float32
        assert hidden_states.shape[1] == weight.shape[1]
        hidden_states = hidden_states.contiguous()
        if normalize_weight:
            weight = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
        weight = weight.contiguous()
        output = hidden_states.new_empty((hidden_states.shape[0], weight.shape[0]))
        if hidden_states.shape[0]:
            grid = (hidden_states.shape[0], triton.cdiv(weight.shape[0], 4))
            _yoco_align_router_kernel[grid](
                hidden_states,
                weight,
                output,
                stride_x=hidden_states.stride(0),
                K=weight.shape[1],
                N=weight.shape[0],
                BLOCK_K=triton.next_power_of_2(weight.shape[1]),
                num_warps=4,
                enable_fp_fusion=False,
            )
        return output
    if normalize_weight:
        return _yoco_normalized_router_linear(hidden_states, weight)
    return F.linear(hidden_states, weight)


if current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_router_linear_bf16",
        op_func=_yoco_router_linear_bf16_cuda,
        fake_impl=_yoco_router_linear_bf16_fake,
    )


if current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_router_linear_tf32",
        op_func=_yoco_router_linear_tf32_cuda,
        fake_impl=_yoco_router_linear_tf32_fake,
    )
