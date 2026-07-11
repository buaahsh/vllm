# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
import os

import torch
import triton
import triton.language as tl

import vllm._custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter


@triton.jit
def _yoco_native_softmax_kernel(
    logits_ptr,
    scores_ptr,
    num_rows,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    cols = tl.arange(0, NUM_EXPERTS)[None, :]
    row_mask = rows < num_rows
    logits = tl.load(
        logits_ptr + rows * NUM_EXPERTS + cols,
        mask=row_mask,
        other=float("-inf"),
    ).to(tl.float32)
    row_max = tl.max(logits, axis=1)[:, None]
    numerator = tl.extra.cuda.libdevice.exp(logits - row_max)
    denominator = tl.sum(tl.where(row_mask, numerator, 0.0), axis=1)[:, None]
    scores = numerator / denominator
    tl.store(
        scores_ptr + rows * NUM_EXPERTS + cols,
        scores,
        mask=row_mask,
    )


@triton.jit
def _yoco_native_topk_renorm_kernel(
    weights_ptr,
    num_rows,
    TOPK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    cols = tl.arange(0, TOPK)[None, :]
    row_mask = rows < num_rows
    weights = tl.load(
        weights_ptr + rows * TOPK + cols,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    denominator = tl.sum(tl.where(row_mask, weights, 0.0), axis=1)[:, None]
    tl.store(
        weights_ptr + rows * TOPK + cols,
        weights / denominator,
        mask=row_mask,
    )


def _yoco_native_topk_routing(
    logits: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert logits.dtype == torch.float32
    assert logits.is_contiguous()
    num_rows, num_experts = logits.shape
    assert num_experts == 128
    assert topk == 8
    scores = torch.empty_like(logits)
    # Match the persistent-reduction layout selected by llm-train Inductor.
    # Reduction association changes a few FP32 ULPs at BF16/FP8 boundaries.
    block_rows = 4
    _yoco_native_softmax_kernel[(triton.cdiv(num_rows, block_rows),)](
        logits,
        scores,
        num_rows,
        NUM_EXPERTS=num_experts,
        BLOCK_ROWS=block_rows,
        num_warps=4,
        num_stages=1,
    )
    topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)
    _yoco_native_topk_renorm_kernel[(triton.cdiv(num_rows, block_rows),)](
        topk_weights,
        num_rows,
        TOPK=topk,
        BLOCK_ROWS=block_rows,
        num_warps=4,
        num_stages=1,
    )
    return topk_weights, topk_ids


def vllm_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]:
    ops.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )

    return topk_weights, topk_indices


def vllm_topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]:
    ops.topk_sigmoid(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )

    return topk_weights, topk_indices


def dispatch_topk_softmax_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_softmax
    return vllm_topk_softmax


def dispatch_topk_sigmoid_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_sigmoid
    return vllm_topk_sigmoid


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
    scoring_func: str = "softmax",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    M, _ = hidden_states.size()

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=hidden_states.device,
    )
    token_expert_indices = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    if scoring_func == "softmax":
        if not rocm_aiter_ops.is_fused_moe_enabled():
            # YOCO/llm-train uses torch softmax + torch.topk(sorted=True), then
            # renormalizes the selected weights.  The CUDA fused topk kernel can
            # choose a different top-k ordering/numerical path, which is visible
            # in tight KL alignment checks.
            use_yoco_compiled = (
                renormalize
                and os.getenv("VLLM_YOCO_COMPILED_TOPK_ROUTING") == "1"
            )
            if use_yoco_compiled:
                topk_weights, topk_ids_torch = _yoco_native_topk_routing(
                    gating_output, topk
                )
            else:
                scores = torch.softmax(gating_output, dim=-1, dtype=torch.float32)
                topk_weights, topk_ids_torch = torch.topk(
                    scores, k=topk, dim=-1, sorted=True
                )
            if renormalize and not use_yoco_compiled:
                topk_weights = topk_weights / topk_weights.sum(
                    dim=-1, keepdim=True
                )
            topk_ids_torch = topk_ids_torch.to(
                torch.int32 if indices_type is None else indices_type
            )
            token_expert_indices.copy_(topk_ids_torch.to(torch.int32))
            return (
                topk_weights.to(torch.float32),
                topk_ids_torch,
                token_expert_indices,
            )
        topk_func = dispatch_topk_softmax_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
        )
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )

        return topk_weights, topk_ids, token_expert_indices
    elif scoring_func == "sigmoid":
        topk_func = dispatch_topk_sigmoid_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
        )
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )

        return topk_weights, topk_ids, token_expert_indices
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")


class FusedTopKRouter(BaseRouter):
    """Default router using standard fused top-k routing."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        scoring_func: str = "softmax",
        renormalize: bool = True,
        eplb_state: EplbLayerState | None = None,
        indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            indices_type_getter=indices_type_getter,
        )
        self.renormalize = renormalize
        self.scoring_func = scoring_func

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return get_routing_method_type(
            scoring_func=self.scoring_func,
            top_k=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=None,
            has_e_score_bias=False,
        )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing using standard fused top-k."""
        topk_weights, topk_ids, token_expert_indices = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            indices_type=indices_type,
            scoring_func=self.scoring_func,
        )
        return topk_weights, topk_ids
