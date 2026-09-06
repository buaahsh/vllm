# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO-only BF16 DeepGEMM W2 path.

The training implementation feeds W2 an expert-major tensor padded to 128
rows per active expert and calls DeepGEMM with a prefix-sum grouped layout.
The regular vLLM Triton MoE path keeps routed rows in token-major order, so
this module owns the required pack and unpack kernels without changing the
shared MoE utilities or their global DeepGEMM alignment.
"""

from collections.abc import Callable

import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    count_expert_num_tokens,
)
from vllm.triton_utils import tl, triton

YOCO_DEEP_GEMM_ALIGNMENT = 128


def yoco_deep_gemm_w2_workspace_rows(
    num_tokens: int, topk: int, num_experts: int
) -> int:
    """Return the graph-stable capacity of the 128-row packed layout."""
    assignments = num_tokens * topk
    rows = assignments + num_experts * (YOCO_DEEP_GEMM_ALIGNMENT - 1)
    rows = (
        (rows + YOCO_DEEP_GEMM_ALIGNMENT - 1) // YOCO_DEEP_GEMM_ALIGNMENT
    ) * YOCO_DEEP_GEMM_ALIGNMENT
    if assignments < num_experts:
        rows = min(assignments * YOCO_DEEP_GEMM_ALIGNMENT, rows)
    return rows


def supports_yoco_deep_gemm_w2() -> bool:
    """Whether the training-compatible BF16 grouped symbol is usable."""
    try:
        from vllm.utils.deep_gemm import (
            _import_deep_gemm,
            get_mk_alignment_for_contiguous_layout,
            is_deep_gemm_supported,
        )

        if not is_deep_gemm_supported():
            return False
        if get_mk_alignment_for_contiguous_layout()[0] != YOCO_DEEP_GEMM_ALIGNMENT:
            return False
        deep_gemm = _import_deep_gemm()
        return deep_gemm is not None and callable(
            getattr(deep_gemm, "m_grouped_bf16_gemm_nt_contiguous", None)
        )
    except (ImportError, RuntimeError):
        return False


@triton.jit
def _pack_expert_major_kernel(
    source_ptr,
    packed_ptr,
    sorted_ids_ptr,
    num_assignments,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    packed_row = tl.program_id(0).to(tl.int64)
    source_row = tl.load(sorted_ids_ptr + packed_row).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < width
    valid = source_row < num_assignments
    values = tl.load(
        source_ptr + source_row * width + offsets,
        mask=valid & mask,
        other=0.0,
    )
    tl.store(packed_ptr + packed_row * width + offsets, values, mask=mask)


@triton.jit
def _unpack_expert_major_kernel(
    packed_ptr,
    output_ptr,
    sorted_ids_ptr,
    num_assignments,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    packed_row = tl.program_id(0).to(tl.int64)
    output_row = tl.load(sorted_ids_ptr + packed_row).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < width
    valid = output_row < num_assignments
    values = tl.load(
        packed_ptr + packed_row * width + offsets,
        mask=valid & mask,
        other=0.0,
    )
    tl.store(
        output_ptr + output_row * width + offsets,
        values,
        mask=valid & mask,
    )


def _grouped_prefix_sum_layout(
    topk_ids: torch.Tensor, num_experts: int, packed_rows: int
) -> torch.Tensor:
    counts = count_expert_num_tokens(topk_ids, num_experts, expert_map=None)
    padded_counts = (
        (counts + YOCO_DEEP_GEMM_ALIGNMENT - 1) // YOCO_DEEP_GEMM_ALIGNMENT
    ) * YOCO_DEEP_GEMM_ALIGNMENT
    grouped_layout = torch.cumsum(padded_counts, dim=0, dtype=torch.int32)

    # The actual number of active 128-row blocks depends on routing. DeepGEMM
    # needs a static A/D shape for CUDA graphs, so assign the unused zero tail
    # to the final expert. Real rows keep exactly the same expert boundaries.
    grouped_layout[-1:].fill_(packed_rows)
    return grouped_layout


def yoco_deep_gemm_w2(
    output: torch.Tensor,
    activation: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    packed_input_workspace: torch.Tensor,
    packed_output_workspace: torch.Tensor,
    *,
    deep_gemm_impl: Callable[..., None] | None = None,
) -> None:
    """Run YOCO W2 with llm-train's BF16 DeepGEMM contract.

    ``activation`` and ``output`` use flattened token-major routed rows.
    Workspaces are reused for the expert-major DeepGEMM input and output.
    A callable can be injected by tests to validate the layout on pre-SM90
    GPUs without executing a DeepGEMM kernel.
    """
    assert activation.ndim == output.ndim == 2
    assert w2.ndim == 3 and topk_ids.ndim == 2
    num_tokens, topk = topk_ids.shape
    num_assignments, intermediate_size = activation.shape
    num_experts, hidden_size, w2_intermediate_size = w2.shape
    assert num_assignments == num_tokens * topk
    assert w2_intermediate_size == intermediate_size
    assert output.shape == (num_assignments, hidden_size)
    assert activation.dtype == w2.dtype == output.dtype == torch.bfloat16
    assert activation.is_contiguous() and w2.is_contiguous() and output.is_contiguous()

    sorted_ids, _, _ = moe_align_block_size(
        topk_ids,
        YOCO_DEEP_GEMM_ALIGNMENT,
        num_experts,
        expert_map=None,
        pad_sorted_ids=True,
    )
    packed_rows = yoco_deep_gemm_w2_workspace_rows(num_tokens, topk, num_experts)
    assert sorted_ids.numel() == packed_rows
    packed_input = _resize_cache(
        packed_input_workspace, (packed_rows, intermediate_size)
    )
    packed_output = _resize_cache(packed_output_workspace, (packed_rows, hidden_size))

    _pack_expert_major_kernel[(packed_rows,)](
        activation,
        packed_input,
        sorted_ids,
        num_assignments,
        width=intermediate_size,
        BLOCK_SIZE=triton.next_power_of_2(intermediate_size),
        num_warps=8,
    )
    grouped_layout = _grouped_prefix_sum_layout(topk_ids, num_experts, packed_rows)

    if deep_gemm_impl is None:
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        if deep_gemm is None:
            raise RuntimeError("DeepGEMM is unavailable for YOCO BF16 W2")
        deep_gemm_impl = deep_gemm.m_grouped_bf16_gemm_nt_contiguous

    deep_gemm_impl(
        packed_input,
        w2,
        packed_output,
        grouped_layout,
        use_psum_layout=True,
        expected_m_for_psum_layout=packed_rows,
    )
    _unpack_expert_major_kernel[(packed_rows,)](
        packed_output,
        output,
        sorted_ids,
        num_assignments,
        width=hidden_size,
        BLOCK_SIZE=triton.next_power_of_2(hidden_size),
        num_warps=8,
    )
