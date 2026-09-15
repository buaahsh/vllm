# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.utils import count_expert_num_tokens
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import get_mk_alignment_for_contiguous_layout
from vllm.utils.math_utils import round_up


def expert_num_tokens_round_up_and_sum(
    expert_num_tokens: torch.Tensor, alignment: int
) -> int:
    # Round up each element in expert_num_tokens to the nearest multiple of
    # alignment.
    ent = (expert_num_tokens.to(torch.int64) + (alignment - 1)) // alignment * alignment
    return torch.sum(ent).item()


def compute_aligned_M(
    M: int,
    num_topk: int,
    local_num_experts: int,
    alignment: int,
    expert_tokens_meta: mk.ExpertTokensMetadata | None,
):
    if (expert_tokens_meta is not None) and (
        expert_tokens_meta.expert_num_tokens_cpu is not None
    ):
        return expert_num_tokens_round_up_and_sum(
            expert_tokens_meta.expert_num_tokens_cpu, alignment=alignment
        )

    # Without CPU counts, bound padding by the number of experts that can
    # receive at least one routed token. In small decode batches this can be
    # much smaller than local_num_experts: M=1, topk=8, E=128 needs at most
    # 8 aligned expert blocks. Padding every expert would reserve 128 blocks
    # and process those extra rows during activation quantization in graphs.
    num_routed_tokens = M * num_topk
    max_active_experts = local_num_experts
    if num_routed_tokens > 0:
        max_active_experts = min(local_num_experts, num_routed_tokens)
    M_sum = num_routed_tokens + max_active_experts * (alignment - 1)
    M_sum = round_up(M_sum, alignment)
    return M_sum


@triton.jit
def apply_expert_map(expert_id, expert_map):
    if expert_id != -1:
        expert_id = tl.load(expert_map + expert_id).to(expert_id.dtype)
    return expert_id


@triton.jit
def round_up_128(x: int) -> int:
    y = 128
    return ((x + y - 1) // y) * y


@triton.jit
def _fwd_kernel_ep_scatter_1(
    num_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)

    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(
        num_recv_tokens_per_expert + offset_cumsum,
        mask=offset_cumsum < num_experts,
        other=0,
    )
    tokens_per_expert = round_up_128(tokens_per_expert)
    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert

    # Extract this block's offset from the register vector (warp shuffle,
    # no global memory round-trip) then write it once to expert_start_loc.
    cur_expert_start = tl.sum(
        tl.where(offset_cumsum == cur_expert, cumsum, tl.zeros_like(cumsum))
    )
    tl.store(expert_start_loc + cur_expert, cur_expert_start)
    cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)

    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)

    # any rows in the per-expert aligned region that do not correspond to
    # real tokens are left untouched here and should remain initialized to
    # -1 so DeepGEMM can skip them
    for start_m in tl.range(0, cur_expert_token_num, BLOCK_E):
        offs = start_m + off_expert
        mask = offs < cur_expert_token_num
        tl.store(
            m_indices_start_ptr + offs,
            cur_expert,
            mask=mask,
        )


@triton.jit
def _expert_prefix_layout_kernel(
    counts,
    starts,
    ends,
    NUM_EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.arange(0, BLOCK)
    count = tl.load(counts + expert, expert < NUM_EXPERTS, 0).to(tl.int64)
    aligned = round_up_128(count)
    end = tl.cumsum(aligned)
    tl.store(starts + expert, end - aligned, expert < NUM_EXPERTS)
    tl.store(ends + expert, end, expert < NUM_EXPERTS)


@triton.jit
def _or_scale_bits(left, right):
    return left | right


@triton.jit
def _fwd_kernel_ep_scatter_2(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_x_scale_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_tensor_scale_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    topk_num: tl.constexpr,
    expert_map,
    HAS_EXPERT_MAP: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
    SCALE_HIDDEN_SIZE: tl.constexpr,
    SCALE_HIDDEN_SIZE_PAD: tl.constexpr,
    PACK_SCALES: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)

    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE

    offset_in_s = tl.arange(0, SCALE_HIDDEN_SIZE_PAD)
    mask_s = offset_in_s < SCALE_HIDDEN_SIZE

    output_tensor_stride0 = output_tensor_stride0.to(tl.int64)

    for token_id in range(start_token_id, total_token_num, grid_num):
        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)
        to_copy_s = tl.load(
            recv_x_scale
            + token_id * recv_x_scale_stride0
            + offset_in_s * recv_x_scale_stride1,
            mask=mask_s,
            other=0.0,
        )
        if PACK_SCALES:
            # Match DeepGEMM's packing of the biased exponent bits. The
            # activation quantizer already rounded scales to powers of two;
            # packing must not perform a second rounding operation.
            bits = to_copy_s.to(tl.uint32, bitcast=True)
            bits = tl.reshape(bits, (SCALE_HIDDEN_SIZE_PAD // 4, 4))
            shifts = 23 - 8 * tl.arange(0, 4)
            parts = tl.where(
                shifts[None, :] >= 0,
                bits >> tl.maximum(shifts[None, :], 0),
                bits << 1,
            )
            packed = tl.reduce(parts, 1, _or_scale_bits)
            packed_offset = tl.arange(0, SCALE_HIDDEN_SIZE_PAD // 4)

        for topk_index in tl.range(0, topk_num, 1, num_stages=4):
            expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 + topk_index)

            if HAS_EXPERT_MAP:
                expert_id = apply_expert_map(expert_id, expert_map)

            if expert_id >= 0:
                dest_token_index = tl.atomic_add(expert_start_loc + expert_id, 1)
                dest_token_index_i64 = dest_token_index.to(tl.int64)
                tl.store(
                    output_index + token_id * output_index_stride0 + topk_index,
                    dest_token_index,
                )
                output_tensor_ptr = (
                    output_tensor + dest_token_index_i64 * output_tensor_stride0
                )
                output_tensor_scale_ptr = (
                    output_tensor_scale + dest_token_index * output_tensor_scale_stride0
                )
                tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)
                if PACK_SCALES:
                    tl.store(
                        output_tensor_scale_ptr
                        + packed_offset * output_tensor_scale_stride1,
                        packed,
                        mask=packed_offset < tl.cdiv(SCALE_HIDDEN_SIZE, 4),
                    )
                else:
                    tl.store(
                        output_tensor_scale_ptr + offset_in_s,
                        to_copy_s,
                        mask=mask_s,
                    )


@torch.no_grad()
def ep_scatter(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_map: torch.Tensor | None,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    m_indices: torch.Tensor | None,
    output_index: torch.Tensor,
    grouped_layout: torch.Tensor | None = None,
):
    BLOCK_E = 128  # token num of per expert is aligned to 128
    BLOCK_D = 128  # block size of quantization
    num_warps = 8
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    # grid = (triton.cdiv(hidden_size, BLOCK_D), num_experts)
    grid = num_experts

    assert expert_start_loc.shape[0] == num_experts

    if grouped_layout is not None:
        _expert_prefix_layout_kernel[(1,)](
            num_recv_tokens_per_expert,
            expert_start_loc,
            grouped_layout,
            NUM_EXPERTS=num_experts,
            BLOCK=triton.next_power_of_2(num_experts),
            num_warps=4,
        )
    else:
        assert m_indices is not None and m_indices.shape[0] % BLOCK_E == 0
        _fwd_kernel_ep_scatter_1[(grid,)](
            num_recv_tokens_per_expert,
            expert_start_loc,
            m_indices,
            num_experts=num_experts,
            num_warps=num_warps,
            BLOCK_E=BLOCK_E,
            BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
        )

    grid = min(recv_topk.shape[0], 1024 * 8)

    _fwd_kernel_ep_scatter_2[(grid,)](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x_scale,
        recv_x_scale.stride(0),
        recv_x_scale.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor_scale,
        output_tensor_scale.stride(0),
        output_tensor_scale.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        topk_num=recv_topk.shape[1],
        expert_map=expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        num_warps=num_warps,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
        SCALE_HIDDEN_SIZE=hidden_size // BLOCK_D,
        SCALE_HIDDEN_SIZE_PAD=(
            max(4, triton.next_power_of_2(hidden_size // BLOCK_D))
            if output_tensor_scale.dtype == torch.int32
            else triton.next_power_of_2(hidden_size // BLOCK_D)
        ),
        PACK_SCALES=output_tensor_scale.dtype == torch.int32,
    )
    return


def deepgemm_moe_permute(
    aq: torch.Tensor,
    aq_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    local_num_experts: int,
    expert_map: torch.Tensor | None,
    expert_tokens_meta: mk.ExpertTokensMetadata | None,
    aq_out: torch.Tensor | None = None,
    use_psum_layout: bool = False,
    pack_scales: bool = False,
):
    assert aq.ndim == 2
    assert topk_ids.dtype.is_signed, "The kernel uses -1 to represent invalid topk_ids"
    H = aq.size(1)
    device = aq.device

    block_m, block_k = get_mk_alignment_for_contiguous_layout()

    M_sum = compute_aligned_M(
        M=topk_ids.size(0),
        num_topk=topk_ids.size(1),
        local_num_experts=local_num_experts,
        alignment=block_m,
        expert_tokens_meta=expert_tokens_meta,
    )

    expert_start_loc = torch.empty(
        (local_num_experts), device=device, dtype=torch.int32
    )

    assert aq_out is None or aq_out.shape == (M_sum, H)
    if aq_out is None:
        aq_out = torch.empty((M_sum, H), device=device, dtype=aq.dtype)

    if pack_scales:
        assert use_psum_layout and aq_scale.dtype == torch.float32
        # Contiguous M and groups of four exponents match DeepGEMM's TMA
        # layout. Scatter writes the packed scales directly into final rows.
        aq_scale_out = torch.empty(
            (triton.cdiv(H // block_k, 4), M_sum), device=device, dtype=torch.int32
        ).T
    else:
        aq_scale_out = torch.empty(
            (M_sum, H // block_k), device=device, dtype=torch.float32
        )

    # DeepGEMM uses negative values in m_indices (here expert_ids) to mark
    # completely invalid / padded blocks that should be skipped. We always
    # initialize expert_ids to -1 so any row that is not explicitly written
    # by the scatter kernel will be treated as invalid and skipped by
    # DeepGEMM's scheduler.
    expert_ids = (
        None
        if use_psum_layout
        else torch.full((M_sum,), fill_value=-1, device=device, dtype=torch.int32)
    )
    # Entries for non-local experts are not written by ep_scatter. Keep them
    # invalid so auxiliary consumers cannot interpret uninitialized indices.
    inv_perm = torch.full(
        topk_ids.shape, fill_value=-1, device=device, dtype=torch.int32
    )

    expert_num_tokens = None
    if expert_tokens_meta is not None:
        expert_num_tokens = expert_tokens_meta.expert_num_tokens
    else:
        expert_num_tokens = count_expert_num_tokens(
            topk_ids, local_num_experts, expert_map
        )

    if use_psum_layout:
        grouped_layout = torch.empty(
            (local_num_experts,), device=device, dtype=torch.int32
        )
    else:
        assert expert_ids is not None
        grouped_layout = expert_ids

    ep_scatter(
        recv_x=aq,
        recv_x_scale=aq_scale,
        recv_topk=topk_ids,
        num_recv_tokens_per_expert=expert_num_tokens,
        expert_start_loc=expert_start_loc,
        expert_map=expert_map,
        output_tensor=aq_out,
        output_tensor_scale=aq_scale_out,
        m_indices=expert_ids,
        output_index=inv_perm,
        grouped_layout=grouped_layout if use_psum_layout else None,
    )

    # Keep the statically sized M_sum buffers. grouped_layout already tells
    # DeepGEMM where each expert's valid rows end, so slicing to its final
    # device value is redundant and would require a capture-unsafe .item().
    return aq_out, aq_scale_out, grouped_layout, inv_perm
