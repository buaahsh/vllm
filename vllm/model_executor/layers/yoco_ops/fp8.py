# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _silu_mul_quant_fp8_packed_kernel(
    input_ptr,
    output_q_ptr,
    output_scale_ptr,
    row_weights_ptr,
    negative_row_weights_ptr,
    row_indices_ptr,
    M,
    NUM_ROWS,
    input_stride_m,
    output_q_stride_m,
    output_scale_stride_m,
    output_scale_stride_k,
    clamp_limit,
    N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    HAS_ROW_WEIGHTS: tl.constexpr,
    HAS_NEGATIVE_ROW_WEIGHTS: tl.constexpr,
    HAS_ROW_INDICES: tl.constexpr,
    ROUND_BEFORE_QUANT: tl.constexpr,
    PACKED_SCALES: tl.constexpr,
):
    N_2: tl.constexpr = N // 2

    pid_pack = tl.program_id(0)
    pid_m = tl.program_id(1)
    m_offset = pid_m.to(tl.int64) * BLOCK_M

    if m_offset >= NUM_ROWS:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, GROUP_SIZE)
    logical_rows = m_offset + offs_m
    row_mask = logical_rows < NUM_ROWS
    if HAS_ROW_INDICES:
        rows = tl.load(row_indices_ptr + logical_rows, mask=row_mask, other=-1).to(
            tl.int64
        )
        row_mask &= (rows >= 0) & (rows < M)
    else:
        rows = logical_rows

    base_row_offset = rows[:, None] * input_stride_m
    base_out_offset = rows[:, None] * output_q_stride_m

    packed_scale = tl.zeros((BLOCK_M,), dtype=tl.int32)

    for pack_idx in tl.static_range(4):
        group_id = pid_pack * 4 + pack_idx

        if group_id < NUM_GROUPS:
            n_offset = group_id * GROUP_SIZE

            act_ptrs = input_ptr + base_row_offset + n_offset + offs_n[None, :]
            act_in = tl.load(act_ptrs, mask=row_mask[:, None], other=0.0)

            mul_ptrs = act_ptrs + N_2
            mul_in = tl.load(mul_ptrs, mask=row_mask[:, None], other=0.0)

            act_f32 = act_in.to(tl.float32)
            mul_f32 = mul_in.to(tl.float32)

            if HAS_CLAMP:
                act_f32 = tl.minimum(act_f32, clamp_limit)
                mul_f32 = tl.clamp(mul_f32, -clamp_limit, clamp_limit)

            y = (act_f32 * tl.sigmoid(act_f32)) * mul_f32
            if HAS_ROW_WEIGHTS:
                row_weights = tl.load(
                    row_weights_ptr + logical_rows, mask=row_mask, other=0.0
                ).to(tl.float32)
                if HAS_NEGATIVE_ROW_WEIGHTS:
                    negative_row_weights = tl.load(
                        negative_row_weights_ptr + logical_rows,
                        mask=row_mask,
                        other=0.0,
                    ).to(tl.float32)
                    y *= tl.where(
                        y < 0.0,
                        negative_row_weights[:, None],
                        row_weights[:, None],
                    )
                else:
                    y *= row_weights[:, None]
            if ROUND_BEFORE_QUANT:
                # Shared experts and existing callers retain their BF16
                # activation boundary. YOCO routed FP8 quantizes directly.
                y = y.to(tl.bfloat16).to(tl.float32)

            absmax = tl.max(tl.abs(y), axis=1)

            scale_raw = tl.maximum(absmax / fp8_max, 1e-4 / fp8_max)
            if ROUND_BEFORE_QUANT:
                exponent = tl.ceil(tl.log2(scale_raw))
                scale = tl.math.exp2(exponent)
                exponent_biased = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.int32)
            else:
                # Same UE8M0 ceiling as llm-train's fused quantizer. Using
                # exponent bits avoids log2 rounding at power-of-two edges.
                bits = scale_raw.to(tl.uint32, bitcast=True)
                exponent_biased = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(
                    tl.uint32
                )
                exponent_biased = tl.minimum(tl.maximum(exponent_biased, 1), 254).to(
                    tl.int32
                )
                scale = (exponent_biased << 23).to(tl.float32, bitcast=True)

            y_q = tl.clamp(y / scale[:, None], fp8_min, fp8_max)

            out_q_ptrs = output_q_ptr + base_out_offset + n_offset + offs_n[None, :]
            tl.store(
                out_q_ptrs,
                y_q.to(output_q_ptr.dtype.element_ty),
                mask=row_mask[:, None],
            )

            if PACKED_SCALES:
                packed_scale = packed_scale | (exponent_biased << (pack_idx * 8))
            else:
                tl.store(
                    output_scale_ptr
                    + rows * output_scale_stride_m
                    + group_id * output_scale_stride_k,
                    scale,
                    mask=row_mask,
                )

    if PACKED_SCALES:
        scale_ptrs = output_scale_ptr + pid_pack * output_scale_stride_k + rows
        tl.store(scale_ptrs, packed_scale, mask=row_mask)


def silu_mul_quant_fp8_triton(
    input: torch.Tensor,
    group_size: int = 128,
    output_q: torch.Tensor | None = None,
    clamp_limit: float | None = None,
    row_weights: torch.Tensor | None = None,
    negative_row_weights: torch.Tensor | None = None,
    row_indices: torch.Tensor | None = None,
    *,
    round_before_quant: bool = True,
    packed_scales: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize dense rows, or only the routed rows of a padded MoE buffer.

    With row_indices, weights are in routing order and each valid index must
    identify a unique input/output row. Invalid indices are skipped. Padding
    in output_q is untouched; consumers must ignore these unreferenced rows.
    """
    assert input.dim() == 2
    assert input.is_contiguous()

    M, N = input.shape
    N_2 = N // 2

    assert N_2 % group_size == 0

    fp8_dtype = torch.float8_e4m3fn
    finfo = torch.finfo(fp8_dtype)
    fp8_min, fp8_max = finfo.min, finfo.max

    num_groups_per_row = N_2 // group_size
    num_packed_groups = (num_groups_per_row + 3) // 4
    tma_aligned_M = ((M + 3) // 4) * 4

    if output_q is None:
        output_q = torch.empty((M, N_2), dtype=fp8_dtype, device=input.device)
    num_rows = M
    if row_indices is not None:
        assert row_indices.dim() == 1 and row_indices.is_contiguous()
        assert row_indices.dtype in (torch.int32, torch.int64)
        num_rows = row_indices.numel()
    if row_weights is not None:
        assert row_weights.dim() == 1
        assert row_weights.numel() == num_rows
        assert row_weights.is_contiguous()
    if negative_row_weights is not None:
        assert row_weights is not None
        assert negative_row_weights.shape == row_weights.shape
        assert negative_row_weights.is_contiguous()

    if packed_scales:
        output_scale = torch.zeros(
            (num_packed_groups, tma_aligned_M),
            dtype=torch.int32,
            device=input.device,
        ).T[:M, :]
    else:
        output_scale = torch.empty(
            (M, num_groups_per_row), dtype=torch.float32, device=input.device
        )

    BLOCK_M = 8
    grid = (num_packed_groups, triton.cdiv(num_rows, BLOCK_M))

    num_warps = max(4, group_size // 32)
    num_stages = 2

    has_clamp = clamp_limit is not None
    has_row_weights = row_weights is not None
    has_negative_row_weights = negative_row_weights is not None
    _silu_mul_quant_fp8_packed_kernel[grid](
        input,
        output_q,
        output_scale,
        row_weights if has_row_weights else input,
        negative_row_weights if has_negative_row_weights else input,
        row_indices if row_indices is not None else input,
        M,
        num_rows,
        input.stride(0),
        output_q.stride(0),
        output_scale.stride(0),
        output_scale.stride(1),
        clamp_limit if has_clamp else 0.0,
        N=N,
        NUM_GROUPS=num_groups_per_row,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        GROUP_SIZE=group_size,
        BLOCK_M=BLOCK_M,
        HAS_CLAMP=has_clamp,
        HAS_ROW_WEIGHTS=has_row_weights,
        HAS_NEGATIVE_ROW_WEIGHTS=has_negative_row_weights,
        HAS_ROW_INDICES=row_indices is not None,
        ROUND_BEFORE_QUANT=round_before_quant,
        PACKED_SCALES=packed_scales,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return output_q, output_scale


def silu_mul_quant_fp8_packed_triton(
    input: torch.Tensor,
    group_size: int = 128,
    output_q: torch.Tensor | None = None,
    clamp_limit: float | None = None,
    row_weights: torch.Tensor | None = None,
    negative_row_weights: torch.Tensor | None = None,
    row_indices: torch.Tensor | None = None,
    *,
    round_before_quant: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    return silu_mul_quant_fp8_triton(
        input,
        group_size,
        output_q,
        clamp_limit,
        row_weights,
        negative_row_weights,
        row_indices,
        round_before_quant=round_before_quant,
        packed_scales=True,
    )


def _silu_mul_quant_fp8_packed(
    input: torch.Tensor, clamp_limit: float
) -> tuple[torch.Tensor, torch.Tensor]:
    # Shared with routed MoE: retain the BF16 activation rounding boundary.
    return silu_mul_quant_fp8_packed_triton(input, clamp_limit=clamp_limit)


def _silu_mul_quant_fp8_packed_fake(
    input: torch.Tensor, clamp_limit: float
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = input.shape
    groups = (width // 2 // 128 + 3) // 4
    aligned_rows = ((rows + 3) // 4) * 4
    quantized = input.new_empty((rows, width // 2), dtype=torch.float8_e4m3fn)
    scales = input.new_empty((groups, aligned_rows), dtype=torch.int32).T[:rows]
    return quantized, scales


direct_register_custom_op(
    "yoco_silu_mul_quant_fp8_packed",
    _silu_mul_quant_fp8_packed,
    fake_impl=_silu_mul_quant_fp8_packed_fake,
)


def silu_mul_quant_fp8_packed(
    input: torch.Tensor, clamp_limit: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.yoco_silu_mul_quant_fp8_packed(input, clamp_limit)
