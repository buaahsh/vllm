# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO rotary operators and their numerical fallbacks."""

from __future__ import annotations

import torch
from torch import nn

from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

if HAS_TRITON:

    @triton.jit
    def _yoco_mul_rn(x, y):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[x, y],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _yoco_fma_rn(x, y, z):
        return tl.inline_asm_elementwise(
            "fma.rn.f32 $0, $1, $2, $3;",
            constraints="=f,f,f,f",
            args=[x, y, z],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _yoco_rotary_kernel(
        query_ptr,
        key_ptr,
        query_output_ptr,
        key_output_ptr,
        positions_ptr,
        cos_sin_cache_ptr,
        num_rows,
        query_row_stride,
        query_head_stride,
        key_row_stride,
        key_head_stride,
        positions_stride,
        QUERY_HEADS: tl.constexpr,
        KEY_HEADS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pair_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        query_pairs = num_rows * QUERY_HEADS * 64
        total_pairs = num_rows * (QUERY_HEADS + KEY_HEADS) * 64
        valid = pair_offsets < total_pairs
        is_query = pair_offsets < query_pairs
        local_offsets = tl.where(
            is_query,
            pair_offsets,
            pair_offsets - query_pairs,
        )
        num_heads = tl.where(is_query, QUERY_HEADS, KEY_HEADS)
        row = local_offsets // (num_heads * 64)
        head = (local_offsets // 64) % num_heads
        rotary_col = local_offsets % 64
        position = tl.load(
            positions_ptr + row * positions_stride,
            mask=valid,
            other=0,
        )
        cos = tl.load(
            cos_sin_cache_ptr + position * 128 + rotary_col,
            mask=valid,
            other=0.0,
        )
        sin = tl.load(
            cos_sin_cache_ptr + position * 128 + 64 + rotary_col,
            mask=valid,
            other=0.0,
        )
        row_stride = tl.where(is_query, query_row_stride, key_row_stride)
        head_stride = tl.where(is_query, query_head_stride, key_head_stride)
        input_offsets = row * row_stride + head * head_stride + rotary_col
        input_ptrs = tl.where(
            is_query,
            query_ptr + input_offsets,
            key_ptr + input_offsets,
        )
        first_half = tl.load(
            input_ptrs,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        second_half = tl.load(
            input_ptrs + 64,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        first_output = _yoco_fma_rn(
            first_half,
            cos,
            -_yoco_mul_rn(second_half, sin),
        )
        second_output = _yoco_fma_rn(
            second_half,
            cos,
            _yoco_mul_rn(first_half, sin),
        )
        output_base = row * num_heads * 128 + head * 128 + rotary_col
        output_ptrs = tl.where(
            is_query,
            query_output_ptr + output_base,
            key_output_ptr + output_base,
        )
        tl.store(
            output_ptrs,
            first_output,
            mask=valid,
        )
        tl.store(
            output_ptrs + 64,
            second_output,
            mask=valid,
        )

    @triton.jit
    def _yoco_qk_rms_clip_rotary_kernel(
        query_ptr,
        key_ptr,
        query_weight_ptr,
        key_weight_ptr,
        query_output_ptr,
        key_output_ptr,
        positions_ptr,
        cos_sin_cache_ptr,
        num_rows,
        query_num_groups,
        query_row_stride,
        query_head_stride,
        key_row_stride,
        key_head_stride,
        positions_stride,
        eps: tl.constexpr,
        limit: tl.constexpr,
        QUERY_HEADS: tl.constexpr,
        KEY_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
        query_scale_ptr=None,
        key_scale_ptr=None,
        value_ptr=None,
        value_output_ptr=None,
        value_scale_ptr=None,
        value_row_stride=0,
        value_head_stride=0,
        FP8_OUTPUT: tl.constexpr = False,
    ):
        group = tl.program_id(0)
        is_query = group < query_num_groups
        local_group = tl.where(is_query, group, group - query_num_groups)
        head_rows = local_group * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        num_heads = tl.where(is_query, QUERY_HEADS, KEY_HEADS)
        num_head_rows = num_rows * num_heads
        row_mask = head_rows < num_head_rows
        token = head_rows // num_heads
        head = head_rows % num_heads
        cols = tl.arange(0, HEAD_DIM)[None, :]

        row_stride = tl.where(is_query, query_row_stride, key_row_stride)
        head_stride = tl.where(is_query, query_head_stride, key_head_stride)
        input_base = token * row_stride + head * head_stride
        input_ptrs = tl.where(
            is_query,
            query_ptr + input_base,
            key_ptr + input_base,
        )
        values = tl.load(
            input_ptrs + cols,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        square_sum = tl.sum(tl.where(row_mask, values * values, 0.0), axis=1)[:, None]
        clip_coef = limit * tl.extra.cuda.libdevice.rsqrt(square_sum / HEAD_DIM + eps)
        clip_coef = tl.minimum(clip_coef, 1.0)

        # Match the unfused RMSClip -> BF16 tensor -> RoPE sequence. The
        # explicit BF16 round is the semantic boundary that used to be the
        # intermediate tensor store/load.
        clipped = values * clip_coef
        if HAS_WEIGHT:
            weight_ptrs = tl.where(
                is_query,
                query_weight_ptr + cols,
                key_weight_ptr + cols,
            )
            weight = tl.load(weight_ptrs, mask=row_mask, other=0.0).to(tl.float32)
            clipped = clipped * weight
        # The affine training kernel fuses the source-level pre-gamma BF16
        # cast away and materializes BF16 only after gamma multiplication.
        clipped = clipped.to(tl.bfloat16).to(tl.float32)
        first_half = cols < HEAD_DIM // 2
        partner_cols = tl.where(
            first_half,
            cols + HEAD_DIM // 2,
            cols - HEAD_DIM // 2,
        )
        partner = tl.load(
            input_ptrs + partner_cols,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        partner = partner * clip_coef
        if HAS_WEIGHT:
            partner_weight_ptrs = tl.where(
                is_query,
                query_weight_ptr + partner_cols,
                key_weight_ptr + partner_cols,
            )
            partner_weight = tl.load(
                partner_weight_ptrs,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            partner = partner * partner_weight
        partner = partner.to(tl.bfloat16).to(tl.float32)

        rotary_col = cols % (HEAD_DIM // 2)
        position = tl.load(
            positions_ptr + token * positions_stride,
            mask=row_mask,
            other=0,
        )
        cos = tl.load(
            cos_sin_cache_ptr + position * HEAD_DIM + rotary_col,
            mask=row_mask,
            other=0.0,
        )
        sin = tl.load(
            cos_sin_cache_ptr + position * HEAD_DIM + HEAD_DIM // 2 + rotary_col,
            mask=row_mask,
            other=0.0,
        )
        signed_product = _yoco_mul_rn(partner, sin)
        signed_product = tl.where(first_half, -signed_product, signed_product)
        output = _yoco_fma_rn(clipped, cos, signed_product)

        output_base = head_rows * HEAD_DIM
        output_ptrs = tl.where(
            is_query,
            query_output_ptr + output_base,
            key_output_ptr + output_base,
        )
        if FP8_OUTPUT:
            scale_ptr = tl.where(is_query, query_scale_ptr, key_scale_ptr)
            scale = tl.load(scale_ptr)
            output = output.to(tl.bfloat16).to(tl.float32)
            output = tl.clamp(output / scale, -448.0, 448.0)
        tl.store(output_ptrs + cols, output, mask=row_mask)
        if FP8_OUTPUT:  # noqa: SIM102
            if not is_query:
                value = tl.load(
                    value_ptr
                    + token * value_row_stride
                    + head * value_head_stride
                    + cols,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                v_scale = tl.load(value_scale_ptr)
                tl.store(
                    value_output_ptr + output_base + cols,
                    tl.clamp(value / v_scale, -448.0, 448.0),
                    mask=row_mask,
                )


def _yoco_rotary_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_output = torch.empty_like(query, memory_format=torch.contiguous_format)
    key_output = torch.empty_like(key, memory_format=torch.contiguous_format)
    num_rows = query.shape[0]
    query_heads = query.shape[1]
    key_heads = key.shape[1]
    block_size = 256
    num_pairs = num_rows * (query_heads + key_heads) * 64
    _yoco_rotary_kernel[(triton.cdiv(num_pairs, block_size),)](
        query,
        key,
        query_output,
        key_output,
        positions,
        cos_sin_cache,
        num_rows,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        positions.stride(0),
        QUERY_HEADS=query_heads,
        KEY_HEADS=key_heads,
        BLOCK_SIZE=block_size,
        num_warps=8,
        num_stages=1,
    )
    return query_output, key_output


def _yoco_rotary_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del positions, cos_sin_cache
    return (
        torch.empty_like(query, memory_format=torch.contiguous_format),
        torch.empty_like(key, memory_format=torch.contiguous_format),
    )


def _yoco_qk_rms_clip_rotary_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_output = torch.empty_like(query, memory_format=torch.contiguous_format)
    key_output = torch.empty_like(key, memory_format=torch.contiguous_format)
    num_rows = query.shape[0]
    query_heads = query.shape[1]
    key_heads = key.shape[1]
    block_rows = 8
    query_num_groups = triton.cdiv(num_rows * query_heads, block_rows)
    key_num_groups = triton.cdiv(num_rows * key_heads, block_rows)
    _yoco_qk_rms_clip_rotary_kernel[(query_num_groups + key_num_groups,)](
        query,
        key,
        query,
        key,
        query_output,
        key_output,
        positions,
        cos_sin_cache,
        num_rows,
        query_num_groups,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        positions.stride(0),
        eps=eps,
        limit=limit,
        QUERY_HEADS=query_heads,
        KEY_HEADS=key_heads,
        HEAD_DIM=128,
        BLOCK_ROWS=block_rows,
        HAS_WEIGHT=False,
        num_warps=4,
        num_stages=1,
    )
    return query_output, key_output


def _yoco_qk_rms_clip_rotary_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del positions, cos_sin_cache, eps, limit
    return (
        torch.empty_like(query, memory_format=torch.contiguous_format),
        torch.empty_like(key, memory_format=torch.contiguous_format),
    )


def _yoco_qk_rms_clip_rotary_weighted_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_output = torch.empty_like(query, memory_format=torch.contiguous_format)
    key_output = torch.empty_like(key, memory_format=torch.contiguous_format)
    query_weight = query_weight.to(torch.bfloat16).contiguous()
    key_weight = key_weight.to(torch.bfloat16).contiguous()
    num_rows = query.shape[0]
    query_heads = query.shape[1]
    key_heads = key.shape[1]
    # Two rows with one warp is the best balanced A6000 configuration across
    # decode and prefill sizes.  The fast path is allowed to choose a fixed
    # reduction tree; the align path keeps Inductor's shape-dependent tree.
    block_rows = 2
    query_num_groups = triton.cdiv(num_rows * query_heads, block_rows)
    key_num_groups = triton.cdiv(num_rows * key_heads, block_rows)
    _yoco_qk_rms_clip_rotary_kernel[(query_num_groups + key_num_groups,)](
        query,
        key,
        query_weight,
        key_weight,
        query_output,
        key_output,
        positions,
        cos_sin_cache,
        num_rows,
        query_num_groups,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        positions.stride(0),
        eps=eps,
        limit=limit,
        QUERY_HEADS=query_heads,
        KEY_HEADS=key_heads,
        HEAD_DIM=128,
        BLOCK_ROWS=block_rows,
        HAS_WEIGHT=True,
        num_warps=1,
        num_stages=1,
    )
    return query_output, key_output


def _yoco_qk_rms_clip_rotary_weighted_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del query_weight, key_weight, positions, cos_sin_cache, eps, limit
    return (
        torch.empty_like(query, memory_format=torch.contiguous_format),
        torch.empty_like(key, memory_format=torch.contiguous_format),
    )


def _yoco_qkv_clip_rotary_fp8_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_weight: torch.Tensor | None,
    key_weight: torch.Tensor | None,
    positions: torch.Tensor,
    cache: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    eps: float,
    limit: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        torch.empty_like(
            x, dtype=torch.float8_e4m3fn, memory_format=torch.contiguous_format
        )
        for x in (query, key, value)
    )


def _yoco_qkv_clip_rotary_fp8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_weight: torch.Tensor | None,
    key_weight: torch.Tensor | None,
    positions: torch.Tensor,
    cache: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    eps: float,
    limit: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert query.dtype == key.dtype == value.dtype == torch.bfloat16
    assert query.shape[-1] == key.shape[-1] == value.shape[-1] == 128
    assert key.shape == value.shape and query.shape[0] == key.shape[0]
    assert query_scale.numel() == key_scale.numel() == value_scale.numel() == 1
    assert (query_weight is None) == (key_weight is None)
    q, k, v = _yoco_qkv_clip_rotary_fp8_fake(
        query,
        key,
        value,
        query_weight,
        key_weight,
        positions,
        cache,
        query_scale,
        key_scale,
        value_scale,
        eps,
        limit,
    )
    tokens, q_heads, _ = query.shape
    k_heads = key.shape[1]
    weighted = query_weight is not None
    rows = 2 if weighted else 8
    q_groups = triton.cdiv(tokens * q_heads, rows)
    k_groups = triton.cdiv(tokens * k_heads, rows)
    if tokens:
        _yoco_qk_rms_clip_rotary_kernel[(q_groups + k_groups,)](
            query,
            key,
            query_weight if weighted else query,
            key_weight if weighted else key,
            q,
            k,
            positions,
            cache,
            tokens,
            q_groups,
            query.stride(0),
            query.stride(1),
            key.stride(0),
            key.stride(1),
            positions.stride(0),
            eps,
            limit,
            q_heads,
            k_heads,
            128,
            rows,
            weighted,
            query_scale,
            key_scale,
            value,
            v,
            value_scale,
            value.stride(0),
            value.stride(1),
            FP8_OUTPUT=True,
            num_warps=1 if weighted else 4,
            num_stages=1,
        )
    return q, k, v


@torch.compile
def _yoco_apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)

    def apply(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat((y1, y2), dim=-1).to(x.dtype)

    return apply(query), apply(key)


@torch.compile
def _yoco_align_rotary_embedding(
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The exact cache lookup and RoPE expression compiled by llm-train."""
    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)

    def apply(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat((y1, y2), dim=-1).to(x.dtype)

    return apply(query), apply(key)


class YOCORotaryEmbedding(nn.Module):
    """YOCO RoPE with llm-train's FP32 cos/sin cache semantics."""

    cos_sin_cache: torch.Tensor | None

    def __init__(
        self,
        head_size: int,
        max_position_embeddings: int,
        base: float,
        execution_mode: str = "fast",
    ) -> None:
        super().__init__()
        if execution_mode not in ("align", "fast"):
            raise ValueError(f"Unsupported YOCO execution mode: {execution_mode!r}")
        self.head_size = head_size
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.execution_mode = execution_mode
        self.register_buffer("cos_sin_cache", None, persistent=False)

    def _get_cos_sin_cache(self, device: torch.device) -> torch.Tensor:
        cache = self.cos_sin_cache
        if cache is not None and cache.device == device:
            return cache
        # llm-train constructs RoPE while the default device is CUDA. Generate
        # the FP32 cache on the execution device as well: CPU-generated trig
        # values differ by a few ULPs and can cross BF16/FP8 boundaries.
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(
                    0,
                    self.head_size,
                    2,
                    dtype=torch.float32,
                    device=device,
                )
                / self.head_size
            )
        )
        positions = torch.arange(
            self.max_position_embeddings,
            dtype=torch.float32,
            device=device,
        )
        freqs = torch.einsum("i,j -> ij", positions, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        self.cos_sin_cache = cache
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self._get_cos_sin_cache(query.device)
        query = query.view(query.shape[0], -1, self.head_size)
        key = key.view(key.shape[0], -1, self.head_size)
        positions = positions.to(device=query.device, dtype=torch.long)
        if self.execution_mode == "align":
            query, key = _yoco_align_rotary_embedding(cache, positions, query, key)
            return query.flatten(-2), key.flatten(-2)
        if (
            HAS_TRITON
            and query.is_cuda
            and query.dtype == torch.bfloat16
            and key.dtype == torch.bfloat16
            and query.stride(-1) == 1
            and key.stride(-1) == 1
            and self.head_size == 128
            and current_platform.is_cuda()
        ):
            query, key = torch.ops.vllm.yoco_rotary(query, key, positions, cache)
        else:
            cos_sin = cache.index_select(0, positions)
            cos, sin = cos_sin.chunk(2, dim=-1)
            query, key = _yoco_apply_rotary_emb(query, key, cos, sin)
        return query.flatten(-2), key.flatten(-2)


if HAS_TRITON and current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_rotary",
        op_func=_yoco_rotary_cuda,
        fake_impl=_yoco_rotary_fake,
    )


if HAS_TRITON and current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_qkv_clip_rotary_fp8",
        op_func=_yoco_qkv_clip_rotary_fp8,
        fake_impl=_yoco_qkv_clip_rotary_fp8_fake,
    )


if HAS_TRITON and current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_qk_rms_clip_rotary",
        op_func=_yoco_qk_rms_clip_rotary_cuda,
        fake_impl=_yoco_qk_rms_clip_rotary_fake,
    )


if HAS_TRITON and current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_qk_rms_clip_rotary_weighted",
        op_func=_yoco_qk_rms_clip_rotary_weighted_cuda,
        fake_impl=_yoco_qk_rms_clip_rotary_weighted_fake,
    )
