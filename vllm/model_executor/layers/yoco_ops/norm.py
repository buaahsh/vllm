# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO norm operators and their numerical fallbacks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton
from vllm.utils.torch_utils import direct_register_custom_op

if HAS_TRITON:

    @triton.jit
    def _yoco_bf16_add(x, y):
        return tl.inline_asm_elementwise(
            "add.rn.bf16 $0, $1, $2;",
            constraints="=h,h,h",
            args=[x, y],
            dtype=tl.bfloat16,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _yoco_rms_clip_kernel(
        x_ptr,
        output_ptr,
        num_rows,
        eps: tl.constexpr,
        limit: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        cols = tl.arange(0, HEAD_DIM)[None, :]
        row_mask = rows < num_rows
        values = tl.load(
            x_ptr + rows * HEAD_DIM + cols,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        square_sum = tl.sum(tl.where(row_mask, values * values, 0.0), axis=1)[:, None]
        clip_coef = limit * tl.extra.cuda.libdevice.rsqrt(square_sum / HEAD_DIM + eps)
        clip_coef = tl.minimum(clip_coef, 1.0)
        tl.store(
            output_ptr + rows * HEAD_DIM + cols,
            values * clip_coef,
            mask=row_mask,
        )

    @triton.jit
    def _yoco_weighted_rms_clip_kernel(
        x_ptr,
        weight_ptr,
        output_ptr,
        num_tokens,
        num_heads,
        token_stride,
        head_stride,
        eps: tl.constexpr,
        limit: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        ROUND_BEFORE_WEIGHT: tl.constexpr,
        scale_ptr=None,
        value_ptr=None,
        value_output_ptr=None,
        value_scale_ptr=None,
        value_stride_m=0,
        FP8_OUTPUT: tl.constexpr = False,
        QUANT_VALUE: tl.constexpr = False,
    ):
        head_rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        cols = tl.arange(0, HEAD_DIM)[None, :]
        row_mask = head_rows < num_tokens * num_heads
        token = head_rows // num_heads
        head = head_rows % num_heads
        input_offsets = token * token_stride + head * head_stride + cols
        values = tl.load(
            x_ptr + input_offsets,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        square_sum = tl.sum(tl.where(row_mask, values * values, 0.0), axis=1)[:, None]
        clip_coef = limit * tl.extra.cuda.libdevice.rsqrt(square_sum / HEAD_DIM + eps)
        clip_coef = tl.minimum(clip_coef, 1.0)

        clipped = values * clip_coef
        if ROUND_BEFORE_WEIGHT:
            # Preserve the source-level BF16 boundary before applying gamma.
            clipped = clipped.to(tl.bfloat16).to(tl.float32)
        weight = tl.load(weight_ptr + cols).to(tl.float32)
        result = clipped * weight
        if FP8_OUTPUT:
            result = result.to(tl.bfloat16).to(tl.float32)
            result = tl.clamp(
                result * tl.div_rn(1.0, tl.load(scale_ptr)), -448.0, 448.0
            )
        tl.store(output_ptr + head_rows * HEAD_DIM + cols, result, mask=row_mask)
        if QUANT_VALUE:
            value = tl.load(
                value_ptr + token * value_stride_m + head * HEAD_DIM + cols,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            tl.store(
                value_output_ptr + head_rows * HEAD_DIM + cols,
                tl.clamp(value / tl.load(value_scale_ptr), -448.0, 448.0),
                mask=row_mask,
            )

    @triton.jit
    def _yoco_rms_norm_kernel(
        x_ptr,
        weight_ptr,
        output_ptr,
        num_rows,
        eps: tl.constexpr,
        HIDDEN_SIZE: tl.constexpr,
        REDUCTION_BLOCK: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        row_mask = row < num_rows
        cols = tl.arange(0, REDUCTION_BLOCK)[None, :]
        square_acc = tl.full([BLOCK_ROWS, REDUCTION_BLOCK], 0.0, tl.float32)

        for offset in tl.range(0, HIDDEN_SIZE, REDUCTION_BLOCK):
            hidden_offsets = offset + cols
            mask = (hidden_offsets < HIDDEN_SIZE) & row_mask
            values = tl.load(
                x_ptr + row * HIDDEN_SIZE + hidden_offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            square_acc = tl.where(mask, square_acc + values * values, square_acc)

        square_sum = tl.sum(square_acc, axis=1)[:, None]
        inv_rms = tldevice.rsqrt(square_sum / HIDDEN_SIZE + eps)

        for offset in tl.range(0, HIDDEN_SIZE, REDUCTION_BLOCK):
            hidden_offsets = offset + cols
            mask = (hidden_offsets < HIDDEN_SIZE) & row_mask
            values = tl.load(
                x_ptr + row * HIDDEN_SIZE + hidden_offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            weight = tl.load(
                weight_ptr + hidden_offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            tl.store(
                output_ptr + row * HIDDEN_SIZE + hidden_offsets,
                values * inv_rms * weight,
                mask=mask,
            )

    @triton.jit
    def _yoco_fused_add_rms_norm_kernel(
        x_ptr,
        residual_ptr,
        weight_ptr,
        output_ptr,
        residual_out_ptr,
        num_rows,
        eps: tl.constexpr,
        HIDDEN_SIZE: tl.constexpr,
        REDUCTION_BLOCK: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        NATIVE_BF16_ADD: tl.constexpr,
    ):
        row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        row_mask = row < num_rows
        cols = tl.arange(0, REDUCTION_BLOCK)[None, :]
        square_acc = tl.full([BLOCK_ROWS, REDUCTION_BLOCK], 0.0, tl.float32)

        # The BF16 chain uses a native BF16 add on SM90+. Other paths add in
        # FP32, then round to storage. Statistics normalize the stored value.
        for offset in tl.range(0, HIDDEN_SIZE, REDUCTION_BLOCK):
            hidden_offsets = offset + cols
            mask = (hidden_offsets < HIDDEN_SIZE) & row_mask
            offsets = row * HIDDEN_SIZE + hidden_offsets
            x = tl.load(
                x_ptr + offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            residual = tl.load(
                residual_ptr + offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            if NATIVE_BF16_ADD:
                values = _yoco_bf16_add(x, residual)
            else:
                values = x.to(tl.float32) + residual.to(tl.float32)
            values = values.to(residual_out_ptr.dtype.element_ty).to(tl.float32)
            tl.store(residual_out_ptr + offsets, values, mask=mask)
            square_acc = tl.where(mask, square_acc + values * values, square_acc)

        square_sum = tl.sum(square_acc, axis=1)[:, None]
        inv_rms = tldevice.rsqrt(square_sum / HIDDEN_SIZE + eps)

        for offset in tl.range(0, HIDDEN_SIZE, REDUCTION_BLOCK):
            hidden_offsets = offset + cols
            mask = (hidden_offsets < HIDDEN_SIZE) & row_mask
            offsets = row * HIDDEN_SIZE + hidden_offsets
            values = tl.load(
                residual_out_ptr + offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            weight = tl.load(
                weight_ptr + hidden_offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            tl.store(
                output_ptr + offsets,
                values * inv_rms * weight,
                mask=mask,
            )


def _yoco_rms_clip_cuda(
    x: torch.Tensor,
    eps: float,
    limit: float,
) -> torch.Tensor:
    x_contiguous = x.contiguous()
    output = torch.empty_like(x_contiguous)
    num_rows = x_contiguous.numel() // x_contiguous.shape[-1]
    _yoco_rms_clip_kernel[(triton.cdiv(num_rows, 8),)](
        x_contiguous,
        output,
        num_rows,
        eps=eps,
        limit=limit,
        HEAD_DIM=128,
        BLOCK_ROWS=8,
        num_warps=4,
        num_stages=1,
    )
    return output


def _yoco_rms_clip_fake(
    x: torch.Tensor,
    eps: float,
    limit: float,
) -> torch.Tensor:
    return torch.empty_like(x)


def _yoco_weighted_rms_clip_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    limit: float,
) -> torch.Tensor:
    assert x.ndim == 3 and x.shape[-1] == 128
    num_tokens, num_heads, _ = x.shape
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    num_head_rows = num_tokens * num_heads
    if num_head_rows == 0:
        return output
    # B200 CUDA-graph tuning for L3's 64 cross-Q heads. Larger row tiles
    # amortize scheduling overhead without changing each head's reduction tree.
    block_rows = 16 if num_head_rows < 12288 else 32
    _yoco_weighted_rms_clip_kernel[(triton.cdiv(num_head_rows, block_rows),)](
        x,
        weight,
        output,
        num_tokens,
        num_heads,
        x.stride(0),
        x.stride(1),
        eps=eps,
        limit=limit,
        HEAD_DIM=128,
        BLOCK_ROWS=block_rows,
        ROUND_BEFORE_WEIGHT=True,
        num_warps=8,
        num_stages=1,
    )
    return output


def _yoco_align_weighted_rms_clip_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    limit: float,
) -> torch.Tensor:
    assert x.ndim == 3 and x.shape[-1] == 128
    num_tokens, num_heads, _ = x.shape
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    num_head_rows = num_tokens * num_heads
    if num_head_rows == 0:
        return output
    # Fix the launch layout as well as the per-head reduction tree. Align
    # must not switch between an Inductor expression and this kernel at M=128.
    block_rows = 16
    _yoco_weighted_rms_clip_kernel[(triton.cdiv(num_head_rows, block_rows),)](
        x,
        weight,
        output,
        num_tokens,
        num_heads,
        x.stride(0),
        x.stride(1),
        eps=eps,
        limit=limit,
        HEAD_DIM=128,
        BLOCK_ROWS=block_rows,
        ROUND_BEFORE_WEIGHT=False,
        num_warps=8,
        num_stages=1,
    )
    return output


def _yoco_weighted_rms_clip_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    limit: float,
) -> torch.Tensor:
    del weight, eps, limit
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def _run_yoco_rms_norm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    reduction_block: int,
) -> torch.Tensor:
    x_contiguous = x.contiguous()
    weight_contiguous = weight.to(torch.bfloat16).contiguous()
    output = torch.empty_like(x_contiguous, dtype=torch.bfloat16)
    num_rows = x_contiguous.numel() // x_contiguous.shape[-1]
    if num_rows == 0:
        return output
    block_rows = 1
    _yoco_rms_norm_kernel[(triton.cdiv(num_rows, block_rows),)](
        x_contiguous,
        weight_contiguous,
        output,
        num_rows,
        eps=eps,
        HIDDEN_SIZE=x_contiguous.shape[-1],
        REDUCTION_BLOCK=reduction_block,
        BLOCK_ROWS=block_rows,
        num_warps=16,
        num_stages=1,
    )
    return output


def _yoco_rms_norm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    num_rows = x.numel() // x.shape[-1]
    reduction_block = 4096 if num_rows >= 128 else 2048
    return _run_yoco_rms_norm_cuda(x, weight, eps, reduction_block)


def _yoco_align_rms_norm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    # One FP32 reduction tree per hidden size, independent of token count,
    # graph padding, input dtype, and enclosing compilation context.
    return _run_yoco_rms_norm_cuda(x, weight, eps, triton.next_power_of_2(x.shape[-1]))


def _yoco_rms_norm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x, dtype=torch.bfloat16)


def _yoco_fused_add_rms_norm_cuda(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows = x.numel() // x.shape[-1]
    reduction_block = 4096 if num_rows >= 128 else 2048
    return _run_yoco_fused_add_rms_norm_cuda(
        x, residual, weight, eps, reduction_block, residual.dtype
    )


def _yoco_align_fused_add_rms_norm_cuda(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _run_yoco_fused_add_rms_norm_cuda(
        x, residual, weight, eps, triton.next_power_of_2(x.shape[-1])
    )


def _yoco_bf16_add_rms_norm_cuda(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dtype == residual.dtype == torch.bfloat16
    num_rows = x.numel() // x.shape[-1]
    capability = current_platform.get_device_capability()
    return _run_yoco_fused_add_rms_norm_cuda(
        x,
        residual,
        weight,
        eps,
        4096 if num_rows >= 128 else 2048,
        torch.bfloat16,
        native_bf16_add=capability is not None and capability.major >= 9,
    )


def _run_yoco_fused_add_rms_norm_cuda(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    reduction_block: int,
    residual_dtype: torch.dtype = torch.float32,
    native_bf16_add: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_contiguous = x.contiguous()
    residual_contiguous = residual.contiguous()
    weight_contiguous = weight.to(torch.bfloat16).contiguous()
    output = torch.empty_like(x_contiguous, dtype=torch.bfloat16)
    residual_out = torch.empty_like(residual_contiguous, dtype=residual_dtype)
    num_rows = x_contiguous.numel() // x_contiguous.shape[-1]
    if num_rows == 0:
        return output, residual_out
    block_rows = 1
    _yoco_fused_add_rms_norm_kernel[(triton.cdiv(num_rows, block_rows),)](
        x_contiguous,
        residual_contiguous,
        weight_contiguous,
        output,
        residual_out,
        num_rows,
        eps=eps,
        HIDDEN_SIZE=x_contiguous.shape[-1],
        REDUCTION_BLOCK=reduction_block,
        BLOCK_ROWS=block_rows,
        NATIVE_BF16_ADD=native_bf16_add,
        num_warps=16,
        num_stages=1,
    )
    return output, residual_out


def _yoco_fused_add_rms_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(x, dtype=torch.bfloat16),
        torch.empty_like(residual),
    )


def _yoco_align_fused_add_rms_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(x, dtype=torch.bfloat16),
        torch.empty_like(residual, dtype=torch.float32),
    )


def _yoco_residual_dtype(execution_mode: str) -> torch.dtype:
    if execution_mode == "fast" and (
        envs.VLLM_YOCO_BF16_RESIDUAL
        or envs.VLLM_YOCO_BF16_CHAIN
        or envs.VLLM_YOCO_BF16_REDUCTIONS
    ):
        return torch.bfloat16
    return torch.float32


def _yoco_add_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    residual_dtype: torch.dtype,
    bf16_chain: bool = False,
) -> torch.Tensor:
    if bf16_chain:
        return residual.to(torch.bfloat16) + x.to(torch.bfloat16)
    return (residual.float() + x.float()).to(residual_dtype)


@torch.compile
def _yoco_align_rms_clip(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    limit: float,
) -> torch.Tensor:
    """The exact affine RMSClip expression compiled by llm-train."""
    x_float = x.float()
    clip_coef = (
        limit * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + eps)
    ).clamp(max=1.0)
    return (x_float * clip_coef).to(x.dtype) * weight.to(x.dtype)


@torch.compile
def _yoco_align_rms_clip_no_weight(
    x: torch.Tensor,
    eps: float,
    limit: float,
) -> torch.Tensor:
    """The exact non-affine RMSClip expression compiled by llm-train."""
    x_float = x.float()
    clip_coef = (
        limit * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + eps)
    ).clamp(max=1.0)
    return (x_float * clip_coef).to(x.dtype)


class RMSClip(nn.Module):
    """RMS-based clipping for YOCO ``qk_rms_clip`` models.

    Scales each ``head_dim`` slice by ``clamp(limit / rms, max=1.0)`` where
    ``rms = sqrt(mean(x**2, -1) + eps)``.  The optional affine weight is
    controlled by ``qk_rms_gamma``, matching training's ``RMSClip``.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        limit: float = 3.0,
        has_weight: bool = False,
        execution_mode: str = "fast",
    ) -> None:
        super().__init__()
        if execution_mode not in ("align", "fast"):
            raise ValueError(f"Unsupported YOCO execution mode: {execution_mode!r}")
        self.dim = dim
        self.eps = eps
        self.limit = limit
        self.execution_mode = execution_mode
        if has_weight:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, limit={self.limit}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.execution_mode == "align":
            device_capability = current_platform.get_device_capability()
            if (
                HAS_TRITON
                and x.is_cuda
                and x.dtype == torch.bfloat16
                and x.ndim == 3
                and x.shape[-1] == 128
                and device_capability is not None
                and device_capability.major == 10
                and self.weight is not None
            ):
                return torch.ops.vllm.yoco_align_weighted_rms_clip(
                    x, self.weight, self.eps, self.limit
                )
            if self.weight is None:
                if (
                    HAS_TRITON
                    and x.is_cuda
                    and x.dtype == torch.bfloat16
                    and x.shape[-1] == 128
                    and current_platform.is_cuda()
                ):
                    return torch.ops.vllm.yoco_rms_clip(x, self.eps, self.limit)
                return _yoco_align_rms_clip_no_weight(x, self.eps, self.limit)
            return _yoco_align_rms_clip(x, self.weight, self.eps, self.limit)
        if (
            HAS_TRITON
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.shape[-1] == 128
            and current_platform.is_cuda()
            and self.weight is None
        ):
            return torch.ops.vllm.yoco_rms_clip(x, self.eps, self.limit)
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        clip_coef = (self.limit * torch.rsqrt(variance + self.eps)).clamp(max=1.0)
        x = (x * clip_coef).to(orig_dtype)
        if self.weight is not None:
            x = x * self.weight.to(orig_dtype)
        return x


@torch.compile
def _yoco_align_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """The exact BF16 RMSNorm expression compiled by llm-train."""
    return F.rms_norm(
        x.to(torch.bfloat16),
        (x.shape[-1],),
        weight=weight.to(torch.bfloat16),
        eps=eps,
    )


class RMSNorm(nn.Module):
    """RMSNorm matching llm-train's compiled reduction semantics.

    Inductor keeps the residual input and reduction in FP32, loads the BF16
    weight as FP32, and casts only the output to BF16. The CUDA path below
    preserves Inductor's reduction order for hidden size 3072. Fast can opt
    into BF16 residual storage with FP32 statistics, normalizing the rounded
    stored sum. The BF16 chain also uses native BF16 addition on SM90+.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
        execution_mode: str = "fast",
    ) -> None:
        super().__init__()
        if execution_mode not in ("align", "fast"):
            raise ValueError(f"Unsupported YOCO execution mode: {execution_mode!r}")
        self.dim = dim
        self.eps = eps
        self.execution_mode = execution_mode
        self.residual_dtype = _yoco_residual_dtype(execution_mode)
        self.bf16_chain = execution_mode == "fast" and envs.VLLM_YOCO_BF16_CHAIN
        self.bf16_reductions = (
            execution_mode == "fast" and envs.VLLM_YOCO_BF16_REDUCTIONS
        )
        weight = torch.ones(dim, dtype=dtype or torch.get_default_dtype())
        if has_weight:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.bf16_reductions:
            x = x.to(torch.bfloat16)
            if residual is None:
                return torch.ops.vllm.yoco_bf16_rms_reduction(x, self.weight, self.eps)
            return torch.ops.vllm.yoco_bf16_add_rms_reduction(
                x, residual.to(torch.bfloat16), self.weight, self.eps
            )
        if self.bf16_chain:
            x = x.to(torch.bfloat16)
            if residual is not None:
                residual = residual.to(torch.bfloat16)
        if residual is not None:
            if self.execution_mode == "align":
                if (
                    HAS_TRITON
                    and x.is_cuda
                    and residual.is_cuda
                    and x.dtype in (torch.bfloat16, torch.float32)
                    and residual.dtype == torch.float32
                    and x.shape == residual.shape
                    and x.shape[-1] in (1024, 3072)
                    and current_platform.is_cuda()
                ):
                    return torch.ops.vllm.yoco_align_fused_add_rms_norm(
                        x, residual, self.weight, self.eps
                    )
                residual_out = residual + x.float()
                normalized = self.forward(residual_out)
                assert isinstance(normalized, torch.Tensor)
                return normalized, residual_out
            if (
                HAS_TRITON
                and x.is_cuda
                and residual.is_cuda
                and x.dtype in (torch.bfloat16, torch.float32)
                and residual.dtype == self.residual_dtype
                and x.shape == residual.shape
                and x.shape[-1] == 3072
                and current_platform.is_cuda()
            ):
                if self.bf16_chain:
                    return torch.ops.vllm.yoco_bf16_add_rms_norm(
                        x, residual, self.weight, self.eps
                    )
                return torch.ops.vllm.yoco_fused_add_rms_norm(
                    x, residual, self.weight, self.eps
                )
            residual_out = _yoco_add_residual(
                x, residual, self.residual_dtype, self.bf16_chain
            )
            normalized = self.forward(residual_out)
            assert isinstance(normalized, torch.Tensor)
            return normalized, residual_out
        if self.execution_mode == "align":
            if (
                HAS_TRITON
                and x.is_cuda
                and x.dtype in (torch.bfloat16, torch.float32)
                and x.shape[-1] in (1024, 3072)
                and current_platform.is_cuda()
            ):
                # Inductor may select a different RMS reduction tree when this
                # expression is compiled inside the full model. Keep the tree
                # fixed so BF16 rounding matches llm-train for every batch M.
                return torch.ops.vllm.yoco_align_rms_norm(x, self.weight, self.eps)
            return _yoco_align_rms_norm(x, self.weight, self.eps)
        if x.is_cuda and x.shape[-1] == 1024:
            # The latent norms use the same expression in both modes.  On
            # B200, Inductor's compiled reduction is faster than the eager
            # fallback while remaining bitwise-aligned with llm-train.
            return _yoco_align_rms_norm(x, self.weight, self.eps)
        if (
            HAS_TRITON
            and x.is_cuda
            and x.dtype in (torch.float32, self.residual_dtype)
            and x.shape[-1] == 3072
            and current_platform.is_cuda()
        ):
            return torch.ops.vllm.yoco_rms_norm(x, self.weight, self.eps)
        return F.rms_norm(
            x.to(torch.bfloat16),
            (x.shape[-1],),
            weight=self.weight.to(torch.bfloat16),
            eps=self.eps,
        )


def _build_qk_norm(
    config: PretrainedConfig,
    head_dim: int,
    rms_eps: float,
    execution_mode: str = "fast",
):
    """Build the per-head Q/K normalization module for YOCO attention.

    Three mutually exclusive modes, matching training (``llm/arch/attention.py``):
    * ``qk_rms_clip=True``  -> :class:`RMSClip` (clips outliers).
    * ``qk_norm=True``      -> :class:`RMSNorm`.
    * otherwise             -> ``None`` (no Q/K norm).
    """
    # Older exported YOCO-v2 configs omitted this field and were weight-free.
    has_weight = bool(getattr(config, "qk_rms_gamma", False))
    if bool(getattr(config, "qk_rms_clip", False)):
        limit = float(getattr(config, "qk_rms_limit", 3.0))
        return RMSClip(
            head_dim,
            eps=rms_eps,
            limit=limit,
            has_weight=has_weight,
            execution_mode=execution_mode,
        )
    if bool(getattr(config, "qk_norm", False)):
        return RMSNorm(
            head_dim,
            eps=rms_eps,
            has_weight=has_weight,
            execution_mode=execution_mode,
        )
    return None


def _apply_per_head_norm(
    x: torch.Tensor, num_heads: int, head_dim: int, norm: nn.Module
) -> torch.Tensor:
    """Apply ``norm`` independently to each head's ``head_dim`` slice.

    ``x`` has shape ``(n_tokens, num_heads * head_dim)``.
    """
    x = x.unflatten(-1, (num_heads, head_dim))
    x = norm(x)
    return x.flatten(-2, -1)


if HAS_TRITON and current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_rms_clip",
        op_func=_yoco_rms_clip_cuda,
        fake_impl=_yoco_rms_clip_fake,
    )


if HAS_TRITON and current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_weighted_rms_clip",
        op_func=_yoco_weighted_rms_clip_cuda,
        fake_impl=_yoco_weighted_rms_clip_fake,
    )


if HAS_TRITON and current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_align_weighted_rms_clip",
        op_func=_yoco_align_weighted_rms_clip_cuda,
        fake_impl=_yoco_weighted_rms_clip_fake,
    )


if HAS_TRITON and current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_rms_norm",
        op_func=_yoco_rms_norm_cuda,
        fake_impl=_yoco_rms_norm_fake,
    )


if HAS_TRITON and current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_align_rms_norm",
        op_func=_yoco_align_rms_norm_cuda,
        fake_impl=_yoco_rms_norm_fake,
    )


if HAS_TRITON and current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_fused_add_rms_norm",
        op_func=_yoco_fused_add_rms_norm_cuda,
        fake_impl=_yoco_fused_add_rms_norm_fake,
    )


if HAS_TRITON and current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_bf16_add_rms_norm",
        op_func=_yoco_bf16_add_rms_norm_cuda,
        fake_impl=_yoco_fused_add_rms_norm_fake,
    )


if HAS_TRITON and current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_align_fused_add_rms_norm",
        op_func=_yoco_align_fused_add_rms_norm_cuda,
        fake_impl=_yoco_align_fused_add_rms_norm_fake,
    )
