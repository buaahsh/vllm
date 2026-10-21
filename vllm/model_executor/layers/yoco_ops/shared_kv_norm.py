# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse Shared-KV norm pointwise work while retaining Torch's FP32 reduction."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _squares(X, R, V, S, N: tl.constexpr, ADD: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    if ADD:
        residual = tl.load(R + offsets, offsets < N, other=0).to(tl.float32)
        value = (residual + value).to(V.dtype.element_ty).to(tl.float32)
        tl.store(V + offsets, value, offsets < N)
    tl.store(S + offsets, value * value, offsets < N)


@triton.jit
def _epilogue(
    V,
    M,
    W,
    Y,
    N: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(V + offsets, offsets < N, other=0).to(tl.float32)
    mean = tl.load(M + offsets // D, offsets < N, other=0)
    inv = tl.extra.cuda.libdevice.rsqrt(mean + EPS)
    normalized = (value * inv).to(V.dtype.element_ty).to(tl.float32)
    weight = tl.load(W + offsets % D, offsets < N, other=0)
    if not PROMOTE:
        weight = weight.to(V.dtype.element_ty)
    result = normalized * weight.to(tl.float32)
    tl.store(Y + offsets, result, offsets < N)


def shared_kv_rms_norm(x, residual, weight, eps, promote):
    if residual is None:
        value = x
    else:
        value = torch.empty_like(x, dtype=torch.promote_types(x.dtype, residual.dtype))
    squares = torch.empty_like(value, dtype=torch.float32)
    n, dim = value.numel(), value.shape[-1]
    grid = (triton.cdiv(n, 1024),)
    _squares[grid](
        x,
        residual if residual is not None else x,
        value,
        squares,
        n,
        residual is not None,
        1024,
        enable_fp_fusion=False,
    )
    # Identical materialized squares, shape and operator preserve reduction order.
    mean = squares.mean(-1, keepdim=True)
    dtype = torch.promote_types(value.dtype, weight.dtype) if promote else value.dtype
    output = torch.empty_like(value, dtype=dtype)
    _epilogue[grid](
        value,
        mean,
        weight,
        output,
        n,
        dim,
        eps,
        promote,
        1024,
        enable_fp_fusion=False,
    )
    return output if residual is None else (output, value)
