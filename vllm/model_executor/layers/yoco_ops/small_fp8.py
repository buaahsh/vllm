# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit experimental small-M FP8 GEMMs for YOCO shared/latent projections.

Consumes the existing DeepGEMM UE8M0 operands without changing quantization.
No model/backend dispatch enables these candidates automatically.
"""

from dataclasses import dataclass

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

SHAPES = {(2560, 3072), (3072, 1280), (1024, 3072), (3072, 1024)}


@dataclass(frozen=True)
class SmallFP8Config:
    backend: str
    block_n: int
    num_warps: int = 4
    num_stages: int = 2


@triton.jit
def _unpack_scale(packed, group):
    exponent = (packed >> ((group % 4) * 8)) & 255
    return (exponent << 23).to(tl.float32, bitcast=True)


@triton.jit
def _small_fp8_tensor_kernel(
    A,
    B,
    As,
    Bs,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AS_M: tl.constexpr,
    AS_K: tl.constexpr,
    BS_N: tl.constexpr,
    BS_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.arange(0, 16)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = tl.arange(0, 128)
    acc = tl.zeros((BLOCK_N, 16), tl.float32)
    for group in range(K // 128):
        kk = group * 128 + offsets
        a = tl.load(
            A + kk[:, None] + rows[None, :] * K, mask=rows[None, :] < M, other=0.0
        )
        b = tl.load(
            B + cols[:, None] * K + kk[None, :], mask=cols[:, None] < N, other=0.0
        )
        ap = tl.load(As + rows * AS_M + (group // 4) * AS_K, mask=rows < M, other=0)
        bp = tl.load(Bs + cols * BS_N + (group // 4) * BS_K, mask=cols < N, other=0)
        a_scale = _unpack_scale(ap, group)
        b_scale = _unpack_scale(bp, group)
        acc += tl.dot(b, a) * b_scale[:, None] * a_scale[None, :]
    tl.store(
        C + cols[:, None] + rows[None, :] * N,
        acc,
        mask=(cols[:, None] < N) & (rows[None, :] < M),
    )


@triton.jit
def _small_fp8_direct_kernel(
    A,
    B,
    As,
    Bs,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AS_M: tl.constexpr,
    AS_K: tl.constexpr,
    BS_N: tl.constexpr,
    BS_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    kk = tl.arange(0, BLOCK_K)
    group = kk // 128
    valid = (cols[:, None] < N) & (kk[None, :] < K)
    b = tl.load(B + cols[:, None] * K + kk[None, :], mask=valid, other=0.0)
    bp = tl.load(
        Bs + cols[:, None] * BS_N + (group[None, :] // 4) * BS_K, mask=valid, other=0
    )
    b_value = b.to(tl.float32) * _unpack_scale(bp, group[None, :])
    for row in tl.static_range(M):
        a = tl.load(A + row * K + kk, mask=kk < K, other=0.0)
        ap = tl.load(As + row * AS_M + (group // 4) * AS_K, mask=kk < K, other=0)
        a_value = a.to(tl.float32) * _unpack_scale(ap, group)
        value = tl.sum(b_value * a_value[None, :], axis=1)
        tl.store(C + row * N + cols, value, mask=cols < N)


def small_fp8_mm(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    config: SmallFP8Config,
) -> torch.Tensor:
    """Multiply existing E4M3 operands with packed, per-row UE8M0 scales.

    Args:
        a: Contiguous quantized input [M, K], M in 1..8.
        weight: Contiguous quantized weight [N, K] for one supported shape.
        a_scale: INT32 packed input scales [M, ceil(K / 512)].
        weight_scale: INT32 packed scales broadcast to all N weight rows.
        config: Explicit candidate tactic; no automatic selection.

    Returns:
        BF16 output [M, N]. Reduction order can differ from DeepGEMM.
    """
    if not current_platform.is_device_capability(100):
        raise ValueError("YOCO small FP8 candidates require SM100")
    if a.ndim != 2 or weight.ndim != 2 or not 1 <= a.shape[0] <= 8:
        raise ValueError("Require [M,K] input with 1 <= M <= 8")
    m, k = a.shape
    n, wk = weight.shape
    if (n, k) not in SHAPES or wk != k:
        raise ValueError("Unsupported YOCO shared/latent matrix shape")
    operands = (a, weight, a_scale, weight_scale)
    if any(not t.is_cuda or t.device != a.device for t in operands):
        raise ValueError("All operands must be on the same CUDA device")
    if a.dtype != torch.float8_e4m3fn or weight.dtype != a.dtype:
        raise ValueError("Require E4M3 inputs and weights")
    if not a.is_contiguous() or not weight.is_contiguous():
        raise ValueError("Input and weight must be contiguous")
    packs = triton.cdiv(k, 512)
    if (
        a_scale.dtype != torch.int32
        or weight_scale.dtype != torch.int32
        or a_scale.shape != (m, packs)
        or weight_scale.shape != (n, packs)
    ):
        raise ValueError("Require native packed UE8M0 scale shapes")
    if config.num_warps not in (4, 8) or config.num_stages not in (1, 2, 3, 4):
        raise ValueError("Unsupported warp/stage configuration")
    allowed = {"tensor": (16, 32, 64), "direct": (1, 2, 4)}
    if config.backend not in allowed or config.block_n not in allowed[config.backend]:
        raise ValueError("Unsupported backend/N tile")
    output = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    args = (
        a,
        weight,
        a_scale,
        weight_scale,
        output,
        m,
        n,
        k,
        *a_scale.stride(),
        *weight_scale.stride(),
    )
    grid = (triton.cdiv(n, config.block_n),)
    if config.backend == "tensor":
        _small_fp8_tensor_kernel[grid](
            *args,
            BLOCK_N=config.block_n,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
    else:
        _small_fp8_direct_kernel[grid](
            *args,
            BLOCK_N=config.block_n,
            BLOCK_K=triton.next_power_of_2(k),
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
    return output
