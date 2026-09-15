# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental BF16 scalar math; this does not change GEMM accumulators."""

import struct
from functools import lru_cache

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.bf16 $0, $1, $2;",
        "=h,h,h",
        [a, b],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.bf16 $0, $1, $2;",
        "=h,h,h",
        [a, b],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.bf16 $0, $1, $2;",
        "=h,h,h",
        [a, b],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _constant(bits: tl.constexpr):
    return tl.full((), bits, tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _lookup(x, table, function: tl.constexpr):
    index = x.to(tl.uint16, bitcast=True).to(tl.int32)
    return tl.load(table + function * 65536 + index)


@triton.jit
def _maximum(x):
    bits = x.to(tl.uint16, bitcast=True).to(tl.uint32)
    ordered = tl.where((bits & 32768) != 0, (~bits) & 65535, bits | 32768)
    largest = tl.max(ordered, axis=0)
    value = tl.where((largest & 32768) != 0, largest & 32767, (~largest) & 65535)
    return value.to(tl.uint16).to(tl.bfloat16, bitcast=True)


def _bits(value: float) -> int:
    bits = struct.unpack("I", struct.pack("f", value))[0]
    return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16


@lru_cache(maxsize=8)
def _tables(device: torch.device) -> torch.Tensor:
    # Generate once on the host in FP64. GPU kernels only load BF16 values;
    # CUDA's BF16 rsqrt/exp wrappers would otherwise execute FP32 instructions.
    raw = torch.arange(65536, dtype=torch.int32, device="cpu").to(torch.uint16)
    values = raw.view(torch.bfloat16).double()
    table = torch.stack((values.rsqrt(), values.exp(), values.log())).bfloat16()
    return table.to(device)


def _check(x: torch.Tensor) -> None:
    capability = current_platform.get_device_capability()
    if not x.is_cuda or capability is None or capability.major < 9:
        raise RuntimeError("YOCO native BF16 reductions require CUDA SM90 or newer")
    assert x.dtype == torch.bfloat16


@triton.jit
def _norm_kernel(
    X,
    R,
    W,
    TABLE,
    OUT,
    RO,
    H: tl.constexpr,
    B: tl.constexpr,
    INV_H: tl.constexpr,
    EPS: tl.constexpr,
    ADD: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * H + col, col < H, other=0)
    if ADD:
        r = tl.load(R + row * H + col, col < H, other=0)
        x = _add(x, r)
        tl.store(RO + row * H + col, x, col < H)
    square = _mul(x, x)
    total = tl.reduce(square, axis=0, combine_fn=_add)
    mean = _mul(total, _constant(INV_H))
    inv = _lookup(_add(mean, _constant(EPS)), TABLE, 0)
    w = tl.load(W + col, col < H, other=0)
    output = _mul(_mul(x, inv), w)
    tl.store(OUT + row * H + col, output, col < H)


def _norm_fake(x, weight, eps):
    return torch.empty_like(x, dtype=torch.bfloat16)


def _add_norm_fake(x, residual, weight, eps):
    return torch.empty_like(x), torch.empty_like(residual)


def _run_norm(x, residual, weight, eps):
    _check(x)
    x = x.contiguous()
    weight = weight.to(torch.bfloat16).contiguous()
    out = torch.empty_like(x)
    r_out = torch.empty_like(x) if residual is not None else out
    if residual is not None:
        assert residual.shape == x.shape and residual.dtype == torch.bfloat16
        residual = residual.contiguous()
    if x.numel():
        width = x.shape[-1]
        _norm_kernel[(x.numel() // width,)](
            x,
            residual if residual is not None else x,
            weight,
            _tables(x.device),
            out,
            r_out,
            width,
            triton.next_power_of_2(width),
            _bits(1.0 / width),
            _bits(eps),
            residual is not None,
            num_warps=4 if width <= 1024 else 8,
        )
    return out, r_out


def _norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _run_norm(x, None, weight, eps)[0]


def _add_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return _run_norm(x, residual, weight, eps)


@triton.jit
def _logprob_max(X, MAX, V: tl.constexpr, C: tl.constexpr, B: tl.constexpr):
    row, chunk = tl.program_id(0), tl.program_id(1)
    col = chunk * B + tl.arange(0, B)
    x = tl.load(X + row * V + col, col < V, other=float("-inf"))
    tl.store(MAX + row * C + chunk, _maximum(x))


@triton.jit
def _logprob_sum(
    X,
    MAX,
    SUM,
    TABLE,
    V: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    BC: tl.constexpr,
):
    row, chunk = tl.program_id(0), tl.program_id(1)
    parts = tl.arange(0, BC)
    maximum = _maximum(tl.load(MAX + row * C + parts, parts < C, other=float("-inf")))
    col = chunk * B + tl.arange(0, B)
    x = tl.load(X + row * V + col, col < V, other=float("-inf"))
    exponent = _lookup(_sub(x, maximum), TABLE, 1)
    total = tl.reduce(exponent, axis=0, combine_fn=_add)
    tl.store(SUM + row * C + chunk, total)


@triton.jit
def _logprob_output(
    X,
    MAX,
    SUM,
    TABLE,
    OUT,
    V: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    BC: tl.constexpr,
):
    row, chunk = tl.program_id(0), tl.program_id(1)
    parts = tl.arange(0, BC)
    maximum = _maximum(tl.load(MAX + row * C + parts, parts < C, other=float("-inf")))
    partial = tl.load(SUM + row * C + parts, parts < C, other=0)
    total = tl.reduce(partial, axis=0, combine_fn=_add)
    log_total = _lookup(total, TABLE, 2)
    col = chunk * B + tl.arange(0, B)
    x = tl.load(X + row * V + col, col < V, other=float("-inf"))
    tl.store(OUT + row * V + col, _sub(_sub(x, maximum), log_total), col < V)


def _logprobs_fake(x):
    return torch.empty_like(x)


def _logprobs(x: torch.Tensor) -> torch.Tensor:
    _check(x)
    assert x.ndim == 2 and x.shape[1] > 0
    x = x.contiguous()
    rows, vocab = x.shape
    out = torch.empty_like(x)
    if not rows:
        return out
    block = 4096
    chunks = triton.cdiv(vocab, block)
    maxima = torch.empty((rows, chunks), dtype=torch.bfloat16, device=x.device)
    sums = torch.empty_like(maxima)
    table = _tables(x.device)
    grid = (rows, chunks)
    _logprob_max[grid](x, maxima, vocab, chunks, block, num_warps=8)
    _logprob_sum[grid](
        x,
        maxima,
        sums,
        table,
        vocab,
        chunks,
        block,
        triton.next_power_of_2(chunks),
        num_warps=8,
    )
    _logprob_output[grid](
        x,
        maxima,
        sums,
        table,
        out,
        vocab,
        chunks,
        block,
        triton.next_power_of_2(chunks),
        num_warps=8,
    )
    return out


direct_register_custom_op("yoco_bf16_rms_reduction", _norm, fake_impl=_norm_fake)
direct_register_custom_op(
    "yoco_bf16_add_rms_reduction", _add_norm, fake_impl=_add_norm_fake
)
direct_register_custom_op("yoco_bf16_logprobs", _logprobs, fake_impl=_logprobs_fake)
