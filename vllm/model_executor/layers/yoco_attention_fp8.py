# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 producers for YOCO Fast attention; reductions remain FP32."""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _diff_quant_kernel(
    attention,
    gates,
    output,
    scales,
    gate_stride_m,
    gate_stride_h,
    scale_stride_k,
    PAIRS: tl.constexpr,
):
    token = tl.program_id(0)
    pack = tl.program_id(1)
    heads = pack * 4 + tl.arange(0, 4)
    dims = tl.arange(0, 128)
    a = tl.load(
        attention + (token * 2 * PAIRS + 2 * heads[:, None]) * 128 + dims[None, :]
    ).to(tl.float32)
    b = tl.load(
        attention + (token * 2 * PAIRS + 2 * heads[:, None] + 1) * 128 + dims[None, :]
    ).to(tl.float32)
    g1 = tl.load(gates + token * gate_stride_m + 2 * heads * gate_stride_h).to(
        tl.float32
    )
    g2 = tl.load(gates + token * gate_stride_m + (2 * heads + 1) * gate_stride_h).to(
        tl.float32
    )
    y = a * tl.sigmoid(g1[:, None]) - b * tl.sigmoid(g2[:, None])
    # Keep the unfused diff-combine's BF16 rounding, without a BF16 store.
    y = y.to(tl.bfloat16).to(tl.float32)
    raw = tl.maximum(tl.max(tl.abs(y), axis=1), 1e-4) / 448.0
    exponent = tl.ceil(tl.log2(raw))
    scale = tl.exp2(exponent)
    q = tl.clamp(y / scale[:, None], -448.0, 448.0)
    tl.store(output + (token * PAIRS + heads[:, None]) * 128 + dims[None, :], q)
    biased = (exponent + 127.0).to(tl.uint32)
    packed = tl.sum(biased << (tl.arange(0, 4) * 8), axis=0)
    tl.store(scales + token + pack * scale_stride_k, packed)


def _diff_quant_fake(
    attention: torch.Tensor,
    gates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, heads, dim = attention.shape
    return (
        torch.empty(
            (tokens, heads // 2 * dim),
            device=attention.device,
            dtype=torch.float8_e4m3fn,
        ),
        torch.empty_strided(
            (tokens, heads // 8),
            (1, triton.cdiv(tokens, 4) * 4),
            device=attention.device,
            dtype=torch.int32,
        ),
    )


def _diff_quant(
    attention: torch.Tensor,
    gates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert attention.dtype == gates.dtype == torch.bfloat16
    assert attention.is_contiguous() and attention.shape[-1] == 128
    assert attention.shape[1] % 8 == 0
    assert gates.shape == attention.shape[:2]
    output, scales = _diff_quant_fake(attention, gates)
    tokens, heads, _ = attention.shape
    if tokens:
        _diff_quant_kernel[(tokens, heads // 8)](
            attention,
            gates,
            output,
            scales,
            gates.stride(0),
            gates.stride(1),
            scales.stride(1),
            PAIRS=heads // 2,
            num_warps=4,
        )
    return output, scales


direct_register_custom_op(
    "yoco_diff_attention_fp8",
    _diff_quant,
    fake_impl=_diff_quant_fake,
)


@triton.jit
def _cache_fp8_kernel(
    key,
    value,
    key_cache,
    value_cache,
    slots,
    key_stride_m,
    value_stride_m,
    cache_stride_b,
    cache_stride_s,
    cache_stride_h,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(slots + token)
    offsets = tl.arange(0, BLOCK)
    valid = (slot >= 0) & (offsets < HEADS * DIM)
    cache_offsets = (
        slot // BLOCK_SIZE * cache_stride_b
        + slot % BLOCK_SIZE * cache_stride_s
        + offsets // DIM * cache_stride_h
        + offsets % DIM
    )
    k = tl.load(key + token * key_stride_m + offsets, mask=valid, other=0)
    v = tl.load(value + token * value_stride_m + offsets, mask=valid, other=0)
    # Byte copies: these inputs already use the cache owner's K/V scales.
    tl.store(key_cache + cache_offsets, k, mask=valid)
    tl.store(value_cache + cache_offsets, v, mask=valid)


def cache_prequantized_fp8(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """Scatter E4M3 bytes, including padded graph slots and strided KV pools."""
    assert key.dtype == value.dtype == torch.float8_e4m3fn
    assert key_cache.dtype == value_cache.dtype == torch.uint8
    assert key.shape == value.shape and slots.numel() <= key.shape[0]
    assert key.stride(1) == value.stride(1) == key.shape[2]
    assert key.stride(2) == value.stride(2) == 1
    assert key_cache.stride() == value_cache.stride()
    assert key_cache.stride(-1) == 1
    if slots.numel():
        _cache_fp8_kernel[(slots.numel(),)](
            key.view(torch.uint8),
            value.view(torch.uint8),
            key_cache,
            value_cache,
            slots,
            key.stride(0),
            value.stride(0),
            *key_cache.stride()[:3],
            HEADS=key.shape[1],
            DIM=key.shape[2],
            BLOCK_SIZE=key_cache.shape[1],
            BLOCK=triton.next_power_of_2(key.shape[1] * key.shape[2]),
        )
