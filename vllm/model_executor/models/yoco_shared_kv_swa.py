# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read the identical last-W strict prefix for all draft slots, directly in pages."""

import torch

from vllm.model_executor.models.yoco_shared_kv import paged_prefix_attention
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version


@triton.jit
def _window(
    Q,
    KV,
    TABLE,
    ENDS,
    OUT,
    K_SCALE,
    V_SCALE,
    QUANTIZED: tl.constexpr,
    H: tl.constexpr,
    KH: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
    G: tl.constexpr,
    M: tl.constexpr,
    PAGE: tl.constexpr,
    SP: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    SC: tl.constexpr,
    TS: tl.constexpr,
    WINDOW: tl.constexpr,
    N: tl.constexpr,
):
    req, head = tl.program_id(0), tl.program_id(1)
    row, dim, col = tl.arange(0, M), tl.arange(0, D), tl.arange(0, N)
    end = tl.load(ENDS + req)
    lower = tl.maximum(0, end - WINDOW)
    qo = req * B * H * D + (row // G)[:, None] * H * D
    qo += (head * G + row % G)[:, None] * D + dim[None, :]
    q = tl.load(Q + qo, row[:, None] < B * G, 0)
    maximum = tl.full((M,), -float("inf"), tl.float32)
    denom = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, D), tl.float32)
    # At most ceil(8192/N)+1 tiles; absolute positions index the target pages.
    for tile in range(lower // N, tl.cdiv(end, N)):
        pos = tile * N + col
        valid = (pos >= lower) & (pos < end)
        # Physical page strides can exceed signed-int32 addressing in large pools.
        page = tl.load(TABLE + req * TS + pos // PAGE, valid, 0).to(tl.int64)
        offset = page[:, None] * SP + head * SH
        offset += (pos % PAGE)[:, None] * ST + dim[None, :] * SC
        key = tl.load(KV + offset, valid[:, None], 0.0)
        value = tl.load(KV + offset + D * SC, valid[:, None], 0.0)
        if QUANTIZED:
            key = (key.to(tl.float32) * tl.load(K_SCALE)).to(q.dtype)
            value = (value.to(tl.float32) * tl.load(V_SCALE)).to(q.dtype)
        score = tl.dot(q, tl.trans(key)) * (D**-0.5 * 1.4426950408889634)
        score = tl.where(valid[None, :], score, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        prob = tl.exp2(score - next_max[:, None])
        alpha = tl.exp2(maximum - next_max)
        acc = acc * alpha[:, None] + tl.dot(prob.to(value.dtype), value)
        denom = denom * alpha + tl.sum(prob, 1)
        maximum = next_max
    # Inert CUDA-graph padding requests have end=0 and must emit finite zeros.
    result = acc / tl.maximum(denom[:, None], 1.0e-20)
    tl.store(OUT + qo, result, row[:, None] < B * G)


def paged_window_attention(
    q, cache, table, ends, window=8192, *, k_scale=None, v_scale=None
):
    quantized = cache.dtype in (torch.uint8, torch.float8_e4m3fn)
    if window != 8192 and not quantized:
        raise ValueError("This kernel implements the trained SWA8192 contract")
    if quantized:
        if window < 1 or k_scale is None or v_scale is None:
            raise ValueError("FP8 KV requires a positive window and owner K/V scales")
        if k_scale.numel() != 1 or v_scale.numel() != 1:
            raise ValueError("Expected per-tensor target K/V scales")
        if any(
            scale.dtype != torch.float32 or scale.device != q.device
            for scale in (k_scale, v_scale)
        ):
            raise ValueError("Expected FP32 target K/V scales on the query device")
        cache = cache.view(torch.float8_e4m3fn)
    batch, block, heads, dim = q.shape
    if (
        cache.ndim != 4
        or cache.shape[-1] != 2 * dim
        or (not quantized and cache.dtype != q.dtype)
        or q.dtype != torch.bfloat16
    ):
        raise ValueError(
            "Expected BF16 query and BF16/E4M3 target cache [page,head,token,K|V]"
        )
    kvheads, page = cache.shape[1:3]
    if heads % kvheads or not 1 <= block <= 8 or dim not in (32, 64, 128):
        raise ValueError("Invalid SWA draft query shape")
    if not quantized and get_flash_attn_version(head_size=dim) == 4:
        return _flash_window_attention(q, cache, table, ends, window)
    q = q.contiguous()
    out = torch.empty_like(q)
    group = heads // kvheads
    _window[(batch, kvheads)](
        q,
        cache,
        table,
        ends,
        out,
        k_scale if quantized else q,
        v_scale if quantized else q,
        quantized,
        heads,
        kvheads,
        dim,
        block,
        group,
        max(16, triton.next_power_of_2(block * group)),
        page,
        *cache.stride(),
        table.stride(0),
        window,
        64,
        num_warps=4,
    )
    return out


def _flash_window_attention(q, cache, table, ends, window):
    batch, block, heads, dim = q.shape
    kvheads, page = cache.shape[1:3]
    group = heads // kvheads
    # Fold slots into each KV head's GQA group. With one query position, FA's
    # window is [end-window, end) for EVERY slot, regardless of its draft offset.
    query = q.reshape(batch, block, kvheads, group, dim).transpose(1, 2)
    query = query.reshape(batch, 1, kvheads * block * group, dim)
    out = paged_prefix_attention(
        query,
        cache,
        table,
        ends,
        table.shape[1] * page,
        window_size=[window - 1, -1],
    )
    return (
        out.reshape(batch, kvheads, block, group, dim)
        .transpose(1, 2)
        .reshape(batch, block, heads, dim)
    )
