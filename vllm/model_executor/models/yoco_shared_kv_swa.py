# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read the identical last-W strict prefix for all draft slots, directly in pages."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _window(
    Q,
    KV,
    TABLE,
    ENDS,
    OUT,
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
        page = tl.load(TABLE + req * TS + pos // PAGE, valid, 0)
        offset = page[:, None] * SP + head * SH
        offset += (pos % PAGE)[:, None] * ST + dim[None, :] * SC
        key = tl.load(KV + offset, valid[:, None], 0)
        value = tl.load(KV + offset + D * SC, valid[:, None], 0)
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


def paged_window_attention(q, cache, table, ends, window=8192):
    if window != 8192:
        raise ValueError("This kernel implements the trained SWA8192 contract")
    batch, block, heads, dim = q.shape
    if (
        cache.ndim != 4
        or cache.shape[-1] != 2 * dim
        or cache.dtype != q.dtype
        or q.dtype != torch.bfloat16
    ):
        raise ValueError("Expected BF16 logical target cache [page,head,token,K|V]")
    kvheads, page = cache.shape[1:3]
    if heads % kvheads or not 1 <= block <= 8 or dim not in (32, 64, 128):
        raise ValueError("Invalid SWA draft query shape")
    q = q.contiguous()
    out = torch.empty_like(q)
    group = heads // kvheads
    _window[(batch, kvheads)](
        q,
        cache,
        table,
        ends,
        out,
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
