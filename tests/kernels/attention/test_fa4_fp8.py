# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B200 FA4: non-unit descales, paged GQA, local attention and graph replay."""

import math

import pytest
import torch

from vllm.platforms import current_platform

try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    from vllm.vllm_flash_attn.fa4_compat import fa4_supports_fp8
except ImportError:
    pytest.skip("requires CUDA FlashAttention extensions", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="requires B200 FA4",
)


def make_case(dtype):
    torch.manual_seed(918)
    qlens, klens = [1, 17], [257, 1024]
    hq, hkv, dim, page = 64, 8, 128, 16
    q = torch.randn(sum(qlens), hq, dim, device="cuda", dtype=torch.float32) * 0.5
    k = torch.randn(128, page, hkv, dim, device="cuda", dtype=torch.float32) * 0.5
    v = torch.randn_like(k) * 0.7
    if dtype == torch.float8_e4m3fn:
        q, k, v = [(x * 8).to(dtype) for x in (q, k, v)]
        sq = torch.tensor([[0.0625, 0.125] * 4, [0.125, 0.0625] * 4], device="cuda")
        sk = sq * 2
        sv = sq * 3
    else:
        q, k, v = [x.to(dtype) for x in (q, k, v)]
        sq = sk = sv = None
    table = torch.randperm(128, device="cuda", dtype=torch.int32).reshape(2, 64)
    return dict(
        q=q,
        k=k,
        v=v,
        qlens=qlens,
        klens=klens,
        sq=sq,
        sk=sk,
        sv=sv,
        table=table,
        cu=torch.tensor([0, 1, 18], device="cuda", dtype=torch.int32),
        lengths=torch.tensor(klens, device="cuda", dtype=torch.int32),
    )


def reference(case, window):
    out, lses = [], []
    offset = 0
    for batch, (m, length) in enumerate(zip(case["qlens"], case["klens"])):
        q = case["q"][offset : offset + m].float()
        k = case["k"][case["table"][batch].long()].reshape(-1, 8, 128)[:length].float()
        v = case["v"][case["table"][batch].long()].reshape(-1, 8, 128)[:length].float()
        if case["sq"] is not None:
            q = q * case["sq"][batch].repeat_interleave(8)[None, :, None]
            k = k * case["sk"][batch][None, :, None]
            v = v * case["sv"][batch][None, :, None]
        k, v = [x.repeat_interleave(8, dim=1) for x in (k, v)]
        scores = torch.einsum("qhd,khd->hqk", q, k) / 128**0.5
        qp = torch.arange(m, device="cuda") + length - m
        kp = torch.arange(length, device="cuda")
        valid = kp[None, :] <= qp[:, None]
        if window is not None:
            valid &= kp[None, :] >= qp[:, None] - window
        scores.masked_fill_(~valid, -torch.inf)
        lses.append(scores.logsumexp(-1))
        out.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v))
        offset += m
    return torch.cat(out), torch.cat(lses, dim=1)


def forward(case, window, splits):
    return flash_attn_varlen_func(
        q=case["q"],
        k=case["k"],
        v=case["v"],
        max_seqlen_q=max(case["qlens"]),
        cu_seqlens_q=case["cu"],
        max_seqlen_k=max(case["klens"]),
        seqused_k=case["lengths"],
        block_table=case["table"],
        causal=True,
        window_size=(-1, -1) if window is None else (window, 0),
        q_descale=case["sq"],
        k_descale=case["sk"],
        v_descale=case["sv"],
        fa_version=4,
        num_splits=splits,
        return_softmax_lse=True,
    )


def check(case, actual, lse, window):
    expected, expected_lse = reference(case, window)
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all() and torch.isfinite(lse).all()
    relative = ((actual.float() - expected).norm() / expected.norm()).item()
    tolerance = 0.045 if case["q"].dtype == torch.float8_e4m3fn else 0.005
    assert relative < tolerance, relative
    torch.testing.assert_close(lse, expected_lse, rtol=2e-4, atol=2e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("window", [None, 512])
@pytest.mark.parametrize("splits", [1, 2])
@torch.inference_mode()
def test_fa4_paged_scales_and_split_kv(dtype, window, splits):
    assert fa4_supports_fp8()
    case = make_case(dtype)
    actual, lse = forward(case, window, splits)
    check(case, actual, lse, window)


@torch.inference_mode()
def test_fa4_fp8_graph_reads_updated_descales():
    case = make_case(torch.float8_e4m3fn)
    for _ in range(3):
        forward(case, 512, 2)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, lse = forward(case, 512, 2)
    base = case["sq"].clone()
    for multiplier in (0.5, 1.0, 2.0):
        case["sq"].copy_(base * multiplier)
        graph.replay()
        check(case, actual, lse, 512)


@torch.inference_mode()
def test_fa4_fp8_short_tail_does_not_saturate_probabilities():
    # FA4 visits the single-token tail first. The next tile raises the maximum
    # by about 2 log2 units: delaying the rescale by 4 would saturate E4M3 P.
    q = torch.ones(1, 64, 128, device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.full((17, 16, 8, 128), 0.125, device="cuda").to(q.dtype)
    v = torch.ones_like(k)
    k.view(-1, 8, 128)[256:].zero_()
    v.view(-1, 8, 128)[256:].zero_()
    out = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=1,
        cu_seqlens_q=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        max_seqlen_k=257,
        seqused_k=torch.tensor([257], device="cuda", dtype=torch.int32),
        block_table=torch.arange(17, device="cuda", dtype=torch.int32)[None],
        causal=True,
        fa_version=4,
        num_splits=1,
    )
    mass = 256 * math.exp(128**0.5 * 0.125)
    expected = mass / (mass + 1)
    torch.testing.assert_close(
        out.float(), torch.full_like(out.float(), expected), rtol=0, atol=0.008
    )
