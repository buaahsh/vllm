# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence of fused FP8 producers to their existing BF16 + quant paths."""

import pytest
import torch

from vllm.model_executor.models import yoco  # noqa: F401
from vllm.platforms import current_platform

if not torch.cuda.is_available() or not current_platform.is_device_capability(100):
    pytest.skip("requires B200", allow_module_level=True)

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8_packed_for_deepgemm,
)
from vllm.model_executor.layers.yoco_attention_fp8 import cache_prequantized_fp8


def quant(x, scale):
    return ops.scaled_fp8_quant(x.flatten(1).contiguous(), scale=scale)[0].view_as(x)


def equal_bytes(a, b):
    torch.testing.assert_close(a.view(torch.uint8), b.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [0, 1, 8, 33, 128, 1025])
@pytest.mark.parametrize("weighted", [False, True])
def test_qkv_clip_rope_quant(tokens, weighted):
    torch.manual_seed(919)
    packed = torch.randn(tokens, 80, 128, dtype=torch.bfloat16, device="cuda") * 8
    q, k, v = packed.split([64, 8, 8], dim=1)
    weights = [torch.randn(128, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    weights = weights if weighted else [None, None]
    pos = torch.arange(tokens * 2, device="cuda")[::2]
    angles = torch.randn(max(1, 2 * tokens), 64, device="cuda")
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    scales = [torch.tensor([x], device="cuda") for x in (0.3, 0.125, 2.0)]
    actual = torch.ops.vllm.yoco_qkv_clip_rotary_fp8(
        q, k, v, *weights, pos, cache, *scales, 1e-6, 4.0
    )
    if not tokens:
        assert all(x.dtype == torch.float8_e4m3fn and x.shape[0] == 0 for x in actual)
        return
    if weighted:
        rq, rk = torch.ops.vllm.yoco_qk_rms_clip_rotary_weighted(
            q, k, *weights, pos, cache, 1e-6, 4.0
        )
    else:
        rq, rk = torch.ops.vllm.yoco_qk_rms_clip_rotary(q, k, pos, cache, 1e-6, 4.0)
    for a, x, scale in zip(actual, (rq, rk, v), scales):
        equal_bytes(a, quant(x, scale))


@pytest.mark.parametrize("tokens", [1, 8, 33, 193, 1025])
@pytest.mark.parametrize("with_value", [False, True])
def test_clip_quant(tokens, with_value):
    torch.manual_seed(919)
    heads = 8 if with_value else 64
    packed = (
        torch.randn(tokens, 2 * heads, 128, dtype=torch.bfloat16, device="cuda") * 6
    )
    x, v = packed.chunk(2, dim=1)
    w = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    scale, vs = [torch.tensor([x], device="cuda") for x in (0.3, 0.25)]
    a, av = torch.ops.vllm.yoco_clip_fp8(
        x, w, scale, 1e-6, 4.0, v if with_value else None, vs if with_value else None
    )
    expected = torch.ops.vllm.yoco_weighted_rms_clip(x, w, 1e-6, 4.0)
    equal_bytes(a, quant(expected, scale))
    if with_value:
        equal_bytes(av, quant(v, vs))


@pytest.mark.parametrize("tokens", [0, 1, 8, 33, 128, 513])
@pytest.mark.parametrize("heads", [16, 64])
def test_diff_quant_packed(tokens, heads):
    torch.manual_seed(919)
    a = torch.randn(tokens, heads, 128, dtype=torch.bfloat16, device="cuda")
    # Include cancellation, tiny values and large sigmoid inputs.
    if tokens:
        a[0, 1] = a[0, 0]
        a[-1] *= 1e-6
    gates = (
        torch.randn(tokens, heads * 2, dtype=torch.bfloat16, device="cuda")[:, ::2] * 10
    )
    q, scales = torch.ops.vllm.yoco_diff_attention_fp8(a, gates)
    if not tokens:
        assert q.shape == (0, heads // 2 * 128)
        return
    ref = torch.ops.vllm.yoco_diff_attention_v3(a, gates).flatten(1)
    rq, rs = per_token_group_quant_fp8_packed_for_deepgemm(
        ref, 128, eps=1e-4, use_ue8m0=True
    )
    equal_bytes(q, rq)
    torch.testing.assert_close(scales, rs, rtol=0, atol=0)
    assert scales.stride() == rs.stride()


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_cache_copy_and_graph_scale_updates(layout):
    torch.manual_seed(919)
    qkv = torch.randn(7, 80, 128, dtype=torch.bfloat16, device="cuda")
    q, k, v = qkv.split([64, 8, 8], dim=1)
    weights = [torch.ones(128, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    scales = [torch.tensor([x], device="cuda") for x in (0.3, 0.7, 1.3)]
    pos = torch.arange(7, device="cuda")
    cache = torch.cat(
        [torch.ones(7, 64, device="cuda"), torch.zeros(7, 64, device="cuda")], -1
    )
    slots = torch.tensor([35, -1, 2, 16, 0], device="cuda")
    # Unified KV allocation has holes between blocks.
    pool = torch.full((4, 3, 2, 16, 8, 128), 157, device="cuda", dtype=torch.uint8)
    target = pool[:, 1].permute(1, 0, 2, 3, 4)
    if layout == "HND":
        pool = torch.full((4, 3, 2, 8, 16, 128), 157, device="cuda", dtype=torch.uint8)
        target = pool[:, 1].permute(1, 0, 3, 2, 4)
    key_cache, value_cache = target.unbind(0)

    def call():
        qa, ka, va = torch.ops.vllm.yoco_qkv_clip_rotary_fp8(
            q, k, v, *weights, pos, cache, *scales, 1e-6, 4.0
        )
        cache_prequantized_fp8(ka, va, key_cache, value_cache, slots)
        return qa, ka, va

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = call()
    for step in range(2):
        for scale in scales:
            scale.mul_(2.0)
        qkv.add_(0.25)
        pool.fill_(157)
        graph.replay()
        rq, rk = torch.ops.vllm.yoco_qk_rms_clip_rotary_weighted(
            q, k, *weights, pos, cache, 1e-6, 4.0
        )
        equal_bytes(actual[0], quant(rq, scales[0]))
        expected = torch.full_like(target, 157)
        ops.reshape_and_cache_flash(
            rk, v, *expected.unbind(0), slots, "fp8", scales[1], scales[2]
        )
        torch.testing.assert_close(target, expected, rtol=0, atol=0)
        assert torch.all(pool[:, 0] == 157) and torch.all(pool[:, 2] == 157)


@pytest.mark.parametrize("tokens", [1, 33, 193])
def test_shared_clip_matches_compiled_norm(tokens):
    torch.manual_seed(919)
    x = torch.randn(tokens, 8, 128, dtype=torch.bfloat16, device="cuda") * 6
    weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor([0.25], device="cuda")
    norm = yoco.RMSClip(128, limit=4.0, has_weight=True).to(
        device="cuda", dtype=torch.bfloat16
    )
    norm.weight.data.copy_(weight)
    expected = torch.compile(norm)(x)
    actual, _ = torch.ops.vllm.yoco_clip_fp8(
        x, weight, scale, 1e-6, 4.0, round_before_weight=False
    )
    equal_bytes(actual, quant(expected, scale))
