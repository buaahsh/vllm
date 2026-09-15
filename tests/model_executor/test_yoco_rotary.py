# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO rotary regression tests."""

import pytest
import torch

from vllm.model_executor.layers.yoco_ops.norm import RMSClip, _yoco_align_rms_clip
from vllm.model_executor.layers.yoco_ops.rotary import (
    YOCORotaryEmbedding,
    _yoco_align_rotary_embedding,
    _yoco_apply_rotary_emb,
)


@torch.compile
def _llm_train_rotary_reference(
    cache: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_sin = cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)

    def apply(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat((y1, y2), dim=-1).to(x.dtype)

    return apply(query), apply(key)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_align_rotary_is_native_compiled_exact() -> None:
    generator = torch.Generator(device="cuda").manual_seed(4100)
    rope = YOCORotaryEmbedding(
        head_size=128,
        max_position_embeddings=4096,
        base=10000.0,
        execution_mode="align",
    ).cuda()
    query = torch.randn(
        17, 64, 128, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    key = torch.randn(
        17, 8, 128, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    positions = torch.arange(17, device="cuda", dtype=torch.long) * 7
    cache = rope._get_cos_sin_cache(query.device)

    with torch.no_grad():
        expected = _llm_train_rotary_reference(cache, positions, query, key)
        direct = _yoco_align_rotary_embedding(cache, positions, query, key)
        actual = rope(positions, query.flatten(-2), key.flatten(-2))

    assert torch.equal(direct[0], expected[0])
    assert torch.equal(direct[1], expected[1])
    assert torch.equal(actual[0].view_as(expected[0]), expected[0])
    assert torch.equal(actual[1].view_as(expected[1]), expected[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 17, 128])
@pytest.mark.parametrize("query_heads,key_heads", [(48, 4), (64, 8)])
def test_yoco_rotary_cuda_matches_compiled_fallback(
    num_tokens: int, query_heads: int, key_heads: int
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(19 + num_tokens)
    head_dim = 128
    total_dim = (query_heads + 2 * key_heads) * head_dim
    qkv = torch.randn(
        num_tokens,
        total_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    query, key, _ = qkv.split(
        [query_heads * head_dim, key_heads * head_dim, key_heads * head_dim],
        dim=-1,
    )
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.long) * 3
    rope = YOCORotaryEmbedding(
        head_size=head_dim,
        max_position_embeddings=max(4096, num_tokens * 3),
        base=10000.0,
    ).cuda()

    cache = rope._get_cos_sin_cache(query.device)
    cos, sin = cache.index_select(0, positions).chunk(2, dim=-1)
    expected_query, expected_key = _yoco_apply_rotary_emb(
        query.view(num_tokens, query_heads, head_dim),
        key.view(num_tokens, key_heads, head_dim),
        cos,
        sin,
    )
    actual_query, actual_key = rope(positions, query, key)

    torch.testing.assert_close(
        actual_query.view_as(expected_query), expected_query, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_key.view_as(expected_key), expected_key, rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_rotary_custom_op_opcheck() -> None:
    qkv = torch.randn(3, 80 * 128, device="cuda", dtype=torch.bfloat16)
    query, key, _ = qkv.split([64 * 128, 8 * 128, 8 * 128], dim=-1)
    query = query.view(3, 64, 128)
    key = key.view(3, 8, 128)
    positions = torch.tensor([0, 7, 31], device="cuda", dtype=torch.long)
    rope = YOCORotaryEmbedding(128, 128, 10000.0).cuda()
    cache = rope._get_cos_sin_cache(query.device)

    torch.library.opcheck(
        torch.ops.vllm.yoco_rotary.default,
        (query, key, positions, cache),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 7, 17, 128])
@pytest.mark.parametrize("query_heads,key_heads", [(64, 8), (32, 4), (48, 4)])
def test_yoco_fused_qk_rms_clip_rotary_is_bitwise_exact(
    num_tokens: int,
    query_heads: int,
    key_heads: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(8120 + num_tokens)
    head_dim = 128
    total_dim = (query_heads + 2 * key_heads) * head_dim
    qkv = torch.randn(
        num_tokens,
        total_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    query, key, _ = qkv.split(
        [query_heads * head_dim, key_heads * head_dim, key_heads * head_dim],
        dim=-1,
    )
    query = query.view(num_tokens, query_heads, head_dim)
    key = key.view(num_tokens, key_heads, head_dim)
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.long) * 7
    rope = YOCORotaryEmbedding(
        head_size=head_dim,
        max_position_embeddings=max(4096, num_tokens * 7),
        base=10000.0,
    ).cuda()
    cache = rope._get_cos_sin_cache(query.device)
    clip = RMSClip(head_dim, eps=1e-6, limit=3.0).cuda()

    expected_query, expected_key = rope(
        positions,
        clip(query),
        clip(key),
    )
    actual_query, actual_key = torch.ops.vllm.yoco_qk_rms_clip_rotary(
        query,
        key,
        positions,
        cache,
        clip.eps,
        clip.limit,
    )

    torch.testing.assert_close(
        actual_query, expected_query.view_as(actual_query), rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_key, expected_key.view_as(actual_key), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 17, 128])
def test_yoco_fused_weighted_qk_rms_clip_rotary_matches_native_bf16(
    num_tokens: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(9200 + num_tokens)
    query = 4 * torch.randn(
        num_tokens,
        64,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    key = 4 * torch.randn(
        num_tokens,
        8,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    query_weight = torch.empty(128, device="cuda", dtype=torch.bfloat16).uniform_(
        -2.0, 2.0, generator=generator
    )
    key_weight = torch.empty(128, device="cuda", dtype=torch.bfloat16).uniform_(
        -2.0, 2.0, generator=generator
    )
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.long) * 7
    rope = YOCORotaryEmbedding(
        128,
        max(4096, num_tokens * 7),
        10000.0,
        execution_mode="align",
    ).cuda()
    cache = rope._get_cos_sin_cache(query.device)

    with torch.no_grad():
        clipped_query = _yoco_align_rms_clip(query, query_weight, 1e-6, 3.0)
        clipped_key = _yoco_align_rms_clip(key, key_weight, 1e-6, 3.0)
        expected_query, expected_key = _yoco_align_rotary_embedding(
            cache, positions, clipped_query, clipped_key
        )
        actual_query, actual_key = torch.ops.vllm.yoco_qk_rms_clip_rotary_weighted(
            query,
            key,
            query_weight,
            key_weight,
            positions,
            cache,
            1e-6,
            3.0,
        )

    # Inductor changes the 128-wide reduction tree between static and dynamic
    # shape compilations.  The fast kernel deliberately fixes one tree, so a
    # handful of values can land on the neighboring BF16 rounding point.
    for actual, expected in (
        (actual_query, expected_query),
        (actual_key, expected_key),
    ):
        error = actual.float() - expected.float()
        nrmse = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(
            expected.float()
        )
        assert error.abs().max() <= 0.0625
        assert nrmse <= 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_fused_qk_rms_clip_rotary_opcheck() -> None:
    qkv = torch.randn(3, 80 * 128, device="cuda", dtype=torch.bfloat16)
    query, key, _ = qkv.split([64 * 128, 8 * 128, 8 * 128], dim=-1)
    query = query.view(3, 64, 128)
    key = key.view(3, 8, 128)
    positions = torch.tensor([0, 7, 31], device="cuda", dtype=torch.long)
    rope = YOCORotaryEmbedding(128, 128, 10000.0).cuda()
    cache = rope._get_cos_sin_cache(query.device)

    torch.library.opcheck(
        torch.ops.vllm.yoco_qk_rms_clip_rotary.default,
        (query, key, positions, cache, 1e-6, 3.0),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_fused_weighted_qk_rms_clip_rotary_opcheck() -> None:
    query = torch.randn(3, 64, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(3, 8, 128, device="cuda", dtype=torch.bfloat16)
    query_weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    key_weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([0, 7, 31], device="cuda", dtype=torch.long)
    rope = YOCORotaryEmbedding(128, 128, 10000.0).cuda()
    cache = rope._get_cos_sin_cache(query.device)

    torch.library.opcheck(
        torch.ops.vllm.yoco_qk_rms_clip_rotary_weighted.default,
        (
            query,
            key,
            query_weight,
            key_weight,
            positions,
            cache,
            1e-6,
            3.0,
        ),
    )
