# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.models.yoco import (
    RMSNorm,
    YOCORotaryEmbedding,
    _yoco_apply_rotary_emb_fallback,
    _yoco_topk_routing,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize("num_tokens", [1, 33, 128])
def test_yoco_topk_routing_matches_full_softmax(num_tokens: int) -> None:
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(num_tokens, 128, generator=generator, dtype=torch.float32)

    topk_weights, topk_ids = _yoco_topk_routing(
        hidden_states=torch.empty(num_tokens, 1),
        gating_output=logits,
        topk=8,
        renormalize=True,
    )

    full_scores = F.softmax(logits, dim=-1, dtype=torch.float32)
    expected_weights, expected_ids = torch.topk(full_scores, k=8, dim=-1)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(topk_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(topk_weights, expected_weights, rtol=2e-6, atol=2e-7)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize("num_rows", [1, 33, 256])
@pytest.mark.parametrize("has_weight", [False, True])
@torch.inference_mode()
def test_yoco_head_rms_norm_matches_reference(num_rows: int, has_weight: bool) -> None:
    torch.manual_seed(11)
    x = torch.randn(num_rows, 128, device="cuda", dtype=torch.bfloat16)
    norm = RMSNorm(128, eps=1e-6, has_weight=has_weight).cuda()
    if has_weight:
        norm.weight.data.normal_(mean=1.0, std=0.1)

    expected = F.rms_norm(
        x,
        (128,),
        weight=norm.weight.to(torch.bfloat16),
        eps=norm.eps,
    )

    torch.testing.assert_close(norm(x), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize("num_tokens", [1, 17, 128])
@torch.inference_mode()
def test_yoco_rotary_matches_reference_with_strided_qk(num_tokens: int) -> None:
    torch.manual_seed(19)
    query_heads, key_heads, head_dim = 48, 4, 128
    total_dim = (query_heads + 2 * key_heads) * head_dim
    qkv = torch.randn(num_tokens, total_dim, device="cuda", dtype=torch.bfloat16)
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
    cos_sin = cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)
    expected_query, expected_key = _yoco_apply_rotary_emb_fallback(
        query.view(num_tokens, query_heads, head_dim),
        key.view(num_tokens, key_heads, head_dim),
        cos,
        sin,
    )

    actual_query, actual_key = rope(positions, query, key)
    torch.testing.assert_close(
        actual_query.view_as(expected_query), expected_query, rtol=0, atol=4e-3
    )
    torch.testing.assert_close(
        actual_key.view_as(expected_key), expected_key, rtol=0, atol=4e-3
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize("diff_v3", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 33, 128])
@torch.inference_mode()
def test_yoco_diff_attention_matches_reference(diff_v3: bool, num_tokens: int) -> None:
    torch.manual_seed(23)
    num_head_pairs = 24
    attn = torch.randn(
        num_tokens,
        2 * num_head_pairs,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    gate_width = (2 if diff_v3 else 1) * num_head_pairs
    gate = torch.randn(num_tokens, gate_width, device="cuda", dtype=torch.bfloat16)
    attn1 = attn[:, 0::2, :].float()
    attn2 = attn[:, 1::2, :].float()
    if diff_v3:
        expected = (
            attn1 * torch.sigmoid(gate[:, 0::2].float()).unsqueeze(-1)
            - attn2 * torch.sigmoid(gate[:, 1::2].float()).unsqueeze(-1)
        ).to(torch.bfloat16)
    else:
        expected = (attn1 - torch.sigmoid(gate.float()).unsqueeze(-1) * attn2).to(
            torch.bfloat16
        )

    actual = torch.ops.vllm.yoco_diff_attention(attn, gate, diff_v3)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
