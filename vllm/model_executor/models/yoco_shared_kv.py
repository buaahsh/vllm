# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-KV DSpark backbone; checkpoint names match SharedKVDraft training."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SharedKVConfig:
    hidden: int = 3072
    ffn: int = 9216
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    layers: int = 4
    vocab: int = 154880
    block: int = 8
    mask_id: int = 154856
    rank: int = 256
    theta: float = 10000.0
    eps: float = 1e-6
    initial_hidden_gate: float = 0.05
    initial_global_gate: float = 0.1


class SharedKVRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, promote_weight: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.promote_weight = promote_weight

    def forward(self, x):
        value = x if x.dtype == torch.float64 else x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        weight = self.weight if self.promote_weight else self.weight.to(x.dtype)
        return value.to(x.dtype) * weight


def rotary(q, k, positions, theta):
    dim = q.shape[-1]
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    inv = theta ** (-torch.arange(0, dim, 2, device=q.device, dtype=dtype) / dim)
    angles = positions.to(dtype)[..., None] * inv
    angles = torch.cat((angles, angles), dim=-1)
    cos, sin = angles.cos().to(q.dtype)[:, None], angles.sin().to(q.dtype)[:, None]

    def rotate(x):
        first, second = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-second, first), dim=-1) * sin

    return rotate(q), rotate(k)


class SharedKVLocalAttention(nn.Module):
    def __init__(self, c: SharedKVConfig):
        super().__init__()
        self.q_proj = nn.Linear(c.hidden, c.heads * c.head_dim, bias=False)
        self.k_proj = nn.Linear(c.hidden, c.kv_heads * c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.hidden, c.kv_heads * c.head_dim, bias=False)
        self.o_proj = nn.Linear(c.heads * c.head_dim, c.hidden, bias=False)
        self.q_norm = SharedKVRMSNorm(c.head_dim, c.eps)
        self.k_norm = SharedKVRMSNorm(c.head_dim, c.eps)


class SharedKVMLP(nn.Module):
    def __init__(self, c: SharedKVConfig):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden, c.ffn, bias=False)
        self.up_proj = nn.Linear(c.hidden, c.ffn, bias=False)
        self.down_proj = nn.Linear(c.ffn, c.hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SharedKVLayer(nn.Module):
    def __init__(self, c: SharedKVConfig):
        super().__init__()
        self.c = c
        self.local = SharedKVLocalAttention(c)
        self.input_norm = SharedKVRMSNorm(c.hidden, c.eps)
        self.post_norm = SharedKVRMSNorm(c.hidden, c.eps)
        self.mlp = SharedKVMLP(c)
        self.global_q = nn.Linear(c.hidden, c.heads * c.head_dim, bias=False)
        self.global_q_norm = SharedKVRMSNorm(c.head_dim, c.eps)
        self.global_o = nn.Linear(c.heads * c.head_dim, c.hidden, bias=False)
        self.hidden_proj = nn.Linear(c.hidden, c.hidden, bias=False)
        self.alpha = nn.Parameter(torch.empty(()))
        self.global_gate_logit = nn.Parameter(torch.empty(()))

    def forward(self, z, current, positions, attend_prefix):
        c = self.c
        batch, block, _ = z.shape
        conditioned = z + self.alpha * self.hidden_proj(current)[:, None]
        normed = self.input_norm(conditioned)
        local = self.local
        q = local.q_norm(local.q_proj(normed).view(batch, block, c.heads, c.head_dim))
        k = local.k_norm(
            local.k_proj(normed).view(batch, block, c.kv_heads, c.head_dim)
        )
        v = local.v_proj(normed).view(batch, block, c.kv_heads, c.head_dim)
        q, k = rotary(q.transpose(1, 2), k.transpose(1, 2), positions, c.theta)
        local_result = F.scaled_dot_product_attention(
            q, k, v.transpose(1, 2), is_causal=True, enable_gqa=True
        )
        local_result = local.o_proj(
            local_result.transpose(1, 2).reshape(batch, block, -1)
        )
        qg = self.global_q_norm(
            self.global_q(normed).view(batch, block, c.heads, c.head_dim)
        )
        global_result = self.global_o(attend_prefix(qg).reshape(batch, block, -1))
        residual = (
            conditioned
            + local_result
            + self.global_gate_logit.sigmoid() * global_result
        )
        return residual + self.mlp(self.post_norm(residual))


class SharedKVMarkov(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.markov_w1 = nn.Embedding(c.vocab, c.rank)
        self.markov_w2 = nn.Linear(c.rank, c.vocab, bias=False)


class SharedKVConfidence(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.proj = nn.Linear(c.hidden + c.rank, 1)

    def forward(self, x):
        return (x.float() * self.proj.weight.float()).sum(-1) + self.proj.bias.float()


class SharedKVDraft(nn.Module):
    def __init__(self, c: SharedKVConfig):
        super().__init__()
        self.c = c
        self.embed_tokens = nn.Embedding(c.vocab, c.hidden)
        self.lm_head = nn.Linear(c.hidden, c.vocab, bias=False)
        self.layers = nn.ModuleList([SharedKVLayer(c) for _ in range(c.layers)])
        self.norm = SharedKVRMSNorm(c.hidden, c.eps)
        self.current_norm = SharedKVRMSNorm(c.hidden, c.eps, promote_weight=False)
        self.slot_embedding = nn.Parameter(torch.empty(c.block, c.hidden))
        self.markov_head = SharedKVMarkov(c)
        self.confidence_head = SharedKVConfidence(c)
        self.draft_id_to_target_id = None

    def forward(self, block_ids, current_hidden, positions, attend_prefix: Callable):
        block = block_ids.shape[1]
        if not 1 <= block <= self.c.block:
            raise ValueError("Draft query length must fit the trained slot embeddings")
        z = self.embed_tokens(block_ids) + self.slot_embedding[None, :block]
        current = self.current_norm(current_hidden)
        for layer in self.layers:
            z = layer(z, current, positions, attend_prefix)
        return self.norm(z)

    def compute_draft_logits(self, hidden_states):
        # The target's ParallelLMHead is a weight container, not a callable.
        return F.linear(
            hidden_states.to(self.lm_head.weight.dtype), self.lm_head.weight
        ).float()

    def markov_embed(self, token_ids):
        return self.markov_head.markov_w1(token_ids)

    def markov_bias(self, embed):
        return self.markov_head.markov_w2(embed).float()

    def map_draft_to_target(self, draft_ids):
        return draft_ids

    def compute_confidence(self, hidden, embed):
        return self.confidence_head(
            torch.cat((hidden, embed.to(hidden.dtype)), -1)
        ).sigmoid()


def paged_prefix_attention(q, cache, block_table, prefix_lens, max_prefix_len):
    """Read normalized YOCO KV without writes, RoPE, or query-dependent masking."""
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        get_flash_attn_version,
    )

    batch, block, heads, dim = q.shape
    # vLLM's logical cache is [page, kv_head, token, K|V].
    if cache.ndim != 4 or cache.shape[-1] != 2 * dim or cache.dtype != q.dtype:
        raise ValueError("Shared-KV requires an unquantized logical YOCO KV cache")
    keys, values = cache.transpose(1, 2).split(dim, dim=-1)
    cu_q = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * block
    fa_version = get_flash_attn_version(head_size=dim)
    if fa_version is None:
        raise RuntimeError("No supported FlashAttention kernel for Shared-KV")
    output = flash_attn_varlen_func(
        q=q.reshape(-1, heads, dim).contiguous(),
        k=keys,
        v=values,
        cu_seqlens_q=cu_q,
        max_seqlen_q=block,
        seqused_k=prefix_lens.to(torch.int32),
        max_seqlen_k=max_prefix_len,
        block_table=block_table,
        causal=False,
        softmax_scale=dim**-0.5,
        fa_version=fa_version,
    )
    return output.reshape(batch, block, heads, dim)
