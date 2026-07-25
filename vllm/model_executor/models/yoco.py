# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only YOCO (You Only Cache Once) model.

The reference training implementation lives at
``llm-train/llm/arch/model.py`` in the YOCO repository. This file
implements the same forward semantics on top of vLLM's primitives
(``Attention``, ``FusedMoE``, ``RMSNorm``, ``RoPE``, ``QKVParallelLinear``,
``MergedColumnParallelLinear`` etc.) so that the model can be served with
torch.compile + CUDAGraph and standard TP / DP / EP parallelism.

Architecture summary (matches the HF checkpoint shipped in ``hf-weights``):

* 20 layers total.

  * Layers 0..9 are *self*-attention with sliding window 512 and per-layer
    QK-norm.  These layers are executed three times in sequence on the same
    hidden state (``universal_loop = 3``).  Each iteration writes to its own
    KV cache, yielding 30 distinct self-attention KV caches.
  * Layers 10..19 are *cross*-attention (YOCO global) layers.  They share a
    single (K, V) pair produced once by a model-level ``yoco_norm`` +
    ``yoco_k_proj`` + ``yoco_v_proj`` on the hidden state at the end of the
    third self-loop pass.  Layer 10 owns that single KV cache; layers
    11..19 use ``kv_sharing_target_layer_name`` to read from it without
    creating new caches.
* All layers use *diff-attention*: ``q_proj`` outputs ``2 * head * head_dim``
  values; attention is computed once with ``2*head`` Q-heads.  Diff-v2 uses
  ``attn1 - sigmoid(gate) * attn2`` while diff-v3 gates both alternating
  heads independently before subtraction.
* All layers run an MoE (128 routed experts, top-k=8, softmax routing with
  post-top-k renormalization) plus a gated shared expert.

The ``cross_head`` field in the HF config is honored as 48 — this is twice
the self-layer ``head`` (24) and matches the checkpoint q_proj shape
``(2 * 48 * 128, 3072) = (12288, 3072)`` for layers 10..19.  Cross layers
therefore have 96 Q-heads (with diff-attention doubling) but still only
4 KV-heads (``cross_kv_head`` defaults to ``kv_head``).

Tokenization note: this model relies on the ``O200kHarmonyTokenizer`` shipped
in ``hf-weights/o200k_harmony_tokenizer.py``.  Launch vLLM with
``--trust-remote-code`` so that the tokenizer registers itself with
``AutoTokenizer``.  The tokenizer's ``encode()`` prepends BOS by default; for
the *completion* code path vLLM passes ``add_special_tokens=True`` (one BOS
prepended).  For the *chat* code path vLLM passes ``add_special_tokens=False``
so the chat template's literal ``<|startoftext|>`` is the only BOS.  In both
cases BOS appears exactly once.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention.attention import Attention, AttentionType
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.utils import KVSharingFastPrefillMetadata

if HAS_TRITON:

    @triton.jit
    def _yoco_rms_clip_kernel(
        x_ptr,
        output_ptr,
        num_rows,
        eps: tl.constexpr,
        limit: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        cols = tl.arange(0, HEAD_DIM)[None, :]
        row_mask = rows < num_rows
        values = tl.load(
            x_ptr + rows * HEAD_DIM + cols,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        square_sum = tl.sum(tl.where(row_mask, values * values, 0.0), axis=1)[:, None]
        clip_coef = limit * tl.extra.cuda.libdevice.rsqrt(square_sum / HEAD_DIM + eps)
        clip_coef = tl.minimum(clip_coef, 1.0)
        tl.store(
            output_ptr + rows * HEAD_DIM + cols,
            values * clip_coef,
            mask=row_mask,
        )

    @triton.jit
    def _yoco_rms_norm_kernel(
        x_ptr,
        weight_ptr,
        output_ptr,
        num_rows,
        eps: tl.constexpr,
        HIDDEN_SIZE: tl.constexpr,
        REDUCTION_BLOCK: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        row_mask = row < num_rows
        cols = tl.arange(0, REDUCTION_BLOCK)[None, :]
        square_acc = tl.full([BLOCK_ROWS, REDUCTION_BLOCK], 0.0, tl.float32)

        for offset in tl.range(0, HIDDEN_SIZE, REDUCTION_BLOCK):
            hidden_offsets = offset + cols
            mask = (hidden_offsets < HIDDEN_SIZE) & row_mask
            values = tl.load(
                x_ptr + row * HIDDEN_SIZE + hidden_offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            square_acc = tl.where(mask, square_acc + values * values, square_acc)

        square_sum = tl.sum(square_acc, axis=1)[:, None]
        inv_rms = tldevice.rsqrt(square_sum / HIDDEN_SIZE + eps)

        for offset in tl.range(0, HIDDEN_SIZE, REDUCTION_BLOCK):
            hidden_offsets = offset + cols
            mask = (hidden_offsets < HIDDEN_SIZE) & row_mask
            values = tl.load(
                x_ptr + row * HIDDEN_SIZE + hidden_offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            weight = tl.load(
                weight_ptr + hidden_offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            tl.store(
                output_ptr + row * HIDDEN_SIZE + hidden_offsets,
                values * inv_rms * weight,
                mask=mask,
            )

    @triton.jit
    def _yoco_routing_softmax_kernel(
        logits_ptr,
        scores_ptr,
        num_rows,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, 128)
        row_mask = row < num_rows
        logits = tl.load(
            logits_ptr + row * 128 + cols,
            mask=row_mask,
            other=0.0,
        )
        masked_logits = tl.where(row_mask, logits, float("-inf"))
        row_max = tl.max(masked_logits, axis=0)
        numerator = tl.extra.cuda.libdevice.exp(logits - row_max)
        denominator = tl.sum(
            tl.where(row_mask, numerator, 0.0),
            axis=0,
        )
        scores = numerator / denominator
        tl.store(
            scores_ptr + row * 128 + cols,
            scores,
            mask=row_mask,
        )

    @triton.jit
    def _yoco_routing_renorm_scatter_kernel(
        topk_weights_ptr,
        topk_ids_ptr,
        output_ptr,
        num_rows,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        ranks = tl.arange(0, 8)[None, :]
        row_mask = rows < num_rows
        weights = tl.load(
            topk_weights_ptr + rows * 8 + ranks,
            mask=row_mask,
            other=0.0,
        )
        ids = tl.load(
            topk_ids_ptr + rows * 8 + ranks,
            mask=row_mask,
            other=0,
        )
        denominator = tl.sum(
            tl.where(row_mask, weights, 0.0),
            axis=1,
        )[:, None]
        tl.store(
            output_ptr + rows * 128 + ids,
            weights / denominator,
            mask=row_mask,
        )


def _yoco_rms_clip_cuda(
    x: torch.Tensor,
    eps: float,
    limit: float,
) -> torch.Tensor:
    x_contiguous = x.contiguous()
    output = torch.empty_like(x_contiguous)
    num_rows = x_contiguous.numel() // x_contiguous.shape[-1]
    _yoco_rms_clip_kernel[(triton.cdiv(num_rows, 8),)](
        x_contiguous,
        output,
        num_rows,
        eps=eps,
        limit=limit,
        HEAD_DIM=128,
        BLOCK_ROWS=8,
        num_warps=4,
        num_stages=1,
    )
    return output


def _yoco_rms_clip_fake(
    x: torch.Tensor,
    eps: float,
    limit: float,
) -> torch.Tensor:
    return torch.empty_like(x)


def _yoco_rms_norm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_contiguous = x.contiguous()
    weight_contiguous = weight.to(torch.bfloat16).contiguous()
    output = torch.empty_like(x_contiguous, dtype=torch.bfloat16)
    num_rows = x_contiguous.numel() // x_contiguous.shape[-1]
    reduction_block = 4096 if num_rows >= 128 else 2048
    block_rows = 1
    _yoco_rms_norm_kernel[(triton.cdiv(num_rows, block_rows),)](
        x_contiguous,
        weight_contiguous,
        output,
        num_rows,
        eps=eps,
        HIDDEN_SIZE=x_contiguous.shape[-1],
        REDUCTION_BLOCK=reduction_block,
        BLOCK_ROWS=block_rows,
        num_warps=16,
        num_stages=1,
    )
    return output


def _yoco_rms_norm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x, dtype=torch.bfloat16)


if HAS_TRITON and current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_rms_clip",
        op_func=_yoco_rms_clip_cuda,
        fake_impl=_yoco_rms_clip_fake,
    )
    direct_register_custom_op(
        op_name="yoco_rms_norm",
        op_func=_yoco_rms_norm_cuda,
        fake_impl=_yoco_rms_norm_fake,
    )


# --------------------------------------------------------------------------- #
# Config helpers                                                              #
# --------------------------------------------------------------------------- #


YOCO_PACKED_MODULES_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
}


YOCO_ONLINE_QUANT_IGNORE = [
    "re:.*\\.self_attn\\.lambda_proj$",
]


def _cfg_int(config: PretrainedConfig, *names: str, default: int | None = None) -> int:
    """Read the first attribute from ``config`` whose name is in ``names``.

    The YOCO HF config aliases canonical HF names (``hidden_size``,
    ``num_hidden_layers`` ...) to YOCO-native names (``d_model``, ``n_layers``
    ...).  This helper returns whichever is set so the model code can use the
    most readable name without caring which alias was used.
    """
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    if default is not None:
        return default
    raise AttributeError(
        f"None of the config fields {names!r} are set on {type(config).__name__}"
    )


def _maybe_build_yoco_quant_config(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    if quant_config is None:
        return None
    quant_config.packed_modules_mapping = YOCO_PACKED_MODULES_MAPPING
    ignored_layers = getattr(quant_config, "ignored_layers", None)
    if ignored_layers is not None:
        for pattern in YOCO_ONLINE_QUANT_IGNORE:
            if pattern not in ignored_layers:
                ignored_layers.append(pattern)
    return quant_config


def _yoco_routing_renorm_block(num_rows: int) -> int:
    # Match the Native Inductor choices for the verified 3/66/110-token shapes
    # without runtime autotuning, which can select different reduction tiles.
    return 32 if num_rows >= 96 else 1


def _yoco_topk_routing_impl(
    router_logits: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert router_logits.dtype == torch.float32
    assert router_logits.shape[-1] == 128
    assert topk == 8
    router_logits = router_logits.contiguous()
    num_rows = router_logits.shape[0]
    gate_scores = torch.empty_like(router_logits)
    _yoco_routing_softmax_kernel[(num_rows,)](
        router_logits,
        gate_scores,
        num_rows,
        num_warps=2,
        num_stages=1,
    )
    scores, topk_ids = torch.topk(gate_scores, k=topk, dim=-1)
    routing_probs = torch.zeros_like(router_logits)
    renorm_block = _yoco_routing_renorm_block(num_rows)
    _yoco_routing_renorm_scatter_kernel[(triton.cdiv(num_rows, renorm_block),)](
        scores,
        topk_ids,
        routing_probs,
        num_rows,
        BLOCK_ROWS=renorm_block,
        num_warps=2,
        num_stages=1,
    )
    routing_map = routing_probs != 0
    return routing_probs, routing_map, gate_scores


def _yoco_topk_routing(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del hidden_states
    assert renormalize
    if gating_output.shape[-1] != 128 or topk != 8:
        gate_scores = F.softmax(gating_output, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(gate_scores, k=topk, dim=-1)
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights, topk_ids
    routing_probs, _, _ = _yoco_topk_routing_impl(gating_output, topk)
    return torch.topk(routing_probs, k=topk, dim=-1)


class RMSClip(nn.Module):
    """RMS-based clipping for YOCO ``qk_rms_clip`` models.

    Scales each ``head_dim`` slice by ``clamp(limit / rms, max=1.0)`` where
    ``rms = sqrt(mean(x**2, -1) + eps)``.  The optional affine weight is
    controlled by ``qk_rms_gamma``, matching training's ``RMSClip``.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        limit: float = 3.0,
        has_weight: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.limit = limit
        if has_weight:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, limit={self.limit}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            HAS_TRITON
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.shape[-1] == 128
            and current_platform.is_cuda()
            and self.weight is None
        ):
            return torch.ops.vllm.yoco_rms_clip(x, self.eps, self.limit)
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        clip_coef = (self.limit * torch.rsqrt(variance + self.eps)).clamp(max=1.0)
        x = (x * clip_coef).to(orig_dtype)
        if self.weight is not None:
            x = x * self.weight.to(orig_dtype)
        return x


class RMSNorm(nn.Module):
    """RMSNorm matching llm-train's compiled reduction semantics.

    Inductor keeps the residual input and reduction in FP32, loads the BF16
    weight as FP32, and casts only the output to BF16.  The CUDA path below
    also preserves Inductor's reduction order for hidden size 3072.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        weight = torch.ones(dim, dtype=dtype or torch.get_default_dtype())
        if has_weight:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            HAS_TRITON
            and x.is_cuda
            and x.dtype == torch.float32
            and x.shape[-1] == 3072
            and current_platform.is_cuda()
        ):
            return torch.ops.vllm.yoco_rms_norm(x, self.weight, self.eps)
        return F.rms_norm(
            x.to(torch.bfloat16),
            (x.shape[-1],),
            weight=self.weight.to(torch.bfloat16),
            eps=self.eps,
        )


@torch.compile
def _yoco_apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)

    def apply(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat((y1, y2), dim=-1).to(x.dtype)

    return apply(query), apply(key)


@torch.compile
def _yoco_diff_attention_v2(
    attn1: torch.Tensor,
    attn2: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    return attn1 - torch.sigmoid(gate).unsqueeze(-1) * attn2


@torch.compile
def _yoco_diff_attention_v3(
    attn1: torch.Tensor,
    attn2: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    gate1 = gate[:, 0::2]
    gate2 = gate[:, 1::2]
    return (
        attn1 * torch.sigmoid(gate1).unsqueeze(-1)
        - attn2 * torch.sigmoid(gate2).unsqueeze(-1)
    )


def _yoco_normalized_router_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    weight = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return F.linear(hidden_states, weight)


class YOCORotaryEmbedding(nn.Module):
    """YOCO RoPE with llm-train's FP32 cos/sin cache semantics."""

    def __init__(
        self,
        head_size: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("cos_sin_cache", None, persistent=False)

    def _get_cos_sin_cache(self, device: torch.device) -> torch.Tensor:
        cache = self.cos_sin_cache
        if cache is not None and cache.device == device:
            return cache
        # llm-train constructs RoPE while the default device is CUDA. Generate
        # the FP32 cache on the execution device as well: CPU-generated trig
        # values differ by a few ULPs and can cross BF16/FP8 boundaries.
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(
                    0,
                    self.head_size,
                    2,
                    dtype=torch.float32,
                    device=device,
                )
                / self.head_size
            )
        )
        positions = torch.arange(
            self.max_position_embeddings,
            dtype=torch.float32,
            device=device,
        )
        freqs = torch.einsum("i,j -> ij", positions, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        self.cos_sin_cache = cache
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self._get_cos_sin_cache(query.device)
        cos_sin = cache.index_select(
            0, positions.to(device=query.device, dtype=torch.long)
        )
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = query.view(query.shape[0], -1, self.head_size)
        key = key.view(key.shape[0], -1, self.head_size)
        query, key = _yoco_apply_rotary_emb(query, key, cos, sin)
        return query.flatten(-2), key.flatten(-2)


def _build_qk_norm(config: PretrainedConfig, head_dim: int, rms_eps: float):
    """Build the per-head Q/K normalization module for YOCO attention.

    Three mutually exclusive modes, matching training (``llm/arch/attention.py``):
    * ``qk_rms_clip=True``  -> :class:`RMSClip` (clips outliers).
    * ``qk_norm=True``      -> :class:`RMSNorm`.
    * otherwise             -> ``None`` (no Q/K norm).
    """
    # Older exported YOCO-v2 configs omitted this field and were weight-free.
    has_weight = bool(getattr(config, "qk_rms_gamma", False))
    if bool(getattr(config, "qk_rms_clip", False)):
        limit = float(getattr(config, "qk_rms_limit", 3.0))
        return RMSClip(head_dim, eps=rms_eps, limit=limit, has_weight=has_weight)
    if bool(getattr(config, "qk_norm", False)):
        return RMSNorm(head_dim, eps=rms_eps, has_weight=has_weight)
    return None


def _swiglu_limit(config: PretrainedConfig) -> float:
    """SwiGLU clamp limit, matching training (``swiglu_limit``, default 10.0).

    Training clamps ``gate`` to ``max=limit`` and ``up`` to ``[-limit, limit]``
    before ``silu(gate) * up`` (see ``llm/arch/ffn.py`` and
    ``llm/arch/all2all_moe.py``).
    """
    return float(getattr(config, "swiglu_limit", 10.0))


def _apply_per_head_norm(
    x: torch.Tensor, num_heads: int, head_dim: int, norm: nn.Module
) -> torch.Tensor:
    """Apply ``norm`` independently to each head's ``head_dim`` slice.

    ``x`` has shape ``(n_tokens, num_heads * head_dim)``.
    """
    x = x.unflatten(-1, (num_heads, head_dim))
    x = norm(x)
    return x.flatten(-2, -1)


# --------------------------------------------------------------------------- #
# Self-attention (sliding window, QK-norm, RoPE, diff-attention)              #
# --------------------------------------------------------------------------- #


class YOCOSelfAttention(nn.Module):
    """Sliding-window self-attention for YOCO layers 0..9.

    Creates ``universal_loop`` distinct ``Attention`` sub-modules that share
    the projection weights but use unique KV cache prefixes so each universal
    loop iteration gets its own KV cache slot.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        universal_loop: int,
        num_hidden_layers: int,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = _cfg_int(config, "hidden_size", "d_model")
        self.total_num_heads = _cfg_int(config, "num_attention_heads", "head")
        self.total_num_kv_heads = _cfg_int(config, "num_key_value_heads", "kv_head")
        self.head_dim = _cfg_int(config, "head_dim")
        self.diff_v3 = bool(getattr(config, "diff_v3", False))
        self.layer_idx = layer_idx
        self.universal_loop = universal_loop
        self.num_hidden_layers = num_hidden_layers
        self.sliding_window = _cfg_int(
            config, "sliding_window_size", "yoco_window_size", default=512
        )
        max_position = _cfg_int(config, "max_position_embeddings", "max_seq_len")
        rope_theta = float(getattr(config, "rope_theta", 10000.0))

        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0, (
            f"num_attention_heads={self.total_num_heads} must be divisible "
            f"by TP size {tp_size} so diff-attention head pairs stay local"
        )
        # ``2 * head`` Q-heads because of diff-attention.
        q_heads = 2 * self.total_num_heads
        assert q_heads % tp_size == 0, (
            f"2*num_attention_heads={q_heads} must be divisible by TP size {tp_size}"
        )
        assert (
            self.total_num_kv_heads % tp_size == 0
            or tp_size % self.total_num_kv_heads == 0
        ), (
            f"num_kv_heads={self.total_num_kv_heads} must be divisible by "
            f"or divide TP size {tp_size}"
        )
        self.num_heads = q_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        gate_heads = (2 if self.diff_v3 else 1) * self.total_num_heads
        assert gate_heads % tp_size == 0, (
            f"gate_heads={gate_heads} must be divisible "
            f"by TP size {tp_size}"
        )
        self.num_lambda_heads = self.total_num_heads // tp_size
        self.num_gate_heads = gate_heads // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=q_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        # Keep the legacy HF name ``lambda_proj`` for checkpoint compatibility.
        # Diff-v2 has one gate per head pair; diff-v3 has one per attention head.
        self.lambda_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=gate_heads,
            bias=False,
            gather_output=False,
            # llm-train constructs lambda_proj with default MixPrecisionLinear,
            # so it stays BF16 even when the rest of attention uses MXFP8.
            quant_config=None,
            prefix=f"{prefix}.lambda_proj",
        )
        rms_eps = float(
            getattr(config, "rms_norm_eps", getattr(config, "norm_eps", 1e-6))
        )
        # Per-head Q/K normalization (RMSClip when ``qk_rms_clip``, weight-free
        # RMSNorm when ``qk_norm``, else nothing).  ``RMSClip`` and weight-free
        # ``RMSNorm`` carry no parameters, so the checkpoint contains no
        # ``q_norm``/``k_norm`` weights in any of these modes.
        self.q_norm = _build_qk_norm(config, self.head_dim, rms_eps)
        self.k_norm = _build_qk_norm(config, self.head_dim, rms_eps)

        self.rotary_emb = YOCORotaryEmbedding(
            head_size=self.head_dim,
            max_position_embeddings=max_position,
            base=rope_theta,
        )

        # Build one Attention module per universal-loop iteration.  Each gets
        # a unique cache prefix so the runtime allocates a distinct KV cache.
        self.attn = nn.ModuleList()
        for loop_idx in range(universal_loop):
            unique_layer_idx = loop_idx * num_hidden_layers + layer_idx
            unique_prefix = prefix.replace(
                f"layers.{layer_idx}", f"layers.{unique_layer_idx}"
            )
            self.attn.append(
                Attention(
                    num_heads=self.num_heads,
                    head_size=self.head_dim,
                    scale=self.scaling,
                    num_kv_heads=self.num_kv_heads,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    per_layer_sliding_window=self.sliding_window,
                    attn_type=AttentionType.DECODER,
                    prefix=f"{unique_prefix}.attn",
                )
            )

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    def _diff_attention_combine(
        self,
        attn_out: torch.Tensor,
        gate: torch.Tensor,
        num_heads_per_pair: int,
    ) -> torch.Tensor:
        """Combine the 2*head attention output via the diff-attention rule.

        Output shape: ``(n_tokens, num_heads_per_pair * head_dim)`` (i.e. the
        pre-o_proj hidden slice owned by this TP rank).
        """
        # (n_tokens, 2 * num_heads_per_pair, head_dim)
        attn_view = attn_out.view(-1, 2 * num_heads_per_pair, self.head_dim)
        attn1 = attn_view[:, 0::2, :]
        attn2 = attn_view[:, 1::2, :]
        if self.diff_v3:
            out = _yoco_diff_attention_v3(attn1, attn2, gate)
        else:
            out = _yoco_diff_attention_v2(attn1, attn2, gate)
        return out.reshape(-1, num_heads_per_pair * self.head_dim)

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        loop_idx: int,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Per-head QK norm/clip on the un-rotated query/key, applied
        # independently to each head's ``head_dim`` slice — matches training's
        # ``q_norm``/``k_norm``.  Skipped entirely when neither ``qk_rms_clip``
        # nor ``qk_norm`` is set.
        if self.q_norm is not None:
            q = _apply_per_head_norm(q, self.num_heads, self.head_dim, self.q_norm)
            k = _apply_per_head_norm(k, self.num_kv_heads, self.head_dim, self.k_norm)
        q, k = self.rotary_emb(positions, q, k)
        attn_out = self.attn[loop_idx](q, k, v)

        gate, _ = self.lambda_proj(hidden_states)
        out = self._diff_attention_combine(attn_out, gate, self.num_lambda_heads)
        out, _ = self.o_proj(out)
        return out


# --------------------------------------------------------------------------- #
# Cross-attention (NoPE, no QK-norm, shared KV via kv_sharing)                #
# --------------------------------------------------------------------------- #


class YOCOCrossAttention(nn.Module):
    """YOCO cross-attention layer (layers 10..19).

    These layers have only ``q_proj`` / ``o_proj`` / ``lambda_proj`` — they
    share a single set of (K, V) produced once at the model level.  Layer 10
    owns the shared KV cache; subsequent cross-layers point their
    ``kv_sharing_target_layer_name`` at layer 10's attention to reuse the
    cache without writing.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        first_cross_layer_idx: int,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = _cfg_int(config, "hidden_size", "d_model")
        # Cross-attention has its OWN Q-head count via ``cross_head``.  In this
        # checkpoint ``cross_head = 48`` (twice the self-attention head count)
        # and the q_proj output is ``2 * cross_head * head_dim = 12288``.
        # ``cross_kv_head`` defaults to ``kv_head`` (= 4 here).
        self.total_num_heads = _cfg_int(config, "cross_head", "head")
        self.total_num_kv_heads = _cfg_int(
            config, "cross_kv_head", "num_key_value_heads", "kv_head"
        )
        self.head_dim = _cfg_int(config, "head_dim")
        self.diff_v3 = bool(getattr(config, "diff_v3", False))
        self.layer_idx = layer_idx
        self.first_cross_layer_idx = first_cross_layer_idx

        tp_size = get_tensor_model_parallel_world_size()
        q_heads = 2 * self.total_num_heads
        assert q_heads % tp_size == 0
        assert self.total_num_heads % tp_size == 0
        self.num_heads = q_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.num_lambda_heads = self.total_num_heads // tp_size
        gate_heads = (2 if self.diff_v3 else 1) * self.total_num_heads
        assert gate_heads % tp_size == 0
        self.num_gate_heads = gate_heads // tp_size
        self.scaling = self.head_dim**-0.5

        # NoPE on cross layers — the checkpoint has ``rope_dim = 0``.
        self.q_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=q_heads * self.head_dim,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.lambda_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=gate_heads,
            bias=False,
            gather_output=False,
            # llm-train leaves lambda_proj at default BF16 precision.
            quant_config=None,
            prefix=f"{prefix}.lambda_proj",
        )

        rms_eps = float(
            getattr(config, "rms_norm_eps", getattr(config, "norm_eps", 1e-6))
        )
        # Cross layers apply the same per-head Q norm/clip as self layers (the
        # shared K is normed once at the model level on ``yoco_key``).
        self.q_norm = _build_qk_norm(config, self.head_dim, rms_eps)

        if layer_idx == first_cross_layer_idx:
            kv_sharing_target = None
        else:
            # Point at layer 10's attention.  ``prefix`` looks like
            # ``model.layers.{i}.self_attn`` so we substitute to layer 10.
            owner_prefix = prefix.replace(
                f"layers.{layer_idx}", f"layers.{first_cross_layer_idx}"
            )
            kv_sharing_target = f"{owner_prefix}.attn"

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=kv_sharing_target,
            prefix=f"{prefix}.attn",
        )

    def _diff_attention_combine(
        self, attn_out: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        attn_view = attn_out.view(-1, 2 * self.num_lambda_heads, self.head_dim)
        attn1 = attn_view[:, 0::2, :]
        attn2 = attn_view[:, 1::2, :]
        if self.diff_v3:
            out = _yoco_diff_attention_v3(attn1, attn2, gate)
        else:
            out = _yoco_diff_attention_v2(attn1, attn2, gate)
        return out.reshape(-1, self.num_lambda_heads * self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        yoco_key: torch.Tensor,
        yoco_value: torch.Tensor,
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        if self.q_norm is not None:
            q = _apply_per_head_norm(q, self.num_heads, self.head_dim, self.q_norm)
        attn_out = self.attn(q, yoco_key, yoco_value)
        gate, _ = self.lambda_proj(hidden_states)
        out = self._diff_attention_combine(attn_out, gate)
        out, _ = self.o_proj(out)
        return out


# --------------------------------------------------------------------------- #
# MoE block                                                                   #
# --------------------------------------------------------------------------- #


class YOCOClampedSwiGLU(nn.Module):
    def __init__(self, swiglu_limit: float) -> None:
        super().__init__()
        self.swiglu_limit = float(swiglu_limit)

    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.float().chunk(2, dim=-1)
        gate = gate.clamp(max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return (F.silu(gate) * up).to(gate_up.dtype)


class YOCOSharedExperts(nn.Module):
    """Shared-expert MLP for YOCO MoE blocks (SwiGLU)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None,
        reduce_results: bool,
        prefix: str,
        swiglu_limit: float = 10.0,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        # Clamped SwiGLU to match training (``swiglu_limit``).  When the limit is
        # non-positive, fall back to the plain (unclamped) activation.
        self.swiglu_limit = float(swiglu_limit)
        if self.swiglu_limit > 0:
            self.act_fn = YOCOClampedSwiGLU(self.swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class YOCOMoE(nn.Module):
    """YOCO MoE block: routed top-k + gated shared expert + final reduce."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = _cfg_int(config, "hidden_size", "d_model")
        self.num_experts = _cfg_int(config, "num_experts", "moe_expert_num")
        self.top_k = _cfg_int(config, "num_experts_per_tok", "moe_top_k", "top_k")
        self.moe_intermediate_size = _cfg_int(
            config, "moe_intermediate_size", "moe_ffn_dim"
        )
        self.moe_latent_dim = _cfg_int(config, "moe_latent_dim", default=0)
        self.moe_latent_norm = bool(getattr(config, "moe_latent_norm", False))
        self.shared_intermediate_size = _cfg_int(
            config, "shared_expert_intermediate_size", "d_shared_expert"
        )
        self.swiglu_limit = _swiglu_limit(config)
        self.router_weights_normalized = bool(
            getattr(config, "router_weights_normalized", False)
        )

        # Router gate — runs in fp32 to match training.
        self.gate = GateLinear(
            input_size=self.hidden_size,
            output_size=self.num_experts,
            bias=False,
            params_dtype=torch.float32,
            force_fp32_compute=True,
            prefix=f"{prefix}.gate",
        )
        self.gate.set_out_dtype(torch.float32)

        # ``reduce_results=True`` so the shared-expert output is tensor-parallel
        # all-reduced on its own; the routed path is reduced inside ``FusedMoE``.
        self.shared_experts = YOCOSharedExperts(
            hidden_size=self.hidden_size,
            intermediate_size=self.shared_intermediate_size,
            quant_config=quant_config,
            reduce_results=True,
            prefix=f"{prefix}.shared_experts",
            swiglu_limit=self.swiglu_limit,
        )

        # Scalar shared-expert sigmoid gate.  Replicated across TP — every
        # rank computes the same per-token scaling factor.
        self.shared_gate = ReplicatedLinear(
            input_size=self.hidden_size,
            output_size=1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_gate",
        )

        expert_hidden_size = self.moe_latent_dim or self.hidden_size
        if self.moe_latent_dim:
            rms_eps = float(
                getattr(config, "rms_norm_eps", getattr(config, "norm_eps", 1e-6))
            )
            self.fc1_latent_proj = ReplicatedLinear(
                input_size=self.hidden_size,
                output_size=self.moe_latent_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fc1_latent_proj",
                return_bias=False,
            )
            self.fc2_latent_proj = ReplicatedLinear(
                input_size=self.moe_latent_dim,
                output_size=self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fc2_latent_proj",
                return_bias=False,
            )
            self.fc1_latent_norm = (
                RMSNorm(self.moe_latent_dim, eps=rms_eps)
                if self.moe_latent_norm
                else nn.Identity()
            )
            self.fc2_latent_norm = (
                RMSNorm(self.moe_latent_dim, eps=rms_eps)
                if self.moe_latent_norm
                else nn.Identity()
            )
        else:
            self.fc1_latent_proj = None
            self.fc2_latent_proj = None
            self.fc1_latent_norm = None
            self.fc2_latent_norm = None

        # NOTE(swiglu_limit): Both the shared expert (above, via
        # ``YOCOClampedSwiGLU``) and the ROUTED experts (below) apply the
        # training ``swiglu_limit`` clamp (clamp-before-silu), for exact
        # train/inference parity. The routed clamp is threaded via
        # ``FusedMoE(swiglu_limit=...)`` -> ``UnquantizedFusedMoEMethod.
        # forward_native`` -> ``FusedMoEExpertsModular.activation`` ->
        # ``swiglu_limit_func`` (gate/up ordering verified identical to
        # ``silu_and_mul``). The W8A8 parity path uses DeepGEMM, which also
        # applies routing probabilities before W2 input quantization. CAUTION:
        # the loose limit=10.0
        # can make some checkpoints (observed: adamw-3000) degenerate under pure
        # greedy decoding; use temperature>0 in production. Kept on per owner's
        # request for training fidelity.
        self.experts = FusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=expert_hidden_size,
            intermediate_size=self.moe_intermediate_size,
            renormalize=True,
            quant_config=quant_config,
            use_grouped_topk=False,
            scoring_func="softmax",
            custom_routing_function=_yoco_topk_routing,
            swiglu_limit=self.swiglu_limit,
            apply_router_weight_before_w2=True,
            prefix=f"{prefix}.experts",
        )

    def _shared_gate_linear(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # llm-train builds shared_gate with default MixPrecisionLinear settings:
        # no MXFP8 path and the parameter follows the module default dtype.
        return F.linear(
            hidden_states,
            self.shared_gate.weight.to(hidden_states.dtype),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        previous_matmul_precision = torch.get_float32_matmul_precision()
        previous_cuda_precision = torch.backends.cuda.matmul.fp32_precision
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        try:
            if self.router_weights_normalized:
                router_logits = F.linear(hidden_states.float(), self.gate.weight)
            else:
                router_logits = _yoco_normalized_router_linear(
                    hidden_states.float(), self.gate.weight
                )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
            torch.set_float32_matmul_precision(previous_matmul_precision)
            torch.backends.cuda.matmul.fp32_precision = previous_cuda_precision
        if self.fc1_latent_proj is not None:
            assert self.fc1_latent_norm is not None
            expert_input = self.fc1_latent_norm(self.fc1_latent_proj(hidden_states))
        else:
            expert_input = hidden_states
        routed_out = self.experts(
            hidden_states=expert_input, router_logits=router_logits
        )
        if self.fc2_latent_proj is not None:
            assert self.fc2_latent_norm is not None
            routed_out = self.fc2_latent_proj(self.fc2_latent_norm(routed_out))

        # YOCO-specific scalar sigmoid gate on the shared expert path.
        # ``shared_gate`` is replicated, so ``scale`` is identical on every TP
        # rank; applying it to the already-reduced shared output is equivalent
        # to applying it before reduction.
        shared_out_unscaled = self.shared_experts(hidden_states)
        scale = self._shared_gate_linear(hidden_states)
        shared_gate_sigmoid = torch.sigmoid(scale)
        shared_out = shared_gate_sigmoid * shared_out_unscaled

        # ``routed_out`` (reduced inside FusedMoE) and ``shared_out`` (reduced
        # inside YOCOSharedExperts.down_proj) are both already tensor-parallel
        # all-reduced, so their sum needs no further reduction.
        final = routed_out + shared_out
        return final.view(num_tokens, hidden_dim)


# --------------------------------------------------------------------------- #
# Decoder layer                                                               #
# --------------------------------------------------------------------------- #


class YOCODecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        hidden_size = _cfg_int(config, "hidden_size", "d_model")
        num_hidden_layers = _cfg_int(config, "num_hidden_layers", "n_layers")
        universal_loop = _cfg_int(config, "universal_loop", default=1)
        yoco_cross_layers = _cfg_int(config, "yoco_cross_layers", default=0)
        first_cross_layer_idx = num_hidden_layers - yoco_cross_layers
        rms_eps = float(
            getattr(config, "rms_norm_eps", getattr(config, "norm_eps", 1e-6))
        )

        self.layer_idx = layer_idx
        self.is_self_layer = layer_idx < first_cross_layer_idx
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_eps)

        if self.is_self_layer:
            self.self_attn = YOCOSelfAttention(
                config=config,
                layer_idx=layer_idx,
                universal_loop=universal_loop,
                num_hidden_layers=num_hidden_layers,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = YOCOCrossAttention(
                config=config,
                layer_idx=layer_idx,
                first_cross_layer_idx=first_cross_layer_idx,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )

        # All layers in this checkpoint are MoE (``dense_layers = 0``).
        self.mlp = YOCOMoE(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        loop_idx: int,
        yoco_key: torch.Tensor | None,
        yoco_value: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        if self.is_self_layer:
            x = self.self_attn(positions, x, loop_idx)
        else:
            assert yoco_key is not None and yoco_value is not None
            x = self.self_attn(x, yoco_key, yoco_value)
        x = residual + x.float()

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x.float()


# --------------------------------------------------------------------------- #
# Cross-decoder block (compiled separately when fast prefill is enabled)      #
# --------------------------------------------------------------------------- #


@support_torch_compile(
    dynamic_arg_dims={
        "positions": 0,
        "hidden_states": 0,
        "yoco_key": 0,
        "yoco_value": 0,
    },
    enable_if=lambda vllm_config: vllm_config.cache_config.kv_sharing_fast_prefill,
)
class YOCOCrossBlock(nn.Module):
    """Wraps cross-attention layers 11..N-1 (KV-sharing layers) for separate
    compilation when --kv-sharing-fast-prefill is enabled."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        cross_layers: list,
        first_cross_layer_idx: int,
    ) -> None:
        super().__init__()
        # Store as plain list to avoid re-registering parameters
        self._cross_layers = cross_layers
        self.first_cross_layer_idx = first_cross_layer_idx

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        yoco_key: torch.Tensor,
        yoco_value: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self._cross_layers:
            hidden_states = layer(
                positions,
                hidden_states,
                0,
                yoco_key,
                yoco_value,
            )
        return hidden_states


# --------------------------------------------------------------------------- #
# Self-decoder block (compiled separately when fast prefill is enabled)       #
# --------------------------------------------------------------------------- #


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
    },
    enable_if=lambda vllm_config: vllm_config.cache_config.kv_sharing_fast_prefill,
)
class YOCOSelfBlock(nn.Module):
    """Self-attention portion compiled as a separate unit for fast prefill.

    Runs (on ALL tokens) the universal-loop self-attention layers, the
    model-level shared-KV producer, and the first cross layer (which owns and
    writes the shared KV cache).  Returns the hidden states together with the
    shared ``yoco_key`` / ``yoco_value`` so the cross block can consume them.
    Keeping this as its own ``@support_torch_compile`` unit (alongside
    ``YOCOCrossBlock``) means the whole fast-prefill forward is an uncompiled
    wrapper around two piecewise CUDA-graph units, which avoids nesting a
    CUDA-graph capture inside an outer full-graph capture."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        model: YOCOModel,
    ) -> None:
        super().__init__()
        # Hold a non-registering reference to the parent model so we reuse its
        # already-registered parameters without duplicating them (assigning an
        # nn.Module attribute directly would re-register it).
        self._model_ref = [model]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self._model_ref[0]
        if inputs_embeds is not None:
            hidden_states = inputs_embeds.float()
        else:
            assert input_ids is not None
            hidden_states = model.embed_tokens(input_ids).float()

        # Self-attention layers (universal loop) — all tokens.
        for loop_idx in range(model.universal_loop):
            for layer_idx in range(model.first_cross_layer_idx):
                hidden_states = model.layers[layer_idx](
                    positions,
                    hidden_states,
                    loop_idx,
                    None,
                    None,
                )

        # Shared-KV producer + first cross layer (owns the shared KV cache);
        # both run on ALL tokens.
        h_norm = model.yoco_norm(hidden_states)
        yoco_key, _ = model.yoco_k_proj(h_norm)
        yoco_value, _ = model.yoco_v_proj(h_norm)
        if model.yoco_k_norm is not None:
            yoco_key = _apply_per_head_norm(
                yoco_key,
                model.yoco_num_kv_heads,
                model.yoco_kv_head_dim,
                model.yoco_k_norm,
            )
        hidden_states = model.layers[model.first_cross_layer_idx](
            positions,
            hidden_states,
            0,
            yoco_key,
            yoco_value,
        )
        return hidden_states, yoco_key, yoco_value


# --------------------------------------------------------------------------- #
# Inner model                                                                 #
# --------------------------------------------------------------------------- #


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    },
    enable_if=lambda vllm_config: not vllm_config.cache_config.kv_sharing_fast_prefill,
)
class YOCOModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: PretrainedConfig = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = _maybe_build_yoco_quant_config(vllm_config.quant_config)
        vllm_config.quant_config = quant_config

        self.config = config
        self.quant_config = quant_config

        self.hidden_size = _cfg_int(config, "hidden_size", "d_model")
        self.vocab_size = _cfg_int(config, "vocab_size")
        self.num_hidden_layers = _cfg_int(config, "num_hidden_layers", "n_layers")
        self.universal_loop = _cfg_int(config, "universal_loop", default=1)
        self.yoco_cross_layers = _cfg_int(config, "yoco_cross_layers", default=0)
        self.first_cross_layer_idx = self.num_hidden_layers - self.yoco_cross_layers
        rms_eps = float(
            getattr(config, "rms_norm_eps", getattr(config, "norm_eps", 1e-6))
        )

        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        # Model-level shared YOCO KV producer.  Active only when there is at
        # least one cross-attention layer.
        if self.yoco_cross_layers > 0:
            cross_kv_head = _cfg_int(
                config, "cross_kv_head", "num_key_value_heads", "kv_head"
            )
            head_dim = _cfg_int(config, "head_dim")
            tp_size = get_tensor_model_parallel_world_size()
            assert cross_kv_head % tp_size == 0 or tp_size % cross_kv_head == 0
            self.yoco_kv_head_dim = head_dim
            self.yoco_num_kv_heads = max(1, cross_kv_head // tp_size)
            # Per-head K norm/clip applied once to the shared ``yoco_key``
            # (mirrors training's model-level ``k_norm`` in ``llm/arch/model.py``).
            self.yoco_k_norm = _build_qk_norm(config, head_dim, rms_eps)
            self.yoco_norm = RMSNorm(self.hidden_size, eps=rms_eps)
            self.yoco_k_proj = ColumnParallelLinear(
                input_size=self.hidden_size,
                output_size=cross_kv_head * head_dim,
                bias=False,
                gather_output=False,
                quant_config=quant_config,
                prefix=f"{prefix}.yoco_k_proj",
            )
            self.yoco_v_proj = ColumnParallelLinear(
                input_size=self.hidden_size,
                output_size=cross_kv_head * head_dim,
                bias=False,
                gather_output=False,
                quant_config=quant_config,
                prefix=f"{prefix}.yoco_v_proj",
            )
        else:
            self.yoco_norm = None
            self.yoco_k_proj = None
            self.yoco_v_proj = None
            self.yoco_k_norm = None

        # Decoder layers.  PP > 1 is out of scope for YOCO (the universal
        # loop and shared cross-KV both couple all layers tightly), so we
        # build the full list and require pp_size == 1.
        assert get_pp_group().world_size == 1, (
            "Pipeline parallelism is not supported for the YOCO model"
        )
        self.layers = nn.ModuleList(
            [
                YOCODecoderLayer(
                    config=config,
                    layer_idx=i,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(self.num_hidden_layers)
            ]
        )
        # ``start_layer``/``end_layer`` are referenced by some shared
        # utilities; expose them for PP=1 coverage.
        self.start_layer = 0
        self.end_layer = self.num_hidden_layers
        self.norm = RMSNorm(self.hidden_size, eps=rms_eps)

        # Fast prefill: split the model into two separately-compiled units so
        # the self portion and the KV-sharing cross layers each get their own
        # piecewise CUDA graph (mirrors gemma3n).  This is required for
        # correctness: the cross block must be invoked during cudagraph warmup,
        # otherwise it tries to capture a graph at inference time (disallowed).
        self.fast_prefill_enabled = cache_config.kv_sharing_fast_prefill
        if self.fast_prefill_enabled and self.yoco_cross_layers > 1:
            # Importing at top level causes issues during tests (see gemma3n).
            from vllm.compilation.backends import set_model_tag

            # Self portion: self layers + shared-KV + first cross layer.
            with set_model_tag("self_decoder"):
                self.self_block = YOCOSelfBlock(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.self_block",
                    model=self,
                )
            # Cross portion: layers first_cross+1 .. N-1 share KV from layer
            # first_cross; only decode tokens are processed during prefill.
            kv_sharing_cross_layers = list(
                self.layers[self.first_cross_layer_idx + 1 :]
            )
            with set_model_tag("cross_decoder"):
                self.cross_block = YOCOCrossBlock(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.cross_block",
                    cross_layers=kv_sharing_cross_layers,
                    first_cross_layer_idx=self.first_cross_layer_idx + 1,
                )

            # Static input buffers for the cross block's CUDA graph.  vLLM runs
            # with cudagraph_copy_inputs=False, so cross-block inputs must have
            # stable addresses across capture/replay.
            max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            kv_dtype = self.embed_tokens.weight.dtype
            device = self.embed_tokens.weight.device
            key_dim = self.yoco_k_proj.weight.shape[0]
            value_dim = self.yoco_v_proj.weight.shape[0]
            self.fp_positions = torch.zeros(
                max_num_tokens, dtype=torch.int64, device=device
            )
            self.fp_hidden_states = torch.zeros(
                (max_num_tokens, self.hidden_size), dtype=torch.float32, device=device
            )
            self.fp_yoco_key = torch.zeros(
                (max_num_tokens, key_dim), dtype=kv_dtype, device=device
            )
            self.fp_yoco_value = torch.zeros(
                (max_num_tokens, value_dim), dtype=kv_dtype, device=device
            )
        else:
            self.self_block = None
            self.cross_block = None

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.hidden_size
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds.float()
        else:
            assert input_ids is not None
            hidden_states = self.embed_tokens(input_ids).float()

        # Universal loop: run layers 0..first_cross_layer_idx-1
        # ``universal_loop`` times.
        for loop_idx in range(self.universal_loop):
            for layer_idx in range(self.first_cross_layer_idx):
                hidden_states = self.layers[layer_idx](
                    positions,
                    hidden_states,
                    loop_idx,
                    None,
                    None,
                )

        # Cross-attention layers (if any).
        if self.yoco_cross_layers > 0:
            assert self.yoco_norm is not None
            assert self.yoco_k_proj is not None
            assert self.yoco_v_proj is not None
            h_norm = self.yoco_norm(hidden_states)
            yoco_key, _ = self.yoco_k_proj(h_norm)
            yoco_value, _ = self.yoco_v_proj(h_norm)
            if self.yoco_k_norm is not None:
                yoco_key = _apply_per_head_norm(
                    yoco_key,
                    self.yoco_num_kv_heads,
                    self.yoco_kv_head_dim,
                    self.yoco_k_norm,
                )
            # No RoPE on cross-layer K (``rope_dim = 0`` in HF config).
            for layer_idx in range(self.first_cross_layer_idx, self.num_hidden_layers):
                hidden_states = self.layers[layer_idx](
                    positions,
                    hidden_states,
                    0,
                    yoco_key,
                    yoco_value,
                )

        hidden_states = self.norm(hidden_states)
        return hidden_states


# --------------------------------------------------------------------------- #
# Top-level CausalLM wrapper                                                  #
# --------------------------------------------------------------------------- #


class YOCOForCausalLM(nn.Module, SupportsPP):
    # Self-attention layers ship q/k/v separately; we fuse into qkv_proj.
    packed_modules_mapping = YOCO_PACKED_MODULES_MAPPING

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: PretrainedConfig = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = _maybe_build_yoco_quant_config(vllm_config.quant_config)
        vllm_config.quant_config = self.quant_config

        self.model = YOCOModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        self.vocab_size = _cfg_int(config, "vocab_size")
        hidden_size = _cfg_int(config, "hidden_size", "d_model")
        self.fast_prefill_enabled = vllm_config.cache_config.kv_sharing_fast_prefill
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=self.vocab_size,
                embedding_dim=hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )

        logit_scale = float(getattr(config, "logit_scale", 1.0))
        self.logits_processor = LogitsProcessor(self.vocab_size, scale=logit_scale)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    # ------------------------------------------------------------------ #
    # standard forward / compute_logits API                              #
    # ------------------------------------------------------------------ #
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.fast_prefill_enabled:
            return self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )
        return self._fast_prefill_forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def _fast_prefill_forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward with fast prefill: cross-attention layers that share KV
        only process decode tokens during prefill.

        The self portion and the cross portion run as two separately-compiled
        ``@support_torch_compile`` units (``self_block`` / ``cross_block``).
        The cross block is *always* executed — during cudagraph warmup and
        dummy/profile runs (when no fast-prefill metadata is available) it
        falls back to ALL tokens — so its CUDA graph is captured during warmup
        instead of (illegally) capturing at inference time."""
        model = self.model

        # No dedicated fast-prefill blocks (e.g. a single cross layer): fall
        # back to the standard dense forward.
        if model.self_block is None or model.cross_block is None:
            return model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )

        # Self portion on ALL tokens (separate piecewise CUDA graph).  Also
        # produces the shared KV (yoco_key / yoco_value) and writes layer
        # ``first_cross_layer_idx``'s shared KV cache.
        hidden_states, yoco_key, yoco_value = model.self_block(
            input_ids,
            positions,
            inputs_embeds,
        )

        # Decode-token indices.  When no fast-prefill metadata is available
        # (cudagraph capture / dummy / profile runs) fall back to ALL tokens so
        # the cross block is always invoked (and thus captured during warmup).
        logits_indices_padded, num_logits_indices = self._get_fast_prefill_indices()
        if logits_indices_padded is None:
            logits_indices_padded = torch.arange(
                positions.size(0),
                dtype=torch.int64,
                device=positions.device,
            )

        # Clone the self-decoder output before it is potentially freed by the
        # piecewise cudagraph machinery when multiple compile units are used.
        out_hidden = hidden_states.clone()

        # Feed the cross block through static buffers — vLLM runs with
        # cudagraph_copy_inputs=False, so inputs need stable addresses.
        n = logits_indices_padded.size(0)
        model.fp_positions[:n].copy_(positions[logits_indices_padded])
        model.fp_hidden_states[:n].copy_(hidden_states[logits_indices_padded])
        model.fp_yoco_key[:n].copy_(yoco_key[logits_indices_padded])
        model.fp_yoco_value[:n].copy_(yoco_value[logits_indices_padded])

        decode_hidden = model.cross_block(
            model.fp_positions[:n],
            model.fp_hidden_states[:n],
            model.fp_yoco_key[:n],
            model.fp_yoco_value[:n],
        )

        # Merge cross-decoder outputs back into the full hidden states.
        if num_logits_indices is not None:
            assert num_logits_indices > 0
            real_indices = logits_indices_padded[:num_logits_indices]
            out_hidden[real_indices] = decode_hidden[:num_logits_indices]
        else:
            out_hidden[logits_indices_padded] = decode_hidden

        return model.norm(out_hidden)

    def _get_fast_prefill_indices(
        self,
    ) -> tuple[torch.Tensor | None, int | None]:
        """Retrieve logits_indices from forward context attention metadata."""
        fwd_ctx = get_forward_context()
        attn_metadata = fwd_ctx.attn_metadata
        if attn_metadata is None:
            return None, None
        if not isinstance(attn_metadata, dict):
            return None, None
        # Find a KV-sharing layer's metadata to get logits_indices.
        # Use the last layer's attention (which is a fast prefill layer).
        last_layer = self.model.layers[-1]
        layer_name = last_layer.self_attn.attn.layer_name
        layer_meta = attn_metadata.get(layer_name)
        if layer_meta is None:
            return None, None
        if isinstance(layer_meta, KVSharingFastPrefillMetadata):
            return layer_meta.logits_indices_padded, layer_meta.num_logits_indices
        return None, None

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, sampling_metadata)

    # ------------------------------------------------------------------ #
    # Weight loading                                                     #
    # ------------------------------------------------------------------ #
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_names: set[str] = set()

        # Stacked Q/K/V mapping (only for self-attention layers 0..9).
        stacked_qkv = [
            ("self_attn.qkv_proj", "self_attn.q_proj", "q"),
            ("self_attn.qkv_proj", "self_attn.k_proj", "k"),
            ("self_attn.qkv_proj", "self_attn.v_proj", "v"),
        ]
        first_cross_layer_idx = _cfg_int(
            self.config, "num_hidden_layers", "n_layers"
        ) - _cfg_int(self.config, "yoco_cross_layers", default=0)
        moe_intermediate_size = _cfg_int(
            self.config, "moe_intermediate_size", "moe_ffn_dim"
        )
        num_experts = _cfg_int(self.config, "num_experts", "moe_expert_num")

        for name, loaded_weight in weights:
            # ------------------------------------------------------------
            # Top-level YOCO shared-KV producer rename.
            # ------------------------------------------------------------
            if name == "model.k_proj.weight":
                name = "model.yoco_k_proj.weight"
            elif name == "model.v_proj.weight":
                name = "model.yoco_v_proj.weight"

            # ------------------------------------------------------------
            # Fused MoE expert tensors (per-expert dispatch).
            # ------------------------------------------------------------
            if name.endswith(".mlp.experts.w13_weight"):
                base = name[: -len(".w13_weight")]
                param_name = f"{base}.w13_weight"
                if param_name not in params_dict:
                    continue
                if is_pp_missing_parameter(param_name, self):
                    continue
                param = params_dict[param_name]
                weight_loader = param.weight_loader
                # HF tensor: (E * 2 * ffn, hidden) — split per expert and
                # then split into w1 (gate, first half) / w3 (up, second).
                w = loaded_weight.view(num_experts, 2 * moe_intermediate_size, -1)
                for expert_id in range(num_experts):
                    w1 = w[expert_id, :moe_intermediate_size, :]
                    w3 = w[expert_id, moe_intermediate_size:, :]
                    weight_loader(
                        param,
                        w1,
                        name,
                        "w1",
                        expert_id,
                    )
                    weight_loader(
                        param,
                        w3,
                        name,
                        "w3",
                        expert_id,
                    )
                loaded_names.add(param_name)
                continue

            if name.endswith(".mlp.experts.w2_weight"):
                base = name[: -len(".w2_weight")]
                param_name = f"{base}.w2_weight"
                if param_name not in params_dict:
                    continue
                if is_pp_missing_parameter(param_name, self):
                    continue
                param = params_dict[param_name]
                weight_loader = param.weight_loader
                # HF tensor: (E * hidden, ffn) — split per expert.
                w = loaded_weight.view(num_experts, -1, moe_intermediate_size)
                for expert_id in range(num_experts):
                    weight_loader(
                        param,
                        w[expert_id],
                        name,
                        "w2",
                        expert_id,
                    )
                loaded_names.add(param_name)
                continue

            # ------------------------------------------------------------
            # Self-attention Q/K/V → qkv_proj (only for self-attn layers).
            # ------------------------------------------------------------
            handled = False
            for stacked, shard_name, shard_id in stacked_qkv:
                if shard_name not in name:
                    continue
                # ``stacked_params_mapping`` would map e.g.
                # ``...self_attn.q_proj.weight`` → ``...self_attn.qkv_proj.weight``.
                # Only applies to layers with stacked qkv (self-attention
                # layers).  Cross-attn layers keep a standalone q_proj.
                layer_idx_str = name.split(".layers.")[-1].split(".")[0]
                try:
                    layer_idx_int = int(layer_idx_str)
                except ValueError:
                    continue
                if (
                    layer_idx_int >= first_cross_layer_idx
                    and shard_name == "self_attn.q_proj"
                ):
                    # Cross-attn standalone q_proj — skip the merging path.
                    break
                # Cross-attn layers don't have k_proj/v_proj at all, so the
                # only way to reach those is on self-attn layers.
                target_name = name.replace(shard_name, stacked)
                if target_name not in params_dict:
                    continue
                if is_pp_missing_parameter(target_name, self):
                    continue
                param = params_dict[target_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_names.add(target_name)
                handled = True
                break
            if handled:
                continue

            # ------------------------------------------------------------
            # Shared-expert gate_up_proj fused tensor.
            # ------------------------------------------------------------
            if name.endswith(".mlp.shared_experts.gate_up_proj.weight"):
                # HF tensor: (2 * intermediate, hidden) — first half gate,
                # second half up.  ``MergedColumnParallelLinear`` accepts
                # the merged tensor via shard_id=0 and shard_id=1 calls.
                if name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                half = loaded_weight.shape[0] // 2
                gate = loaded_weight[:half, :]
                up = loaded_weight[half:, :]
                param.weight_loader(param, gate, 0)
                param.weight_loader(param, up, 1)
                loaded_names.add(name)
                continue

            # ------------------------------------------------------------
            # Default loader: name maps 1:1 to a registered parameter.
            # ------------------------------------------------------------
            if name not in params_dict:
                # Unknown weight — skip silently.  This is rare but happens
                # for e.g. quantization scales we don't use.
                continue
            if is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]
            # Cast bf16 router gate weights to fp32 to match training.
            if name.endswith(".mlp.gate.weight") and param.dtype != loaded_weight.dtype:
                loaded_weight = loaded_weight.to(param.dtype)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            try:
                weight_loader(param, loaded_weight)
            except TypeError:
                # Some param weight loaders take extra positional args; fall
                # back to the default one.
                default_weight_loader(param, loaded_weight)
            loaded_names.add(name)

        return loaded_names
