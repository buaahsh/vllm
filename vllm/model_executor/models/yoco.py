# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO model assembly for vLLM.

Self-attention layers repeat with separate KV caches; cross-attention layers
share K/V produced after the final self pass. Numerical operators, layer
composition, loading, and fast-prefill orchestration have separate modules.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

import torch
from torch import nn
from transformers import PretrainedConfig

import vllm.envs as envs
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.config.kernel import MoEBackend
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.layers.yoco_attention import (
    YOCOCrossAttention as YOCOCrossAttention,
)
from vllm.model_executor.layers.yoco_attention import (
    YOCOSelfAttention as YOCOSelfAttention,
)
from vllm.model_executor.layers.yoco_fast import (
    yoco_fast_linear_fusion,
)
from vllm.model_executor.layers.yoco_moe import YOCOMoE as YOCOMoE
from vllm.model_executor.layers.yoco_ops.norm import RMSClip as RMSClip
from vllm.model_executor.layers.yoco_ops.norm import RMSNorm as RMSNorm
from vllm.model_executor.layers.yoco_ops.norm import (
    _apply_per_head_norm as _apply_per_head_norm,
)
from vllm.model_executor.layers.yoco_ops.norm import _build_qk_norm as _build_qk_norm
from vllm.model_executor.layers.yoco_ops.norm import (
    _yoco_add_residual as _yoco_add_residual,
)
from vllm.model_executor.layers.yoco_ops.norm import (
    _yoco_residual_dtype as _yoco_residual_dtype,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _YOCO_L3_HIDDEN_SIZE as _YOCO_L3_HIDDEN_SIZE,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _YOCO_L3_VOCAB_SIZE as _YOCO_L3_VOCAB_SIZE,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _supports_yoco_sm100_lm_head_kernel as _supports_yoco_sm100_lm_head_kernel,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_can_fuse_fp8_attention as _yoco_can_fuse_fp8_attention,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_lm_head_dispatch as _yoco_lm_head_dispatch,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _YocoAlignEmbeddingMethod as _YocoAlignEmbeddingMethod,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _YocoAlignLinearMethod as _YocoAlignLinearMethod,
)
from vllm.model_executor.models import yoco_prefill, yoco_weights
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.model_executor.models.yoco_config import (
    YOCO_PACKED_MODULES_MAPPING as YOCO_PACKED_MODULES_MAPPING,
)
from vllm.model_executor.models.yoco_config import _cfg_int as _cfg_int
from vllm.model_executor.models.yoco_config import (
    _get_yoco_execution_mode as _get_yoco_execution_mode,
)
from vllm.model_executor.models.yoco_config import (
    _maybe_build_yoco_quant_config as _maybe_build_yoco_quant_config,
)
from vllm.model_executor.models.yoco_config import (
    _yoco_standalone_prefill_min_tokens as _yoco_standalone_prefill_min_tokens,
)
from vllm.model_executor.models.yoco_config import (
    resolve_yoco_fast_moe_backend,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import _encode_layer_name

logger = init_logger(__name__)


# --------------------------------------------------------------------------- #
# Config helpers                                                              #
# --------------------------------------------------------------------------- #


# B200 full-CUDA-graph measurements for L3 TP4 show a stable win once there
# are at least 32 token rows. The TP1 head-group layout is non-regressing from
# the first row and becomes progressively faster as the token count grows.


# --------------------------------------------------------------------------- #
# Self-attention (sliding window, QK-norm, RoPE, diff-attention)              #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Cross-attention (NoPE, no QK-norm, shared KV via kv_sharing)                #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# MoE block                                                                   #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Decoder layer                                                               #
# --------------------------------------------------------------------------- #


class YOCODecoderLayer(nn.Module):
    self_attn: YOCOSelfAttention | YOCOCrossAttention

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
        execution_mode: str = "fast",
        moe_backend_override: MoEBackend | None = None,
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
        self.input_layernorm = RMSNorm(
            hidden_size,
            eps=rms_eps,
            execution_mode=execution_mode,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            eps=rms_eps,
            execution_mode=execution_mode,
        )

        if self.is_self_layer:
            self.self_attn = YOCOSelfAttention(
                config=config,
                layer_idx=layer_idx,
                universal_loop=universal_loop,
                num_hidden_layers=num_hidden_layers,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                execution_mode=execution_mode,
            )
        else:
            self.self_attn = YOCOCrossAttention(
                config=config,
                layer_idx=layer_idx,
                first_cross_layer_idx=first_cross_layer_idx,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                execution_mode=execution_mode,
            )

        # All layers in this checkpoint are MoE (``dense_layers = 0``).
        self.mlp = YOCOMoE(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            execution_mode=execution_mode,
            moe_backend_override=moe_backend_override,
            layer_idx=layer_idx,
        )

    def forward_with_residual(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        loop_idx: int,
        yoco_key: torch.Tensor | None,
        yoco_value: torch.Tensor | None,
        kv_cache_dummy_dep: torch.Tensor | None = None,
        skip_kv_cache_update: bool = False,
        input_residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_residual is None:
            hidden_states = hidden_states.to(self.input_layernorm.residual_dtype)
            residual = hidden_states
            x = self.input_layernorm(hidden_states)
            assert isinstance(x, torch.Tensor)
        else:
            norm_output = self.input_layernorm(hidden_states, input_residual)
            assert isinstance(norm_output, tuple)
            x, residual = norm_output
        if self.is_self_layer:
            x = self.self_attn(positions, x, loop_idx)
        else:
            assert yoco_key is not None and yoco_value is not None
            x = self.self_attn(
                x,
                yoco_key,
                yoco_value,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
                skip_kv_cache_update=skip_kv_cache_update,
            )
        norm_output = self.post_attention_layernorm(x, residual)
        assert isinstance(norm_output, tuple)
        x, residual = norm_output
        x = self.mlp(x, loop_idx)
        return x, residual

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        loop_idx: int,
        yoco_key: torch.Tensor | None,
        yoco_value: torch.Tensor | None,
        kv_cache_dummy_dep: torch.Tensor | None = None,
        skip_kv_cache_update: bool = False,
    ) -> torch.Tensor:
        x, residual = self.forward_with_residual(
            positions,
            hidden_states,
            loop_idx,
            yoco_key,
            yoco_value,
            kv_cache_dummy_dep=kv_cache_dummy_dep,
            skip_kv_cache_update=skip_kv_cache_update,
        )
        # Align and direct layer callers materialize the block output here.
        # Fast model loops call ``forward_with_residual`` instead and carry
        # these two tensors into the next RMSNorm without executing this add.
        return _yoco_add_residual(
            x,
            residual,
            self.input_layernorm.residual_dtype,
            self.input_layernorm.bf16_chain,
        )


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
    """Runs every cross-attention layer on compact logits-token inputs.

    The self block has already written the full shared K/V tensors to the
    first cross layer's cache.  The first layer therefore skips its normal
    cache update while retaining a dependency on that write; all cross layers
    only compute the tokens that need logits.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        cross_layers: list[YOCODecoderLayer],
        first_cross_layer_idx: int,
    ) -> None:
        super().__init__()
        # Store as plain list to avoid re-registering parameters
        self._cross_layers = cross_layers
        self.first_cross_layer_idx = first_cross_layer_idx
        self.execution_mode = _get_yoco_execution_mode(vllm_config)
        self.bf16_chain = self.execution_mode == "fast" and envs.VLLM_YOCO_BF16_CHAIN

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        yoco_key: torch.Tensor,
        yoco_value: torch.Tensor,
        kv_cache_dummy_dep: torch.Tensor,
    ) -> torch.Tensor:
        if getattr(self, "execution_mode", "align") == "fast":
            residual = None
            for index, layer in enumerate(self._cross_layers):
                hidden_states, residual = layer.forward_with_residual(
                    positions,
                    hidden_states,
                    0,
                    yoco_key,
                    yoco_value,
                    kv_cache_dummy_dep=kv_cache_dummy_dep if index == 0 else None,
                    skip_kv_cache_update=index == 0,
                    input_residual=residual,
                )
            assert residual is not None
            # The compact cross result must be materialized before the outer
            # fast-prefill wrapper scatters it into the full-token tensor.
            return _yoco_add_residual(
                hidden_states,
                residual,
                residual.dtype,
                getattr(self, "bf16_chain", False),
            )

        for index, layer in enumerate(self._cross_layers):
            hidden_states = layer(
                positions,
                hidden_states,
                0,
                yoco_key,
                yoco_value,
                kv_cache_dummy_dep=kv_cache_dummy_dep if index == 0 else None,
                skip_kv_cache_update=index == 0,
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

    Runs (on ALL tokens) the universal-loop self-attention layers and the
    model-level shared-KV producer, then writes K/V directly to the first cross
    layer's cache without running that layer.  Returns the hidden states,
    shared ``yoco_key`` / ``yoco_value``, and the cache-write dependency so the
    compact cross block can consume them.
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self._model_ref[0]
        if inputs_embeds is not None:
            hidden_states = inputs_embeds.to(model.residual_dtype)
        else:
            assert input_ids is not None
            hidden_states = model.embed_tokens(input_ids).to(model.residual_dtype)

        # Self-attention layers (universal loop) — all tokens.
        residual = None
        for loop_idx in range(model.universal_loop):
            for layer_idx in range(model.first_cross_layer_idx):
                if model.execution_mode == "fast":
                    hidden_states, residual = model.decoder_layers[
                        layer_idx
                    ].forward_with_residual(
                        positions,
                        hidden_states,
                        loop_idx,
                        None,
                        None,
                        input_residual=residual,
                    )
                else:
                    hidden_states = model.decoder_layers[layer_idx](
                        positions,
                        hidden_states,
                        loop_idx,
                        None,
                        None,
                    )

        # Produce shared K/V for all tokens and write them directly to the
        # first cross layer's cache.  The cross layer itself is deferred to the
        # compact cross block and therefore only runs for logits tokens.
        assert model.yoco_norm is not None
        if residual is None:
            h_norm = model.yoco_norm(hidden_states)
            assert isinstance(h_norm, torch.Tensor)
        else:
            norm_output = model.yoco_norm(hidden_states, residual)
            assert isinstance(norm_output, tuple)
            h_norm, hidden_states = norm_output
        yoco_key, yoco_value = model.project_yoco_kv(h_norm)
        yoco_key, yoco_value = model.normalize_yoco_kv(yoco_key, yoco_value)
        owner_attn = model.get_shared_kv_attention()
        yoco_key_view = yoco_key.view(-1, owner_attn.num_kv_heads, owner_attn.head_size)
        yoco_value_view = yoco_value.view(
            -1, owner_attn.num_kv_heads, owner_attn.head_size_v
        )
        kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
            yoco_key_view,
            yoco_value_view,
            _encode_layer_name(owner_attn.layer_name),
        )
        return hidden_states, yoco_key, yoco_value, kv_cache_dummy_dep


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
)
class YOCOModel(nn.Module):
    yoco_norm: RMSNorm | None
    yoco_kv_proj: MergedColumnParallelLinear | None
    self_block: YOCOSelfBlock | None
    cross_block: YOCOCrossBlock | None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: PretrainedConfig = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = _maybe_build_yoco_quant_config(vllm_config.quant_config)
        vllm_config.quant_config = quant_config

        self.config = config
        self.quant_config = quant_config
        self.execution_mode = _get_yoco_execution_mode(vllm_config)
        self.residual_dtype = _yoco_residual_dtype(self.execution_mode)

        self.hidden_size = _cfg_int(config, "hidden_size", "d_model")
        self.vocab_size = _cfg_int(config, "vocab_size")
        self.num_hidden_layers = _cfg_int(config, "num_hidden_layers", "n_layers")
        self.universal_loop = _cfg_int(config, "universal_loop", default=1)
        self.yoco_cross_layers = _cfg_int(config, "yoco_cross_layers", default=0)
        self.first_cross_layer_idx = self.num_hidden_layers - self.yoco_cross_layers
        tp_size = get_tensor_model_parallel_world_size()
        moe_backend_decision = resolve_yoco_fast_moe_backend(
            execution_mode=self.execution_mode,
            quant_config=quant_config,
            tp_size=tp_size,
            config=config,
            vllm_config=vllm_config,
        )
        moe_backend_decision.apply(vllm_config)
        moe_backend_override = moe_backend_decision.backend
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
            assert cross_kv_head % tp_size == 0 or tp_size % cross_kv_head == 0
            self.yoco_kv_head_dim = head_dim
            self.yoco_num_kv_heads = max(1, cross_kv_head // tp_size)
            self.yoco_local_kv_dim = cross_kv_head * head_dim // tp_size
            # Per-head K norm/clip applied once to the shared ``yoco_key``
            # (mirrors training's model-level ``k_norm`` in ``llm/arch/model.py``).
            self.yoco_k_norm = _build_qk_norm(
                config, head_dim, rms_eps, self.execution_mode
            )
            self.yoco_norm = RMSNorm(
                self.hidden_size,
                eps=rms_eps,
                execution_mode=self.execution_mode,
            )
            kv_width = cross_kv_head * head_dim
            use_merged_yoco_kv, merged_kv_quant_config = yoco_fast_linear_fusion(
                self.execution_mode,
                tp_size,
                quant_config,
                (f"{prefix}.yoco_k_proj", f"{prefix}.yoco_v_proj"),
                self.hidden_size,
                (kv_width, kv_width),
                allow_fp8=True,
            )
            if use_merged_yoco_kv:
                self.yoco_kv_proj = MergedColumnParallelLinear(
                    input_size=self.hidden_size,
                    output_sizes=[
                        cross_kv_head * head_dim,
                        cross_kv_head * head_dim,
                    ],
                    bias=False,
                    gather_output=False,
                    quant_config=merged_kv_quant_config,
                    prefix=f"{prefix}.yoco_kv_proj",
                )
                self.yoco_k_proj = None
                self.yoco_v_proj = None
            else:
                self.yoco_kv_proj = None
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
            self.yoco_kv_proj = None
            self.yoco_k_proj = None
            self.yoco_v_proj = None
            self.yoco_k_norm = None
            self.yoco_local_kv_dim = 0

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
                    execution_mode=self.execution_mode,
                    moe_backend_override=moe_backend_override,
                )
                for i in range(self.num_hidden_layers)
            ]
        )
        self.fuse_fp8_shared_kv = (
            self.yoco_cross_layers > 0
            and isinstance(self.yoco_k_norm, RMSClip)
            and self.yoco_k_norm.weight is not None
            and self.yoco_kv_head_dim == 128
            and _yoco_can_fuse_fp8_attention(
                self.get_shared_kv_attention(),
                self.execution_mode,
            )
        )
        kv_transfer = vllm_config.kv_transfer_config
        if moe_backend_override == "flashinfer_cutlass" and (
            kv_transfer is None or kv_transfer.kv_connector is None
        ):
            fallback_max = _yoco_standalone_prefill_min_tokens(vllm_config) - 1
            use_decode_cutlass = False
            additional = (
                vllm_config.additional_config
                if isinstance(vllm_config.additional_config, Mapping)
                else {}
            )
            capture_max = vllm_config.compilation_config.max_cudagraph_capture_size or 0
            if (
                self.execution_mode == "fast"
                and 0 < capture_max <= 256
                and additional.get("yoco_fast_decode_cutlass", True)
                and not os.getenv("VLLM_YOCO_FLASHINFER_AUTOTUNE_CACHE")
            ):
                from vllm.model_executor.layers.fused_moe.experts.yoco_flashinfer_decode import (  # noqa: E501
                    load_yoco_decode_cutlass_cache,
                )

                use_decode_cutlass = load_yoco_decode_cutlass_cache()
            for layer in self.decoder_layers:
                moe_config = layer.mlp.experts.moe_config
                moe_config.yoco = replace(
                    moe_config.yoco,
                    triton_fallback_max_tokens=fallback_max,
                    fast_decode_cutlass=use_decode_cutlass,
                )
        # ``start_layer``/``end_layer`` are referenced by some shared
        # utilities; expose them for PP=1 coverage.
        self.start_layer = 0
        self.end_layer = self.num_hidden_layers
        self.norm = RMSNorm(
            self.hidden_size,
            eps=rms_eps,
            execution_mode=self.execution_mode,
        )

        # Fast prefill: split the model into two separately-compiled units so
        # the self portion and the KV-sharing cross layers each get their own
        # piecewise CUDA graph (mirrors gemma3n).  This is required for
        # correctness: the cross block must be invoked during cudagraph warmup,
        # otherwise it tries to capture a graph at inference time (disallowed).
        self.fast_prefill_enabled = cache_config.kv_sharing_fast_prefill
        if self.fast_prefill_enabled and self.yoco_cross_layers > 1:
            # Importing at top level causes issues during tests (see gemma3n).
            from vllm.compilation.backends import set_model_tag

            # Self portion: self layers + shared-KV cache write.
            with set_model_tag("self_decoder"):
                self.self_block = YOCOSelfBlock(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.self_block",
                    model=self,
                )
            # Cross portion: every cross layer only processes logits/decode
            # tokens. The first layer's full shared-KV cache was populated by
            # the self block above.
            kv_sharing_cross_layers = list(
                self.decoder_layers[self.first_cross_layer_idx :]
            )
            with set_model_tag("cross_decoder"):
                self.cross_block = YOCOCrossBlock(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.cross_block",
                    cross_layers=kv_sharing_cross_layers,
                    first_cross_layer_idx=self.first_cross_layer_idx + 1,
                )
            self.full_model_warmed = False

            # Static input buffers for the cross block's CUDA graph.  vLLM runs
            # with cudagraph_copy_inputs=False, so cross-block inputs must have
            # stable addresses across capture/replay.
            max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            kv_dtype = (
                torch.float8_e4m3fn
                if self.fuse_fp8_shared_kv
                else cast(torch.Tensor, self.embed_tokens.weight).dtype
            )
            device = cast(torch.Tensor, self.embed_tokens.weight).device
            key_dim = self.yoco_local_kv_dim
            value_dim = self.yoco_local_kv_dim
            self.fp_positions = torch.zeros(
                max_num_tokens, dtype=torch.int64, device=device
            )
            self.fp_hidden_states = torch.zeros(
                (max_num_tokens, self.hidden_size),
                dtype=self.residual_dtype,
                device=device,
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
            self.full_model_warmed = True

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.hidden_size
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    @property
    def decoder_layers(self) -> Sequence[YOCODecoderLayer]:
        return cast(Sequence[YOCODecoderLayer], self.layers)

    def get_shared_kv_attention(self) -> Attention:
        layer = self.decoder_layers[self.first_cross_layer_idx]
        return cast(YOCOCrossAttention, layer.self_attn).attn

    def normalize_yoco_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fuse_fp8_shared_kv:
            owner = self.get_shared_kv_attention()
            key, value = torch.ops.vllm.yoco_clip_fp8(
                key.view(-1, self.yoco_num_kv_heads, self.yoco_kv_head_dim),
                self.yoco_k_norm.weight,
                owner._k_scale,
                self.yoco_k_norm.eps,
                self.yoco_k_norm.limit,
                value.view(-1, self.yoco_num_kv_heads, self.yoco_kv_head_dim),
                owner._v_scale,
                # Inductor fuses the shared-key pre-affine cast away. Eager
                # RMSClip materializes it; preserve each existing behavior.
                round_before_weight=not torch.compiler.is_compiling(),
            )
            return key.flatten(1), value.flatten(1)
        if self.yoco_k_norm is not None:
            key = _apply_per_head_norm(
                key, self.yoco_num_kv_heads, self.yoco_kv_head_dim, self.yoco_k_norm
            )
        return key, value

    def project_yoco_kv(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the model-level K/V, merging the two GEMMs in Fast TP1."""
        if self.yoco_kv_proj is not None:
            yoco_kv, _ = self.yoco_kv_proj(hidden_states)
            yoco_key, yoco_value = yoco_kv.split(self.yoco_local_kv_dim, dim=-1)
            return yoco_key, yoco_value
        assert self.yoco_k_proj is not None and self.yoco_v_proj is not None
        yoco_key, _ = self.yoco_k_proj(hidden_states)
        yoco_value, _ = self.yoco_v_proj(hidden_states)
        return yoco_key, yoco_value

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds.to(self.residual_dtype)
        else:
            assert input_ids is not None
            hidden_states = self.embed_tokens(input_ids).to(self.residual_dtype)

        # Universal loop: run layers 0..first_cross_layer_idx-1
        # ``universal_loop`` times.
        residual = None
        for loop_idx in range(self.universal_loop):
            for layer_idx in range(self.first_cross_layer_idx):
                if self.execution_mode == "fast":
                    hidden_states, residual = self.decoder_layers[
                        layer_idx
                    ].forward_with_residual(
                        positions,
                        hidden_states,
                        loop_idx,
                        None,
                        None,
                        input_residual=residual,
                    )
                else:
                    hidden_states = self.decoder_layers[layer_idx](
                        positions,
                        hidden_states,
                        loop_idx,
                        None,
                        None,
                    )

        # Cross-attention layers (if any).
        if self.yoco_cross_layers > 0:
            assert self.yoco_norm is not None
            if residual is None:
                h_norm = self.yoco_norm(hidden_states)
                assert isinstance(h_norm, torch.Tensor)
            else:
                norm_output = self.yoco_norm(hidden_states, residual)
                assert isinstance(norm_output, tuple)
                h_norm, hidden_states = norm_output
                residual = None
            yoco_key, yoco_value = self.project_yoco_kv(h_norm)
            yoco_key, yoco_value = self.normalize_yoco_kv(yoco_key, yoco_value)
            # No RoPE on cross-layer K (``rope_dim = 0`` in HF config).
            for layer_idx in range(self.first_cross_layer_idx, self.num_hidden_layers):
                if self.execution_mode == "fast":
                    hidden_states, residual = self.decoder_layers[
                        layer_idx
                    ].forward_with_residual(
                        positions,
                        hidden_states,
                        0,
                        yoco_key,
                        yoco_value,
                        input_residual=residual,
                    )
                else:
                    hidden_states = self.decoder_layers[layer_idx](
                        positions,
                        hidden_states,
                        0,
                        yoco_key,
                        yoco_value,
                    )

        if residual is None:
            output = self.norm(hidden_states)
            assert isinstance(output, torch.Tensor)
        else:
            norm_output = self.norm(hidden_states, residual)
            assert isinstance(norm_output, tuple)
            output, _ = norm_output
        return output


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
        self._yoco_weight_load_report: yoco_weights.YocoWeightLoadReport | None = None
        self.quant_config = _maybe_build_yoco_quant_config(vllm_config.quant_config)
        vllm_config.quant_config = self.quant_config

        self.model = YOCOModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.execution_mode = self.model.execution_mode
        self.bf16_sampling = (
            self.execution_mode == "fast" and envs.VLLM_YOCO_BF16_SAMPLING
        )

        self.vocab_size = _cfg_int(config, "vocab_size")
        hidden_size = _cfg_int(config, "hidden_size", "d_model")
        self.fast_prefill_enabled = vllm_config.cache_config.kv_sharing_fast_prefill
        tie_word_embeddings = bool(getattr(config, "tie_word_embeddings", False))
        if tie_word_embeddings:
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
        self.use_sm100_lm_head_kernel = (
            _supports_yoco_sm100_lm_head_kernel(self.execution_mode)
            # Online FP8 leaves ParallelLMHead unquantized. Check this layer,
            # rather than disabling the BF16 kernel for the entire model.
            and type(getattr(self.lm_head, "quant_method", None))
            is UnquantizedEmbeddingMethod
            and not tie_word_embeddings
            and get_tensor_model_parallel_world_size() == 1
            and hidden_size == _YOCO_L3_HIDDEN_SIZE
            and self.vocab_size == _YOCO_L3_VOCAB_SIZE
            and logit_scale == 1.0
        )

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self._rotary_caches_initialized = False
        self._router_weight_caches_initialized = False
        self._shared_expert_weight_caches_initialized = False
        if self.execution_mode == "align" and current_platform.is_cuda():
            # Scope the launch policy to this YOCO instance, including latent,
            # output, shared-KV, shared-expert projections and the LM head.
            for module in self.modules():
                method = getattr(module, "quant_method", None)
                if isinstance(method, UnquantizedLinearMethod):
                    cast(Any, module).quant_method = _YocoAlignLinearMethod()
                elif isinstance(method, UnquantizedEmbeddingMethod):
                    cast(Any, module).quant_method = _YocoAlignEmbeddingMethod()

    # ------------------------------------------------------------------ #
    # standard forward / compute_logits API                              #
    # ------------------------------------------------------------------ #
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def _initialize_rotary_caches(self) -> None:
        return yoco_weights._initialize_rotary_caches(self)

    def _initialize_router_weight_caches(self) -> None:
        return yoco_weights._initialize_router_weight_caches(self)

    def _initialize_shared_expert_weight_caches(self) -> None:
        return yoco_weights._initialize_shared_expert_weight_caches(self)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        kv_only_prefill: bool = False,
    ) -> torch.Tensor:
        # Default loading initializes this eagerly. This fallback covers
        # loaders that assign tensors without calling this model's load_weights.
        self._initialize_rotary_caches()
        self._initialize_router_weight_caches()
        self._initialize_shared_expert_weight_caches()
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
            kv_only_prefill=kv_only_prefill,
        )

    def _fast_prefill_forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        kv_only_prefill: bool = False,
    ) -> torch.Tensor:
        return yoco_prefill._fast_prefill_forward(
            self,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            kv_only_prefill,
        )

    def _get_fast_prefill_indices(
        self,
    ) -> tuple[torch.Tensor | None, int | None, torch.Tensor | None]:
        return yoco_prefill._get_fast_prefill_indices(self)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor:
        if (
            getattr(self, "use_sm100_lm_head_kernel", False)
            and sampling_metadata is None
        ):
            logits = _yoco_lm_head_dispatch(
                hidden_states,
                cast(torch.Tensor, self.lm_head.weight),
                use_sm100_kernel=True,
                output_dtype=torch.bfloat16
                if getattr(self, "bf16_sampling", False)
                else torch.float32,
            )
            if self.logits_processor.scale != 1.0:
                logits *= self.logits_processor.scale
            return logits
        return self.logits_processor(self.lm_head, hidden_states, sampling_metadata)

    # ------------------------------------------------------------------ #
    # Weight loading                                                     #
    # ------------------------------------------------------------------ #
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return yoco_weights.load_weights(self, weights)


def __getattr__(name: str):
    # Legacy imports for external experiment snapshots; new code uses the
    # canonical operator and layer modules.
    from vllm.model_executor.models import yoco_compat

    return getattr(yoco_compat, name)
