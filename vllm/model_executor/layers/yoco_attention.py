# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention.attention import Attention, AttentionType
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.yoco_fast import yoco_fast_linear_fusion
from vllm.model_executor.layers.yoco_ops.norm import RMSClip as RMSClip
from vllm.model_executor.layers.yoco_ops.norm import (
    _apply_per_head_norm as _apply_per_head_norm,
)
from vllm.model_executor.layers.yoco_ops.norm import _build_qk_norm as _build_qk_norm
from vllm.model_executor.layers.yoco_ops.projection import (
    _YOCO_Q_LAMBDA_MERGED_MAX_TOKENS as _YOCO_Q_LAMBDA_MERGED_MAX_TOKENS,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _YOCO_QKV_LAMBDA_MERGED_MAX_TOKENS as _YOCO_QKV_LAMBDA_MERGED_MAX_TOKENS,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _supports_yoco_sm100_diff_v3_kernel as _supports_yoco_sm100_diff_v3_kernel,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_align_qkv_linear as _yoco_align_qkv_linear,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_can_fuse_fp8_attention as _yoco_can_fuse_fp8_attention,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_can_fuse_fp8_output as _yoco_can_fuse_fp8_output,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_diff_attention_v2 as _yoco_diff_attention_v2,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_diff_attention_v3_dispatch as _yoco_diff_attention_v3_dispatch,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_project_fp8_output as _yoco_project_fp8_output,
)
from vllm.model_executor.layers.yoco_ops.rotary import (
    YOCORotaryEmbedding as YOCORotaryEmbedding,
)
from vllm.model_executor.models.yoco_config import (
    _cfg_int,
    _yoco_runtime_sliding_window,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON


class YOCOSelfAttention(nn.Module):
    """Sliding-window self-attention for YOCO layers 0..9.

    Creates ``universal_loop`` distinct ``Attention`` sub-modules that share
    the projection weights but use unique KV cache prefixes so each universal
    loop iteration gets its own KV cache slot.
    """

    qkv_lambda_proj: MergedColumnParallelLinear | None

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        universal_loop: int,
        num_hidden_layers: int,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
        execution_mode: str = "fast",
    ) -> None:
        super().__init__()
        if execution_mode not in ("align", "fast"):
            raise ValueError(f"Unsupported YOCO execution mode: {execution_mode!r}")
        if execution_mode == "align" and quant_config is not None:
            raise ValueError("YOCO --align currently supports BF16 weights only")
        self.execution_mode = execution_mode
        self.use_sm100_diff_v3_kernel = _supports_yoco_sm100_diff_v3_kernel(
            execution_mode
        )
        self.hidden_size = _cfg_int(config, "hidden_size", "d_model")
        self.total_num_heads = _cfg_int(config, "num_attention_heads", "head")
        self.total_num_kv_heads = _cfg_int(config, "num_key_value_heads", "kv_head")
        self.head_dim = _cfg_int(config, "head_dim")
        self.diff_v3 = bool(getattr(config, "diff_v3", False))
        self.layer_idx = layer_idx
        self.universal_loop = universal_loop
        self.num_hidden_layers = num_hidden_layers
        self.training_sliding_window = _cfg_int(
            config, "sliding_window_size", "yoco_window_size", default=512
        )
        self.sliding_window = _yoco_runtime_sliding_window(self.training_sliding_window)
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
            f"gate_heads={gate_heads} must be divisible by TP size {tp_size}"
        )
        self.num_lambda_heads = self.total_num_heads // tp_size
        self.num_gate_heads = gate_heads // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        use_merged_qkv_lambda, _ = yoco_fast_linear_fusion(
            execution_mode,
            tp_size,
            quant_config,
            (f"{prefix}.qkv_proj",),
            self.hidden_size,
            (self.q_size, self.kv_size, self.kv_size, gate_heads),
        )
        use_merged_qkv_lambda &= torch.get_default_dtype() == torch.bfloat16
        if use_merged_qkv_lambda:
            self.qkv_lambda_proj = MergedColumnParallelLinear(
                input_size=self.hidden_size,
                output_sizes=[
                    q_heads * self.head_dim,
                    self.total_num_kv_heads * self.head_dim,
                    self.total_num_kv_heads * self.head_dim,
                    gate_heads,
                ],
                bias=False,
                gather_output=False,
                quant_config=None,
                prefix=f"{prefix}.qkv_lambda_proj",
            )
            self.qkv_proj = None
            self.lambda_proj = None
        else:
            self.qkv_lambda_proj = None
            self.qkv_proj = QKVParallelLinear(
                hidden_size=self.hidden_size,
                head_size=self.head_dim,
                total_num_heads=q_heads,
                total_num_kv_heads=self.total_num_kv_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            # Keep the legacy HF name ``lambda_proj`` for checkpoint
            # compatibility. Diff-v2 has one gate per head pair; diff-v3 has
            # one per attention head.
            self.lambda_proj = ColumnParallelLinear(
                input_size=self.hidden_size,
                output_size=gate_heads,
                bias=False,
                gather_output=False,
                # llm-train constructs lambda_proj with default
                # MixPrecisionLinear, so it stays BF16 even when the rest of
                # attention uses MXFP8.
                quant_config=None,
                prefix=f"{prefix}.lambda_proj",
            )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.fuse_fp8_output = (
            self.diff_v3
            and self.head_dim == 128
            and self.num_lambda_heads % 4 == 0
            and _yoco_can_fuse_fp8_output(self.o_proj, execution_mode)
        )
        rms_eps = float(
            getattr(config, "rms_norm_eps", getattr(config, "norm_eps", 1e-6))
        )
        # Per-head Q/K normalization (RMSClip when ``qk_rms_clip``, RMSNorm
        # when ``qk_norm``, else nothing). L3 uses affine RMSClip weights.
        self.q_norm = _build_qk_norm(config, self.head_dim, rms_eps, execution_mode)
        self.k_norm = _build_qk_norm(config, self.head_dim, rms_eps, execution_mode)

        self.rotary_emb = YOCORotaryEmbedding(
            head_size=self.head_dim,
            max_position_embeddings=max_position,
            base=rope_theta,
            execution_mode=execution_mode,
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

        self.fuse_fp8_attention = all(
            _yoco_can_fuse_fp8_attention(cast(Attention, attention), execution_mode)
            for attention in self.attn
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
        if self.diff_v3:
            out = _yoco_diff_attention_v3_dispatch(
                attn_view, gate, self.use_sm100_diff_v3_kernel
            )
        else:
            attn1 = attn_view[:, 0::2, :]
            attn2 = attn_view[:, 1::2, :]
            out = _yoco_diff_attention_v2(attn1, attn2, gate)
        return out.reshape(-1, num_heads_per_pair * self.head_dim)

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #
    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.execution_mode == "align":
            assert self.qkv_proj is not None
            q, k, v = _yoco_align_qkv_linear(
                hidden_states,
                cast(torch.Tensor, self.qkv_proj.weight),
                self.q_size,
                self.kv_size,
            )
            return q, k, v, None

        if self.qkv_lambda_proj is None:
            assert self.qkv_proj is not None
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            return q, k, v, None

        qkv_size = self.q_size + 2 * self.kv_size
        if hidden_states.shape[0] <= _YOCO_QKV_LAMBDA_MERGED_MAX_TOKENS:
            qkv_lambda, _ = self.qkv_lambda_proj(hidden_states)
            q, k, v, gate = qkv_lambda.split(
                [
                    self.q_size,
                    self.kv_size,
                    self.kv_size,
                    self.num_gate_heads,
                ],
                dim=-1,
            )
            return q, k, v, gate

        # B200's merged L3 N=10304 GEMM wins through M=4096, but regresses
        # beyond that. Reuse the packed parameter's QKV prefix without keeping
        # a duplicate QKV weight.
        qkv_weight = cast(torch.Tensor, self.qkv_lambda_proj.weight).narrow(
            0, 0, qkv_size
        )
        qkv = F.linear(hidden_states, qkv_weight)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v, None

    def _project_lambda(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lambda_proj is not None:
            gate, _ = self.lambda_proj(hidden_states)
            return gate
        assert self.qkv_lambda_proj is not None
        qkv_size = self.q_size + 2 * self.kv_size
        lambda_weight = cast(torch.Tensor, self.qkv_lambda_proj.weight).narrow(
            0,
            qkv_size,
            self.num_gate_heads,
        )
        return F.linear(hidden_states, lambda_weight)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        loop_idx: int,
    ) -> torch.Tensor:
        q, k, v, gate = self._project_qkv(hidden_states)
        # Per-head QK norm/clip on the un-rotated query/key, applied
        # independently to each head's ``head_dim`` slice — matches training's
        # ``q_norm``/``k_norm``.  Skipped entirely when neither ``qk_rms_clip``
        # nor ``qk_norm`` is set.
        qk_clip_weight_mode_matches = (
            isinstance(self.q_norm, RMSClip)
            and isinstance(self.k_norm, RMSClip)
            and (self.q_norm.weight is None) == (self.k_norm.weight is None)
        )
        can_fuse_qk_clip_rotary = (
            self.execution_mode == "fast"
            and HAS_TRITON
            and q.is_cuda
            and q.dtype == torch.bfloat16
            and k.dtype == torch.bfloat16
            and self.head_dim == 128
            and isinstance(self.q_norm, RMSClip)
            and isinstance(self.k_norm, RMSClip)
            and qk_clip_weight_mode_matches
            and self.q_norm.eps == self.k_norm.eps
            and self.q_norm.limit == self.k_norm.limit
            and current_platform.is_cuda()
        )
        if can_fuse_qk_clip_rotary:
            cache = self.rotary_emb._get_cos_sin_cache(q.device)
            q = q.view(q.shape[0], self.num_heads, self.head_dim)
            k = k.view(k.shape[0], self.num_kv_heads, self.head_dim)
            positions = positions.to(device=q.device, dtype=torch.long)
            if self.fuse_fp8_attention:
                attention = self.attn[loop_idx]
                q, k, v = torch.ops.vllm.yoco_qkv_clip_rotary_fp8(
                    q,
                    k,
                    v.view(-1, self.num_kv_heads, self.head_dim),
                    self.q_norm.weight,
                    self.k_norm.weight,
                    positions,
                    cache,
                    attention._q_scale,
                    attention._k_scale,
                    attention._v_scale,
                    self.q_norm.eps,
                    self.q_norm.limit,
                )
            elif self.q_norm.weight is None:
                q, k = torch.ops.vllm.yoco_qk_rms_clip_rotary(
                    q,
                    k,
                    positions,
                    cache,
                    self.q_norm.eps,
                    self.q_norm.limit,
                )
            else:
                assert self.k_norm.weight is not None
                q, k = torch.ops.vllm.yoco_qk_rms_clip_rotary_weighted(
                    q,
                    k,
                    self.q_norm.weight,
                    self.k_norm.weight,
                    positions,
                    cache,
                    self.q_norm.eps,
                    self.q_norm.limit,
                )
            q = q.flatten(-2)
            k = k.flatten(-2)
        else:
            if self.q_norm is not None:
                q = _apply_per_head_norm(q, self.num_heads, self.head_dim, self.q_norm)
                k = _apply_per_head_norm(
                    k, self.num_kv_heads, self.head_dim, self.k_norm
                )
            q, k = self.rotary_emb(positions, q, k)
        attn_out = self.attn[loop_idx](
            q,
            k,
            v,
            output_dtype=torch.bfloat16 if q.dtype == torch.float8_e4m3fn else None,
        )

        if gate is None:
            gate = self._project_lambda(hidden_states)
        if self.fuse_fp8_output:
            return _yoco_project_fp8_output(
                self.o_proj, attn_out, gate, self.num_lambda_heads, self.head_dim
            )
        out = self._diff_attention_combine(attn_out, gate, self.num_lambda_heads)
        out, _ = self.o_proj(out)
        return out


class YOCOCrossAttention(nn.Module):
    """YOCO cross-attention layer (layers 10..19).

    These layers have only ``q_proj`` / ``o_proj`` / ``lambda_proj`` — they
    share a single set of (K, V) produced once at the model level.  Layer 10
    owns the shared KV cache; subsequent cross-layers point their
    ``kv_sharing_target_layer_name`` at layer 10's attention to reuse the
    cache without writing.
    """

    q_lambda_proj: MergedColumnParallelLinear | None

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        first_cross_layer_idx: int,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
        execution_mode: str = "fast",
    ) -> None:
        super().__init__()
        self.execution_mode = execution_mode
        self.use_sm100_diff_v3_kernel = _supports_yoco_sm100_diff_v3_kernel(
            execution_mode
        )
        # FP8 projections still produce BF16 activations. The dispatch checks
        # the actual activation and norm-weight dtype, shape and device.
        self.use_sm100_weighted_rms_clip_kernel = self.use_sm100_diff_v3_kernel
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
        self.q_size = self.num_heads * self.head_dim
        use_merged_q_lambda, _ = yoco_fast_linear_fusion(
            execution_mode,
            tp_size,
            quant_config,
            (f"{prefix}.q_proj",),
            self.hidden_size,
            (self.q_size, gate_heads),
        )
        if use_merged_q_lambda:
            self.q_lambda_proj = MergedColumnParallelLinear(
                input_size=self.hidden_size,
                output_sizes=[q_heads * self.head_dim, gate_heads],
                bias=False,
                gather_output=False,
                quant_config=None,
                prefix=f"{prefix}.q_lambda_proj",
            )
            self.q_proj = None
            self.lambda_proj = None
        else:
            self.q_lambda_proj = None
            self.q_proj = ColumnParallelLinear(
                input_size=self.hidden_size,
                output_size=q_heads * self.head_dim,
                bias=False,
                gather_output=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
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
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.fuse_fp8_output = (
            self.diff_v3
            and self.head_dim == 128
            and self.num_lambda_heads % 4 == 0
            and _yoco_can_fuse_fp8_output(self.o_proj, execution_mode)
        )
        rms_eps = float(
            getattr(config, "rms_norm_eps", getattr(config, "norm_eps", 1e-6))
        )
        # Cross layers apply the same per-head Q norm/clip as self layers (the
        # shared K is normed once at the model level on ``yoco_key``).
        self.q_norm = _build_qk_norm(config, self.head_dim, rms_eps, execution_mode)

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

        self.fuse_fp8_attention = _yoco_can_fuse_fp8_attention(
            self.attn, execution_mode
        )

    def _diff_attention_combine(
        self, attn_out: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        attn_view = attn_out.view(-1, 2 * self.num_lambda_heads, self.head_dim)
        if self.diff_v3:
            out = _yoco_diff_attention_v3_dispatch(
                attn_view, gate, self.use_sm100_diff_v3_kernel
            )
        else:
            attn1 = attn_view[:, 0::2, :]
            attn2 = attn_view[:, 1::2, :]
            out = _yoco_diff_attention_v2(attn1, attn2, gate)
        return out.reshape(-1, self.num_lambda_heads * self.head_dim)

    def _project_query(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.q_lambda_proj is None:
            assert self.q_proj is not None
            q, _ = self.q_proj(hidden_states)
            return q, None

        if hidden_states.shape[0] <= _YOCO_Q_LAMBDA_MERGED_MAX_TOKENS:
            q_lambda, _ = self.q_lambda_proj(hidden_states)
            q, gate = q_lambda.split(
                [self.q_size, self.num_gate_heads],
                dim=-1,
            )
            return q, gate

        # B200's merged N=8256 GEMM wins through M=2048, but can regress at
        # larger prefill shapes (notably M=4096). Reuse the packed parameter
        # as two contiguous views and preserve the original projection order.
        q_weight = cast(torch.Tensor, self.q_lambda_proj.weight).narrow(
            0, 0, self.q_size
        )
        return F.linear(hidden_states, q_weight), None

    def _project_lambda(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lambda_proj is not None:
            gate, _ = self.lambda_proj(hidden_states)
            return gate
        assert self.q_lambda_proj is not None
        lambda_weight = cast(torch.Tensor, self.q_lambda_proj.weight).narrow(
            0,
            self.q_size,
            self.num_gate_heads,
        )
        return F.linear(hidden_states, lambda_weight)

    def _normalize_query(self, q: torch.Tensor) -> torch.Tensor:
        if self.q_norm is None:
            return q
        can_use_sm100_weighted_rms_clip = (
            self.use_sm100_weighted_rms_clip_kernel
            and isinstance(self.q_norm, RMSClip)
            and self.q_norm.weight is not None
            and q.is_cuda
            and q.dtype == torch.bfloat16
            and self.q_norm.weight.dtype == torch.bfloat16
            and q.ndim == 2
            and q.shape[-1] == 64 * 128
            and self.num_heads == 64
            and self.head_dim == 128
            and q.stride(-1) == 1
        )
        if can_use_sm100_weighted_rms_clip:
            q_view = q.unflatten(-1, (self.num_heads, self.head_dim))
            if self.fuse_fp8_attention:
                output, _ = torch.ops.vllm.yoco_clip_fp8(
                    q_view,
                    self.q_norm.weight,
                    self.attn._q_scale,
                    self.q_norm.eps,
                    self.q_norm.limit,
                )
                return output.flatten(-2)
            return torch.ops.vllm.yoco_weighted_rms_clip(
                q_view,
                self.q_norm.weight,
                self.q_norm.eps,
                self.q_norm.limit,
            ).flatten(-2)
        return _apply_per_head_norm(q, self.num_heads, self.head_dim, self.q_norm)

    def forward(
        self,
        hidden_states: torch.Tensor,
        yoco_key: torch.Tensor,
        yoco_value: torch.Tensor,
        kv_cache_dummy_dep: torch.Tensor | None = None,
        skip_kv_cache_update: bool = False,
    ) -> torch.Tensor:
        q, gate = self._project_query(hidden_states)
        q = self._normalize_query(q)
        attn_out = self.attn(
            q,
            yoco_key,
            yoco_value,
            kv_cache_dummy_dep=kv_cache_dummy_dep,
            skip_kv_cache_update=skip_kv_cache_update,
            output_dtype=torch.bfloat16 if q.dtype == torch.float8_e4m3fn else None,
        )
        if gate is None:
            gate = self._project_lambda(hidden_states)
        if self.fuse_fp8_output:
            return _yoco_project_fp8_output(
                self.o_proj, attn_out, gate, self.num_lambda_heads, self.head_dim
            )
        out = self._diff_attention_combine(attn_out, gate)
        out, _ = self.o_proj(out)
        return out
