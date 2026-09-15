# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm.config import VllmConfig
from vllm.config.kernel import MoEBackend
from vllm.config.yoco import get_yoco_execution_mode
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.yoco_ops.projection import (
    _YOCO_L3_HIDDEN_SIZE as _YOCO_L3_HIDDEN_SIZE,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _yoco_verified_trtllm_cache_max_capture(cache_path: str | None) -> int:
    """Return the largest graph backed by a YOCO-validated tactic cache."""
    if not cache_path:
        return 32
    # Any environment-compatible persisted cache retains the already validated
    # graph64 behavior. Graph128/256 are admitted only for the real-routing
    # tactic sets; the earlier synthetic M128 tactic regressed end-to-end.
    max_capture = 64
    try:
        with open(cache_path) as cache_file:
            configs = json.load(cache_file)
    except (OSError, TypeError, ValueError):
        return max_capture

    expected = {
        32: [32, 24],
        64: [32, 17],
        128: [32, 17],
    }
    for num_tokens, tactic in expected.items():
        key = (
            "('flashinfer::trtllm_bf16_moe', 'MoERunner', "
            f"(({num_tokens}, 1024), (0,), ({num_tokens}, 8), "
            f"({num_tokens}, 8), ({num_tokens}, 1024), (0,), (0,), (0,)), ())"
        )
        if configs.get(key) != ["MoERunner", tactic]:
            return max_capture
    max_capture = 128
    key = (
        "('flashinfer::trtllm_bf16_moe', 'MoERunner', "
        "((256, 1024), (0,), (256, 8), (256, 8), (256, 1024), "
        "(0,), (0,), (0,)), ())"
    )
    if configs.get(key) == ["MoERunner", [64, 0]]:
        return 256
    return max_capture


def _yoco_standalone_prefill_min_tokens(vllm_config: VllmConfig) -> int:
    """Keep every configured pure-decode graph on the Triton path."""
    return max(
        1024,
        vllm_config.scheduler_config.max_num_seqs + 1,
        int(vllm_config.compilation_config.max_cudagraph_capture_size or 0) + 1,
    )


@dataclass(frozen=True)
class YocoMoEBackendDecision:
    backend: MoEBackend | None = None
    enable_flashinfer_autotune: bool | None = None
    reason: str = "configured backend retained"

    def apply(self, vllm_config: VllmConfig) -> None:
        if self.enable_flashinfer_autotune is not None:
            vllm_config.kernel_config.enable_flashinfer_autotune = (
                self.enable_flashinfer_autotune
            )
        logger.info_once(
            "YOCO MoE backend %s: %s",
            self.backend or vllm_config.kernel_config.moe_backend,
            self.reason,
        )


def resolve_yoco_fast_moe_backend(
    *,
    execution_mode: str,
    quant_config: QuantizationConfig | None,
    tp_size: int,
    config: PretrainedConfig,
    vllm_config: VllmConfig,
) -> YocoMoEBackendDecision:
    """Resolve the backend and startup adjustments without modifying configuration."""
    if execution_mode != "fast" or quant_config is not None or tp_size != 1:
        return YocoMoEBackendDecision(reason="Fast BF16 TP1 policy does not apply")
    additional_config = (
        vllm_config.additional_config
        if isinstance(vllm_config.additional_config, Mapping)
        else {}
    )
    kv_transfer_config = vllm_config.kv_transfer_config
    standalone = kv_transfer_config is None or kv_transfer_config.kv_connector is None
    role = (
        kv_transfer_config.kv_role
        if kv_transfer_config is not None and not standalone
        else None
    )
    kernel_config = vllm_config.kernel_config
    enable_decode_autotune = False
    if standalone:
        if not bool(additional_config.get("yoco_fast_standalone_flashinfer_moe", True)):
            return YocoMoEBackendDecision(reason="standalone FlashInfer disabled")
        parallel = getattr(vllm_config, "parallel_config", None)
        model_config = getattr(vllm_config, "model_config", None)
        if (
            parallel is None
            or getattr(parallel, "data_parallel_size", 0) != 1
            or getattr(parallel, "pipeline_parallel_size", 0) != 1
            or (getattr(parallel, "prefill_context_parallel_size", 1) != 1)
            or (getattr(parallel, "decode_context_parallel_size", 1) != 1)
            or (getattr(model_config, "dtype", None) != torch.bfloat16)
            or (not vllm_config.cache_config.kv_sharing_fast_prefill)
            or (kernel_config.enable_flashinfer_autotune is True)
            or (
                vllm_config.scheduler_config.max_num_batched_tokens
                < _yoco_standalone_prefill_min_tokens(vllm_config)
            )
        ):
            return YocoMoEBackendDecision(
                reason="standalone topology or graph policy does not match"
            )
    elif role == "kv_producer":
        if (
            not bool(additional_config.get("yoco_fast_prefill_flashinfer_moe", True))
            or not vllm_config.cache_config.kv_sharing_fast_prefill
            or kernel_config.enable_flashinfer_autotune is True
        ):
            return YocoMoEBackendDecision(
                reason="prefill policy disabled or incompatible"
            )
    elif role == "kv_consumer":
        scheduler_config = vllm_config.scheduler_config
        if (
            not bool(additional_config.get("yoco_fast_decode_flashinfer_moe", True))
            or scheduler_config.max_num_seqs < 64
            or scheduler_config.max_num_batched_tokens < 8192
        ):
            return YocoMoEBackendDecision(
                reason="decode service below optimized capacity"
            )
        enable_decode_autotune = True
    else:
        return YocoMoEBackendDecision(reason="unsupported KV transfer role")
    if (
        _cfg_int(config, "hidden_size", "d_model") != _YOCO_L3_HIDDEN_SIZE
        or _cfg_int(config, "num_experts", "moe_expert_num") != 128
        or _cfg_int(config, "num_experts_per_tok", "moe_top_k", "top_k") != 8
        or (_cfg_int(config, "moe_intermediate_size", "moe_ffn_dim") != 3840)
        or (_cfg_int(config, "moe_latent_dim", default=0) != 1024)
        or (_swiglu_limit(config) <= 0)
    ):
        return YocoMoEBackendDecision(
            reason="model dimensions differ from validated L3"
        )
    if kernel_config.moe_backend not in ("auto", "triton", "flashinfer_cutlass"):
        return YocoMoEBackendDecision(reason="configured MoE backend retained")
    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 10:
        return YocoMoEBackendDecision(
            reason="device is outside the validated SM100 family"
        )
    if standalone and getattr(capability, "minor", 0) != 0:
        return YocoMoEBackendDecision(reason="standalone policy requires SM100")
    selected_backend: MoEBackend = "flashinfer_cutlass"
    if enable_decode_autotune:
        compilation_config = getattr(vllm_config, "compilation_config", None)
        max_capture_size = int(
            getattr(compilation_config, "max_cudagraph_capture_size", 0) or 0
        )
        default_trtllm_max_capture = _yoco_verified_trtllm_cache_max_capture(
            os.getenv("VLLM_YOCO_FLASHINFER_AUTOTUNE_CACHE")
        )
        trtllm_max_capture = int(
            additional_config.get(
                "yoco_fast_decode_trtllm_max_capture", default_trtllm_max_capture
            )
        )
        use_trtllm = (
            bool(additional_config.get("yoco_fast_decode_trtllm_moe", True))
            and 0 < max_capture_size <= trtllm_max_capture
        )
        if use_trtllm:
            from vllm.model_executor.layers.fused_moe.experts.yoco_trtllm_bf16 import (
                has_yoco_trtllm_bf16_clamp,
            )

            if has_yoco_trtllm_bf16_clamp():
                selected_backend = "yoco_flashinfer_trtllm"
        if selected_backend == "flashinfer_cutlass":
            from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe

            if not has_flashinfer_cutlass_fused_moe():
                return YocoMoEBackendDecision(reason="FlashInfer CUTLASS unavailable")
    else:
        from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe

        if not has_flashinfer_cutlass_fused_moe():
            return YocoMoEBackendDecision(reason="FlashInfer CUTLASS unavailable")
    if standalone:
        return YocoMoEBackendDecision(
            selected_backend,
            False if standalone else True if enable_decode_autotune else None,
            "validated L3 role and device policy",
        )
    return YocoMoEBackendDecision(
        selected_backend,
        False if standalone else True if enable_decode_autotune else None,
        "validated L3 role and device policy",
    )


def _select_yoco_fast_moe_backend(**kwargs) -> MoEBackend | None:
    """Compatibility entry for older experimental callers."""
    decision = resolve_yoco_fast_moe_backend(**kwargs)
    decision.apply(kwargs["vllm_config"])
    return decision.backend


YOCO_PACKED_MODULES_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "yoco_kv_proj": ["yoco_k_proj", "yoco_v_proj"],
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


def _get_yoco_execution_mode(vllm_config: VllmConfig) -> str:
    return get_yoco_execution_mode(vllm_config.additional_config)


def _select_yoco_online_fp8_moe_backend(
    quant_config: QuantizationConfig,
    prefix: str,
    tp_size: int,
) -> MoEBackend:
    """Resolve YOCO's automatic online FP8 policy per routed-expert layer.

    BF16 experts excluded by ignore rules cannot use the FP8 DeepGEMM
    backend. Explicit user backend choices never enter this function.
    """
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )
    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Static128BlockSym,
    )
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    if isinstance(quant_config, OnlineQuantizationConfig):
        spec = quant_config.args.moe
        if (
            spec is not None
            and spec.weight == kFp8Static128BlockSym
            and not should_ignore_layer(
                prefix,
                ignore=quant_config.ignored_layers,
                fused_mapping=quant_config.packed_modules_mapping,
            )
            and tp_size == 1
            and is_deep_gemm_supported()
        ):
            return "deep_gemm"
    return "triton"


def _yoco_runtime_sliding_window(training_window_left: int) -> int:
    """Translate llm-train's left-window count to vLLM's total window size.

    FlashAttention's ``(left, 0)`` includes ``left`` previous tokens and the
    current token. vLLM accepts the total token count and subtracts one before
    calling the backend, so training's 512 becomes 513 here.
    """
    if training_window_left <= 0:
        raise ValueError("YOCO self-attention requires a positive sliding window")
    return training_window_left + 1


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


def _swiglu_limit(config: PretrainedConfig) -> float:
    """SwiGLU clamp limit, matching training (``swiglu_limit``, default 10.0).

    Training clamps ``gate`` to ``max=limit`` and ``up`` to ``[-limit, limit]``
    before ``silu(gate) * up`` (see ``llm/arch/ffn.py`` and
    ``llm/arch/all2all_moe.py``).
    """
    return float(getattr(config, "swiglu_limit", 10.0))
