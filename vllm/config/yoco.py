# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO execution options shared by model construction and MoE backends."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from vllm.config import VllmConfig

YocoExecutionMode = Literal["align", "fast"]
YOCO_FP8_GROUP_EPS = 1e-4


def yoco_fp8_group_quant_eps(config: "VllmConfig | None") -> float | None:
    model_config = config.model_config if config is not None else None
    hf_config = getattr(model_config, "hf_text_config", None)
    if getattr(hf_config, "model_type", None) == "yoco":
        return YOCO_FP8_GROUP_EPS
    return None


def yoco_v1_runner_features(config: "VllmConfig") -> list[str]:
    """Features whose YOCO integration uses upstream's retained V1 runner."""
    import vllm.envs as envs

    hf_config = getattr(config.model_config, "hf_text_config", None)
    if getattr(hf_config, "model_type", None) != "yoco":
        return []
    features = []
    if config.kv_transfer_config is not None:
        features.append("YOCO disaggregated KV transfer")
    if (
        config.cache_config.kv_sharing_fast_prefill
        and config.parallel_config.data_parallel_size > 1
    ):
        features.append("YOCO fast prefill across data-parallel ranks")
    if (
        envs.VLLM_YOCO_BF16_SAMPLING
        and get_yoco_execution_mode(config.additional_config) == "fast"
    ):
        features.append("YOCO BF16 sampling")
    return features


def get_yoco_execution_mode(additional_config: object) -> YocoExecutionMode:
    mode = (
        additional_config.get("yoco_execution_mode", "fast")
        if isinstance(additional_config, Mapping)
        else "fast"
    )
    if mode not in ("align", "fast"):
        raise ValueError(
            f"YOCO execution mode must be 'align' or 'fast', but got {mode!r}"
        )
    return cast(YocoExecutionMode, mode)


@dataclass(frozen=True)
class YocoMoEPolicy:
    """Numerical and tuning choices propagated with the MoE configuration."""

    enabled: bool = False
    align_weighted_swiglu: bool = False
    direct_fp8_activation: bool = False
    align_deep_gemm_w2: bool = False
    separate_w2_config: bool = False
    fast_w13_config: bool = False
    triton_fallback_max_tokens: int = 0
    fast_decode_cutlass: bool = False
    align_moe_sum: bool = False
    fast_moe_sum: bool = False
    swapped_w13: bool = False
    fp8_decode_aligned: bool = False
    fp8_group_quant_eps: float = field(default=YOCO_FP8_GROUP_EPS, init=False)

    def __post_init__(self) -> None:
        if self.align_moe_sum and self.fast_moe_sum:
            raise ValueError("YOCO Align and Fast MoE reductions are exclusive")
        if self.triton_fallback_max_tokens < 0:
            raise ValueError("YOCO fallback token limit cannot be negative")

    @classmethod
    def for_mode(cls, mode: YocoExecutionMode) -> "YocoMoEPolicy":
        if mode not in ("align", "fast"):
            raise ValueError(f"Unsupported YOCO execution mode: {mode!r}")
        fast = mode == "fast"
        return cls(
            enabled=True,
            align_weighted_swiglu=not fast,
            direct_fp8_activation=fast,
            separate_w2_config=fast,
            fast_w13_config=fast,
            triton_fallback_max_tokens=int(fast),
            align_moe_sum=not fast,
            fast_moe_sum=fast,
        )
