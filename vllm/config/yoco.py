# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO execution options shared by model construction and MoE backends."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from vllm.config import VllmConfig

YocoExecutionMode = Literal["align", "fast"]
YocoMegaMoEMode = Literal["off", "fp8_group32"]
YOCO_FP8_GROUP_EPS = 1e-4


def is_yoco_shared_kv_speculation(config: "VllmConfig") -> bool:
    spec = config.speculative_config
    draft = getattr(spec, "draft_model_config", None)
    return bool(
        config.model_config.hf_text_config.model_type == "yoco"
        and spec is not None
        and spec.method == "dspark"
        and draft is not None
        and getattr(draft.hf_config, "shared_kv_config", None)
    )


def yoco_fp8_group_quant_eps(config: "VllmConfig | None") -> float | None:
    model_config = config.model_config if config is not None else None
    hf_config = getattr(model_config, "hf_text_config", None)
    if getattr(hf_config, "model_type", None) == "yoco":
        return YOCO_FP8_GROUP_EPS
    return None


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


def get_yoco_mega_moe_mode(additional_config: object) -> YocoMegaMoEMode:
    mode = (
        additional_config.get("yoco_mega_moe", "off")
        if isinstance(additional_config, Mapping)
        else "off"
    )
    if mode not in ("off", "fp8_group32"):
        raise ValueError("yoco_mega_moe must be 'off' or explicitly 'fp8_group32'")
    return cast(YocoMegaMoEMode, mode)


@dataclass(frozen=True)
class YocoMoEDispatch:
    """Resolved, graph-stable routed-expert thresholds (local token rows)."""

    profile: str = "original"
    triton_max_tokens: int | None = None
    mega_min_tokens: int | None = None
    mega_prefill: bool = False

    def use_mega(self, num_tokens: int, *, decode_graph: bool) -> bool:
        return (
            self.mega_min_tokens is not None
            and num_tokens >= self.mega_min_tokens
            and (self.mega_prefill or decode_graph)
        )


def resolve_yoco_moe_dispatch(
    additional_config: object,
    *,
    gpu_name: str,
    parallel_sizes: tuple[int, int, int, int],
    profile_compatible: bool,
    speculative_tokens: int,
) -> YocoMoEDispatch:
    """Use measured profiles only; manual YAML thresholds override selection."""
    extra = additional_config if isinstance(additional_config, Mapping) else {}
    options = extra.get("yoco_moe_dispatch", {})
    if not isinstance(options, Mapping):
        raise ValueError("yoco_moe_dispatch must be a mapping")
    unknown = options.keys() - {
        "mode",
        "triton_max_tokens",
        "mega_min_tokens",
        "mega_prefill",
    }
    if unknown:
        raise ValueError(f"Unknown yoco_moe_dispatch options: {sorted(unknown)}")
    mode = options.get("mode", "auto")
    if mode not in ("auto", "manual"):
        raise ValueError("yoco_moe_dispatch.mode must be 'auto' or 'manual'")
    mega_enabled = get_yoco_mega_moe_mode(extra) != "off"
    if mode == "auto":
        if options.keys() - {"mode"}:
            raise ValueError("Use yoco_moe_dispatch.mode=manual to set thresholds")
        # Evidence and full matching contract: docs/yoco/performance/MEGA_MOE.md.
        if (
            mega_enabled
            and gpu_name == "NVIDIA B200"
            and parallel_sizes == (1, 1, 1, 1)
            and profile_compatible
            and speculative_tokens == 8
        ):
            return YocoMoEDispatch("b200-l3-fp8-tp1-k8", 16, 17, False)
        return YocoMoEDispatch("unprofiled-original")

    triton = options.get("triton_max_tokens")
    mega = options.get("mega_min_tokens")
    prefill = options.get("mega_prefill", False)
    if triton is not None and (type(triton) is not int or not 0 <= triton <= 64):
        raise ValueError("triton_max_tokens must be null or an integer in [0,64]")
    if mega is not None and (type(mega) is not int or mega < 1):
        raise ValueError("mega_min_tokens must be null or a positive integer")
    if type(prefill) is not bool:
        raise ValueError("mega_prefill must be a boolean")
    if mega is not None and not mega_enabled:
        raise ValueError("mega_min_tokens requires explicit yoco_mega_moe=fp8_group32")
    if mega is not None and triton is not None and mega <= triton:
        raise ValueError("mega_min_tokens must exceed triton_max_tokens")
    return YocoMoEDispatch("manual", triton, mega, prefill)


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
    mega_moe: bool = False
    fp8_group_quant_eps: float = field(default=YOCO_FP8_GROUP_EPS, init=False)

    def __post_init__(self) -> None:
        if self.align_moe_sum and self.fast_moe_sum:
            raise ValueError("YOCO Align and Fast MoE reductions are exclusive")
        if self.triton_fallback_max_tokens < 0:
            raise ValueError("YOCO fallback token limit cannot be negative")
        if self.mega_moe and (not self.enabled or self.align_weighted_swiglu):
            raise ValueError("YOCO MegaMoE requires Fast execution")

    @classmethod
    def for_mode(
        cls, mode: YocoExecutionMode, *, mega_moe: bool = False
    ) -> "YocoMoEPolicy":
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
            mega_moe=mega_moe,
        )
