# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared, opt-in launch tuning for YOCO Align's two BF16 expert GEMMs.

K tiles, split-K, activation rounding and route reduction are not tuning axes.
Matching these constraints does not prove bitwise equality: each profile still
needs byte comparisons on its recorded device and software environment.
"""

import functools
import json
from pathlib import Path

import torch

from vllm import envs
from vllm.platforms import current_platform
from vllm.triton_utils import triton

_BASE = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
    "SPLIT_K": 1,
}
_CHOICES = {
    "BLOCK_SIZE_M": (16, 32, 64, 128),
    "BLOCK_SIZE_N": (64, 128, 256),
    "BLOCK_SIZE_K": (32,),
    "GROUP_SIZE_M": (1, 8),
    "SPLIT_K": (1,),
    "num_warps": (4, 8),
    "num_stages": (2, 3, 4),
}


def _validate_config(config: dict[str, int]) -> None:
    if not isinstance(config, dict) or config.keys() != _CHOICES.keys():
        raise ValueError("Align MoE profile requires exactly the supported launch keys")
    for key, allowed in _CHOICES.items():
        if type(config[key]) is not int or config[key] not in allowed:
            raise ValueError(
                f"Unsupported Align MoE {key}={config[key]!r}; expected {allowed}"
            )


@functools.lru_cache
def load_yoco_align_moe_profile(path: str) -> dict:
    """Validate once per process; use a new path/process after editing a profile."""
    profile = json.loads(Path(path).read_text())
    if (
        not isinstance(profile, dict)
        or type(profile.get("schema_version")) is not int
        or profile["schema_version"] != 1
    ):
        raise ValueError("Align MoE profile requires schema_version=1")
    if (
        profile.get("w13_shape") != [128, 7680, 1024]
        or profile.get("w2_shape") != [128, 1024, 3840]
        or profile.get("top_k") != 8
    ):
        raise ValueError("Align MoE tuning currently requires YOCO L3 TP1 shapes")
    metadata = profile.get("environment")
    keys = ("device_name", "torch_version", "triton_version", "cuda_version")
    if not isinstance(metadata, dict) or any(
        not isinstance(metadata.get(key), str) or not metadata[key] for key in keys
    ):
        raise ValueError("Align MoE profile requires device and dependency versions")
    configs = profile.get("configs")
    if not isinstance(configs, dict) or not configs:
        raise ValueError("Align MoE profile requires measured token-row configurations")
    for rows, pair in configs.items():
        if not rows.isdecimal() or int(rows) < 1 or str(int(rows)) != rows:
            raise ValueError(f"Invalid Align MoE token-row key: {rows!r}")
        if not isinstance(pair, dict) or pair.keys() != {"w13", "w2"}:
            raise ValueError("Each Align MoE entry must contain w13 and w2")
        _validate_config(pair["w13"])
        _validate_config(pair["w2"])
        if pair["w13"]["BLOCK_SIZE_M"] != pair["w2"]["BLOCK_SIZE_M"]:
            raise ValueError("Align W13/W2 must share the M tile for expert assignment")
    return profile


@functools.lru_cache
def _verified_profile(path: str, device_index: int) -> dict:
    # Device queries and dependency checks must not run once per MoE layer.
    profile = load_yoco_align_moe_profile(path)
    actual = {
        "device_name": current_platform.get_device_name(device_index),
        "torch_version": str(torch.__version__),
        "triton_version": triton.__version__,
        "cuda_version": torch.version.cuda,
    }
    if any(profile["environment"][key] != value for key, value in actual.items()):
        raise ValueError(
            "Align MoE profile environment mismatch: "
            f"recorded={profile['environment']}, "
            f"actual={actual}. Use a profile validated in this environment."
        )
    return profile


def select_yoco_align_moe_configs(
    profile: dict,
    num_tokens: int,
    base_config: dict[str, int],
) -> tuple[dict[str, int], dict[str, int]] | None:
    """Only override the canonical invariant launch at an exact recorded size."""
    # Explicit caller overrides take precedence over this experimental profile.
    canonical = {**_BASE, "num_warps": 4, "num_stages": 3}
    if any(base_config.get(key) != value for key, value in _BASE.items()) or any(
        key not in canonical or value != canonical[key]
        for key, value in base_config.items()
    ):
        return None
    pair = profile["configs"].get(str(num_tokens))
    if pair is None:
        return None
    # Launch helpers may mutate dictionaries; never expose the cached entries.
    return pair["w13"].copy(), pair["w2"].copy()


def get_yoco_align_moe_configs(
    num_tokens: int,
    w13_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
    top_k: int,
    base_config: dict[str, int],
    *,
    device_index: int = 0,
) -> tuple[dict[str, int], dict[str, int]] | None:
    """Shared inference/training selection, gated by shape and environment.

    Callers additionally require unquantized BF16 inputs/weights, the weighted
    Align activation, and no expert parallelism, bias or LoRA.
    """
    path = envs.VLLM_YOCO_ALIGN_MOE_CONFIG
    if not envs.VLLM_BATCH_INVARIANT or not path:
        return None
    if (
        tuple(w13_shape) != (128, 7680, 1024)
        or tuple(w2_shape) != (128, 1024, 3840)
        or top_k != 8
        or not current_platform.is_cuda()
    ):
        return None
    profile = _verified_profile(path, device_index)
    return select_yoco_align_moe_configs(profile, num_tokens, base_config)
