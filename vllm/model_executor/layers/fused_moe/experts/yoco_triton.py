# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO-specific Triton kernels and tuning used by the modular MoE backend."""

import functools
import json
from pathlib import Path
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_TRITON_CONFIG_KEYS = {
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
}


@triton.jit
def _yoco_swapped_clamped_swiglu_kernel(
    input_ptr,
    output_ptr,
    stride_input_m,
    stride_input_n,
    stride_output_m,
    stride_output_n,
    N,
    BLOCK_SIZE_N: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
):
    """Clamped SwiGLU for FlashInfer's [up, gate] W13 row order."""
    row_id = tl.program_id(axis=0).to(tl.int64)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    mask = offs_n < N
    up = tl.load(
        input_ptr + row_id * stride_input_m + offs_n * stride_input_n,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        input_ptr + row_id * stride_input_m + (offs_n + N) * stride_input_n,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.minimum(gate, SWIGLU_LIMIT)
    up = tl.minimum(tl.maximum(up, -SWIGLU_LIMIT), SWIGLU_LIMIT)
    result = gate * tl.sigmoid(gate) * up
    tl.store(
        output_ptr + row_id * stride_output_m + offs_n * stride_output_n,
        result,
        mask=mask,
    )


def yoco_swapped_clamped_swiglu(
    output: torch.Tensor,
    input: torch.Tensor,
    swiglu_limit: float,
) -> None:
    """Apply YOCO SwiGLU to W13 rows converted for FlashInfer CUTLASS."""
    assert input.ndim == output.ndim == 2
    assert input.shape[0] == output.shape[0]
    assert input.shape[1] == 2 * output.shape[1]
    assert input.dtype == output.dtype == torch.bfloat16
    assert input.is_contiguous() and output.is_contiguous()
    assert swiglu_limit > 0
    hidden_dim = output.shape[1]
    _yoco_swapped_clamped_swiglu_kernel[(input.shape[0],)](
        input,
        output,
        input.stride(0),
        input.stride(1),
        output.stride(0),
        output.stride(1),
        hidden_dim,
        BLOCK_SIZE_N=triton.next_power_of_2(hidden_dim),
        SWIGLU_LIMIT=float(swiglu_limit),
        num_warps=4,
    )


def _yoco_w2_config_file_name(
    num_experts: int, output_size: int, input_size: int, device_name: str
) -> str:
    return (
        f"w2_E={num_experts},N={output_size},K={input_size},"
        f"device_name={device_name.replace(' ', '_')}.json"
    )


def _yoco_w13_config_file_name(
    num_experts: int, intermediate_size: int, hidden_size: int, device_name: str
) -> str:
    return (
        f"w13_E={num_experts},N={intermediate_size},K={hidden_size},"
        f"device_name={device_name.replace(' ', '_')}.json"
    )


def _parse_yoco_configs(path: Path) -> dict[int, dict[str, int]]:
    with path.open() as config_file:
        raw: dict[str, Any] = json.load(config_file)
    raw.pop("triton_version", None)
    configs: dict[int, dict[str, int]] = {}
    for tokens, config in raw.items():
        if not isinstance(config, dict) or not _TRITON_CONFIG_KEYS.issubset(config):
            raise ValueError(f"Invalid YOCO expert config in {path}: {tokens}")
        configs[int(tokens)] = {key: int(config[key]) for key in _TRITON_CONFIG_KEYS}
    return configs


@functools.lru_cache
def _load_yoco_w13_configs(
    num_experts: int,
    intermediate_size: int,
    hidden_size: int,
    device_name: str,
) -> dict[int, dict[str, int]] | None:
    """Load a YOCO-only W13 map without affecting equal-shaped models."""
    file_name = _yoco_w13_config_file_name(
        num_experts, intermediate_size, hidden_size, device_name
    )
    paths: list[Path] = []
    if envs.VLLM_TUNED_CONFIG_FOLDER is not None:
        paths.append(Path(envs.VLLM_TUNED_CONFIG_FOLDER) / file_name)
    paths.append(Path(__file__).with_name("yoco_configs") / file_name)

    for path in paths:
        if not path.is_file():
            continue
        configs = _parse_yoco_configs(path)
        logger.info_once("Using YOCO W13 tuning from %s", path)
        return configs
    return None


@functools.lru_cache
def _load_yoco_w2_configs(
    num_experts: int, output_size: int, input_size: int, device_name: str
) -> dict[int, dict[str, int]] | None:
    """Load only YOCO's W2-specific tuning map."""
    file_name = _yoco_w2_config_file_name(
        num_experts, output_size, input_size, device_name
    )
    paths: list[Path] = []
    if envs.VLLM_TUNED_CONFIG_FOLDER is not None:
        paths.append(Path(envs.VLLM_TUNED_CONFIG_FOLDER) / file_name)
    paths.append(Path(__file__).with_name("yoco_configs") / file_name)

    for path in paths:
        if not path.is_file():
            continue
        configs = _parse_yoco_configs(path)
        logger.info_once("Using YOCO W2 tuning from %s", path)
        return configs
    return None


def select_yoco_w13_config(
    configs: dict[int, dict[str, int]], num_tokens: int
) -> dict[str, int] | None:
    """Select the nearest measured bucket without small-M extrapolation."""
    if num_tokens < min(configs):
        return None
    nearest = min(configs, key=lambda candidate: abs(candidate - num_tokens))
    return configs[nearest]


def try_get_yoco_w13_config(
    num_tokens: int,
    num_experts: int,
    intermediate_size: int,
    hidden_size: int,
) -> dict[str, int] | None:
    """Return private YOCO W13 tuning, or None for the common lookup."""
    if envs.VLLM_BATCH_INVARIANT:
        return None
    device_name = current_platform.get_device_name()
    configs = _load_yoco_w13_configs(
        num_experts, intermediate_size, hidden_size, device_name
    )
    if not configs:
        return None
    # Do not extrapolate a large-M map into decode/small-prefill shapes.
    # Those shapes keep the common vLLM config unless measured entries are
    # explicitly present in this private file.
    return select_yoco_w13_config(configs, num_tokens)


def select_yoco_w2_config(
    configs: dict[int, dict[str, int]],
    num_tokens: int,
    w13_config: dict[str, int],
) -> dict[str, int]:
    """Select W2 tuning while retaining W13's dispatch block size."""
    nearest = min(configs, key=lambda candidate: abs(candidate - num_tokens))
    config = configs[nearest]
    if config["BLOCK_SIZE_M"] != w13_config["BLOCK_SIZE_M"]:
        # A different M tile requires a second token-dispatch operation. The
        # dedicated tuner intentionally excludes that cost for now.
        return w13_config
    return config


def try_get_yoco_w2_config(
    num_tokens: int,
    num_experts: int,
    output_size: int,
    input_size: int,
    w13_config: dict[str, int],
) -> dict[str, int]:
    """Return a W2-only tuned config, or the shared config as fallback."""
    if envs.VLLM_BATCH_INVARIANT:
        return w13_config
    device_name = current_platform.get_device_name()
    configs = _load_yoco_w2_configs(num_experts, output_size, input_size, device_name)
    if not configs or num_tokens < min(configs):
        return w13_config
    return select_yoco_w2_config(configs, num_tokens, w13_config)


@triton.jit
def _yoco_weighted_swiglu_kernel(
    input_ptr,
    output_ptr,
    routing_weights_ptr,
    stride_input_m,
    stride_input_n,
    stride_output_m,
    stride_output_n,
    N,
    BLOCK_SIZE_N: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
):
    """Training-compatible weighted clamped SwiGLU."""
    row_id = tl.program_id(axis=0).to(tl.int64)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    mask = offs_n < N

    gate = tl.load(
        input_ptr + row_id * stride_input_m + offs_n * stride_input_n,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        input_ptr + row_id * stride_input_m + (offs_n + N) * stride_input_n,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.minimum(gate, SWIGLU_LIMIT)
    up = tl.minimum(tl.maximum(up, -SWIGLU_LIMIT), SWIGLU_LIMIT)
    routing_weight = tl.load(routing_weights_ptr + row_id).to(tl.float32)
    result = gate * tl.sigmoid(gate) * up * routing_weight

    tl.store(
        output_ptr + row_id * stride_output_m + offs_n * stride_output_n,
        result,
        mask=mask,
    )


def yoco_weighted_swiglu(
    output: torch.Tensor,
    input: torch.Tensor,
    routing_weights: torch.Tensor,
    swiglu_limit: float,
) -> None:
    """Match llm-train's BF16 routed activation and rounding boundary.

    Clamp, SiLU, gate/up multiplication, and FP32 routing-weight
    multiplication happen in one kernel. The result is rounded only once when
    stored to ``output`` before the expert W2 GEMM.
    """
    assert input.ndim == 2 and output.ndim == 2
    assert input.shape[0] == output.shape[0] == routing_weights.numel()
    assert input.shape[1] == 2 * output.shape[1]
    assert input.is_contiguous() and output.is_contiguous()
    assert routing_weights.is_contiguous()
    assert routing_weights.dtype == torch.float32
    assert swiglu_limit > 0

    hidden_dim = output.shape[1]
    _yoco_weighted_swiglu_kernel[(input.shape[0],)](
        input,
        output,
        routing_weights,
        input.stride(0),
        input.stride(1),
        output.stride(0),
        output.stride(1),
        hidden_dim,
        BLOCK_SIZE_N=triton.next_power_of_2(hidden_dim),
        SWIGLU_LIMIT=float(swiglu_limit),
        num_warps=4,
    )


@triton.jit
def _yoco_topk8_sum_kernel(
    input_ptr,
    output_ptr,
    stride_input_m,
    stride_input_route,
    stride_output_m,
    K,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Accumulate routes in their stored order using an FP32 accumulator."""
    row_id = tl.program_id(axis=0).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    mask = offs_k < K
    accumulator = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)
    row_ptr = input_ptr + row_id * stride_input_m + offs_k

    for route_idx in tl.static_range(8):
        route = tl.load(
            row_ptr + route_idx * stride_input_route,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += route

    tl.store(output_ptr + row_id * stride_output_m + offs_k, accumulator, mask=mask)


def yoco_topk8_sum(input: torch.Tensor, output: torch.Tensor) -> None:
    """Reduce YOCO's eight expert outputs in training accumulation order."""
    assert input.ndim == 3 and output.ndim == 2
    assert input.shape[0] == output.shape[0]
    assert input.shape[1] == 8
    assert input.shape[2] == output.shape[1]
    assert input.dtype == output.dtype == torch.bfloat16
    assert input.is_contiguous() and output.is_contiguous()

    num_tokens, _, hidden_size = input.shape
    _yoco_topk8_sum_kernel[(num_tokens,)](
        input,
        output,
        input.stride(0),
        input.stride(1),
        output.stride(0),
        hidden_size,
        BLOCK_SIZE_K=triton.next_power_of_2(hidden_size),
        num_warps=16 if num_tokens < 8192 else 8,
    )
