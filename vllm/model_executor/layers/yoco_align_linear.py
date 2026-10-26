# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental shared BF16 forward provider for YOCO training and inference.

The legacy K=64 Align output is the numerical contract. The native operator
owns its BLAS handle and plans, leaving ATen's training backward untouched.
"""

from functools import lru_cache
from importlib import import_module

import torch

from vllm import envs
from vllm.utils.torch_utils import direct_register_custom_op

ALIGN_GEMM_CONTRACT = "yoco-bf16-k64-v1"
_FUSED_SHAPES = frozenset({(10240, 3072), (2560, 3072)})
# Development admission list; full-model acceptance is recorded separately.
_SHAPES = frozenset(
    {
        (8192, 3072),
        (1024, 3072),
        (3072, 4096),
        (64, 3072),
        (1280, 3072),
        (3072, 1280),
        (3072, 1024),
    }
    | _FUSED_SHAPES
)


# The backend and loaded library are fixed for the process. The environment
# selection is included in vLLM and NNScaler compilation cache identities.
@lru_cache(maxsize=1)
@torch.compiler.assume_constant_result
def align_gemm_backend() -> str:
    backend = envs.VLLM_YOCO_ALIGN_GEMM_BACKEND
    if backend not in ("legacy", "cublaslt_v1"):
        raise ValueError(f"Unknown YOCO Align GEMM backend: {backend!r}")
    if backend == "cublaslt_v1":
        if not hasattr(torch.ops._C, "yoco_align_gemm_version"):
            try:
                import_module("vllm._yoco_align_C")
            except ImportError as exc:
                raise RuntimeError(
                    "Rebuild vLLM with the YOCO Align cuBLASLt extension"
                ) from exc
        version = torch.ops._C.yoco_align_gemm_version()
        if version != 130101:
            raise RuntimeError(
                "YOCO Align cuBLASLt v1 requires validated version 130101, "
                f"got {version}"
            )
    return backend


def align_gemm_identity() -> dict[str, str]:
    """Record shared forward selection in the training compilation cache."""
    return {
        "backend": align_gemm_backend(),
        "contract": ALIGN_GEMM_CONTRACT,
        "policy": "dense-and-head-v5",
        "fuse_projections": str(_projection_fusion_enabled()),
    }


@lru_cache(maxsize=1)
@torch.compiler.assume_constant_result
def _projection_fusion_enabled() -> bool:
    enabled = envs.VLLM_YOCO_ALIGN_FUSE_PROJECTIONS
    if enabled and align_gemm_backend() != "cublaslt_v1":
        raise ValueError("YOCO Align projection fusion requires cublaslt_v1")
    return enabled


def use_align_projection_fusion(m: int, n: int, k: int) -> bool:
    return (
        _projection_fusion_enabled()
        and (n, k) in _FUSED_SHAPES
        and cublaslt_workspace(m, n, k) is not None
    )


def cublaslt_workspace(m: int, n: int, k: int) -> int | None:
    # Decode and the trainer's default 1024-row cross-entropy chunks. Larger
    # full-vocabulary teacher-forcing tensors retain the legacy implementation.
    if (n, k) == (154880, 3072):
        return 0 if 0 < m <= 2048 else None
    if not 0 < m <= 32768 or (n, k) not in _SHAPES:
        return None
    # The wider search picks a slow output-projection plan for small batches.
    # M=1 benefits from that search; M=2..32 use the validated zero-budget plans.
    if (n, k) == (3072, 4096) and 1 < m <= 32:
        return 0
    # A wider search budget changes tiling even when the chosen plan needs no
    # scratch. Narrow N uses the zero-workspace search to avoid a regression.
    # Merged QKV's wider search can select algorithm 13, which changes bytes.
    return 0 if n in (64, 10240) else 32 * 1024 * 1024


@lru_cache
def _check_device(device_index: int) -> None:
    if torch.cuda.get_device_capability(device_index) != (10, 0):
        raise RuntimeError("YOCO Align cuBLASLt v1 currently requires SM100")


def _cublaslt_forward(
    x: torch.Tensor, weight: torch.Tensor, workspace_bytes: int
) -> torch.Tensor:
    _check_device(x.device.index)
    # Unaligned operands can select plans with different BF16 rounding. Check
    # actual addresses inside the opaque operator so this also covers DLPack
    # storage and leaves fake-tensor tracing independent of storage addresses.
    if (
        x.data_ptr() % 16
        or weight.data_ptr() % 16
        or x.stride(0) % 8
        or weight.stride(0) % 8
    ):
        from .yoco_ops.projection import _yoco_align_linear_legacy

        return _yoco_align_linear_legacy(x, weight)
    return torch.ops._C.yoco_align_gemm(x, weight, workspace_bytes)


def _cublaslt_fake(
    x: torch.Tensor, weight: torch.Tensor, workspace_bytes: int
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


direct_register_custom_op(
    op_name="yoco_align_cublaslt",
    op_func=_cublaslt_forward,
    fake_impl=_cublaslt_fake,
)
