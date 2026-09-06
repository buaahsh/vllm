# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The online reduction follows llm-train/kernel/cross_entropy.py
# (Copyright (c) 2023, Tri Dao).
"""Shared YOCO Align log-probability and cross-entropy forward arithmetic.

One FP32 online reduction per row is used by full-vocabulary log-probs,
selected-token log-probs and training CE. Tile shape depends only on vocabulary
size. Enabling this policy is process-scoped, like VLLM_BATCH_INVARIANT.
"""

import torch

from vllm.triton_utils import tl, triton

_enabled = False
_training_library = None


def is_enabled() -> bool:
    return _enabled


def enable(*, register_aten: bool = False) -> None:
    """Select the policy; training installs only the log-softmax CUDA override.

    ATen's existing Autograd dispatch remains in place and supplies
    LogSoftmaxBackward. The other inference-only overrides are not installed
    by the training adapter.
    """
    global _enabled, _training_library
    _enabled = True
    if register_aten and _training_library is None:
        from vllm.model_executor.layers import batch_invariant

        if not batch_invariant._batch_invariant_MODE:
            _training_library = torch.library.Library("aten", "IMPL")
            _training_library.impl(
                "aten::_log_softmax",
                batch_invariant._log_softmax_batch_invariant,
                "CUDA",
            )


def has_training_override() -> bool:
    return _training_library is not None


@triton.jit
def _statistics_kernel(
    x,
    lse,
    labels,
    loss,
    zloss,
    stride,
    n_cols,
    ignore_index,
    BLOCK: tl.constexpr,
    WRITE_CE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    maximum = -float("inf")
    total = 0.0
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        values = tl.load(
            x + row * stride + cols, cols < n_cols, other=-float("inf")
        ).to(tl.float32)
        updated = tl.maximum(maximum, tl.max(values))
        # Top-k/top-p may mask every value in an early tile. Avoid inf-inf
        # poisoning the online accumulator before the first finite tile.
        scale = tl.exp(
            tl.where(maximum == -float("inf"), -float("inf"), maximum - updated)
        )
        shifted = tl.where(updated == -float("inf"), -float("inf"), values - updated)
        total = scale * total + tl.sum(tl.exp(shifted))
        maximum = updated
    normalizer = tl.log(total) + maximum
    tl.store(lse + row, normalizer)
    if WRITE_CE:
        label = tl.load(labels + row)
        valid = label != ignore_index
        in_range = (label >= 0) & (label < n_cols)
        target = tl.load(x + row * stride + label, valid & in_range, other=0).to(
            tl.float32
        )
        nll = tl.where(in_range, normalizer - target, float("nan"))
        tl.store(loss + row, tl.where(valid, nll, 0.0))
        tl.store(zloss + row, tl.where(valid, normalizer * normalizer, 0.0))


@triton.jit
def _negative_nll(normalizer, value):
    result = -(normalizer - value)
    # Preserve CE = -logprob byte-for-byte when the CE rounds to positive zero.
    bits = result.to(tl.int32, bitcast=True)
    bits = tl.where(result == 0, bits | -2147483648, bits)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _logprobs_kernel(x, lse, output, stride, n_cols, numel, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // n_cols
    col = offsets % n_cols
    value = tl.load(x + row * stride + col, offsets < numel, other=0).to(tl.float32)
    normalizer = tl.load(lse + row, offsets < numel, other=0)
    tl.store(output + offsets, _negative_nll(normalizer, value), offsets < numel)


@triton.jit
def _gather_kernel(
    x, lse, ids, output, stride, n_cols, width, numel, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // width
    col = tl.load(ids + offsets, offsets < numel, other=0)
    valid = (offsets < numel) & (col >= 0) & (col < n_cols)
    value = tl.load(x + row * stride + col, valid, other=0).to(tl.float32)
    normalizer = tl.load(lse + row, offsets < numel, other=0)
    result = tl.where(valid, _negative_nll(normalizer, value), float("nan"))
    tl.store(output + offsets, result, offsets < numel)


def _matrix(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("YOCO Align probabilities require CUDA BF16/FP16/FP32 logits")
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("Logits must have a nonempty vocabulary dimension")
    return x.reshape(-1, x.shape[-1]).contiguous()


def _statistics(x, labels=None, ignore_index=-100):
    # FP16/BF16 loads can induce a different Triton reduction layout. Converting
    # here makes training FP32 logits and inference BF16 projections share bits.
    x = x.float().contiguous()
    m, n = x.shape
    lse = torch.empty(m, device=x.device, dtype=torch.float32)
    loss = torch.empty_like(lse) if labels is not None else None
    zloss = torch.empty_like(lse) if labels is not None else None
    if m:
        block = min(triton.next_power_of_2(n), 16384)
        warps = 4 if block < 2048 else 8 if block < 8192 else 16
        _statistics_kernel[(m,)](
            x,
            lse,
            labels,
            loss,
            zloss,
            x.stride(0),
            n,
            ignore_index,
            BLOCK=block,
            WRITE_CE=labels is not None,
            num_warps=warps,
        )
    return loss, lse, zloss


def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Inference forward returning FP32; differentiable through the ATen API."""
    if x.ndim == 0 or not -x.ndim <= dim < x.ndim:
        raise ValueError("Invalid log-softmax dimension")
    dim = dim % x.ndim
    moved = x.movedim(dim, -1)
    matrix = _matrix(moved)
    _, lse, _ = _statistics(matrix)
    output = torch.empty_like(matrix, dtype=torch.float32)
    if matrix.numel():
        _logprobs_kernel[(triton.cdiv(matrix.numel(), 1024),)](
            matrix,
            lse,
            output,
            matrix.stride(0),
            matrix.shape[1],
            matrix.numel(),
            BLOCK=1024,
        )
    return output.view(moved.shape).movedim(-1, dim).contiguous()


def token_logprobs(x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Selected-token values identical to gathering from full log_softmax."""
    matrix = _matrix(x)
    if x.ndim != 2 or ids.ndim not in (1, 2) or ids.shape[0] != x.shape[0]:
        raise ValueError(
            "Expected [rows,vocab] logits and [rows] or [rows,k] token IDs"
        )
    if ids.device != x.device or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Token IDs must be int32/int64 on the logits device")
    width = 1 if ids.ndim == 1 else ids.shape[1]
    selected = ids.reshape(x.shape[0], width).long().contiguous()
    _, lse, _ = _statistics(matrix)
    output = torch.empty_like(selected, dtype=torch.float32)
    if selected.numel():
        _gather_kernel[(triton.cdiv(selected.numel(), 1024),)](
            matrix,
            lse,
            selected,
            output,
            matrix.stride(0),
            matrix.shape[1],
            selected.shape[1],
            selected.numel(),
            BLOCK=1024,
        )
    return output.view(ids.shape)


def cross_entropy_forward(
    x: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
):
    """Fused CE/LSE/z-loss outputs, without a full-vocabulary log-prob tensor.

    The caller supplies autograd using the saved logits and this exact LSE.
    Valid labels or ignore_index are required. z-loss is the raw squared LSE.
    """
    matrix = _matrix(x)
    if x.ndim != 2 or labels.shape != (x.shape[0],):
        raise ValueError("Expected [rows,vocab] logits and [rows] labels")
    if labels.device != x.device or labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("Labels must be int32/int64 on the logits device")
    labels = labels.contiguous()
    if labels.dtype == torch.int64 and labels.data_ptr() % 16:
        labels = torch.nn.functional.pad(labels, (0, 1))[:-1]
    return _statistics(matrix, labels, ignore_index)
