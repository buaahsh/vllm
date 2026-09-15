# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO latent RMSNorm producing DeepGEMM's packed FP8 activation format."""

import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _rms_norm_fp8_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    scale_ptr,
    input_stride,
    scale_stride,
    num_rows,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, 1024)
    # Routed-expert outputs are BF16; keep the existing FP32 statistics.
    x = tl.load(x_ptr + row * input_stride + cols).to(tl.float32)
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    variance = tl.sum(x * x, axis=0) * (1.0 / 1024)
    inv_rms = tldevice.rsqrt(variance + EPS)
    normalized = x * inv_rms * weight
    # Preserve the existing Norm -> BF16 -> quantizer boundary in registers.
    normalized = normalized.to(tl.bfloat16).to(tl.float32)
    groups = normalized.reshape(8, 128)
    # CUDA's fmaxf ignores NaNs (including inactive graph rows). Match its
    # quantizer instead of letting a NaN amax produce invalid scale bytes.
    absolute = tl.abs(groups)
    absolute = tl.where(absolute == absolute, absolute, 0.0)
    amax = tl.maximum(tl.max(absolute, axis=1), 1e-4)
    scale_raw = tl.maximum(amax / 448.0, 1e-10)
    bits = scale_raw.to(tl.uint32, bitcast=True)
    biased = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
    scale = (biased << 23).to(tl.float32, bitcast=True)
    scaled = groups / scale[:, None]
    scaled = tl.where(scaled == scaled, scaled, -448.0)
    quantized = tl.clamp(scaled, -448.0, 448.0)
    tl.store(output_ptr + row * 1024 + cols, quantized.reshape(1024))
    packed = tl.sum(biased.reshape(2, 4) << (tl.arange(0, 4)[None, :] * 8), axis=1)
    tl.store(scale_ptr + row + tl.arange(0, 2) * scale_stride, packed)
    if row == 0:
        # empty_strided contains these holes between the two packed columns.
        # The CUDA quantizer initializes them too. Stay inside its storage.
        padding = num_rows + tl.arange(0, 4)
        tl.store(scale_ptr + padding, 0, mask=padding < scale_stride)


def _rms_norm_fp8_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = x.shape[0]
    return (
        torch.empty((rows, 1024), device=x.device, dtype=torch.float8_e4m3fn),
        torch.empty_strided(
            (rows, 2),
            (1, triton.cdiv(rows, 4) * 4),
            device=x.device,
            dtype=torch.int32,
        ),
    )


def _rms_norm_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 2 and x.shape[1] == 1024 and x.stride(1) == 1
    assert x.dtype == torch.bfloat16
    assert weight.shape == (1024,) and weight.dtype == torch.bfloat16
    assert weight.is_contiguous()
    output, scales = _rms_norm_fp8_fake(x, weight, eps)
    if x.shape[0]:
        _rms_norm_fp8_kernel[(x.shape[0],)](
            x,
            weight,
            output,
            scales,
            x.stride(0),
            scales.stride(1),
            x.shape[0],
            eps,
            num_warps=4,
        )
    return output, scales


direct_register_custom_op(
    "yoco_latent_rms_norm_fp8",
    _rms_norm_fp8,
    fake_impl=_rms_norm_fp8_fake,
)
