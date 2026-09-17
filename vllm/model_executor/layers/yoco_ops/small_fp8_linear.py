# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M1-only shared/latent dispatch, retaining native quantization and fallback."""

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
    DeepGemmFp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.yoco_ops.small_fp8 import (
    SHAPES,
    _small_fp8_direct_kernel,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.deep_gemm import fp8_gemm_nt
from vllm.utils.torch_utils import direct_register_custom_op


def _can_use_m1(a, weight, a_scale, weight_scale):
    if a.ndim != 2 or weight.ndim != 2 or a.shape[0] != 1:
        return False
    n, k = weight.shape
    packs = triton.cdiv(k, 512)
    return (
        (n, k) in SHAPES
        and a.shape[1] == k
        and a.dtype == weight.dtype == torch.float8_e4m3fn
        and a.is_contiguous()
        and weight.is_contiguous()
        and a_scale.dtype == weight_scale.dtype == torch.int32
        and a_scale.shape == (1, packs)
        and weight_scale.shape == (n, packs)
        and all(
            t.is_cuda and t.device == a.device
            for t in (a, weight, a_scale, weight_scale)
        )
    )


def _yoco_m1_fp8_gemm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> None:
    # Keep the shape dispatch inside the opaque op: prefill compilation can
    # remain dynamic, and CUDA Graph capture selects the actual padded M.
    if _can_use_m1(a, weight, a_scale, weight_scale):
        n, k = weight.shape
        _small_fp8_direct_kernel[(n,)](
            a,
            weight,
            a_scale,
            weight_scale,
            output,
            1,
            n,
            k,
            *a_scale.stride(),
            *weight_scale.stride(),
            BLOCK_N=1,
            BLOCK_K=triton.next_power_of_2(k),
            num_warps=4,
            num_stages=1,
        )
    else:
        fp8_gemm_nt(
            (a, a_scale), (weight, weight_scale), output, is_deep_gemm_e8m0_used=True
        )


direct_register_custom_op(
    "yoco_m1_fp8_gemm", _yoco_m1_fp8_gemm, mutates_args=["output"]
)


class YocoM1Fp8LinearKernel(DeepGemmFp8BlockScaledMMKernel):
    """Native DeepGEMM linear semantics with a measured M1 GEMM replacement."""

    def apply_block_scaled_mm(self, A, B, As, Bs):
        output = torch.empty(
            (A.shape[0], B.shape[0]), device=A.device, dtype=self.config.out_dtype
        )
        torch.ops.vllm.yoco_m1_fp8_gemm(A, As, B, Bs, output)
        return output


def configure_yoco_m1_fp8_linear(layer, execution_mode: str) -> bool:
    """Select during YOCO construction, before weight loading and compilation."""
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PerBlockOnlineLinearMethod,
    )

    method = getattr(layer, "quant_method", None)
    if (
        execution_mode != "fast"
        or not envs.VLLM_YOCO_FP8_SMALL_M
        or envs.VLLM_BATCH_INVARIANT
        or not isinstance(method, Fp8PerBlockOnlineLinearMethod)
        or method.input_dtype != torch.bfloat16
        or method.out_dtype != torch.bfloat16
        or getattr(layer, "bias", None) is not None
        or not current_platform.is_device_capability(100)
    ):
        return False
    parallel = get_current_vllm_config().parallel_config
    if (
        parallel.tensor_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.enable_expert_parallel
        or parallel.enable_eplb
    ):
        return False
    original = method.fp8_linear
    if (
        type(original) is not DeepGemmFp8BlockScaledMMKernel
        or not original.use_deep_gemm_e8m0
        or tuple(original.config.weight_shape) not in SHAPES
    ):
        return False
    replacement = YocoM1Fp8LinearKernel(original.config)
    replacement.quant_fp8 = original.quant_fp8
    replacement.use_triton = original.use_triton
    method.fp8_linear = replacement
    return True
