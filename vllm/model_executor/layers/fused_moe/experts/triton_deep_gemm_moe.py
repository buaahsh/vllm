# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
    DeepGemmExperts,
    _valid_deep_gemm,
    _valid_deep_gemm_shape,
)
from vllm.model_executor.layers.fused_moe.experts.fallback import FallbackExperts
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    is_deep_gemm_e8m0_used,
)


class TritonOrDeepGemmExperts(FallbackExperts):
    """DeepGemm with fallback to Triton for low latency shapes."""

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(
            experts=DeepGemmExperts(moe_config, quant_config),
            fallback_experts=TritonExperts(moe_config, quant_config),
        )
        self._yoco_fp8_decode_limit = 0
        self._yoco_fp8_scale_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    @torch.no_grad()
    def configure_yoco_fp8_decode(
        self,
        max_tokens: int = 16,
        cached_scales: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> bool:
        """Cache compatible small-batch scales during online weight loading.

        DeepGEMM broadcasts each weight block's scale across 128 N rows and
        packs four K exponents per INT32. Triton expects the original block
        scales as FP32. Decode these once, after the final weight conversion;
        no weights or activation quantization rules change.
        """
        self._yoco_fp8_decode_limit = 0
        if not 0 <= max_tokens <= 64:
            raise ValueError("YOCO FP8 decode threshold must be between 0 and 64")
        parallel = self.moe_config.moe_parallel_config
        s1, s2 = self.w1_scale, self.w2_scale
        if (
            max_tokens == 0
            or not is_deep_gemm_e8m0_used()
            or not current_platform.is_device_capability(100)
            or parallel.tp_size != 1
            or parallel.dp_size != 1
            or parallel.ep_size != 1
            or parallel.enable_eplb
            or self.moe_config.in_dtype != torch.bfloat16
            or self.block_shape != [128, 128]
            or not isinstance(s1, torch.Tensor)
            or not isinstance(s2, torch.Tensor)
            or s1.dtype != torch.int32
            or s2.dtype != torch.int32
            or not s1.is_cuda
            or not s2.is_cuda
            or s1.shape != (128, 7680, 2)
            or s2.shape != (128, 1024, 8)
        ):
            return False

        def unpack(scale: torch.Tensor, groups: int) -> torch.Tensor:
            packed = scale[:, ::128, :].contiguous().view(torch.uint8)
            bits = packed[..., :groups].to(torch.int32) << 23
            return bits.view(torch.float32).contiguous()

        decoded = unpack(s1, 8), unpack(s2, 30)
        cache = (
            cached_scales if cached_scales is not None else self._yoco_fp8_scale_cache
        )
        from vllm.model_executor.layers.yoco_fast import refresh_yoco_weight_cache

        cache = (
            refresh_yoco_weight_cache(None if cache is None else cache[0], decoded[0]),
            refresh_yoco_weight_cache(None if cache is None else cache[1], decoded[1]),
        )
        self._yoco_fp8_scale_cache = cache
        quant = replace(
            self.quant_config,
            _w1=replace(self.quant_config._w1, scale=cache[0]),
            _w2=replace(self.quant_config._w2, scale=cache[1]),
        )
        self.fallback_experts = TritonExperts(self.moe_config, quant)
        self.fallback_experts.yoco_fp8_decode_aligned = True
        self._yoco_fp8_decode_limit = max_tokens
        return True

    @staticmethod
    def get_clses() -> tuple[
        type[mk.FusedMoEExpertsModular],
        type[mk.FusedMoEExpertsModular],
    ]:
        return (DeepGemmExperts, TritonExperts)

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Note: the deep gemm workspaces are strictly larger than the triton
        # workspaces so we can be pessimistic here and allocate for DeepGemm
        # even if we fall back to triton later, e.g. if expert maps are set.
        if is_deep_gemm_e8m0_used() or _valid_deep_gemm_shape(M, N, K):
            return self.experts.workspace_shapes(
                M,
                N,
                K,
                topk,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
                activation,
            )
        else:
            return self.fallback_experts.workspace_shapes(
                M,
                N,
                K,
                topk,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
                activation,
            )

    def _select_experts_impl(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
    ) -> mk.FusedMoEExpertsModular:
        if (
            0 < hidden_states.shape[0] <= self._yoco_fp8_decode_limit
            and hidden_states.dtype == torch.float8_e4m3fn
            and w1.dtype == w2.dtype == torch.float8_e4m3fn
            and w1.shape == (128, 7680, 1024)
            and w2.shape == (128, 1024, 3840)
        ):
            return self.fallback_experts
        if is_deep_gemm_e8m0_used() or _valid_deep_gemm(hidden_states, w1, w2):
            return self.experts
        else:
            return self.fallback_experts
