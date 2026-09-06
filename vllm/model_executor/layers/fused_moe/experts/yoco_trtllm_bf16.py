# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO-private modular FlashInfer TRTLLM BF16 experts."""

from __future__ import annotations

import functools
import inspect

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe


@functools.cache
def has_yoco_trtllm_bf16_clamp() -> bool:
    """Whether FlashInfer exposes the BF16 clamp and output arguments."""
    if not has_flashinfer_trtllm_fused_moe():
        return False
    try:
        from flashinfer.fused_moe import trtllm_bf16_routed_moe

        parameters = inspect.signature(trtllm_bf16_routed_moe).parameters
    except (ImportError, TypeError, ValueError):
        return False
    return {
        "gemm1_alpha",
        "gemm1_beta",
        "gemm1_clamp_limit",
        "output",
    }.issubset(parameters)


class YocoTrtLlmBf16Experts(mk.FusedMoEExpertsModular):
    """Pre-routed TRTLLM-Gen experts with YOCO's clamped SwiGLU."""

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ) -> None:
        super().__init__(moe_config, quant_config)
        self.num_experts = moe_config.num_local_experts
        self.intermediate_size = moe_config.intermediate_size_per_partition
        self.max_capture_size = max(
            get_current_vllm_config().compilation_config.max_cudagraph_capture_size
            or 0,
            1,
        )
        self._swiglu_limit: float | None = None
        self._swiglu_alpha: torch.Tensor | None = None
        self._swiglu_beta: torch.Tensor | None = None
        self._swiglu_limit_tensor: torch.Tensor | None = None

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
            and has_yoco_trtllm_bf16_clamp()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            parallel_config.tp_size == 1
            and parallel_config.pcp_size == 1
            and parallel_config.dp_size == 1
            and parallel_config.ep_size == 1
            and not parallel_config.enable_eplb
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        if w1.ndim == 4:
            experts = w1.shape[0]
            output_size = w1.shape[2]
            hidden_size = a1.shape[-1]
            tokens = a1.shape[-2]
            return experts, tokens, output_size, hidden_size, topk_ids.shape[1]
        return super().moe_problem_size(a1, w1, w2, topk_ids)

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
        del N, topk, global_num_experts, local_num_experts
        del expert_tokens_meta, activation
        return (0,), (0,), (M, K)

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def _activation_params(
        self, limit: float, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._swiglu_limit != limit:
            self._swiglu_limit = limit
            self._swiglu_alpha = torch.ones(
                self.num_experts, dtype=torch.float32, device=device
            )
            self._swiglu_beta = torch.zeros_like(self._swiglu_alpha)
            self._swiglu_limit_tensor = torch.full_like(self._swiglu_alpha, limit)
        assert self._swiglu_alpha is not None
        assert self._swiglu_beta is not None
        assert self._swiglu_limit_tensor is not None
        return self._swiglu_alpha, self._swiglu_beta, self._swiglu_limit_tensor

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ) -> None:
        del a2_scale, workspace13, workspace2, expert_tokens_meta
        assert a1q_scale is None
        assert expert_map is None
        assert not apply_router_weight_on_input
        assert activation == MoEActivation.SILU
        assert hidden_states.dtype == w1.dtype == w2.dtype == torch.bfloat16

        limit = float(getattr(self, "swiglu_limit", 0.0) or 0.0)
        assert limit > 0
        alpha, beta, clamp_limit = self._activation_params(limit, hidden_states.device)

        from flashinfer.fused_moe import (
            WeightLayout,
            trtllm_bf16_routed_moe,
        )

        trtllm_bf16_routed_moe(
            topk_ids=(topk_ids.to(torch.int32), topk_weights),
            hidden_states=hidden_states,
            gemm1_weights=w1,
            gemm2_weights=w2,
            num_experts=global_num_experts,
            top_k=topk_ids.shape[1],
            n_group=None,
            topk_group=None,
            intermediate_size=self.intermediate_size,
            local_expert_offset=0,
            local_num_experts=self.num_experts,
            routing_method_type=1,
            use_shuffled_weight=True,
            weight_layout=WeightLayout.BlockMajorK,
            do_finalize=True,
            enable_pdl=True,
            tune_max_num_tokens=self.max_capture_size,
            activation_type=activation_to_flashinfer_int(activation),
            gemm1_alpha=alpha,
            gemm1_beta=beta,
            gemm1_clamp_limit=clamp_limit,
            output=output,
        )

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError("TRTLLM BF16 finalizes Top-K internally")
