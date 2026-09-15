# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _prepare_expert_assignment,
    invoke_fused_moe_triton_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    moe_kernel_quantize_input,
)
from vllm.triton_utils import tl


def workspace_shapes(
    self: TritonExperts,
    M: int,
    N: int,
    K: int,
    topk: int,
    global_num_experts: int,
    local_num_experts: int,
    expert_tokens_meta: mk.ExpertTokensMetadata | None,
    activation: MoEActivation,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    activation_out_dim = self.adjust_N_for_activation(N, activation)
    workspace1: tuple[int, ...] = (M, topk, max(activation_out_dim, K))
    workspace2: tuple[int, ...] = (M, topk, max(N, K))
    output = (M, K)

    if (
        self.moe_config.yoco.align_deep_gemm_w2
        and not self.quant_config.is_quantized
        and self.quant_config.weight_quant_dtype is None
        and self.w2_bias is None
    ):
        from vllm.model_executor.layers.fused_moe.experts.yoco_deep_gemm import (
            supports_yoco_deep_gemm_w2,
            yoco_deep_gemm_w2_workspace_rows,
        )

        if supports_yoco_deep_gemm_w2():
            packed_rows = yoco_deep_gemm_w2_workspace_rows(M, topk, local_num_experts)
            workspace1_numel = max(
                M * topk * max(activation_out_dim, K), packed_rows * K
            )
            workspace2_numel = max(
                M * topk * max(N, K), packed_rows * activation_out_dim
            )
            # Keep M as the leading dimension for activation chunking.
            workspace1 = (M, (workspace1_numel + M - 1) // M)
            workspace2 = (M, (workspace2_numel + M - 1) // M)
    return (workspace1, workspace2, output)


def apply(
    self: TritonExperts,
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
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    expert_tokens_meta: mk.ExpertTokensMetadata | None,
    apply_router_weight_on_input: bool,
):
    # Check constraints.
    if self.quant_config.use_int4_w4a16:
        assert hidden_states.size(-1) // 2 == w1.size(2), "Hidden size mismatch"
    else:
        assert hidden_states.size(-1) == w1.size(2), (
            f"Hidden size mismatch {hidden_states.size(-1)} != {w1.size(2)}"
        )

    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert hidden_states.dim() == 2
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ]

    E, num_tokens, N, K, top_k_num = self.moe_problem_size(
        hidden_states, w1, w2, topk_ids
    )

    if global_num_experts == -1:
        global_num_experts = E

    config = None
    if (
        self.moe_config.yoco.fast_w13_config
        and not self.quant_config.is_quantized
        and self.quant_config.weight_quant_dtype is None
        and hidden_states.dtype == torch.bfloat16
    ):
        from vllm.model_executor.layers.fused_moe.experts.yoco_triton import (
            try_get_yoco_w13_config,
        )

        config = try_get_yoco_w13_config(
            num_tokens,
            E,
            w1.size(1) // 2,
            w1.size(2),
        )
    if config is None:
        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            self.quant_config.config_name(hidden_states.dtype),
            num_tokens,
            block_shape=self.block_shape,
            use_tuned_config=self.moe_config.use_tuned_config,
        )

    align_configs = None
    if (
        self.moe_config.yoco.align_weighted_swiglu
        and not self.quant_config.is_quantized
        and self.quant_config.weight_quant_dtype is None
        and hidden_states.dtype == w1.dtype == w2.dtype == torch.bfloat16
        and expert_map is None
        and global_num_experts == E
        and self.w1_bias is None
        and self.w2_bias is None
        and self._lora_context is None
        and not apply_router_weight_on_input
    ):
        from vllm.model_executor.layers.yoco_align_moe import (
            get_yoco_align_moe_configs,
        )

        align_configs = get_yoco_align_moe_configs(
            num_tokens,
            w1.shape,
            w2.shape,
            top_k_num,
            config,
            device_index=hidden_states.device.index or 0,
        )
        if align_configs is not None:
            config = align_configs[0]

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    elif (
        hidden_states.dtype == torch.float8_e4m3fn
        or hidden_states.dtype == torch.float8_e4m3fnuz
    ):
        compute_type = tl.bfloat16
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    # Note that the output tensor might be in workspace1
    intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, N))
    cache2_dim = self.adjust_N_for_activation(N, activation)
    intermediate_cache2 = _resize_cache(
        workspace13, (num_tokens * top_k_num, cache2_dim)
    )
    intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

    sorted_token_ids: torch.Tensor | None
    if (
        self.quant_config.use_fp8_w8a8
        and self.moe_config.yoco.fp8_decode_aligned
        and 2 < num_tokens <= 4
    ):
        # The generic small-M heuristic gives each route its own block.
        # Grouping M=3/4 routes avoids repeated expert-weight reads when
        # YOCO tokens select the same experts; M=1/2 keep the cheaper setup.
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], global_num_experts, expert_map
        )
    else:
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            _prepare_expert_assignment(
                topk_ids,
                config,
                num_tokens,
                top_k_num,
                global_num_experts,
                expert_map,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                block_shape=self.block_shape,
            )
        )

    invoke_fused_moe_triton_kernel(
        hidden_states,
        w1,
        intermediate_cache1,
        a1q_scale,
        self.w1_scale,
        None,  # topk_weights
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,  # mul_routed_weights
        top_k_num,
        config,
        compute_type=compute_type,
        use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
        use_int8_w8a8=self.quant_config.use_int8_w8a8,
        use_int8_w8a16=self.quant_config.use_int8_w8a16,
        use_int4_w4a16=self.quant_config.use_int4_w4a16,
        per_channel_quant=self.per_act_token_quant,
        block_shape=self.block_shape,
        B_bias=self.w1_bias,
    )

    # LoRA w13: applied to intermediate_cache1 before activation, using
    # hidden_states as the lora_a input.  moe_lora_align_block_size is
    # called once here and results reused for the w2 LoRA below.
    sorted_token_ids_lora = None
    expert_ids_lora = None
    num_tokens_post_padded_lora = None
    token_lora_mapping = None
    lora_context = self._lora_context
    if lora_context is not None:
        (
            sorted_token_ids_lora,
            expert_ids_lora,
            num_tokens_post_padded_lora,
            token_lora_mapping,
        ) = self.apply_w13_lora(
            lora_context,
            y=intermediate_cache1,
            x=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            expert_map=expert_map,
            w1=w1,
            w2=w2,
            num_tokens=num_tokens,
            top_k_num=top_k_num,
        )

    yoco_align_weighted_swiglu = self.moe_config.yoco.align_weighted_swiglu
    yoco_swapped_w13 = self.moe_config.yoco.swapped_w13
    # W8A8 must include routing weights when computing the W2 input
    # quantization scale. Multiplying in the W2 epilogue is too late.
    # Keep the existing Fast BF16 epilogue policy unchanged.
    weight_before_w2 = yoco_align_weighted_swiglu or (
        self.quant_config.use_fp8_w8a8 and self.moe_config.apply_router_weight_before_w2
    )
    direct_fp8_activation = (
        weight_before_w2
        and not yoco_align_weighted_swiglu
        and self.moe_config.yoco.direct_fp8_activation
        and self.quant_config.use_fp8_w8a8
        and self.moe_config.yoco.fp8_decode_aligned
        and self.block_shape == [128, 128]
        and intermediate_cache1.dtype == torch.bfloat16
    )
    a2q_scale: torch.Tensor | None
    if direct_fp8_activation:
        from vllm.model_executor.layers.yoco_ops.fp8 import silu_mul_quant_fp8_triton

        assert lora_context is None
        assert not apply_router_weight_on_input
        assert activation == MoEActivation.SILU
        swiglu_limit = self.moe_config.swiglu_limit
        assert swiglu_limit is not None and swiglu_limit > 0
        qintermediate_cache2, a2q_scale = silu_mul_quant_fp8_triton(
            intermediate_cache1.view(-1, N),
            clamp_limit=float(swiglu_limit),
            row_weights=topk_weights.reshape(-1),
            round_before_quant=False,
            packed_scales=False,
        )
    elif weight_before_w2:
        assert lora_context is None, (
            "Applying router weights before W2 is not supported with MoE LoRA"
        )
        from vllm.model_executor.layers.fused_moe.experts.yoco_triton import (
            yoco_weighted_swiglu,
        )

        assert not apply_router_weight_on_input
        assert activation == MoEActivation.SILU
        swiglu_limit = self.moe_config.swiglu_limit
        assert swiglu_limit is not None and swiglu_limit > 0
        yoco_weighted_swiglu(
            intermediate_cache2,
            intermediate_cache1.view(-1, N),
            topk_weights.reshape(-1),
            float(swiglu_limit),
        )
    elif yoco_swapped_w13:
        from vllm.model_executor.layers.fused_moe.experts.yoco_triton import (
            yoco_swapped_clamped_swiglu,
        )

        assert not apply_router_weight_on_input
        assert activation == MoEActivation.SILU
        swiglu_limit = self.moe_config.swiglu_limit
        assert swiglu_limit is not None and swiglu_limit > 0
        yoco_swapped_clamped_swiglu(
            intermediate_cache2,
            intermediate_cache1.view(-1, N),
            float(swiglu_limit),
        )
    else:
        self.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

    use_yoco_deep_gemm_w2 = (
        self.moe_config.yoco.align_deep_gemm_w2
        and yoco_align_weighted_swiglu
        and not self.quant_config.is_quantized
        and self.quant_config.weight_quant_dtype is None
        and self.w2_bias is None
        and lora_context is None
        and expert_map is None
        and global_num_experts == E
        and intermediate_cache2.dtype == torch.bfloat16
        and w2.dtype == torch.bfloat16
        and w2.is_contiguous()
        and w2.size(1) % 8 == 0
        and w2.size(2) % 64 == 0
    )
    if use_yoco_deep_gemm_w2:
        from vllm.model_executor.layers.fused_moe.experts.yoco_deep_gemm import (
            supports_yoco_deep_gemm_w2,
            yoco_deep_gemm_w2,
        )

        if supports_yoco_deep_gemm_w2():
            yoco_deep_gemm_w2(
                intermediate_cache3.view(-1, K),
                intermediate_cache2,
                w2,
                topk_ids,
                workspace2,
                workspace13,
            )
            _yoco_moe_sum(self, intermediate_cache3, output)
            return

    if not direct_fp8_activation:
        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            intermediate_cache2,
            a2_scale,
            self.quant_dtype,
            self.per_act_token_quant,
            self.block_shape,
            quantization_emulation=self.quantization_emulation,
            group_quant_eps=self.quant_config.group_quant_eps,
        )

    w2_config = config if align_configs is None else align_configs[1]
    if (
        align_configs is None
        and direct_fp8_activation
        and self.moe_config.yoco.fp8_decode_aligned
        and self.moe_config.use_tuned_config
        and self.quant_config.use_fp8_w8a8
        and self.block_shape == [128, 128]
        and top_k_num == 8
        and qintermediate_cache2.dtype == w2.dtype == torch.float8_e4m3fn
        and expert_map is None
        and self.w2_bias is None
        and lora_context is None
    ):
        from vllm.model_executor.layers.fused_moe.experts.yoco_triton import (
            try_get_yoco_fp8_w2_config,
        )

        w2_config = try_get_yoco_fp8_w2_config(
            num_tokens, E, w2.size(1), w2.size(2), config
        )
    elif (
        align_configs is None
        and self.moe_config.yoco.separate_w2_config
        and not self.quant_config.is_quantized
        and self.quant_config.weight_quant_dtype is None
        and hidden_states.dtype == torch.bfloat16
    ):
        from vllm.model_executor.layers.fused_moe.experts.yoco_triton import (
            try_get_yoco_w2_config,
        )

        w2_config = try_get_yoco_w2_config(
            num_tokens,
            E,
            w2.size(1),
            w2.size(2),
            config,
        )

    invoke_fused_moe_triton_kernel(
        qintermediate_cache2,
        w2,
        intermediate_cache3,
        a2q_scale,
        self.w2_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not (apply_router_weight_on_input or weight_before_w2),
        1,
        w2_config,
        compute_type=compute_type,
        use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
        use_int8_w8a8=self.quant_config.use_int8_w8a8,
        use_int8_w8a16=self.quant_config.use_int8_w8a16,
        use_int4_w4a16=self.quant_config.use_int4_w4a16,
        per_channel_quant=self.per_act_token_quant,
        block_shape=self.block_shape,
        B_bias=self.w2_bias,
    )

    # LoRA w2: applied to intermediate_cache3 before moe_sum, using the
    # unquantized intermediate_cache2 as the lora_a input.  Reuses the
    # sorted_token_ids_lora computed above.
    if lora_context is not None:
        assert not yoco_align_weighted_swiglu, (
            "Applying router weights before W2 is not supported with MoE LoRA"
        )
        self.apply_w2_lora(
            lora_context,
            y=intermediate_cache3,
            x=intermediate_cache2,
            topk_weights=topk_weights,
            sorted_token_ids_lora=sorted_token_ids_lora,
            expert_ids_lora=expert_ids_lora,
            num_tokens_post_padded_lora=num_tokens_post_padded_lora,
            token_lora_mapping=token_lora_mapping,
            num_tokens=num_tokens,
            w1=w1,
            w2=w2,
            top_k_num=top_k_num,
        )

    # separate function is required for MoE + LoRA
    _yoco_moe_sum(self, intermediate_cache3, output)


def _yoco_moe_sum(
    self: TritonExperts, input: torch.Tensor, output: torch.Tensor
) -> None:
    use_yoco_sum = self.moe_config.yoco.align_moe_sum or (
        self.moe_config.yoco.fast_moe_sum and input.shape[0] >= 2048
    )
    if use_yoco_sum:
        from vllm.model_executor.layers.fused_moe.experts.yoco_triton import (
            yoco_topk8_sum,
        )

        yoco_topk8_sum(input, output)
    else:
        self.moe_sum(input, output)
