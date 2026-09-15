# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Legacy YOCO imports for existing experiment and alignment scripts."""

from vllm.model_executor.layers.yoco_ops.norm import RMSClip as RMSClip
from vllm.model_executor.layers.yoco_ops.norm import RMSNorm as RMSNorm
from vllm.model_executor.layers.yoco_ops.norm import (
    _apply_per_head_norm as _apply_per_head_norm,
)
from vllm.model_executor.layers.yoco_ops.norm import _build_qk_norm as _build_qk_norm
from vllm.model_executor.layers.yoco_ops.norm import (
    _run_yoco_fused_add_rms_norm_cuda as _run_yoco_fused_add_rms_norm_cuda,
)
from vllm.model_executor.layers.yoco_ops.norm import (
    _run_yoco_rms_norm_cuda as _run_yoco_rms_norm_cuda,
)
from vllm.model_executor.layers.yoco_ops.norm import (
    _yoco_add_residual as _yoco_add_residual,
)
from vllm.model_executor.layers.yoco_ops.norm import (
    _yoco_align_fused_add_rms_norm_cuda as _yoco_align_fused_add_rms_norm_cuda,
)
from vllm.model_executor.layers.yoco_ops.norm import (
    _yoco_align_fused_add_rms_norm_fake as _yoco_align_fused_add_rms_norm_fake,
)
from vllm.model_executor.layers.yoco_ops.norm import (
    _yoco_align_rms_clip as _yoco_align_rms_clip,
)
from vllm.model_executor.layers.yoco_ops.norm import (
    _yoco_align_rms_clip_no_weight as _yoco_align_rms_clip_no_weight,
)
from vllm.model_executor.layers.yoco_ops.norm import (
    _yoco_align_rms_norm as _yoco_align_rms_norm,
)
from vllm.model_executor.layers.yoco_ops.norm import (
    _yoco_align_rms_norm_cuda as _yoco_align_rms_norm_cuda,
)
from vllm.model_executor.layers.yoco_ops.norm import (
    _yoco_align_weighted_rms_clip_cuda as _yoco_align_weighted_rms_clip_cuda,
)
from vllm.model_executor.layers.yoco_ops.norm import (
    _yoco_bf16_add_rms_norm_cuda as _yoco_bf16_add_rms_norm_cuda,
)
from vllm.model_executor.layers.yoco_ops.norm import (
    _yoco_fused_add_rms_norm_cuda as _yoco_fused_add_rms_norm_cuda,
)
from vllm.model_executor.layers.yoco_ops.norm import (
    _yoco_fused_add_rms_norm_fake as _yoco_fused_add_rms_norm_fake,
)
from vllm.model_executor.layers.yoco_ops.norm import (
    _yoco_residual_dtype as _yoco_residual_dtype,
)
from vllm.model_executor.layers.yoco_ops.norm import (
    _yoco_rms_clip_cuda as _yoco_rms_clip_cuda,
)
from vllm.model_executor.layers.yoco_ops.norm import (
    _yoco_rms_clip_fake as _yoco_rms_clip_fake,
)
from vllm.model_executor.layers.yoco_ops.norm import (
    _yoco_rms_norm_cuda as _yoco_rms_norm_cuda,
)
from vllm.model_executor.layers.yoco_ops.norm import (
    _yoco_rms_norm_fake as _yoco_rms_norm_fake,
)
from vllm.model_executor.layers.yoco_ops.norm import (
    _yoco_weighted_rms_clip_cuda as _yoco_weighted_rms_clip_cuda,
)
from vllm.model_executor.layers.yoco_ops.norm import (
    _yoco_weighted_rms_clip_fake as _yoco_weighted_rms_clip_fake,
)
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.model_executor.layers.yoco_ops.norm import (
        _yoco_bf16_add as _yoco_bf16_add,
    )
    from vllm.model_executor.layers.yoco_ops.norm import (
        _yoco_fused_add_rms_norm_kernel as _yoco_fused_add_rms_norm_kernel,
    )
    from vllm.model_executor.layers.yoco_ops.norm import (
        _yoco_rms_clip_kernel as _yoco_rms_clip_kernel,
    )
    from vllm.model_executor.layers.yoco_ops.norm import (
        _yoco_rms_norm_kernel as _yoco_rms_norm_kernel,
    )
    from vllm.model_executor.layers.yoco_ops.norm import (
        _yoco_weighted_rms_clip_kernel as _yoco_weighted_rms_clip_kernel,
    )

from vllm.model_executor.layers.yoco_ops.projection import (
    _YOCO_L3_HIDDEN_SIZE as _YOCO_L3_HIDDEN_SIZE,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _YOCO_L3_VOCAB_SIZE as _YOCO_L3_VOCAB_SIZE,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _YOCO_Q_LAMBDA_MERGED_MAX_TOKENS as _YOCO_Q_LAMBDA_MERGED_MAX_TOKENS,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _YOCO_QKV_LAMBDA_MERGED_MAX_TOKENS as _YOCO_QKV_LAMBDA_MERGED_MAX_TOKENS,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _YOCO_SM100_DIFF_V3_TP4_MIN_TOKENS as _YOCO_SM100_DIFF_V3_TP4_MIN_TOKENS,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _YOCO_SM100_LM_HEAD_MAX_TOKENS as _YOCO_SM100_LM_HEAD_MAX_TOKENS,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _run_yoco_lm_head_cuda as _run_yoco_lm_head_cuda,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _supports_yoco_sm100_diff_v3_kernel as _supports_yoco_sm100_diff_v3_kernel,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _supports_yoco_sm100_lm_head_kernel as _supports_yoco_sm100_lm_head_kernel,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_align_linear as _yoco_align_linear,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_align_qkv_linear as _yoco_align_qkv_linear,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_can_fuse_fp8_attention as _yoco_can_fuse_fp8_attention,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_can_fuse_fp8_output as _yoco_can_fuse_fp8_output,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_clip_fp8 as _yoco_clip_fp8,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_clip_fp8_fake as _yoco_clip_fp8_fake,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_diff_attention_v2 as _yoco_diff_attention_v2,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_diff_attention_v3 as _yoco_diff_attention_v3,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_diff_attention_v3_cuda as _yoco_diff_attention_v3_cuda,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_diff_attention_v3_dispatch as _yoco_diff_attention_v3_dispatch,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_diff_attention_v3_fake as _yoco_diff_attention_v3_fake,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_fused_shared_gate_moe_output_cuda as _yoco_fused_shared_gate_moe_output_cuda,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_fused_shared_gate_moe_output_fake as _yoco_fused_shared_gate_moe_output_fake,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_lm_head_bf16_cuda as _yoco_lm_head_bf16_cuda,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_lm_head_bf16_fake as _yoco_lm_head_bf16_fake,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_lm_head_cuda as _yoco_lm_head_cuda,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_lm_head_dispatch as _yoco_lm_head_dispatch,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_lm_head_fake as _yoco_lm_head_fake,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_project_fp8_output as _yoco_project_fp8_output,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _YocoAlignEmbeddingMethod as _YocoAlignEmbeddingMethod,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _YocoAlignLinearMethod as _YocoAlignLinearMethod,
)

if HAS_TRITON:
    from vllm.model_executor.layers.yoco_ops.projection import (
        _yoco_diff_attention_v3_kernel as _yoco_diff_attention_v3_kernel,
    )
    from vllm.model_executor.layers.yoco_ops.projection import (
        _yoco_fused_shared_gate_moe_output_kernel as _yoco_fused_shared_gate_moe_output_kernel,  # noqa: E501
    )
    from vllm.model_executor.layers.yoco_ops.projection import (
        _yoco_lm_head_kernel as _yoco_lm_head_kernel,
    )

from vllm.model_executor.layers.yoco_ops.rotary import (
    YOCORotaryEmbedding as YOCORotaryEmbedding,
)
from vllm.model_executor.layers.yoco_ops.rotary import (
    _yoco_align_rotary_embedding as _yoco_align_rotary_embedding,
)
from vllm.model_executor.layers.yoco_ops.rotary import (
    _yoco_apply_rotary_emb as _yoco_apply_rotary_emb,
)
from vllm.model_executor.layers.yoco_ops.rotary import (
    _yoco_qk_rms_clip_rotary_cuda as _yoco_qk_rms_clip_rotary_cuda,
)
from vllm.model_executor.layers.yoco_ops.rotary import (
    _yoco_qk_rms_clip_rotary_fake as _yoco_qk_rms_clip_rotary_fake,
)
from vllm.model_executor.layers.yoco_ops.rotary import (
    _yoco_qk_rms_clip_rotary_weighted_cuda as _yoco_qk_rms_clip_rotary_weighted_cuda,
)
from vllm.model_executor.layers.yoco_ops.rotary import (
    _yoco_qk_rms_clip_rotary_weighted_fake as _yoco_qk_rms_clip_rotary_weighted_fake,
)
from vllm.model_executor.layers.yoco_ops.rotary import (
    _yoco_qkv_clip_rotary_fp8 as _yoco_qkv_clip_rotary_fp8,
)
from vllm.model_executor.layers.yoco_ops.rotary import (
    _yoco_qkv_clip_rotary_fp8_fake as _yoco_qkv_clip_rotary_fp8_fake,
)
from vllm.model_executor.layers.yoco_ops.rotary import (
    _yoco_rotary_cuda as _yoco_rotary_cuda,
)
from vllm.model_executor.layers.yoco_ops.rotary import (
    _yoco_rotary_fake as _yoco_rotary_fake,
)

if HAS_TRITON:
    from vllm.model_executor.layers.yoco_ops.rotary import (
        _yoco_fma_rn as _yoco_fma_rn,
    )
    from vllm.model_executor.layers.yoco_ops.rotary import (
        _yoco_mul_rn as _yoco_mul_rn,
    )
    from vllm.model_executor.layers.yoco_ops.rotary import (
        _yoco_qk_rms_clip_rotary_kernel as _yoco_qk_rms_clip_rotary_kernel,
    )
    from vllm.model_executor.layers.yoco_ops.rotary import (
        _yoco_rotary_kernel as _yoco_rotary_kernel,
    )

from vllm.model_executor.layers.yoco_attention import (
    YOCOCrossAttention as YOCOCrossAttention,
)
from vllm.model_executor.layers.yoco_attention import (
    YOCOSelfAttention as YOCOSelfAttention,
)
from vllm.model_executor.layers.yoco_moe import (
    YOCOCombinedOutputTransform as YOCOCombinedOutputTransform,
)
from vllm.model_executor.layers.yoco_moe import (
    YOCOLatentInputTransform as YOCOLatentInputTransform,
)
from vllm.model_executor.layers.yoco_moe import (
    YOCOLatentOutputTransform as YOCOLatentOutputTransform,
)
from vllm.model_executor.layers.yoco_moe import YOCOMoE as YOCOMoE
from vllm.model_executor.layers.yoco_moe import YOCOSharedExperts as YOCOSharedExperts
from vllm.model_executor.layers.yoco_moe import (
    _yoco_align_shared_expert_swiglu as _yoco_align_shared_expert_swiglu,
)
from vllm.model_executor.layers.yoco_moe import (
    _yoco_align_shared_expert_swiglu_unclamped as _yoco_align_shared_expert_swiglu_unclamped,  # noqa: E501
)
from vllm.model_executor.layers.yoco_ops.routing import (
    _yoco_align_router_linear as _yoco_align_router_linear,
)
from vllm.model_executor.layers.yoco_ops.routing import (
    _yoco_align_topk_routing as _yoco_align_topk_routing,
)
from vllm.model_executor.layers.yoco_ops.routing import (
    _yoco_align_topk_routing_impl as _yoco_align_topk_routing_impl,
)
from vllm.model_executor.layers.yoco_ops.routing import (
    _yoco_normalized_router_linear as _yoco_normalized_router_linear,
)
from vllm.model_executor.layers.yoco_ops.routing import (
    _yoco_router_linear_bf16_cuda as _yoco_router_linear_bf16_cuda,
)
from vllm.model_executor.layers.yoco_ops.routing import (
    _yoco_router_linear_bf16_fake as _yoco_router_linear_bf16_fake,
)
from vllm.model_executor.layers.yoco_ops.routing import (
    _yoco_router_linear_tf32_cuda as _yoco_router_linear_tf32_cuda,
)
from vllm.model_executor.layers.yoco_ops.routing import (
    _yoco_router_linear_tf32_fake as _yoco_router_linear_tf32_fake,
)
from vllm.model_executor.layers.yoco_ops.routing import (
    _yoco_topk_routing as _yoco_topk_routing,
)
from vllm.model_executor.layers.yoco_ops.routing import (
    _yoco_topk_routing_impl as _yoco_topk_routing_impl,
)
from vllm.model_executor.models.yoco_config import (
    YOCO_ONLINE_QUANT_IGNORE as YOCO_ONLINE_QUANT_IGNORE,
)
from vllm.model_executor.models.yoco_config import (
    YOCO_PACKED_MODULES_MAPPING as YOCO_PACKED_MODULES_MAPPING,
)
from vllm.model_executor.models.yoco_config import _cfg_int as _cfg_int
from vllm.model_executor.models.yoco_config import (
    _get_yoco_execution_mode as _get_yoco_execution_mode,
)
from vllm.model_executor.models.yoco_config import (
    _maybe_build_yoco_quant_config as _maybe_build_yoco_quant_config,
)
from vllm.model_executor.models.yoco_config import (
    _select_yoco_fast_moe_backend as _select_yoco_fast_moe_backend,
)
from vllm.model_executor.models.yoco_config import (
    _select_yoco_online_fp8_moe_backend as _select_yoco_online_fp8_moe_backend,
)
from vllm.model_executor.models.yoco_config import _swiglu_limit as _swiglu_limit
from vllm.model_executor.models.yoco_config import (
    _yoco_runtime_sliding_window as _yoco_runtime_sliding_window,
)
from vllm.model_executor.models.yoco_config import (
    _yoco_standalone_prefill_min_tokens as _yoco_standalone_prefill_min_tokens,
)
from vllm.model_executor.models.yoco_config import (
    _yoco_verified_trtllm_cache_max_capture as _yoco_verified_trtllm_cache_max_capture,
)
from vllm.model_executor.models.yoco_diagnostics import (
    _YOCO_LOGICAL_ROUTE_DUMP_BATCHES as _YOCO_LOGICAL_ROUTE_DUMP_BATCHES,
)
from vllm.model_executor.models.yoco_diagnostics import (
    _YOCO_LOGICAL_ROUTE_DUMP_INDEX as _YOCO_LOGICAL_ROUTE_DUMP_INDEX,
)
from vllm.model_executor.models.yoco_diagnostics import (
    _YOCO_LOGICAL_ROUTE_DUMP_ROOT as _YOCO_LOGICAL_ROUTE_DUMP_ROOT,
)
from vllm.model_executor.models.yoco_diagnostics import (
    _maybe_dump_yoco_logical_routes as _maybe_dump_yoco_logical_routes,
)
from vllm.model_executor.models.yoco_diagnostics import (
    _yoco_logical_moe_layer_id as _yoco_logical_moe_layer_id,
)

if HAS_TRITON:
    from vllm.model_executor.layers.yoco_ops.routing import (
        _yoco_align_router_kernel as _yoco_align_router_kernel,
    )
    from vllm.model_executor.layers.yoco_ops.routing import (
        _yoco_fused_topk_routing_kernel as _yoco_fused_topk_routing_kernel,
    )
