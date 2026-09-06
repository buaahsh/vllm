# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only YOCO (You Only Cache Once) model.

The reference training implementation lives at
``llm-train/llm/arch/model.py`` in the YOCO repository. This file
implements the same forward semantics on top of vLLM's primitives
(``Attention``, ``FusedMoE``, ``RMSNorm``, ``RoPE``, ``QKVParallelLinear``,
``MergedColumnParallelLinear`` etc.) so that the model can be served with
torch.compile + CUDAGraph and standard TP / DP / EP parallelism.

Architecture summary (matches the HF checkpoint shipped in ``hf-weights``):

* 20 layers total.

  * Layers 0..9 are *self*-attention with sliding window 512 and per-layer
    QK-norm.  These layers are executed three times in sequence on the same
    hidden state (``universal_loop = 3``).  Each iteration writes to its own
    KV cache, yielding 30 distinct self-attention KV caches.
  * Layers 10..19 are *cross*-attention (YOCO global) layers.  They share a
    single (K, V) pair produced once by a model-level ``yoco_norm`` +
    K/V projection on the hidden state at the end of the third self-loop pass.
    Fast BF16 TP1 packs the two checkpoint projections into one GEMM; Align
    preserves the original two-GEMM order. Layer 10 owns that single KV cache;
    layers 11..19 use ``kv_sharing_target_layer_name`` to read from it without
    creating new caches.
* All layers use *diff-attention*: ``q_proj`` outputs ``2 * head * head_dim``
  values; attention is computed once with ``2*head`` Q-heads.  Diff-v2 uses
  ``attn1 - sigmoid(gate) * attn2`` while diff-v3 gates both alternating
  heads independently before subtraction.
* All layers run an MoE (128 routed experts, top-k=8, softmax routing with
  post-top-k renormalization) plus a gated shared expert.

The ``cross_head`` field in the HF config is honored as 48 — this is twice
the self-layer ``head`` (24) and matches the checkpoint q_proj shape
``(2 * 48 * 128, 3072) = (12288, 3072)`` for layers 10..19.  Cross layers
therefore have 96 Q-heads (with diff-attention doubling) but still only
4 KV-heads (``cross_kv_head`` defaults to ``kv_head``).

Tokenization note: this model relies on the ``O200kHarmonyTokenizer`` shipped
in ``hf-weights/o200k_harmony_tokenizer.py``.  Launch vLLM with
``--trust-remote-code`` so that the tokenizer registers itself with
``AutoTokenizer``.  The tokenizer's ``encode()`` prepends BOS by default; for
the *completion* code path vLLM passes ``add_special_tokens=True`` (one BOS
prepended).  For the *chat* code path vLLM passes ``add_special_tokens=False``
so the chat template's literal ``<|startoftext|>`` is the only BOS.  In both
cases BOS appears exactly once.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, CUDAGraphMode, VllmConfig
from vllm.config.kernel import MoEBackend
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import DPMetadata, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import (
    SiluAndMul,
    SiluAndMulWithClampFP32,
)
from vllm.model_executor.layers.attention.attention import Attention, AttentionType
from vllm.model_executor.layers.batch_invariant import (
    linear_batch_invariant,
    matmul_kernel_persistent,
)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import _encode_layer_name, direct_register_custom_op
from vllm.v1.attention.backends.utils import KVSharingFastPrefillMetadata

_YOCO_Q_LAMBDA_MERGED_MAX_TOKENS = 2048
_YOCO_QKV_LAMBDA_MERGED_MAX_TOKENS = 4096
_YOCO_L3_HIDDEN_SIZE = 3072
_YOCO_L3_VOCAB_SIZE = 154880
_YOCO_SM100_LM_HEAD_MAX_TOKENS = 16
_YOCO_LOGICAL_ROUTE_DUMP_ROOT = os.getenv("VLLM_YOCO_LOGICAL_ROUTE_DUMP")
_YOCO_LOGICAL_ROUTE_DUMP_BATCHES = frozenset(
    int(value)
    for value in os.getenv("VLLM_YOCO_LOGICAL_ROUTE_DUMP_BATCHES", "").split(",")
    if value
)
_YOCO_LOGICAL_ROUTE_DUMP_INDEX = 0

logger = init_logger(__name__)


def _yoco_verified_trtllm_cache_max_capture(cache_path: str | None) -> int:
    """Return the largest graph backed by a YOCO-validated tactic cache."""
    if not cache_path:
        return 32
    # Any environment-compatible persisted cache retains the already validated
    # graph64 behavior. Graph128/256 are admitted only for the real-routing
    # tactic sets; the earlier synthetic M128 tactic regressed end-to-end.
    max_capture = 64
    try:
        with open(cache_path) as cache_file:
            configs = json.load(cache_file)
    except (OSError, TypeError, ValueError):
        return max_capture

    expected = {
        32: [32, 24],
        64: [32, 17],
        128: [32, 17],
    }
    for num_tokens, tactic in expected.items():
        key = (
            "('flashinfer::trtllm_bf16_moe', 'MoERunner', "
            f"(({num_tokens}, 1024), (0,), ({num_tokens}, 8), "
            f"({num_tokens}, 8), ({num_tokens}, 1024), (0,), (0,), (0,)), ())"
        )
        if configs.get(key) != ["MoERunner", tactic]:
            return max_capture
    max_capture = 128
    key = (
        "('flashinfer::trtllm_bf16_moe', 'MoERunner', "
        "((256, 1024), (0,), (256, 8), (256, 8), (256, 1024), "
        "(0,), (0,), (0,)), ())"
    )
    if configs.get(key) == ["MoERunner", [64, 0]]:
        return 256
    return max_capture


def _yoco_logical_moe_layer_id(
    layer_idx: int,
    loop_idx: int,
    first_cross_layer_idx: int,
    universal_loop: int,
) -> int:
    if layer_idx < first_cross_layer_idx:
        if not 0 <= loop_idx < universal_loop:
            raise ValueError(f"invalid YOCO universal loop index {loop_idx}")
        return loop_idx * first_cross_layer_idx + layer_idx
    return first_cross_layer_idx * universal_loop + layer_idx - first_cross_layer_idx


def _maybe_dump_yoco_logical_routes(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    logical_route_info: tuple[int, int, int] | None,
    loop_idx: int,
) -> None:
    root = _YOCO_LOGICAL_ROUTE_DUMP_ROOT
    num_tokens = hidden_states.shape[0]
    if (
        root is None
        or logical_route_info is None
        or num_tokens not in _YOCO_LOGICAL_ROUTE_DUMP_BATCHES
        or not os.path.exists(os.path.join(root, "ENABLED"))
    ):
        return
    if torch.compiler.is_compiling():
        raise RuntimeError("YOCO logical routing dump requires eager execution")
    layer_idx, first_cross_layer_idx, universal_loop = logical_route_info
    logical_layer_id = _yoco_logical_moe_layer_id(
        layer_idx,
        loop_idx,
        first_cross_layer_idx,
        universal_loop,
    )
    _, topk_ids = _yoco_topk_routing(
        hidden_states,
        router_logits,
        top_k,
        True,
    )
    global _YOCO_LOGICAL_ROUTE_DUMP_INDEX
    index = _YOCO_LOGICAL_ROUTE_DUMP_INDEX
    _YOCO_LOGICAL_ROUTE_DUMP_INDEX += 1
    os.makedirs(root, exist_ok=True)
    torch.save(
        {
            "index": index,
            "num_tokens": num_tokens,
            "logical_layer_id": logical_layer_id,
            "topk_ids": topk_ids.to(torch.int16).cpu(),
        },
        os.path.join(
            root,
            f"{index:08d}-m{num_tokens}-l{logical_layer_id:02d}.pt",
        ),
    )


def _yoco_standalone_prefill_min_tokens(vllm_config: VllmConfig) -> int:
    """Keep every configured pure-decode graph on the Triton path."""
    return max(
        1024,
        vllm_config.scheduler_config.max_num_seqs + 1,
        int(vllm_config.compilation_config.max_cudagraph_capture_size or 0) + 1,
    )


def _select_yoco_fast_moe_backend(
    *,
    execution_mode: str,
    quant_config: QuantizationConfig | None,
    tp_size: int,
    config: PretrainedConfig,
    vllm_config: VllmConfig,
) -> MoEBackend | None:
    """Select the per-role FlashInfer policy for L3 BF16."""
    if execution_mode != "fast" or quant_config is not None or tp_size != 1:
        return None
    additional_config = vllm_config.additional_config or {}
    kv_transfer_config = vllm_config.kv_transfer_config
    standalone = kv_transfer_config is None or kv_transfer_config.kv_connector is None
    role = None if standalone else kv_transfer_config.kv_role
    kernel_config = vllm_config.kernel_config
    enable_decode_autotune = False
    if standalone:
        if not bool(additional_config.get("yoco_fast_standalone_flashinfer_moe", True)):
            return None
        parallel = getattr(vllm_config, "parallel_config", None)
        model_config = getattr(vllm_config, "model_config", None)
        if (
            parallel is None
            or getattr(parallel, "data_parallel_size", 0) != 1
            or getattr(parallel, "pipeline_parallel_size", 0) != 1
            or getattr(parallel, "prefill_context_parallel_size", 1) != 1
            or getattr(parallel, "decode_context_parallel_size", 1) != 1
            or getattr(model_config, "dtype", None) != torch.bfloat16
            or not vllm_config.cache_config.kv_sharing_fast_prefill
            or kernel_config.enable_flashinfer_autotune is True
            or vllm_config.scheduler_config.max_num_batched_tokens
            < _yoco_standalone_prefill_min_tokens(vllm_config)
        ):
            return None
    elif role == "kv_producer":
        if (
            not bool(additional_config.get("yoco_fast_prefill_flashinfer_moe", True))
            or not vllm_config.cache_config.kv_sharing_fast_prefill
            # Autotuning at the P scheduler's maximum M regressed the
            # measured 1410-input C8 throughput. Keep P on the heuristic.
            or kernel_config.enable_flashinfer_autotune is True
        ):
            return None
    elif role == "kv_consumer":
        scheduler_config = vllm_config.scheduler_config
        if (
            not bool(additional_config.get("yoco_fast_decode_flashinfer_moe", True))
            or scheduler_config.max_num_seqs < 64
            or scheduler_config.max_num_batched_tokens < 8192
        ):
            return None
        enable_decode_autotune = True
    else:
        return None
    if (
        _cfg_int(config, "hidden_size", "d_model") != _YOCO_L3_HIDDEN_SIZE
        or _cfg_int(config, "num_experts", "moe_expert_num") != 128
        or _cfg_int(config, "num_experts_per_tok", "moe_top_k", "top_k") != 8
        or _cfg_int(config, "moe_intermediate_size", "moe_ffn_dim") != 3840
        or _cfg_int(config, "moe_latent_dim", default=0) != 1024
        or _swiglu_limit(config) <= 0
    ):
        return None
    if kernel_config.moe_backend not in ("auto", "triton", "flashinfer_cutlass"):
        return None
    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 10:
        return None
    if standalone and getattr(capability, "minor", 0) != 0:
        return None
    selected_backend: MoEBackend = "flashinfer_cutlass"
    if enable_decode_autotune:
        compilation_config = getattr(vllm_config, "compilation_config", None)
        max_capture_size = int(
            getattr(compilation_config, "max_cudagraph_capture_size", 0) or 0
        )
        default_trtllm_max_capture = _yoco_verified_trtllm_cache_max_capture(
            os.getenv("VLLM_YOCO_FLASHINFER_AUTOTUNE_CACHE")
        )
        trtllm_max_capture = int(
            additional_config.get(
                "yoco_fast_decode_trtllm_max_capture",
                default_trtllm_max_capture,
            )
        )
        use_trtllm = (
            bool(additional_config.get("yoco_fast_decode_trtllm_moe", True))
            and 0 < max_capture_size <= trtllm_max_capture
        )
        if use_trtllm:
            from vllm.model_executor.layers.fused_moe.experts.yoco_trtllm_bf16 import (
                has_yoco_trtllm_bf16_clamp,
            )

            if has_yoco_trtllm_bf16_clamp():
                selected_backend = "yoco_flashinfer_trtllm"
        if selected_backend == "flashinfer_cutlass":
            from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe

            if not has_flashinfer_cutlass_fused_moe():
                return None
        # Kernel warmup runs after model construction, so this per-service
        # policy reaches FlashInfer's official max-M tactic tuner.
        kernel_config.enable_flashinfer_autotune = True
    else:
        from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe

        if not has_flashinfer_cutlass_fused_moe():
            return None
    backend_label = (
        "YOCO FlashInfer TRTLLM"
        if selected_backend == "yoco_flashinfer_trtllm"
        else "FlashInfer CUTLASS"
    )
    if standalone:
        kernel_config.enable_flashinfer_autotune = False
        logger.info_once(
            "YOCO Fast standalone uses FlashInfer CUTLASS for M>=%d; "
            "smaller batches retain Triton with shared workspace",
            _yoco_standalone_prefill_min_tokens(vllm_config),
        )
        return selected_backend
    logger.info_once(
        "YOCO Fast %s selected %s BF16 MoE%s; "
        "unmatched configurations keep the configured backend",
        "pure Prefill" if role == "kv_producer" else "high-throughput Decode",
        backend_label,
        " with autotune" if enable_decode_autotune else "",
    )
    return selected_backend


if HAS_TRITON:

    @triton.jit
    def _yoco_align_router_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        stride_x: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # One fixed IEEE FP32 reduction per token/expert pair. No padding the
        # token batch to 128 rows and no process-global TF32 mode switching.
        token = tl.program_id(0)
        expert = tl.program_id(1) * 4 + tl.arange(0, 4)
        k = tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + token * stride_x + k, k < K, 0).to(tl.float32)
        weight = tl.load(
            weight_ptr + expert[:, None] * K + k[None, :],
            (expert[:, None] < N) & (k[None, :] < K),
            0,
        ).to(tl.float32)
        logits = tl.sum(weight * x[None, :], axis=1)
        tl.store(out_ptr + token * N + expert, logits, expert < N)

    @triton.jit
    def _yoco_rms_clip_kernel(
        x_ptr,
        output_ptr,
        num_rows,
        eps: tl.constexpr,
        limit: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        cols = tl.arange(0, HEAD_DIM)[None, :]
        row_mask = rows < num_rows
        values = tl.load(
            x_ptr + rows * HEAD_DIM + cols,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        square_sum = tl.sum(tl.where(row_mask, values * values, 0.0), axis=1)[:, None]
        clip_coef = limit * tl.extra.cuda.libdevice.rsqrt(square_sum / HEAD_DIM + eps)
        clip_coef = tl.minimum(clip_coef, 1.0)
        tl.store(
            output_ptr + rows * HEAD_DIM + cols,
            values * clip_coef,
            mask=row_mask,
        )

    @triton.jit
    def _yoco_weighted_rms_clip_kernel(
        x_ptr,
        weight_ptr,
        output_ptr,
        num_tokens,
        num_heads,
        token_stride,
        head_stride,
        eps: tl.constexpr,
        limit: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        ROUND_BEFORE_WEIGHT: tl.constexpr,
    ):
        head_rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        cols = tl.arange(0, HEAD_DIM)[None, :]
        row_mask = head_rows < num_tokens * num_heads
        token = head_rows // num_heads
        head = head_rows % num_heads
        input_offsets = token * token_stride + head * head_stride + cols
        values = tl.load(
            x_ptr + input_offsets,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        square_sum = tl.sum(tl.where(row_mask, values * values, 0.0), axis=1)[:, None]
        clip_coef = limit * tl.extra.cuda.libdevice.rsqrt(square_sum / HEAD_DIM + eps)
        clip_coef = tl.minimum(clip_coef, 1.0)

        clipped = values * clip_coef
        if ROUND_BEFORE_WEIGHT:
            # Preserve the source-level BF16 boundary before applying gamma.
            clipped = clipped.to(tl.bfloat16).to(tl.float32)
        weight = tl.load(weight_ptr + cols).to(tl.float32)
        tl.store(
            output_ptr + head_rows * HEAD_DIM + cols,
            clipped * weight,
            mask=row_mask,
        )

    @triton.jit
    def _yoco_rms_norm_kernel(
        x_ptr,
        weight_ptr,
        output_ptr,
        num_rows,
        eps: tl.constexpr,
        HIDDEN_SIZE: tl.constexpr,
        REDUCTION_BLOCK: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        row_mask = row < num_rows
        cols = tl.arange(0, REDUCTION_BLOCK)[None, :]
        square_acc = tl.full([BLOCK_ROWS, REDUCTION_BLOCK], 0.0, tl.float32)

        for offset in tl.range(0, HIDDEN_SIZE, REDUCTION_BLOCK):
            hidden_offsets = offset + cols
            mask = (hidden_offsets < HIDDEN_SIZE) & row_mask
            values = tl.load(
                x_ptr + row * HIDDEN_SIZE + hidden_offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            square_acc = tl.where(mask, square_acc + values * values, square_acc)

        square_sum = tl.sum(square_acc, axis=1)[:, None]
        inv_rms = tldevice.rsqrt(square_sum / HIDDEN_SIZE + eps)

        for offset in tl.range(0, HIDDEN_SIZE, REDUCTION_BLOCK):
            hidden_offsets = offset + cols
            mask = (hidden_offsets < HIDDEN_SIZE) & row_mask
            values = tl.load(
                x_ptr + row * HIDDEN_SIZE + hidden_offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            weight = tl.load(
                weight_ptr + hidden_offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            tl.store(
                output_ptr + row * HIDDEN_SIZE + hidden_offsets,
                values * inv_rms * weight,
                mask=mask,
            )

    @triton.jit
    def _yoco_fused_add_rms_norm_kernel(
        x_ptr,
        residual_ptr,
        weight_ptr,
        output_ptr,
        residual_out_ptr,
        num_rows,
        eps: tl.constexpr,
        HIDDEN_SIZE: tl.constexpr,
        REDUCTION_BLOCK: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        row_mask = row < num_rows
        cols = tl.arange(0, REDUCTION_BLOCK)[None, :]
        square_acc = tl.full([BLOCK_ROWS, REDUCTION_BLOCK], 0.0, tl.float32)

        # Materialize the FP32 residual sum while accumulating its norm. This
        # preserves the existing add-then-normalize order but removes the
        # standalone residual-add kernel and its extra read.
        for offset in tl.range(0, HIDDEN_SIZE, REDUCTION_BLOCK):
            hidden_offsets = offset + cols
            mask = (hidden_offsets < HIDDEN_SIZE) & row_mask
            offsets = row * HIDDEN_SIZE + hidden_offsets
            x = tl.load(
                x_ptr + offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            residual = tl.load(
                residual_ptr + offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            values = x + residual
            tl.store(residual_out_ptr + offsets, values, mask=mask)
            square_acc = tl.where(mask, square_acc + values * values, square_acc)

        square_sum = tl.sum(square_acc, axis=1)[:, None]
        inv_rms = tldevice.rsqrt(square_sum / HIDDEN_SIZE + eps)

        for offset in tl.range(0, HIDDEN_SIZE, REDUCTION_BLOCK):
            hidden_offsets = offset + cols
            mask = (hidden_offsets < HIDDEN_SIZE) & row_mask
            offsets = row * HIDDEN_SIZE + hidden_offsets
            values = tl.load(
                residual_out_ptr + offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            weight = tl.load(
                weight_ptr + hidden_offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            tl.store(
                output_ptr + offsets,
                values * inv_rms * weight,
                mask=mask,
            )

    @triton.jit
    def _yoco_fused_shared_gate_moe_output_kernel(
        shared_output_ptr,
        routed_output_ptr,
        hidden_states_ptr,
        gate_weight_ptr,
        output_ptr,
        num_rows,
        HIDDEN_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fuse YOCO's shared gate and final shared+routed MoE add."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = (row < num_rows) & (cols < HIDDEN_SIZE)
        offsets = row * HIDDEN_SIZE + cols

        hidden = tl.load(
            hidden_states_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        gate_weight = tl.load(
            gate_weight_ptr + cols,
            mask=cols < HIDDEN_SIZE,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        gate = tl.sum(hidden * gate_weight, axis=0)

        # Keep the same BF16 boundaries as the unfused serving expression:
        # BF16 GEMV output -> BF16 sigmoid -> BF16 multiply -> BF16 add.
        gate = gate.to(tl.bfloat16).to(tl.float32)
        scale = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
        shared_output = tl.load(
            shared_output_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        routed_output = tl.load(
            routed_output_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        gated_shared = (scale * shared_output).to(tl.bfloat16).to(tl.float32)
        output = (routed_output + gated_shared).to(tl.bfloat16)
        tl.store(output_ptr + offsets, output, mask=mask)

    @triton.jit
    def _yoco_fused_topk_routing_kernel(
        logits_ptr,
        topk_weights_ptr,
        topk_ids_ptr,
        num_rows,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        cols = tl.arange(0, 128)[None, :]
        row_mask = rows < num_rows
        logits = tl.load(
            logits_ptr + rows * 128 + cols,
            mask=row_mask,
            other=float("-inf"),
        )

        # Keep the current fast path's FP32 full-row softmax.  Although its
        # denominator cancels during the final Top-8 renormalization, retaining
        # it minimizes numerical differences from the unfused implementation.
        row_max = tl.max(logits, axis=1)[:, None]
        numerator = tl.extra.cuda.libdevice.exp(logits - row_max)
        scores = numerator / tl.sum(numerator, axis=1)[:, None]

        ranks = tl.arange(0, 8)[None, :]
        selected_scores = tl.full((BLOCK_ROWS, 8), 0.0, tl.float32)
        selected_ids = tl.full((BLOCK_ROWS, 8), 0, tl.int32)
        for rank in tl.static_range(8):
            value, index = tl.max(
                scores,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            selected_scores = tl.where(
                ranks == rank,
                value[:, None],
                selected_scores,
            )
            selected_ids = tl.where(
                ranks == rank,
                index[:, None],
                selected_ids,
            )
            scores = tl.where(cols == index[:, None], float("-inf"), scores)

        selected_scores /= tl.sum(selected_scores, axis=1)[:, None]
        output_offsets = rows * 8 + ranks
        tl.store(topk_weights_ptr + output_offsets, selected_scores, mask=row_mask)
        tl.store(topk_ids_ptr + output_offsets, selected_ids, mask=row_mask)

    @triton.jit
    def _yoco_diff_attention_v3_kernel(
        attention_ptr,
        gate_ptr,
        output_ptr,
        gate_token_stride,
        gate_head_stride,
        NUM_HEAD_PAIRS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HEAD_GROUP: tl.constexpr,
    ):
        """Apply both gates once per head pair, then broadcast over HEAD_DIM."""
        group = tl.program_id(0)
        groups_per_token: tl.constexpr = NUM_HEAD_PAIRS // HEAD_GROUP
        token = group // groups_per_token
        first_pair = token * NUM_HEAD_PAIRS + (group % groups_per_token) * HEAD_GROUP
        heads = tl.arange(0, HEAD_GROUP)[:, None]
        dims = tl.arange(0, HEAD_DIM)[None, :]
        pair = first_pair + heads
        first_head = 2 * pair
        pair_in_token = (group % groups_per_token) * HEAD_GROUP + heads
        first_gate_offset = (
            token * gate_token_stride + 2 * pair_in_token * gate_head_stride
        )

        first_gate = tl.load(gate_ptr + first_gate_offset).to(tl.float32)
        second_gate = tl.load(gate_ptr + first_gate_offset + gate_head_stride).to(
            tl.float32
        )
        first = tl.load(attention_ptr + first_head * HEAD_DIM + dims).to(tl.float32)
        second = tl.load(attention_ptr + (first_head + 1) * HEAD_DIM + dims).to(
            tl.float32
        )
        result = first * tl.sigmoid(first_gate) - second * tl.sigmoid(second_gate)
        tl.store(output_ptr + pair * HEAD_DIM + dims, result)

    @triton.jit
    def _yoco_lm_head_kernel(
        hidden_ptr,
        weight_ptr,
        output_ptr,
        num_tokens,
        HIDDEN_SIZE: tl.constexpr,
        VOCAB_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Small-M BF16 LM head over the checkpoint's row-major weight."""
        rows = tl.arange(0, BLOCK_M)
        cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offsets = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, HIDDEN_SIZE, BLOCK_K):
            hidden = tl.load(
                hidden_ptr + rows[:, None] * HIDDEN_SIZE + k_start + k_offsets[None, :],
                mask=rows[:, None] < num_tokens,
                other=0.0,
            )
            weight = tl.load(
                weight_ptr + cols[None, :] * HIDDEN_SIZE + k_start + k_offsets[:, None],
                mask=cols[None, :] < VOCAB_SIZE,
                other=0.0,
            )
            accumulator += tl.dot(hidden, weight)

        # llm-train rounds the GEMM result to BF16, then casts logits to FP32.
        accumulator = accumulator.to(tl.bfloat16).to(tl.float32)
        tl.store(
            output_ptr + rows[:, None] * VOCAB_SIZE + cols[None, :],
            accumulator,
            mask=(rows[:, None] < num_tokens) & (cols[None, :] < VOCAB_SIZE),
        )

    # Match the current Inductor fallback's PTX exactly: it rounds the second
    # product, then fuses the first product with the final add/subtract.
    @triton.jit
    def _yoco_mul_rn(x, y):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[x, y],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _yoco_fma_rn(x, y, z):
        return tl.inline_asm_elementwise(
            "fma.rn.f32 $0, $1, $2, $3;",
            constraints="=f,f,f,f",
            args=[x, y, z],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _yoco_rotary_kernel(
        query_ptr,
        key_ptr,
        query_output_ptr,
        key_output_ptr,
        positions_ptr,
        cos_sin_cache_ptr,
        num_rows,
        query_row_stride,
        query_head_stride,
        key_row_stride,
        key_head_stride,
        positions_stride,
        QUERY_HEADS: tl.constexpr,
        KEY_HEADS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pair_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        query_pairs = num_rows * QUERY_HEADS * 64
        total_pairs = num_rows * (QUERY_HEADS + KEY_HEADS) * 64
        valid = pair_offsets < total_pairs
        is_query = pair_offsets < query_pairs
        local_offsets = tl.where(
            is_query,
            pair_offsets,
            pair_offsets - query_pairs,
        )
        num_heads = tl.where(is_query, QUERY_HEADS, KEY_HEADS)
        row = local_offsets // (num_heads * 64)
        head = (local_offsets // 64) % num_heads
        rotary_col = local_offsets % 64
        position = tl.load(
            positions_ptr + row * positions_stride,
            mask=valid,
            other=0,
        )
        cos = tl.load(
            cos_sin_cache_ptr + position * 128 + rotary_col,
            mask=valid,
            other=0.0,
        )
        sin = tl.load(
            cos_sin_cache_ptr + position * 128 + 64 + rotary_col,
            mask=valid,
            other=0.0,
        )
        row_stride = tl.where(is_query, query_row_stride, key_row_stride)
        head_stride = tl.where(is_query, query_head_stride, key_head_stride)
        input_offsets = row * row_stride + head * head_stride + rotary_col
        input_ptrs = tl.where(
            is_query,
            query_ptr + input_offsets,
            key_ptr + input_offsets,
        )
        first_half = tl.load(
            input_ptrs,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        second_half = tl.load(
            input_ptrs + 64,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        first_output = _yoco_fma_rn(
            first_half,
            cos,
            -_yoco_mul_rn(second_half, sin),
        )
        second_output = _yoco_fma_rn(
            second_half,
            cos,
            _yoco_mul_rn(first_half, sin),
        )
        output_base = row * num_heads * 128 + head * 128 + rotary_col
        output_ptrs = tl.where(
            is_query,
            query_output_ptr + output_base,
            key_output_ptr + output_base,
        )
        tl.store(
            output_ptrs,
            first_output,
            mask=valid,
        )
        tl.store(
            output_ptrs + 64,
            second_output,
            mask=valid,
        )

    @triton.jit
    def _yoco_qk_rms_clip_rotary_kernel(
        query_ptr,
        key_ptr,
        query_weight_ptr,
        key_weight_ptr,
        query_output_ptr,
        key_output_ptr,
        positions_ptr,
        cos_sin_cache_ptr,
        num_rows,
        query_num_groups,
        query_row_stride,
        query_head_stride,
        key_row_stride,
        key_head_stride,
        positions_stride,
        eps: tl.constexpr,
        limit: tl.constexpr,
        QUERY_HEADS: tl.constexpr,
        KEY_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
    ):
        group = tl.program_id(0)
        is_query = group < query_num_groups
        local_group = tl.where(is_query, group, group - query_num_groups)
        head_rows = local_group * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        num_heads = tl.where(is_query, QUERY_HEADS, KEY_HEADS)
        num_head_rows = num_rows * num_heads
        row_mask = head_rows < num_head_rows
        token = head_rows // num_heads
        head = head_rows % num_heads
        cols = tl.arange(0, HEAD_DIM)[None, :]

        row_stride = tl.where(is_query, query_row_stride, key_row_stride)
        head_stride = tl.where(is_query, query_head_stride, key_head_stride)
        input_base = token * row_stride + head * head_stride
        input_ptrs = tl.where(
            is_query,
            query_ptr + input_base,
            key_ptr + input_base,
        )
        values = tl.load(
            input_ptrs + cols,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        square_sum = tl.sum(tl.where(row_mask, values * values, 0.0), axis=1)[:, None]
        clip_coef = limit * tl.extra.cuda.libdevice.rsqrt(square_sum / HEAD_DIM + eps)
        clip_coef = tl.minimum(clip_coef, 1.0)

        # Match the unfused RMSClip -> BF16 tensor -> RoPE sequence. The
        # explicit BF16 round is the semantic boundary that used to be the
        # intermediate tensor store/load.
        clipped = values * clip_coef
        if HAS_WEIGHT:
            weight_ptrs = tl.where(
                is_query,
                query_weight_ptr + cols,
                key_weight_ptr + cols,
            )
            weight = tl.load(weight_ptrs, mask=row_mask, other=0.0).to(tl.float32)
            clipped = clipped * weight
        # The affine training kernel fuses the source-level pre-gamma BF16
        # cast away and materializes BF16 only after gamma multiplication.
        clipped = clipped.to(tl.bfloat16).to(tl.float32)
        first_half = cols < HEAD_DIM // 2
        partner_cols = tl.where(
            first_half,
            cols + HEAD_DIM // 2,
            cols - HEAD_DIM // 2,
        )
        partner = tl.load(
            input_ptrs + partner_cols,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        partner = partner * clip_coef
        if HAS_WEIGHT:
            partner_weight_ptrs = tl.where(
                is_query,
                query_weight_ptr + partner_cols,
                key_weight_ptr + partner_cols,
            )
            partner_weight = tl.load(
                partner_weight_ptrs,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            partner = partner * partner_weight
        partner = partner.to(tl.bfloat16).to(tl.float32)

        rotary_col = cols % (HEAD_DIM // 2)
        position = tl.load(
            positions_ptr + token * positions_stride,
            mask=row_mask,
            other=0,
        )
        cos = tl.load(
            cos_sin_cache_ptr + position * HEAD_DIM + rotary_col,
            mask=row_mask,
            other=0.0,
        )
        sin = tl.load(
            cos_sin_cache_ptr + position * HEAD_DIM + HEAD_DIM // 2 + rotary_col,
            mask=row_mask,
            other=0.0,
        )
        signed_product = _yoco_mul_rn(partner, sin)
        signed_product = tl.where(first_half, -signed_product, signed_product)
        output = _yoco_fma_rn(clipped, cos, signed_product)

        output_base = head_rows * HEAD_DIM
        output_ptrs = tl.where(
            is_query,
            query_output_ptr + output_base,
            key_output_ptr + output_base,
        )
        tl.store(output_ptrs + cols, output, mask=row_mask)


def _yoco_rms_clip_cuda(
    x: torch.Tensor,
    eps: float,
    limit: float,
) -> torch.Tensor:
    x_contiguous = x.contiguous()
    output = torch.empty_like(x_contiguous)
    num_rows = x_contiguous.numel() // x_contiguous.shape[-1]
    _yoco_rms_clip_kernel[(triton.cdiv(num_rows, 8),)](
        x_contiguous,
        output,
        num_rows,
        eps=eps,
        limit=limit,
        HEAD_DIM=128,
        BLOCK_ROWS=8,
        num_warps=4,
        num_stages=1,
    )
    return output


def _yoco_diff_attention_v3_cuda(
    attention: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    num_tokens, twice_num_heads, head_dim = attention.shape
    num_head_pairs = twice_num_heads // 2
    output = torch.empty(
        (num_tokens, num_head_pairs, head_dim),
        dtype=attention.dtype,
        device=attention.device,
    )
    # L3 TP1 has 32 head pairs. Splitting a token across several CTAs exposes
    # substantially more parallelism than the old one-CTA-per-token layout.
    # These three ranges are tuned on B200; TP4 keeps its original 8-pair CTA.
    if num_head_pairs == 32:
        if num_tokens < 64:
            head_group, num_warps = 4, 4
        elif num_tokens < 512:
            head_group, num_warps = 8, 4
        else:
            head_group, num_warps = 16, 8
    else:
        head_group, num_warps = num_head_pairs, 4
    grid = (num_tokens * num_head_pairs // head_group,)
    _yoco_diff_attention_v3_kernel[grid](
        attention,
        gate,
        output,
        gate.stride(0),
        gate.stride(1),
        NUM_HEAD_PAIRS=num_head_pairs,
        HEAD_DIM=head_dim,
        HEAD_GROUP=head_group,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


def _yoco_diff_attention_v3_fake(
    attention: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    del gate
    return attention.new_empty(
        (attention.shape[0], attention.shape[1] // 2, attention.shape[2])
    )


def _yoco_lm_head_cuda(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    assert hidden_states.ndim == 2 and hidden_states.is_contiguous()
    assert weight.ndim == 2 and weight.is_contiguous()
    assert hidden_states.dtype == torch.bfloat16
    assert weight.dtype == torch.bfloat16
    assert hidden_states.shape[0] <= _YOCO_SM100_LM_HEAD_MAX_TOKENS
    assert hidden_states.shape[1] == _YOCO_L3_HIDDEN_SIZE
    assert weight.shape == (_YOCO_L3_VOCAB_SIZE, _YOCO_L3_HIDDEN_SIZE)
    output = torch.empty(
        (hidden_states.shape[0], _YOCO_L3_VOCAB_SIZE),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    _yoco_lm_head_kernel[(triton.cdiv(_YOCO_L3_VOCAB_SIZE, 128),)](
        hidden_states,
        weight,
        output,
        hidden_states.shape[0],
        HIDDEN_SIZE=_YOCO_L3_HIDDEN_SIZE,
        VOCAB_SIZE=_YOCO_L3_VOCAB_SIZE,
        BLOCK_M=16,
        BLOCK_N=128,
        BLOCK_K=128,
        num_warps=4,
        num_stages=3,
    )
    return output


def _yoco_lm_head_fake(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return hidden_states.new_empty(
        (*hidden_states.shape[:-1], weight.shape[0]), dtype=torch.float32
    )


def _yoco_rms_clip_fake(
    x: torch.Tensor,
    eps: float,
    limit: float,
) -> torch.Tensor:
    return torch.empty_like(x)


def _yoco_weighted_rms_clip_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    limit: float,
) -> torch.Tensor:
    assert x.ndim == 3 and x.shape[-1] == 128
    num_tokens, num_heads, _ = x.shape
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    num_head_rows = num_tokens * num_heads
    if num_head_rows == 0:
        return output
    # B200 CUDA-graph tuning for L3's 64 cross-Q heads. Larger row tiles
    # amortize scheduling overhead without changing each head's reduction tree.
    block_rows = 16 if num_head_rows < 12288 else 32
    _yoco_weighted_rms_clip_kernel[(triton.cdiv(num_head_rows, block_rows),)](
        x,
        weight,
        output,
        num_tokens,
        num_heads,
        x.stride(0),
        x.stride(1),
        eps=eps,
        limit=limit,
        HEAD_DIM=128,
        BLOCK_ROWS=block_rows,
        ROUND_BEFORE_WEIGHT=True,
        num_warps=8,
        num_stages=1,
    )
    return output


def _yoco_align_weighted_rms_clip_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    limit: float,
) -> torch.Tensor:
    assert x.ndim == 3 and x.shape[-1] == 128
    num_tokens, num_heads, _ = x.shape
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    num_head_rows = num_tokens * num_heads
    if num_head_rows == 0:
        return output
    # Fix the launch layout as well as the per-head reduction tree. Align
    # must not switch between an Inductor expression and this kernel at M=128.
    block_rows = 16
    _yoco_weighted_rms_clip_kernel[(triton.cdiv(num_head_rows, block_rows),)](
        x,
        weight,
        output,
        num_tokens,
        num_heads,
        x.stride(0),
        x.stride(1),
        eps=eps,
        limit=limit,
        HEAD_DIM=128,
        BLOCK_ROWS=block_rows,
        ROUND_BEFORE_WEIGHT=False,
        num_warps=8,
        num_stages=1,
    )
    return output


def _yoco_weighted_rms_clip_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    limit: float,
) -> torch.Tensor:
    del weight, eps, limit
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def _run_yoco_rms_norm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    reduction_block: int,
) -> torch.Tensor:
    x_contiguous = x.contiguous()
    weight_contiguous = weight.to(torch.bfloat16).contiguous()
    output = torch.empty_like(x_contiguous, dtype=torch.bfloat16)
    num_rows = x_contiguous.numel() // x_contiguous.shape[-1]
    if num_rows == 0:
        return output
    block_rows = 1
    _yoco_rms_norm_kernel[(triton.cdiv(num_rows, block_rows),)](
        x_contiguous,
        weight_contiguous,
        output,
        num_rows,
        eps=eps,
        HIDDEN_SIZE=x_contiguous.shape[-1],
        REDUCTION_BLOCK=reduction_block,
        BLOCK_ROWS=block_rows,
        num_warps=16,
        num_stages=1,
    )
    return output


def _yoco_rms_norm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    num_rows = x.numel() // x.shape[-1]
    reduction_block = 4096 if num_rows >= 128 else 2048
    return _run_yoco_rms_norm_cuda(x, weight, eps, reduction_block)


def _yoco_align_rms_norm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    # One FP32 reduction tree per hidden size, independent of token count,
    # graph padding, input dtype, and enclosing compilation context.
    return _run_yoco_rms_norm_cuda(x, weight, eps, triton.next_power_of_2(x.shape[-1]))


def _yoco_rms_norm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x, dtype=torch.bfloat16)


def _yoco_fused_add_rms_norm_cuda(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows = x.numel() // x.shape[-1]
    reduction_block = 4096 if num_rows >= 128 else 2048
    return _run_yoco_fused_add_rms_norm_cuda(x, residual, weight, eps, reduction_block)


def _yoco_align_fused_add_rms_norm_cuda(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _run_yoco_fused_add_rms_norm_cuda(
        x, residual, weight, eps, triton.next_power_of_2(x.shape[-1])
    )


def _run_yoco_fused_add_rms_norm_cuda(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    reduction_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_contiguous = x.contiguous()
    residual_contiguous = residual.contiguous()
    weight_contiguous = weight.to(torch.bfloat16).contiguous()
    output = torch.empty_like(x_contiguous, dtype=torch.bfloat16)
    residual_out = torch.empty_like(residual_contiguous, dtype=torch.float32)
    num_rows = x_contiguous.numel() // x_contiguous.shape[-1]
    if num_rows == 0:
        return output, residual_out
    block_rows = 1
    _yoco_fused_add_rms_norm_kernel[(triton.cdiv(num_rows, block_rows),)](
        x_contiguous,
        residual_contiguous,
        weight_contiguous,
        output,
        residual_out,
        num_rows,
        eps=eps,
        HIDDEN_SIZE=x_contiguous.shape[-1],
        REDUCTION_BLOCK=reduction_block,
        BLOCK_ROWS=block_rows,
        num_warps=16,
        num_stages=1,
    )
    return output, residual_out


def _yoco_fused_add_rms_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(x, dtype=torch.bfloat16),
        torch.empty_like(residual, dtype=torch.float32),
    )


def _yoco_fused_shared_gate_moe_output_cuda(
    shared_output: torch.Tensor,
    routed_output: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
) -> torch.Tensor:
    shared_output = shared_output.contiguous()
    routed_output = routed_output.contiguous()
    hidden_states = hidden_states.contiguous()
    gate_weight = gate_weight.to(torch.bfloat16).contiguous()
    output = torch.empty_like(routed_output, dtype=torch.bfloat16)
    num_rows = routed_output.numel() // routed_output.shape[-1]
    if num_rows == 0:
        return output
    num_warps = 8 if num_rows >= 4096 else 4
    _yoco_fused_shared_gate_moe_output_kernel[(num_rows,)](
        shared_output,
        routed_output,
        hidden_states,
        gate_weight,
        output,
        num_rows,
        HIDDEN_SIZE=routed_output.shape[-1],
        BLOCK_SIZE=4096,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


def _yoco_fused_shared_gate_moe_output_fake(
    shared_output: torch.Tensor,
    routed_output: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
) -> torch.Tensor:
    del shared_output, hidden_states, gate_weight
    return torch.empty_like(routed_output, dtype=torch.bfloat16)


def _yoco_rotary_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_output = torch.empty_like(query, memory_format=torch.contiguous_format)
    key_output = torch.empty_like(key, memory_format=torch.contiguous_format)
    num_rows = query.shape[0]
    query_heads = query.shape[1]
    key_heads = key.shape[1]
    block_size = 256
    num_pairs = num_rows * (query_heads + key_heads) * 64
    _yoco_rotary_kernel[(triton.cdiv(num_pairs, block_size),)](
        query,
        key,
        query_output,
        key_output,
        positions,
        cos_sin_cache,
        num_rows,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        positions.stride(0),
        QUERY_HEADS=query_heads,
        KEY_HEADS=key_heads,
        BLOCK_SIZE=block_size,
        num_warps=8,
        num_stages=1,
    )
    return query_output, key_output


def _yoco_rotary_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del positions, cos_sin_cache
    return (
        torch.empty_like(query, memory_format=torch.contiguous_format),
        torch.empty_like(key, memory_format=torch.contiguous_format),
    )


def _yoco_qk_rms_clip_rotary_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_output = torch.empty_like(query, memory_format=torch.contiguous_format)
    key_output = torch.empty_like(key, memory_format=torch.contiguous_format)
    num_rows = query.shape[0]
    query_heads = query.shape[1]
    key_heads = key.shape[1]
    block_rows = 8
    query_num_groups = triton.cdiv(num_rows * query_heads, block_rows)
    key_num_groups = triton.cdiv(num_rows * key_heads, block_rows)
    _yoco_qk_rms_clip_rotary_kernel[(query_num_groups + key_num_groups,)](
        query,
        key,
        query,
        key,
        query_output,
        key_output,
        positions,
        cos_sin_cache,
        num_rows,
        query_num_groups,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        positions.stride(0),
        eps=eps,
        limit=limit,
        QUERY_HEADS=query_heads,
        KEY_HEADS=key_heads,
        HEAD_DIM=128,
        BLOCK_ROWS=block_rows,
        HAS_WEIGHT=False,
        num_warps=4,
        num_stages=1,
    )
    return query_output, key_output


def _yoco_qk_rms_clip_rotary_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del positions, cos_sin_cache, eps, limit
    return (
        torch.empty_like(query, memory_format=torch.contiguous_format),
        torch.empty_like(key, memory_format=torch.contiguous_format),
    )


def _yoco_qk_rms_clip_rotary_weighted_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_output = torch.empty_like(query, memory_format=torch.contiguous_format)
    key_output = torch.empty_like(key, memory_format=torch.contiguous_format)
    query_weight = query_weight.to(torch.bfloat16).contiguous()
    key_weight = key_weight.to(torch.bfloat16).contiguous()
    num_rows = query.shape[0]
    query_heads = query.shape[1]
    key_heads = key.shape[1]
    # Two rows with one warp is the best balanced A6000 configuration across
    # decode and prefill sizes.  The fast path is allowed to choose a fixed
    # reduction tree; the align path keeps Inductor's shape-dependent tree.
    block_rows = 2
    query_num_groups = triton.cdiv(num_rows * query_heads, block_rows)
    key_num_groups = triton.cdiv(num_rows * key_heads, block_rows)
    _yoco_qk_rms_clip_rotary_kernel[(query_num_groups + key_num_groups,)](
        query,
        key,
        query_weight,
        key_weight,
        query_output,
        key_output,
        positions,
        cos_sin_cache,
        num_rows,
        query_num_groups,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        positions.stride(0),
        eps=eps,
        limit=limit,
        QUERY_HEADS=query_heads,
        KEY_HEADS=key_heads,
        HEAD_DIM=128,
        BLOCK_ROWS=block_rows,
        HAS_WEIGHT=True,
        num_warps=1,
        num_stages=1,
    )
    return query_output, key_output


def _yoco_qk_rms_clip_rotary_weighted_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del query_weight, key_weight, positions, cos_sin_cache, eps, limit
    return (
        torch.empty_like(query, memory_format=torch.contiguous_format),
        torch.empty_like(key, memory_format=torch.contiguous_format),
    )


def _yoco_router_linear_tf32_cuda(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    normalize_weight: bool,
) -> torch.Tensor:
    if normalize_weight:
        weight = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
    # The fast serving path intentionally uses the real token-row count.  Shape
    # padding belongs to the explicit alignment policy, not this default op:
    # CUDA Graph removes host launch overhead but cannot remove padded GEMM work.
    assert hidden_states.ndim == 2
    previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_matmul_precision = torch.get_float32_matmul_precision()
    previous_cuda_precision = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    try:
        return F.linear(hidden_states, weight)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
        torch.set_float32_matmul_precision(previous_matmul_precision)
        torch.backends.cuda.matmul.fp32_precision = previous_cuda_precision


def _yoco_router_linear_tf32_fake(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    normalize_weight: bool,
) -> torch.Tensor:
    del normalize_weight
    return hidden_states.new_empty((*hidden_states.shape[:-1], weight.shape[0]))


if current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_router_linear_tf32",
        op_func=_yoco_router_linear_tf32_cuda,
        fake_impl=_yoco_router_linear_tf32_fake,
    )

if HAS_TRITON and current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_diff_attention_v3",
        op_func=_yoco_diff_attention_v3_cuda,
        fake_impl=_yoco_diff_attention_v3_fake,
    )
    direct_register_custom_op(
        op_name="yoco_lm_head",
        op_func=_yoco_lm_head_cuda,
        fake_impl=_yoco_lm_head_fake,
    )
    direct_register_custom_op(
        op_name="yoco_rms_clip",
        op_func=_yoco_rms_clip_cuda,
        fake_impl=_yoco_rms_clip_fake,
    )
    direct_register_custom_op(
        op_name="yoco_weighted_rms_clip",
        op_func=_yoco_weighted_rms_clip_cuda,
        fake_impl=_yoco_weighted_rms_clip_fake,
    )
    direct_register_custom_op(
        op_name="yoco_align_weighted_rms_clip",
        op_func=_yoco_align_weighted_rms_clip_cuda,
        fake_impl=_yoco_weighted_rms_clip_fake,
    )
    direct_register_custom_op(
        op_name="yoco_rms_norm",
        op_func=_yoco_rms_norm_cuda,
        fake_impl=_yoco_rms_norm_fake,
    )
    direct_register_custom_op(
        op_name="yoco_align_rms_norm",
        op_func=_yoco_align_rms_norm_cuda,
        fake_impl=_yoco_rms_norm_fake,
    )
    direct_register_custom_op(
        op_name="yoco_fused_add_rms_norm",
        op_func=_yoco_fused_add_rms_norm_cuda,
        fake_impl=_yoco_fused_add_rms_norm_fake,
    )
    direct_register_custom_op(
        op_name="yoco_align_fused_add_rms_norm",
        op_func=_yoco_align_fused_add_rms_norm_cuda,
        fake_impl=_yoco_fused_add_rms_norm_fake,
    )
    direct_register_custom_op(
        op_name="yoco_fused_shared_gate_moe_output",
        op_func=_yoco_fused_shared_gate_moe_output_cuda,
        fake_impl=_yoco_fused_shared_gate_moe_output_fake,
    )
    direct_register_custom_op(
        op_name="yoco_rotary",
        op_func=_yoco_rotary_cuda,
        fake_impl=_yoco_rotary_fake,
    )
    direct_register_custom_op(
        op_name="yoco_qk_rms_clip_rotary",
        op_func=_yoco_qk_rms_clip_rotary_cuda,
        fake_impl=_yoco_qk_rms_clip_rotary_fake,
    )
    direct_register_custom_op(
        op_name="yoco_qk_rms_clip_rotary_weighted",
        op_func=_yoco_qk_rms_clip_rotary_weighted_cuda,
        fake_impl=_yoco_qk_rms_clip_rotary_weighted_fake,
    )


# --------------------------------------------------------------------------- #
# Config helpers                                                              #
# --------------------------------------------------------------------------- #


YOCO_PACKED_MODULES_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
}


YOCO_ONLINE_QUANT_IGNORE = [
    "re:.*\\.self_attn\\.lambda_proj$",
]


def _cfg_int(config: PretrainedConfig, *names: str, default: int | None = None) -> int:
    """Read the first attribute from ``config`` whose name is in ``names``.

    The YOCO HF config aliases canonical HF names (``hidden_size``,
    ``num_hidden_layers`` ...) to YOCO-native names (``d_model``, ``n_layers``
    ...).  This helper returns whichever is set so the model code can use the
    most readable name without caring which alias was used.
    """
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    if default is not None:
        return default
    raise AttributeError(
        f"None of the config fields {names!r} are set on {type(config).__name__}"
    )


def _get_yoco_execution_mode(vllm_config: VllmConfig) -> str:
    """Return the requested YOCO kernel policy.

    ``fast`` preserves the optimized serving path and remains the default.
    ``align`` preserves llm-train's BF16 operator and rounding boundaries.
    """
    additional_config = vllm_config.additional_config
    mode = (
        additional_config.get("yoco_execution_mode", "fast")
        if isinstance(additional_config, dict)
        else "fast"
    )
    if mode not in ("align", "fast"):
        raise ValueError(
            f"YOCO execution mode must be 'align' or 'fast', but got {mode!r}"
        )
    return mode


def _yoco_runtime_sliding_window(training_window_left: int) -> int:
    """Translate llm-train's left-window count to vLLM's total window size.

    FlashAttention's ``(left, 0)`` includes ``left`` previous tokens and the
    current token. vLLM accepts the total token count and subtracts one before
    calling the backend, so training's 512 becomes 513 here.
    """
    if training_window_left <= 0:
        raise ValueError("YOCO self-attention requires a positive sliding window")
    return training_window_left + 1


def _maybe_build_yoco_quant_config(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    if quant_config is None:
        return None
    quant_config.packed_modules_mapping = YOCO_PACKED_MODULES_MAPPING
    ignored_layers = getattr(quant_config, "ignored_layers", None)
    if ignored_layers is not None:
        for pattern in YOCO_ONLINE_QUANT_IGNORE:
            if pattern not in ignored_layers:
                ignored_layers.append(pattern)
    return quant_config


def _yoco_topk_routing_impl(
    router_logits: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert router_logits.dtype == torch.float32
    assert router_logits.shape[-1] == 128
    assert topk == 8
    router_logits = router_logits.contiguous()
    num_rows = router_logits.shape[0]
    output_shape = (num_rows, 8)
    topk_weights = torch.empty(
        output_shape,
        dtype=torch.float32,
        device=router_logits.device,
    )
    topk_ids = torch.empty(
        output_shape,
        # Every YOCO Fast MoE backend consumes INT32 expert ids.  Returning
        # INT64 here made CustomRoutingRouter launch a standalone cast after
        # each of the 40 logical Router calls.
        dtype=torch.int32,
        device=router_logits.device,
    )
    block_rows = 4
    _yoco_fused_topk_routing_kernel[(triton.cdiv(num_rows, block_rows),)](
        router_logits,
        topk_weights,
        topk_ids,
        num_rows,
        BLOCK_ROWS=block_rows,
        num_warps=4,
        num_stages=1,
    )
    return topk_weights, topk_ids


@torch.compile
def _yoco_align_topk_routing_impl(
    logits: torch.Tensor,
    topk: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """The complete routing expression compiled by llm-train."""
    gate_scores = F.softmax(logits, dim=-1, dtype=torch.float32)
    scores, top_indices = torch.topk(gate_scores, k=topk, dim=-1)
    probs = scores / scores.sum(dim=-1, keepdim=True)
    routing_probs = torch.zeros_like(logits).scatter(
        1, top_indices, probs.to(logits.dtype)
    )
    routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    return probs, top_indices, routing_probs, routing_map, gate_scores


def _yoco_align_topk_routing(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del hidden_states
    assert renormalize
    if HAS_TRITON and gating_output.is_cuda and current_platform.is_cuda():
        if gating_output.shape[-1] != 128 or topk != 8:
            raise ValueError("YOCO invariant routing requires 128 experts and Top-8")
        # Fix both reductions and the tie rule. Inductor's compiled softmax /
        # renormalization is not a stable numerical contract across warmup
        # shapes and enclosing graph-capture contexts.
        topk_weights, topk_ids = _yoco_topk_routing_impl(gating_output, topk)
    else:
        topk_weights, topk_ids, _, _, _ = _yoco_align_topk_routing_impl(
            gating_output, topk
        )
    # TransformerEngine's token unpermute traverses selected experts in
    # expert-id order. Preserve each probability/id pair while matching that
    # order for the subsequent fixed-order FP32 reduction.
    expert_order = torch.argsort(topk_ids, dim=-1)
    topk_ids = torch.gather(topk_ids, dim=-1, index=expert_order)
    topk_weights = torch.gather(topk_weights, dim=-1, index=expert_order)
    return topk_weights, topk_ids


def _yoco_topk_routing(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del hidden_states
    assert renormalize
    if gating_output.shape[-1] != 128 or topk != 8:
        gate_scores = F.softmax(gating_output, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(gate_scores, k=topk, dim=-1)
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights, topk_ids
    return _yoco_topk_routing_impl(gating_output, topk)


@torch.compile
def _yoco_align_rms_clip(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    limit: float,
) -> torch.Tensor:
    """The exact affine RMSClip expression compiled by llm-train."""
    x_float = x.float()
    clip_coef = (
        limit * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + eps)
    ).clamp(max=1.0)
    return (x_float * clip_coef).to(x.dtype) * weight.to(x.dtype)


@torch.compile
def _yoco_align_rms_clip_no_weight(
    x: torch.Tensor,
    eps: float,
    limit: float,
) -> torch.Tensor:
    """The exact non-affine RMSClip expression compiled by llm-train."""
    x_float = x.float()
    clip_coef = (
        limit * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + eps)
    ).clamp(max=1.0)
    return (x_float * clip_coef).to(x.dtype)


class RMSClip(nn.Module):
    """RMS-based clipping for YOCO ``qk_rms_clip`` models.

    Scales each ``head_dim`` slice by ``clamp(limit / rms, max=1.0)`` where
    ``rms = sqrt(mean(x**2, -1) + eps)``.  The optional affine weight is
    controlled by ``qk_rms_gamma``, matching training's ``RMSClip``.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        limit: float = 3.0,
        has_weight: bool = False,
        execution_mode: str = "fast",
    ) -> None:
        super().__init__()
        if execution_mode not in ("align", "fast"):
            raise ValueError(f"Unsupported YOCO execution mode: {execution_mode!r}")
        self.dim = dim
        self.eps = eps
        self.limit = limit
        self.execution_mode = execution_mode
        if has_weight:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, limit={self.limit}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.execution_mode == "align":
            device_capability = current_platform.get_device_capability()
            if (
                HAS_TRITON
                and x.is_cuda
                and x.dtype == torch.bfloat16
                and x.ndim == 3
                and x.shape[-1] == 128
                and device_capability is not None
                and device_capability.major == 10
                and self.weight is not None
            ):
                return torch.ops.vllm.yoco_align_weighted_rms_clip(
                    x, self.weight, self.eps, self.limit
                )
            if self.weight is None:
                if (
                    HAS_TRITON
                    and x.is_cuda
                    and x.dtype == torch.bfloat16
                    and x.shape[-1] == 128
                    and current_platform.is_cuda()
                ):
                    return torch.ops.vllm.yoco_rms_clip(x, self.eps, self.limit)
                return _yoco_align_rms_clip_no_weight(x, self.eps, self.limit)
            return _yoco_align_rms_clip(x, self.weight, self.eps, self.limit)
        if (
            HAS_TRITON
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.shape[-1] == 128
            and current_platform.is_cuda()
            and self.weight is None
        ):
            return torch.ops.vllm.yoco_rms_clip(x, self.eps, self.limit)
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        clip_coef = (self.limit * torch.rsqrt(variance + self.eps)).clamp(max=1.0)
        x = (x * clip_coef).to(orig_dtype)
        if self.weight is not None:
            x = x * self.weight.to(orig_dtype)
        return x


@torch.compile
def _yoco_align_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """The exact BF16 RMSNorm expression compiled by llm-train."""
    return F.rms_norm(
        x.to(torch.bfloat16),
        (x.shape[-1],),
        weight=weight.to(torch.bfloat16),
        eps=eps,
    )


class RMSNorm(nn.Module):
    """RMSNorm matching llm-train's compiled reduction semantics.

    Inductor keeps the residual input and reduction in FP32, loads the BF16
    weight as FP32, and casts only the output to BF16.  The CUDA path below
    also preserves Inductor's reduction order for hidden size 3072.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
        execution_mode: str = "fast",
    ) -> None:
        super().__init__()
        if execution_mode not in ("align", "fast"):
            raise ValueError(f"Unsupported YOCO execution mode: {execution_mode!r}")
        self.dim = dim
        self.eps = eps
        self.execution_mode = execution_mode
        weight = torch.ones(dim, dtype=dtype or torch.get_default_dtype())
        if has_weight:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            if self.execution_mode == "align":
                if (
                    HAS_TRITON
                    and x.is_cuda
                    and residual.is_cuda
                    and x.dtype in (torch.bfloat16, torch.float32)
                    and residual.dtype == torch.float32
                    and x.shape == residual.shape
                    and x.shape[-1] in (1024, 3072)
                    and current_platform.is_cuda()
                ):
                    return torch.ops.vllm.yoco_align_fused_add_rms_norm(
                        x, residual, self.weight, self.eps
                    )
                residual_out = residual + x.float()
                normalized = self.forward(residual_out)
                assert isinstance(normalized, torch.Tensor)
                return normalized, residual_out
            if (
                HAS_TRITON
                and x.is_cuda
                and residual.is_cuda
                and x.dtype in (torch.bfloat16, torch.float32)
                and residual.dtype == torch.float32
                and x.shape == residual.shape
                and x.shape[-1] == 3072
                and current_platform.is_cuda()
            ):
                return torch.ops.vllm.yoco_fused_add_rms_norm(
                    x, residual, self.weight, self.eps
                )
            residual_out = residual + x.float()
            normalized = self.forward(residual_out)
            assert isinstance(normalized, torch.Tensor)
            return normalized, residual_out
        if self.execution_mode == "align":
            if (
                HAS_TRITON
                and x.is_cuda
                and x.dtype in (torch.bfloat16, torch.float32)
                and x.shape[-1] in (1024, 3072)
                and current_platform.is_cuda()
            ):
                # Inductor may select a different RMS reduction tree when this
                # expression is compiled inside the full model. Keep the tree
                # fixed so BF16 rounding matches llm-train for every batch M.
                return torch.ops.vllm.yoco_align_rms_norm(x, self.weight, self.eps)
            return _yoco_align_rms_norm(x, self.weight, self.eps)
        if x.is_cuda and x.shape[-1] == 1024:
            # The latent norms use the same expression in both modes.  On
            # B200, Inductor's compiled reduction is faster than the eager
            # fallback while remaining bitwise-aligned with llm-train.
            return _yoco_align_rms_norm(x, self.weight, self.eps)
        if (
            HAS_TRITON
            and x.is_cuda
            and x.dtype == torch.float32
            and x.shape[-1] == 3072
            and current_platform.is_cuda()
        ):
            return torch.ops.vllm.yoco_rms_norm(x, self.weight, self.eps)
        return F.rms_norm(
            x.to(torch.bfloat16),
            (x.shape[-1],),
            weight=self.weight.to(torch.bfloat16),
            eps=self.eps,
        )


@torch.compile
def _yoco_apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)

    def apply(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat((y1, y2), dim=-1).to(x.dtype)

    return apply(query), apply(key)


@torch.compile
def _yoco_align_rotary_embedding(
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The exact cache lookup and RoPE expression compiled by llm-train."""
    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)

    def apply(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat((y1, y2), dim=-1).to(x.dtype)

    return apply(query), apply(key)


@torch.compile
def _yoco_diff_attention_v2(
    attn1: torch.Tensor,
    attn2: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    return attn1 - torch.sigmoid(gate).unsqueeze(-1) * attn2


@torch.compile
def _yoco_diff_attention_v3(
    attn1: torch.Tensor,
    attn2: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    gate1 = gate[:, 0::2]
    gate2 = gate[:, 1::2]
    return attn1 * torch.sigmoid(gate1).unsqueeze(-1) - attn2 * torch.sigmoid(
        gate2
    ).unsqueeze(-1)


# B200 full-CUDA-graph measurements for L3 TP4 show a stable win once there
# are at least 32 token rows. The TP1 head-group layout is non-regressing from
# the first row and becomes progressively faster as the token count grows.
_YOCO_SM100_DIFF_V3_TP4_MIN_TOKENS = 32


def _yoco_diff_attention_v3_dispatch(
    attention: torch.Tensor,
    gate: torch.Tensor,
    use_sm100_kernel: bool,
) -> torch.Tensor:
    if (
        use_sm100_kernel
        and attention.is_cuda
        and gate.is_cuda
        and attention.dtype == torch.bfloat16
        and gate.dtype == torch.bfloat16
        and attention.is_contiguous()
        and gate.stride(-1) == 1
        and (
            attention.shape[1] == 64
            or (
                attention.shape[1] == 16
                and attention.shape[0] >= _YOCO_SM100_DIFF_V3_TP4_MIN_TOKENS
            )
        )
        and attention.shape[2] == 128
        and gate.shape == attention.shape[:2]
    ):
        return torch.ops.vllm.yoco_diff_attention_v3(attention, gate)
    return _yoco_diff_attention_v3(attention[:, 0::2, :], attention[:, 1::2, :], gate)


def _supports_yoco_sm100_diff_v3_kernel(execution_mode: str) -> bool:
    if execution_mode != "fast" or not HAS_TRITON or not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major == 10


def _yoco_lm_head_dispatch(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    use_sm100_kernel: bool,
) -> torch.Tensor:
    """Use the B200 small-M kernel only for the measured L3 BF16 shape."""
    if (
        use_sm100_kernel
        and hidden_states.is_cuda
        and weight.is_cuda
        and hidden_states.ndim == 2
        and weight.ndim == 2
        and 0 < hidden_states.shape[0] <= _YOCO_SM100_LM_HEAD_MAX_TOKENS
        and hidden_states.shape[1] == _YOCO_L3_HIDDEN_SIZE
        and weight.shape == (_YOCO_L3_VOCAB_SIZE, _YOCO_L3_HIDDEN_SIZE)
        and hidden_states.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and weight.is_contiguous()
    ):
        return torch.ops.vllm.yoco_lm_head(hidden_states, weight)
    logits = F.linear(hidden_states, weight)
    return logits.float() if use_sm100_kernel else logits


def _supports_yoco_sm100_lm_head_kernel(execution_mode: str) -> bool:
    if execution_mode != "fast" or not HAS_TRITON or not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major == 10


def _yoco_normalized_router_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    weight = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return F.linear(hidden_states, weight)


def _yoco_align_router_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    normalize_weight: bool,
) -> torch.Tensor:
    """Fixed IEEE FP32 Router reduction, independent of token-row shape."""
    assert hidden_states.ndim == 2
    if hidden_states.is_cuda and current_platform.is_cuda():
        assert hidden_states.dtype == weight.dtype == torch.float32
        assert hidden_states.shape[1] == weight.shape[1]
        hidden_states = hidden_states.contiguous()
        if normalize_weight:
            weight = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
        weight = weight.contiguous()
        output = hidden_states.new_empty((hidden_states.shape[0], weight.shape[0]))
        if hidden_states.shape[0]:
            grid = (hidden_states.shape[0], triton.cdiv(weight.shape[0], 4))
            _yoco_align_router_kernel[grid](
                hidden_states,
                weight,
                output,
                stride_x=hidden_states.stride(0),
                K=weight.shape[1],
                N=weight.shape[0],
                BLOCK_K=triton.next_power_of_2(weight.shape[1]),
                num_warps=4,
                enable_fp_fusion=False,
            )
        return output
    if normalize_weight:
        return _yoco_normalized_router_linear(hidden_states, weight)
    return F.linear(hidden_states, weight)


def _yoco_align_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if not hidden_states.is_cuda:
        return F.linear(hidden_states, weight, bias)
    if (
        hidden_states.ndim != 2
        or hidden_states.dtype != torch.bfloat16
        or not 0 < hidden_states.shape[0] <= 32
    ):
        return linear_batch_invariant(hidden_states, weight, bias)
    # Keep the same K=64 MMA traversal as linear_batch_invariant. A smaller
    # M tile avoids doing 128 rows of work for one decode token; B200 gates
    # require this launch to be bitwise equal to the large-M launch.
    m, k = hidden_states.shape
    assert weight.ndim == 2 and weight.shape[1] == k
    assert weight.dtype == hidden_states.dtype
    n = weight.shape[0]
    output = hidden_states.new_empty((m, n))
    sms = num_compute_units(hidden_states.device.index)
    grid = (min(sms, triton.cdiv(m, 16) * triton.cdiv(n, 128)),)
    matmul_kernel_persistent[grid](
        hidden_states,
        weight.t(),
        output,
        None,
        m,
        n,
        k,
        hidden_states.stride(0),
        hidden_states.stride(1),
        weight.stride(1),
        weight.stride(0),
        output.stride(0),
        output.stride(1),
        NUM_SMS=sms,
        A_LARGE=hidden_states.numel() > 2**31,
        B_LARGE=weight.numel() > 2**31,
        C_LARGE=output.numel() > 2**31,
        HAS_BIAS=False,
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
        num_stages=3,
        num_warps=4,
    )
    # Match the generic invariant linear's BF16 store before bias addition.
    if bias is not None:
        output = output + bias
    return output


class _YocoAlignLinearMethod(UnquantizedLinearMethod):
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _yoco_align_linear(x, layer.weight, bias)


class _YocoAlignEmbeddingMethod(UnquantizedEmbeddingMethod):
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _yoco_align_linear(x, layer.weight, bias)


class YOCORotaryEmbedding(nn.Module):
    """YOCO RoPE with llm-train's FP32 cos/sin cache semantics."""

    def __init__(
        self,
        head_size: int,
        max_position_embeddings: int,
        base: float,
        execution_mode: str = "fast",
    ) -> None:
        super().__init__()
        if execution_mode not in ("align", "fast"):
            raise ValueError(f"Unsupported YOCO execution mode: {execution_mode!r}")
        self.head_size = head_size
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.execution_mode = execution_mode
        self.register_buffer("cos_sin_cache", None, persistent=False)

    def _get_cos_sin_cache(self, device: torch.device) -> torch.Tensor:
        cache = self.cos_sin_cache
        if cache is not None and cache.device == device:
            return cache
        # llm-train constructs RoPE while the default device is CUDA. Generate
        # the FP32 cache on the execution device as well: CPU-generated trig
        # values differ by a few ULPs and can cross BF16/FP8 boundaries.
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(
                    0,
                    self.head_size,
                    2,
                    dtype=torch.float32,
                    device=device,
                )
                / self.head_size
            )
        )
        positions = torch.arange(
            self.max_position_embeddings,
            dtype=torch.float32,
            device=device,
        )
        freqs = torch.einsum("i,j -> ij", positions, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        self.cos_sin_cache = cache
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self._get_cos_sin_cache(query.device)
        query = query.view(query.shape[0], -1, self.head_size)
        key = key.view(key.shape[0], -1, self.head_size)
        positions = positions.to(device=query.device, dtype=torch.long)
        if self.execution_mode == "align":
            query, key = _yoco_align_rotary_embedding(cache, positions, query, key)
            return query.flatten(-2), key.flatten(-2)
        if (
            HAS_TRITON
            and query.is_cuda
            and query.dtype == torch.bfloat16
            and key.dtype == torch.bfloat16
            and query.stride(-1) == 1
            and key.stride(-1) == 1
            and self.head_size == 128
            and current_platform.is_cuda()
        ):
            query, key = torch.ops.vllm.yoco_rotary(query, key, positions, cache)
        else:
            cos_sin = cache.index_select(0, positions)
            cos, sin = cos_sin.chunk(2, dim=-1)
            query, key = _yoco_apply_rotary_emb(query, key, cos, sin)
        return query.flatten(-2), key.flatten(-2)


def _build_qk_norm(
    config: PretrainedConfig,
    head_dim: int,
    rms_eps: float,
    execution_mode: str = "fast",
):
    """Build the per-head Q/K normalization module for YOCO attention.

    Three mutually exclusive modes, matching training (``llm/arch/attention.py``):
    * ``qk_rms_clip=True``  -> :class:`RMSClip` (clips outliers).
    * ``qk_norm=True``      -> :class:`RMSNorm`.
    * otherwise             -> ``None`` (no Q/K norm).
    """
    # Older exported YOCO-v2 configs omitted this field and were weight-free.
    has_weight = bool(getattr(config, "qk_rms_gamma", False))
    if bool(getattr(config, "qk_rms_clip", False)):
        limit = float(getattr(config, "qk_rms_limit", 3.0))
        return RMSClip(
            head_dim,
            eps=rms_eps,
            limit=limit,
            has_weight=has_weight,
            execution_mode=execution_mode,
        )
    if bool(getattr(config, "qk_norm", False)):
        return RMSNorm(
            head_dim,
            eps=rms_eps,
            has_weight=has_weight,
            execution_mode=execution_mode,
        )
    return None


def _swiglu_limit(config: PretrainedConfig) -> float:
    """SwiGLU clamp limit, matching training (``swiglu_limit``, default 10.0).

    Training clamps ``gate`` to ``max=limit`` and ``up`` to ``[-limit, limit]``
    before ``silu(gate) * up`` (see ``llm/arch/ffn.py`` and
    ``llm/arch/all2all_moe.py``).
    """
    return float(getattr(config, "swiglu_limit", 10.0))


def _apply_per_head_norm(
    x: torch.Tensor, num_heads: int, head_dim: int, norm: nn.Module
) -> torch.Tensor:
    """Apply ``norm`` independently to each head's ``head_dim`` slice.

    ``x`` has shape ``(n_tokens, num_heads * head_dim)``.
    """
    x = x.unflatten(-1, (num_heads, head_dim))
    x = norm(x)
    return x.flatten(-2, -1)


# --------------------------------------------------------------------------- #
# Self-attention (sliding window, QK-norm, RoPE, diff-attention)              #
# --------------------------------------------------------------------------- #


def _yoco_align_qkv_linear(
    hidden_states: torch.Tensor,
    packed_weight: torch.Tensor,
    q_size: int,
    kv_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run llm-train's three independent BF16 Q/K/V projections."""
    expected_rows = q_size + 2 * kv_size
    if packed_weight.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} packed QKV rows, got {packed_weight.shape[0]}"
        )
    hidden_states = hidden_states.to(torch.bfloat16)
    packed_weight = packed_weight.to(torch.bfloat16)
    q_weight, k_weight, v_weight = packed_weight.split(
        (q_size, kv_size, kv_size), dim=0
    )
    return (
        _yoco_align_linear(hidden_states, q_weight),
        _yoco_align_linear(hidden_states, k_weight),
        _yoco_align_linear(hidden_states, v_weight),
    )


class YOCOSelfAttention(nn.Module):
    """Sliding-window self-attention for YOCO layers 0..9.

    Creates ``universal_loop`` distinct ``Attention`` sub-modules that share
    the projection weights but use unique KV cache prefixes so each universal
    loop iteration gets its own KV cache slot.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        universal_loop: int,
        num_hidden_layers: int,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
        execution_mode: str = "fast",
    ) -> None:
        super().__init__()
        if execution_mode not in ("align", "fast"):
            raise ValueError(f"Unsupported YOCO execution mode: {execution_mode!r}")
        if execution_mode == "align" and quant_config is not None:
            raise ValueError("YOCO --align currently supports BF16 weights only")
        self.execution_mode = execution_mode
        self.use_sm100_diff_v3_kernel = _supports_yoco_sm100_diff_v3_kernel(
            execution_mode
        )
        self.hidden_size = _cfg_int(config, "hidden_size", "d_model")
        self.total_num_heads = _cfg_int(config, "num_attention_heads", "head")
        self.total_num_kv_heads = _cfg_int(config, "num_key_value_heads", "kv_head")
        self.head_dim = _cfg_int(config, "head_dim")
        self.diff_v3 = bool(getattr(config, "diff_v3", False))
        self.layer_idx = layer_idx
        self.universal_loop = universal_loop
        self.num_hidden_layers = num_hidden_layers
        self.training_sliding_window = _cfg_int(
            config, "sliding_window_size", "yoco_window_size", default=512
        )
        self.sliding_window = _yoco_runtime_sliding_window(self.training_sliding_window)
        max_position = _cfg_int(config, "max_position_embeddings", "max_seq_len")
        rope_theta = float(getattr(config, "rope_theta", 10000.0))

        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0, (
            f"num_attention_heads={self.total_num_heads} must be divisible "
            f"by TP size {tp_size} so diff-attention head pairs stay local"
        )
        # ``2 * head`` Q-heads because of diff-attention.
        q_heads = 2 * self.total_num_heads
        assert q_heads % tp_size == 0, (
            f"2*num_attention_heads={q_heads} must be divisible by TP size {tp_size}"
        )
        assert (
            self.total_num_kv_heads % tp_size == 0
            or tp_size % self.total_num_kv_heads == 0
        ), (
            f"num_kv_heads={self.total_num_kv_heads} must be divisible by "
            f"or divide TP size {tp_size}"
        )
        self.num_heads = q_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        gate_heads = (2 if self.diff_v3 else 1) * self.total_num_heads
        assert gate_heads % tp_size == 0, (
            f"gate_heads={gate_heads} must be divisible by TP size {tp_size}"
        )
        self.num_lambda_heads = self.total_num_heads // tp_size
        self.num_gate_heads = gate_heads // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        use_merged_qkv_lambda = (
            execution_mode == "fast"
            and tp_size == 1
            and quant_config is None
            and torch.get_default_dtype() == torch.bfloat16
        )
        if use_merged_qkv_lambda:
            self.qkv_lambda_proj = MergedColumnParallelLinear(
                input_size=self.hidden_size,
                output_sizes=[
                    q_heads * self.head_dim,
                    self.total_num_kv_heads * self.head_dim,
                    self.total_num_kv_heads * self.head_dim,
                    gate_heads,
                ],
                bias=False,
                gather_output=False,
                quant_config=None,
                prefix=f"{prefix}.qkv_lambda_proj",
            )
            self.qkv_proj = None
            self.lambda_proj = None
        else:
            self.qkv_lambda_proj = None
            self.qkv_proj = QKVParallelLinear(
                hidden_size=self.hidden_size,
                head_size=self.head_dim,
                total_num_heads=q_heads,
                total_num_kv_heads=self.total_num_kv_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            # Keep the legacy HF name ``lambda_proj`` for checkpoint
            # compatibility. Diff-v2 has one gate per head pair; diff-v3 has
            # one per attention head.
            self.lambda_proj = ColumnParallelLinear(
                input_size=self.hidden_size,
                output_size=gate_heads,
                bias=False,
                gather_output=False,
                # llm-train constructs lambda_proj with default
                # MixPrecisionLinear, so it stays BF16 even when the rest of
                # attention uses MXFP8.
                quant_config=None,
                prefix=f"{prefix}.lambda_proj",
            )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        rms_eps = float(
            getattr(config, "rms_norm_eps", getattr(config, "norm_eps", 1e-6))
        )
        # Per-head Q/K normalization (RMSClip when ``qk_rms_clip``, RMSNorm
        # when ``qk_norm``, else nothing). L3 uses affine RMSClip weights.
        self.q_norm = _build_qk_norm(config, self.head_dim, rms_eps, execution_mode)
        self.k_norm = _build_qk_norm(config, self.head_dim, rms_eps, execution_mode)

        self.rotary_emb = YOCORotaryEmbedding(
            head_size=self.head_dim,
            max_position_embeddings=max_position,
            base=rope_theta,
            execution_mode=execution_mode,
        )

        # Build one Attention module per universal-loop iteration.  Each gets
        # a unique cache prefix so the runtime allocates a distinct KV cache.
        self.attn = nn.ModuleList()
        for loop_idx in range(universal_loop):
            unique_layer_idx = loop_idx * num_hidden_layers + layer_idx
            unique_prefix = prefix.replace(
                f"layers.{layer_idx}", f"layers.{unique_layer_idx}"
            )
            self.attn.append(
                Attention(
                    num_heads=self.num_heads,
                    head_size=self.head_dim,
                    scale=self.scaling,
                    num_kv_heads=self.num_kv_heads,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    per_layer_sliding_window=self.sliding_window,
                    attn_type=AttentionType.DECODER,
                    prefix=f"{unique_prefix}.attn",
                )
            )

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    def _diff_attention_combine(
        self,
        attn_out: torch.Tensor,
        gate: torch.Tensor,
        num_heads_per_pair: int,
    ) -> torch.Tensor:
        """Combine the 2*head attention output via the diff-attention rule.

        Output shape: ``(n_tokens, num_heads_per_pair * head_dim)`` (i.e. the
        pre-o_proj hidden slice owned by this TP rank).
        """
        # (n_tokens, 2 * num_heads_per_pair, head_dim)
        attn_view = attn_out.view(-1, 2 * num_heads_per_pair, self.head_dim)
        if self.diff_v3:
            out = _yoco_diff_attention_v3_dispatch(
                attn_view, gate, self.use_sm100_diff_v3_kernel
            )
        else:
            attn1 = attn_view[:, 0::2, :]
            attn2 = attn_view[:, 1::2, :]
            out = _yoco_diff_attention_v2(attn1, attn2, gate)
        return out.reshape(-1, num_heads_per_pair * self.head_dim)

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #
    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.execution_mode == "align":
            assert self.qkv_proj is not None
            q, k, v = _yoco_align_qkv_linear(
                hidden_states,
                self.qkv_proj.weight,
                self.q_size,
                self.kv_size,
            )
            return q, k, v, None

        if self.qkv_lambda_proj is None:
            assert self.qkv_proj is not None
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            return q, k, v, None

        qkv_size = self.q_size + 2 * self.kv_size
        if hidden_states.shape[0] <= _YOCO_QKV_LAMBDA_MERGED_MAX_TOKENS:
            qkv_lambda, _ = self.qkv_lambda_proj(hidden_states)
            q, k, v, gate = qkv_lambda.split(
                [
                    self.q_size,
                    self.kv_size,
                    self.kv_size,
                    self.num_gate_heads,
                ],
                dim=-1,
            )
            return q, k, v, gate

        # B200's merged L3 N=10304 GEMM wins through M=4096, but regresses
        # beyond that. Reuse the packed parameter's QKV prefix without keeping
        # a duplicate QKV weight.
        qkv_weight = self.qkv_lambda_proj.weight.narrow(0, 0, qkv_size)
        qkv = F.linear(hidden_states, qkv_weight)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v, None

    def _project_lambda(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lambda_proj is not None:
            gate, _ = self.lambda_proj(hidden_states)
            return gate
        assert self.qkv_lambda_proj is not None
        qkv_size = self.q_size + 2 * self.kv_size
        lambda_weight = self.qkv_lambda_proj.weight.narrow(
            0,
            qkv_size,
            self.num_gate_heads,
        )
        return F.linear(hidden_states, lambda_weight)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        loop_idx: int,
    ) -> torch.Tensor:
        q, k, v, gate = self._project_qkv(hidden_states)
        # Per-head QK norm/clip on the un-rotated query/key, applied
        # independently to each head's ``head_dim`` slice — matches training's
        # ``q_norm``/``k_norm``.  Skipped entirely when neither ``qk_rms_clip``
        # nor ``qk_norm`` is set.
        qk_clip_weight_mode_matches = (
            isinstance(self.q_norm, RMSClip)
            and isinstance(self.k_norm, RMSClip)
            and (self.q_norm.weight is None) == (self.k_norm.weight is None)
        )
        can_fuse_qk_clip_rotary = (
            self.execution_mode == "fast"
            and HAS_TRITON
            and q.is_cuda
            and q.dtype == torch.bfloat16
            and k.dtype == torch.bfloat16
            and self.head_dim == 128
            and isinstance(self.q_norm, RMSClip)
            and isinstance(self.k_norm, RMSClip)
            and qk_clip_weight_mode_matches
            and self.q_norm.eps == self.k_norm.eps
            and self.q_norm.limit == self.k_norm.limit
            and current_platform.is_cuda()
        )
        if can_fuse_qk_clip_rotary:
            cache = self.rotary_emb._get_cos_sin_cache(q.device)
            q = q.view(q.shape[0], self.num_heads, self.head_dim)
            k = k.view(k.shape[0], self.num_kv_heads, self.head_dim)
            positions = positions.to(device=q.device, dtype=torch.long)
            if self.q_norm.weight is None:
                q, k = torch.ops.vllm.yoco_qk_rms_clip_rotary(
                    q,
                    k,
                    positions,
                    cache,
                    self.q_norm.eps,
                    self.q_norm.limit,
                )
            else:
                assert self.k_norm.weight is not None
                q, k = torch.ops.vllm.yoco_qk_rms_clip_rotary_weighted(
                    q,
                    k,
                    self.q_norm.weight,
                    self.k_norm.weight,
                    positions,
                    cache,
                    self.q_norm.eps,
                    self.q_norm.limit,
                )
            q = q.flatten(-2)
            k = k.flatten(-2)
        else:
            if self.q_norm is not None:
                q = _apply_per_head_norm(q, self.num_heads, self.head_dim, self.q_norm)
                k = _apply_per_head_norm(
                    k, self.num_kv_heads, self.head_dim, self.k_norm
                )
            q, k = self.rotary_emb(positions, q, k)
        attn_out = self.attn[loop_idx](q, k, v)

        if gate is None:
            gate = self._project_lambda(hidden_states)
        out = self._diff_attention_combine(attn_out, gate, self.num_lambda_heads)
        out, _ = self.o_proj(out)
        return out


# --------------------------------------------------------------------------- #
# Cross-attention (NoPE, no QK-norm, shared KV via kv_sharing)                #
# --------------------------------------------------------------------------- #


class YOCOCrossAttention(nn.Module):
    """YOCO cross-attention layer (layers 10..19).

    These layers have only ``q_proj`` / ``o_proj`` / ``lambda_proj`` — they
    share a single set of (K, V) produced once at the model level.  Layer 10
    owns the shared KV cache; subsequent cross-layers point their
    ``kv_sharing_target_layer_name`` at layer 10's attention to reuse the
    cache without writing.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        first_cross_layer_idx: int,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
        execution_mode: str = "fast",
    ) -> None:
        super().__init__()
        self.execution_mode = execution_mode
        self.use_sm100_diff_v3_kernel = _supports_yoco_sm100_diff_v3_kernel(
            execution_mode
        )
        self.use_sm100_weighted_rms_clip_kernel = (
            self.use_sm100_diff_v3_kernel and quant_config is None
        )
        self.hidden_size = _cfg_int(config, "hidden_size", "d_model")
        # Cross-attention has its OWN Q-head count via ``cross_head``.  In this
        # checkpoint ``cross_head = 48`` (twice the self-attention head count)
        # and the q_proj output is ``2 * cross_head * head_dim = 12288``.
        # ``cross_kv_head`` defaults to ``kv_head`` (= 4 here).
        self.total_num_heads = _cfg_int(config, "cross_head", "head")
        self.total_num_kv_heads = _cfg_int(
            config, "cross_kv_head", "num_key_value_heads", "kv_head"
        )
        self.head_dim = _cfg_int(config, "head_dim")
        self.diff_v3 = bool(getattr(config, "diff_v3", False))
        self.layer_idx = layer_idx
        self.first_cross_layer_idx = first_cross_layer_idx

        tp_size = get_tensor_model_parallel_world_size()
        q_heads = 2 * self.total_num_heads
        assert q_heads % tp_size == 0
        assert self.total_num_heads % tp_size == 0
        self.num_heads = q_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.num_lambda_heads = self.total_num_heads // tp_size
        gate_heads = (2 if self.diff_v3 else 1) * self.total_num_heads
        assert gate_heads % tp_size == 0
        self.num_gate_heads = gate_heads // tp_size
        self.scaling = self.head_dim**-0.5

        # NoPE on cross layers — the checkpoint has ``rope_dim = 0``.
        self.q_size = self.num_heads * self.head_dim
        use_merged_q_lambda = (
            execution_mode == "fast" and tp_size == 1 and quant_config is None
        )
        if use_merged_q_lambda:
            self.q_lambda_proj = MergedColumnParallelLinear(
                input_size=self.hidden_size,
                output_sizes=[q_heads * self.head_dim, gate_heads],
                bias=False,
                gather_output=False,
                quant_config=None,
                prefix=f"{prefix}.q_lambda_proj",
            )
            self.q_proj = None
            self.lambda_proj = None
        else:
            self.q_lambda_proj = None
            self.q_proj = ColumnParallelLinear(
                input_size=self.hidden_size,
                output_size=q_heads * self.head_dim,
                bias=False,
                gather_output=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
            self.lambda_proj = ColumnParallelLinear(
                input_size=self.hidden_size,
                output_size=gate_heads,
                bias=False,
                gather_output=False,
                # llm-train leaves lambda_proj at default BF16 precision.
                quant_config=None,
                prefix=f"{prefix}.lambda_proj",
            )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        rms_eps = float(
            getattr(config, "rms_norm_eps", getattr(config, "norm_eps", 1e-6))
        )
        # Cross layers apply the same per-head Q norm/clip as self layers (the
        # shared K is normed once at the model level on ``yoco_key``).
        self.q_norm = _build_qk_norm(config, self.head_dim, rms_eps, execution_mode)

        if layer_idx == first_cross_layer_idx:
            kv_sharing_target = None
        else:
            # Point at layer 10's attention.  ``prefix`` looks like
            # ``model.layers.{i}.self_attn`` so we substitute to layer 10.
            owner_prefix = prefix.replace(
                f"layers.{layer_idx}", f"layers.{first_cross_layer_idx}"
            )
            kv_sharing_target = f"{owner_prefix}.attn"

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=kv_sharing_target,
            prefix=f"{prefix}.attn",
        )

    def _diff_attention_combine(
        self, attn_out: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        attn_view = attn_out.view(-1, 2 * self.num_lambda_heads, self.head_dim)
        if self.diff_v3:
            out = _yoco_diff_attention_v3_dispatch(
                attn_view, gate, self.use_sm100_diff_v3_kernel
            )
        else:
            attn1 = attn_view[:, 0::2, :]
            attn2 = attn_view[:, 1::2, :]
            out = _yoco_diff_attention_v2(attn1, attn2, gate)
        return out.reshape(-1, self.num_lambda_heads * self.head_dim)

    def _project_query(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.q_lambda_proj is None:
            assert self.q_proj is not None
            q, _ = self.q_proj(hidden_states)
            return q, None

        if hidden_states.shape[0] <= _YOCO_Q_LAMBDA_MERGED_MAX_TOKENS:
            q_lambda, _ = self.q_lambda_proj(hidden_states)
            q, gate = q_lambda.split(
                [self.q_size, self.num_gate_heads],
                dim=-1,
            )
            return q, gate

        # B200's merged N=8256 GEMM wins through M=2048, but can regress at
        # larger prefill shapes (notably M=4096). Reuse the packed parameter
        # as two contiguous views and preserve the original projection order.
        q_weight = self.q_lambda_proj.weight.narrow(0, 0, self.q_size)
        return F.linear(hidden_states, q_weight), None

    def _project_lambda(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lambda_proj is not None:
            gate, _ = self.lambda_proj(hidden_states)
            return gate
        assert self.q_lambda_proj is not None
        lambda_weight = self.q_lambda_proj.weight.narrow(
            0,
            self.q_size,
            self.num_gate_heads,
        )
        return F.linear(hidden_states, lambda_weight)

    def _normalize_query(self, q: torch.Tensor) -> torch.Tensor:
        if self.q_norm is None:
            return q
        can_use_sm100_weighted_rms_clip = (
            self.use_sm100_weighted_rms_clip_kernel
            and isinstance(self.q_norm, RMSClip)
            and self.q_norm.weight is not None
            and q.is_cuda
            and q.dtype == torch.bfloat16
            and self.q_norm.weight.dtype == torch.bfloat16
            and q.ndim == 2
            and q.shape[-1] == 64 * 128
            and self.num_heads == 64
            and self.head_dim == 128
            and q.stride(-1) == 1
        )
        if can_use_sm100_weighted_rms_clip:
            q_view = q.unflatten(-1, (self.num_heads, self.head_dim))
            return torch.ops.vllm.yoco_weighted_rms_clip(
                q_view,
                self.q_norm.weight,
                self.q_norm.eps,
                self.q_norm.limit,
            ).flatten(-2)
        return _apply_per_head_norm(q, self.num_heads, self.head_dim, self.q_norm)

    def forward(
        self,
        hidden_states: torch.Tensor,
        yoco_key: torch.Tensor,
        yoco_value: torch.Tensor,
        kv_cache_dummy_dep: torch.Tensor | None = None,
        skip_kv_cache_update: bool = False,
    ) -> torch.Tensor:
        q, gate = self._project_query(hidden_states)
        q = self._normalize_query(q)
        attn_out = self.attn(
            q,
            yoco_key,
            yoco_value,
            kv_cache_dummy_dep=kv_cache_dummy_dep,
            skip_kv_cache_update=skip_kv_cache_update,
        )
        if gate is None:
            gate = self._project_lambda(hidden_states)
        out = self._diff_attention_combine(attn_out, gate)
        out, _ = self.o_proj(out)
        return out


# --------------------------------------------------------------------------- #
# MoE block                                                                   #
# --------------------------------------------------------------------------- #


@torch.compile
def _yoco_align_shared_expert_swiglu(
    up: torch.Tensor,
    gate: torch.Tensor,
    swiglu_limit: float,
) -> torch.Tensor:
    """Mirror llm-train's compiled shared-expert SwiGLU expression."""
    gate = gate.clamp(max=swiglu_limit)
    up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    return up * F.silu(gate)


@torch.compile
def _yoco_align_shared_expert_swiglu_unclamped(
    up: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    return up * F.silu(gate)


class YOCOSharedExperts(nn.Module):
    """Shared-expert MLP for YOCO MoE blocks (SwiGLU)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None,
        reduce_results: bool,
        prefix: str,
        swiglu_limit: float = 10.0,
        execution_mode: str = "fast",
    ) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size
        # llm-train evaluates two independent BF16 projections in this order:
        # ``up_proj(x)``, then ``gate_proj(x)``.  A merged-N cuBLAS GEMM can
        # select a different reduction kernel and is therefore not a strict
        # numerical substitute.  Align reuses contiguous views of the packed
        # checkpoint parameter but restores the two original GEMM boundaries.
        tp_size = get_tensor_model_parallel_world_size()
        self.use_separate_projection = (
            execution_mode == "align" and quant_config is None and tp_size == 1
        )
        self.use_fast_down_transpose = (
            execution_mode == "fast"
            and quant_config is None
            and tp_size == 1
            and hidden_size == _YOCO_L3_HIDDEN_SIZE
            and intermediate_size == 1280
        )
        self.register_buffer("_fast_down_weight_t", None, persistent=False)
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        # Clamped SwiGLU to match training (``swiglu_limit``).  When the limit is
        # non-positive, fall back to the plain (unclamped) activation.
        self.swiglu_limit = float(swiglu_limit)
        if self.swiglu_limit > 0:
            self.act_fn = SiluAndMulWithClampFP32(
                self.swiglu_limit,
                enforce_enable=True,
            )
        else:
            self.act_fn = SiluAndMul()

    def initialize_fast_weight_cache(self) -> None:
        """Cache B200's faster M=1 down-projection operand layout."""
        self._fast_down_weight_t = None
        if not self.use_fast_down_transpose:
            return
        weight = self.down_proj.weight
        if not weight.is_cuda or weight.dtype != torch.bfloat16:
            return
        capability = torch.cuda.get_device_capability(weight.device)
        if capability[0] != 10:
            return
        with torch.no_grad():
            self._fast_down_weight_t = weight.t().contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_separate_projection:
            weight = self.gate_up_proj.weight
            gate_weight = weight.narrow(0, 0, self.intermediate_size)
            up_weight = weight.narrow(
                0,
                self.intermediate_size,
                self.intermediate_size,
            )
            # Python evaluates llm-train's ``swiglu(up_proj(x), gate_proj(x))``
            # arguments left-to-right. Keep that order as well as each BF16
            # GEMM store before the compiled FP32 activation expression.
            up = _yoco_align_linear(x, up_weight)
            gate = _yoco_align_linear(x, gate_weight)
            if self.swiglu_limit > 0:
                x = _yoco_align_shared_expert_swiglu(
                    up,
                    gate,
                    self.swiglu_limit,
                )
            else:
                x = _yoco_align_shared_expert_swiglu_unclamped(up, gate)
        else:
            gate_up, _ = self.gate_up_proj(x)
            x = self.act_fn(gate_up)
        if self._fast_down_weight_t is not None and x.shape[0] == 1:
            x = torch.mm(x, self._fast_down_weight_t)
        else:
            x, _ = self.down_proj(x)
        return x


class YOCOLatentInputTransform(nn.Module):
    """Project and normalize only the routed-expert input."""

    def __init__(self, proj: nn.Module, norm: nn.Module) -> None:
        super().__init__()
        self.proj = proj
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.proj(x)
        if isinstance(projected, tuple):
            projected = projected[0]
        return self.norm(projected)


class YOCOLatentOutputTransform(nn.Module):
    """Normalize and project only the routed-expert output."""

    def __init__(self, norm: nn.Module, proj: nn.Module) -> None:
        super().__init__()
        self.norm = norm
        self.proj = proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.proj(self.norm(x))
        return projected[0] if isinstance(projected, tuple) else projected


class YOCOCombinedOutputTransform(nn.Module):
    """Apply YOCO's shared gate and combine reduced shared+routed outputs."""

    def __init__(
        self,
        shared_gate: ReplicatedLinear,
        execution_mode: str = "fast",
    ) -> None:
        super().__init__()
        if execution_mode not in ("align", "fast"):
            raise ValueError(f"Unsupported YOCO execution mode: {execution_mode!r}")
        self.shared_gate = shared_gate
        self.execution_mode = execution_mode

    def forward(
        self,
        shared_output: torch.Tensor,
        routed_output: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.execution_mode == "fast"
            and HAS_TRITON
            and current_platform.is_cuda()
            and shared_output.is_cuda
            and routed_output.is_cuda
            and hidden_states.is_cuda
            and shared_output.dtype == torch.bfloat16
            and routed_output.dtype == torch.bfloat16
            and hidden_states.dtype == torch.bfloat16
            and shared_output.shape == routed_output.shape == hidden_states.shape
            and shared_output.shape[-1] == 3072
            and shared_output.is_contiguous()
            and routed_output.is_contiguous()
            and hidden_states.is_contiguous()
        ):
            return torch.ops.vllm.yoco_fused_shared_gate_moe_output(
                shared_output,
                routed_output,
                hidden_states,
                self.shared_gate.weight,
            )

        # llm-train builds shared_gate with default MixPrecisionLinear settings:
        # no MXFP8 path and the parameter follows the module default dtype.
        linear = _yoco_align_linear if self.execution_mode == "align" else F.linear
        scale = linear(
            hidden_states,
            self.shared_gate.weight.to(hidden_states.dtype),
        )
        gated_shared = torch.sigmoid(scale) * shared_output
        # Keep the same operand order as llm-train's
        # ``final_hidden_states + shared_gate_score * self.shared(x)``.
        return routed_output + gated_shared


class YOCOMoE(nn.Module):
    """YOCO MoE block: routed top-k + gated shared expert + final reduce."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
        execution_mode: str = "fast",
        moe_backend_override: MoEBackend | None = None,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.execution_mode = execution_mode
        self.hidden_size = _cfg_int(config, "hidden_size", "d_model")
        self.num_experts = _cfg_int(config, "num_experts", "moe_expert_num")
        self.top_k = _cfg_int(config, "num_experts_per_tok", "moe_top_k", "top_k")
        self.moe_intermediate_size = _cfg_int(
            config, "moe_intermediate_size", "moe_ffn_dim"
        )
        self.moe_latent_dim = _cfg_int(config, "moe_latent_dim", default=0)
        self.moe_latent_norm = bool(getattr(config, "moe_latent_norm", False))
        self.shared_intermediate_size = _cfg_int(
            config, "shared_expert_intermediate_size", "d_shared_expert"
        )
        self.swiglu_limit = _swiglu_limit(config)
        self.router_weights_normalized = bool(
            getattr(config, "router_weights_normalized", False)
        )
        num_hidden_layers = _cfg_int(config, "num_hidden_layers", "n_layers")
        yoco_cross_layers = _cfg_int(config, "yoco_cross_layers", default=0)
        self._yoco_logical_route_info = (
            (
                layer_idx,
                num_hidden_layers - yoco_cross_layers,
                _cfg_int(config, "universal_loop", default=1),
            )
            if layer_idx is not None
            else None
        )
        # Older YOCO checkpoints keep the raw router weights and normalize
        # them at inference time. The weights are immutable after loading, so
        # cache that normalized FP32 tensor once instead of rebuilding it in
        # every decoder-block execution. Keep it non-persistent so checkpoint
        # names and serialization remain unchanged.
        self.register_buffer("_normalized_gate_weight", None, persistent=False)

        # Router gate — runs in fp32 to match training.
        self.gate = GateLinear(
            input_size=self.hidden_size,
            output_size=self.num_experts,
            bias=False,
            params_dtype=torch.float32,
            force_fp32_compute=True,
            prefix=f"{prefix}.gate",
        )
        self.gate.set_out_dtype(torch.float32)

        # Keep the shared output local so FusedMoE can execute these GEMMs on its
        # auxiliary stream. The runner still performs a separate TP all-reduce
        # after the routed reduction to preserve YOCO's numerical boundaries.
        self.shared_experts = YOCOSharedExperts(
            hidden_size=self.hidden_size,
            intermediate_size=self.shared_intermediate_size,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
            swiglu_limit=self.swiglu_limit,
            execution_mode=execution_mode,
        )

        # Scalar shared-expert sigmoid gate.  Replicated across TP — every
        # rank computes the same per-token scaling factor.
        self.shared_gate = ReplicatedLinear(
            input_size=self.hidden_size,
            output_size=1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_gate",
        )

        expert_hidden_size = self.moe_latent_dim or self.hidden_size
        if self.moe_latent_dim:
            rms_eps = float(
                getattr(config, "rms_norm_eps", getattr(config, "norm_eps", 1e-6))
            )
            self.fc1_latent_proj = ReplicatedLinear(
                input_size=self.hidden_size,
                output_size=self.moe_latent_dim,
                bias=False,
                # llm-train does not forward MoE.quant_mode to either latent
                # projection, so MixPrecisionLinear keeps its BF16 default
                # even when the routed experts use MXFP8.
                quant_config=None,
                prefix=f"{prefix}.fc1_latent_proj",
                return_bias=False,
            )
            self.fc2_latent_proj = ReplicatedLinear(
                input_size=self.moe_latent_dim,
                output_size=self.hidden_size,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.fc2_latent_proj",
                return_bias=False,
            )
            self.fc1_latent_norm = (
                RMSNorm(
                    self.moe_latent_dim,
                    eps=rms_eps,
                    execution_mode=execution_mode,
                )
                if self.moe_latent_norm
                else nn.Identity()
            )
            self.fc2_latent_norm = (
                RMSNorm(
                    self.moe_latent_dim,
                    eps=rms_eps,
                    execution_mode=execution_mode,
                )
                if self.moe_latent_norm
                else nn.Identity()
            )
        else:
            self.fc1_latent_proj = None
            self.fc2_latent_proj = None
            self.fc1_latent_norm = None
            self.fc2_latent_norm = None

        routed_input_transform = (
            YOCOLatentInputTransform(self.fc1_latent_proj, self.fc1_latent_norm)
            if self.fc1_latent_proj is not None and self.fc1_latent_norm is not None
            else None
        )
        routed_output_transform = (
            YOCOLatentOutputTransform(self.fc2_latent_norm, self.fc2_latent_proj)
            if self.fc2_latent_proj is not None and self.fc2_latent_norm is not None
            else None
        )
        combined_output_transform = YOCOCombinedOutputTransform(
            self.shared_gate,
            execution_mode=execution_mode,
        )

        # NOTE(swiglu_limit): Both the shared expert (above, via the fused FP32
        # clamped SwiGLU op) and the ROUTED experts (below) apply the training
        # ``swiglu_limit`` clamp (clamp-before-silu), for exact train/inference
        # parity. In align mode, the routed clamp and FP32 routing probability
        # are fused before the BF16 store and W2 GEMM, matching llm-train's
        # ``fused_silu`` rounding boundary. Fast BF16 keeps the cheaper
        # mathematically equivalent W2-epilogue weighting. The routed clamp is
        # threaded through FusedMoE into the modular experts. Only align mode
        # dispatches TritonExperts to the isolated ``yoco_weighted_swiglu``
        # kernel; other models retain the common activation path. The W8A8
        # parity path uses DeepGEMM, which also
        # applies routing probabilities before W2 input quantization. CAUTION:
        # the loose limit=10.0
        # can make some checkpoints (observed: adamw-3000) degenerate under pure
        # greedy decoding; use temperature>0 in production. Kept on per owner's
        # request for training fidelity.
        routing_function = (
            _yoco_align_topk_routing
            if execution_mode == "align"
            else _yoco_topk_routing
        )
        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=expert_hidden_size,
            intermediate_size=self.moe_intermediate_size,
            renormalize=True,
            quant_config=quant_config,
            use_grouped_topk=False,
            scoring_func="softmax",
            custom_routing_function=routing_function,
            swiglu_limit=self.swiglu_limit,
            apply_router_weight_before_w2=True,
            routed_input_transform=routed_input_transform,
            routed_output_transform=routed_output_transform,
            combined_output_transform=combined_output_transform,
            reduce_shared_experts_separately=True,
            use_tuned_config=execution_mode == "fast",
            moe_backend_override=moe_backend_override,
            prefix=f"{prefix}.experts",
        )
        # Keep the training-specific rounding boundary out of the common MoE
        # path. The modular Triton backend reads this only for YOCO align.
        self.experts.yoco_align_weighted_swiglu = execution_mode == "align"
        # DeepGEMM chooses shape-dependent tiles. Align uses fixed Triton W2.
        self.experts.yoco_align_deep_gemm_w2 = False
        self.experts.yoco_separate_w2_config = execution_mode == "fast"
        self.experts.yoco_fast_w13_config = execution_mode == "fast"
        self.experts.yoco_triton_fallback_max_tokens = (
            1 if execution_mode == "fast" else 0
        )
        self.experts.yoco_align_moe_sum = execution_mode == "align"
        self.experts.yoco_fast_moe_sum = execution_mode == "fast"

    def initialize_router_weight_cache(self) -> None:
        if self.execution_mode == "align" or self.router_weights_normalized:
            self._normalized_gate_weight = None
            return
        with torch.no_grad():
            weight = self.gate.weight
            self._normalized_gate_weight = weight / weight.norm(
                dim=1, keepdim=True
            ).clamp_min(1e-6)

    def forward(self, hidden_states: torch.Tensor, loop_idx: int = 0) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        gate_weight = self.gate.weight
        normalize_gate_weight = not self.router_weights_normalized
        if self._normalized_gate_weight is not None:
            gate_weight = self._normalized_gate_weight
            normalize_gate_weight = False

        if self.execution_mode == "align":
            router_logits = _yoco_align_router_linear(
                hidden_states.float(), gate_weight, normalize_gate_weight
            )
        elif hidden_states.is_cuda and current_platform.is_cuda():
            router_logits = torch.ops.vllm.yoco_router_linear_tf32(
                hidden_states.float(),
                gate_weight,
                normalize_gate_weight,
            )
        else:
            if not normalize_gate_weight:
                router_logits = F.linear(hidden_states.float(), gate_weight)
            else:
                router_logits = _yoco_normalized_router_linear(
                    hidden_states.float(), gate_weight
                )
        _maybe_dump_yoco_logical_routes(
            hidden_states,
            router_logits,
            self.top_k,
            self._yoco_logical_route_info,
            loop_idx,
        )
        # FusedMoE overlaps the local shared-expert GEMMs with routed dispatch
        # and expert compute. It then preserves YOCO's original order: routed
        # TP reduction, shared TP reduction, latent output transform, sigmoid
        # shared gate, and finally the sum.
        final = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        return final.view(num_tokens, hidden_dim)


# --------------------------------------------------------------------------- #
# Decoder layer                                                               #
# --------------------------------------------------------------------------- #


class YOCODecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
        execution_mode: str = "fast",
        moe_backend_override: MoEBackend | None = None,
    ) -> None:
        super().__init__()
        hidden_size = _cfg_int(config, "hidden_size", "d_model")
        num_hidden_layers = _cfg_int(config, "num_hidden_layers", "n_layers")
        universal_loop = _cfg_int(config, "universal_loop", default=1)
        yoco_cross_layers = _cfg_int(config, "yoco_cross_layers", default=0)
        first_cross_layer_idx = num_hidden_layers - yoco_cross_layers
        rms_eps = float(
            getattr(config, "rms_norm_eps", getattr(config, "norm_eps", 1e-6))
        )

        self.layer_idx = layer_idx
        self.is_self_layer = layer_idx < first_cross_layer_idx
        self.input_layernorm = RMSNorm(
            hidden_size,
            eps=rms_eps,
            execution_mode=execution_mode,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            eps=rms_eps,
            execution_mode=execution_mode,
        )

        if self.is_self_layer:
            self.self_attn = YOCOSelfAttention(
                config=config,
                layer_idx=layer_idx,
                universal_loop=universal_loop,
                num_hidden_layers=num_hidden_layers,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                execution_mode=execution_mode,
            )
        else:
            self.self_attn = YOCOCrossAttention(
                config=config,
                layer_idx=layer_idx,
                first_cross_layer_idx=first_cross_layer_idx,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                execution_mode=execution_mode,
            )

        # All layers in this checkpoint are MoE (``dense_layers = 0``).
        self.mlp = YOCOMoE(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            execution_mode=execution_mode,
            moe_backend_override=moe_backend_override,
            layer_idx=layer_idx,
        )

    def forward_with_residual(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        loop_idx: int,
        yoco_key: torch.Tensor | None,
        yoco_value: torch.Tensor | None,
        kv_cache_dummy_dep: torch.Tensor | None = None,
        skip_kv_cache_update: bool = False,
        input_residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_residual is None:
            residual = hidden_states
            x = self.input_layernorm(hidden_states)
            assert isinstance(x, torch.Tensor)
        else:
            norm_output = self.input_layernorm(hidden_states, input_residual)
            assert isinstance(norm_output, tuple)
            x, residual = norm_output
        if self.is_self_layer:
            x = self.self_attn(positions, x, loop_idx)
        else:
            assert yoco_key is not None and yoco_value is not None
            x = self.self_attn(
                x,
                yoco_key,
                yoco_value,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
                skip_kv_cache_update=skip_kv_cache_update,
            )
        norm_output = self.post_attention_layernorm(x, residual)
        assert isinstance(norm_output, tuple)
        x, residual = norm_output
        x = self.mlp(x, loop_idx)
        return x, residual

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        loop_idx: int,
        yoco_key: torch.Tensor | None,
        yoco_value: torch.Tensor | None,
        kv_cache_dummy_dep: torch.Tensor | None = None,
        skip_kv_cache_update: bool = False,
    ) -> torch.Tensor:
        x, residual = self.forward_with_residual(
            positions,
            hidden_states,
            loop_idx,
            yoco_key,
            yoco_value,
            kv_cache_dummy_dep=kv_cache_dummy_dep,
            skip_kv_cache_update=skip_kv_cache_update,
        )
        # Align and direct layer callers materialize the block output here.
        # Fast model loops call ``forward_with_residual`` instead and carry
        # these two tensors into the next RMSNorm without executing this add.
        return residual + x.float()


# --------------------------------------------------------------------------- #
# Cross-decoder block (compiled separately when fast prefill is enabled)      #
# --------------------------------------------------------------------------- #


@support_torch_compile(
    dynamic_arg_dims={
        "positions": 0,
        "hidden_states": 0,
        "yoco_key": 0,
        "yoco_value": 0,
    },
    enable_if=lambda vllm_config: vllm_config.cache_config.kv_sharing_fast_prefill,
)
class YOCOCrossBlock(nn.Module):
    """Runs every cross-attention layer on compact logits-token inputs.

    The self block has already written the full shared K/V tensors to the
    first cross layer's cache.  The first layer therefore skips its normal
    cache update while retaining a dependency on that write; all cross layers
    only compute the tokens that need logits.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        cross_layers: list,
        first_cross_layer_idx: int,
    ) -> None:
        super().__init__()
        # Store as plain list to avoid re-registering parameters
        self._cross_layers = cross_layers
        self.first_cross_layer_idx = first_cross_layer_idx
        self.execution_mode = _get_yoco_execution_mode(vllm_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        yoco_key: torch.Tensor,
        yoco_value: torch.Tensor,
        kv_cache_dummy_dep: torch.Tensor,
    ) -> torch.Tensor:
        if getattr(self, "execution_mode", "align") == "fast":
            residual = None
            for index, layer in enumerate(self._cross_layers):
                hidden_states, residual = layer.forward_with_residual(
                    positions,
                    hidden_states,
                    0,
                    yoco_key,
                    yoco_value,
                    kv_cache_dummy_dep=kv_cache_dummy_dep if index == 0 else None,
                    skip_kv_cache_update=index == 0,
                    input_residual=residual,
                )
            assert residual is not None
            # The compact cross result must be materialized before the outer
            # fast-prefill wrapper scatters it into the full-token tensor.
            return residual + hidden_states.float()

        for index, layer in enumerate(self._cross_layers):
            hidden_states = layer(
                positions,
                hidden_states,
                0,
                yoco_key,
                yoco_value,
                kv_cache_dummy_dep=kv_cache_dummy_dep if index == 0 else None,
                skip_kv_cache_update=index == 0,
            )
        return hidden_states


# --------------------------------------------------------------------------- #
# Self-decoder block (compiled separately when fast prefill is enabled)       #
# --------------------------------------------------------------------------- #


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
    },
    enable_if=lambda vllm_config: vllm_config.cache_config.kv_sharing_fast_prefill,
)
class YOCOSelfBlock(nn.Module):
    """Self-attention portion compiled as a separate unit for fast prefill.

    Runs (on ALL tokens) the universal-loop self-attention layers and the
    model-level shared-KV producer, then writes K/V directly to the first cross
    layer's cache without running that layer.  Returns the hidden states,
    shared ``yoco_key`` / ``yoco_value``, and the cache-write dependency so the
    compact cross block can consume them.
    Keeping this as its own ``@support_torch_compile`` unit (alongside
    ``YOCOCrossBlock``) means the whole fast-prefill forward is an uncompiled
    wrapper around two piecewise CUDA-graph units, which avoids nesting a
    CUDA-graph capture inside an outer full-graph capture."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        model: YOCOModel,
    ) -> None:
        super().__init__()
        # Hold a non-registering reference to the parent model so we reuse its
        # already-registered parameters without duplicating them (assigning an
        # nn.Module attribute directly would re-register it).
        self._model_ref = [model]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self._model_ref[0]
        if inputs_embeds is not None:
            hidden_states = inputs_embeds.float()
        else:
            assert input_ids is not None
            hidden_states = model.embed_tokens(input_ids).float()

        # Self-attention layers (universal loop) — all tokens.
        residual = None
        for loop_idx in range(model.universal_loop):
            for layer_idx in range(model.first_cross_layer_idx):
                if model.execution_mode == "fast":
                    hidden_states, residual = model.layers[
                        layer_idx
                    ].forward_with_residual(
                        positions,
                        hidden_states,
                        loop_idx,
                        None,
                        None,
                        input_residual=residual,
                    )
                else:
                    hidden_states = model.layers[layer_idx](
                        positions,
                        hidden_states,
                        loop_idx,
                        None,
                        None,
                    )

        # Produce shared K/V for all tokens and write them directly to the
        # first cross layer's cache.  The cross layer itself is deferred to the
        # compact cross block and therefore only runs for logits tokens.
        assert model.yoco_norm is not None
        if residual is None:
            h_norm = model.yoco_norm(hidden_states)
            assert isinstance(h_norm, torch.Tensor)
        else:
            norm_output = model.yoco_norm(hidden_states, residual)
            assert isinstance(norm_output, tuple)
            h_norm, hidden_states = norm_output
        yoco_key, yoco_value = model.project_yoco_kv(h_norm)
        if model.yoco_k_norm is not None:
            yoco_key = _apply_per_head_norm(
                yoco_key,
                model.yoco_num_kv_heads,
                model.yoco_kv_head_dim,
                model.yoco_k_norm,
            )
        owner_attn = model.layers[model.first_cross_layer_idx].self_attn.attn
        yoco_key_view = yoco_key.view(-1, owner_attn.num_kv_heads, owner_attn.head_size)
        yoco_value_view = yoco_value.view(
            -1, owner_attn.num_kv_heads, owner_attn.head_size_v
        )
        kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
            yoco_key_view,
            yoco_value_view,
            _encode_layer_name(owner_attn.layer_name),
        )
        return hidden_states, yoco_key, yoco_value, kv_cache_dummy_dep


# --------------------------------------------------------------------------- #
# Inner model                                                                 #
# --------------------------------------------------------------------------- #


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    },
)
class YOCOModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: PretrainedConfig = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = _maybe_build_yoco_quant_config(vllm_config.quant_config)
        vllm_config.quant_config = quant_config

        self.config = config
        self.quant_config = quant_config
        self.execution_mode = _get_yoco_execution_mode(vllm_config)

        self.hidden_size = _cfg_int(config, "hidden_size", "d_model")
        self.vocab_size = _cfg_int(config, "vocab_size")
        self.num_hidden_layers = _cfg_int(config, "num_hidden_layers", "n_layers")
        self.universal_loop = _cfg_int(config, "universal_loop", default=1)
        self.yoco_cross_layers = _cfg_int(config, "yoco_cross_layers", default=0)
        self.first_cross_layer_idx = self.num_hidden_layers - self.yoco_cross_layers
        tp_size = get_tensor_model_parallel_world_size()
        moe_backend_override = _select_yoco_fast_moe_backend(
            execution_mode=self.execution_mode,
            quant_config=quant_config,
            tp_size=tp_size,
            config=config,
            vllm_config=vllm_config,
        )
        rms_eps = float(
            getattr(config, "rms_norm_eps", getattr(config, "norm_eps", 1e-6))
        )

        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        # Model-level shared YOCO KV producer.  Active only when there is at
        # least one cross-attention layer.
        if self.yoco_cross_layers > 0:
            cross_kv_head = _cfg_int(
                config, "cross_kv_head", "num_key_value_heads", "kv_head"
            )
            head_dim = _cfg_int(config, "head_dim")
            assert cross_kv_head % tp_size == 0 or tp_size % cross_kv_head == 0
            self.yoco_kv_head_dim = head_dim
            self.yoco_num_kv_heads = max(1, cross_kv_head // tp_size)
            self.yoco_local_kv_dim = cross_kv_head * head_dim // tp_size
            # Per-head K norm/clip applied once to the shared ``yoco_key``
            # (mirrors training's model-level ``k_norm`` in ``llm/arch/model.py``).
            self.yoco_k_norm = _build_qk_norm(
                config, head_dim, rms_eps, self.execution_mode
            )
            self.yoco_norm = RMSNorm(
                self.hidden_size,
                eps=rms_eps,
                execution_mode=self.execution_mode,
            )
            use_merged_yoco_kv = (
                self.execution_mode == "fast" and tp_size == 1 and quant_config is None
            )
            if use_merged_yoco_kv:
                self.yoco_kv_proj = MergedColumnParallelLinear(
                    input_size=self.hidden_size,
                    output_sizes=[
                        cross_kv_head * head_dim,
                        cross_kv_head * head_dim,
                    ],
                    bias=False,
                    gather_output=False,
                    quant_config=None,
                    prefix=f"{prefix}.yoco_kv_proj",
                )
                self.yoco_k_proj = None
                self.yoco_v_proj = None
            else:
                self.yoco_kv_proj = None
                self.yoco_k_proj = ColumnParallelLinear(
                    input_size=self.hidden_size,
                    output_size=cross_kv_head * head_dim,
                    bias=False,
                    gather_output=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.yoco_k_proj",
                )
                self.yoco_v_proj = ColumnParallelLinear(
                    input_size=self.hidden_size,
                    output_size=cross_kv_head * head_dim,
                    bias=False,
                    gather_output=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.yoco_v_proj",
                )
        else:
            self.yoco_norm = None
            self.yoco_kv_proj = None
            self.yoco_k_proj = None
            self.yoco_v_proj = None
            self.yoco_k_norm = None
            self.yoco_local_kv_dim = 0

        # Decoder layers.  PP > 1 is out of scope for YOCO (the universal
        # loop and shared cross-KV both couple all layers tightly), so we
        # build the full list and require pp_size == 1.
        assert get_pp_group().world_size == 1, (
            "Pipeline parallelism is not supported for the YOCO model"
        )
        self.layers = nn.ModuleList(
            [
                YOCODecoderLayer(
                    config=config,
                    layer_idx=i,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}",
                    execution_mode=self.execution_mode,
                    moe_backend_override=moe_backend_override,
                )
                for i in range(self.num_hidden_layers)
            ]
        )
        kv_transfer = vllm_config.kv_transfer_config
        if moe_backend_override == "flashinfer_cutlass" and (
            kv_transfer is None or kv_transfer.kv_connector is None
        ):
            fallback_max = _yoco_standalone_prefill_min_tokens(vllm_config) - 1
            for layer in self.layers:
                layer.mlp.experts.yoco_triton_fallback_max_tokens = fallback_max
        # ``start_layer``/``end_layer`` are referenced by some shared
        # utilities; expose them for PP=1 coverage.
        self.start_layer = 0
        self.end_layer = self.num_hidden_layers
        self.norm = RMSNorm(
            self.hidden_size,
            eps=rms_eps,
            execution_mode=self.execution_mode,
        )

        # Fast prefill: split the model into two separately-compiled units so
        # the self portion and the KV-sharing cross layers each get their own
        # piecewise CUDA graph (mirrors gemma3n).  This is required for
        # correctness: the cross block must be invoked during cudagraph warmup,
        # otherwise it tries to capture a graph at inference time (disallowed).
        self.fast_prefill_enabled = cache_config.kv_sharing_fast_prefill
        if self.fast_prefill_enabled and self.yoco_cross_layers > 1:
            # Importing at top level causes issues during tests (see gemma3n).
            from vllm.compilation.backends import set_model_tag

            # Self portion: self layers + shared-KV cache write.
            with set_model_tag("self_decoder"):
                self.self_block = YOCOSelfBlock(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.self_block",
                    model=self,
                )
            # Cross portion: every cross layer only processes logits/decode
            # tokens. The first layer's full shared-KV cache was populated by
            # the self block above.
            kv_sharing_cross_layers = list(self.layers[self.first_cross_layer_idx :])
            with set_model_tag("cross_decoder"):
                self.cross_block = YOCOCrossBlock(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.cross_block",
                    cross_layers=kv_sharing_cross_layers,
                    first_cross_layer_idx=self.first_cross_layer_idx + 1,
                )
            self.full_model_warmed = False

            # Static input buffers for the cross block's CUDA graph.  vLLM runs
            # with cudagraph_copy_inputs=False, so cross-block inputs must have
            # stable addresses across capture/replay.
            max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            kv_dtype = self.embed_tokens.weight.dtype
            device = self.embed_tokens.weight.device
            key_dim = self.yoco_local_kv_dim
            value_dim = self.yoco_local_kv_dim
            self.fp_positions = torch.zeros(
                max_num_tokens, dtype=torch.int64, device=device
            )
            self.fp_hidden_states = torch.zeros(
                (max_num_tokens, self.hidden_size), dtype=torch.float32, device=device
            )
            self.fp_yoco_key = torch.zeros(
                (max_num_tokens, key_dim), dtype=kv_dtype, device=device
            )
            self.fp_yoco_value = torch.zeros(
                (max_num_tokens, value_dim), dtype=kv_dtype, device=device
            )
        else:
            self.self_block = None
            self.cross_block = None
            self.full_model_warmed = True

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.hidden_size
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_yoco_kv(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the model-level K/V, merging the two GEMMs in Fast TP1."""
        if self.yoco_kv_proj is not None:
            yoco_kv, _ = self.yoco_kv_proj(hidden_states)
            yoco_key, yoco_value = yoco_kv.split(self.yoco_local_kv_dim, dim=-1)
            return yoco_key, yoco_value
        assert self.yoco_k_proj is not None and self.yoco_v_proj is not None
        yoco_key, _ = self.yoco_k_proj(hidden_states)
        yoco_value, _ = self.yoco_v_proj(hidden_states)
        return yoco_key, yoco_value

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds.float()
        else:
            assert input_ids is not None
            hidden_states = self.embed_tokens(input_ids).float()

        # Universal loop: run layers 0..first_cross_layer_idx-1
        # ``universal_loop`` times.
        residual = None
        for loop_idx in range(self.universal_loop):
            for layer_idx in range(self.first_cross_layer_idx):
                if self.execution_mode == "fast":
                    hidden_states, residual = self.layers[
                        layer_idx
                    ].forward_with_residual(
                        positions,
                        hidden_states,
                        loop_idx,
                        None,
                        None,
                        input_residual=residual,
                    )
                else:
                    hidden_states = self.layers[layer_idx](
                        positions,
                        hidden_states,
                        loop_idx,
                        None,
                        None,
                    )

        # Cross-attention layers (if any).
        if self.yoco_cross_layers > 0:
            assert self.yoco_norm is not None
            if residual is None:
                h_norm = self.yoco_norm(hidden_states)
                assert isinstance(h_norm, torch.Tensor)
            else:
                norm_output = self.yoco_norm(hidden_states, residual)
                assert isinstance(norm_output, tuple)
                h_norm, hidden_states = norm_output
                residual = None
            yoco_key, yoco_value = self.project_yoco_kv(h_norm)
            if self.yoco_k_norm is not None:
                yoco_key = _apply_per_head_norm(
                    yoco_key,
                    self.yoco_num_kv_heads,
                    self.yoco_kv_head_dim,
                    self.yoco_k_norm,
                )
            # No RoPE on cross-layer K (``rope_dim = 0`` in HF config).
            for layer_idx in range(self.first_cross_layer_idx, self.num_hidden_layers):
                if self.execution_mode == "fast":
                    hidden_states, residual = self.layers[
                        layer_idx
                    ].forward_with_residual(
                        positions,
                        hidden_states,
                        0,
                        yoco_key,
                        yoco_value,
                        input_residual=residual,
                    )
                else:
                    hidden_states = self.layers[layer_idx](
                        positions,
                        hidden_states,
                        0,
                        yoco_key,
                        yoco_value,
                    )

        if residual is None:
            output = self.norm(hidden_states)
            assert isinstance(output, torch.Tensor)
        else:
            norm_output = self.norm(hidden_states, residual)
            assert isinstance(norm_output, tuple)
            output, _ = norm_output
        return output


# --------------------------------------------------------------------------- #
# Top-level CausalLM wrapper                                                  #
# --------------------------------------------------------------------------- #


class YOCOForCausalLM(nn.Module, SupportsPP):
    # Self-attention layers ship q/k/v separately; we fuse into qkv_proj.
    packed_modules_mapping = YOCO_PACKED_MODULES_MAPPING

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: PretrainedConfig = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = _maybe_build_yoco_quant_config(vllm_config.quant_config)
        vllm_config.quant_config = self.quant_config

        self.model = YOCOModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.execution_mode = self.model.execution_mode

        self.vocab_size = _cfg_int(config, "vocab_size")
        hidden_size = _cfg_int(config, "hidden_size", "d_model")
        self.fast_prefill_enabled = vllm_config.cache_config.kv_sharing_fast_prefill
        tie_word_embeddings = bool(getattr(config, "tie_word_embeddings", False))
        if tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=self.vocab_size,
                embedding_dim=hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )

        logit_scale = float(getattr(config, "logit_scale", 1.0))
        self.logits_processor = LogitsProcessor(self.vocab_size, scale=logit_scale)
        self.use_sm100_lm_head_kernel = (
            _supports_yoco_sm100_lm_head_kernel(self.execution_mode)
            and self.quant_config is None
            and not tie_word_embeddings
            and get_tensor_model_parallel_world_size() == 1
            and hidden_size == _YOCO_L3_HIDDEN_SIZE
            and self.vocab_size == _YOCO_L3_VOCAB_SIZE
            and logit_scale == 1.0
        )

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self._rotary_caches_initialized = False
        self._router_weight_caches_initialized = False
        self._shared_expert_weight_caches_initialized = False
        if self.execution_mode == "align" and current_platform.is_cuda():
            # Scope the launch policy to this YOCO instance, including latent,
            # output, shared-KV, shared-expert projections and the LM head.
            for module in self.modules():
                method = getattr(module, "quant_method", None)
                if isinstance(method, UnquantizedLinearMethod):
                    module.quant_method = _YocoAlignLinearMethod()
                elif isinstance(method, UnquantizedEmbeddingMethod):
                    module.quant_method = _YocoAlignEmbeddingMethod()

    # ------------------------------------------------------------------ #
    # standard forward / compute_logits API                              #
    # ------------------------------------------------------------------ #
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def _initialize_rotary_caches(self) -> None:
        if self._rotary_caches_initialized:
            return
        device = self.model.embed_tokens.weight.device
        if device.type != "cuda":
            return

        rotary_caches: dict[tuple[int, int, float], torch.Tensor] = {}
        for module in self.modules():
            if not isinstance(module, YOCORotaryEmbedding):
                continue
            cache_key = (
                module.head_size,
                module.max_position_embeddings,
                module.base,
            )
            cache = rotary_caches.get(cache_key)
            if cache is None:
                cache = module._get_cos_sin_cache(device)
                rotary_caches[cache_key] = cache
            else:
                module.cos_sin_cache = cache
        self._rotary_caches_initialized = True

    def _initialize_router_weight_caches(self) -> None:
        if self._router_weight_caches_initialized:
            return
        for module in self.modules():
            if isinstance(module, YOCOMoE):
                module.initialize_router_weight_cache()
        self._router_weight_caches_initialized = True

    def _initialize_shared_expert_weight_caches(self) -> None:
        if getattr(self, "_shared_expert_weight_caches_initialized", False):
            return
        for module in self.modules():
            if isinstance(module, YOCOSharedExperts):
                module.initialize_fast_weight_cache()
        self._shared_expert_weight_caches_initialized = True

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        kv_only_prefill: bool = False,
    ) -> torch.Tensor:
        # Default loading initializes this eagerly. This fallback covers
        # loaders that assign tensors without calling this model's load_weights.
        self._initialize_rotary_caches()
        self._initialize_router_weight_caches()
        self._initialize_shared_expert_weight_caches()
        if not self.fast_prefill_enabled:
            return self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )
        return self._fast_prefill_forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            kv_only_prefill=kv_only_prefill,
        )

    def _fast_prefill_forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        kv_only_prefill: bool = False,
    ) -> torch.Tensor:
        """Forward with fast prefill: cross-attention layers that share KV
        only process decode tokens during prefill.

        The self portion and the cross portion run as two separately-compiled
        ``@support_torch_compile`` units (``self_block`` / ``cross_block``).
        The first eager profile run compiles the ordinary full model before
        CUDA graph capture, then piecewise profiling compiles both split
        blocks. Uniform FULL decode reuses the ordinary model path; prefill
        uses the split blocks."""
        model = self.model

        # No dedicated fast-prefill blocks (e.g. a single cross layer): fall
        # back to the standard dense forward.
        if model.self_block is None or model.cross_block is None:
            return model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )

        # Decode-token indices. Piecewise profile/warmup runs without fast
        # metadata fall back to all tokens so both split blocks are compiled.
        (
            logits_indices_padded,
            num_logits_indices,
            fast_prefill_num_tokens_across_dp_cpu,
        ) = self._get_fast_prefill_indices()
        fwd_ctx = get_forward_context()
        global_fast_prefill = kv_only_prefill or logits_indices_padded is not None
        if (
            not global_fast_prefill
            and fwd_ctx.dp_metadata is not None
            and fast_prefill_num_tokens_across_dp_cpu is not None
        ):
            global_fast_prefill = not torch.equal(
                fast_prefill_num_tokens_across_dp_cpu,
                fwd_ctx.dp_metadata.num_tokens_across_dp_cpu,
            )

        if (
            not global_fast_prefill
            and fwd_ctx.cudagraph_runtime_mode == CUDAGraphMode.NONE
            and not model.full_model_warmed
        ):
            model(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
            )
            model.full_model_warmed = True

        if (
            not global_fast_prefill
            and fwd_ctx.cudagraph_runtime_mode == CUDAGraphMode.FULL
        ):
            assert model.full_model_warmed, (
                "YOCO full model must be compiled during eager profiling "
                "before CUDA graph capture."
            )
            return model(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
            )

        # Self portion on ALL tokens (separate piecewise CUDA graph). It also
        # produces and writes the full shared KV cache without executing any
        # cross layer.
        hidden_states, yoco_key, yoco_value, kv_cache_dummy_dep = model.self_block(
            input_ids,
            positions,
            inputs_embeds,
        )

        # A disaggregated P request only needs the shared K/V produced above.
        # Its sampled token is ignored by the proxy, so use the self-decoder
        # state as a disposable logits input and avoid all ten cross layers.
        if kv_only_prefill:
            return model.norm(hidden_states)

        if logits_indices_padded is None:
            logits_indices_padded = torch.arange(
                positions.size(0),
                dtype=torch.int64,
                device=positions.device,
            )

        # Clone the self-decoder output before it is potentially freed by the
        # piecewise cudagraph machinery when multiple compile units are used.
        out_hidden = hidden_states.clone()

        # Feed the cross block through static buffers — vLLM runs with
        # cudagraph_copy_inputs=False, so inputs need stable addresses.
        n = logits_indices_padded.size(0)
        model.fp_positions[:n].copy_(positions[logits_indices_padded])
        model.fp_hidden_states[:n].copy_(hidden_states[logits_indices_padded])
        model.fp_yoco_key[:n].copy_(yoco_key[logits_indices_padded])
        model.fp_yoco_value[:n].copy_(yoco_value[logits_indices_padded])

        original_dp_metadata = fwd_ctx.dp_metadata
        if (
            original_dp_metadata is not None
            and fast_prefill_num_tokens_across_dp_cpu is not None
        ):
            fwd_ctx.dp_metadata = DPMetadata(fast_prefill_num_tokens_across_dp_cpu)
        try:
            decode_hidden = model.cross_block(
                model.fp_positions[:n],
                model.fp_hidden_states[:n],
                model.fp_yoco_key[:n],
                model.fp_yoco_value[:n],
                kv_cache_dummy_dep,
            )
        finally:
            fwd_ctx.dp_metadata = original_dp_metadata

        # Merge cross-decoder outputs back into the full hidden states.
        if num_logits_indices is not None:
            assert num_logits_indices > 0
            real_indices = logits_indices_padded[:num_logits_indices]
            out_hidden[real_indices] = decode_hidden[:num_logits_indices]
        else:
            out_hidden[logits_indices_padded] = decode_hidden

        return model.norm(out_hidden)

    def _get_fast_prefill_indices(
        self,
    ) -> tuple[torch.Tensor | None, int | None, torch.Tensor | None]:
        """Retrieve logits_indices from forward context attention metadata."""
        fwd_ctx = get_forward_context()
        attn_metadata = fwd_ctx.attn_metadata
        fast_prefill_num_tokens_across_dp_cpu = (
            fwd_ctx.fast_prefill_num_tokens_across_dp_cpu
        )
        if attn_metadata is None:
            return None, None, fast_prefill_num_tokens_across_dp_cpu
        if not isinstance(attn_metadata, dict):
            return None, None, fast_prefill_num_tokens_across_dp_cpu
        # Find a KV-sharing layer's metadata to get logits_indices.
        # Use the last layer's attention (which is a fast prefill layer).
        last_layer = self.model.layers[-1]
        layer_name = last_layer.self_attn.attn.layer_name
        layer_meta = attn_metadata.get(layer_name)
        if layer_meta is None:
            return None, None, fast_prefill_num_tokens_across_dp_cpu
        if isinstance(layer_meta, KVSharingFastPrefillMetadata):
            return (
                layer_meta.logits_indices_padded,
                layer_meta.num_logits_indices,
                fast_prefill_num_tokens_across_dp_cpu,
            )
        return None, None, fast_prefill_num_tokens_across_dp_cpu

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor:
        if (
            getattr(self, "use_sm100_lm_head_kernel", False)
            and sampling_metadata is None
        ):
            logits = _yoco_lm_head_dispatch(
                hidden_states,
                self.lm_head.weight,
                use_sm100_kernel=True,
            )
            if self.logits_processor.scale != 1.0:
                logits *= self.logits_processor.scale
            return logits
        return self.logits_processor(self.lm_head, hidden_states, sampling_metadata)

    # ------------------------------------------------------------------ #
    # Weight loading                                                     #
    # ------------------------------------------------------------------ #
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_names: set[str] = set()

        # Stacked Q/K/V mapping (only for self-attention layers 0..9).
        stacked_qkv = [
            ("self_attn.qkv_proj", "self_attn.q_proj", "q"),
            ("self_attn.qkv_proj", "self_attn.k_proj", "k"),
            ("self_attn.qkv_proj", "self_attn.v_proj", "v"),
        ]
        first_cross_layer_idx = _cfg_int(
            self.config, "num_hidden_layers", "n_layers"
        ) - _cfg_int(self.config, "yoco_cross_layers", default=0)
        moe_intermediate_size = _cfg_int(
            self.config, "moe_intermediate_size", "moe_ffn_dim"
        )
        num_experts = _cfg_int(self.config, "num_experts", "moe_expert_num")
        yoco_kv_shards = {
            "model.yoco_k_proj.weight": 0,
            "model.yoco_v_proj.weight": 1,
        }
        self_qkv_lambda_shards = {
            "self_attn.q_proj": 0,
            "self_attn.k_proj": 1,
            "self_attn.v_proj": 2,
            "self_attn.lambda_proj": 3,
        }

        for name, loaded_weight in weights:
            # ------------------------------------------------------------
            # Top-level YOCO shared-KV producer rename.
            # ------------------------------------------------------------
            if name == "model.k_proj.weight":
                name = "model.yoco_k_proj.weight"
            elif name == "model.v_proj.weight":
                name = "model.yoco_v_proj.weight"

            # Fast BF16 TP1 packs the model-level K/V projections into one
            # GEMM. Keep this YOCO-specific instead of extending the global
            # quantization packed-module mapping: quantized and TP paths still
            # instantiate the original two projections.
            yoco_kv_shard_id = yoco_kv_shards.get(name)
            if yoco_kv_shard_id is not None:
                target_name = "model.yoco_kv_proj.weight"
                if target_name in params_dict:
                    if not is_pp_missing_parameter(target_name, self):
                        param = params_dict[target_name]
                        param.weight_loader(param, loaded_weight, yoco_kv_shard_id)
                        loaded_names.add(target_name)
                    continue

            # Fast BF16 TP1 packs each self-attention layer's Q/K/V/lambda
            # projections into one YOCO-private GEMM. Align, quantized, and TP
            # paths keep the standard qkv_proj plus standalone lambda_proj.
            self_qkv_lambda_source = next(
                (
                    source
                    for source in self_qkv_lambda_shards
                    if name.endswith(f".{source}.weight")
                ),
                None,
            )
            if self_qkv_lambda_source is not None:
                layer_idx_str = name.split(".layers.")[-1].split(".")[0]
                try:
                    layer_idx_int = int(layer_idx_str)
                except ValueError:
                    layer_idx_int = -1
                if 0 <= layer_idx_int < first_cross_layer_idx:
                    target_name = name.replace(
                        self_qkv_lambda_source,
                        "self_attn.qkv_lambda_proj",
                    )
                    if target_name in params_dict:
                        if not is_pp_missing_parameter(target_name, self):
                            param = params_dict[target_name]
                            param.weight_loader(
                                param,
                                loaded_weight,
                                self_qkv_lambda_shards[self_qkv_lambda_source],
                            )
                            loaded_names.add(target_name)
                        continue

            # Fast BF16 TP1 packs each cross-attention layer's Q and lambda
            # projections into one GEMM. Gate this mapping on the cross-layer
            # index as well as the presence of the private merged parameter.
            q_lambda_shard_id = None
            if name.endswith(".self_attn.q_proj.weight"):
                q_lambda_shard_id = 0
                q_lambda_source = "self_attn.q_proj"
            elif name.endswith(".self_attn.lambda_proj.weight"):
                q_lambda_shard_id = 1
                q_lambda_source = "self_attn.lambda_proj"
            if q_lambda_shard_id is not None:
                layer_idx_str = name.split(".layers.")[-1].split(".")[0]
                try:
                    layer_idx_int = int(layer_idx_str)
                except ValueError:
                    layer_idx_int = -1
                if layer_idx_int >= first_cross_layer_idx:
                    target_name = name.replace(
                        q_lambda_source,
                        "self_attn.q_lambda_proj",
                    )
                    if target_name in params_dict:
                        if not is_pp_missing_parameter(target_name, self):
                            param = params_dict[target_name]
                            param.weight_loader(
                                param,
                                loaded_weight,
                                q_lambda_shard_id,
                            )
                            loaded_names.add(target_name)
                        continue

            # ------------------------------------------------------------
            # Fused MoE expert tensors (per-expert dispatch).
            # ------------------------------------------------------------
            if name.endswith(".mlp.experts.w13_weight"):
                base = name[: -len(".w13_weight")]
                param_name = f"{base}.w13_weight"
                if param_name not in params_dict:
                    continue
                if is_pp_missing_parameter(param_name, self):
                    continue
                param = params_dict[param_name]
                weight_loader = param.weight_loader
                # HF tensor: (E * 2 * ffn, hidden) — split per expert and
                # then split into w1 (gate, first half) / w3 (up, second).
                w = loaded_weight.view(num_experts, 2 * moe_intermediate_size, -1)
                for expert_id in range(num_experts):
                    w1 = w[expert_id, :moe_intermediate_size, :]
                    w3 = w[expert_id, moe_intermediate_size:, :]
                    weight_loader(
                        param,
                        w1,
                        name,
                        "w1",
                        expert_id,
                    )
                    weight_loader(
                        param,
                        w3,
                        name,
                        "w3",
                        expert_id,
                    )
                loaded_names.add(param_name)
                continue

            if name.endswith(".mlp.experts.w2_weight"):
                base = name[: -len(".w2_weight")]
                param_name = f"{base}.w2_weight"
                if param_name not in params_dict:
                    continue
                if is_pp_missing_parameter(param_name, self):
                    continue
                param = params_dict[param_name]
                weight_loader = param.weight_loader
                # HF tensor: (E * hidden, ffn) — split per expert.
                w = loaded_weight.view(num_experts, -1, moe_intermediate_size)
                for expert_id in range(num_experts):
                    weight_loader(
                        param,
                        w[expert_id],
                        name,
                        "w2",
                        expert_id,
                    )
                loaded_names.add(param_name)
                continue

            # ------------------------------------------------------------
            # Self-attention Q/K/V → qkv_proj (only for self-attn layers).
            # ------------------------------------------------------------
            handled = False
            for stacked, shard_name, shard_id in stacked_qkv:
                if shard_name not in name:
                    continue
                # ``stacked_params_mapping`` would map e.g.
                # ``...self_attn.q_proj.weight`` → ``...self_attn.qkv_proj.weight``.
                # Only applies to layers with stacked qkv (self-attention
                # layers).  Cross-attn layers keep a standalone q_proj.
                layer_idx_str = name.split(".layers.")[-1].split(".")[0]
                try:
                    layer_idx_int = int(layer_idx_str)
                except ValueError:
                    continue
                if (
                    layer_idx_int >= first_cross_layer_idx
                    and shard_name == "self_attn.q_proj"
                ):
                    # Cross-attn standalone q_proj — skip the merging path.
                    break
                # Cross-attn layers don't have k_proj/v_proj at all, so the
                # only way to reach those is on self-attn layers.
                target_name = name.replace(shard_name, stacked)
                if target_name not in params_dict:
                    continue
                if is_pp_missing_parameter(target_name, self):
                    continue
                param = params_dict[target_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_names.add(target_name)
                handled = True
                break
            if handled:
                continue

            # ------------------------------------------------------------
            # Shared-expert gate_up_proj fused tensor.
            # ------------------------------------------------------------
            if name.endswith(".mlp.shared_experts.gate_up_proj.weight"):
                # HF tensor: (2 * intermediate, hidden) — first half gate,
                # second half up.  ``MergedColumnParallelLinear`` accepts
                # the merged tensor via shard_id=0 and shard_id=1 calls.
                if name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                half = loaded_weight.shape[0] // 2
                gate = loaded_weight[:half, :]
                up = loaded_weight[half:, :]
                param.weight_loader(param, gate, 0)
                param.weight_loader(param, up, 1)
                loaded_names.add(name)
                continue

            # ------------------------------------------------------------
            # Default loader: name maps 1:1 to a registered parameter.
            # ------------------------------------------------------------
            if name not in params_dict:
                # Unknown weight — skip silently.  This is rare but happens
                # for e.g. quantization scales we don't use.
                continue
            if is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]
            # Cast bf16 router gate weights to fp32 to match training.
            if name.endswith(".mlp.gate.weight") and param.dtype != loaded_weight.dtype:
                loaded_weight = loaded_weight.to(param.dtype)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            try:
                weight_loader(param, loaded_weight)
            except TypeError:
                # Some param weight loaders take extra positional args; fall
                # back to the default one.
                default_weight_loader(param, loaded_weight)
            loaded_names.add(name)

        # Piecewise compilation cannot trace a forward that mutates module
        # buffers, so initialize caches before profile/warmup forwards begin.
        self._initialize_rotary_caches()
        self._router_weight_caches_initialized = False
        self._initialize_router_weight_caches()
        self._shared_expert_weight_caches_initialized = False
        self._initialize_shared_expert_weight_caches()

        return loaded_names
