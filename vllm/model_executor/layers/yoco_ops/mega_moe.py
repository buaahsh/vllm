# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in FP8 group32 MegaMoE for frozen, single-rank YOCO L3 experts."""

from typing import Any
from weakref import WeakValueDictionary

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.config.yoco import resolve_yoco_moe_dispatch
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.deep_gemm import get_deep_gemm_mega_moe, is_deep_gemm_e8m0_used

logger = init_logger(__name__)


@triton.jit
def _latent_norm_megamoe_kernel(
    x,
    gamma,
    ids,
    weights,
    out,
    scales,
    out_ids,
    out_weights,
    x_stride: tl.constexpr,
    ids_stride: tl.constexpr,
    weights_stride: tl.constexpr,
    out_stride: tl.constexpr,
    scales_stride: tl.constexpr,
    out_ids_stride: tl.constexpr,
    out_weights_stride: tl.constexpr,
    EPS: tl.constexpr,
    STAGE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, 1024)
    if not STAGE:
        # Retain the original norm layout on the small-M and prefill paths.
        cols = tl.max_contiguous(cols, 8)
    value = tl.load(x + row * x_stride + cols).to(tl.float32)
    affine = tl.load(gamma + cols).to(tl.float32)
    variance = tl.sum(value * value, 0) * (1.0 / 1024)
    normalized = value * tldevice.rsqrt(variance + EPS) * affine
    # Keep the existing norm -> BF16 -> group32 quantizer rounding boundary.
    normalized = normalized.to(tl.bfloat16)
    if STAGE:
        normalized = normalized.to(tl.float32).reshape(32, 32)
        amax = tl.maximum(tl.max(tl.abs(normalized), 1), 1.0e-4)
        bits = (amax / 448.0).to(tl.uint32, bitcast=True)
        exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
        exponent = tl.minimum(tl.maximum(exponent, 1), 254)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        quantized = (normalized * (1.0 / scale)[:, None]).reshape(1024)
        tl.store(out + row * out_stride + cols, quantized.to(tl.float8e4nv))
        packed = tl.sum(exponent.reshape(8, 4) << (tl.arange(0, 4)[None, :] * 8), 1).to(
            tl.int32
        )
        tl.store(scales + row * scales_stride + tl.arange(0, 8), packed)
        topk = tl.arange(0, 8)
        tl.store(
            out_ids + row * out_ids_stride + topk,
            tl.load(ids + row * ids_stride + topk).to(tl.int64),
        )
        tl.store(
            out_weights + row * out_weights_stride + topk,
            tl.load(weights + row * weights_stride + topk),
        )
    else:
        tl.store(out + row * out_stride + cols, normalized)


def prepare_yoco_latent_megamoe_inputs(
    hidden: torch.Tensor,
    norm: torch.nn.Module,
    weights: torch.Tensor,
    ids: torch.Tensor,
    buffer: Any,
) -> None:
    """Normalize and stage into the existing Mega workspace on its owner stream."""
    if hidden.shape[0]:
        _latent_norm_megamoe_kernel[(hidden.shape[0],)](
            hidden,
            norm.weight,
            ids,
            weights,
            buffer.x,
            buffer.x_sf,
            buffer.topk_idx,
            buffer.topk_weights,
            hidden.stride(0),
            ids.stride(0),
            weights.stride(0),
            buffer.x.stride(0),
            buffer.x_sf.stride(0),
            buffer.topk_idx.stride(0),
            buffer.topk_weights.stride(0),
            norm.eps,
            STAGE=True,
            num_warps=4,
            enable_fp_fusion=False,
        )


def normalize_yoco_mega_input(
    hidden: torch.Tensor, norm: torch.nn.Module
) -> torch.Tensor:
    """Retain the compiled model's single-warp FP32 reduction after deferral."""
    if hidden.dtype != torch.bfloat16:
        return norm(hidden)
    output = torch.empty_like(hidden)
    if hidden.shape[0]:
        _latent_norm_megamoe_kernel[(hidden.shape[0],)](
            hidden,
            norm.weight,
            None,
            None,
            output,
            None,
            None,
            None,
            hidden.stride(0),
            0,
            0,
            output.stride(0),
            0,
            0,
            0,
            norm.eps,
            STAGE=False,
            num_warps=1,
            enable_fp_fusion=False,
        )
    return output


class YOCOLatentRoutedExperts(RoutedExperts):
    """Keep the latent norm inside the existing opaque routed-expert boundary."""

    def __init__(self, *args: Any, input_norm: torch.nn.Module, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.input_norm = input_norm

    def forward_modular(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dtype == torch.bfloat16 and use_yoco_mega_moe(self, x.shape[0]):
            return self._yoco_mega_moe(
                x, topk_weights, topk_ids, input_norm=self.input_norm
            )
        return super().forward_modular(
            normalize_yoco_mega_input(x, self.input_norm),
            topk_weights,
            topk_ids,
            shared_experts,
            shared_experts_input,
        )


def expand_block128_scales(
    scales: torch.Tensor, rows: int, columns: int
) -> torch.Tensor:
    """Broadcast existing power-of-two scales into packed 1x32 Mega layout."""
    if scales.ndim != 3 or rows % 128 or columns % 128:
        raise ValueError("Expected block128 weight scale dimensions")
    groups = columns // 128
    if scales.dtype == torch.float32:
        if scales.shape[1:] != (rows // 128, groups):
            raise ValueError("FP32 scales must have shape [E, N/128, K/128]")
        bits = scales.contiguous().view(torch.int32).to(torch.int64)
        exponent = (bits >> 23) & 255
        if not bool(((bits > 0) & ((bits & 0x7FFFFF) == 0) & (exponent < 255)).all()):
            raise ValueError("Weight scales must be positive finite powers of two")
        exponent = exponent.repeat_interleave(128, dim=1)
    elif scales.dtype == torch.int32:
        if scales.shape[1:] != (rows, (groups + 3) // 4):
            raise ValueError("Packed scales must have shape [E, N, ceil(K/512)]")
        shifts = torch.arange(4, device=scales.device) * 8
        exponent = (scales.to(torch.int64).unsqueeze(-1) >> shifts) & 255
        exponent = exponent.flatten(-2)[..., :groups]
        if not bool(((exponent > 0) & (exponent < 255)).all()):
            raise ValueError("Packed scales must encode finite normal powers of two")
    else:
        raise ValueError("Weight scales must be FP32 block128 or packed INT32")
    packed = (exponent * 0x01010101).to(torch.int32)
    return packed.transpose(-1, -2).contiguous().transpose(-1, -2)


class _Workspace:
    def __init__(self, backend: Any, max_tokens: int) -> None:
        self.buffer = backend.get_symm_buffer_for_mega_moe(
            get_tp_group().device_group,
            128,
            max_tokens,
            8,
            1024,
            3840,
            num_shared_experts=0,
            mma_type="fp8xfp8",
        )


# Every layer executes on the same stream; DBO and parallel groups are rejected.
# Weak ownership lets a destroyed model release its workspace and device tensors.
_workspaces: WeakValueDictionary[tuple[int, int, int], _Workspace] = (
    WeakValueDictionary()
)


class YOCOMegaMoERouted:
    def __init__(
        self,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scales: torch.Tensor,
        w2_scales: torch.Tensor,
        workspace: _Workspace,
        backend: Any,
    ) -> None:
        self.workspace = workspace
        self.backend = backend
        self.w13, self.w2 = backend.transform_weights_for_mega_moe(
            (w13, expand_block128_scales(w13_scales, 7680, 1024)),
            (w2, expand_block128_scales(w2_scales, 1024, 3840)),
        )

    def __call__(
        self,
        hidden: torch.Tensor,
        weights: torch.Tensor,
        ids: torch.Tensor,
        *,
        input_norm: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import (
            prepare_megamoe_inputs,
        )

        n = hidden.shape[0]
        buffer = self.workspace.buffer
        if (
            hidden.ndim != 2
            or hidden.shape[1] != 1024
            or hidden.dtype != torch.bfloat16
            or hidden.device != buffer.x.device
            or n > buffer.num_max_tokens_per_rank
            or ids.shape != (n, 8)
            or weights.shape != ids.shape
            or weights.dtype != torch.float32
            or ids.dtype not in (torch.int32, torch.int64)
            or hidden.device != weights.device
            or hidden.device != ids.device
        ):
            raise ValueError("Inputs exceed the prepared YOCO MegaMoE contract")
        output = torch.empty_like(hidden)
        if n:
            if input_norm is None:
                prepare_megamoe_inputs(
                    hidden,
                    weights,
                    ids,
                    buffer.x[:n],
                    buffer.x_sf[:n],
                    buffer.topk_idx[:n],
                    buffer.topk_weights[:n],
                )
            else:
                prepare_yoco_latent_megamoe_inputs(
                    hidden, input_norm, weights, ids, buffer
                )
            self.backend.fp8_fp4_mega_moe(
                output,
                self.w13,
                self.w2,
                buffer,
                recipe=(1, 1, 32),
                activation_clamp=10.0,
                fast_math=False,
            )
        return output


def _supports_yoco_mega_moe(layer: Any, kernel: Any) -> bool:
    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        DeepGemmExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe import (
        TritonOrDeepGemmExperts,
    )

    config = get_current_vllm_config()
    parallel = config.parallel_config
    moe = layer.moe_config
    mp = moe.moe_parallel_config
    experts = kernel.fused_experts
    if isinstance(experts, TritonOrDeepGemmExperts):
        experts = experts.experts
    return not (
        not current_platform.is_device_capability_family(100)
        or envs.VLLM_BATCH_INVARIANT
        or parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.enable_dbo
        or mp.enable_eplb
        or (mp.tp_size, mp.dp_size, mp.ep_size) != (1, 1, 1)
        or moe.is_sequence_parallel
        or moe.is_lora_enabled
        or moe.has_bias
        or moe.in_dtype != torch.bfloat16
        or not moe.yoco.direct_fp8_activation
        or not moe.apply_router_weight_before_w2
        or moe.experts_per_token != 8
        or layer.swiglu_limit != 10.0
        or not isinstance(experts, DeepGemmExperts)
        or experts.block_shape != [128, 128]
        or kernel.prepare_finalize.supports_async()
        or tuple(layer.w13_weight.shape) != (128, 7680, 1024)
        or tuple(layer.w2_weight.shape) != (128, 1024, 3840)
        or layer.w13_weight.dtype != torch.float8_e4m3fn
        or layer.w2_weight.dtype != torch.float8_e4m3fn
    )


@torch.no_grad()
def configure_yoco_moe_dispatch(layer: Any, kernel: Any) -> None:
    """Resolve configuration once, before warmup and CUDA Graph capture."""
    from vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe import (
        TritonOrDeepGemmExperts,
    )

    config = get_current_vllm_config()
    parallel = config.parallel_config
    decode_graphs = (
        config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
    )
    spec = config.speculative_config
    dispatch = resolve_yoco_moe_dispatch(
        config.additional_config,
        gpu_name=current_platform.get_device_name(layer.w13_weight.device.index),
        parallel_sizes=(
            parallel.tensor_parallel_size,
            parallel.pipeline_parallel_size,
            parallel.data_parallel_size,
            layer.moe_config.moe_parallel_config.ep_size,
        ),
        profile_compatible=(
            _supports_yoco_mega_moe(layer, kernel)
            and is_deep_gemm_e8m0_used()
            and decode_graphs
            and not config.model_config.enforce_eager
            and bool(envs.VLLM_USE_V2_MODEL_RUNNER)
            and getattr(spec, "method", None) == "dspark"
        ),
        speculative_tokens=getattr(spec, "num_speculative_tokens", 0),
    )
    experts = kernel.fused_experts
    if dispatch.triton_max_tokens is not None:
        if not isinstance(experts, TritonOrDeepGemmExperts):
            raise ValueError("triton_max_tokens requires TritonOrDeepGemmExperts")
        if dispatch.triton_max_tokens == 0:
            experts._yoco_fp8_decode_limit = 0
        elif not experts.configure_yoco_fp8_decode(dispatch.triton_max_tokens):
            raise ValueError("triton_max_tokens is unsupported by this FP8 layout")
    layer._yoco_moe_dispatch = dispatch
    layer._yoco_moe_decode_only_graphs = decode_graphs
    layer._yoco_moe_capture_backends = {}
    logger.info_once(
        "YOCO MoE dispatch profile=%s, Triton M<=%s, Mega M>=%s, "
        "Mega prefill=%s. Unselected rows retain the original backend.",
        dispatch.profile,
        dispatch.triton_max_tokens,
        dispatch.mega_min_tokens,
        dispatch.mega_prefill,
    )
    if dispatch.mega_min_tokens is not None:
        prepare_yoco_mega_moe(layer, kernel)


def use_yoco_mega_moe(layer: Any, num_tokens: int) -> bool:
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    dispatch = layer._yoco_moe_dispatch
    # Profiling without metadata must exercise the original path to reserve its
    # largest workspace before the workspace manager locks graph addresses.
    if not is_forward_context_available():
        return False
    context = get_forward_context()
    if context.attn_metadata is None:
        return False
    selected = dispatch.use_mega(
        num_tokens,
        decode_graph=(
            layer._yoco_moe_decode_only_graphs
            and CUDAGraphMode.FULL
            in (context.cudagraph_runtime_mode, context.cudagraph_capture_mode)
        ),
    )
    if context.cudagraph_capture_mode != CUDAGraphMode.NONE:
        layer._yoco_moe_capture_backends[num_tokens] = selected
    return selected


@torch.no_grad()
def prepare_yoco_mega_moe(layer: Any, kernel: Any) -> None:
    """Prepare frozen layouts before profiling/capture; retain runner overlap."""
    if getattr(layer, "_yoco_mega_moe", None) is not None:
        raise RuntimeError("YOCO MegaMoE weights are frozen; create a new engine")
    if not _supports_yoco_mega_moe(layer, kernel):
        raise ValueError(
            "YOCO MegaMoE requires SM100 Fast L3 online block128 FP8, BF16 inputs, "
            "TP1/PP1/DP1/EP1, without CP, EPLB, DBO, LoRA or batch invariance"
        )
    from vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe import (
        TritonOrDeepGemmExperts,
    )

    config = get_current_vllm_config()
    moe = layer.moe_config
    experts = kernel.fused_experts
    if isinstance(experts, TritonOrDeepGemmExperts):
        experts = experts.experts
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare YOCO MegaMoE before CUDA graph capture")
    backend = get_deep_gemm_mega_moe()
    max_tokens = max(
        moe.max_num_tokens, config.compilation_config.max_cudagraph_capture_size or 0
    )
    key = (id(config), layer.w13_weight.device.index, max_tokens)
    workspace = _workspaces.get(key)
    if workspace is None:
        workspace = _Workspace(backend, max_tokens)
        _workspaces[key] = workspace
        logger.info(
            "YOCO MegaMoE fp8_group32 enabled; shared workspace %d bytes. "
            "Activation quantization differs from group128; weights remain frozen.",
            workspace.buffer.buffer.nbytes,
        )
    layer._yoco_mega_moe = YOCOMegaMoERouted(
        layer.w13_weight,
        layer.w2_weight,
        experts.w1_scale,
        experts.w2_scale,
        workspace,
        backend,
    )
