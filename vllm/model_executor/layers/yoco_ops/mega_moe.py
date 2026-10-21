# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in FP8 group32 MegaMoE for frozen, single-rank YOCO L3 experts."""

from typing import Any
from weakref import WeakValueDictionary

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import get_deep_gemm_mega_moe

logger = init_logger(__name__)


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
        self, hidden: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor
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
            prepare_megamoe_inputs(
                hidden,
                weights,
                ids,
                buffer.x[:n],
                buffer.x_sf[:n],
                buffer.topk_idx[:n],
                buffer.topk_weights[:n],
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


@torch.no_grad()
def prepare_yoco_mega_moe(layer: Any, kernel: Any) -> None:
    """Prepare frozen layouts before profiling/capture; retain runner overlap."""
    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        DeepGemmExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe import (
        TritonOrDeepGemmExperts,
    )

    if getattr(layer, "_yoco_mega_moe", None) is not None:
        raise RuntimeError("YOCO MegaMoE weights are frozen; create a new engine")
    config = get_current_vllm_config()
    parallel = config.parallel_config
    moe = layer.moe_config
    mp = moe.moe_parallel_config
    experts = kernel.fused_experts
    if isinstance(experts, TritonOrDeepGemmExperts):
        experts = experts.experts
    if (
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
    ):
        raise ValueError(
            "YOCO MegaMoE requires SM100 Fast L3 online block128 FP8, BF16 inputs, "
            "TP1/PP1/DP1/EP1, without CP, EPLB, DBO, LoRA or batch invariance"
        )
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
