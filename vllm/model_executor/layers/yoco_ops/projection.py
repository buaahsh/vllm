# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO projection operators and their numerical fallbacks."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
        Fp8BlockScaledMMLinearKernel,
    )
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PerBlockOnlineLinearMethod,
    )

import torch
import torch.nn.functional as F
from torch import nn

import vllm.envs as envs
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.batch_invariant import (
    linear_batch_invariant,
    matmul_kernel_persistent,
)
from vllm.model_executor.layers.linear import RowParallelLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import direct_register_custom_op

if HAS_TRITON:
    from .norm import _yoco_weighted_rms_clip_kernel


_YOCO_Q_LAMBDA_MERGED_MAX_TOKENS = 2048


_YOCO_QKV_LAMBDA_MERGED_MAX_TOKENS = 4096


_YOCO_L3_HIDDEN_SIZE = 3072


_YOCO_L3_VOCAB_SIZE = 154880


_YOCO_SM100_LM_HEAD_MAX_TOKENS = 16


_YOCO_SM100_DIFF_V3_TP4_MIN_TOKENS = 32


if HAS_TRITON:

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
    return _run_yoco_lm_head_cuda(hidden_states, weight, torch.float32)


def _yoco_lm_head_bf16_cuda(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return _run_yoco_lm_head_cuda(hidden_states, weight, torch.bfloat16)


def _run_yoco_lm_head_cuda(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    output_dtype: torch.dtype,
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
        dtype=output_dtype,
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


def _yoco_lm_head_bf16_fake(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return hidden_states.new_empty(
        (*hidden_states.shape[:-1], weight.shape[0]), dtype=torch.bfloat16
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


def _yoco_clip_fp8_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    limit: float,
    value: torch.Tensor | None = None,
    value_scale: torch.Tensor | None = None,
    round_before_weight: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(
        x, dtype=torch.float8_e4m3fn, memory_format=torch.contiguous_format
    )
    v = torch.empty_like(output) if value is not None else output.new_empty((0,))
    return output, v


def _yoco_clip_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    limit: float,
    value: torch.Tensor | None = None,
    value_scale: torch.Tensor | None = None,
    round_before_weight: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dtype == weight.dtype == torch.bfloat16 and x.shape[-1] == 128
    assert scale.numel() == 1
    if value is not None:
        assert value.dtype == x.dtype and value.shape == x.shape
        assert value.stride(1) == 128 and value.stride(2) == 1
        assert value_scale is not None and value_scale.numel() == 1
    output, v = _yoco_clip_fp8_fake(x, weight, scale, eps, limit, value, value_scale)
    tokens, heads, _ = x.shape
    rows = 16 if tokens * heads < 12288 else 32
    if tokens:
        _yoco_weighted_rms_clip_kernel[(triton.cdiv(tokens * heads, rows),)](
            x,
            weight,
            output,
            tokens,
            heads,
            x.stride(0),
            x.stride(1),
            eps,
            limit,
            128,
            rows,
            round_before_weight,
            scale,
            value,
            v,
            value_scale,
            value.stride(0) if value is not None else 0,
            FP8_OUTPUT=True,
            QUANT_VALUE=value is not None,
            # FP8 stores otherwise double the elements per thread and alter
            # the RMS reduction tree. Keep the original eight elements/lane.
            num_warps=rows // 2,
            num_stages=1,
        )
    return output, v


def _yoco_can_fuse_fp8_attention(attention: Attention, execution_mode: str) -> bool:
    # Resolve once at model construction; scales are read from device at runtime.
    return (
        execution_mode == "fast"
        and envs.VLLM_YOCO_FP8_ATTENTION_FUSION
        and HAS_TRITON
        and current_platform.is_cuda()
        and current_platform.is_device_capability(100)
        and torch.get_default_dtype() == torch.bfloat16
        and attention.kv_cache_dtype in ("fp8", "fp8_e4m3")
        and not getattr(attention, "calculate_kv_scales", False)
        and attention.query_quant is not None
        and getattr(attention.impl, "vllm_flash_attn_version", None) == 4
        and attention._q_scale.numel()
        == attention._k_scale.numel()
        == attention._v_scale.numel()
        == 1
    )


def _yoco_can_fuse_fp8_output(
    projection: RowParallelLinear, execution_mode: str
) -> bool:
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PerBlockOnlineLinearMethod,
    )

    method = projection.quant_method
    return (
        execution_mode == "fast"
        and envs.VLLM_YOCO_FP8_ATTENTION_FUSION
        and projection.tp_size == 1
        and projection.bias is None
        and isinstance(method, Fp8PerBlockOnlineLinearMethod)
        and method.supports_silu_mul_fusion()
    )


def _yoco_project_fp8_output(
    projection: RowParallelLinear,
    attention: torch.Tensor,
    gate: torch.Tensor,
    heads: int,
    dim: int,
) -> torch.Tensor:
    # This path is admitted only for bias-free TP1; TP>1 retains RowParallelLinear.
    q, scales = torch.ops.vllm.yoco_diff_attention_fp8(
        attention.view(-1, 2 * heads, dim), gate
    )
    method = cast("Fp8PerBlockOnlineLinearMethod", projection.quant_method)
    kernel = cast("Fp8BlockScaledMMLinearKernel", method.fp8_linear)
    return kernel.apply_quantized_weights(projection, q, scales)


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
    output_dtype: torch.dtype = torch.float32,
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
        if output_dtype == torch.bfloat16:
            return torch.ops.vllm.yoco_lm_head_bf16(hidden_states, weight)
        return torch.ops.vllm.yoco_lm_head(hidden_states, weight)
    logits = F.linear(hidden_states, weight)
    if output_dtype == torch.bfloat16:
        return logits.to(torch.bfloat16)
    return logits.float() if use_sm100_kernel else logits


def _supports_yoco_sm100_lm_head_kernel(execution_mode: str) -> bool:
    if execution_mode != "fast" or not HAS_TRITON or not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major == 10


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


if HAS_TRITON and current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_diff_attention_v3",
        op_func=_yoco_diff_attention_v3_cuda,
        fake_impl=_yoco_diff_attention_v3_fake,
    )


if HAS_TRITON and current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_lm_head",
        op_func=_yoco_lm_head_cuda,
        fake_impl=_yoco_lm_head_fake,
    )


if HAS_TRITON and current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_lm_head_bf16",
        op_func=_yoco_lm_head_bf16_cuda,
        fake_impl=_yoco_lm_head_bf16_fake,
    )


if HAS_TRITON and current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_fused_shared_gate_moe_output",
        op_func=_yoco_fused_shared_gate_moe_output_cuda,
        fake_impl=_yoco_fused_shared_gate_moe_output_fake,
    )


if HAS_TRITON and current_platform.is_cuda():
    direct_register_custom_op(
        op_name="yoco_clip_fp8",
        op_func=_yoco_clip_fp8,
        fake_impl=_yoco_clip_fp8_fake,
    )
