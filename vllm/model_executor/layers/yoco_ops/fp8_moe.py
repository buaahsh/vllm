# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        DeepGemmExperts,
    )

import os

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    deepgemm_unpermute_and_reduce,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    per_token_group_quant_fp8_packed_for_deepgemm,
    silu_mul_per_token_group_quant_fp8_colmajor,
)
from vllm.model_executor.layers.yoco_ops.fp8 import (
    silu_mul_quant_fp8_packed_triton as fused_silu_mul_fp8_quant_packed,
)
from vllm.model_executor.layers.yoco_ops.fp8_permute import (
    compute_aligned_M,
    deepgemm_moe_permute,
)
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    DeepGemmQuantScaleFMT,
    get_mk_alignment_for_contiguous_layout,
    m_grouped_fp8_gemm_nt_contiguous,
)


@triton.jit
def _scatter_routed_row_weights_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    inv_perm_ptr,
    row_weights_ptr,
    numel,
    num_rows,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offsets < numel
    expert_ids = tl.load(topk_ids_ptr + offsets, mask=in_bounds, other=-1)
    valid = in_bounds & (expert_ids >= 0)
    rows = tl.load(inv_perm_ptr + offsets, mask=valid, other=0).to(tl.int64)
    valid &= (rows >= 0) & (rows < num_rows)
    weights = tl.load(topk_weights_ptr + offsets, mask=valid, other=0.0)
    tl.store(row_weights_ptr + rows, weights, mask=valid)


def _scatter_routed_row_weights(
    row_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_perm: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    _scatter_routed_row_weights_kernel[(triton.cdiv(numel, 256),)](
        topk_ids,
        topk_weights,
        inv_perm,
        row_weights,
        numel,
        row_weights.numel(),
        BLOCK_SIZE=256,
        num_warps=4,
        num_stages=1,
    )


def workspace_shapes(
    self: DeepGemmExperts,
    M: int,
    N: int,
    K: int,
    topk: int,
    global_num_experts: int,
    local_num_experts: int,
    expert_tokens_meta: mk.ExpertTokensMetadata | None,
    activation: MoEActivation,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    assert self.block_shape is not None
    block_m = self.block_shape[0]
    M_sum = compute_aligned_M(M, topk, local_num_experts, block_m, expert_tokens_meta)
    assert M_sum % block_m == 0

    activation_out_dim = self.adjust_N_for_activation(N, activation)
    workspace1 = (M_sum, max(activation_out_dim, K))
    workspace2 = (M_sum, max(N, K))
    output = (M, K)
    return (workspace1, workspace2, output)


def _act_mul_quant(
    self: DeepGemmExperts,
    input: torch.Tensor,
    output: torch.Tensor,
    activation: MoEActivation,
    row_weights: torch.Tensor | None = None,
    negative_row_weights: torch.Tensor | None = None,
    row_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert self.block_shape is not None
    block_k = self.block_shape[1]
    scale_fmt = DeepGemmQuantScaleFMT.from_oracle()

    M_sum, N = input.size()
    activation_out_dim = self.adjust_N_for_activation(N, activation)
    swiglu_limit = self.gemm1_clamp_limit
    if row_weights is not None:
        if (
            scale_fmt == DeepGemmQuantScaleFMT.UE8M0
            and activation == MoEActivation.SILU
        ):
            return fused_silu_mul_fp8_quant_packed(
                input=input,
                output_q=output,
                group_size=block_k,
                clamp_limit=swiglu_limit,
                row_weights=row_weights.contiguous(),
                negative_row_weights=(
                    negative_row_weights.contiguous()
                    if negative_row_weights is not None
                    else None
                ),
                row_indices=row_indices,
                round_before_quant=not self.moe_config.yoco.direct_fp8_activation,
            )
        act_out = torch.empty(
            (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
        )
        self.activation(activation, act_out, input)
        act_out.mul_(row_weights.to(act_out.dtype).unsqueeze(-1))
        if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
            return per_token_group_quant_fp8_packed_for_deepgemm(
                act_out,
                block_k,
                out_q=output,
            )
        return per_token_group_quant_fp8(
            act_out,
            block_k,
            eps=1e-4,
            column_major_scales=True,
            out_q=output,
        )

    # 1. DeepGemm UE8M0: fused SiLU+mul+clamp+quant+pack
    if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
        if activation == MoEActivation.SILU:
            return fused_silu_mul_fp8_quant_packed(
                input=input,
                output_q=output,
                group_size=block_k,
                clamp_limit=swiglu_limit,
            )
        act_out = torch.empty(
            (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
        )
        self.activation(activation, act_out, input)
        a2q, a2q_scale = per_token_group_quant_fp8_packed_for_deepgemm(
            act_out,
            block_k,
            out_q=output,
        )
        return a2q, a2q_scale

    # 2. Hopper / non‑E8M0: prefer the fused SiLU+mul+quant kernel
    if activation == MoEActivation.SILU:
        use_ue8m0 = scale_fmt == DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
        return silu_mul_per_token_group_quant_fp8_colmajor(
            input=input,
            output=output,
            use_ue8m0=use_ue8m0,
        )

    # 3. fallback path for non-SiLU activations in non‑UE8M0 cases.
    act_out = torch.empty(
        (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
    )
    self.activation(activation, act_out, input)
    return per_token_group_quant_fp8(
        act_out,
        block_k,
        eps=1e-4,
        column_major_scales=True,
        out_q=output,
    )


def apply(
    self: DeepGemmExperts,
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
    assert a1q_scale is not None
    assert a2_scale is None
    assert self.block_shape is not None
    assert self.w1_scale is not None
    assert self.w2_scale is not None

    a1q = hidden_states
    _, N, K = w1.size()

    local_num_experts = w1.size(0)
    if global_num_experts == -1:
        global_num_experts = local_num_experts

    assert w2.size(1) == K

    block_m = get_mk_alignment_for_contiguous_layout()[0]
    M_sum = compute_aligned_M(
        M=topk_ids.size(0),
        num_topk=topk_ids.size(1),
        local_num_experts=local_num_experts,
        alignment=block_m,
        expert_tokens_meta=expert_tokens_meta,
    )
    probs_before_w2 = self.moe_config.apply_router_weight_before_w2
    use_psum_layout = (
        probs_before_w2 or os.getenv("VLLM_DEEPGEMM_MOE_PSUM_LAYOUT") == "1"
    )
    pack_scales = (
        use_psum_layout
        and DeepGemmQuantScaleFMT.from_oracle() == DeepGemmQuantScaleFMT.UE8M0
    )

    a1q_perm = _resize_cache(workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, K))
    a1q, a1q_scale, grouped_layout, inv_perm = deepgemm_moe_permute(
        aq=a1q,
        aq_scale=a1q_scale,
        topk_ids=topk_ids,
        local_num_experts=local_num_experts,
        expert_map=expert_map,
        expert_tokens_meta=expert_tokens_meta,
        aq_out=a1q_perm,
        use_psum_layout=use_psum_layout,
        pack_scales=pack_scales,
    )
    if (
        use_psum_layout
        # Packed TMA scales use the static workspace pitch. The prefix
        # layout already bounds real expert blocks, so this path can keep
        # the full buffers and avoid synchronizing a device scalar to CPU.
        and not pack_scales
        and not torch.cuda.is_current_stream_capturing()
        and not torch.compiler.is_compiling()
    ):
        actual_m = int(grouped_layout[-1].item())
        a1q = a1q[:actual_m]
        a1q_scale = a1q_scale[:actual_m]
    assert use_psum_layout or a1q.size(0) == M_sum
    m_gemm = a1q.size(0)
    mm1_out = _resize_cache(workspace2, (m_gemm, N))
    grouped_gemm_kwargs = {}
    if use_psum_layout:
        grouped_gemm_kwargs.update(
            {
                "use_psum_layout": True,
                "expected_m_for_psum_layout": m_gemm,
            }
        )
    m_grouped_fp8_gemm_nt_contiguous(
        (a1q, a1q_scale),
        (w1, self.w1_scale),
        mm1_out,
        grouped_layout,
        **grouped_gemm_kwargs,
    )
    activation_out_dim = self.adjust_N_for_activation(N, activation)
    quant_out = _resize_cache(
        workspace13.view(dtype=torch.float8_e4m3fn), (m_gemm, activation_out_dim)
    )
    row_weights = None
    negative_row_weights = None
    row_indices = None
    if probs_before_w2:
        if (
            DeepGemmQuantScaleFMT.from_oracle() == DeepGemmQuantScaleFMT.UE8M0
            and activation == MoEActivation.SILU
        ):
            # inv_perm maps real routes to their padded expert rows. Read
            # routing weights directly and quantize only those rows; the
            # graph's static workspace shape and addresses stay unchanged.
            row_weights = topk_weights.reshape(-1)
            row_indices = inv_perm.reshape(-1)
        else:
            row_weights = torch.zeros(
                (m_gemm,), device=topk_weights.device, dtype=topk_weights.dtype
            )
            _scatter_routed_row_weights(row_weights, topk_ids, topk_weights, inv_perm)
        negative_row_weights = row_weights

    a2q, a2q_scale = _act_mul_quant(
        self,
        input=mm1_out.view(-1, N),
        output=quant_out,
        activation=activation,
        row_weights=row_weights,
        negative_row_weights=negative_row_weights,
        row_indices=row_indices,
    )
    mm2_out = _resize_cache(workspace2, (m_gemm, K))
    m_grouped_fp8_gemm_nt_contiguous(
        (a2q, a2q_scale),
        (w2, self.w2_scale),
        mm2_out,
        grouped_layout,
        **grouped_gemm_kwargs,
    )
    if apply_router_weight_on_input or probs_before_w2:
        topk_weights = torch.ones_like(topk_weights)

    deepgemm_unpermute_and_reduce(
        a=mm2_out,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        inv_perm=inv_perm,
        expert_map=expert_map,
        output=output,
    )
