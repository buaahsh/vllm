# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the FP8 Triton fallback's routing/quantization boundary on CUDA."""

import pytest
import torch
from torch.nn import functional as F

from tests.kernels.moe.utils import (
    make_dummy_moe_config,
    make_test_weights,
    modular_triton_fused_moe,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts import triton_moe


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FP8")
@pytest.mark.parametrize("tokens", [1, 7, 33])
@pytest.mark.parametrize("block_shape", [None, [128, 128]])
def test_fp8_triton_weights_before_quantization(
    monkeypatch, workspace_init, tokens, block_shape
):
    torch.manual_seed(9208)
    experts, hidden, intermediate, topk = 4, 128, 256, 2
    (_, w1, s1, _), (_, w2, s2, _) = make_test_weights(
        experts,
        intermediate,
        hidden,
        quant_dtype=torch.float8_e4m3fn,
        block_shape=block_shape,
    )
    quant = fp8_w8a8_moe_quant_config(w1_scale=s1, w2_scale=s2, block_shape=block_shape)
    config = make_dummy_moe_config(experts, topk, hidden, intermediate)
    moe = modular_triton_fused_moe(config, quant)
    moe.fused_experts.swiglu_limit = 10.0
    moe.fused_experts.apply_router_weight_before_w2 = True
    x = (
        torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 32
    ).contiguous()
    ids = (
        torch.arange(tokens * topk, device="cuda").view(tokens, topk) % experts
    ).int()
    weights = torch.linspace(0.03, 0.91, tokens * topk, device="cuda").view(
        tokens, topk
    )
    observed = {}
    invoke = triton_moe.invoke_fused_moe_triton_kernel
    quantize = triton_moe.moe_kernel_quantize_input

    def observe_gemm(*args, **kwargs):
        invoke(*args, **kwargs)
        if args[1] is w1:
            observed["w13"] = args[2].clone().reshape(-1, 2 * intermediate)
        else:
            observed["weight_in_epilogue"] = args[9]

    def observe_quantize(a, *args, **kwargs):
        observed["w2_input"] = a.clone()
        return quantize(a, *args, **kwargs)

    monkeypatch.setattr(triton_moe, "invoke_fused_moe_triton_kernel", observe_gemm)
    monkeypatch.setattr(triton_moe, "moe_kernel_quantize_input", observe_quantize)
    output = moe.apply(
        hidden_states=x,
        w1=w1,
        w2=w2,
        topk_weights=weights,
        topk_ids=ids,
        activation=MoEActivation.SILU,
        global_num_experts=experts,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    gate, up = observed["w13"].float().chunk(2, dim=-1)
    unweighted = F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)
    expected = (unweighted * weights.reshape(-1, 1)).to(torch.bfloat16)
    torch.testing.assert_close(observed["w2_input"], expected, rtol=0.008, atol=0.001)
    assert not torch.equal(observed["w2_input"], unweighted.to(torch.bfloat16))
    assert observed["weight_in_epilogue"] is False
    assert torch.isfinite(output).all()

    # An ordinary FP8 MoE without the YOCO flag retains epilogue weighting.
    moe.fused_experts.apply_router_weight_before_w2 = False
    moe.apply(
        hidden_states=x,
        w1=w1,
        w2=w2,
        topk_weights=weights,
        topk_ids=ids,
        activation=MoEActivation.SILU,
        global_num_experts=experts,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    assert observed["weight_in_epilogue"] is True
