# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check direct quantization against llm-train's routed activation contract."""

import pytest
import torch

from vllm.model_executor.layers.yoco_ops.fp8 import (
    silu_mul_quant_fp8_packed_triton,
    silu_mul_quant_fp8_triton,
)
from vllm.triton_utils import tl, triton

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@triton.jit
def _training_reference(X, W, Q, S, N: tl.constexpr, B: tl.constexpr):
    # Independent row-wise reference from llm-train/kernel/quant.py:
    # FP32 weighted SwiGLU -> UE8M0 scale -> E4M3, without a BF16 round.
    row = tl.program_id(0)
    cols = tl.arange(0, B)
    gate = tl.load(X + row * 2 * N + cols, cols < N, 0).to(tl.float32)
    up = tl.load(X + row * 2 * N + N + cols, cols < N, 0).to(tl.float32)
    gate = tl.minimum(gate, 10.0)
    up = tl.clamp(up, -10.0, 10.0)
    y = gate * tl.sigmoid(gate) * up * tl.load(W + row).to(tl.float32)
    y = tl.reshape(y, (B // 128, 128))
    raw = tl.maximum(tl.max(tl.abs(y), axis=1), 1e-4) / 448.0
    bits = raw.to(tl.uint32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
    exponent = tl.minimum(tl.maximum(exponent, 1), 254)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    q = tl.reshape(tl.clamp(y / scale[:, None], -448.0, 448.0), (B,))
    tl.store(Q + row * N + cols, q, cols < N)
    groups = tl.arange(0, B // 128)
    tl.store(S + row * (N // 128) + groups, scale, groups < N // 128)


def training_reference(x, weights):
    rows, width = x.shape[0], x.shape[1] // 2
    q = torch.empty(rows, width, device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(rows, width // 128, device=x.device)
    _training_reference[(rows,)](
        x, weights, q, scale, width, triton.next_power_of_2(width)
    )
    return q, scale


@pytest.mark.parametrize("rows,width", [(1, 128), (8, 3840), (31, 1280), (128, 3840)])
@pytest.mark.parametrize("amplitude", [0.001, 1.0, 32.0])
def test_direct_fp8_matches_training_and_scale_layouts(rows, width, amplitude):
    torch.manual_seed(9310)
    x = torch.randn(rows, 2 * width, device="cuda", dtype=torch.bfloat16) * amplitude
    weights = torch.rand(rows, device="cuda")
    weights[0] = 0
    expected, expected_scale = training_reference(x, weights)
    outputs = []
    for packed in [False, True]:
        q, scale = silu_mul_quant_fp8_triton(
            x,
            clamp_limit=10.0,
            row_weights=weights,
            round_before_quant=False,
            packed_scales=packed,
        )
        if packed:
            groups = torch.arange(width // 128, device=x.device)
            exponents = (scale[:, groups // 4] >> ((groups % 4) * 8)) & 255
            scale = (exponents.int() << 23).view(torch.float32)
        assert torch.equal(q.view(torch.uint8), expected.view(torch.uint8))
        assert torch.equal(scale, expected_scale)
        outputs.append(q.view(torch.uint8))
    assert torch.equal(*outputs)


def test_direct_quant_graph_replay_routes_and_preserves_shared_boundary():
    torch.manual_seed(9311)
    rows, width = 256, 3840
    x = torch.randn(rows, 2 * width, device="cuda", dtype=torch.bfloat16) * 4
    indices = torch.arange(8, device="cuda", dtype=torch.int32) * 16
    weights = torch.rand(8, device="cuda")
    storage = torch.full(
        (rows + 1, width), 32, device="cuda", dtype=torch.float8_e4m3fn
    )

    def run():
        return silu_mul_quant_fp8_packed_triton(
            x,
            output_q=storage[:rows],
            clamp_limit=10.0,
            row_weights=weights,
            row_indices=indices,
            round_before_quant=False,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, _ = run()
    for shift in [1, 7]:
        storage.fill_(32)
        indices.copy_(torch.arange(8, device="cuda", dtype=torch.int32) * 16 + shift)
        indices[0] = -1
        weights.copy_(torch.rand_like(weights))
        valid = indices[1:].long()
        x.fill_(float("nan"))
        x[valid] = torch.randn(7, 2 * width, device="cuda").bfloat16() * 4
        graph.replay()
        expected, _ = training_reference(x[valid].contiguous(), weights[1:])
        assert torch.equal(q.view(torch.uint8)[valid], expected.view(torch.uint8))
        untouched = torch.ones(rows + 1, device="cuda", dtype=torch.bool)
        untouched[valid] = False
        assert (storage.float()[untouched] == 32).all()

    # The shared-expert caller keeps the default BF16 activation boundary.
    dense = x[valid].contiguous()
    default, ds = silu_mul_quant_fp8_packed_triton(dense, clamp_limit=10.0)
    rounded, rs = silu_mul_quant_fp8_triton(
        dense, clamp_limit=10.0, round_before_quant=True
    )
    direct, _ = silu_mul_quant_fp8_triton(
        dense, clamp_limit=10.0, round_before_quant=False
    )
    assert torch.equal(default.view(torch.uint8), rounded.view(torch.uint8))
    assert torch.equal(ds, rs)
    assert not torch.equal(default.view(torch.uint8), direct.view(torch.uint8))


def test_backend_policy_preserves_direct_quantization_after_rebuild():
    from dataclasses import replace

    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.config.yoco import YocoMoEPolicy
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        DeepGemmExperts,
    )
    from vllm.model_executor.layers.yoco_ops.fp8_moe import _act_mul_quant
    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

    if not is_deep_gemm_e8m0_used():
        pytest.skip("requires DeepGEMM UE8M0")
    torch.manual_seed(9312)
    rows, width = 31, 1280
    x = torch.randn(rows, 2 * width, device="cuda", dtype=torch.bfloat16) * 4
    weights = torch.rand(rows, device="cuda")
    config = make_dummy_moe_config(hidden_dim=width, intermediate_size=width)
    config.yoco = YocoMoEPolicy.for_mode("fast")
    config.apply_router_weight_before_w2 = True
    quant = fp8_w8a8_moe_quant_config(
        w1_scale=torch.ones(1, 2 * width // 128, width // 128, device="cuda"),
        w2_scale=torch.ones(1, width // 128, width // 128, device="cuda"),
        block_shape=[128, 128],
        gemm1_clamp_limit=10.0,
    )
    expected, _ = training_reference(x, weights)
    rounded, _ = silu_mul_quant_fp8_packed_triton(
        x,
        clamp_limit=10.0,
        row_weights=weights,
        round_before_quant=True,
    )
    assert not torch.equal(expected.view(torch.uint8), rounded.view(torch.uint8))
    for enabled in (True, False, True):
        config.yoco = replace(config.yoco, direct_fp8_activation=enabled)
        # Online quantization/reload constructs another backend with this config.
        experts = DeepGemmExperts(config, quant)
        actual, _ = _act_mul_quant(
            experts,
            x,
            torch.empty_like(expected),
            MoEActivation.SILU,
            row_weights=weights,
        )
        reference = expected if enabled else rounded
        assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))
