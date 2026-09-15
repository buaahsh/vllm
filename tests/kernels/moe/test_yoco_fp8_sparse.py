# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed FP8 quantization must preserve bytes and tolerate graph reuse."""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    silu_mul_quant_fp8_packed_triton,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("routes", [1, 8, 31, 128])
@pytest.mark.parametrize("width", [128, 3840])
@pytest.mark.parametrize("weighted", [False, True])
def test_routed_quant_preserves_valid_bytes_and_guard_rows(routes, width, weighted):
    torch.manual_seed(9209)
    rows = 1024
    # Permuted, noncontiguous destinations; extra invalid routes model EP.
    indices = torch.cat(
        [
            torch.randperm(rows, device="cuda")[:routes],
            torch.tensor([-1, rows, -9], device="cuda"),
        ]
    ).int()
    x = torch.randn(rows, 2 * width, device="cuda", dtype=torch.bfloat16) * 16
    positive = torch.rand(routes + 3, device="cuda") if weighted else None
    negative = torch.rand_like(positive) if weighted else None
    positive_dense, negative_dense = None, None
    if weighted:
        assert positive is not None and negative is not None
        positive[0] = 0
        positive_dense = torch.zeros(rows, device="cuda")
        negative_dense = torch.zeros_like(positive_dense)
        positive_dense[indices[:routes].long()] = positive[:routes]
        negative_dense[indices[:routes].long()] = negative[:routes]
    ref_q, ref_s = silu_mul_quant_fp8_packed_triton(
        x,
        clamp_limit=10.0,
        row_weights=positive_dense,
        negative_row_weights=negative_dense,
    )
    storage = torch.full(
        (rows + 2, width), 32, device="cuda", dtype=torch.float8_e4m3fn
    )
    q, s = silu_mul_quant_fp8_packed_triton(
        x,
        output_q=storage[:rows],
        clamp_limit=10.0,
        row_weights=positive,
        negative_row_weights=negative,
        row_indices=indices,
    )
    valid = indices[:routes].long()
    assert torch.equal(q.view(torch.uint8)[valid], ref_q.view(torch.uint8)[valid])
    assert torch.equal(s[valid], ref_s[valid])
    untouched = torch.ones(rows + 2, device="cuda", dtype=torch.bool)
    untouched[valid] = False
    assert bool((storage.float()[untouched] == 32).all())


def test_routed_quant_graph_replay_changes_routes_and_ignores_poison_padding():
    torch.manual_seed(9210)
    rows, routes, width = 1024, 8, 3840
    x = torch.randn(rows, 2 * width, device="cuda", dtype=torch.bfloat16)
    indices = (torch.arange(routes, device="cuda") * 128).int()
    weights = torch.rand(routes, device="cuda")

    def run():
        return silu_mul_quant_fp8_packed_triton(
            x, clamp_limit=10.0, row_weights=weights, row_indices=indices
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, s = run()
    for offset in [0, 7, 3]:
        indices.copy_((torch.arange(routes, device="cuda") * 128 + offset).int())
        weights.copy_(torch.rand_like(weights))
        if offset == 3:
            indices[0] = -1
        valid = indices >= 0
        dest = indices[valid].long()
        x.fill_(float("nan"))
        x[dest] = torch.randn(len(dest), 2 * width, device="cuda").bfloat16() * 16
        graph.replay()
        dense_weights = torch.zeros(rows, device="cuda")
        dense_weights[dest] = weights[valid]
        ref_q, ref_s = silu_mul_quant_fp8_packed_triton(
            x, clamp_limit=10.0, row_weights=dense_weights
        )
        assert torch.equal(q.view(torch.uint8)[dest], ref_q.view(torch.uint8)[dest])
        assert torch.equal(s[dest], ref_s[dest])
        assert bool(torch.isfinite(q.float()[dest]).all())


@pytest.mark.parametrize("tokens", [1, 7, 33])
@pytest.mark.parametrize("nonlocal_experts", [False, True])
def test_deepgemm_sparse_matches_dense_pipeline(
    monkeypatch, workspace_init, tokens, nonlocal_experts
):
    from tests.kernels.moe.test_deepgemm import make_block_quant_fp8_weights
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )
    from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
    from vllm.model_executor.layers.fused_moe.experts import deep_gemm_moe
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

    if not is_deep_gemm_e8m0_used():
        pytest.skip("requires DeepGEMM UE8M0")
    torch.manual_seed(9211)
    experts, topk, hidden, intermediate = 8, 4, 1024, 3840
    w1, w2, s1, s2 = make_block_quant_fp8_weights(
        experts, intermediate, hidden, [128, 128]
    )
    quant = fp8_w8a8_moe_quant_config(w1_scale=s1, w2_scale=s2, block_shape=[128, 128])
    config = make_dummy_moe_config(experts, topk, hidden, intermediate)
    impl = deep_gemm_moe.DeepGemmExperts(config, quant)
    impl.swiglu_limit = 10.0
    impl.apply_router_weight_before_w2 = True
    moe = FusedMoEKernel(
        prepare_finalize=maybe_make_prepare_finalize(
            moe=config,
            quant_config=quant,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        fused_experts=impl,
        inplace=False,
    )
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    ids = torch.rand(tokens, experts, device="cuda").topk(topk).indices.int()
    weights = torch.softmax(torch.randn(tokens, topk, device="cuda"), dim=-1)
    expert_map = None
    if nonlocal_experts:
        expert_map = torch.arange(experts, device="cuda", dtype=torch.int32)
        expert_map[1::2] = -1

    def run():
        return moe.apply(
            hidden_states=x,
            w1=w1,
            w2=w2,
            topk_weights=weights,
            topk_ids=ids,
            global_num_experts=experts,
            activation=config.activation,
            apply_router_weight_on_input=False,
            expert_map=expert_map,
        )

    original_quant = deep_gemm_moe.fused_silu_mul_fp8_quant_packed

    def dense_reference(**kwargs):
        indices = kwargs.pop("row_indices")
        for name in ("row_weights", "negative_row_weights"):
            routed = kwargs[name]
            dense = torch.zeros(kwargs["input"].shape[0], device="cuda")
            deep_gemm_moe._scatter_routed_row_weights(
                dense, ids, routed.view_as(weights), indices.view_as(ids)
            )
            kwargs[name] = dense
        return original_quant(**kwargs)

    # The established full-buffer quantizer is the reference; every other
    # stage, including actual W2 GEMM and top-k reduction, is exercised.
    with monkeypatch.context() as context:
        context.setattr(
            deep_gemm_moe, "fused_silu_mul_fp8_quant_packed", dense_reference
        )
        expected = run().clone()
    actual = run().clone()
    assert torch.equal(actual, expected)
    assert bool(torch.isfinite(actual).all())

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    reference_graph = torch.cuda.CUDAGraph()
    with monkeypatch.context() as context:
        context.setattr(
            deep_gemm_moe, "fused_silu_mul_fp8_quant_packed", dense_reference
        )
        with torch.cuda.graph(reference_graph):
            captured_reference = run()
    for _ in range(2):
        ids.copy_(torch.rand(tokens, experts, device="cuda").topk(topk).indices.int())
        weights.copy_(torch.softmax(torch.randn_like(weights), dim=-1))
        x.copy_(torch.randn_like(x))
        graph.replay()
        actual = captured.clone()
        reference_graph.replay()
        expected = captured_reference.clone()
        assert torch.equal(actual, expected)
        assert bool(torch.isfinite(actual).all())
