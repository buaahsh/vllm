# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fast selects logits directly; Align retains the full-softmax boundary."""

import pytest
import torch

from vllm.model_executor.layers.yoco_ops import routing as yoco
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="requires SM100"
)
@pytest.mark.parametrize("rows", [0, 128, 144, 288, 576, 1152, 2048])
def test_mega_gate_preserves_bf16_rounding_ties_and_negative_softmax(rows):
    # A single nonzero K term gives an exact GEMM reference, including a
    # near-tie that must be rounded before ranking and all-negative logits.
    x = torch.zeros(rows, 3072, dtype=torch.bfloat16, device="cuda")
    x[:, 0] = 1.0078125
    x[::2, 0] = 0
    weight = torch.zeros(128, 3072, dtype=torch.bfloat16, device="cuda")
    weight[:, 0] = -2048
    weight[:9, 0] = -1020
    weight[1, 0] = -1016
    logits = torch.nn.functional.linear(x, weight)
    expected_ids = logits.float().argsort(dim=-1, descending=True, stable=True)[:, :8]
    expected_weights = logits.float().gather(1, expected_ids).softmax(-1)
    weights, ids = torch.ops.vllm.yoco_mega_gate(x, weight)
    assert ids.dtype == torch.int32 and weights.dtype == torch.float32
    assert torch.equal(ids.long(), expected_ids)
    torch.testing.assert_close(weights, expected_weights, rtol=2e-6, atol=1e-7)


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="requires SM100"
)
def test_mega_gate_graph_replay_observes_input_and_weight_updates():
    x = torch.zeros(576, 3072, device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros(128, 3072, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.ops.vllm.yoco_mega_gate(x, weight)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            weights, ids = torch.ops.vllm.yoco_mega_gate(x, weight)
    torch.cuda.current_stream().wait_stream(stream)
    for scale in (0.0, 1.0, -1.0):
        x[:, 0] = scale
        weight[:, 0] = torch.arange(128, device="cuda", dtype=torch.bfloat16)
        graph.replay()
        expected_w, expected_i = yoco._yoco_topk_routing_impl(
            torch.nn.functional.linear(x, weight), 8, topk_logits=True
        )
        assert torch.equal(ids, expected_i)
        torch.testing.assert_close(weights, expected_w, rtol=2e-6, atol=1e-7)


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="requires SM100"
)
@pytest.mark.parametrize("rows", [128, 144, 576, 1152])
def test_mega_gate_masks_nonfinite_padding_when_graph_batch_shrinks(rows):
    torch.manual_seed(20261025)
    x = torch.randn(rows, 3072, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, 3072, device="cuda", dtype=torch.bfloat16) / 32
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.ops.vllm.yoco_mega_gate(x, weight)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual_weights, actual_ids = torch.ops.vllm.yoco_mega_gate(x, weight)
    torch.cuda.current_stream().wait_stream(stream)
    for active in (rows, rows - 9, rows - 18):
        x.copy_(torch.randn_like(x))
        clean = x.clone()
        clean[active:] = 0
        expected_weights, expected_ids = torch.ops.vllm.yoco_mega_gate(clean, weight)
        x[active:] = float("nan")
        if active < rows:
            x[active] = float("inf")
        graph.replay()
        assert torch.equal(actual_ids[:active], expected_ids[:active])
        assert torch.equal(actual_weights[:active], expected_weights[:active])
        assert (actual_ids[active:] == -1).all()
        assert (actual_weights[active:] == 0).all()


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="requires SM100"
)
def test_mega_gate_and_align_gemm_keep_independent_graph_outputs(monkeypatch):
    from vllm import envs
    from vllm.model_executor.layers.yoco_align_linear import align_gemm_backend
    from vllm.model_executor.layers.yoco_ops.projection import (
        _yoco_align_linear,
        _yoco_align_linear_legacy,
    )

    monkeypatch.setattr(envs, "VLLM_YOCO_ALIGN_GEMM_BACKEND", "cublaslt_v1")
    align_gemm_backend.cache_clear()
    try:
        torch.manual_seed(20261027)
        x = torch.randn(128, 3072, device="cuda", dtype=torch.bfloat16) / 8
        gate = torch.randn(128, 3072, device="cuda", dtype=torch.bfloat16) / 32
        dense = torch.randn(1024, 3072, device="cuda", dtype=torch.bfloat16) / 32
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                torch.ops.vllm.yoco_mega_gate(x, gate)
                _yoco_align_linear(x, dense)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                weights, ids = torch.ops.vllm.yoco_mega_gate(x, gate)
                output = _yoco_align_linear(x, dense)
        torch.cuda.current_stream().wait_stream(stream)
        for delta in (0.0, 0.125):
            x.add_(delta)
            gate.add_(delta / 8)
            dense.sub_(delta / 16)
            graph.replay()
            expected_weights, expected_ids = torch.ops.vllm.yoco_mega_gate(x, gate)
            expected_output = _yoco_align_linear_legacy(x, dense)
            assert torch.equal(weights, expected_weights)
            assert torch.equal(ids, expected_ids)
            assert torch.equal(
                output.view(torch.int16), expected_output.view(torch.int16)
            )
    finally:
        align_gemm_backend.cache_clear()


@pytest.mark.parametrize("scale", [0.05, 1.0, 8.0, 100.0])
def test_fast_logits_topk_matches_high_precision_reference(scale):
    torch.manual_seed(9310)
    logits = torch.randn(257, 128, device="cuda") * scale
    logits[0].zero_()
    logits[1] = torch.arange(128, device="cuda") * 1e-8
    ids = logits.double().argsort(dim=-1, descending=True, stable=True)[:, :8]
    expected = logits.double().gather(1, ids).softmax(-1).float()
    weights, actual_ids = yoco._yoco_topk_routing(logits, logits, 8, True)
    assert torch.equal(actual_ids.long(), ids)
    torch.testing.assert_close(weights, expected, rtol=2e-6, atol=1e-7)
    assert torch.isfinite(weights).all()


def test_fast_and_align_dispatch_separate_rounding_contracts(monkeypatch):
    observed = []

    def observe(logits, topk, *, topk_logits=False):
        observed.append(topk_logits)
        return torch.ones(1, 8, device="cuda"), torch.arange(8, device="cuda")[
            None
        ].int()

    monkeypatch.setattr(yoco, "_yoco_topk_routing_impl", observe)
    logits = torch.zeros(1, 128, device="cuda")
    yoco._yoco_topk_routing(logits, logits, 8, True)
    yoco._yoco_align_topk_routing(logits, logits, 8, True)
    assert observed == [True, False]


def test_fast_routing_graph_replay_and_batch_independence():
    torch.manual_seed(9312)
    logits = torch.randn(33, 128, device="cuda")

    def run():
        return yoco._yoco_topk_routing(logits, logits, 8, True)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, ids = run()
    for _ in range(3):
        logits.copy_(torch.randn_like(logits))
        graph.replay()
        for row in [0, 16, 32]:
            one = logits[row : row + 1].contiguous()
            ref_w, ref_i = yoco._yoco_topk_routing(one, one, 8, True)
            assert torch.equal(weights[row], ref_w[0])
            assert torch.equal(ids[row], ref_i[0])
