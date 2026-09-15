# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fast selects logits directly; Align retains the full-softmax boundary."""

import pytest
import torch

from vllm.model_executor.layers.yoco_ops import routing as yoco

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


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
