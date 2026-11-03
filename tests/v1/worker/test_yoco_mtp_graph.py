# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check draft input semantics with real CUDA replay and changing request slots."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    prepare_prefill_inputs,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_yoco_draft_inputs_replay_accept_reject_and_request_slot_reuse():
    device = "cuda"
    max_reqs, max_tokens = 8, 16

    def buffer(size, dtype=torch.int32):
        return torch.zeros(size, dtype=dtype, device=device)

    target = SimpleNamespace(
        num_reqs=3,
        input_ids=buffer(max_tokens),
        positions=buffer(max_tokens, torch.int64),
        idx_mapping=buffer(max_reqs),
        query_start_loc=buffer(max_reqs + 1),
        seq_lens=buffer(max_reqs),
    )
    draft = SimpleNamespace(
        input_ids=buffer(max_tokens),
        positions=buffer(max_tokens, torch.int64),
        query_start_loc=buffer(max_reqs + 1),
        seq_lens=buffer(max_reqs),
    )
    last_indices = buffer(max_reqs, torch.int64)
    current_step = buffer((), torch.int64)
    sampled, rejected = buffer(max_reqs), buffer(max_reqs)
    last_sampled, next_prefill = buffer(max_reqs), buffer(max_reqs)

    def assign(tensor, values):
        tensor.zero_()
        tensor[: len(values)].copy_(torch.tensor(values, device=device))

    def run():
        prepare_prefill_inputs(
            last_indices,
            current_step,
            draft,
            target,
            sampled,
            rejected,
            last_sampled,
            next_prefill,
            max_reqs,
        )

    assign(target.input_ids, [10, 11, 20, 21, 30, 31, 32])
    assign(target.positions, [5, 6, 9, 10, 100, 101, 102])
    assign(target.idx_mapping, [6, 1, 4])
    assign(target.query_start_loc, [0, 2, 4, 7])
    assign(target.seq_lens, [7, 11, 103])
    assign(sampled, [2, 2, 0])
    assign(last_sampled, [100, 111, 122, 133, 144, 155, 166, 177])
    assign(next_prefill, [200, 211, 222, 233, 244, 255, 266, 277])
    run()
    torch.accelerator.synchronize()
    graph_three = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_three):
        run()

    # Rejection and new sampled tokens change after capture, on the device.
    assign(rejected, [0, 1, 0])
    assign(sampled, [2, 1, 0])
    current_step.fill_(7)
    target_before = target.input_ids.clone()
    graph_three.replay()
    assert last_indices.tolist() == [1, 2, 6, 0, 0, 0, 0, 0]
    assert draft.input_ids[[0, 1, 2, 4, 5, 6]].tolist() == [11, 166, 111, 31, 32, 244]
    assert draft.positions[[0, 1, 2, 4, 5, 6]].tolist() == [5, 6, 9, 100, 101, 102]
    assert draft.seq_lens.tolist() == [7, 11, 103, 0, 0, 0, 0, 0]
    assert draft.query_start_loc.tolist() == [0, 2, 4, 7, 7, 7, 7, 7, 7]
    assert current_step.item() == 0
    assert torch.equal(target.input_ids, target_before)

    # A smaller batch reuses a different request slot after the other requests end.
    target.num_reqs = 1
    assign(target.input_ids, [90])
    assign(target.positions, [3])
    assign(target.idx_mapping, [2])
    assign(target.query_start_loc, [0, 1])
    assign(target.seq_lens, [4])
    assign(sampled, [1])
    rejected.zero_()
    run()
    torch.accelerator.synchronize()
    graph_one = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_one):
        run()
    last_sampled[2] = 99
    graph_one.replay()
    assert draft.input_ids[0].item() == 99
    assert draft.positions[0].item() == 3
    assert last_indices.tolist() == [0] * max_reqs
    assert draft.seq_lens.tolist() == [4, 0, 0, 0, 0, 0, 0, 0]
    assert draft.query_start_loc.tolist() == [0] + [1] * max_reqs
