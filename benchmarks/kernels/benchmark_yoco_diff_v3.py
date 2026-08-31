# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the YOCO L3 differential-attention combine kernel.

This models the TP=4 decode shape used by YOCO-MoE-30B-A3B-L3:
16 interleaved attention heads are reduced to 8 heads, with head_dim=128.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


def _training_diff_v3(output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    output = output * torch.sigmoid(gate).unsqueeze(-1)
    return output[:, 0::2] - output[:, 1::2]


def _vllm_diff_v3(
    attn1: torch.Tensor,
    attn2: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    gate1 = gate[:, 0::2]
    gate2 = gate[:, 1::2]
    return attn1 * torch.sigmoid(gate1).unsqueeze(-1) - attn2 * torch.sigmoid(
        gate2
    ).unsqueeze(-1)


@triton.jit
def _diff_v3_head_group_kernel(
    output_ptr,
    gate_ptr,
    result_ptr,
    num_head_pairs: tl.constexpr,
    head_dim: tl.constexpr,
    head_group: tl.constexpr,
    block_dim: tl.constexpr,
):
    group_idx = tl.program_id(0)
    groups_per_token: tl.constexpr = num_head_pairs // head_group
    token_idx = group_idx // groups_per_token
    first_pair_idx = (
        token_idx * num_head_pairs + (group_idx % groups_per_token) * head_group
    )

    head_offsets = tl.arange(0, head_group)[:, None]
    dim_offsets = tl.arange(0, block_dim)[None, :]
    mask = dim_offsets < head_dim
    pair_idx = first_pair_idx + head_offsets
    first_head = 2 * pair_idx
    first_offsets = first_head * head_dim + dim_offsets
    second_offsets = first_offsets + head_dim

    first_gate = tl.load(gate_ptr + first_head).to(tl.float32)
    second_gate = tl.load(gate_ptr + first_head + 1).to(tl.float32)
    first_scale = tl.sigmoid(first_gate)
    second_scale = tl.sigmoid(second_gate)
    first = tl.load(output_ptr + first_offsets, mask=mask).to(tl.float32)
    second = tl.load(output_ptr + second_offsets, mask=mask).to(tl.float32)
    result = first * first_scale - second * second_scale
    tl.store(result_ptr + pair_idx * head_dim + dim_offsets, result, mask=mask)


def _triton_diff_v3(
    output: torch.Tensor,
    gate: torch.Tensor,
    *,
    head_group: int,
    num_warps: int,
) -> torch.Tensor:
    num_tokens, twice_num_heads, head_dim = output.shape
    num_head_pairs = twice_num_heads // 2
    result = torch.empty(
        (num_tokens, num_head_pairs, head_dim),
        device=output.device,
        dtype=output.dtype,
    )
    assert num_head_pairs % head_group == 0
    _diff_v3_head_group_kernel[(num_tokens * num_head_pairs // head_group,)](
        output,
        gate,
        result,
        num_head_pairs=num_head_pairs,
        head_dim=head_dim,
        head_group=head_group,
        block_dim=triton.next_power_of_2(head_dim),
        num_warps=num_warps,
    )
    return result


def _capture(
    fn: Callable[[], torch.Tensor], graph_nodes: int
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    for _ in range(3):
        result = fn()
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_nodes):
            result = fn()
    return graph, result


def _time_graph(
    graph: torch.cuda.CUDAGraph, iterations: int, graph_nodes: int
) -> float:
    for _ in range(10):
        graph.replay()
    torch.accelerator.synchronize()
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (iterations * graph_nodes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batches", type=int, nargs="+", default=[1, 8, 32, 64, 128, 224, 256, 512]
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--graph-nodes", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--num-head-pairs", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(20260831)
    device = torch.device("cuda")
    print(f"device={current_platform.get_device_name()} torch={torch.__version__}")
    configs = ((1, 1), (2, 2), (4, 4), (8, 4), (8, 8))
    config_names = [f"g{group}w{warps}_us" for group, warps in configs]
    print(
        "batch  training_us  current_us  "
        + "  ".join(f"{name:>9}" for name in config_names)
        + "  max_abs"
    )

    for batch in args.batches:
        output = torch.randn(
            batch,
            2 * args.num_head_pairs,
            args.head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        gate = torch.randn(
            batch,
            2 * args.num_head_pairs,
            device=device,
            dtype=torch.bfloat16,
        )

        training_fn = torch.compile(_training_diff_v3, fullgraph=True)
        current_fn = torch.compile(_vllm_diff_v3, fullgraph=True)
        training_graph, expected = _capture(
            lambda training_fn=training_fn, output=output, gate=gate: training_fn(
                output, gate
            ),
            args.graph_nodes,
        )
        current_graph, actual = _capture(
            lambda current_fn=current_fn, output=output, gate=gate: current_fn(
                output[:, 0::2], output[:, 1::2], gate
            ),
            args.graph_nodes,
        )

        triton_graphs: list[torch.cuda.CUDAGraph] = []
        triton_results: list[torch.Tensor] = []
        for head_group, num_warps in configs:
            graph, result = _capture(
                lambda output=output,
                gate=gate,
                head_group=head_group,
                num_warps=num_warps: _triton_diff_v3(
                    output,
                    gate,
                    head_group=head_group,
                    num_warps=num_warps,
                ),
                args.graph_nodes,
            )
            triton_graphs.append(graph)
            triton_results.append(result)

        torch.accelerator.synchronize()
        assert torch.equal(actual, expected), f"compiled mismatch at batch={batch}"
        max_abs = max(
            (result.float() - expected.float()).abs().max().item()
            for result in triton_results
        )
        graphs = [training_graph, current_graph, *triton_graphs]
        samples: list[list[float]] = [[] for _ in graphs]
        for round_idx in range(args.rounds):
            order = list(range(len(graphs)))
            if round_idx % 2:
                order.reverse()
            order = order[round_idx % len(order) :] + order[: round_idx % len(order)]
            for graph_idx in order:
                samples[graph_idx].append(
                    _time_graph(graphs[graph_idx], args.iterations, args.graph_nodes)
                )
        times = [statistics.median(graph_samples) for graph_samples in samples]
        print(
            f"{batch:5d} "
            + " ".join(f"{elapsed:11.3f}" for elapsed in times)
            + f" {max_abs:8.6f}"
        )


if __name__ == "__main__":
    main()
