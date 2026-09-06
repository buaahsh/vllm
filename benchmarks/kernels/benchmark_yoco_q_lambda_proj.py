# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B200 CUDA-graph benchmark for YOCO L3 Q/QKV+lambda projection."""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F


def _capture(
    fn: Callable[[], Any], graph_nodes: int
) -> tuple[torch.cuda.CUDAGraph, Any]:
    for _ in range(5):
        output = fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_nodes):
            output = fn()
    return graph, output


def _time_graph(graph: torch.cuda.CUDAGraph, repeats: int, graph_nodes: int) -> float:
    for _ in range(10):
        graph.replay()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / (repeats * graph_nodes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1410, 2048, 4096],
    )
    parser.add_argument("--graph-nodes", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--projection",
        choices=("cross-q", "self-qkv"),
        default="cross-q",
    )
    parser.add_argument("--weight-scale", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(20260901)
    device = torch.device("cuda")
    hidden_size = 3072
    q_size = 8192 if args.projection == "cross-q" else 10240
    lambda_size = 64
    max_tokens = max(args.tokens)
    inputs = torch.randn(
        max_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    q_weight = torch.randn(
        q_size,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    lambda_weight = torch.randn(
        lambda_size,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    q_weight.mul_(args.weight_scale)
    lambda_weight.mul_(args.weight_scale)
    merged_weight = torch.cat((q_weight, lambda_weight))

    print(
        f"device={torch.cuda.get_device_name()} torch={torch.__version__} "
        f"cuda={torch.version.cuda} projection={args.projection} "
        f"weight_scale={args.weight_scale}"
    )
    print(
        "tokens  separate_us  merged_us  speedup_pct  q_exact  lambda_exact  "
        "q_max_abs  lambda_max_abs"
    )

    for tokens in args.tokens:
        x = inputs[:tokens]

        def separate(x=x) -> tuple[torch.Tensor, torch.Tensor]:
            return F.linear(x, q_weight), F.linear(x, lambda_weight)

        def merged(x=x) -> torch.Tensor:
            return F.linear(x, merged_weight)

        expected_q, expected_lambda = separate()
        actual_q, actual_lambda = merged().split((q_size, lambda_size), dim=-1)
        q_exact = torch.equal(actual_q, expected_q)
        lambda_exact = torch.equal(actual_lambda, expected_lambda)
        q_max_abs = (actual_q - expected_q).abs().max().item()
        lambda_max_abs = (actual_lambda - expected_lambda).abs().max().item()

        separate_graph, _ = _capture(separate, args.graph_nodes)
        merged_graph, _ = _capture(merged, args.graph_nodes)
        graphs = (separate_graph, merged_graph)
        samples: list[list[float]] = [[], []]
        for round_index in range(args.rounds):
            order = (0, 1) if round_index % 2 == 0 else (1, 0)
            for graph_index in order:
                samples[graph_index].append(
                    _time_graph(graphs[graph_index], args.repeats, args.graph_nodes)
                )
        separate_us, merged_us = (statistics.median(values) for values in samples)
        speedup_pct = 100.0 * (separate_us / merged_us - 1.0)
        print(
            f"{tokens:6d}  {separate_us:11.3f}  {merged_us:9.3f}  "
            f"{speedup_pct:11.2f}  {str(q_exact):>7}  "
            f"{str(lambda_exact):>12}  {q_max_abs:9.6f}  "
            f"{lambda_max_abs:14.6f}"
        )


if __name__ == "__main__":
    main()
