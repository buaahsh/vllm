# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B200 CUDA-graph benchmark for the YOCO L3 MoE router GEMM.

The training shape is ``[tokens, 3072] @ [3072, 128]`` in FP32.  This
compares the real token shape with the historical pad-to-128 policy under
both IEEE FP32 (training default) and TF32 (serving fast path).
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch
import torch.nn.functional as F

from vllm.platforms import current_platform


def _capture(
    fn: Callable[[], torch.Tensor], graph_nodes: int
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
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
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
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
        default=[1, 2, 4, 8, 16, 32, 64, 127, 128, 256],
    )
    parser.add_argument("--graph-nodes", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(20260831)
    device = torch.device("cuda")
    max_tokens = max(max(args.tokens), 128)
    hidden_all = torch.randn(
        max_tokens, 3072, dtype=torch.bfloat16, device=device
    ).float()
    weight = torch.randn(128, 3072, dtype=torch.float32, device=device)
    weight = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-6)

    print(
        f"device={current_platform.get_device_name()} torch={torch.__version__} "
        f"cuda={torch.version.cuda}"
    )
    print(
        "tokens  ieee_actual_us  ieee_pad128_us  tf32_actual_us  "
        "tf32_pad128_us  ieee_pad_max_abs  tf32_pad_max_abs  "
        "tf32_vs_ieee_max_abs  tf32_row0_batch_exact"
    )

    row0_by_tokens: dict[int, torch.Tensor] = {}
    for tokens in args.tokens:
        hidden = hidden_all[:tokens]
        pad_rows = max(0, 128 - tokens)

        def actual(hidden=hidden, weight=weight) -> torch.Tensor:
            return F.linear(hidden, weight)

        def padded(
            hidden=hidden, weight=weight, pad_rows=pad_rows, tokens=tokens
        ) -> torch.Tensor:
            if pad_rows == 0:
                return F.linear(hidden, weight)
            return F.linear(F.pad(hidden, (0, 0, 0, pad_rows)), weight)[:tokens]

        graphs: list[torch.cuda.CUDAGraph] = []
        outputs: list[torch.Tensor] = []
        for precision, fn in (
            ("highest", actual),
            ("highest", padded),
            ("high", actual),
            ("high", padded),
        ):
            torch.set_float32_matmul_precision(precision)
            torch.backends.cuda.matmul.fp32_precision = (
                "ieee" if precision == "highest" else "tf32"
            )
            graph, output = _capture(fn, args.graph_nodes)
            graphs.append(graph)
            outputs.append(output)

        samples: list[list[float]] = [[] for _ in graphs]
        for round_index in range(args.rounds):
            order = list(range(len(graphs)))
            if round_index % 2:
                order.reverse()
            for graph_index in order:
                samples[graph_index].append(
                    _time_graph(graphs[graph_index], args.repeats, args.graph_nodes)
                )
        times = [statistics.median(values) for values in samples]

        ieee_actual, ieee_pad, tf32_actual, tf32_pad = outputs
        row0_by_tokens[tokens] = tf32_actual[0].clone()
        row0_exact = torch.equal(row0_by_tokens[tokens], row0_by_tokens[args.tokens[0]])
        print(
            f"{tokens:6d}  "
            + "  ".join(f"{value:14.3f}" for value in times)
            + f"  {(ieee_actual - ieee_pad).abs().max().item():16.8f}"
            + f"  {(tf32_actual - tf32_pad).abs().max().item():16.8f}"
            + f"  {(tf32_actual - ieee_actual).abs().max().item():20.8f}"
            + f"  {str(row0_exact):>21}"
        )


if __name__ == "__main__":
    main()
