#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark YOCO Fast's fused multi-KV-group slot mapping."""

from __future__ import annotations

import argparse
import math

import torch

from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.worker.block_table import MultiGroupBlockTable


def capture(fn) -> torch.cuda.CUDAGraph:
    fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.accelerator.synchronize()
    return graph


def graph_latency_us(graph: torch.cuda.CUDAGraph, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=31)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--position", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=2000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_grad_enabled(False)
    block_sizes = [args.block_size] * args.groups
    table = MultiGroupBlockTable(
        max_num_reqs=args.batch,
        max_model_len=max(8192, args.position + 1),
        max_num_batched_tokens=max(8192, args.batch),
        pin_memory=is_pin_memory_available(),
        device=torch.device("cuda"),
        block_sizes=block_sizes,
        kernel_block_sizes=block_sizes,
        use_yoco_fused_slot_mapping=True,
    )
    assert table.yoco_fused_slot_mapping is not None

    blocks_per_req = math.ceil((args.position + 1) / args.block_size)
    for req_idx in range(args.batch):
        table.add_row(
            tuple(
                [
                    (group_idx * args.batch + req_idx) * blocks_per_req + block_idx
                    for block_idx in range(blocks_per_req)
                ]
                for group_idx in range(args.groups)
            ),
            req_idx,
        )
    table.commit_block_table(args.batch)
    query_start_loc = torch.arange(args.batch + 1, dtype=torch.int32, device="cuda")
    positions = torch.full(
        (args.batch,), args.position, dtype=torch.int64, device="cuda"
    )

    def baseline() -> None:
        for block_table in table.block_tables:
            block_table.compute_slot_mapping(args.batch, query_start_loc, positions)

    def fused() -> None:
        table.compute_slot_mapping(args.batch, query_start_loc, positions)

    baseline()
    reference = [x.slot_mapping.gpu.clone() for x in table.block_tables]
    fused()
    assert all(
        torch.equal(x.slot_mapping.gpu, expected)
        for x, expected in zip(table.block_tables, reference)
    )

    baseline_graph = capture(baseline)
    fused_graph = capture(fused)
    baseline_us = graph_latency_us(baseline_graph, args.iterations)
    fused_us = graph_latency_us(fused_graph, args.iterations)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"groups={args.groups} batch={args.batch} position={args.position}")
    print(f"baseline: {baseline_us:.3f} us")
    print(f"fused:    {fused_us:.3f} us")
    print(f"speedup:  {baseline_us / fused_us:.3f}x")


if __name__ == "__main__":
    main()
