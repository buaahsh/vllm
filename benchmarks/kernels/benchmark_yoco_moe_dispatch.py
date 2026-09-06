# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B200 benchmark for YOCO's E=128, top-k=8 MoE token dispatch."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from vllm.platforms import current_platform

_EXTENSION = None


def _output_sizes(
    tokens: int, topk: int, experts: int, block_m: int
) -> tuple[int, int]:
    assignments = tokens * topk
    padded = assignments + experts * (block_m - 1)
    if assignments < experts:
        padded = min(assignments * block_m, padded)
    return padded, (padded + block_m - 1) // block_m


def _allocate_and_dispatch(
    topk_ids: torch.Tensor,
    experts: int,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sorted_size, expert_size = _output_sizes(
        topk_ids.shape[0], topk_ids.shape[1], experts, block_m
    )
    sorted_ids = torch.empty(sorted_size, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.empty(expert_size, dtype=torch.int32, device=topk_ids.device)
    post_pad = torch.empty(1, dtype=torch.int32, device=topk_ids.device)
    assert _EXTENSION is not None
    _EXTENSION.moe_align_block_size(
        topk_ids,
        experts,
        block_m,
        sorted_ids,
        expert_ids,
        post_pad,
        None,
    )
    return sorted_ids, expert_ids, post_pad


def _capture(
    topk_ids: torch.Tensor,
    experts: int,
    block_m: int,
    graph_nodes: int,
) -> tuple[torch.cuda.CUDAGraph, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    for _ in range(10):
        outputs = _allocate_and_dispatch(topk_ids, experts, block_m)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_nodes):
            outputs = _allocate_and_dispatch(topk_ids, experts, block_m)
    graph.replay()
    torch.accelerator.synchronize()
    return graph, outputs


def _time_graph(graph: torch.cuda.CUDAGraph, repeats: int, graph_nodes: int) -> float:
    for _ in range(20):
        graph.replay()
    torch.accelerator.synchronize()
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (repeats * graph_nodes)


def _validate(
    topk_ids: torch.Tensor,
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    experts: int,
    block_m: int,
) -> None:
    sorted_ids, expert_ids, post_pad = outputs
    assignments = topk_ids.numel()
    valid_length = int(post_pad.item())
    assert valid_length % block_m == 0
    actual_by_expert: list[list[int]] = [[] for _ in range(experts)]
    for block_start in range(0, valid_length, block_m):
        expert = int(expert_ids[block_start // block_m].item())
        assert 0 <= expert < experts
        for index in sorted_ids[block_start : block_start + block_m].tolist():
            if index < assignments:
                actual_by_expert[expert].append(index)
    flat_experts = topk_ids.flatten().tolist()
    for expert in range(experts):
        expected = [
            index for index, value in enumerate(flat_experts) if value == expert
        ]
        actual = sorted(actual_by_expert[expert])
        if actual != expected:
            raise AssertionError(
                f"expert={expert} actual={actual} expected={expected} "
                f"post_pad={valid_length} block_m={block_m}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[8, 16, 32, 66, 128, 256, 512, 1024, 4096],
    )
    parser.add_argument("--graph-nodes", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()

    global _EXTENSION
    from torch.utils.cpp_extension import load

    _EXTENSION = load(
        name="yoco_moe_dispatch_ext",
        sources=[
            str(Path(__file__).with_name("yoco_moe_dispatch_bindings.cpp")),
            str(args.source_root / "moe" / "moe_align_sum_kernels.cu"),
        ],
        extra_include_paths=[str(args.source_root)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=True,
    )
    torch.manual_seed(20260831)
    experts = 128
    topk = 8
    for tokens in (8, 32, 66, 127):
        random_order = torch.rand(tokens, experts, device="cuda").argsort(dim=1)
        topk_ids = random_order[:, :topk].to(torch.int32).contiguous()
        for block_m in (16, 32, 64):
            outputs = _allocate_and_dispatch(topk_ids, experts, block_m)
            torch.accelerator.synchronize()
            _validate(topk_ids, outputs, experts, block_m)
    print("correctness=pass (E128, top-k=8, assignments<1024, block_m=16/32/64)")

    # Keep the performance workload independent of the correctness sweep.
    torch.manual_seed(20260831)
    # L3 uses the generic BF16 config today. This is its BLOCK_SIZE_M choice.
    default_block_m = {
        8: 16,
        16: 16,
        32: 16,
        66: 32,
        128: 64,
        256: 64,
        512: 64,
        1024: 128,
        4096: 128,
    }

    print(
        f"device={current_platform.get_device_name()} torch={torch.__version__} "
        f"cuda={torch.version.cuda}"
    )
    print("tokens  block_m  dispatch_us  post_pad  padding_x  deterministic")
    for tokens in args.tokens:
        block_m = default_block_m[tokens]
        random_order = torch.rand(tokens, experts, device="cuda").argsort(dim=1)
        topk_ids = random_order[:, :topk].to(torch.int32).contiguous()
        graph, outputs = _capture(topk_ids, experts, block_m, args.graph_nodes)
        _validate(topk_ids, outputs, experts, block_m)
        baseline = tuple(value.clone() for value in outputs)
        deterministic = True
        for _ in range(5):
            graph.replay()
            torch.accelerator.synchronize()
            deterministic &= all(
                torch.equal(actual, expected)
                for actual, expected in zip(outputs, baseline)
            )
        samples = [
            _time_graph(graph, args.repeats, args.graph_nodes)
            for _ in range(args.rounds)
        ]
        elapsed = statistics.median(samples)
        post_pad = int(outputs[2].item())
        padding_x = post_pad / (tokens * topk)
        print(
            f"{tokens:6d}  {block_m:7d}  {elapsed:11.3f}  {post_pad:8d}  "
            f"{padding_x:9.3f}  {str(deterministic):>13}"
        )


if __name__ == "__main__":
    main()
