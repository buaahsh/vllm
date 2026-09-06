# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare YOCO TP1's complete DeepGEMM W2 path with tuned Triton W2.

The DeepGEMM timing includes its additional 128-row expert dispatch, pack,
prefix-sum layout construction, grouped GEMM, and unpack. This is the cost
that matters before enabling the training-shaped path in ``--fast``.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_yoco_moe_w13 import (
    _launch,
    _load_vllm_fused_moe_kernel,
    _make_assignment,
)
from vllm.model_executor.layers.fused_moe.experts.yoco_deep_gemm import (
    supports_yoco_deep_gemm_w2,
    yoco_deep_gemm_w2,
    yoco_deep_gemm_w2_workspace_rows,
)
from vllm.model_executor.layers.fused_moe.experts.yoco_triton import (
    try_get_yoco_w2_config,
    try_get_yoco_w13_config,
)

EXPERTS = 128
TOPK = 8
INTERMEDIATE_SIZE = 3840
HIDDEN_SIZE = 1024


def _capture(fn: Callable[[], None]) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph, repeats: int, rounds: int) -> float:
    for _ in range(3):
        graph.replay()
    torch.accelerator.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / repeats)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-source", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[2048, 7168, 8192])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    if not supports_yoco_deep_gemm_w2():
        raise RuntimeError("YOCO BF16 DeepGEMM W2 is unavailable")

    torch.manual_seed(20260903)
    kernel = _load_vllm_fused_moe_kernel(args.kernel_source)
    w2 = torch.randn(
        EXPERTS,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )

    print(
        "tokens triton_us deep_gemm_complete_us deep_over_triton "
        "packed_rows workspace_mib"
    )
    for num_tokens in args.tokens:
        topk_ids = (
            torch.rand(num_tokens, EXPERTS, device="cuda")
            .argsort(dim=1)[:, :TOPK]
            .to(torch.int32)
            .contiguous()
        )
        routed_weights = torch.rand(
            num_tokens, TOPK, device="cuda", dtype=torch.float32
        )
        routed_weights /= routed_weights.sum(dim=-1, keepdim=True)
        activation = torch.randn(
            num_tokens * TOPK,
            INTERMEDIATE_SIZE,
            device="cuda",
            dtype=torch.bfloat16,
        )
        triton_output = torch.empty(
            num_tokens,
            TOPK,
            HIDDEN_SIZE,
            device="cuda",
            dtype=torch.bfloat16,
        )
        deep_output = torch.empty_like(triton_output).view(-1, HIDDEN_SIZE)

        w13_config = try_get_yoco_w13_config(
            num_tokens, EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE
        )
        if w13_config is None:
            raise RuntimeError(f"No YOCO W13 config for M={num_tokens}")
        w2_config = try_get_yoco_w2_config(
            num_tokens,
            EXPERTS,
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            w13_config,
        )
        assignment = _make_assignment(topk_ids, w13_config["BLOCK_SIZE_M"], EXPERTS)

        packed_rows = yoco_deep_gemm_w2_workspace_rows(num_tokens, TOPK, EXPERTS)
        packed_input = torch.empty(
            packed_rows,
            INTERMEDIATE_SIZE,
            device="cuda",
            dtype=torch.bfloat16,
        )
        packed_output = torch.empty(
            packed_rows,
            HIDDEN_SIZE,
            device="cuda",
            dtype=torch.bfloat16,
        )

        run_triton = partial(
            _launch,
            kernel,
            activation,
            w2,
            triton_output,
            routed_weights,
            assignment,
            1,
            w2_config,
        )
        run_deep_gemm = partial(
            yoco_deep_gemm_w2,
            deep_output,
            activation,
            w2,
            topk_ids,
            packed_input,
            packed_output,
        )

        triton_graph = _capture(run_triton)
        deep_graph = _capture(run_deep_gemm)
        triton_us = _time_graph(triton_graph, args.repeats, args.rounds)
        deep_us = _time_graph(deep_graph, args.repeats, args.rounds)
        workspace_mib = (
            (packed_input.numel() + packed_output.numel())
            * packed_input.element_size()
            / 2**20
        )
        print(
            f"{num_tokens:6d} {triton_us:9.3f} {deep_us:21.3f} "
            f"{deep_us / triton_us:16.3f} {packed_rows:11d} "
            f"{workspace_mib:13.1f}"
        )
        triton_graph.reset()
        deep_graph.reset()
        del (
            topk_ids,
            routed_weights,
            activation,
            triton_output,
            deep_output,
            assignment,
            packed_input,
            packed_output,
        )
        torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
