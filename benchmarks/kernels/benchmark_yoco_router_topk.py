# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark YOCO's current routing path against vLLM fused topk_softmax.

This script is intentionally self-contained so it can run in a minimal B200
development Pod.  Build ``csrc/moe/topk_softmax_kernels.cu`` together with
``yoco_topk_softmax_bindings.cpp`` and pass the resulting shared library via
``--library``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def routing_softmax_kernel(logits_ptr, scores_ptr, num_rows):
    row = tl.program_id(0)
    cols = tl.arange(0, 128)
    row_mask = row < num_rows
    logits = tl.load(
        logits_ptr + row * 128 + cols,
        mask=row_mask,
        other=0.0,
    )
    masked_logits = tl.where(row_mask, logits, float("-inf"))
    row_max = tl.max(masked_logits, axis=0)
    numerator = tl.extra.cuda.libdevice.exp(logits - row_max)
    denominator = tl.sum(tl.where(row_mask, numerator, 0.0), axis=0)
    tl.store(
        scores_ptr + row * 128 + cols,
        numerator / denominator,
        mask=row_mask,
    )


@triton.jit
def routing_renorm_kernel(topk_weights_ptr, num_rows, BLOCK_ROWS: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    ranks = tl.arange(0, 8)[None, :]
    row_mask = rows < num_rows
    weights = tl.load(
        topk_weights_ptr + rows * 8 + ranks,
        mask=row_mask,
        other=0.0,
    )
    denominator = tl.sum(tl.where(row_mask, weights, 0.0), axis=1)[:, None]
    tl.store(
        topk_weights_ptr + rows * 8 + ranks,
        weights / denominator,
        mask=row_mask,
    )


@triton.jit
def fused_routing_triton_kernel(
    logits_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    num_rows,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    cols = tl.arange(0, 128)[None, :]
    row_mask = rows < num_rows
    logits = tl.load(
        logits_ptr + rows * 128 + cols,
        mask=row_mask,
        other=float("-inf"),
    )

    # Preserve the existing fast path's full 128-way softmax before selecting
    # experts.  Its common denominator later cancels during Top-8 renorm, but
    # retaining it gives the closest comparison with the current path.
    row_max = tl.max(logits, axis=1)[:, None]
    numerator = tl.extra.cuda.libdevice.exp(logits - row_max)
    scores = numerator / tl.sum(numerator, axis=1)[:, None]

    ranks = tl.arange(0, 8)[None, :]
    selected_scores = tl.full((BLOCK_ROWS, 8), 0.0, tl.float32)
    selected_ids = tl.full((BLOCK_ROWS, 8), 0, tl.int32)
    for rank in tl.static_range(8):
        value, index = tl.max(
            scores,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        selected_scores = tl.where(
            ranks == rank,
            value[:, None],
            selected_scores,
        )
        selected_ids = tl.where(
            ranks == rank,
            index[:, None],
            selected_ids,
        )
        scores = tl.where(cols == index[:, None], float("-inf"), scores)

    selected_scores /= tl.sum(selected_scores, axis=1)[:, None]
    output_offsets = rows * 8 + ranks
    tl.store(topk_weights_ptr + output_offsets, selected_scores, mask=row_mask)
    tl.store(topk_ids_ptr + output_offsets, selected_ids, mask=row_mask)


@triton.jit
def fused_logits_topk_kernel(
    logits_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    num_rows,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    cols = tl.arange(0, 128)[None, :]
    row_mask = rows < num_rows
    values = tl.load(
        logits_ptr + rows * 128 + cols,
        mask=row_mask,
        other=float("-inf"),
    )
    ranks = tl.arange(0, 8)[None, :]
    selected_logits = tl.full((BLOCK_ROWS, 8), 0.0, tl.float32)
    selected_ids = tl.full((BLOCK_ROWS, 8), 0, tl.int32)
    for rank in tl.static_range(8):
        value, index = tl.max(
            values,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        selected_logits = tl.where(ranks == rank, value[:, None], selected_logits)
        selected_ids = tl.where(ranks == rank, index[:, None], selected_ids)
        values = tl.where(cols == index[:, None], float("-inf"), values)
    selected_max = tl.max(selected_logits, axis=1)[:, None]
    numerators = tl.extra.cuda.libdevice.exp(selected_logits - selected_max)
    weights = numerators / tl.sum(numerators, axis=1)[:, None]
    output_offsets = rows * 8 + ranks
    tl.store(topk_weights_ptr + output_offsets, weights, mask=row_mask)
    tl.store(topk_ids_ptr + output_offsets, selected_ids, mask=row_mask)


def renorm_block(num_rows: int) -> int:
    return 32 if num_rows >= 96 else 1


def current_routing(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows = logits.shape[0]
    scores = torch.empty_like(logits)
    routing_softmax_kernel[(num_rows,)](
        logits,
        scores,
        num_rows,
        num_warps=2,
        num_stages=1,
    )
    weights, ids = torch.topk(scores, k=8, dim=-1)
    block_rows = renorm_block(num_rows)
    routing_renorm_kernel[(triton.cdiv(num_rows, block_rows),)](
        weights,
        num_rows,
        BLOCK_ROWS=block_rows,
        num_warps=2,
        num_stages=1,
    )
    return weights, ids


def fused_routing(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (logits.shape[0], 8)
    weights = torch.empty(shape, dtype=torch.float32, device=logits.device)
    ids = torch.empty(shape, dtype=torch.int32, device=logits.device)
    source_rows = torch.empty_like(ids)
    torch.ops.yoco_topk_bench.topk_softmax(
        weights,
        ids,
        source_rows,
        logits,
        True,
        None,
    )
    return weights, ids, source_rows


def triton_fused_routing(
    logits: torch.Tensor,
    block_rows: int = 4,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (logits.shape[0], 8)
    weights = torch.empty(shape, dtype=torch.float32, device=logits.device)
    ids = torch.empty(shape, dtype=torch.int64, device=logits.device)
    fused_routing_triton_kernel[(triton.cdiv(logits.shape[0], block_rows),)](
        logits,
        weights,
        ids,
        logits.shape[0],
        BLOCK_ROWS=block_rows,
        num_warps=num_warps,
        num_stages=1,
    )
    return weights, ids


def triton_logits_routing(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (logits.shape[0], 8)
    weights = torch.empty(shape, dtype=torch.float32, device=logits.device)
    ids = torch.empty(shape, dtype=torch.int64, device=logits.device)
    fused_logits_topk_kernel[(triton.cdiv(logits.shape[0], 4),)](
        logits,
        weights,
        ids,
        logits.shape[0],
        BLOCK_ROWS=4,
        num_warps=4,
        num_stages=1,
    )
    return weights, ids


def torch_reference(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    weights, ids = torch.topk(scores, k=8, dim=-1, sorted=True)
    return weights / weights.sum(dim=-1, keepdim=True), ids


@dataclass
class GraphRun:
    graph: torch.cuda.CUDAGraph
    outputs: tuple[torch.Tensor, ...]


def capture(fn, logits: torch.Tensor) -> GraphRun:
    fn(logits)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn(logits)
    graph.replay()
    torch.accelerator.synchronize()
    return GraphRun(graph, outputs)


def graph_latency_us(run: GraphRun, iterations: int) -> float:
    for _ in range(20):
        run.graph.replay()
    torch.accelerator.synchronize()
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        run.graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def compare(name: str, logits: torch.Tensor) -> None:
    ref_weights, ref_ids = torch_reference(logits)
    cur_weights, cur_ids = current_routing(logits)
    fused_weights, fused_ids, _ = fused_routing(logits)
    triton_weights, triton_ids = triton_fused_routing(logits)
    logits_weights, logits_ids = triton_logits_routing(logits)
    torch.accelerator.synchronize()

    def report(
        candidate_weights: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> str:
        ordered = (candidate_ids.to(torch.int64) == ref_ids).all(dim=1)
        candidate_sets = torch.sort(candidate_ids.to(torch.int64), dim=1).values
        reference_sets = torch.sort(ref_ids, dim=1).values
        same_sets = (candidate_sets == reference_sets).all(dim=1)
        error = (candidate_weights - ref_weights).abs()
        return (
            f"ordered={ordered.float().mean().item():.6f} "
            f"set={same_sets.float().mean().item():.6f} "
            f"max_abs={error.max().item():.9g} "
            f"mean_abs={error.mean().item():.9g}"
        )

    print(f"accuracy {name:>12} current {report(cur_weights, cur_ids)}")
    print(f"accuracy {name:>12} fused   {report(fused_weights, fused_ids)}")
    print(f"accuracy {name:>12} triton  {report(triton_weights, triton_ids)}")
    print(f"accuracy {name:>12} logits  {report(logits_weights, logits_ids)}")
    same_ids = (triton_ids == cur_ids).all(dim=1).float().mean().item()
    current_error = (triton_weights - cur_weights).abs()
    print(
        f"direct   {name:>12} triton/current "
        f"ordered={same_ids:.6f} max_abs={current_error.max().item():.9g} "
        f"mean_abs={current_error.mean().item():.9g}"
    )
    if logits.shape[0] <= 4:
        print(f"  ref ids[0]   {ref_ids[0].tolist()}")
        print(f"  fused ids[0] {fused_ids[0].tolist()}")
        print(f"  triton ids[0] {triton_ids[0].tolist()}")


def check_batch_independence() -> None:
    torch.manual_seed(20260831)
    target = torch.randn(1, 128, dtype=torch.float32, device="cuda")
    expected_weights, expected_ids = triton_fused_routing(target)
    for num_rows, position in ((3, 1), (66, 37), (110, 109), (1024, 511)):
        logits = torch.randn(num_rows, 128, dtype=torch.float32, device="cuda")
        logits[position].copy_(target[0])
        weights, ids = triton_fused_routing(logits)
        if not torch.equal(ids[position], expected_ids[0]):
            raise AssertionError(f"batch-dependent IDs at rows={num_rows}")
        if not torch.equal(weights[position], expected_weights[0]):
            error = (weights[position] - expected_weights[0]).abs().max().item()
            raise AssertionError(
                f"batch-dependent weights at rows={num_rows}: max_abs={error}"
            )
    print("batch independence: bitwise exact for rows=1/3/66/110/1024")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 3, 16, 66, 110, 256, 1024, 4096, 16384],
    )
    parser.add_argument("--iterations", type=int, default=2000)
    args = parser.parse_args()

    torch.ops.load_library(args.library)
    torch.manual_seed(0)
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"device={current_platform.get_device_name()}"
    )

    # Random logits cover several useful scales; exact and near ties expose
    # ordering changes hidden by ordinary continuous random inputs.
    compare("normal", torch.randn(4096, 128, device="cuda"))
    compare("small", torch.randn(4096, 128, device="cuda") * 0.05)
    compare("large", torch.randn(4096, 128, device="cuda") * 8.0)
    compare("all_tie", torch.zeros(4, 128, device="cuda"))
    near_tie = torch.zeros(4, 128, device="cuda")
    near_tie[:, :16] = torch.arange(16, device="cuda", dtype=torch.float32) * 1e-7
    compare("near_tie", near_tie)
    check_batch_independence()

    print("\nCUDA Graph latency (us)")
    print(
        f"{'tokens':>8} {'current':>10} {'cuda':>10} "
        f"{'b4w1':>10} {'b4w2':>10} {'b4w4':>10} {'b4w8':>10} {'b8w8':>10} "
        f"{'logits':>10}"
    )
    for num_tokens in args.tokens:
        logits = torch.randn(num_tokens, 128, dtype=torch.float32, device="cuda")
        current_run = capture(current_routing, logits)
        fused_run = capture(fused_routing, logits)
        triton_configs = ((4, 1), (4, 2), (4, 4), (4, 8), (8, 8))
        triton_runs = {
            config: capture(
                lambda x, config=config: triton_fused_routing(x, config[0], config[1]),
                logits,
            )
            for config in triton_configs
        }
        logits_run = capture(triton_logits_routing, logits)
        iterations = (
            args.iterations if num_tokens <= 4096 else max(500, args.iterations // 4)
        )
        current_us = graph_latency_us(current_run, iterations)
        fused_us = graph_latency_us(fused_run, iterations)
        triton_us = {
            config: graph_latency_us(run, iterations)
            for config, run in triton_runs.items()
        }
        logits_us = graph_latency_us(logits_run, iterations)
        print(
            f"{num_tokens:8d} {current_us:10.3f} {fused_us:10.3f} "
            + " ".join(f"{triton_us[config]:10.3f}" for config in triton_configs)
            + f" {logits_us:10.3f}"
        )


if __name__ == "__main__":
    main()
