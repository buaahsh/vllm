# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B200 benchmark for YOCO L3 cross-query affine RMSClip."""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch
from torch import Tensor

from vllm.triton_utils import tl, triton


@torch.compile
def _training_rms_clip(
    x: Tensor,
    weight: Tensor,
    eps: float,
    limit: float,
) -> Tensor:
    x_float = x.float()
    clip_coef = (
        limit * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + eps)
    ).clamp(max=1.0)
    return (x_float * clip_coef).to(x.dtype) * weight.to(x.dtype)


@triton.jit
def _weighted_rms_clip_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    num_tokens,
    num_heads,
    token_stride,
    head_stride,
    eps: tl.constexpr,
    limit: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    head_rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    cols = tl.arange(0, 128)[None, :]
    row_mask = head_rows < num_tokens * num_heads
    token = head_rows // num_heads
    head = head_rows % num_heads
    input_offsets = token * token_stride + head * head_stride + cols
    values = tl.load(
        x_ptr + input_offsets,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    square_sum = tl.sum(tl.where(row_mask, values * values, 0.0), axis=1)[:, None]
    clip_coef = limit * tl.extra.cuda.libdevice.rsqrt(square_sum / 128.0 + eps)
    clip_coef = tl.minimum(clip_coef, 1.0)
    # Preserve training's explicit BF16 boundary before applying gamma.
    clipped = (values * clip_coef).to(tl.bfloat16).to(tl.float32)
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    tl.store(
        output_ptr + head_rows * 128 + cols,
        clipped * weight,
        mask=row_mask,
    )


def _triton_rms_clip(
    x: Tensor,
    weight: Tensor,
    eps: float,
    limit: float,
    block_rows: int,
    num_warps: int,
) -> Tensor:
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    num_tokens, num_heads, _ = x.shape
    num_head_rows = num_tokens * num_heads
    _weighted_rms_clip_kernel[(triton.cdiv(num_head_rows, block_rows),)](
        x,
        weight,
        output,
        num_tokens,
        num_heads,
        x.stride(0),
        x.stride(1),
        eps,
        limit,
        BLOCK_ROWS=block_rows,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


def _capture(
    fn: Callable[[], Tensor], graph_nodes: int
) -> tuple[torch.cuda.CUDAGraph, Tensor]:
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
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1410, 2048],
    )
    parser.add_argument("--graph-nodes", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(20260901)
    device = torch.device("cuda")
    num_heads = 64
    head_dim = 128
    max_tokens = max(args.tokens)
    projected = 4 * torch.randn(
        max_tokens,
        num_heads * head_dim + 64,
        dtype=torch.bfloat16,
        device=device,
    )
    weight = torch.empty(head_dim, dtype=torch.bfloat16, device=device)
    weight.uniform_(-2.0, 2.0)
    eps = 1e-6
    limit = 3.0
    configs = (
        (1, 1),
        (2, 1),
        (4, 1),
        (4, 2),
        (8, 2),
        (8, 4),
        (16, 4),
        (16, 8),
        (32, 8),
    )

    print(
        f"device={torch.cuda.get_device_name()} torch={torch.__version__} "
        f"cuda={torch.version.cuda} heads={num_heads} head_dim={head_dim} "
        f"token_stride={projected.stride(0)}"
    )
    labels = ["inductor"] + [f"r{rows}w{warps}" for rows, warps in configs]
    print("tokens  " + "  ".join(f"{label:>11}" for label in labels) + "  best")
    first_rows: dict[str, Tensor] = {}

    for tokens in args.tokens:
        x = projected[:tokens, : num_heads * head_dim].unflatten(
            -1,
            (num_heads, head_dim),
        )

        def inductor(x=x) -> Tensor:
            return _training_rms_clip(x, weight, eps, limit)

        functions = [inductor]
        for block_rows, num_warps in configs:

            def candidate(
                x=x,
                block_rows=block_rows,
                num_warps=num_warps,
            ) -> Tensor:
                return _triton_rms_clip(
                    x,
                    weight,
                    eps,
                    limit,
                    block_rows,
                    num_warps,
                )

            functions.append(candidate)

        graphs: list[torch.cuda.CUDAGraph] = []
        outputs: list[Tensor] = []
        for function in functions:
            graph, output = _capture(function, args.graph_nodes)
            graphs.append(graph)
            outputs.append(output)

        reference = outputs[0]
        for label, output in zip(labels[1:], outputs[1:]):
            if not torch.equal(output, reference):
                max_abs = (output - reference).abs().max().item()
                raise AssertionError(f"{tokens=} {label=} is not exact: {max_abs=}")
        for label, output in zip(labels, outputs):
            first_row = output[0].clone()
            if label in first_rows and not torch.equal(first_row, first_rows[label]):
                raise AssertionError(f"{tokens=} {label=} is not batch invariant")
            first_rows[label] = first_row

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
        best_index = min(range(len(times)), key=times.__getitem__)
        print(
            f"{tokens:6d}  "
            + "  ".join(f"{value:11.3f}" for value in times)
            + f"  {labels[best_index]}"
        )


if __name__ == "__main__":
    main()
