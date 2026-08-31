# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B200 benchmark for YOCO L3's BF16 fc1 latent projection and RMSNorm."""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


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


@torch.compile
def _compiled_proj_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    projected = F.linear(x, weight)
    return F.rms_norm(
        projected,
        (projected.shape[-1],),
        weight=norm_weight,
        eps=eps,
    )


@torch.compile
def _compiled_rmsnorm(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return F.rms_norm(
        x.to(torch.bfloat16),
        (x.shape[-1],),
        weight=norm_weight.to(torch.bfloat16),
        eps=eps,
    )


@triton.jit
def _rmsnorm_1024_kernel(
    x,
    weight,
    output,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(x + row * BLOCK_SIZE + offsets).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / BLOCK_SIZE
    scale = tl.rsqrt(variance + eps)
    weights = tl.load(weight + offsets).to(tl.float32)
    tl.store(output + row * BLOCK_SIZE + offsets, values * scale * weights)


def _triton_rmsnorm_1024(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    num_warps: int,
) -> torch.Tensor:
    output = torch.empty_like(x)
    _rmsnorm_1024_kernel[(x.shape[0],)](
        x,
        norm_weight,
        output,
        eps,
        BLOCK_SIZE=1024,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 66, 128, 256, 1024, 4096],
    )
    parser.add_argument("--graph-nodes", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(20260831)
    device = torch.device("cuda")
    max_tokens = max(args.tokens)
    target = torch.randn(1, 3072, dtype=torch.bfloat16, device=device)
    inputs = torch.randn(max_tokens, 3072, dtype=torch.bfloat16, device=device)
    weight = torch.randn(1024, 3072, dtype=torch.bfloat16, device=device)
    norm_weight = torch.randn(1024, dtype=torch.bfloat16, device=device)
    eps = 1e-6

    print(
        f"device={current_platform.get_device_name()} torch={torch.__version__} "
        f"cuda={torch.version.cuda}"
    )
    print(
        "tokens  linear_us  eager_norm_us  compiled_norm_us  triton4_norm_us  "
        "triton8_norm_us  compiled_pair_us  triton4_max_abs  "
        "linear_row_exact  compiled_row_exact"
    )

    reference_linear_row: torch.Tensor | None = None
    reference_compiled_row: torch.Tensor | None = None
    for tokens in args.tokens:
        x = inputs[:tokens].clone()
        x[0].copy_(target[0])
        projected = F.linear(x, weight)

        def linear(x=x, weight=weight) -> torch.Tensor:
            return F.linear(x, weight)

        def rmsnorm(projected=projected, norm_weight=norm_weight) -> torch.Tensor:
            return F.rms_norm(
                projected,
                (projected.shape[-1],),
                weight=norm_weight,
                eps=eps,
            )

        def eager_pair(x=x, weight=weight, norm_weight=norm_weight) -> torch.Tensor:
            result = F.linear(x, weight)
            return F.rms_norm(
                result,
                (result.shape[-1],),
                weight=norm_weight,
                eps=eps,
            )

        def compiled_pair(x=x, weight=weight, norm_weight=norm_weight) -> torch.Tensor:
            return _compiled_proj_norm(x, weight, norm_weight, eps)

        def compiled_norm(projected=projected, norm_weight=norm_weight) -> torch.Tensor:
            return _compiled_rmsnorm(projected, norm_weight, eps)

        def triton4_norm(projected=projected, norm_weight=norm_weight) -> torch.Tensor:
            return _triton_rmsnorm_1024(projected, norm_weight, eps, 4)

        def triton8_norm(projected=projected, norm_weight=norm_weight) -> torch.Tensor:
            return _triton_rmsnorm_1024(projected, norm_weight, eps, 8)

        graphs: list[torch.cuda.CUDAGraph] = []
        outputs: list[torch.Tensor] = []
        for fn in (
            linear,
            rmsnorm,
            compiled_norm,
            triton4_norm,
            triton8_norm,
            compiled_pair,
        ):
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

        linear_row = outputs[0][0].clone()
        compiled_row = outputs[5][0].clone()
        if reference_linear_row is None:
            reference_linear_row = linear_row
            reference_compiled_row = compiled_row
        linear_row_exact = torch.equal(linear_row, reference_linear_row)
        compiled_row_exact = torch.equal(compiled_row, reference_compiled_row)
        triton_error = (outputs[2] - outputs[3]).abs().max().item()
        print(
            f"{tokens:6d}  "
            + "  ".join(f"{value:9.3f}" for value in times)
            + f"  {triton_error:15.8f}  {str(linear_row_exact):>16}"
            + f"  {str(compiled_row_exact):>18}"
        )


if __name__ == "__main__":
    main()
