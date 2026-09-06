# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B200 benchmark for YOCO L3's TP1 BF16 LM-head GEMM.

The checkpoint uses an untied ``[154880, 3072]`` BF16 output weight. The
benchmark compares the vLLM/training ``F.linear`` layout with alternative
cuBLAS operand layouts and checks whether a fixed input row changes with the
number of batch rows.

Only PyTorch is imported so this can run on the standalone B200 tuning pod.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton


@triton.jit
def _lm_head_kernel(
    hidden_ptr,
    weight_ptr,
    output_ptr,
    num_tokens,
    HIDDEN_SIZE: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUTPUT_FP32: tl.constexpr,
):
    program = tl.program_id(0)
    num_programs_m = tl.cdiv(num_tokens, BLOCK_M)
    program_m = program % num_programs_m
    program_n = program // num_programs_m

    rows = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, HIDDEN_SIZE, BLOCK_K):
        hidden = tl.load(
            hidden_ptr + rows[:, None] * HIDDEN_SIZE + k_start + k_offsets[None, :],
            mask=rows[:, None] < num_tokens,
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + cols[None, :] * HIDDEN_SIZE + k_start + k_offsets[:, None],
            mask=cols[None, :] < VOCAB_SIZE,
            other=0.0,
        )
        accumulator += tl.dot(hidden, weight)

    if OUTPUT_FP32:
        accumulator = accumulator.to(tl.bfloat16).to(tl.float32)
    tl.store(
        output_ptr + rows[:, None] * VOCAB_SIZE + cols[None, :],
        accumulator,
        mask=(rows[:, None] < num_tokens) & (cols[None, :] < VOCAB_SIZE),
    )


def _triton_lm_head(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    output = torch.empty(
        (hidden.shape[0], weight.shape[0]),
        device=hidden.device,
        dtype=output_dtype,
    )
    grid = (
        triton.cdiv(hidden.shape[0], block_m) * triton.cdiv(weight.shape[0], block_n),
    )
    _lm_head_kernel[grid](
        hidden,
        weight,
        output,
        hidden.shape[0],
        HIDDEN_SIZE=hidden.shape[1],
        VOCAB_SIZE=weight.shape[0],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        OUTPUT_FP32=output_dtype == torch.float32,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def _capture(
    fn: Callable[[], torch.Tensor],
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    for _ in range(3):
        output = fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def _time_graph(graph: torch.cuda.CUDAGraph, repeats: int) -> float:
    for _ in range(100):
        graph.replay()
    torch.accelerator.synchronize()
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeats


def _time_eager(fn: Callable[[], torch.Tensor], repeats: int) -> float:
    for _ in range(100):
        fn()
    torch.accelerator.synchronize()
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1410],
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=3072)
    parser.add_argument("--vocab-size", type=int, default=154880)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--fp32-output", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(20260901)
    device = torch.device("cuda")
    max_tokens = max(args.tokens)
    hidden_all = torch.randn(
        max_tokens,
        args.hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    weight = (
        torch.randn(
            args.vocab_size,
            args.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    # This costs another ~907 MiB and is benchmark-only. Production should
    # only cache it if the NN layout shows a stable, material speedup.
    weight_t = weight.t().contiguous()
    weight_col_major = weight_t.t()

    print(
        f"device={torch.cuda.get_device_name()} torch={torch.__version__} "
        f"weight_mib={weight.numel() * weight.element_size() / 2**20:.1f}"
    )
    configs = (
        # The first sweep found M16/N128/K128 near-optimal for one-to-sixteen
        # rows. Sweep its neighboring launch shapes without changing the
        # checkpoint's row-major weight layout.
        (16, 64, 64, 4, 2),
        (16, 64, 64, 4, 3),
        (16, 64, 64, 4, 4),
        (16, 64, 128, 4, 2),
        (16, 64, 128, 4, 3),
        (16, 64, 128, 8, 3),
        (16, 128, 32, 4, 3),
        (16, 128, 64, 4, 2),
        (16, 128, 64, 4, 3),
        (16, 128, 64, 4, 4),
        (16, 128, 64, 8, 2),
        (16, 128, 64, 8, 3),
        (16, 128, 64, 8, 4),
        (16, 128, 128, 4, 2),
        (16, 128, 128, 4, 3),
        (16, 128, 128, 4, 4),
        (16, 128, 128, 8, 2),
        (16, 128, 128, 8, 3),
        (16, 128, 128, 8, 4),
        (16, 256, 64, 8, 2),
        (16, 256, 64, 8, 3),
        (16, 256, 128, 8, 2),
        (16, 256, 128, 8, 3),
    )
    config_names = [
        f"m{block_m}n{block_n}k{block_k}w{warps}s{stages}"
        for block_m, block_n, block_k, warps, stages in configs
    ]
    selected_config = (16, 128, 128, 4, 3)
    selected_index = configs.index(selected_config)
    print(
        "tokens  linear_nt_us  linear_col_us  col_speedup_pct  "
        + "  ".join(config_names)
        + "  "
        "column_exact  column_max_abs  row0_batch_exact"
    )

    first_row: torch.Tensor | None = None
    selected_first_row: torch.Tensor | None = None
    for tokens in args.tokens:
        hidden = hidden_all[:tokens]
        output_dtype = torch.float32 if args.fp32_output else torch.bfloat16

        def linear(hidden=hidden, weight=weight):
            output = F.linear(hidden, weight)
            return output.float() if args.fp32_output else output

        def linear_column(hidden=hidden, weight=weight_col_major):
            output = F.linear(hidden, weight)
            return output.float() if args.fp32_output else output

        fns: tuple[Callable[[], torch.Tensor], ...] = (
            linear,
            linear_column,
            *(
                partial(
                    _triton_lm_head,
                    hidden,
                    weight,
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                    num_warps=warps,
                    num_stages=stages,
                    output_dtype=output_dtype,
                )
                for block_m, block_n, block_k, warps, stages in configs
            ),
        )
        if args.eager:
            outputs = [fn() for fn in fns]
            timers: list[Callable[[], float]] = [
                lambda fn=fn: _time_eager(fn, args.repeats) for fn in fns
            ]
        else:
            captures = [_capture(fn) for fn in fns]
            outputs = [capture[1] for capture in captures]
            timers = [
                lambda graph=graph: _time_graph(graph, args.repeats)
                for graph, _ in captures
            ]
        torch.accelerator.synchronize()

        reference = outputs[0]
        column_exact = torch.equal(outputs[1], reference)
        column_max_abs = (outputs[1].float() - reference.float()).abs().max().item()
        triton_max_abs = max(
            (output.float() - reference.float()).abs().max().item()
            for output in outputs[2:]
        )
        triton_exact = [torch.equal(output, reference) for output in outputs[2:]]
        selected_output = outputs[2 + selected_index]
        selected_abs_diff = (selected_output.float() - reference.float()).abs()
        selected_max_abs = selected_abs_diff.max().item()
        selected_mean_abs = selected_abs_diff.mean().item()
        selected_mismatch_pct = (
            selected_output != reference
        ).float().mean().item() * 100.0
        if first_row is None:
            first_row = reference[0].clone()
        row0_batch_exact = torch.equal(reference[0], first_row)
        if selected_first_row is None:
            selected_first_row = selected_output[0].clone()
        selected_row0_batch_exact = torch.equal(selected_output[0], selected_first_row)

        samples: list[list[float]] = [[] for _ in timers]
        for round_index in range(args.rounds):
            order = list(range(len(timers)))
            if round_index % 2:
                order.reverse()
            order = (
                order[round_index % len(order) :] + order[: round_index % len(order)]
            )
            for timer_index in order:
                samples[timer_index].append(timers[timer_index]())
        times = [statistics.median(values) for values in samples]
        column_speedup = (times[0] / times[1] - 1.0) * 100.0
        print(
            f"{tokens:6d}  "
            + "  ".join(f"{elapsed:12.3f}" for elapsed in times)
            + f"  {column_speedup:15.2f}"
            + f"  {str(column_exact):>12}"
            + f"  {column_max_abs:14.8f}"
            + f"  {str(row0_batch_exact):>16}"
            + f" triton_max_abs={triton_max_abs:.8f}"
            + f" selected_max_abs={selected_max_abs:.8f}"
            + f" selected_mean_abs={selected_mean_abs:.10f}"
            + f" selected_mismatch_pct={selected_mismatch_pct:.4f}"
            + f" selected_row0_batch_exact={selected_row0_batch_exact}"
        )
        non_exact = [
            name for name, exact in zip(config_names, triton_exact) if not exact
        ]
        print(f"         triton_non_exact={non_exact}")


if __name__ == "__main__":
    main()
