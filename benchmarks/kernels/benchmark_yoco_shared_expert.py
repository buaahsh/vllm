# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B200 benchmark for YOCO L3's TP1 BF16 shared-expert MLP.

The L3 shape is ``3072 -> (gate=1280, up=1280) -> 1280 -> 3072``.
This standalone benchmark compares the current merged cuBLAS GEMM followed by
an FP32 clamped-SwiGLU kernel with a Triton GEMM whose epilogue performs the
same BF16 projection rounding and activation.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton

HIDDEN_SIZE = 3072
INTERMEDIATE_SIZE = 1280
SWIGLU_LIMIT = 10.0


@torch.compile(fullgraph=True)
def _clamped_swiglu(merged: torch.Tensor) -> torch.Tensor:
    gate, up = merged.chunk(2, dim=-1)
    gate = gate.float().clamp(max=SWIGLU_LIMIT)
    up = up.float().clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    return (F.silu(gate) * up).to(merged.dtype)


@torch.compile(fullgraph=True)
def _clamped_swiglu_separate(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    # Literal llm-train expression (arguments are passed as gate/up here only
    # to keep the benchmark's packed-weight order obvious).
    gate = gate.clamp(max=SWIGLU_LIMIT)
    up = up.clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    return up * F.silu(gate)


@triton.jit
def _fused_gate_up_swiglu_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    num_tokens,
    HIDDEN: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    program = tl.program_id(0)
    num_programs_m = tl.cdiv(num_tokens, BLOCK_M)
    program_m = program % num_programs_m
    program_n = program // num_programs_m

    rows = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, HIDDEN, BLOCK_K):
        hidden = tl.load(
            input_ptr + rows[:, None] * HIDDEN + k_start + k_offsets[None, :],
            mask=rows[:, None] < num_tokens,
            other=0.0,
        )
        gate_weight = tl.load(
            weight_ptr + cols[None, :] * HIDDEN + k_start + k_offsets[:, None],
            mask=cols[None, :] < INTERMEDIATE,
            other=0.0,
        )
        up_weight = tl.load(
            weight_ptr
            + (INTERMEDIATE + cols[None, :]) * HIDDEN
            + k_start
            + k_offsets[:, None],
            mask=cols[None, :] < INTERMEDIATE,
            other=0.0,
        )
        gate_acc += tl.dot(hidden, gate_weight)
        up_acc += tl.dot(hidden, up_weight)

    # F.linear stores BF16 before the standalone training SwiGLU runs.
    gate = gate_acc.to(tl.bfloat16).to(tl.float32)
    up = up_acc.to(tl.bfloat16).to(tl.float32)
    gate = tl.minimum(gate, LIMIT)
    up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    result = gate * tl.sigmoid(gate) * up
    output_offsets = rows[:, None] * INTERMEDIATE + cols[None, :]
    tl.store(
        output_ptr + output_offsets,
        result,
        mask=(rows[:, None] < num_tokens) & (cols[None, :] < INTERMEDIATE),
    )


@dataclass(frozen=True)
class Config:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int

    @property
    def label(self) -> str:
        return (
            f"m{self.block_m}n{self.block_n}k{self.block_k}"
            f"w{self.num_warps}s{self.num_stages}"
        )


CONFIGS = (
    Config(16, 16, 64, 4, 3),
    Config(16, 32, 64, 4, 3),
    Config(16, 64, 64, 4, 3),
    Config(16, 32, 128, 4, 3),
    Config(16, 64, 128, 4, 3),
    Config(32, 32, 64, 4, 3),
    Config(32, 64, 64, 4, 3),
    Config(32, 32, 128, 8, 3),
    Config(64, 32, 64, 4, 3),
    Config(64, 32, 128, 8, 3),
)


def _fused_gate_up_swiglu(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    output = torch.empty(
        (hidden.shape[0], INTERMEDIATE_SIZE),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    grid = (
        triton.cdiv(hidden.shape[0], config.block_m)
        * triton.cdiv(INTERMEDIATE_SIZE, config.block_n),
    )
    _fused_gate_up_swiglu_kernel[grid](
        hidden,
        weight,
        output,
        hidden.shape[0],
        HIDDEN=HIDDEN_SIZE,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        LIMIT=SWIGLU_LIMIT,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return output


def _capture(fn: Callable[[], Any]) -> torch.cuda.CUDAGraph:
    for _ in range(5):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph, repeats: int) -> float:
    for _ in range(20):
        graph.replay()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeats


def _median_time(fn: Callable[[], Any], repeats: int, rounds: int) -> float:
    graph = _capture(fn)
    return statistics.median(_time_graph(graph, repeats) for _ in range(rounds))


def _paired_median_time(
    first: Callable[[], Any],
    second: Callable[[], Any],
    repeats: int,
    rounds: int,
) -> tuple[float, float]:
    graphs = (_capture(first), _capture(second))
    samples: tuple[list[float], list[float]] = ([], [])
    for round_index in range(rounds):
        order = (0, 1) if round_index % 2 == 0 else (1, 0)
        for graph_index in order:
            samples[graph_index].append(_time_graph(graphs[graph_index], repeats))
    return statistics.median(samples[0]), statistics.median(samples[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1410, 2048],
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--whole-mlp", action="store_true")
    parser.add_argument("--first-weight-scale", type=float, default=0.02)
    parser.add_argument("--down-weight-scale", type=float, default=0.03)
    args = parser.parse_args()

    torch.manual_seed(20260901)
    max_tokens = max(args.tokens)
    hidden = torch.randn(
        max_tokens,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    merged_weight = torch.randn(
        2 * INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    down_weight = torch.randn(
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    merged_weight.mul_(args.first_weight_scale)
    down_weight.mul_(args.down_weight_scale)
    gate_weight, up_weight = merged_weight.chunk(2, dim=0)
    training_gate_weight = gate_weight.clone()
    training_up_weight = up_weight.clone()
    training_down_weight = down_weight.clone()
    merged_weight_t = merged_weight.t().contiguous()
    down_weight_t = down_weight.t().contiguous()

    # Compile the two pointwise references before entering CUDA Graph capture.
    _clamped_swiglu(F.linear(hidden[:1], merged_weight))
    _clamped_swiglu_separate(
        F.linear(hidden[:1], gate_weight),
        F.linear(hidden[:1], up_weight),
    )
    torch.accelerator.synchronize()

    print(
        f"device={torch.cuda.get_device_name()} torch={torch.__version__} "
        f"cuda={torch.version.cuda} triton={triton.__version__} "
        f"whole_mlp={args.whole_mlp} first_scale={args.first_weight_scale} "
        f"down_scale={args.down_weight_scale}"
    )
    print(
        "tokens baseline_us align_us first_t_us down_t_us transposed_us "
        "transpose_gain_pct "
        "best_fused_us best_config speedup_pct exact max_abs mean_abs"
    )

    for num_tokens in args.tokens:
        x = hidden[:num_tokens]

        def baseline(x=x) -> torch.Tensor:
            activated = _clamped_swiglu(F.linear(x, merged_weight))
            return F.linear(activated, down_weight) if args.whole_mlp else activated

        def align(x=x) -> torch.Tensor:
            # Keep llm-train's Python argument evaluation order: up, then gate.
            up = F.linear(x, up_weight)
            gate = F.linear(x, gate_weight)
            activated = _clamped_swiglu_separate(gate, up)
            return F.linear(activated, down_weight) if args.whole_mlp else activated

        def training(x=x) -> torch.Tensor:
            up = F.linear(x, training_up_weight)
            gate = F.linear(x, training_gate_weight)
            activated = _clamped_swiglu_separate(gate, up)
            return (
                F.linear(activated, training_down_weight)
                if args.whole_mlp
                else activated
            )

        def transposed(x=x) -> torch.Tensor:
            activated = _clamped_swiglu(torch.mm(x, merged_weight_t))
            return torch.mm(activated, down_weight_t) if args.whole_mlp else activated

        def first_transposed(x=x) -> torch.Tensor:
            activated = _clamped_swiglu(torch.mm(x, merged_weight_t))
            return F.linear(activated, down_weight) if args.whole_mlp else activated

        def down_transposed(x=x) -> torch.Tensor:
            activated = _clamped_swiglu(F.linear(x, merged_weight))
            return torch.mm(activated, down_weight_t) if args.whole_mlp else activated

        fast_output = baseline()
        align_output = align()
        train_output = training()
        baseline_us, down_transposed_us = _paired_median_time(
            baseline,
            down_transposed,
            args.repeats,
            args.rounds,
        )
        align_us = _median_time(align, args.repeats, args.rounds)
        first_transposed_us = _median_time(
            first_transposed,
            args.repeats,
            args.rounds,
        )
        transposed_output = transposed()
        transposed_us = _median_time(transposed, args.repeats, args.rounds)

        fused_results: list[tuple[float, Config, torch.Tensor]] = []
        for config in CONFIGS:

            def fused(config=config, x=x) -> torch.Tensor:
                activated = _fused_gate_up_swiglu(x, merged_weight, config)
                return F.linear(activated, down_weight) if args.whole_mlp else activated

            try:
                actual = fused()
                elapsed = _median_time(fused, args.repeats, args.rounds)
            except Exception as error:  # noqa: BLE001
                print(f"skip {config.label}: {type(error).__name__}: {error}")
                continue
            fused_results.append((elapsed, config, actual))

        best_us, best_config, actual = min(fused_results, key=lambda item: item[0])
        difference = (actual.float() - fast_output.float()).abs()
        fast_align_difference = (fast_output.float() - train_output.float()).abs()
        print(
            f"{num_tokens:6d} {baseline_us:11.3f} {align_us:8.3f} "
            f"{first_transposed_us:10.3f} {down_transposed_us:9.3f} "
            f"{transposed_us:13.3f} "
            f"{100.0 * (baseline_us / transposed_us - 1.0):18.2f} "
            f"{best_us:13.3f} {best_config.label:>17} "
            f"{100.0 * (baseline_us / best_us - 1.0):11.2f} "
            f"{str(torch.equal(actual, fast_output)):>5} "
            f"{difference.max().item():7.5f} {difference.mean().item():8.6f} "
            f"fast_align_exact={torch.equal(fast_output, train_output)} "
            f"fast_align_max={fast_align_difference.max().item():.6f} "
            f"align_train_exact={torch.equal(align_output, train_output)} "
            f"transpose_exact={torch.equal(fast_output, transposed_output)}"
        )


if __name__ == "__main__":
    main()
