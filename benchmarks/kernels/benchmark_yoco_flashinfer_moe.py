# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark FlashInfer CUTLASS BF16 MoE launch choices for YOCO L3."""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.fused_moe import ActivationType, cutlass_fused_moe


def _capture(fn: Callable[[], torch.Tensor]) -> torch.cuda.CUDAGraph:
    for _ in range(10):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--tune-max-num-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()

    experts = 128
    topk = 8
    hidden = 1024
    intermediate = 3840
    max_tokens = max(args.tokens)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    generator = torch.Generator(device=device).manual_seed(20260903)
    inputs = torch.randn(
        max_tokens,
        hidden,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    routing = torch.randn(
        max_tokens,
        experts,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    topk_logits, topk_ids = torch.topk(routing, topk, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    # Performance depends on shape/layout, not the numerical weight values.
    w13 = torch.empty(
        experts,
        2 * intermediate,
        hidden,
        dtype=dtype,
        device=device,
    ).fill_(0.001)
    w2 = torch.empty(
        experts,
        hidden,
        intermediate,
        dtype=dtype,
        device=device,
    ).fill_(0.001)
    alpha = torch.ones(experts, dtype=torch.float32, device=device)
    beta = torch.zeros_like(alpha)
    limit = torch.full_like(alpha, 10.0)
    outputs = {
        pdl: torch.empty(max_tokens, hidden, dtype=dtype, device=device)
        for pdl in (False, True)
    }

    def make_run(num_tokens: int, enable_pdl: bool) -> Callable[[], torch.Tensor]:
        def run() -> torch.Tensor:
            cutlass_fused_moe(
                input=inputs[:num_tokens],
                token_selected_experts=topk_ids[:num_tokens],
                token_final_scales=topk_weights[:num_tokens],
                fc1_expert_weights=w13,
                fc2_expert_weights=w2,
                output_dtype=dtype,
                quant_scales=[],
                swiglu_alpha=alpha,
                swiglu_beta=beta,
                swiglu_limit=limit,
                output=outputs[enable_pdl][:num_tokens],
                tune_max_num_tokens=args.tune_max_num_tokens,
                enable_pdl=enable_pdl,
                activation_type=ActivationType.Swiglu,
            )
            return outputs[enable_pdl][:num_tokens]

        return run

    AutoTuner.get().clear_cache()
    print(
        f"device={torch.cuda.get_device_name()} torch={torch.__version__} "
        f"cuda={torch.version.cuda} tune_max={args.tune_max_num_tokens}"
    )
    print("tokens pdl_off_us pdl_on_us pdl_gain_pct exact max_abs")
    with autotune(True):
        for num_tokens in args.tokens:
            make_run(num_tokens, False)()
            make_run(num_tokens, True)()

    for num_tokens in args.tokens:
        functions = (make_run(num_tokens, False), make_run(num_tokens, True))
        graphs = tuple(_capture(fn) for fn in functions)
        samples: tuple[list[float], list[float]] = ([], [])
        for round_index in range(args.rounds):
            order = (0, 1) if round_index % 2 == 0 else (1, 0)
            for graph_index in order:
                samples[graph_index].append(
                    _time_graph(graphs[graph_index], args.repeats)
                )
        times = tuple(statistics.median(sample) for sample in samples)
        references = tuple(fn().clone() for fn in functions)
        difference = (references[0].float() - references[1].float()).abs()
        print(
            f"{num_tokens:6d} {times[0]:10.3f} {times[1]:9.3f} "
            f"{100.0 * (times[0] / times[1] - 1.0):12.3f} "
            f"{str(torch.equal(references[0], references[1])):>5} "
            f"{difference.max().item():.8f}"
        )


if __name__ == "__main__":
    main()
