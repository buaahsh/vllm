# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare full BF16 GEMM + Top-8 routing against YOCO Softmax Mega Gate.

Rows are tokens reaching one router call, not request concurrency. Use
--speculative-tokens 8 for the nominal M=C*9 verification shape; actual
scheduler batches may contain fewer rows. Tensor files must contain a single
BF16 tensor: normalized/cached weights [128,3072], activations [at least M,3072].
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm.model_executor.layers.yoco_ops import routing as yoco
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import get_deep_gemm_yoco_mega_gate


def capture(fn, stream, repeats):
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(repeats):
                output = fn()
        graph.replay()
    stream.synchronize()
    return graph, output


def latency_us(graph, stream, repeats):
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    with torch.cuda.stream(stream):
        graph.replay()
        start.record()
        graph.replay()
        end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=[16, 24, 32, 40, 48, 56, 64, 128],
    )
    parser.add_argument("--speculative-tokens", type=int, default=8)
    parser.add_argument(
        "--prefill-rows", type=int, nargs="*", default=[1024, 2048, 4096]
    )
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--activations", type=Path)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--calls-per-graph", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not current_platform.is_device_capability_family(100):
        parser.error("native YOCO Mega Gate requires SM100")
    if (
        args.speculative_tokens < 0
        or min(args.concurrencies) < 1
        or any(m < 1 for m in args.prefill_rows)
        or min(args.rounds, args.calls_per_graph) < 1
    ):
        parser.error("counts must be positive and speculative-tokens nonnegative")
    get_deep_gemm_yoco_mega_gate()
    torch.manual_seed(20261025)
    weight = (
        torch.load(args.weights, map_location="cuda", weights_only=True)
        if args.weights
        else torch.randn(128, 3072, device="cuda", dtype=torch.bfloat16) / 3072**0.5
    )
    if weight.shape != (128, 3072) or weight.dtype != torch.bfloat16:
        parser.error("weights must be the BF16 [128,3072] router compute cache")
    weight = weight.contiguous()
    activations = (
        torch.load(args.activations, map_location="cuda", weights_only=True)
        if args.activations
        else None
    )
    cases = [
        (
            "verify" if args.speculative_tokens else "decode",
            c,
            c * (args.speculative_tokens + 1),
        )
        for c in args.concurrencies
    ] + [("prefill", None, m) for m in args.prefill_rows]
    results = []
    for mode, concurrency, rows in cases:
        if activations is not None:
            if (
                activations.ndim != 2
                or activations.shape[0] < rows
                or activations.shape[1] != 3072
                or activations.dtype != torch.bfloat16
            ):
                parser.error(f"activations must be BF16 [at least {rows},3072]")
            x = activations[:rows].contiguous()
        else:
            x = torch.randn(rows, 3072, device="cuda", dtype=torch.bfloat16)

        def baseline(x=x):
            logits = torch.ops.vllm.yoco_router_linear_bf16(x, weight)
            return yoco._yoco_topk_routing_impl(logits, 8, topk_logits=True)

        def candidate(x=x):
            return torch.ops.vllm.yoco_mega_gate(x, weight)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        old_graph, (old_weights, old_ids) = capture(
            baseline, stream, args.calls_per_graph
        )
        new_graph, (new_weights, new_ids) = capture(
            candidate, stream, args.calls_per_graph
        )
        # Copies synchronize the completed capture stream before correctness checks.
        mismatch = (old_ids != new_ids).any(-1).sum().item()
        probability_error = (old_weights - new_weights).abs().max().item()
        equivalent = mismatch == 0 and torch.allclose(
            old_weights, new_weights, rtol=2e-6, atol=1e-7
        )
        measurements = {"baseline": [], "fused": []}
        for index in range(args.rounds):
            pair = [("baseline", old_graph), ("fused", new_graph)]
            if index % 2:
                pair.reverse()
            for name, graph in pair:
                measurements[name].append(
                    latency_us(graph, stream, args.calls_per_graph)
                )
        old_us = statistics.median(measurements["baseline"])
        new_us = statistics.median(measurements["fused"])
        result = {
            "mode": mode,
            "nominal_concurrency": concurrency,
            "router_rows": rows,
            "baseline_us": old_us,
            "fused_us": new_us,
            "speedup": old_us / new_us,
            "mismatched_rows": mismatch,
            "max_probability_error": probability_error,
            "routing_equivalent": equivalent,
            "samples_us": measurements,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "weights": str(args.weights) if args.weights else "synthetic",
        "activations": str(args.activations) if args.activations else "synthetic",
        "speculative_tokens": args.speculative_tokens,
        "timing": "CUDA Graph GPU events, alternating order, no profiler",
        "results": results,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if any(not r["routing_equivalent"] for r in results):
        raise SystemExit("Routing mismatch: candidate is not eligible for activation")


if __name__ == "__main__":
    main()
