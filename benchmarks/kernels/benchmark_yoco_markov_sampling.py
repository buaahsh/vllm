# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ABBA device timing of K sequential Markov GEMMs and full-vocabulary sampling.

Uses synthetic weights and precomputed base logits; excludes the backbone,
LM head, target, scheduling and prefill. Extra audit graphs are benchmark-only.
"""

import argparse
import json
import statistics
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

from vllm.model_executor.models.yoco_shared_kv import SharedKVConfig, SharedKVDraft
from vllm.v1.worker.gpu.spec_decode.dspark.shared_kv import SharedKVSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


@torch.inference_mode()
def run_case(batch, steps, vocab, rounds, iterations):
    torch.manual_seed(42)
    config = SharedKVConfig(hidden=16, layers=0, vocab=vocab, block=steps, rank=256)
    model = SharedKVDraft(config).cuda().eval().requires_grad_(False)
    model.prepare_for_inference()
    base = torch.randn(batch * steps, vocab, device="cuda")
    model.compute_draft_logits = lambda hidden: base
    hidden = torch.randn(batch * steps, config.hidden, device="cuda")
    spec = SharedKVSpeculator.__new__(SharedKVSpeculator)
    spec.model = model
    spec.num_speculative_steps = steps
    spec._draft_topk = spec._d2t_scatter_index = spec.acceptance_estimator = None
    spec.draft_watermarker = None
    spec.use_confidence_head = spec.use_fp64_gumbel = False
    spec.sample_indices = torch.arange(batch * steps, device="cuda")
    spec.sample_idx_mapping = torch.arange(
        batch, device="cuda", dtype=torch.int32
    ).repeat_interleave(steps)
    spec.sample_pos = torch.arange(steps, device="cuda").repeat(batch) + 100
    spec.input_buffers = SimpleNamespace(
        input_ids=torch.arange(batch * steps, device="cuda") % vocab
    )
    spec._anchor_idx = torch.arange(batch, device="cuda") * steps
    spec._step_cols = torch.arange(steps, device="cuda", dtype=torch.int32)
    spec.draft_logits = torch.zeros(batch, steps, vocab, device="cuda")
    spec.draft_tokens = torch.zeros(batch, steps, device="cuda", dtype=torch.int64)
    spec.temperature = torch.ones(batch, device="cuda")
    spec.seeds = torch.arange(batch, device="cuda", dtype=torch.int64) + 17
    forwards = {
        "baseline": MethodType(DSparkSpeculator._sample_markov_logits, spec),
        "fused": spec._sample_markov_logits,
    }

    def run():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            spec._sample_sequential(batch, hidden)

    graphs = {}
    snapshots = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for name, forward in forwards.items():
        spec._sample_markov_logits = forward
        with torch.cuda.stream(stream):
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
        graphs[name] = graph
        graph.replay()
        torch.accelerator.synchronize()
        snapshots[name] = (spec.draft_tokens.clone(), spec.draft_logits.clone())
    for actual, expected in zip(snapshots["fused"], snapshots["baseline"]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    samples = {name: [] for name in graphs}
    for trial in range(rounds):
        order = (
            ["baseline", "fused", "fused", "baseline"]
            if trial % 2 == 0
            else ["fused", "baseline", "baseline", "fused"]
        )
        for name in order:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(iterations):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / iterations)

    profiles = {}
    for name, forward in forwards.items():
        spec._sample_markov_logits = forward
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            run()
            torch.accelerator.synchronize()
        profiles[name] = {
            event.key: event.count
            for event in profile.key_averages()
            if event.key in {"aten::add", "aten::_to_copy"}
        }
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "batch": batch,
        "steps": steps,
        "vocab": vocab,
        "q_and_tokens_exact": True,
        "latency_ms": medians,
        "latency_reduction_pct": (1 - medians["fused"] / medians["baseline"]) * 100,
        "samples_ms": samples,
        "eager_operator_counts": profiles,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=154880)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if min(*args.batches, args.steps, args.vocab, args.rounds, args.iterations) < 1:
        parser.error("All shape and timing arguments must be positive")
    results = {
        "scope": __doc__,
        "gpu": torch.cuda.get_device_name(),
        "cases": [
            run_case(batch, args.steps, args.vocab, args.rounds, args.iterations)
            for batch in args.batches
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
