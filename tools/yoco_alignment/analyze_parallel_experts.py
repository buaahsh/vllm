# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split native YOCO parallel-expert elapsed time using recorded CUDA kernels.

Offline only. Start-to-start segments partition elapsed time; kernel duration
sums are reported separately, with overlap and gaps explicitly accounted for.
"""

import argparse
import gzip
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from analyze_decode_critical_path import classify, end, layer_phases, require, stats
from analyze_fp8_decode import interval_union

ROUTED = (
    "topk",
    "routed_quant",
    "routed_layout",
    "routed_w13",
    "routed_activation",
    "routed_w2",
    "routed_reduce",
)
SHARED = (
    "quant_shared_w13",
    "shared_w13",
    "shared_scale_zero",
    "shared_activation",
    "shared_w2",
)


def busy_time(events):
    origin = min(event["ts"] for event in events)
    return interval_union(
        [{"ts": event["ts"] - origin, "dur": event["dur"]} for event in events]
    )


def summarize(path):
    with gzip.open(path, "rt") as handle:
        trace = json.load(handle)
    kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
    groups = defaultdict(list)
    for event in kernels:
        groups[event["args"]["correlation"]].append(event)
    graphs = sorted(
        (
            sorted(group, key=lambda e: e["ts"])
            for group in groups.values()
            if any("FlashAttentionForwardSm100" in e["name"] for e in group)
        ),
        key=lambda group: group[0]["ts"],
    )
    require(len(graphs) == 12, "Expected 12 native model Graphs")
    observations, operator_events = [], defaultdict(list)
    for step, graph in enumerate(graphs):
        require(len(graph) == 1490, "Unexpected graph kernel count")
        classify(graph, candidate=False)
        layers, _ = layer_phases(graph)
        for layer in layers:
            begin = graph[0]["ts"] + layer["start_us"]
            selected = [e for e in graph if begin <= e["ts"] < begin + layer["span_us"]]
            rows = defaultdict(list)
            for event in selected:
                if event["category"] in ROUTED + SHARED:
                    rows[event["category"]].append(event)
                    operator_events[event["category"]].append(event)
            require(
                all(len(rows[category]) == 1 for category in ROUTED + SHARED),
                "Each layer must have 7 routed and 5 shared kernels",
            )
            routed = [rows[category][0] for category in ROUTED]
            shared = [rows[category][0] for category in SHARED]
            require(
                routed == sorted(routed, key=lambda e: e["ts"]),
                "Routed stage ordering differs from the audited chain",
            )
            require(
                len({e["args"]["stream"] for e in routed}) == 1,
                "Routed kernels must use one stream",
            )
            fork = min(routed[0]["ts"], shared[0]["ts"])
            routed_end, shared_end = end(routed[-1]), end(shared[-1])
            join = max(routed_end, shared_end)
            parts = {
                "fork_to_topk": routed[0]["ts"] - fork,
                "topk_quant_layout": routed[3]["ts"] - routed[0]["ts"],
                "w13_to_activation": routed[4]["ts"] - routed[3]["ts"],
                "activation_to_w2": routed[5]["ts"] - routed[4]["ts"],
                "w2_to_reduce": routed[6]["ts"] - routed[5]["ts"],
                "reduce": routed_end - routed[-1]["ts"],
                "wait_for_shared": join - routed_end,
            }
            require(
                abs(sum(parts.values()) - (join - fork)) < 0.002,
                "Elapsed segments fail to reconstruct branch span",
            )
            require(
                abs(join - fork - layer["phase_us"]["parallel_experts"]) < 0.002,
                "Branch span differs from previous whole-graph partition",
            )
            routed_span = routed[-1]["ts"] - routed[0]["ts"] + routed[-1]["dur"]
            shared_span = shared[-1]["ts"] - shared[0]["ts"] + shared[-1]["dur"]
            gaps = [b["ts"] - end(a) for a, b in zip(routed, routed[1:])]
            observations.append(
                {
                    "step": step,
                    "logical_layer": layer["index"],
                    "span_us": join - fork,
                    "elapsed_parts_us": parts,
                    "routed_span_us": routed_span,
                    "routed_kernel_us": sum(e["dur"] for e in routed),
                    "routed_no_kernel_us": max(0, routed_span - busy_time(routed)),
                    "routed_overlap_us": max(
                        0, sum(e["dur"] for e in routed) - busy_time(routed)
                    ),
                    "routed_overlapping_pairs": sum(gap < -0.002 for gap in gaps),
                    "shared_span_us": shared_span,
                    "shared_no_kernel_us": max(0, shared_span - busy_time(shared)),
                    "shared_slack_us": routed_end - shared_end,
                    "both_streams_no_kernel_us": join
                    - fork
                    - busy_time(routed + shared),
                }
            )
    parts = observations[0]["elapsed_parts_us"]
    scalar_metrics = (
        "span_us",
        "routed_span_us",
        "routed_kernel_us",
        "routed_no_kernel_us",
        "routed_overlap_us",
        "shared_span_us",
        "shared_no_kernel_us",
        "shared_slack_us",
        "both_streams_no_kernel_us",
    )
    return {
        "trace": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "steps": 12,
        "layer_observations": len(observations),
        "mean_elapsed_parts_us_per_layer": {
            part: statistics.mean(o["elapsed_parts_us"][part] for o in observations)
            for part in parts
        },
        "metrics_us_per_layer": {
            metric: stats([o[metric] for o in observations])
            for metric in scalar_metrics
        },
        "routed_overlapping_pairs": sum(
            o["routed_overlapping_pairs"] for o in observations
        ),
        "operators": {
            category: {
                "duration_us": stats([e["dur"] for e in events]),
                "grids": dict(Counter(str(e["args"]["grid"]) for e in events)),
                "blocks": dict(Counter(str(e["args"]["block"]) for e in events)),
                "registers_per_thread": dict(
                    Counter(str(e["args"].get("registers per thread")) for e in events)
                ),
                "shared_memory_bytes": dict(
                    Counter(str(e["args"].get("shared memory")) for e in events)
                ),
            }
            for category, events in sorted(operator_events.items())
        },
        "observations": observations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.runs_root / "m1-final-a1"
    benchmark = json.loads((run / "benchmark.json").read_text())
    require(
        benchmark["completed"] and benchmark["precision"] == "fp8",
        "Require a completed FP8 benchmark",
    )
    require(benchmark["env"]["VLLM_YOCO_FP8_SMALL_M"] == "0", "Require native run")
    runtime = benchmark["runtime"]
    require(
        runtime["tp"] == runtime["dp"] == 1 and not runtime["enable_expert_parallel"],
        "This analysis requires TP1/DP1/no EP",
    )
    result = {
        "scope": "Native B200 FP8 TP1, 12 steps x 40 layers per batch",
        "runs": {},
    }
    for profile in benchmark["profiles"]:
        batch = profile["batch"]
        path = run / "traces" / Path(profile["files"][0]).name
        result["runs"][f"b{batch}"] = summarize(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    for label, data in result["runs"].items():
        print(label, json.dumps(data["mean_elapsed_parts_us_per_layer"]))
        print("routed overlap pairs:", data["routed_overlapping_pairs"])


if __name__ == "__main__":
    main()
