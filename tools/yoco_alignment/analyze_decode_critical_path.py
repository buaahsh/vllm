# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit elapsed YOCO layer phases and shared/routed overlap in saved traces.

This is a version/shape-specific offline analysis, not a general dependency-DAG
profiler. Phase boundaries partition wall time; kernel sums include overlap.
"""

import argparse
import gzip
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from analyze_fp8_decode import dense_category, interval_union

PHASES = (
    "attention_and_norm",
    "latent_in_and_router",
    "parallel_experts",
    "join_to_latent_out",
    "latent_out_and_merge",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def end(event):
    return event["ts"] + event["dur"]


def stats(values):
    values = sorted(values)
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": values[0],
        "p95": values[min(len(values) - 1, int(0.95 * len(values)))],
        "max": values[-1],
    }


def classify(graph, candidate):
    streams = defaultdict(list)
    for event in graph:
        streams[event["args"]["stream"]].append(event)
        event["category"] = dense_category(event["name"])
    routed = [e for e in graph if e["name"] == "fused_moe_kernel"]
    require(len(routed) == 80, "Expected 40 routed W13/W2 pairs")
    for i, event in enumerate(routed):
        event["category"] = "routed_w13" if i % 2 == 0 else "routed_w2"
    for stream in streams.values():
        for i, event in enumerate(stream):
            if event["name"] != "_small_fp8_direct_kernel":
                continue
            grid = event["args"]["grid"]
            require(grid[1:] == [1, 1], "Unexpected direct GEMV grid")
            if grid[0] == 1024:
                category = "latent_in"
            elif grid[0] == 2560:
                category = "shared_w13"
            else:
                require(grid[0] == 3072, "Unexpected direct GEMV N")
                category = (
                    "shared_w2"
                    if stream[i - 1]["name"] == "_silu_mul_quant_fp8_packed_kernel"
                    else "latent_out"
                )
            event["category"] = category
    for stream in streams.values():
        for i, event in enumerate(stream):
            if event["category"]:
                continue
            name = event["name"]
            previous = stream[i - 1]["name"] if i else ""
            following = stream[i + 1] if i + 1 < len(stream) else {}
            next_category = following.get("category")
            if "packed_register_kernel" in name and next_category:
                category = "quant_" + next_category
            elif name == "_silu_mul_quant_fp8_packed_kernel":
                require(next_category in ("routed_w2", "shared_w2"), name)
                category = next_category.replace("w2", "activation")
            elif "FlashAttentionForwardSm100" in name:
                category = "attention_main"
            elif "FlashAttentionForwardCombine" in name:
                category = "attention_combine"
            elif "FlashPrepareScheduler" in name:
                category = "attention_scheduler"
            elif "direct_copy_kernel_cuda" in name and "lambda(float)" in name:
                category = "attention_descale"
            elif "nvjet_" in name or "splitKreduce_kernel" in name:
                gemm = previous if "splitKreduce" in name else name
                if "nvjet_sm100_tst_32x64_64x16_4x1_" in gemm:
                    category = "router"
                else:
                    require(
                        "nvjet_sm100_tst_32x64_64x16_2x1_" in gemm
                        or "cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6" in gemm,
                        f"Unrecognized BF16 GEMM: {gemm}",
                    )
                    category = "lambda"
            elif "cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6" in name:
                category = "lambda"
            elif "per_token_group_quant_8bit_kernel<" in name:
                category = "routed_quant"
            elif "moe_sum_vec_kernel<c10::BFloat16" in name:
                category = "routed_reduce"
            elif "moe_align_block_size_e128_small_kernel" in name:
                category = "routed_layout"
            elif "FillFunctor<int>" in name:
                category = (
                    "shared_scale_zero"
                    if following.get("name") == "_silu_mul_quant_fp8_packed_kernel"
                    else "routed_layout"
                )
            else:
                category = {
                    "moe_forward_shared_0": "latent_in_norm",
                    "per_token_group_fp8_quant_packed_1": "latent_out_norm",
                    "_yoco_fused_topk_routing_kernel": "topk",
                    "_yoco_fused_shared_gate_moe_output_kernel": "merge_gate",
                }.get(name.removeprefix("triton_per_fused__fused_rms_norm_"), name)
            event["category"] = category
    expected = {
        **dict.fromkeys(
            (
                "latent_in",
                "latent_out",
                "shared_w13",
                "shared_w2",
                "latent_in_norm",
                "latent_out_norm",
                "routed_w13",
                "routed_w2",
                "shared_activation",
                "routed_activation",
                "routed_reduce",
                "topk",
                "merge_gate",
                "attention_main",
                "attention_combine",
                "attention_scheduler",
                "quant_latent_in",
                "quant_latent_out",
                "quant_shared_w13",
                "shared_scale_zero",
                "routed_layout",
            ),
            40,
        ),
        "router": 80,
        "lambda": 80,
        "attention_descale": 120,
        "self_qkv": 30,
        "cross_q": 10,
        "shared_kv_proj": 1,
    }
    counts = Counter(e["category"] for e in graph)
    for category, count in expected.items():
        require(counts[category] == count, f"{category}: {counts[category]} != {count}")
    direct = [e for e in graph if e["name"] == "_small_fp8_direct_kernel"]
    require(len(direct) == (160 if candidate else 0), "Incorrect direct dispatch")
    if direct:
        require(
            Counter(e["category"] for e in direct)
            == dict.fromkeys(
                ("latent_in", "latent_out", "shared_w13", "shared_w2"), 40
            ),
            "Unproven direct GEMV categories",
        )


def layer_phases(graph):
    gates = [e for e in graph if e["category"] == "merge_gate"]
    begin = graph[0]["ts"]
    layers = []
    for i, gate in enumerate(gates):
        selected = [e for e in graph if begin <= e["ts"] < end(gate)]
        rows = defaultdict(list)
        for event in selected:
            rows[event["category"]].append(event)

        def one(category, rows=rows, index=i):
            require(len(rows[category]) == 1, f"Layer {index}: ambiguous {category}")
            return rows[category][0]

        shared_start = one("quant_shared_w13")["ts"]
        one("self_qkv" if i < 30 else "cross_q")
        one("attention_main")
        routed_start = one("topk")["ts"]
        shared_end = end(one("shared_w2"))
        routed_end = end(one("routed_reduce"))
        fork, join = min(shared_start, routed_start), max(shared_end, routed_end)
        boundaries = [
            begin,
            one("quant_latent_in")["ts"],
            fork,
            join,
            one("latent_out_norm")["ts"],
            end(gate),
        ]
        phases = dict(zip(PHASES, (b - a for a, b in zip(boundaries, boundaries[1:]))))
        require(min(phases.values()) >= 0, f"Layer {i}: reversed phase boundaries")
        require(len(rows["router"]) == 2, f"Layer {i}: expected router/reduce")
        # These are observed timestamp landmarks, not CUDA dependency edges.
        require(
            one("latent_in")["ts"] < rows["router"][0]["ts"] < fork,
            "Latent/router/fork scheduling differs from the audited graph",
        )
        require(
            one("shared_w13")["args"]["stream"]
            == one("shared_w2")["args"]["stream"]
            != one("routed_w13")["args"]["stream"]
            == one("routed_w2")["args"]["stream"],
            "Shared and routed stream roles not proven",
        )
        layers.append(
            {
                "index": i,
                "kind": "self" if i < 30 else "cross",
                "start_us": begin - graph[0]["ts"],
                "span_us": end(gate) - begin,
                "phase_us": phases,
                "shared_span_us": shared_end - shared_start,
                "routed_span_us": routed_end - routed_start,
                "shared_slack_us": routed_end - shared_end,
                "shared_late_us": max(0, shared_end - routed_end),
                "branch_us": [
                    shared_start - begin,
                    shared_end - begin,
                    routed_start - begin,
                    routed_end - begin,
                ],
                "kernel_count": len(selected),
            }
        )
        begin = end(gate)
    tail = [e for e in graph if e["ts"] >= begin]
    require(
        sum(layer["kernel_count"] for layer in layers) + len(tail) == len(graph),
        "Layer partition omitted or duplicated kernels",
    )
    span = max(map(end, graph)) - graph[0]["ts"]
    phase_us = {p: sum(layer["phase_us"][p] for layer in layers) for p in PHASES}
    phase_us["final_norm"] = span - sum(phase_us.values())
    require(phase_us["final_norm"] >= 0, "Invalid final norm tail")
    return layers, phase_us


def analyze(path, batch, candidate, annotated_dir):
    with gzip.open(path, "rt") as handle:
        trace = json.load(handle)
    events = trace["traceEvents"]
    annotations = [
        e
        for e in events
        if e.get("cat") == "user_annotation" and e["name"].startswith("execute_context")
    ]
    require(
        len(annotations) == 12
        and {e["name"] for e in annotations}
        == {f"execute_context_0(0)_generation_{batch}({batch})"},
        "Require exactly 12 pure-generation steps",
    )
    kernels = [e for e in events if e.get("cat") == "kernel"]
    groups = defaultdict(list)
    for event in kernels:
        groups[event["args"]["correlation"]].append(event)
    model_graphs = sorted(
        [
            sorted(g, key=lambda e: e["ts"])
            for g in groups.values()
            if any("FlashAttentionForwardSm100" in e["name"] for e in g)
        ],
        key=lambda g: g[0]["ts"],
    )
    require(len(model_graphs) == 12, "Require 12 actual model graph replays")
    graphs, all_layers, operators, slices = [], [], defaultdict(list), []
    for step, graph in enumerate(model_graphs):
        correlation = graph[0]["args"]["correlation"]
        launches = [
            e
            for e in events
            if e.get("cat") == "cuda_runtime"
            and e["name"].startswith("cudaGraphLaunch")
            and e.get("args", {}).get("correlation") == correlation
        ]
        require(
            len(launches) == 1 and len(graph) == 1490,
            "Require one 1490-kernel CUDA Graph replay",
        )
        classify(graph, candidate and batch == 1)
        layers, phases = layer_phases(graph)
        all_layers.extend(layers)
        span = max(map(end, graph)) - graph[0]["ts"]
        graphs.append(
            {
                "step": step,
                "span_us": span,
                "phase_us": phases,
                "kernel_sum_us": sum(e["dur"] for e in graph),
                "busy_union_us": interval_union(graph),
            }
        )
        for event in graph:
            operators[event["category"]].append(event)
        for layer in layers:
            start = graph[0]["ts"] + layer["start_us"]
            for phase, duration in layer["phase_us"].items():
                slices.append(
                    {
                        "name": phase,
                        "cat": "yoco_elapsed_phase",
                        "ph": "X",
                        "pid": 99001,
                        "tid": 1,
                        "ts": start,
                        "dur": duration,
                        "args": {"step": step, "logical_layer": layer["index"]},
                    }
                )
                start += duration
    correlations = {g[0]["args"]["correlation"] for g in model_graphs}
    outside = [e for e in kernels if e["args"]["correlation"] not in correlations]
    require(len(outside) == 12 * 18, "Expected 18 non-model kernels per step")
    require(
        sum(e["name"] == "_yoco_lm_head_kernel" for e in outside) == 12,
        "LM head outside model graph not proven",
    )
    if annotated_dir:
        annotated_dir.mkdir(parents=True, exist_ok=True)
        trace["traceEvents"] = (
            events
            + slices
            + [
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": 99001,
                    "tid": 0,
                    "args": {"name": "YOCO inferred elapsed phases (not NVTX)"},
                }
            ]
        )
        label = "candidate" if candidate else "native"
        with gzip.open(annotated_dir / f"{label}-b{batch}.json.gz", "wt") as handle:
            json.dump(trace, handle, separators=(",", ":"))
    names = sorted(operators)
    first = model_graphs[0]
    layers, _ = layer_phases(first)
    representative = {
        "step": 0,
        "layers": layers,
        "categories": names,
        "kernels": [
            [
                names.index(e["category"]),
                round(e["ts"] - first[0]["ts"], 3),
                round(e["dur"], 3),
                e["args"]["stream"],
            ]
            for e in first
        ],
    }
    return {
        "batch": batch,
        "candidate": candidate,
        "trace": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "graphs": graphs,
        "layer_observations": len(all_layers),
        "model_graph_us": stats([g["span_us"] for g in graphs]),
        "mean_phase_us_per_step": {
            p: statistics.mean(g["phase_us"][p] for g in graphs)
            for p in graphs[0]["phase_us"]
        },
        "shared_finishes_first_count": sum(
            layer["shared_slack_us"] > 0 for layer in all_layers
        ),
        "shared_slack_us": stats([layer["shared_slack_us"] for layer in all_layers]),
        "shared_late_us_per_step": sum(layer["shared_late_us"] for layer in all_layers)
        / 12,
        "routed_span_us_per_layer": stats(
            [layer["routed_span_us"] for layer in all_layers]
        ),
        "shared_span_us_per_layer": stats(
            [layer["shared_span_us"] for layer in all_layers]
        ),
        "outside_model_kernel_sum_us_per_step": sum(e["dur"] for e in outside) / 12,
        "operators": {
            c: {
                "count_per_step": len(rows) / 12,
                "mean_kernel_us": statistics.mean(e["dur"] for e in rows),
                "sum_us_per_step": sum(e["dur"] for e in rows) / 12,
                "grids": dict(Counter(str(e["args"]["grid"]) for e in rows)),
            }
            for c, rows in sorted(operators.items())
        },
        "representative": representative,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotated-dir", type=Path)
    args = parser.parse_args()
    result = {
        "schema": 1,
        "scope": "FP8 B200 TP1, 40 logical blocks; model graph only",
        "phase_definition": (
            "Consecutive timestamp landmarks; "
            "not kernel sums or a measured CUDA dependency DAG"
        ),
        "runs": {},
    }
    identities = []
    for case, candidate in (("a1", False), ("b1", True)):
        run_dir = args.runs_root / f"m1-final-{case}"
        benchmark = json.loads((run_dir / "benchmark.json").read_text())
        require(
            benchmark["completed"] and benchmark["precision"] == "fp8",
            "Expected completed FP8 run",
        )
        runtime = benchmark["runtime"]
        require(
            runtime["tp"] == runtime["dp"] == 1
            and not runtime["enable_expert_parallel"],
            "Require TP1/DP1/no EP",
        )
        require(
            benchmark["env"]["VLLM_YOCO_FP8_SMALL_M"] == str(int(candidate)),
            "Historical run flag does not match case",
        )
        identity = {
            "gpu_uuid": runtime["gpu_uuid"],
            "deep_gemm_library_sha256": runtime["deep_gemm_library_sha256"],
            "tokens_sha256": benchmark["tokens_sha256"],
        }
        identities.append(identity)
        for profile in benchmark["profiles"]:
            batch = profile["batch"]
            path = run_dir / "traces" / Path(profile["files"][0]).name
            label = "candidate" if candidate else "native"
            result["runs"][f"{label}_b{batch}"] = analyze(
                path, batch, candidate, args.annotated_dir
            )
    require(identities[0] == identities[1], "Mismatched GPU, DeepGEMM or prompt tokens")
    result["identity"] = identities[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    for label, run in result["runs"].items():
        print(
            label,
            "model mean/median us:",
            run["model_graph_us"]["mean"],
            run["model_graph_us"]["median"],
            "shared first:",
            run["shared_finishes_first_count"],
            "/",
            run["layer_observations"],
        )


if __name__ == "__main__":
    main()
