# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit ABBA timings, actual M1 graph dispatch, and forced-token raw scores."""

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(path, batch, enabled):
    with gzip.open(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    annotations = [
        e
        for e in events
        if e.get("cat") == "user_annotation" and e["name"].startswith("execute_context")
    ]
    assert len(annotations) == 12
    assert {e["name"] for e in annotations} == {
        f"execute_context_0(0)_generation_{batch}({batch})"
    }
    kernels = [e for e in events if e.get("cat") == "kernel"]
    groups = defaultdict(list)
    for event in kernels:
        groups[event["args"]["correlation"]].append(event)
    graphs = []
    for correlation, group in groups.items():
        attention = [e for e in group if "FlashAttentionForwardSm100" in e["name"]]
        if not attention:
            continue
        assert len(attention) == 40
        assert sum(e["name"] == "fused_moe_kernel" for e in group) == 80
        launches = [
            e
            for e in events
            if e.get("cat") == "cuda_runtime"
            and e["name"] == "cudaGraphLaunch"
            and e["args"]["correlation"] == correlation
        ]
        assert len(launches) == 1
        direct = [e for e in group if e["name"] == "_small_fp8_direct_kernel"]
        expected = 160 if enabled and batch == 1 else 0
        assert len(direct) == expected, (path, len(direct), expected)
        if direct:
            assert Counter(tuple(e["args"]["grid"]) for e in direct) == {
                (1024, 1, 1): 40,
                (2560, 1, 1): 40,
                (3072, 1, 1): 80,
            }
        dense = [
            e for e in group if "deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl" in e["name"]
        ]
        assert len(dense) == 241 - expected
        graphs.append(
            {
                "span_us": max(e["ts"] + e["dur"] for e in group)
                - min(e["ts"] for e in group),
                "kernel_count": len(group),
                "direct_count": len(direct),
                "direct_sum_us": sum(e["dur"] for e in direct),
                "native_dense_count": len(dense),
            }
        )
    assert len(graphs) == 12
    return {
        "batch": batch,
        "enabled": enabled,
        "trace": path.name,
        "sha256": digest(path),
        "steps": 12,
        "median_model_graph_ms": statistics.median(g["span_us"] for g in graphs) / 1000,
        "graphs": graphs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--prefix", default="m1-final")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs, sources = {}, {}
    for case in ("a1", "b1", "b2", "a2"):
        path = args.runs_root / f"{args.prefix}-{case}" / "benchmark.json"
        run = json.loads(path.read_text())
        assert run["completed"]
        flag = "1" if case.startswith("b") else "0"
        assert run["env"]["VLLM_YOCO_FP8_SMALL_M"] == flag
        linears = run["runtime"]["small_fp8_linears"]
        assert len(linears) == 80
        expected = (
            "YocoM1Fp8LinearKernel" if flag == "1" else "DeepGemmFp8BlockScaledMMKernel"
        )
        assert {row["backend"] for row in linears} == {expected}
        assert [r["batch"] for r in run["benchmark"]] == [1, 2, 8]
        for row in run["benchmark"]:
            assert len(row["seconds"]) == 15
            assert row["output_tokens_per_request"] == 128
            assert {hit for wave in row["cached_prompt_tokens"] for hit in wave} == {
                496
            }
        runs[case] = run
        sources[str(path.relative_to(args.runs_root))] = digest(path)
    assert len({r["runtime"]["gpu_uuid"] for r in runs.values()}) == 1
    native_hash = {r["runtime"]["deep_gemm_library_sha256"] for r in runs.values()}
    assert len(native_hash) == 1 and None not in native_hash
    assert len({r["tokens_sha256"] for r in runs.values()}) == 1
    assert len({r["implementation_sha256"] for r in runs.values()}) == 1
    result = {
        "sources": sources,
        "native_library_sha256": next(iter(native_hash)),
        "runtime": runs["b1"]["runtime"],
        "timings": [],
        "profiles": [],
    }
    for batch in (1, 2, 8):
        row = {"batch": batch, "samples_seconds": {}, "case_tok_s": {}}
        trajectories = {}
        for case, run in runs.items():
            point = next(p for p in run["benchmark"] if p["batch"] == batch)
            row["case_tok_s"][case] = point["total_tokens_per_second"]
            row["samples_seconds"][case] = point["seconds"]
            trajectories[case] = point["last_generated_token_ids"]
        for name, cases in [("native", ["a1", "a2"]), ("candidate", ["b1", "b2"])]:
            seconds = [t for case in cases for t in row["samples_seconds"][case]]
            median = statistics.median(seconds)
            row[name + "_tok_s"] = batch * 128 / median
            row[name + "_mean_step_ms"] = median * 1000 / 128
        row["throughput_change_percent"] = (
            row["candidate_tok_s"] / row["native_tok_s"] - 1
        ) * 100
        row["last_trajectories_equal"] = trajectories["a1"] == trajectories["b1"]
        result["timings"].append(row)
    for case in ("a1", "b1"):
        for p in runs[case]["profiles"]:
            for name in p["files"]:
                path = (
                    args.runs_root
                    / f"{args.prefix}-{case}"
                    / "traces"
                    / Path(name).name
                )
                result["profiles"].append(profile(path, p["batch"], case == "b1"))
    quality = {}
    for case in ("native", "candidate"):
        path = args.runs_root / f"m1-quality-{case}" / "quality.json"
        quality[case] = json.loads(path.read_text())
        assert quality[case]["completed"]
        assert quality[case]["raw_logprob_anchor"]["max_abs"] < 1e-5
        assert quality[case]["runtime"]["deep_gemm_library_sha256"] in native_hash
        assert quality[case]["runtime"]["gpu_uuid"] == runs["b1"]["runtime"]["gpu_uuid"]
        sources[str(path.relative_to(args.runs_root))] = digest(path)
    assert quality["native"]["tokens_sha256"] == quality["candidate"]["tokens_sha256"]
    result["quality"] = {}
    for batch in ("b1", "b8"):
        assert len(quality["native"][batch]) == len(quality["candidate"][batch]) == 8
        native, candidate = [], []
        for a, b in zip(quality["native"][batch], quality["candidate"][batch]):
            assert a["tokens"] == b["tokens"]
            assert len(a["tokens"]) == 128
            native.extend(a["raw_logprobs"])
            candidate.extend(b["raw_logprobs"])
        assert len(native) == len(candidate) == 1024
        assert all(math.isfinite(x) for x in native + candidate)
        delta = -statistics.mean(candidate) + statistics.mean(native)
        result["quality"][batch] = {
            "positions": 1024,
            "native_nll": -statistics.mean(native),
            "candidate_nll": -statistics.mean(candidate),
            "nll_delta": delta,
            "exp_nll_change_percent": 100 * math.expm1(delta),
            "mean_abs_logprob_difference": statistics.mean(
                abs(a - b) for a, b in zip(native, candidate)
            ),
            "max_abs_logprob_difference": max(
                abs(a - b) for a, b in zip(native, candidate)
            ),
            "changed_positions": sum(a != b for a, b in zip(native, candidate)),
            "native_raw_logprobs": native,
            "candidate_raw_logprobs": candidate,
        }
    result["quality"]["long_smoke"] = {
        case: r["long_smoke_tokens"] for case, r in quality.items()
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        [
            (
                r["batch"],
                round(r["native_tok_s"], 2),
                round(r["candidate_tok_s"], 2),
                round(r["throughput_change_percent"], 2),
            )
            for r in result["timings"]
        ]
    )
    print(
        {
            k: {
                name: value
                for name, value in v.items()
                if not name.endswith("raw_logprobs")
            }
            for k, v in result["quality"].items()
            if k != "long_smoke"
        }
    )


if __name__ == "__main__":
    main()
