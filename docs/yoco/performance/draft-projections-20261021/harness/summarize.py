# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODES = ["baseline", "rope", "gate-up"]
COUNTERS = [
    "spec_decode_num_accepted_tokens",
    "spec_decode_num_draft_tokens",
    "spec_decode_num_drafts",
]


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def counter(row, name):
    def value(when):
        return sum(
            m["value"] for m in row["metrics_" + when] if m["name"] == "vllm:" + name
        )

    return value("after") - value("before")


def per_pos(row):
    def value(when):
        return next(
            m["values"]
            for m in row["metrics_" + when]
            if m["name"] == "vllm:spec_decode_num_accepted_tokens_per_pos"
        )

    return [a - b for a, b in zip(value("after"), value("before"))]


def stats(rows):
    return {
        "tokens_per_s": sum(r["tokens"] for r in rows)
        / sum(r["elapsed_s"] for r in rows),
        "accepted": sum(counter(r, COUNTERS[0]) for r in rows),
        "proposed": sum(counter(r, COUNTERS[1]) for r in rows),
        "rounds": sum(counter(r, COUNTERS[2]) for r in rows),
        "median_ttft_ms": statistics.median(
            m["first_token_latency"] * 1000 for r in rows for m in r["request_metrics"]
        ),
    }


def compare(actual, reference):
    return {
        "outputs_equal": actual["output_ids"] == reference["output_ids"],
        "counts_equal": all(
            counter(actual, k) == counter(reference, k) for k in COUNTERS
        ),
        "per_position_equal": per_pos(actual) == per_pos(reference),
    }


def csvsave(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if (RESULTS / "paired/complete.json").exists():
    out = RESULTS / "paired"
    complete = read(out / "complete.json")
    data = read(out / "trials.json")
    assert complete["status"] == "PASS"
    w = complete["worker"][0]
    assert w["pending"] is None and len(w["same_input"]) == 6
    assert w["worker"]["graphs"]["target"]["captured_tokens"] == [9, 18, 36, 72]
    assert all(
        value["tokens"] == [8, 16, 24, 32, 40, 64] and value["replays"]
        for value in w["audit_graphs"].values()
    )
    cells = []
    comparisons = []
    trials = []
    for row in data:
        assert all(len(ids) == 128 for ids in row["output_ids"])
        assert all(
            not m["is_corrupted"] and not m["num_preemptions"]
            for m in row["request_metrics"]
        )
        trials.append(
            {
                k: row[k]
                for k in [
                    "length",
                    "concurrency",
                    "mode",
                    "trial",
                    "warmup",
                    "seed",
                    "elapsed_s",
                    "tokens",
                ]
            }
            | {name: counter(row, name) for name in COUNTERS}
            | {
                "output_sha256": hashlib.sha256(
                    json.dumps(row["output_ids"], separators=(",", ":")).encode()
                ).hexdigest()
            }
        )
    rng = random.Random(42)
    for length in [0, 8193]:
        for c in [1, 4, 8]:
            variants = {
                mode: [
                    r
                    for r in data
                    if r["length"] == length
                    and r["concurrency"] == c
                    and r["mode"] == mode
                    and not r["warmup"]
                ]
                for mode in MODES
            }
            assert all(len(rows) == 7 for rows in variants.values())
            speeds = {mode: stats(rows) for mode, rows in variants.items()}
            for mode in MODES[1:]:
                baseline, candidate = variants["baseline"], variants[mode]
                for a, b in zip(candidate, baseline):
                    assert a["trial"] == b["trial"] and a["seed"] == b["seed"]
                    comparisons.append(
                        {
                            "length": length,
                            "concurrency": c,
                            "mode": mode,
                            "trial": a["trial"],
                        }
                        | compare(a, b)
                    )
                boot = []
                for _ in range(4000):
                    sample = rng.choices(range(7), k=7)
                    boot.append(
                        (
                            sum(baseline[i]["elapsed_s"] for i in sample)
                            / sum(candidate[i]["elapsed_s"] for i in sample)
                            - 1
                        )
                        * 100
                    )
                boot.sort()
                cell = {
                    "length": length,
                    "concurrency": c,
                    "mode": mode,
                    "baseline_tokens_per_s": speeds["baseline"]["tokens_per_s"],
                    "candidate_tokens_per_s": speeds[mode]["tokens_per_s"],
                    "throughput_change_pct": (
                        speeds[mode]["tokens_per_s"]
                        / speeds["baseline"]["tokens_per_s"]
                        - 1
                    )
                    * 100,
                    "bootstrap_p025_pct": boot[100],
                    "bootstrap_p975_pct": boot[3900],
                    "baseline_acceptance": speeds["baseline"]["accepted"]
                    / speeds["baseline"]["proposed"],
                    "candidate_acceptance": speeds[mode]["accepted"]
                    / speeds[mode]["proposed"],
                    "baseline_ttft_ms": speeds["baseline"]["median_ttft_ms"],
                    "candidate_ttft_ms": speeds[mode]["median_ttft_ms"],
                }
                cells.append(cell)
    summary = {
        "status": "PASS",
        "cells": cells,
        "comparisons": comparisons,
        "all_outputs_equal": all(c["outputs_equal"] for c in comparisons),
        "all_counters_equal": all(
            c["counts_equal"] and c["per_position_equal"] for c in comparisons
        ),
        "scored_batches": sum(not r["warmup"] for r in data),
        "scored_requests": sum(r["concurrency"] for r in data if not r["warmup"]),
        "scored_output_tokens": sum(r["tokens"] for r in data if not r["warmup"]),
        "ci_scope": (
            "Paired bootstrap over seven fixed-panel seed/timing pairs. "
            "Includes order/time noise and seed-dependent work; not a "
            "population/task-quality interval."
        ),
    }
    save(out / "summary.json", summary)
    csvsave(out / "summary.csv", cells)
    csvsave(out / "trials.csv", trials)
    print(
        json.dumps(
            {
                "paired": cells,
                "all_outputs_equal": summary["all_outputs_equal"],
                "all_counters_equal": summary["all_counters_equal"],
            },
            indent=2,
        )
    )

runs = ["parent-a", "rope-r", "gate-up-g", "parent-b"]
if all((RESULTS / run / "complete.json").exists() for run in runs):
    rows = {run: read(RESULTS / run / "trials.json") for run in runs}
    cells = []
    comparisons = []
    trials = []
    quality = {}
    for run in runs:
        complete = read(RESULTS / run / "complete.json")
        assert complete["status"] == "PASS"
        worker = complete["worker"][0]
        assert worker["graphs"]["target"]["captured_tokens"] == [9, 18, 36, 72]
        assert worker["graphs"]["draft"]["captured_tokens"] == [8, 16, 24, 32, 40, 64]
        assert all(g["replays"] for g in worker["graphs"].values())
        assert (
            worker["fast_prefill"] and worker["draft_logits_dtype"] == "torch.float32"
        )
        assert worker["dispatch"]["profile"] == "b200-l3-fp8-tp1-k8"
        assert read(RESULTS / run / "manifest.json") == read(
            RESULTS / "parent-a/manifest.json"
        )
        assert read(RESULTS / run / "prompt-ids.json") == read(
            RESULTS / "parent-a/prompt-ids.json"
        )
        quality[run] = {
            name: read(RESULTS / run / (name + ".json"))
            == read(RESULTS / "parent-a" / (name + ".json"))
            for name in ["first-token", "teacher-forced"]
        }
        for row in rows[run]:
            assert all(len(ids) == 128 for ids in row["output_ids"])
            assert all(
                not m["is_corrupted"] and not m["num_preemptions"]
                for m in row["request_metrics"]
            )
            trials.append(
                {
                    k: row[k]
                    for k in [
                        "profile",
                        "concurrency",
                        "trial",
                        "warmup",
                        "seed",
                        "elapsed_s",
                        "tokens",
                    ]
                }
                | {"run": run}
                | {name: counter(row, name) for name in COUNTERS}
                | {
                    "output_sha256": hashlib.sha256(
                        json.dumps(row["output_ids"], separators=(",", ":")).encode()
                    ).hexdigest()
                }
            )
        if run != "parent-a":
            for a, b in zip(rows[run], rows["parent-a"]):
                assert all(
                    a[k] == b[k]
                    for k in ["profile", "concurrency", "trial", "warmup", "seed"]
                )
                comparisons.append(
                    {
                        k: a[k]
                        for k in ["profile", "concurrency", "trial", "warmup", "seed"]
                    }
                    | {"run": run}
                    | compare(a, b)
                )
    for profile in ["short", "context8193"]:
        for c in [1, 4, 8]:
            selected = {
                run: [
                    r
                    for r in rows[run]
                    if r["profile"] == profile
                    and r["concurrency"] == c
                    and not r["warmup"]
                ]
                for run in runs
            }
            speeds = {run: stats(data) for run, data in selected.items()}
            baseline = stats(selected["parent-a"] + selected["parent-b"])
            for run in ["rope-r", "gate-up-g"]:
                cells.append(
                    {
                        "profile": profile,
                        "concurrency": c,
                        "run": run,
                        "baseline_tokens_per_s": baseline["tokens_per_s"],
                        "candidate_tokens_per_s": speeds[run]["tokens_per_s"],
                        "throughput_change_pct": (
                            speeds[run]["tokens_per_s"] / baseline["tokens_per_s"] - 1
                        )
                        * 100,
                        "relative_parent_a_pct": (
                            speeds[run]["tokens_per_s"]
                            / speeds["parent-a"]["tokens_per_s"]
                            - 1
                        )
                        * 100,
                        "relative_parent_b_pct": (
                            speeds[run]["tokens_per_s"]
                            / speeds["parent-b"]["tokens_per_s"]
                            - 1
                        )
                        * 100,
                        "baseline_acceptance": baseline["accepted"]
                        / baseline["proposed"],
                        "candidate_acceptance": speeds[run]["accepted"]
                        / speeds[run]["proposed"],
                        "baseline_ttft_ms": baseline["median_ttft_ms"],
                        "candidate_ttft_ms": speeds[run]["median_ttft_ms"],
                    }
                )
    summary = {
        "status": "PASS",
        "run_order": runs,
        "quality": quality,
        "cells": cells,
        "comparisons": comparisons,
        "all_outputs_equal": all(c["outputs_equal"] for c in comparisons),
        "all_counters_equal": all(
            c["counts_equal"] and c["per_position_equal"] for c in comparisons
        ),
        "scored_batches": sum(not r["warmup"] for data in rows.values() for r in data),
        "scored_requests": sum(
            r["concurrency"] for data in rows.values() for r in data if not r["warmup"]
        ),
        "scored_output_tokens": sum(
            r["tokens"] for data in rows.values() for r in data if not r["warmup"]
        ),
        "scope": (
            "Two bracketing baseline processes and one native process per "
            "candidate. Three scored seeds per cell per process. Check "
            "same-engine paired experiment for order control."
        ),
    }
    save(RESULTS / "native-summary.json", summary)
    csvsave(RESULTS / "native-summary.csv", cells)
    csvsave(RESULTS / "native-trials.csv", trials)
    print(
        json.dumps(
            {
                "native": cells,
                "all_outputs_equal": summary["all_outputs_equal"],
                "all_counters_equal": summary["all_counters_equal"],
            },
            indent=2,
        )
    )
