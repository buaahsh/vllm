# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize the combined draft acceptance panel and numerical controls."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
REFERENCE = Path(
    "/data/vllm_test/work/yoco-draft-projections-20261021/results/parent-a"
)
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
            metric["value"]
            for metric in row["metrics_" + when]
            if metric["name"] == "vllm:" + name
        )

    return value("after") - value("before")


def per_position(row):
    def value(when):
        return next(
            metric["values"]
            for metric in row["metrics_" + when]
            if metric["name"] == "vllm:spec_decode_num_accepted_tokens_per_pos"
        )

    return [a - b for a, b in zip(value("after"), value("before"))]


def compare(actual, expected):
    return {
        "outputs_equal": actual["output_ids"] == expected["output_ids"],
        "counts_equal": all(
            counter(actual, key) == counter(expected, key) for key in COUNTERS
        ),
        "per_position_equal": per_position(actual) == per_position(expected),
    }


def check_rows(rows):
    for row in rows:
        assert all(len(ids) == 128 for ids in row["output_ids"])
        assert all(
            not metric["is_corrupted"] and not metric["num_preemptions"]
            for metric in row["request_metrics"]
        )


def main():
    native = read(RESULTS / "native/complete.json")
    assert native["status"] == "PASS"
    native_rows = read(RESULTS / "native/trials.json")
    parent_rows = read(REFERENCE / "trials.json")
    assert len(native_rows) == len(parent_rows) == 30
    assert read(RESULTS / "native/manifest.json") == read(REFERENCE / "manifest.json")
    assert read(RESULTS / "native/prompt-ids.json") == read(
        REFERENCE / "prompt-ids.json"
    )
    for name in ["first-token.json", "teacher-forced.json"]:
        assert read(RESULTS / "native" / name) == read(REFERENCE / name)
    check_rows(native_rows)
    native_comparisons = []
    for actual, expected in zip(native_rows, parent_rows):
        assert all(
            actual[key] == expected[key]
            for key in ["profile", "concurrency", "trial", "warmup", "seed"]
        )
        result = compare(actual, expected)
        assert all(result.values()), result
        native_comparisons.append(
            {
                key: actual[key]
                for key in ["profile", "concurrency", "trial", "warmup", "seed"]
            }
            | result
        )
    worker = native["worker"][0]
    assert worker["graphs"]["target"]["captured_tokens"] == [9, 18, 36, 72]
    assert worker["graphs"]["draft"]["captured_tokens"] == [8, 16, 24, 32, 40, 64]
    assert all(graph["replays"] for graph in worker["graphs"].values())
    assert worker["draft_logits_dtype"] == "torch.float32"
    assert worker["fast_prefill"]
    assert worker["dispatch"]["profile"] == "b200-l3-fp8-tp1-k8"

    paired = read(RESULTS / "paired/complete.json")
    assert paired["status"] == "PASS"
    paired_worker = paired["worker"][0]
    assert paired_worker["pending"] is None and len(paired_worker["checks"]) == 6
    assert all(
        graph["tokens"] == [8, 16, 24, 32, 40, 64] and graph["replays"]
        for graph in paired_worker["audit_graphs"].values()
    )
    data = read(RESULTS / "paired/trials.json")
    check_rows(data)
    comparisons = []
    cells = []
    for length in [0, 8193]:
        for c in [1, 4, 8]:
            selected = {
                mode: [
                    row
                    for row in data
                    if row["length"] == length
                    and row["concurrency"] == c
                    and row["mode"] == mode
                ]
                for mode in ["baseline", "combined"]
            }
            assert all(len(rows) == 5 for rows in selected.values())
            for actual, expected in zip(selected["combined"], selected["baseline"]):
                assert actual["trial"] == expected["trial"]
                result = compare(actual, expected)
                assert all(result.values()), result
                comparisons.append(
                    {
                        "length": length,
                        "concurrency": c,
                        "trial": actual["trial"],
                        "warmup": actual["warmup"],
                    }
                    | result
                )
            speed = {}
            acceptance = {}
            for mode, rows in selected.items():
                scored = [row for row in rows if not row["warmup"]]
                assert len(scored) == 3
                speed[mode] = sum(row["tokens"] for row in scored) / sum(
                    row["elapsed_s"] for row in scored
                )
                acceptance[mode] = sum(
                    counter(row, COUNTERS[0]) for row in scored
                ) / sum(counter(row, COUNTERS[1]) for row in scored)
            component = paired_worker["checks"][
                [0, 8193].index(length) * 3 + [1, 4, 8].index(c)
            ]
            assert component["batch"] == c
            if length:
                assert min(component["prefix_lens"]) >= length
            cells.append(
                {
                    "length": length,
                    "concurrency": c,
                    "baseline_draft_ms": component["timing"]["median_ms"]["baseline"],
                    "combined_draft_ms": component["timing"]["median_ms"]["combined"],
                    "draft_latency_reduction_pct": component["timing"][
                        "latency_reduction_pct"
                    ],
                    "baseline_tokens_per_s": speed["baseline"],
                    "combined_tokens_per_s": speed["combined"],
                    "throughput_change_pct": (speed["combined"] / speed["baseline"] - 1)
                    * 100,
                    "baseline_acceptance": acceptance["baseline"],
                    "combined_acceptance": acceptance["combined"],
                }
            )
    summary = {
        "status": "PASS",
        "decision": "ACCEPTED_DRAFT_SIDE_GAINS",
        "base_commit": "7b525571d0af65f89d72970c9a3462fac1f5c990",
        "accepted_code_commits": [
            "04088223dfb93d805caf17e4e8e81bc40aab1177",
            "8d8bd197c7f4e87c7d4b5c294243967a13c953e9",
        ],
        "native_formal_pairs": 18,
        "native_warmup_pairs": 12,
        "native_reference": str(REFERENCE),
        "native_reference_scope": (
            "Archived numerical control only; not a current timing comparator."
        ),
        "native_comparisons": native_comparisons,
        "native_first_token_and_teacher_forced_exact": True,
        "paired_formal_pairs": 18,
        "paired_warmup_pairs": 12,
        "paired_comparisons": comparisons,
        "same_input_exact_cells": 6,
        "production_graph_counts": {"target": 4, "draft": 6},
        "paired_timing_scope": (
            "Three scored seed pairs per cell in one engine. GPU events and"
            " same-input comparisons occur only in separate unscored audit "
            "warmups. No stable end-to-end gain claimed."
        ),
        "cells": cells,
    }
    save(ROOT / "summary.json", summary)
    with (ROOT / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(cells[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(cells)
    save(ROOT / "worker-receipts.json", {"native": native, "paired": paired})
    print(json.dumps({"status": summary["status"], "cells": cells}, indent=2))


if __name__ == "__main__":
    main()
