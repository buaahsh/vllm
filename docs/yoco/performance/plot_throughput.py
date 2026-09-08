# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rebuild comparison figures from the current table and retained case records."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
NAMES = {"qwen3": "Qwen3", "align": "Align GEMM", "fast": "Fast"}
COLORS = {"qwen3": "#607d8b", "align": "#dc7633", "fast": "#2471a3"}
MODES = ["qwen3", "align", "fast"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "throughput/figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    current = json.loads((ROOT / "throughput/current.json").read_text())
    topologies = [("standalone", "Standalone (GPU5)"), ("pd", "1P1D (P4 / D5)")]
    caption = (
        "Mooncake toolagent 300-900s, 1x; B200; context <=81920; "
        "same physical GPUs, different run times.\n"
        "Shared-node, single-run diagnostic; achieved throughput, "
        "not peak capacity. * Incomplete run."
    )

    def finish(fig, name):
        fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        for ext in ["svg", "png"]:
            fig.savefig(args.out / (name + "." + ext), dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for j, (topology, title) in enumerate(topologies):
        rows = [
            current[f"{mode}/{topology}"]
            for mode in MODES
            if f"{mode}/{topology}" in current
        ]
        labels = [
            NAMES[r["mode"]] + ("*" if r["completed"] != r["planned"] else "")
            for r in rows
        ]
        for i, (metric, label) in enumerate(
            [("output_tok_s", "Output tokens/s"), ("input_tok_s", "Input tokens/s")]
        ):
            ax = axes[i, j]
            values = [r[metric] for r in rows]
            bars = ax.bar(labels, values, color=[COLORS[r["mode"]] for r in rows])
            ax.bar_label(bars, fmt="%.1f", padding=3)
            ax.set_ylim(0, max(values) * 1.22)
            ax.set_ylabel(label)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.2)
    finish(fig, "throughput")
    fig, axes = plt.subplots(3, 2, figsize=(11, 9))
    for j, (topology, title) in enumerate(topologies):
        rows = [
            current[f"{mode}/{topology}"]
            for mode in MODES
            if f"{mode}/{topology}" in current
        ]
        labels = [
            NAMES[r["mode"]] + ("*" if r["completed"] != r["planned"] else "")
            for r in rows
        ]
        for i, (metric, label, scale) in enumerate(
            [
                ("time_to_first_token", "TTFT (s)", 1000),
                ("inter_token_latency", "ITL (ms)", 1),
                ("request_latency", "E2E (s)", 1000),
            ]
        ):
            ax = axes[i, j]
            x = np.arange(len(rows))
            width = 0.36
            for offset, percentile, color in [
                (-width / 2, "p95", "#2471a3"),
                (width / 2, "p99", "#a9cce3"),
            ]:
                ax.bar(
                    x + offset,
                    [r["source_metrics"][metric][percentile] / scale for r in rows],
                    width,
                    label=percentile.upper(),
                    color=color,
                )
            ax.set_xticks(x, labels)
            ax.set_ylabel(label)
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.2)
    finish(fig, "latency")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ratios = []
    for ax, (topology, title) in zip(axes, topologies):
        fast = current.get(f"fast/{topology}")
        if fast:
            peers = [
                current[f"{mode}/{topology}"]
                for mode in ["qwen3", "align"]
                if f"{mode}/{topology}" in current
            ]
            values = [
                100 * (fast["output_tok_s"] / r["output_tok_s"] - 1) for r in peers
            ]
            labels = [
                "vs "
                + NAMES[r["mode"]]
                + ("*" if r["completed"] != r["planned"] else "")
                for r in peers
            ]
            bars = ax.bar(labels, values, color=[COLORS[r["mode"]] for r in peers])
            ax.bar_label(bars, fmt="%+.1f%%", padding=3)
            for peer, value in zip(peers, values):
                ratios.append(
                    dict(
                        topology=topology,
                        reference=peer["mode"],
                        fast_relative_pct=value,
                        reference_complete=peer["completed"] == peer["planned"],
                    )
                )
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(title)
        ax.set_ylabel("Fast achieved output throughput change (%)")
        ax.margins(y=0.3)
    finish(fig, "fast-relative")
    (args.out / "fast-relative.json").write_text(json.dumps(ratios, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    for ax, (topology, title) in zip(axes, topologies):
        for mode in MODES:
            r = current.get(f"{mode}/{topology}")
            if r is None:
                continue
            case = (ROOT / r["manifest"]).parent
            records = [
                json.loads(line)
                for line in (case / "artifacts/profile_export.jsonl")
                .read_text()
                .splitlines()
            ]
            starts = [row["metadata"]["request_start_ns"] / 1e9 for row in records]
            first = min(starts)
            # All attempts have a request end; success is distinguished below.
            ends = [
                row["metadata"]["request_end_ns"] / 1e9 - first
                for row in records
                if not row.get("error")
            ]
            sends = np.sort(np.array(starts) - first)
            label = NAMES[mode] + ("*" if r["completed"] != r["planned"] else "")
            ax.step(
                sends,
                np.arange(1, len(sends) + 1),
                where="post",
                color=COLORS[mode],
                alpha=0.4,
                linestyle=":",
            )
            ax.step(
                np.sort(ends),
                np.arange(1, len(ends) + 1),
                where="post",
                color=COLORS[mode],
                label=label,
            )
        # Same trace bytes for all entries; first planned arrival aligns to first send.
        trace_path = (
            ROOT
            / "fast-compare-b200-20260907/traces/toolagent-ctx81920-t300-900s-f1.jsonl"
        )
        trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
        arrivals = np.array([row["timestamp"] for row in trace])
        arrivals = (arrivals - arrivals[0]) / 1000
        ax.step(
            arrivals,
            np.arange(1, len(arrivals) + 1),
            where="post",
            color="black",
            linestyle="--",
            label="Planned",
        )
        ax.set_title(title + "\nSolid: successful completions; dotted: sends")
        ax.set_xlabel("Seconds since first actual send")
        ax.set_ylabel("Requests")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    finish(fig, "completion")
    print(args.out)


if __name__ == "__main__":
    main()
