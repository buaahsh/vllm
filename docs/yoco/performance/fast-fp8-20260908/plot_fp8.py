# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rebuild figures from the Fast FP8 diagnostic summaries."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    root = p.parse_args().root
    data = json.loads((root / "comparison.json").read_text())
    out = root / "figures"
    out.mkdir(exist_ok=True)
    colors = ["#606b7a", "#176fca"]
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, routing in enumerate(["spread", "concentrated"]):
        rows = [r for r in data["kernel"] if r["routing"] == routing]
        axes[0].plot(
            [r["tokens"] for r in rows],
            [r["speedup"] for r in rows],
            marker="o",
            label=routing,
        )
    axes[0].axhline(1, color="gray", ls="--", lw=0.8)
    axes[0].set(
        xlabel="Tokens in MoE call (M)",
        ylabel="Baseline / candidate latency",
        title="FP8 MoE CUDA graphs: 3 x ABBA",
        xscale="log",
    )
    axes[0].set_xticks([1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32])
    axes[0].legend()
    rows = [r for r in data["probes"] if r["isl"] == 128]
    x = np.arange(len(rows))
    for i, variant in enumerate(["baseline", "candidate"]):
        axes[1].bar(
            x + (i - 0.5) * 0.36,
            [r[variant]["output_tok_s"] for r in rows],
            0.36,
            label=variant,
            color=colors[i],
        )
        axes[2].plot(
            [r["concurrency"] for r in rows],
            [r[variant]["tpot_ms"] for r in rows],
            marker="o",
            label=variant,
            color=colors[i],
        )
    axes[1].set_xticks(x, [r["concurrency"] for r in rows])
    axes[1].set(
        xlabel="Client concurrency",
        ylabel="Output tokens / second",
        title="ISL128 / OSL128: median of 3 repeats",
    )
    axes[2].set(
        xlabel="Client concurrency",
        ylabel="TPOT (ms)",
        title="Streaming decode latency",
    )
    axes[1].legend()
    fig.suptitle("Fast FP8: bound MoE padding by reachable experts", fontsize=14)
    fig.text(
        0.5,
        0.015,
        "B200; block-128 W8A8 / DeepGEMM UE8M0. Operat"
        "or and fixed-shape diagnostics; client concur"
        "rency is not physical microbatch size.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    for ext in ["png", "svg"]:
        fig.savefig(out / f"kernel-probes.{ext}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    old, new = data["trace"]
    for i, (label, keys, unit) in enumerate(
        [
            (
                "Achieved throughput",
                ["achieved_input_tok_s", "achieved_output_tok_s"],
                "tokens/s",
            ),
            ("Latency P95", ["ttft_p95_ms", "itl_p95_ms", "e2e_p95_ms"], "ms"),
            ("Latency P99", ["ttft_p99_ms", "itl_p99_ms", "e2e_p99_ms"], "ms"),
        ]
    ):
        x = np.arange(len(keys))
        for j, row in enumerate([old, new]):
            bars = axes[i].bar(
                x + (j - 0.5) * 0.36,
                [row[k] for k in keys],
                0.36,
                label=row["variant"],
                color=colors[j],
            )
            axes[i].bar_label(
                bars,
                fmt=lambda value: f"{value / 1000:.1f}k"
                if value >= 1000
                else f"{value:.1f}",
                fontsize=8,
                padding=3,
            )
        axes[i].set_yscale("log")
        axes[i].set_xticks(x, ["Input", "Output"] if i == 0 else ["TTFT", "ITL", "E2E"])
        axes[i].set(title=label, ylabel=unit)
        axes[i].margins(y=0.2)
    axes[0].legend()
    fig.suptitle("Mooncake FAST'25 toolagent: Fast FP8 before / after", fontsize=14)
    status = "; ".join(
        f"{r['variant']} {r['completed']}/{r['planned']}"
        f", errors={r['error_records']}, drained={r['drained']}"
        for r in data["trace"]
    )
    fig.text(
        0.5,
        0.02,
        "1.2x timestamps, 500s arrivals, 3643 requests"
        "; same B200 GPU2, TP1 standalone, maxlen81920"
        ", unique salts; SHA5317e230...960c.\n"
        + status
        + (
            ".\nBoth client schedule/ceiling gates FAIL. S"
            "ingle-pair shared-node diagnostic; no latency"
            " SLO."
        ),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    for ext in ["png", "svg"]:
        fig.savefig(out / f"mooncake.{ext}", dpi=160)
    plt.close(fig)
    timeline = list(csv.DictReader((root / "timeline.csv").open()))
    arrivals = list(csv.DictReader((root / "arrivals.csv").open()))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.step(
        [float(r["time_s"]) for r in arrivals],
        [int(r["planned"]) for r in arrivals],
        label="Planned arrivals",
        color="black",
        lw=1.5,
    )
    for j, variant in enumerate(["baseline", "candidate"]):
        for event, style in [("completed", "-"), ("sent", "--")]:
            rows = [
                r for r in timeline if r["variant"] == variant and r["event"] == event
            ]
            ax.step(
                [float(r["time_s"]) for r in rows],
                [int(r["completed"]) for r in rows],
                label=f"{variant} {event}",
                color=colors[j],
                ls=style,
                lw=1.2,
            )
    ax.axvline(500, color="gray", ls="--", lw=0.8)
    ax.set(
        xlabel="Seconds since each case's first actual send",
        ylabel="Cumulative requests",
        title="Mooncake 1.2x: arrivals and complete drain",
    )
    ax.legend()
    fig.text(
        0.5,
        0.015,
        "Same frozen trace (SHA256 5317e230...960c), G"
        "PU2 standalone, isolated cache salts. Shared-"
        "node diagnostic; completion lines include err"
        "ors if any.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ["png", "svg"]:
        fig.savefig(out / f"arrival-completion.{ext}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
