# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export the audited Fast FP8 decode analysis as standalone figures."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output stem")
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    profiles = {p["batch"]: p for p in data["profiles"]}
    benchmark = {r["batch"]: r for r in data["benchmark"]}
    b1 = profiles[1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), width_ratios=[1, 1.6])
    labels = [
        "8 TB/s operand reference",
        "800 tok/s target",
        "Model graph",
        "Offline step",
    ]
    values = [
        data["reference"]["kv_scenarios"][1]["weight_plus_kv_ms_at_8TBps"],
        1.25,
        b1["median_graph_span_us"] / 1000,
        benchmark[1]["mean_output_step_ms"],
    ]
    axes[0].barh(labels, values, color=["#94a3b8", "#f59e0b", "#2563eb", "#123b76"])
    for i, value in enumerate(values):
        axes[0].text(value + 0.06, i, f"{value:.3f}", va="center", fontsize=10)
    axes[0].set_xlim(0, 6.8)
    axes[0].invert_yaxis()
    axes[0].set_title("B1: bandwidth arithmetic vs. observed time", loc="left")
    axes[0].set_xlabel("ms / output step")
    groups = [r["label"] for r in b1["groups"]]
    y = np.arange(len(groups))
    for batch, offset, color in [(1, -0.18, "#2563eb"), (8, 0.18, "#8b5cf6")]:
        times = {r["label"]: r["us_per_step"] / 1000 for r in profiles[batch]["groups"]}
        axes[1].barh(
            y + offset,
            [times[g] for g in groups],
            height=0.34,
            color=color,
            label=f"B{batch}",
        )
    axes[1].set_yticks(y, groups)
    axes[1].invert_yaxis()
    axes[1].set_title("GPU kernel duration sums by stage", loc="left")
    axes[1].set_xlabel("ms / generation step (overlapping streams counted separately)")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle(
        "YOCO v0.29 Fast FP8 | single B200 | 512-token warm prefix", fontsize=15
    )
    fig.text(
        0.02,
        0.035,
        "15 unprofiled timings; 12 pure decode graph replays per batch. "
        "B1 = 170.64 tok/s; B8 total = 815.13 tok/s.\n"
        "Operand reference excludes other traffic/cache effects. Model graph "
        "excludes LM head and sampling. Kernel sums are not request latency.",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    svg = args.output.with_suffix(".svg")
    fig.savefig(svg, bbox_inches="tight")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    fig.savefig(args.output.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
