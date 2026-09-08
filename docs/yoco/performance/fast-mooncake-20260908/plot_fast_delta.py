# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone artifact for the dated same-GPU Fast comparison."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
rows = json.loads((ROOT / "FAST_DELTA.json").read_text())
fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
metrics = [
    ("output_token_throughput", "Output tokens/s", 1),
    ("ttft_p95_ms", "TTFT P95 (s)", 1000),
    ("itl_p95_ms", "ITL P95 (ms)", 1),
    ("e2e_p95_ms", "E2E P95 (s)", 1000),
]
for ax, (key, label, scale) in zip(axes.flat, metrics):
    for offset, field, name, color in [
        (-0.18, "old", "Fast Sep 7", "#9ca3af"),
        (0.18, "current", "Fast Sep 8", "#2471a3"),
    ]:
        bars = ax.bar(
            [i + offset for i in range(2)],
            [r[key][field] / scale for r in rows],
            0.35,
            label=name,
            color=color,
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xticks([0, 1], ["Standalone GPU5", "1P1D P4 / D5"])
    ax.set_ylabel(label)
    ax.grid(axis="y", alpha=0.18)
    ax.set_ylim(
        0, max(r[key][f] / scale for r in rows for f in ["old", "current"]) * 1.25
    )
    ax.legend(fontsize=8)
fig.suptitle("Fast split-KV: Mooncake fixed 1x replay", fontsize=15)
fig.text(
    0.5,
    0.01,
    (
        "Toolagent 300-900s; 3,643 requests; context <=81,920; B200; "
        "separate cache salts.\n"
        "Same physical GPUs, different run times; shared node, one run, no SLO. "
        "Achieved throughput, not peak capacity."
    ),
    ha="center",
    va="bottom",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.085, 1, 0.96))
(ROOT / "figures").mkdir(exist_ok=True)
for ext in ["png", "svg"]:
    fig.savefig(ROOT / "figures" / ("fast-before-after." + ext), dpi=180)
    if ext == "svg":
        output = ROOT / "figures/fast-before-after.svg"
        output.write_text(
            "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n"
        )
