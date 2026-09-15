# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
data = json.loads((root / "comparison.json").read_text())
rows = data["trace"]
labels = ["Fast BF16", "Fast FP8"]
colors = ["#727c8e", "#187d8d"]
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    }
)
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, key, title, unit, scale in [
    (axes[0, 0], "achieved_output_tok_s", "Output throughput", "tokens/s", 1),
    (axes[0, 1], "achieved_input_tok_s", "Input throughput", "tokens/s", 1),
    (
        axes[0, 2],
        "overhang_from_first_actual_send_s",
        "Drain after arrivals",
        "seconds",
        1,
    ),
]:
    values = [r[key] / scale for r in rows]
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.bar_label(bars, labels=[f"{v:,.2f}" for v in values], padding=4)
    ax.set(title=title, ylabel=unit, ylim=(0, max(values) * 1.23))
for ax, key, title, unit, scale in [
    (axes[1, 0], "ttft", "TTFT", "seconds", 1000),
    (axes[1, 1], "itl", "ITL", "ms", 1),
    (axes[1, 2], "e2e", "E2E latency", "seconds", 1000),
]:
    maxima = []
    for i, (r, color, label) in enumerate(zip(rows, colors, labels)):
        values = [r[f"{key}_p{p}_ms"] / scale for p in [95, 99]]
        maxima.extend(values)
        bars = ax.bar(
            [x + (i - 0.5) * 0.36 for x in [0, 1]],
            values,
            width=0.34,
            color=color,
            label=label,
        )
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=3, fontsize=9)
    ax.set(
        xticks=[0, 1],
        xticklabels=["P95", "P99"],
        title=title,
        ylabel=unit,
        ylim=(0, max(maxima) * 1.3),
    )
    if key == "e2e":
        ax.legend(loc="upper left", fontsize=8)
fig.suptitle(
    f"YOCO Fast | Mooncake toolagent 2x | FP8 / BF16 {data['trace_speedup']:.3f}x"
)
fig.text(
    0.5,
    0.02,
    "Same B200 GPU2, standalone TP1/DP1; 3,643 requests; source 300–900 s; "
    "maxlen 81,920; unique cache salts; full drain.\n"
    "BF16 newly measured; FP8 retained from preceding run. "
    "Shared node, one replay each, no latency SLO: diagnostic.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.09, 1, 0.95))
for ext in ["svg", "png"]:
    fig.savefig(root / f"mooncake.{ext}", dpi=160)
plt.close(fig)
times = list(csv.DictReader((root / "timeline.csv").open()))
planned = list(csv.DictReader((root / "planned-arrivals.csv").open()))
fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.step(
    [float(r["time_s"]) for r in planned],
    [int(r["requests"]) for r in planned],
    where="post",
    color="#444",
    linestyle="--",
    label="Planned arrivals",
)
for variant, color, label in zip(["bf16", "fp8"], colors, labels):
    completed = [
        r for r in times if r["variant"] == variant and r["event"] == "completed"
    ]
    ax.step(
        [float(r["time_s"]) for r in completed],
        [int(r["completed"]) for r in completed],
        where="post",
        color=color,
        label=label,
    )
ax.axvline(300, color="#aaa", linestyle=":")
ax.set(
    xlabel="Seconds from first actual send",
    ylabel="Cumulative requests",
    title="Mooncake 2x arrivals and complete drain",
)
ax.legend()
fig.text(
    0.5,
    0.015,
    "Trace e0df4ca65eea…; same GPU, request lengths and prefix relationships; "
    "shared-node diagnostic.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
for ext in ["svg", "png"]:
    fig.savefig(root / f"timeline.{ext}", dpi=160)
