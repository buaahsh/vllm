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
old, new = data["trace"]
labels = ["Before reuse", "Current FP8"]
colors = ["#727c8e", "#187d8d"]
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    }
)
fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
for ax, key, title, unit in [
    (axes[0], "achieved_output_tok_s", "Output throughput", "tokens/s"),
    (axes[1], "ttft_p95_ms", "TTFT P95", "seconds"),
    (axes[2], "itl_p95_ms", "ITL P95", "ms"),
]:
    values = [r[key] / (1000 if key == "ttft_p95_ms" else 1) for r in [old, new]]
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.bar_label(bars, labels=[f"{v:,.2f}" for v in values], padding=4)
    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.set_ylim(0, max(values) * 1.2)
fig.suptitle(
    "YOCO Fast FP8 | Mooncake toolagent 2x | "
    f"throughput {(data['trace_speedup'] - 1) * 100:+.2f}%"
)
fig.text(
    0.5,
    0.02,
    "Same B200 GPU2, standalone TP1/DP1; 3,643 requests; source 300–900 s; "
    "clean cache salts; full drain.\n"
    "One pair on a shared node, no latency SLO: diagnostic, "
    "not capacity certification.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.13, 1, 0.92))
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
for label, color in zip(["baseline", "candidate"], colors):
    completed = [
        r for r in times if r["variant"] == label and r["event"] == "completed"
    ]
    ax.step(
        [float(r["time_s"]) for r in completed],
        [int(r["completed"]) for r in completed],
        where="post",
        color=color,
        label=labels[0 if label == "baseline" else 1],
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
    "Trace e0df4ca65eea…; maxlen 81,920; same GPU and request "
    "lengths/prefix relationships.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
for ext in ["svg", "png"]:
    fig.savefig(root / f"timeline.{ext}", dpi=160)
