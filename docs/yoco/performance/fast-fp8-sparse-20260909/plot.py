# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Render the frozen FP8 implementation comparison."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
result = json.loads((root / "comparison.json").read_text())
micro = json.loads((root / "quant-microbench.json").read_text())
colors = {"baseline": "#3264a3", "candidate": "#d86c26"}


def save(fig, name):
    for extension in ["svg", "png"]:
        fig.savefig(root / f"{name}.{extension}", dpi=170)
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for key, label in [("old", "Before"), ("new", "After")]:
    axes[0].loglog(
        [row["m"] for row in micro],
        [row[f"{key}_us"] for row in micro],
        marker="o",
        label=label,
    )
axes[0].set_ylabel("Quantization stage (microseconds)")
axes[0].legend()
axes[1].semilogx(
    [row["m"] for row in micro],
    [row["speedup"] for row in micro],
    marker="o",
    color=colors["candidate"],
)
axes[1].set_ylabel("Stage speedup (before / after)")
for ax in axes:
    ax.set_xlabel("MoE input rows M")
    ax.grid(alpha=0.2)
fig.suptitle("B200 GPU3 | CUDA graphs | median of 3 measurements")
fig.text(
    0.5,
    0.01,
    "Includes row-weight setup; valid FP8 bytes and packed scales match.",
    ha="center",
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "quantization")

fig, ax = plt.subplots(figsize=(8, 4))
rows = result["probes"]
x = list(range(len(rows)))
for key, label, shift in [
    ("baseline", "Before FP8", -0.18),
    ("candidate", "After FP8", 0.18),
]:
    ax.bar(
        [value + shift for value in x],
        [row[key]["output_tok_s"] for row in rows],
        width=0.36,
        label=label,
        color=colors[key],
    )
ax.set_xticks(x, ["C1", "C2", "C4", "C8", "8K / C1"])
ax.set_ylabel("Aggregate output tokens/s")
ax.set_title("Fast FP8 | same B200 GPU2 | 128 output tokens")
ax.legend()
ax.grid(axis="y", alpha=0.2)
fig.text(
    0.5,
    0.01,
    "128 input tokens except 8K/C1; 2 warmups + 3 measurements; shared node.",
    ha="center",
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "low-concurrency")

fig, axes = plt.subplots(1, 4, figsize=(13, 4))
for ax, metric, title in zip(
    axes,
    ["achieved_output_tok_s", "ttft_p95_ms", "itl_p95_ms", "e2e_p95_ms"],
    ["Output tokens/s", "TTFT P95 (ms)", "ITL P95 (ms)", "E2E P95 (ms)"],
):
    ax.bar(
        ["Before", "After"],
        [row[metric] for row in result["trace"]],
        color=list(colors.values()),
    )
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
gates = ["PASS" if row["client_pass"] else "FAIL" for row in result["trace"]]
fig.suptitle("Mooncake toolagent 1.2x | 3643 requests | same B200 GPU2")
fig.text(
    0.5,
    0.01,
    f"Full drain included; client gates {gates[0]} / {gates[1]}. "
    "Shared node, single pair: diagnostic.",
    ha="center",
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "mooncake")
