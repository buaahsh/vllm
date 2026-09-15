# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
r = json.loads((root / "comparison.json").read_text())
colors = {"bf16": "#3264a3", "fp8": "#d86c26"}
rows = r["probes"]
x = list(range(len(rows)))
labels = ["C1", "C2", "C4", "C8", "8K / C1"]
fig, ax = plt.subplots(figsize=(8, 4))
for variant, shift in [("bf16", -0.18), ("fp8", 0.18)]:
    val = [q[variant]["output_tok_s"] for q in rows]
    lo = [q[variant]["output_tok_s"] - q[variant]["min_tok_s"] for q in rows]
    hi = [q[variant]["max_tok_s"] - q[variant]["output_tok_s"] for q in rows]
    ax.bar(
        [v + shift for v in x],
        val,
        width=0.36,
        yerr=[lo, hi],
        capsize=3,
        color=colors[variant],
        label=variant.upper(),
    )
ax.set_xticks(x, labels)
ax.set_ylabel("Output tokens/s (end to end)")
ax.legend()
ax.grid(axis="y", alpha=0.2)
ax.set_title("Fast BF16 vs block FP8 | B200 TP1 | 128 output tokens")
fig.text(
    0.5,
    0.01,
    "128 input tokens except 8K/C1; median and range of 3 runs. "
    "One same-GPU pair; diagnostic.",
    ha="center",
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
for ext in ["svg", "png"]:
    fig.savefig(root / f"low-concurrency.{ext}", dpi=170)
plt.close(fig)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
for ax, key, title in zip(
    axes,
    ["achieved_output_tok_s", "ttft_p95_ms", "itl_p95_ms", "e2e_p95_ms"],
    ["Output tokens/s", "TTFT P95 (ms)", "ITL P95 (ms)", "E2E P95 (ms)"],
):
    vals = [v[key] for v in r["trace"]]
    ax.bar(["BF16", "FP8"], vals, color=[colors["bf16"], colors["fp8"]])
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
fig.suptitle("Mooncake FAST25 toolagent 1.2x | 3643 requests | same B200 | diagnostic")
fig.tight_layout()
for ext in ["svg", "png"]:
    fig.savefig(root / f"mooncake.{ext}", dpi=170)
plt.close(fig)
