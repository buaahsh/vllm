# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
data = json.loads((root / "comparison.json").read_text())
plt.rcParams.update(
    {
        "font.size": 10,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, component, title in zip(
    axes,
    ["router", "triton_activation_quant", "moe"],
    ["Routing", "Weighted activation + FP8", "Routed expert chain"],
):
    rows = [r.copy() for r in data["microbench"]["rows"] if r["component"] == component]
    if component == "triton_activation_quant":
        rows = [r for r in rows if r["m"] <= 16]
    if component == "router":
        single = {r["name"]: r["median_us"] for r in data["router_single"]}
        rows[0]["speedup"] = single["full_b4w4"] / single["logits_b1w1"]
    ax.plot([r["m"] for r in rows], [r["speedup"] for r in rows], "o-", color="#187d8d")
    ax.axhline(1, color="#888", linestyle="--", linewidth=1)
    ax.set(
        xscale="log",
        xlabel="Input token rows M",
        ylabel="Old / new latency",
        title=title,
    )
    ax.grid(alpha=0.2)
fig.suptitle("YOCO Fast FP8: direct activation quantization and logits Top-8")
fig.text(
    0.5,
    0.02,
    "Same B200 GPU3, interleaved CUDA graphs; repeated hot weights. "
    "M=1 router uses separately verified 1-row / 1-warp config.\n"
    "Routed chain includes dispatch, W13, activation, W2 and merge; "
    "excludes latent/shared experts and attention. Not service throughput.",
    ha="center",
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.14, 1, 0.92))
for extension in ["svg", "png"]:
    fig.savefig(root / f"microbench.{extension}", dpi=160)
