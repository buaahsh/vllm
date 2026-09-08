# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot explicit Fast arrival-rate points; run next to load-response.json."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
COLORS = {"standalone": "#146b8c", "pd": "#d55e28"}
LABELS = {"standalone": "Standalone / GPU5", "pd": "1P1D / GPU4+5"}


def save(fig, name):
    target = ROOT / "figures"
    target.mkdir(exist_ok=True)
    fig.savefig(target / (name + ".png"), dpi=180, bbox_inches="tight")
    path = target / (name + ".svg")
    fig.savefig(path, bbox_inches="tight")
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def main():
    data = json.loads((ROOT / "load-response.json").read_text())
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [
        ("achieved_output_tok_s", "Output throughput (tok/s)", 1),
        ("achieved_input_tok_s", "Input throughput (Ktok/s)", 1000),
        ("overhang_from_first_send_s", "Drain overhang (s)", 1),
        ("ttft_p95_ms", "TTFT P95 (s)", 1000),
        ("itl_p95_ms", "ITL P95 (ms)", 1),
        ("e2e_p95_ms", "E2E P95 (s)", 1000),
    ]
    for ax, (key, label, scale) in zip(axes.flat, panels):
        for topology in ["standalone", "pd"]:
            rows = sorted(
                (r for r in data["rows"] if r["topology"] == topology),
                key=lambda r: r["speedup"],
            )
            vals = [r[key] / scale for r in rows]
            ax.plot(
                [r["speedup"] for r in rows],
                vals,
                marker="o",
                color=COLORS[topology],
                label=LABELS[topology],
            )
            for r, val in zip(rows, vals):
                ax.annotate(
                    f"{val:,.2f}" + ("*" if not r["client_pass"] else ""),
                    (r["speedup"], val),
                    xytext=(
                        0,
                        7
                        if (
                            (
                                topology == "pd"
                                and key
                                in [
                                    "achieved_output_tok_s",
                                    "achieved_input_tok_s",
                                    "itl_p95_ms",
                                ]
                            )
                            or (
                                topology == "standalone"
                                and key
                                not in ["achieved_output_tok_s", "achieved_input_tok_s"]
                            )
                        )
                        else -16,
                    ),
                    textcoords="offset points",
                    ha="center",
                    color=COLORS[topology],
                    fontsize=9,
                )
        ax.set(
            title=label,
            xlabel="Arrival-rate multiplier",
            xticks=[1.0, 1.2],
            xlim=(0.97, 1.23),
        )
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(loc="lower left", fontsize=9)
    fig.suptitle(
        "YOCO Fast: load response at 1.0x and 1.2x | same implementation", fontsize=15
    )
    caption = (
        "Mooncake toolagent source 300-900s; ctx<=81920; 3,643 requests "
        "per case; same B200 UUIDs.\n1x: 600s arrivals; 1.2x: 500s. "
        "Unique cache salts. One run each, shared node, no SLO: diagnostic."
    )
    caption += "\n* Client-gate failure. " + ", ".join(
        f"{r['topology']} 1.2x: {r['completed']}/{r['planned']} valid"
        for r in data["rows"]
        if r["speedup"] == 1.2
    )
    fig.text(0.5, 0.01, caption, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    save(fig, "load-response")
    timeline = list(csv.DictReader((ROOT / "arrival-completion.csv").open()))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for row_index, topology in enumerate(["standalone", "pd"]):
        for col_index, speedup in enumerate([1.0, 1.2]):
            ax = axes[row_index, col_index]
            points = [
                r
                for r in timeline
                if r["topology"] == topology and float(r["speedup"]) == speedup
            ]
            times = [float(r["seconds"]) for r in points]
            for key, label, color in [
                ("planned", "Planned arrivals", "#555555"),
                ("sent", "Actually sent", "#729b24"),
                ("completed", "Valid completions", COLORS[topology]),
            ]:
                ax.plot(
                    times,
                    [int(r[key]) for r in points],
                    label=label,
                    color=color,
                    linewidth=1.5,
                )
            ax.axvline(600 / speedup, color="#777777", linestyle=":", linewidth=1)
            ax.set(
                title=f"{LABELS[topology]} | {speedup:.1f}x",
                xlabel="Seconds since first actual send",
                ylabel="Cumulative requests",
            )
            measured = next(
                r
                for r in data["rows"]
                if r["topology"] == topology and r["speedup"] == speedup
            )
            ax.text(
                0.98,
                0.06,
                f"Valid: {measured['completed']}/{measured['planned']}",
                transform=ax.transAxes,
                ha="right",
                fontsize=9,
                color="#a32a21" if not measured["client_pass"] else "#333333",
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("Fast arrival schedule and complete drain", fontsize=15)
    fig.text(
        0.5,
        0.01,
        (
            "Same source requests, separately labelled rates/topologies. "
            "First actual send anchors the planned curve.\nExact trace has"
            "hes, completion gates and overhang reference are in REPORT.m"
            "d and load-response.json."
        ),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    save(fig, "arrival-completion")


if __name__ == "__main__":
    main()
