# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Update only measured rows of the frozen Mooncake 1.2x cohort."""

import argparse
import csv
import json
from pathlib import Path

TRACE_SHA256 = "5317e2301656c7d5441bbd63e58b7e8b9d3d445e0118729cd7fde0b8717b960c"
PERF = Path(__file__).resolve().parent
COHORT = {
    "trace_source": "Mooncake FAST25 toolagent_trace",
    "source_sha256": "48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71",
    "trace_sha256": TRACE_SHA256,
    "source_window_ms": [300000, 900000],
    "max_model_len": 81920,
    "timestamp_speedup": 1.2,
    "aiperf_cli_speedup": 1.0,
    "requests": 3643,
    "arrival_span_seconds": 499.99916666666667,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--topology", choices=["standalone", "pd"], required=True)
    args = parser.parse_args()
    rows = [
        r
        for r in json.loads(args.source.read_text())["rows"]
        if r["speedup"] == 1.2 and r["topology"] == args.topology
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["mode"] == "fast" and row["trace_sha256"] == TRACE_SHA256
    assert row["planned"] == COHORT["requests"]
    assert abs(row["arrival_span_s"] - COHORT["arrival_span_seconds"]) < 1e-6
    assert row["cache_salt"]
    expected = (
        {"standalone": "GPU-14379c29-e601-fc6d-b27c-4fd778ab772a"}
        if args.topology == "standalone"
        else {
            "p": "GPU-4e279853-e2ff-de3e-70a8-d3623fd038b3",
            "d": "GPU-14379c29-e601-fc6d-b27c-4fd778ab772a",
        }
    )
    assert {k: v["uuid"] for k, v in row["devices"].items()} == expected
    folder = PERF / "throughput-f1p2"
    (folder / "history").mkdir(parents=True, exist_ok=True)
    path = folder / "current.json"
    state = (
        json.loads(path.read_text())
        if path.exists()
        else dict(schema_version=1, cohort=COHORT, rows={})
    )
    assert state["cohort"] == COHORT
    key = f"fast/{args.topology}"
    state["rows"][key] = row
    stamp = row["measured_start_utc"].replace(":", "").replace("+", "_")
    history = folder / "history" / f"{stamp}-fast-{args.topology}.json"
    entry = json.dumps(dict(cohort=COHORT, row=row), indent=2) + "\n"
    if history.exists():
        assert history.read_text() == entry, "History is immutable"
    else:
        history.write_text(entry)
    path.write_text(json.dumps(state, indent=2) + "\n")
    fields = [k for k, v in row.items() if not isinstance(v, (dict, list))]
    with (folder / "current.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(state["rows"].values())
    lines = [
        "# Mooncake 1.2× 持续表",
        "",
        (
            "同一 toolagent 源时间300–900秒，时间戳除以1.2："
            "约500秒到达、3643请求。与[1×持续表](THROUGHPUT.md)分开维护。"
            "当前只测Fast；Qwen3和Align的1.2×结果尚未测量。"
        ),
        "",
        (
            "单卡GPU5；1P1D为P GPU4、D GPU5，B200、TP1、BF16、上下文81920。"
            "每case独立cache_salt，AIPerf CLI speedup=1.0、并发上限512。"
            "单次、共享节点、未声明SLO，均为diagnostic。"
        ),
        "",
        (
            "| 模式/拓扑 | 测量开始 UTC | 完成/计划 | 输出tok/s"
            " | 输入tok/s | TTFT P95 ms"
            " | ITL P95 ms | E2E P95 ms | client/server/drain |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, item in sorted(state["rows"].items()):
        gates = "/".join(
            "PASS" if item[k] else "FAIL"
            for k in ["client_pass", "server_pass", "drained"]
        )
        lines.append(
            f"| [{key}]({item['source_directory']}/REPORT.md) | "
            f"{item['measured_start_utc']} | "
            f"{item['completed']}/{item['planned']} | "
            f"{item['achieved_output_tok_s']:.2f} | "
            f"{item['achieved_input_tok_s']:.2f} | "
            f"{item['ttft_p95_ms']:.2f} | "
            f"{item['itl_p95_ms']:.2f} | "
            f"{item['e2e_p95_ms']:.2f} | "
            f"{gates} |"
        )
    lines += [
        "",
        (
            "吞吐按完整完成区间计算，包含到达结束后的排空。client/server/drain通过不代表延迟SLO或容量验收通过。"
            "错误、迟发、队列和P99详见每行报告。"
        ),
        "",
        f"Trace SHA256：`{TRACE_SHA256}`。",
        "",
        (
            "数据：[current.json](throughput-f1p2/current.json) · [CSV](thro"
            "ughput-f1p2/current.csv) · [不可变历史](throughput-f1p2/history/)"
            "。"
        ),
        "",
        "只更新实际测量的拓扑：",
        "",
        "```bash",
        (
            "python update_throughput_f1p2.py --source fast-mooncake-f1p2"
            "-20260908/load-response.json --topology standalone"
        ),
        (
            "python update_throughput_f1p2.py --source fast-mooncake-f1p2"
            "-20260908/load-response.json --topology pd"
        ),
        "```",
        "",
        "脚本校验1.2× trace、请求数、到达跨度和物理GPU，拒绝混入1×或修改已有历史记录。",
        "",
    ]
    (PERF / "THROUGHPUT_F1P2.md").write_text("\n".join(lines))
    print(
        json.dumps(
            dict(
                updated=f"fast/{args.topology}",
                trace_sha256=TRACE_SHA256,
                rows=len(state["rows"]),
            )
        )
    )


if __name__ == "__main__":
    main()
