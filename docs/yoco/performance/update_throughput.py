# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Update one measured mode/topology; preserve every prior measurement as a file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STORE = ROOT / "throughput"
LABELS = {
    "align": "YOCO Align GEMM",
    "fast": "YOCO Fast",
    "qwen3": "Qwen3-30B-A3B-Instruct-2507",
}
TRACE = "680e526d49545258c0ca5b635d0004a44cce2ce990d533ac32024cefee1f0170"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def argument(command, name):
    i = command.index(name)
    return command[i + 1]


def relative(path):
    return str(path.resolve().relative_to(ROOT))


def read_row(source, case, mode):
    all_rows = json.loads(source.read_text())
    matches = [r for r in all_rows if r["name"] == case]
    if len(matches) != 1:
        raise ValueError("Select exactly one existing case")
    r = matches[0]
    if r["client"].get("aiperf_version") != "0.12.0":
        raise ValueError(
            "Different AIPerf version: create a separate comparison cohort"
        )
    case_dir = source.parent / "cases" / case
    manifest_path = case_dir / "manifest.json"
    m = json.loads(manifest_path.read_text())
    command = m["command"]
    if r["trace_sha256"] != TRACE or m["trace_sha256"] != TRACE:
        raise ValueError("Different trace: create a separate comparison cohort")
    for key, value in {
        "--synthesis-speedup-ratio": "1.0",
        "--concurrency": "512",
        "--workers-max": "32",
        "--record-processors": "1",
        "--request-timeout-seconds": "600",
        "--random-seed": "42",
        "--endpoint-type": "completions",
    }.items():
        if argument(command, key) != value:
            raise ValueError(f"Workload differs at {key}; create a separate cohort")
    for flag in [
        "--fixed-schedule",
        "--fixed-schedule-auto-offset",
        "--streaming",
        "--use-server-token-count",
        "--use-legacy-max-tokens",
    ]:
        if flag not in command:
            raise ValueError(f"Missing comparable client flag {flag}")
    if not json.loads(argument(command, "--extra-inputs")).get("cache_salt"):
        raise ValueError("Cache isolation evidence missing")
    state = m["state"]
    expected_devices = {
        "standalone": "GPU-14379c29-e601-fc6d-b27c-4fd778ab772a",
        "p": "GPU-4e279853-e2ff-de3e-70a8-d3623fd038b3",
        "d": "GPU-14379c29-e601-fc6d-b27c-4fd778ab772a",
    }
    for role, device in state["devices"].items():
        if device["uuid"] != expected_devices[role]:
            raise ValueError(
                "Different physical GPU: create a separate comparison cohort"
            )
    serving = {}
    for role, item in state["processes"].items():
        if role == "proxy":
            continue
        cmd = item["command"]
        expected_model = (
            "/mnt/pvc/lidong1/models/Qwen3-30B-A3B-Instruct-2507"
            if mode == "qwen3"
            else "/mnt/pvc/lidong1/exp/agens/30A3B-180M-L3/0000-28000-hf"
        )
        if argument(cmd, "serve") != expected_model:
            raise ValueError(
                "Different model/checkpoint: create a separate comparison cohort"
            )
        for key, value in {
            "--dtype": "bfloat16",
            "--tensor-parallel-size": "1",
            "--data-parallel-size": "1",
            "--gpu-memory-utilization": "0.85",
            "--max-model-len": "81920",
            "--max-num-seqs": "256",
            "--max-num-batched-tokens": "8192" if role == "d" else "32768",
        }.items():
            if argument(cmd, key) != value:
                raise ValueError(f"Server {role} differs at {key}")
        if mode == "align" and not item.get("align_profile"):
            raise ValueError(
                "This row is the tuned Align candidate; profile is required"
            )
        if mode in ("align", "fast") and "--" + mode not in cmd:
            raise ValueError("Mode does not match actual server command")
        if mode == "qwen3" and ("--align" in cmd or "--fast" in cmd):
            raise ValueError("Qwen row unexpectedly uses a YOCO mode")
        serving[role] = dict(command=cmd, align_profile=item.get("align_profile"))
    c, done = r["client"], r["complete"]
    problems = []
    errors = int(r.get("error_request_count", done.get("error_request_count", 0)))
    if c["completed"] != c["planned"] or errors or not done["token_accounting_pass"]:
        problems.append("请求/长度未全通过")
    if not c["client_pass"]:
        problems.append("客户端负载门槛失败")
    if not done["server_pass"] or not done["drained"]:
        problems.append("服务/排空门槛失败")
    if mode == "qwen3" and r["topology"] == "pd":
        problems.append("P/D log-prob差异待定位")
    if mode == "fast" and r["topology"] == "pd":
        functional_path = source.parent / "functional-comparison.json"
        if functional_path.exists():
            probes = json.loads(functional_path.read_text())["rows"]
            if any(not probe["same_output_tokens"] for probe in probes):
                problems.append("单卡/P-D探针存在输出token差异")
            differences = [
                probe["max_logprob_difference"]
                for probe in probes
                if probe["max_logprob_difference"] is not None
            ]
            if differences and max(differences) > 0:
                problems.append(f"单卡/P-D探针log-prob差{max(differences):.4g}；待定位")
    ended = datetime.fromtimestamp(done["finished"], timezone.utc).isoformat(
        timespec="seconds"
    )
    runtime_manifest = source.parent / "ALIGN_RUNTIME_SHA256.json"
    if not runtime_manifest.exists():
        runtime_manifest = source.parent / "SOURCE_SHA256.json"
    return dict(
        key=f"{mode}/{r['topology']}",
        mode=mode,
        label=LABELS[mode],
        topology=r["topology"],
        case=case,
        measured_at_utc=ended,
        started_at_unix=m["started"],
        source=relative(source),
        source_sha256=sha(source),
        manifest=relative(manifest_path),
        manifest_sha256=sha(manifest_path),
        runtime_manifest=relative(runtime_manifest)
        if runtime_manifest.exists()
        else None,
        runtime_manifest_sha256=sha(runtime_manifest)
        if runtime_manifest.exists()
        else None,
        trace_sha256=TRACE,
        active_gpus=r["active_gpus"],
        allocated_gpus=r["allocated_gpus"],
        devices=r["devices"],
        serving=serving,
        output_tok_s=c["output_token_throughput"],
        input_tok_s=c["input_token_throughput"],
        output_tok_s_per_active_gpu=r["output_tok_s_per_active_gpu"],
        planned=c["planned"],
        completed=c["completed"],
        errors=errors,
        ttft_p95_ms=c["ttft_p95_ms"],
        itl_p95_ms=c["itl_p95_ms"],
        e2e_p95_ms=c["e2e_p95_ms"],
        schedule_lag_p99_ms=c["schedule_lag_p99_ms"],
        concurrency_max=c["effective_concurrency_max"],
        client_pass=c["client_pass"],
        server_pass=done["server_pass"],
        drained=done["drained"],
        token_accounting_pass=done["token_accounting_pass"],
        limitations=problems,
        diagnostic=True,
        source_metrics=r["metrics"],
    )


def render(current):
    fast_reports = sorted(
        {
            str(Path(row["source"]).parent / "REPORT.md")
            for row in current.values()
            if row["mode"] == "fast"
        }
    )
    fast_report_links = "、".join(
        f"[Fast报告{index + 1}]({path})" for index, path in enumerate(fast_reports)
    )
    lines = [
        "# YOCO / Qwen3 持续吞吐对照表",
        "",
        "只重测本次修改涉及的模式；只更新已实际测量的对应行。其他行保留原值、测量时间和证据。",
        "",
        "固定工作负载：Mooncake FAST’25 toolagent，源时间 300–900 秒，1×，3643 请求；"
        "AIPerf 0.12.0、concurrency512、workers32、timeout600。"
        "BF16、TP1/DP1、maxlen81920、maxseq256；单卡/P预算32768、D预算8192。",
        "",
        f"Trace SHA256：`{TRACE}`。每轮独立 cache_salt。"
        "本表使用同一节点上的 GPU5（单卡）、GPU4/5（1P1D）；UUID保存在JSON。",
        "",
        "**全部是共享节点、单次重复、无预设延迟SLO的诊断结果。** "
        "这是固定到达负载下的实测吞吐，不是峰值吞吐；接近负载上限时不能据此排序最大能力。"
        "各行不是同一时段重测，节点上其他工作的干扰可能变化。",
        "",
        "Qwen3与YOCO是不同模型。各模式采用自身执行路径：Align强制的后端/图模式、"
        "Fast按角色的MoE策略、Qwen普通执行路径不保证相同；启动命令和配置证据保存在每行manifest。",
        "",
    ]
    for topology, title in [
        ("standalone", "单卡（GPU5）"),
        ("pd", "1P1D（P GPU4 / D GPU5）"),
    ]:
        lines += [
            "## " + title,
            "",
            "| 模式 | 输出 tok/s | 每推理GPU tok/s | 完成/计划；错误 | "
            "TTFT P95 s | ITL P95 ms | 测量时间 UTC | 状态/证据 |",
            "| --- | ---: | ---: | --- | ---: | ---: | --- | --- |",
        ]
        for mode in ["qwen3", "align", "fast"]:
            r = current.get(f"{mode}/{topology}")
            if r is None:
                lines.append(
                    f"| {LABELS[mode]} | — | — | — | — | — | — | 尚无本组测量 |"
                )
                continue
            status = "；".join(r["limitations"]) or "诊断"
            if r["client_pass"] and r["server_pass"]:
                status = "客户端/服务门槛通过；" + status
            evidence = r["manifest"].removesuffix("/manifest.json") + "/COMPLETE.json"
            measured_at = r["measured_at_utc"].replace("T", " ").replace("+00:00", "")
            lines.append(
                f"| {r['label']} | {r['output_tok_s']:.2f} | "
                f"{r['output_tok_s_per_active_gpu']:.2f} | "
                f"{r['completed']}/{r['planned']}；{r['errors']} | "
                f"{r['ttft_p95_ms'] / 1000:.2f} | "
                f"{r['itl_p95_ms']:.2f} | {measured_at} | "
                f"[{status}]({evidence}) |"
            )
        lines.append("")
    lines += [
        "## 解释与更新规则",
        "",
        "- 未完成的轮次照实显示；其成功请求吞吐/延迟有幸存者偏差，"
        "不能当作完成相同工作量的等价比较。",
        "- Qwen3 1P1D性能门槛通过，但独立探针发现log-prob差异，"
        "数值问题仍未定位；Fast不声明bitwise。",
        "- Align引用的是带16项精确M配置的Align GEMM候选；"
        "其bitwise结论只覆盖原报告已测条件。",
        "- 主表只显示每个模式/拓扑的最近一次实际测量。旧记录永久保存在"
        "[history](throughput/history/)，完整当前字段见"
        "[current.json](throughput/current.json)及[current.csv](throughput/current.csv)。",
        "- 单个更新命令只导入一个case、更新一个模式/拓扑；不启动任何测试。"
        "失败记录不会被隐藏或改成通过。",
        "- 更换trace、速率、上下文、客户端或调度预算时，应新建可比组，不能覆盖本表。",
        "",
        "在本目录中运行（一次导入一个已审计的 case）：",
        "",
        "```bash",
        "uv run --python 3.12 update_throughput.py \\",
        "  --mode fast --source fast-compare-b200-20260907/trace-comparison.json \\",
        "  --case pd-fast-r1-long600s",
        "```",
        "",
        "原始报告：[Align](align-4gpu-1p1d-20260906/REPORT.md)、"
        "[Qwen3](qwen3-compare-b200-20260907/REPORT.md)、" + fast_report_links + "。",
        "",
        "对比图：[吞吐](throughput/figures/throughput.png)、"
        "[延迟](throughput/figures/latency.png)、"
        "[Fast相对变化](throughput/figures/fast-relative.png)、"
        "[到达与完成](throughput/figures/completion.png)。",
        "",
    ]
    (ROOT / "THROUGHPUT.md").write_text("\n".join(lines))
    fields = [
        "mode",
        "topology",
        "case",
        "measured_at_utc",
        "output_tok_s",
        "input_tok_s",
        "output_tok_s_per_active_gpu",
        "planned",
        "completed",
        "errors",
        "ttft_p95_ms",
        "itl_p95_ms",
        "e2e_p95_ms",
        "schedule_lag_p99_ms",
        "concurrency_max",
        "client_pass",
        "server_pass",
        "drained",
        "token_accounting_pass",
        "source",
        "manifest",
    ]
    with (STORE / "current.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(current[key] for key in sorted(current))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=list(LABELS), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    row = read_row(args.source.resolve(), args.case, args.mode)
    STORE.mkdir(exist_ok=True)
    (STORE / "history").mkdir(exist_ok=True)
    path = STORE / "current.json"
    current = json.loads(path.read_text()) if path.exists() else {}
    old = current.get(row["key"])
    if old and row["measured_at_utc"] < old["measured_at_utc"]:
        raise ValueError("Refusing to replace a newer result with an older measurement")
    record = json.dumps(row, ensure_ascii=False, indent=2) + "\n"
    identifier = (
        row["measured_at_utc"].replace(":", "").replace("+", "_")
        + "-"
        + row["key"].replace("/", "-")
    )
    history = STORE / "history" / (identifier + ".json")
    if history.exists() and history.read_text() != record:
        raise ValueError(
            "Historical measurement already exists with different evidence"
        )
    if not history.exists():
        history.write_text(record)
    current[row["key"]] = row
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
    render(current)
    print(
        json.dumps(
            dict(
                updated=row["key"],
                case=row["case"],
                output_tok_s=row["output_tok_s"],
                kept_other_rows=len(current) - 1,
                table=str(ROOT / "THROUGHPUT.md"),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
