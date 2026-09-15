# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Render the measured FP8 result tables from comparison.json."""

import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--docs", type=Path, required=True)
    args = p.parse_args()
    root, docs = args.root, args.docs
    data = json.loads((root / "comparison.json").read_text())
    old, new = data["trace"]
    probes = data["probes"]
    small = probes[0]
    matched = sum(r["same_text_hash"] for r in data["probe_outputs"])
    count = len(data["probe_outputs"])
    lines = [
        "## 实测结果",
        "",
        "本轮完成 MoE padding 和在线 block-FP8 启动预热两处修改。"
        f"M=1 的完整 MoE 算子在两种路由下加速 1.76–1.84×；"
        f"编译缓存预热后 ISL128/OSL128、并发1的输出吞吐为 "
        f"{small['baseline']['output_tok_s']:.2f} → "
        f"{small['candidate']['output_tok_s']:.2f}"
        f" tok/s"
        f"（{small['change_percent']['output_tok_s']:+.2f}%）。",
        "",
        f"同卡 Mooncake 1.2× 编译缓存预热后回放的输出吞"
        f"吐为 {old['achieved_output_tok_s']:.2f} → "
        f"{new['achieved_output_tok_s']:.2f} tok/s（"
        f"{data['trace_change_percent']['achieved_output_tok_s']:+.2f}"
        f"%）。"
        "必须结合下列完成数、调度门槛和尾延迟解读；客户"
        "端触及并发上限时，实际到达会受限，"
        "吞吐与排空变化属于过载响应，不能作为无损容量或"
        "固定实际到达时刻下的纯 kernel 加速结论。",
        "",
        "### 完整 MoE CUDA graph",
        "",
        "3×ABBA，每个样本100次 replay，取每版本6个样本的中位数。"
        "随机 top-k 与集中 top-8 路由使用相同输入和权重进行版本内对照。",
        "",
        "| M | 随机路由：旧→新 μs | 加速比 | 集中路由：旧→新 μs | 加速比 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    kernel_csv = []
    for m in [1, 2, 4, 8, 16, 32]:
        rows = {r["routing"]: r for r in data["kernel"] if r["tokens"] == m}
        spread, concentrated = rows["spread"], rows["concentrated"]
        lines.append(
            f"| {m} | {spread['median_us']['baseline']:.2f}"
            f" → {spread['median_us']['candidate']:.2f} | {spread['speedup']:.3f}"
            f"× | {concentrated['median_us']['baseline']:.2f}"
            f" → {concentrated['median_us']['candidate']:.2f}"
            f" | {concentrated['speedup']:.3f}× |"
        )
        for r in rows.values():
            kernel_csv.append(
                dict(
                    tokens=m,
                    routing=r["routing"],
                    baseline_us=r["median_us"]["baseline"],
                    candidate_us=r["median_us"]["candidate"],
                    speedup=r["speedup"],
                    baseline_rows=r["baseline_rows"],
                    candidate_rows=r["candidate_rows"],
                    bitwise=r["bitwise"],
                )
            )
    lines += [
        "",
        "M≥16 的 padding 上限未改变，计时约持平。12组输"
        "出均 finite、bitwise equal；这只覆盖所列算子测"
        "试。",
        "",
        "### 逐档预热后的完整服务",
        "",
        "| ISL / OSL | 客户端并发 | 旧 tok/s | 新 tok/"
        "s | 吞吐变化 | 旧→新 TPOT中位数 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    probe_csv = []
    for r in probes:
        b, c = r["baseline"], r["candidate"]
        lines.append(
            f"| {r['isl']} / {r['osl']} | {r['concurrency']}"
            f" | {b['output_tok_s']:.2f} | {c['output_tok_s']:.2f}"
            f" | {r['change_percent']['output_tok_s']:+.2f}"
            f"% | {b['tpot_ms']:.3f} → {c['tpot_ms']:.3f} |"
        )
        probe_csv.append(
            dict(
                isl=r["isl"],
                osl=r["osl"],
                concurrency=r["concurrency"],
                baseline_tok_s=b["output_tok_s"],
                candidate_tok_s=c["output_tok_s"],
                change_percent=r["change_percent"]["output_tok_s"],
                baseline_tpot_ms=b["tpot_ms"],
                candidate_tpot_ms=c["tpot_ms"],
            )
        )
    lines += [
        "",
        f"每档2次预热、3次测量，预热另存。正式探针中 {matched}"
        f"/{count} 条输出文本 SHA256 相同；这不是 token"
        f" ID/logits 的完整比较或质量评测。客户端并发不"
        f"等于物理 MoE batch。",
        "",
        "![MoE及低并发](figures/kernel-probes.png)",
        "",
        "### Mooncake 1.2× 编译缓存预热后回放",
        "",
        "| 指标 | 旧 Fast FP8 | 优化 Fast FP8 |",
        "| --- | ---: | ---: |",
    ]
    metrics = [
        ("完成 / 计划", lambda r: f"{r['completed']} / {r['planned']}"),
        ("失败请求", lambda r: str(r["error_records"])),
        ("输出 tok/s", lambda r: f"{r['achieved_output_tok_s']:.2f}"),
        ("输入 tok/s", lambda r: f"{r['achieved_input_tok_s']:.2f}"),
        ("req/s", lambda r: f"{r['achieved_req_s']:.4f}"),
        ("总完成时间 s", lambda r: f"{r['benchmark_duration_s']:.3f}"),
        ("overhang s", lambda r: f"{r['overhang_from_first_actual_send_s']:.3f}"),
        ("调度迟到 P99 ms", lambda r: f"{r['schedule_lag_p99_ms']:.2f}"),
        ("调度退化", lambda r: str(int(r["schedule_degraded"]))),
        ("最大在途并发", lambda r: str(int(r["concurrency_max"]))),
        (
            "运行 / 等待队列最大值",
            lambda r: (
                f"{int(r['queue_maxima']['vllm:num_requests_running'])}"
                f" / {int(r['queue_maxima']['vllm:num_requests_waiting'])}"
            ),
        ),
        ("prefix cache 命中", lambda r: f"{r['cache_hit_fraction'] * 100:.2f}%"),
        (
            "新增 DeepGEMM / Triton kernel",
            lambda r: (
                f"{len(r['newly_compiled_kernels']['deep_gemm'])}"
                f" / {len(r['newly_compiled_kernels']['triton'])}"
            ),
        ),
        ("客户端门槛", lambda r: "PASS" if r["client_pass"] else "FAIL"),
        (
            "服务 / 计数 / 排空",
            lambda r: "PASS"
            if r["server_pass"] and r["token_accounting_pass"] and r["drained"]
            else "FAIL",
        ),
    ]
    for label, func in metrics:
        lines.append(f"| {label} | {func(old)} | {func(new)} |")
    lines += [
        "",
        "| 延迟 ms | 旧 P50 | 旧 P95 | 旧 P99 | 新 P50 | 新 P95 | 新 P99 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in ["ttft", "itl", "e2e"]:
        values = [r[f"{label}_p{p}_ms"] for r in [old, new] for p in [50, 95, 99]]
        lines.append(
            f"| {label.upper()} | " + " | ".join(f"{v:.2f}" for v in values) + " |"
        )
    lines += [
        "",
        f"实际输入总数 {old['actual_input_tokens']} → {new['actual_input_tokens']}"
        f"；有 {len(data['token_differences'])} 个请求的"
        f"实际 token 计数不同，逐条差异保留。ISL允许历史"
        f"合成器的±2计数差，OSL逐条严格检查。AIPerf ITL"
        f"是逐请求平均 token 间隔的分布；overhang以首个"
        f"实际发送时刻加499.999秒为参照。",
        "",
        "![Mooncake吞吐与延迟](figures/mooncake.png)",
        "",
        "![到达与完成](figures/arrival-completion.png)",
        "",
        "两处修改分别解决小 batch 的 workspace 过大和在"
        "线 FP8 层未进入启动预热的问题。当前仍保留专家"
        "内部的 alignment padding；其 activation/quant"
        "ization 工作可作为下一轮优化对象，需单独验证有"
        "效行掩码和 scale 布局。",
        "",
    ]
    block = "\n".join(lines)
    report = docs / "REPORT.md"
    text = report.read_text()
    start, end = "<!-- MEASURED_RESULTS_BEGIN -->", "<!-- MEASURED_RESULTS_END -->"
    if start in text:
        text = text[: text.index(start)] + text[text.index(end) + len(end) :]
    pos = text.index("## 优化内容")
    text = text[:pos] + start + "\n\n" + block + "\n" + end + "\n\n" + text[pos:]
    report.write_text(text)
    for name, rows in [("kernel", kernel_csv), ("probes", probe_csv)]:
        with (root / f"{name}.csv").open("w", newline="") as out:
            writer = csv.DictWriter(out, list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    table = docs.parent / "FP8.md"
    text = table.read_text()
    if start in text:
        text = text[: text.index(start)] + text[text.index(end) + len(end) :]
    table_lines = [
        "## 当前同卡对照",
        "",
        "测量时间使用各 case 的实际 UTC 时间；基线包含本轮之前的 Fast FP8 实现。",
        "",
        "| 实现 | 实际开始 UTC | 输出 tok/s | TTFT / I"
        "TL / E2E P95 ms | 成功 | 调度 / 并发门槛 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in [old, new]:
        table_lines.append(
            f"| {r['variant']} | {r['measured_start_utc']} "
            f"| {r['achieved_output_tok_s']:.2f} | {r['ttft_p95_ms']:.2f}"
            f" / {r['itl_p95_ms']:.2f} / {r['e2e_p95_ms']:.2f}"
            f" | {r['completed']}/{r['planned']} | "
            f"{'PASS' if r['client_pass'] else 'FAIL（诊断）'}"
            f" |"
        )
    table_lines += [
        "",
        "### 低并发完整服务",
        "",
        "每档2次预热、3次测量，取输出吞吐中位数。下列点与 Mooncake 分开比较。",
        "",
        "| ISL / OSL | 并发 | 旧 tok/s | 新 tok/s | 变化 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in probes:
        table_lines.append(
            f"| {r['isl']} / {r['osl']} | {r['concurrency']}"
            f" | {r['baseline']['output_tok_s']:.2f} | "
            f"{r['candidate']['output_tok_s']:.2f}"
            f" | {r['change_percent']['output_tok_s']:+.2f}"
            f"% |"
        )
    table_lines += [
        "",
        "全部延迟分位数、排空、cache、输入 token 差异及"
        "冷启动失败见完整报告和 [comparison.json](fast"
        "-fp8-20260908/comparison.json)。",
        "",
    ]
    pos = text.index("## 固定条件")
    text = (
        text[:pos]
        + start
        + "\n\n"
        + "\n".join(table_lines)
        + "\n"
        + end
        + "\n\n"
        + text[pos:]
    )
    table.write_text(text)


if __name__ == "__main__":
    main()
