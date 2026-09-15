# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize retained Fast FP8 A/B artifacts without rerunning GPU work."""

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import regex as re


def read(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def prom(path):
    result = {}
    for line in path.read_text().splitlines():
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^\n]*\})?\s+([^\s]+)", line)
        if match:
            key, value = match.groups()
            result[key] = result.get(key, 0) + float(value)
    return result


def change(old, new):
    return (new / old - 1) * 100 if old else None


def collect_trace(root, variant):
    case = root / "cases" / f"{variant}-b-f1p2"
    manifest = read(case / "manifest.json")
    raw = read(case / "artifacts/profile_export_aiperf.json")
    audit = read(case / "client-audit.json")
    done = read(case / "COMPLETE.json")
    trace = records(root / "traces" / Path(manifest["trace"]).name)
    requests = records(case / "artifacts/profile_export.jsonl")
    telemetry = records(case / "telemetry.jsonl")
    first = min(r["metadata"]["request_start_ns"] for r in requests) / 1e9
    last = max(r["metadata"]["request_end_ns"] for r in requests) / 1e9
    span = (trace[-1]["timestamp"] - trace[0]["timestamp"]) / 1000
    before, after = [
        prom(case / f"standalone-{when}.prom") for when in ["before", "after"]
    ]
    prefix = {}
    for key in after:
        if "prefix_cache" in key and key.endswith("_total"):
            prefix[key] = after[key] - before.get(key, 0)
    cache_queries = prefix.get("vllm:prefix_cache_queries_total", 0)
    cache_hits = prefix.get("vllm:prefix_cache_hits_total", 0)
    queue_keys = ["vllm:num_requests_running", "vllm:num_requests_waiting"]
    queues = {
        key: max(
            t["services"]["standalone"].get("metrics", {}).get(key, 0)
            for t in telemetry
        )
        for key in queue_keys
    }
    gpu_samples = []
    gpu_index = manifest["state"]["devices"]["standalone"]["index"]
    other_busy = set()
    for sample in telemetry:
        if not first <= sample["timestamp"] <= last:
            continue
        for line in sample.get("gpus", []):
            values = [float(value.strip()) for value in line.split(",")]
            if int(values[0]) == gpu_index:
                gpu_samples.append(values[1:])
            elif values[2] > 0:
                other_busy.add(int(values[0]))
    gpu_summary = (
        {
            key: dict(
                mean=statistics.mean(row[i] for row in gpu_samples),
                minimum=min(row[i] for row in gpu_samples),
                maximum=max(row[i] for row in gpu_samples),
            )
            for i, key in enumerate(
                ["memory_mib", "utilization_percent", "power_w", "temperature_c"]
            )
        }
        if gpu_samples
        else {}
    )
    accounting = read(case / "token-accounting.json")["rows"]
    caches = [read(case / f"kernel-cache-{when}.json") for when in ["before", "after"]]
    new_kernels = {
        key: sorted(set(caches[1][key]) - set(caches[0][key])) for key in caches[0]
    }
    row = dict(
        variant=variant,
        case=case.name,
        measured_start_utc=datetime.fromtimestamp(first, timezone.utc).isoformat(),
        measured_end_utc=datetime.fromtimestamp(last, timezone.utc).isoformat(),
        trace_sha256=manifest["trace_sha256"],
        gpu_uuid=manifest["state"]["devices"]["standalone"]["uuid"],
        cache_salt=manifest["cache_salt"],
        planned=audit["planned"],
        completed=audit["completed"],
        error_records=sum(bool(r.get("error")) for r in requests),
        arrival_span_s=span,
        offered_req_s=len(trace) / span,
        offered_input_tok_s=sum(r["input_length"] for r in trace) / span,
        offered_output_tok_s=sum(r["output_length"] for r in trace) / span,
        achieved_req_s=audit["request_throughput"],
        achieved_input_tok_s=audit["input_token_throughput"],
        achieved_output_tok_s=audit["output_token_throughput"],
        actual_input_tokens=sum(r["actual_input"] or 0 for r in accounting),
        actual_output_tokens=sum(r["actual_output"] or 0 for r in accounting),
        benchmark_duration_s=raw["benchmark_duration"]["avg"],
        overhang_from_first_actual_send_s=last - first - span,
        schedule_lag_p99_ms=audit["schedule_lag_p99_ms"],
        schedule_degraded=audit["schedule_degraded"],
        concurrency_max=audit["effective_concurrency_max"],
        client_pass=audit["client_pass"],
        client_gates=audit["client_gates"],
        server_pass=done["server_pass"],
        token_accounting_pass=done["token_accounting_pass"],
        drained=done["drained"],
        passed=done["passed"],
        metrics_gates=done["services"],
        queue_maxima=queues,
        prefix_cache_deltas=prefix,
        cache_hit_fraction=cache_hits / cache_queries if cache_queries else None,
        manifest_sha256=hashlib.sha256(
            (case / "manifest.json").read_bytes()
        ).hexdigest(),
        diagnostic=True,
        newly_compiled_kernels=new_kernels,
        gpu_telemetry=gpu_summary,
        gpu_telemetry_samples=len(gpu_samples),
        other_busy_gpus=sorted(other_busy),
    )
    for label, metric in [
        ("ttft", "time_to_first_token"),
        ("itl", "inter_token_latency"),
        ("e2e", "request_latency"),
    ]:
        for percentile in [50, 95, 99]:
            row[f"{label}_p{percentile}_ms"] = raw[metric][f"p{percentile}"]
    timeline = []
    completed_at = sorted(
        (r["metadata"]["request_end_ns"] / 1e9 - first, bool(r.get("error")))
        for r in requests
    )
    for index, (time_s, error) in enumerate(completed_at, 1):
        timeline.append(
            dict(
                variant=variant,
                event="completed",
                time_s=time_s,
                completed=index,
                error=error,
            )
        )
    for index, request in enumerate(
        sorted(requests, key=lambda r: r["metadata"]["request_start_ns"]), 1
    ):
        timeline.append(
            dict(
                variant=variant,
                event="sent",
                time_s=request["metadata"]["request_start_ns"] / 1e9 - first,
                completed=index,
                error=False,
            )
        )
    return row, timeline, accounting


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root
    kernel = read(root / "kernel-timing-a.json")
    bytewise = read(root / "kernel-correctness-bytes.json")
    assert {(r["tokens"], r["routing"]) for r in bytewise} == {
        (r["tokens"], r["routing"]) for r in kernel
    }
    assert all(
        r["finite"] and r["bitwise"] and r["comparison"] == "uint8 byte view"
        for r in bytewise
    )
    for row in kernel:
        assert row["finite"] and row["bitwise"], row
        old, new = row["median_us"]["baseline"], row["median_us"]["candidate"]
        row["speedup"] = old / new
        row["latency_change_percent"] = change(old, new)
    probes = {
        v: read(root / "probes" / f"{v}-a-steady" / "results.json")
        for v in ["baseline", "candidate"]
    }
    probe_rows, output_checks = [], []
    for isl, concurrency in [(128, 1), (128, 2), (128, 4), (128, 8), (8192, 1)]:
        row = dict(isl=isl, osl=128, concurrency=concurrency)
        for variant, measured in probes.items():
            selected = [
                r
                for r in measured
                if r["isl"] == isl and r["concurrency"] == concurrency
            ]
            assert len(selected) == 3
            requests = [r for batch in selected for r in batch["requests"]]
            row[variant] = dict(
                output_tok_s=statistics.median(r["output_tok_s"] for r in selected),
                ttft_ms=statistics.median(r["ttft_ms"] for r in requests),
                tpot_ms=statistics.median(r["tpot_ms"] for r in requests),
                e2e_ms=statistics.median(r["e2e_ms"] for r in requests),
                repeats=3,
                request_count=len(requests),
            )
        row["change_percent"] = {
            k: change(row["baseline"][k], row["candidate"][k])
            for k in ["output_tok_s", "ttft_ms", "tpot_ms", "e2e_ms"]
        }
        probe_rows.append(row)
    candidate_by_label = {r["label"]: r for r in probes["candidate"]}
    for batch in probes["baseline"]:
        other = candidate_by_label[batch["label"]]
        for old, new in zip(batch["requests"], other["requests"], strict=True):
            assert old["index"] == new["index"]
            output_checks.append(
                dict(
                    label=batch["label"],
                    index=old["index"],
                    same_text_hash=old["text_sha256"] == new["text_sha256"],
                    same_usage=old["usage"] == new["usage"],
                )
            )
    trace_rows, timeline, token_counts = [], [], []
    for variant in ["baseline", "candidate"]:
        row, rows, counts = collect_trace(root, variant)
        trace_rows.append(row)
        timeline.extend(rows)
        token_counts.append({r["index"]: r for r in counts})
    old, new = trace_rows
    assert old["trace_sha256"] == new["trace_sha256"]
    assert old["gpu_uuid"] == new["gpu_uuid"]
    assert old["cache_salt"] != new["cache_salt"]
    assert old["planned"] == new["planned"] == 3643
    trace_delta = {
        k: change(old[k], new[k])
        for k in [
            "achieved_output_tok_s",
            "achieved_input_tok_s",
            "achieved_req_s",
            "ttft_p95_ms",
            "itl_p95_ms",
            "e2e_p95_ms",
            "overhang_from_first_actual_send_s",
        ]
    }
    token_differences = []
    for index, before in token_counts[0].items():
        after = token_counts[1][index]
        if (before["actual_input"], before["actual_output"]) != (
            after["actual_input"],
            after["actual_output"],
        ):
            token_differences.append(
                dict(index=index, baseline=before, candidate=after)
            )
    result = dict(
        kernel=kernel,
        kernel_bytewise=bytewise,
        probes=probe_rows,
        probe_outputs=output_checks,
        trace=trace_rows,
        trace_change_percent=trace_delta,
        token_differences=token_differences,
        diagnostic=True,
    )
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    frozen = records(root / "traces/toolagent-ctx81920-t300-900s-f1p2.jsonl")
    arrivals = [
        dict(time_s=(r["timestamp"] - frozen[0]["timestamp"]) / 1000, planned=i)
        for i, r in enumerate(frozen, 1)
    ]
    for name, rows in [
        ("trace", trace_rows),
        ("timeline", timeline),
        ("arrivals", arrivals),
    ]:
        fields = [k for k, v in rows[0].items() if not isinstance(v, (dict, list))]
        with (root / f"{name}.csv").open("w", newline="") as out:
            writer = csv.DictWriter(
                out, fields, extrasaction="ignore", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    print(
        json.dumps(
            dict(
                kernel=[
                    dict(tokens=r["tokens"], routing=r["routing"], speedup=r["speedup"])
                    for r in kernel
                ],
                probes=probe_rows,
                trace_change_percent=trace_delta,
                trace=trace_rows,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
