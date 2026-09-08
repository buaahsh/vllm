# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Fast load response at two explicit rates from retained raw records."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import regex as re

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / "fast-mooncake-20260908"


def load(path):
    return json.loads(path.read_text())


def prom(path):
    result = {}
    for line in path.read_text().splitlines():
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^\n]*\})?\s+([^\s]+)", line)
        if match:
            key, value = match.groups()
            result[key] = result.get(key, 0) + float(value)
    return result


def collect(root, speedup):
    output = []
    for summary in load(root / "trace-comparison.json"):
        case = root / "cases" / summary["name"]
        manifest = load(case / "manifest.json")
        raw = load(case / "artifacts/profile_export_aiperf.json")
        audit, done = summary["client"], summary["complete"]
        trace = [
            json.loads(line)
            for line in (root / "traces" / Path(manifest["trace"]).name)
            .read_text()
            .splitlines()
        ]
        records = [
            json.loads(line)
            for line in (case / "artifacts/profile_export.jsonl")
            .read_text()
            .splitlines()
        ]
        first = min(r["metadata"]["request_start_ns"] for r in records) / 1e9
        last = max(r["metadata"]["request_end_ns"] for r in records) / 1e9
        span = (trace[-1]["timestamp"] - trace[0]["timestamp"]) / 1000
        assert audit["output_token_throughput"] == raw["output_token_throughput"]["avg"]
        buckets = []
        for start in range(0, 600, 100):
            selected = []
            for record in records:
                index = int(record["metadata"]["conversation_id"].rsplit("_", 1)[-1])
                source_offset = (
                    (trace[index]["timestamp"] - trace[0]["timestamp"]) * speedup / 1000
                )
                if start <= source_offset < start + 100:
                    selected.append(record)
            entry = dict(
                source_start_s=300 + start,
                source_end_s=400 + start,
                requests=len(selected),
                errors=sum(bool(r.get("error")) for r in selected),
            )
            for key, metric in [
                ("ttft", "time_to_first_token"),
                ("itl", "inter_token_latency"),
                ("e2e", "request_latency"),
            ]:
                vals = [
                    r["metrics"][metric]["value"]
                    for r in selected
                    if metric in r["metrics"]
                ]
                for percentile in [50, 95, 99]:
                    entry[f"{key}_p{percentile}_ms"] = (
                        float(np.percentile(vals, percentile)) if vals else None
                    )
            buckets.append(entry)
        means = {}
        for role in manifest["state"]["processes"]:
            before, after = (
                prom(case / f"{role}-before.prom"),
                prom(case / f"{role}-after.prom"),
            )
            means[role] = {}
            for key in after:
                if key.endswith("_sum") and ("seconds" in key or "latency" in key):
                    count_key = key[:-4] + "_count"
                    count = after.get(count_key, 0) - before.get(count_key, 0)
                    if count:
                        means[role][key[:-4]] = dict(
                            mean_seconds=(after[key] - before.get(key, 0)) / count,
                            count=count,
                        )
        row = dict(
            mode="fast",
            topology=summary["topology"],
            speedup=speedup,
            case=summary["name"],
            source_directory=root.name,
            source_commit=load(root / "GIT_SOURCE.json")["commit"],
            trace_sha256=manifest["trace_sha256"],
            cache_salt=manifest["cache_salt"],
            measured_start_utc=datetime.fromtimestamp(first, timezone.utc).isoformat(
                timespec="seconds"
            ),
            measured_end_utc=datetime.fromtimestamp(last, timezone.utc).isoformat(
                timespec="seconds"
            ),
            devices=summary["devices"],
            active_gpus=summary["active_gpus"],
            planned=audit["planned"],
            completed=audit["completed"],
            errors=summary["error_request_count"],
            arrival_span_s=span,
            offered_req_s=len(trace) / span,
            offered_input_tok_s=sum(r["input_length"] for r in trace) / span,
            offered_output_tok_s=sum(r["output_length"] for r in trace) / span,
            achieved_req_s=audit["request_throughput"],
            achieved_input_tok_s=audit["input_token_throughput"],
            achieved_output_tok_s=audit["output_token_throughput"],
            output_tok_s_per_active_gpu=summary["output_tok_s_per_active_gpu"],
            actual_prompt_tokens=raw["total_usage_prompt_tokens"]["avg"],
            actual_output_tokens=raw["total_usage_completion_tokens"]["avg"],
            measured_duration_s=raw["benchmark_duration"]["avg"],
            overhang_from_first_send_s=summary[
                "overhang_from_first_actual_send_seconds"
            ],
            schedule_lag_p99_ms=audit["schedule_lag_p99_ms"],
            schedule_degraded=audit["schedule_degraded"],
            concurrency_max=audit["effective_concurrency_max"],
            client_pass=audit["client_pass"],
            client_gates=audit["client_gates"],
            token_accounting_pass=done["token_accounting_pass"],
            server_pass=done["server_pass"],
            transfer_pass=done["transfer_pass"],
            drained=done["drained"],
            metrics_gates=done["services"],
            queues=summary["queues"],
            transfer=summary["transfer"],
            cache_hit_fraction=summary["cache_hit_fraction"],
            gpu_telemetry=summary["gpu_telemetry_during_requests"],
            server_means=means,
            source_time_buckets=buckets,
            diagnostic=True,
            manifest_sha256=hashlib.sha256(
                (case / "manifest.json").read_bytes()
            ).hexdigest(),
        )
        for key, metric in [
            ("ttft", "time_to_first_token"),
            ("itl", "inter_token_latency"),
            ("e2e", "request_latency"),
        ]:
            for percentile in [50, 95, 99]:
                row[f"{key}_p{percentile}_ms"] = raw[metric][f"p{percentile}"]
        output.append(row)
    return output


def main():
    assert load(ROOT / "TRACE_VERIFICATION.json")["only_timestamps_changed"]
    assert load(ROOT / "SOURCE_EQUIVALENCE.json")["passed"]
    rows = collect(BASELINE, 1.0) + collect(ROOT, 1.2)
    delta = []
    for topology in ["standalone", "pd"]:
        old, new = [r for r in rows if r["topology"] == topology]
        assert {k: v["uuid"] for k, v in old["devices"].items()} == {
            k: v["uuid"] for k, v in new["devices"].items()
        }
        assert old["planned"] == new["planned"] == 3643
        assert old["cache_salt"] != new["cache_salt"]
        entry = dict(topology=topology)
        for key in [
            "achieved_output_tok_s",
            "achieved_input_tok_s",
            "achieved_req_s",
            "ttft_p95_ms",
            "itl_p95_ms",
            "e2e_p95_ms",
            "overhang_from_first_send_s",
            "concurrency_max",
        ]:
            entry[key] = dict(
                rate1=old[key],
                rate1p2=new[key],
                change_percent=(new[key] / old[key] - 1) * 100,
            )
        entry["output_scaling_vs_ideal_percent"] = (
            new["achieved_output_tok_s"] / (old["achieved_output_tok_s"] * 1.2) * 100
        )
        delta.append(entry)
    result = dict(
        schema_version=1,
        rows=rows,
        delta=delta,
        diagnostic=True,
        claim=(
            "Same Fast implementation, same source requests, two explicit "
            "arrival rates. This is a load response comparison, "
            "not a kernel speedup or cross-model ranking."
        ),
        limitations=[
            "One repeat per topology and rate; shared node; no declared latency SLO.",
            (
                "1.2x arrivals span 500 seconds, less than the "
                "600-second qualification minimum."
            ),
            (
                "Achieved throughput includes full completion/drain time; "
                "overhang uses first actual send plus the planned arrival "
                "span as the reference."
            ),
            (
                "P/D server mean timers can overlap; Prefill totals may "
                "include one post-timing drain-control request."
            ),
            (
                "AIPerf ITL is a request-level average; public trace "
                "synthesizes lengths and hash-based prefixes, not model quality."
            ),
        ],
    )
    (ROOT / "load-response.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = [
        key for key, value in rows[0].items() if not isinstance(value, (dict, list))
    ]
    with (ROOT / "load-response.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            dict(
                delta=delta,
                gates=[
                    {
                        k: r[k]
                        for k in [
                            "topology",
                            "speedup",
                            "completed",
                            "errors",
                            "client_pass",
                            "server_pass",
                            "drained",
                        ]
                    }
                    for r in rows
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
