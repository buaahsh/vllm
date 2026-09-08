# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export cumulative arrivals and completions with explicit rate/topology labels."""

import bisect
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT / "load-response.json").read_text())
    table = []
    for row in data["rows"]:
        root = ROOT.parent / row["source_directory"]
        case = root / "cases" / row["case"]
        manifest = json.loads((case / "manifest.json").read_text())
        records = [
            json.loads(s)
            for s in (case / "artifacts/profile_export.jsonl").read_text().splitlines()
        ]
        trace = [
            json.loads(s)
            for s in (root / "traces" / Path(manifest["trace"]).name)
            .read_text()
            .splitlines()
        ]
        first = min(r["metadata"]["request_start_ns"] for r in records) / 1e9
        starts = sorted(
            r["metadata"]["request_start_ns"] / 1e9 - first for r in records
        )
        ends = sorted(
            r["metadata"]["request_end_ns"] / 1e9 - first
            for r in records
            if not r.get("error") and not r["metadata"].get("was_cancelled")
        )
        planned = [(r["timestamp"] - trace[0]["timestamp"]) / 1000 for r in trace]
        for second in range(math.ceil(max(planned[-1], starts[-1], ends[-1])) + 2):
            table.append(
                dict(
                    topology=row["topology"],
                    speedup=row["speedup"],
                    seconds=second,
                    planned=bisect.bisect_right(planned, second),
                    sent=bisect.bisect_right(starts, second),
                    completed=bisect.bisect_right(ends, second),
                )
            )
    with (ROOT / "arrival-completion.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    buckets = [
        dict(topology=r["topology"], speedup=r["speedup"], **b)
        for r in data["rows"]
        for b in r["source_time_buckets"]
    ]
    with (ROOT / "source-time-latency.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(buckets[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(buckets)


if __name__ == "__main__":
    main()
