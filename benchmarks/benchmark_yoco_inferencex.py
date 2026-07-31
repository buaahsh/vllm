# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark YOCO with the token lengths used by InferenceX GPT-OSS runs."""

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path

import aiohttp


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def _make_prompt(query_id: int, length: int) -> list[int]:
    unique_prefix = [1000 + query_id * 32 + offset for offset in range(32)]
    return unique_prefix + [100] * (length - len(unique_prefix))


async def _request(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    model: str,
    query_id: int,
    prompt_tokens: int,
    output_tokens: int,
) -> dict:
    payload = {
        "model": model,
        "prompt": _make_prompt(query_id, prompt_tokens),
        "temperature": 0,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "ignore_eos": True,
    }
    async with semaphore:
        started = time.perf_counter()
        try:
            async with session.post(url, json=payload) as response:
                body = await response.text()
                elapsed = time.perf_counter() - started
                if response.status != 200:
                    return {
                        "query_id": query_id,
                        "ok": False,
                        "latency_s": elapsed,
                        "status": response.status,
                        "error": body[:2000],
                    }
                data = json.loads(body)
                usage = data.get("usage") or {}
                return {
                    "query_id": query_id,
                    "ok": True,
                    "latency_s": elapsed,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                }
        except Exception as error:
            return {
                "query_id": query_id,
                "ok": False,
                "latency_s": time.perf_counter() - started,
                "error": repr(error),
            }


async def _run(args: argparse.Namespace) -> dict:
    url = f"{args.base_url.rstrip('/')}/v1/completions"
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        warmup = await _request(
            session,
            semaphore,
            url,
            args.model,
            args.queries,
            args.prompt_tokens,
            16,
        )
        if not warmup["ok"]:
            raise RuntimeError(f"Warmup failed: {warmup}")

        started = time.perf_counter()
        results = await asyncio.gather(
            *[
                _request(
                    session,
                    semaphore,
                    url,
                    args.model,
                    query_id,
                    args.prompt_tokens,
                    args.output_tokens,
                )
                for query_id in range(args.queries)
            ]
        )
        wall_time = time.perf_counter() - started

    successes = [result for result in results if result["ok"]]
    failures = [result for result in results if not result["ok"]]
    latencies = [result["latency_s"] for result in successes]
    total_prompt_tokens = sum(result["prompt_tokens"] or 0 for result in successes)
    total_output_tokens = sum(result["output_tokens"] or 0 for result in successes)
    return {
        "label": args.label,
        "model": args.model,
        "queries": args.queries,
        "concurrency": args.concurrency,
        "prompt_tokens_per_query": args.prompt_tokens,
        "output_tokens_per_query": args.output_tokens,
        "wall_time_s": wall_time,
        "successes": len(successes),
        "failures": len(failures),
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "aggregate_output_tokens_per_s": total_output_tokens / wall_time,
        "latency_s": {
            "mean": statistics.mean(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.50) if latencies else None,
            "p90": _percentile(latencies, 0.90) if latencies else None,
            "p99": _percentile(latencies, 0.99) if latencies else None,
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="yoco")
    parser.add_argument("--label", required=True)
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument(
        "--prompt-tokens", type=int, choices=[1024, 8192], required=True
    )
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.queries < args.concurrency:
        parser.error("--queries must be greater than or equal to --concurrency")

    summary = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "results"},
            indent=2,
        )
    )
    if summary["failures"]:
        raise SystemExit(1)
    if summary["total_prompt_tokens"] != args.queries * args.prompt_tokens:
        raise SystemExit("Prompt token count did not match the requested workload")
    if summary["total_output_tokens"] != args.queries * args.output_tokens:
        raise SystemExit("Output token count did not match the requested workload")


if __name__ == "__main__":
    main()
