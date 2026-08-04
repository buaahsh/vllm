#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
import shutil
import statistics
import time

import httpx
import regex as re
from transformers import AutoTokenizer

CORPUS = """
Agent observation:
The repository contains Python services, JSON configuration, shell launchers,
spreadsheet inputs, benchmark logs, and unit tests. Inspect the current state,
read the relevant files, execute the requested tool, validate the result, and
continue from the returned output.

Tool output:
{"status":"ok","files":["src/main.py","src/service.py","tests/test_service.py"],
"metrics":{"requests":128,"latency_ms":42.7,"queue":3}}

def transform(records):
    result = []
    for record in records:
        if record.get("enabled"):
            result.append({"id": record["id"], "value": record["value"] * 2})
    return result

$ python -m pytest -q tests/test_service.py
........................................
40 passed in 12.48s
"""


def percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]


def parse_metrics(text):
    metrics = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([^{\s]+)(?:\{([^}]*)\})?\s+([^\s]+)$", line)
        if not match:
            continue
        name, raw_labels, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        labels = tuple(
            sorted(
                re.findall(
                    r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"',
                    raw_labels or "",
                )
            )
        )
        metrics[(name, labels)] = value
    return metrics


def metric_sum(metrics, name, **labels):
    total = 0.0
    for (metric_name, metric_labels), value in metrics.items():
        if metric_name != name:
            continue
        current = dict(metric_labels)
        if all(current.get(key) == expected for key, expected in labels.items()):
            total += value
    return total


def metric_delta(before, after, name, **labels):
    return metric_sum(after, name, **labels) - metric_sum(before, name, **labels)


def cycle_slice(values, offset, length):
    return [values[(offset + index) % len(values)] for index in range(length)]


async def fetch_metrics(client, root_url):
    response = await client.get(f"{root_url}/metrics")
    response.raise_for_status()
    return parse_metrics(response.text)


async def reset_prefix_cache(client, root_url):
    response = await client.post(
        f"{root_url}/reset_prefix_cache?reset_running_requests=true"
    )
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return True


async def stream_completion(client, endpoint, body, headers=None):
    started = time.perf_counter()
    first_token_at = None
    token_ids = []
    finish_reason = None
    async with client.stream("POST", endpoint, json=body, headers=headers) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if "choices" not in chunk:
                raise RuntimeError(f"completion stream returned error payload: {chunk}")
            choice = chunk["choices"][0]
            delta = choice.get("token_ids") or []
            if delta and first_token_at is None:
                first_token_at = time.perf_counter()
            token_ids.extend(delta)
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    ended = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("completion returned no token IDs")
    return token_ids, {
        "latency_seconds": ended - started,
        "ttft_seconds": first_token_at - started,
        "itl_seconds": (ended - first_token_at) / max(len(token_ids) - 1, 1),
        "finish_reason": finish_reason,
    }


async def sample_runtime(stop, root_url, gpu_indices, metric_samples, gpu_samples):
    timeout = httpx.Timeout(30)
    nvidia_smi = shutil.which("nvidia-smi")
    async with httpx.AsyncClient(timeout=timeout) as client:
        while not stop.is_set():
            sampled_at = time.time()
            try:
                metrics = await fetch_metrics(client, root_url)
                metric_samples.append(
                    {
                        "timestamp": sampled_at,
                        "running": metric_sum(metrics, "vllm:num_requests_running"),
                        "waiting": metric_sum(metrics, "vllm:num_requests_waiting"),
                        "kv": metric_sum(metrics, "vllm:kv_cache_usage_perc")
                        / max(
                            sum(
                                1
                                for name, _ in metrics
                                if name == "vllm:kv_cache_usage_perc"
                            ),
                            1,
                        ),
                    }
                )
            except Exception as error:
                metric_samples.append({"timestamp": sampled_at, "error": repr(error)})
            if nvidia_smi is not None:
                process = await asyncio.create_subprocess_exec(
                    nvidia_smi,
                    "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                    "-i",
                    gpu_indices,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await process.communicate()
                if process.returncode == 0:
                    for line in stdout.decode().splitlines():
                        fields = [field.strip() for field in line.split(",")]
                        if len(fields) == 5:
                            gpu_samples.append(
                                {
                                    "timestamp": sampled_at,
                                    "index": fields[0],
                                    "utilization": float(fields[1]),
                                    "memory_utilization": float(fields[2]),
                                    "memory_mib": float(fields[3]),
                                    "power_w": float(fields[4]),
                                }
                            )
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=1)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--trajectories", type=int)
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--prefill-per-turn", type=int, default=1800)
    parser.add_argument("--output-per-turn", type=int, default=200)
    parser.add_argument("--cache-alignment", type=int, default=1056)
    parser.add_argument("--gpu-indices", default="1,7")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument(
        "--warmup-turns",
        type=int,
        default=2,
        help=(
            "Number of trajectory turns used by the excluded warmup. Use the "
            "full trajectory length to compile long-context YOCO kernels before "
            "serving production traffic."
        ),
    )
    parser.add_argument(
        "--warmup-only",
        action="store_true",
        help="Run the excluded warmup, reset prefix cache, and exit.",
    )
    args = parser.parse_args()
    if not 1 <= args.warmup_turns <= args.turns:
        raise ValueError("warmup-turns must be between 1 and turns")
    if args.warmup_only and args.skip_warmup:
        raise ValueError("warmup-only cannot be combined with skip-warmup")

    trajectory_count = args.trajectories or args.concurrency
    if trajectory_count < args.concurrency:
        raise ValueError("trajectories must be at least concurrency")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    root_url = args.base_url.removesuffix("/v1").rstrip("/")
    endpoint = f"{args.base_url.rstrip('/')}/completions"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    corpus_ids = tokenizer.encode(CORPUS, add_special_tokens=False)
    if not corpus_ids:
        raise RuntimeError("tokenizer produced an empty corpus")

    final_tokens = args.turns * (args.prefill_per_turn + args.output_per_turn)
    prefill_schedule = []
    previous_prompt = 0
    for turn in range(1, args.turns + 1):
        ideal_prompt = (
            turn * (args.prefill_per_turn + args.output_per_turn) - args.output_per_turn
        )
        target_prompt = (
            final_tokens - args.output_per_turn
            if turn == args.turns
            else round(ideal_prompt / args.cache_alignment) * args.cache_alignment
        )
        delta = (
            target_prompt
            if turn == 1
            else target_prompt - previous_prompt - args.output_per_turn
        )
        if delta <= 0:
            raise ValueError("cache alignment produced a non-positive prefill delta")
        prefill_schedule.append(delta)
        previous_prompt = target_prompt

    def prompt_delta(trajectory_id, turn):
        length = prefill_schedule[turn]
        marker = tokenizer.encode(
            f"\n[trajectory={trajectory_id} turn={turn} tool_observation]\n",
            add_special_tokens=False,
        )
        needed = length - len(marker)
        if needed < 0:
            return marker[:length]
        offset = (trajectory_id * 1009 + turn * 97) % len(corpus_ids)
        return marker + cycle_slice(corpus_ids, offset, needed)

    timeout = httpx.Timeout(args.timeout, connect=30)
    limits = httpx.Limits(max_connections=1, max_keepalive_connections=1)
    control = httpx.AsyncClient(timeout=timeout)
    try:
        health = await control.get(f"{root_url}/health")
        health.raise_for_status()
        if not args.skip_warmup:

            async def warmup_rank(rank):
                async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
                    history = []
                    salt = hashlib.sha256(
                        f"agent-trace-warmup:{rank}".encode()
                    ).hexdigest()
                    for turn in range(args.warmup_turns):
                        history.extend(prompt_delta(999999 + rank, turn))
                        output, _ = await stream_completion(
                            client,
                            endpoint,
                            {
                                "model": args.model,
                                "prompt": history,
                                "max_tokens": args.output_per_turn,
                                "min_tokens": args.output_per_turn,
                                "ignore_eos": True,
                                "temperature": 0.0,
                                "add_special_tokens": False,
                                "skip_special_tokens": False,
                                "return_token_ids": True,
                                "stream": True,
                                "cache_salt": salt,
                            },
                            {"X-data-parallel-rank": str(rank)},
                        )
                        history.extend(output)

            await asyncio.gather(*(warmup_rank(rank) for rank in range(args.dp_size)))
            await reset_prefix_cache(control, root_url)
            if args.warmup_only:
                print(
                    "completed warmup-only run: "
                    f"{args.warmup_turns} turns on {args.dp_size} DP ranks",
                    flush=True,
                )
                return

        before = await fetch_metrics(control, root_url)
        records = []
        completed_turns = 0
        progress_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(args.concurrency)

        async def run_trajectory(trajectory_id):
            nonlocal completed_turns
            async with semaphore:
                history = []
                started = time.perf_counter()
                salt = hashlib.sha256(
                    f"{args.seed}:{trajectory_id}".encode()
                ).hexdigest()
                async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
                    for turn in range(args.turns):
                        history.extend(prompt_delta(trajectory_id, turn))
                        prompt_tokens = len(history)
                        output, timing = await stream_completion(
                            client,
                            endpoint,
                            {
                                "model": args.model,
                                "prompt": history,
                                "max_tokens": args.output_per_turn,
                                "min_tokens": args.output_per_turn,
                                "ignore_eos": True,
                                "temperature": 0.0,
                                "seed": (args.seed + trajectory_id * args.turns + turn),
                                "add_special_tokens": False,
                                "skip_special_tokens": False,
                                "return_token_ids": True,
                                "stream": True,
                                "request_id": (
                                    f"trace-{args.seed}-{trajectory_id}-{turn}"
                                ),
                                "cache_salt": salt,
                            },
                            {"X-data-parallel-rank": str(trajectory_id % args.dp_size)},
                        )
                        if len(output) != args.output_per_turn:
                            raise RuntimeError(
                                f"trajectory {trajectory_id} turn {turn}: "
                                f"expected {args.output_per_turn} tokens, "
                                f"got {len(output)}"
                            )
                        history.extend(output)
                        records.append(
                            {
                                "trajectory": trajectory_id,
                                "turn": turn + 1,
                                "prompt_tokens": prompt_tokens,
                                "output_tokens": len(output),
                                **timing,
                            }
                        )
                        async with progress_lock:
                            completed_turns += 1
                            if completed_turns % args.turns == 0:
                                print(
                                    "completed "
                                    f"{completed_turns}/"
                                    f"{trajectory_count * args.turns} turns",
                                    flush=True,
                                )
                return {
                    "trajectory": trajectory_id,
                    "seconds": time.perf_counter() - started,
                    "final_tokens": len(history),
                }

        metric_samples = []
        gpu_samples = []
        stop = asyncio.Event()
        sampler = asyncio.create_task(
            sample_runtime(
                stop,
                root_url,
                args.gpu_indices,
                metric_samples,
                gpu_samples,
            )
        )
        started = time.perf_counter()
        try:
            trajectories = await asyncio.gather(
                *(run_trajectory(index) for index in range(trajectory_count))
            )
        finally:
            stop.set()
            await sampler
        wall_seconds = time.perf_counter() - started
        await asyncio.sleep(1)
        after = await fetch_metrics(control, root_url)
    finally:
        await control.aclose()

    logical_prefill = trajectory_count * sum(prefill_schedule)
    logical_generation = trajectory_count * args.turns * args.output_per_turn
    submitted_prompt = sum(record["prompt_tokens"] for record in records)
    computed_prefill = metric_delta(
        before,
        after,
        "vllm:prompt_tokens_by_source_total",
        source="local_compute",
    )
    cached_prefill = metric_delta(
        before,
        after,
        "vllm:prompt_tokens_by_source_total",
        source="local_cache_hit",
    )
    generated = metric_delta(before, after, "vllm:generation_tokens_total")
    prefill_service_seconds = metric_delta(
        before, after, "vllm:request_prefill_time_seconds_sum"
    )
    decode_service_seconds = metric_delta(
        before, after, "vllm:request_decode_time_seconds_sum"
    )
    cache_queries = metric_delta(before, after, "vllm:prefix_cache_queries_total")
    cache_hits = metric_delta(before, after, "vllm:prefix_cache_hits_total")
    queue = [sample for sample in metric_samples if "waiting" in sample]

    gpu_summary = {}
    for index in sorted({sample["index"] for sample in gpu_samples}):
        points = [sample for sample in gpu_samples if sample["index"] == index]
        gpu_summary[index] = {}
        for field in (
            "utilization",
            "memory_utilization",
            "memory_mib",
            "power_w",
        ):
            values = [point[field] for point in points]
            gpu_summary[index][field] = {
                "mean": statistics.fmean(values),
                "p95": percentile(values, 0.95),
                "max": max(values),
            }

    latencies = [record["latency_seconds"] for record in records]
    ttfts = [record["ttft_seconds"] for record in records]
    itls = [record["itl_seconds"] for record in records]
    trajectory_times = [item["seconds"] for item in trajectories]
    summary = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "concurrency": args.concurrency,
        "dp_size": args.dp_size,
        "trajectories": trajectory_count,
        "turns_per_trajectory": args.turns,
        "average_prefill_tokens_per_turn": (sum(prefill_schedule) / args.turns),
        "prefill_schedule": prefill_schedule,
        "output_tokens_per_turn": args.output_per_turn,
        "cache_alignment": args.cache_alignment,
        "final_tokens_per_trajectory": final_tokens,
        "wall_seconds": wall_seconds,
        "trajectories_per_second": trajectory_count / wall_seconds,
        "logical_prefill_tokens": logical_prefill,
        "logical_generation_tokens": logical_generation,
        "submitted_prompt_tokens": submitted_prompt,
        "computed_prefill_tokens": computed_prefill,
        "cached_prefill_tokens": cached_prefill,
        "generated_tokens": generated,
        "computed_prefill_tokens_per_second": (computed_prefill / wall_seconds),
        "generation_tokens_per_second": generated / wall_seconds,
        "prefill_service_seconds": prefill_service_seconds,
        "decode_service_seconds": decode_service_seconds,
        "computed_prefill_tokens_per_service_second": (
            computed_prefill / max(prefill_service_seconds, 1e-9)
        ),
        "generation_tokens_per_service_second": (
            generated / max(decode_service_seconds, 1e-9)
        ),
        "logical_trajectory_tokens_per_second": (logical_prefill + logical_generation)
        / wall_seconds,
        "generation_compute_fraction": generated / max(computed_prefill + generated, 1),
        "prefix_cache_queries": cache_queries,
        "prefix_cache_hits": cache_hits,
        "prefix_cache_hit_rate": cache_hits / max(cache_queries, 1),
        "request_latency_seconds": {
            "mean": statistics.fmean(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies),
        },
        "ttft_seconds": {
            "mean": statistics.fmean(ttfts),
            "p50": percentile(ttfts, 0.50),
            "p95": percentile(ttfts, 0.95),
            "max": max(ttfts),
        },
        "itl_seconds": {
            "mean": statistics.fmean(itls),
            "p50": percentile(itls, 0.50),
            "p95": percentile(itls, 0.95),
            "max": max(itls),
        },
        "trajectory_seconds": {
            "mean": statistics.fmean(trajectory_times),
            "p50": percentile(trajectory_times, 0.50),
            "p95": percentile(trajectory_times, 0.95),
            "max": max(trajectory_times),
        },
        "queue": {
            "running_mean": statistics.fmean(sample["running"] for sample in queue),
            "running_max": max(sample["running"] for sample in queue),
            "waiting_mean": statistics.fmean(sample["waiting"] for sample in queue),
            "waiting_p95": percentile([sample["waiting"] for sample in queue], 0.95),
            "waiting_max": max(sample["waiting"] for sample in queue),
            "kv_mean": statistics.fmean(sample["kv"] for sample in queue),
            "kv_p95": percentile([sample["kv"] for sample in queue], 0.95),
            "kv_max": max(sample["kv"] for sample in queue),
        },
        "gpu": gpu_summary,
        "gpu_telemetry_available": bool(gpu_samples),
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    with open(f"{args.output}.turns.jsonl", "w", encoding="utf-8") as file:
        for record in sorted(
            records, key=lambda item: (item["trajectory"], item["turn"])
        ):
            file.write(json.dumps(record) + "\n")
    with open(f"{args.output}.runtime.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "metrics": metric_samples,
                "gpu": gpu_samples,
                "trajectories": trajectories,
            },
            file,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
