#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RL-style YOCO rollout throughput and llm-train logprob alignment."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import torch

BASE_PROMPTS = [
    (
        "physics",
        "Explain why the daytime sky appears blue and sunsets often appear red.",
    ),
    (
        "probability",
        "A fair coin is flipped until two consecutive heads appear. Derive the "
        "expected number of flips.",
    ),
    (
        "python_debug",
        "A Python service gradually consumes more memory while processing a "
        "large queue. Describe a systematic debugging plan and likely causes.",
    ),
    (
        "distributed_systems",
        "Compare leader-based replication with leaderless replication for a "
        "globally distributed key-value store.",
    ),
    (
        "model_serving",
        "Explain how continuous batching, KV cache management, and CUDA Graphs "
        "affect large-language-model serving throughput and latency.",
    ),
    (
        "sql",
        "Design a SQL query that returns each customer's latest successful "
        "order and the customer's rolling 30-day spend.",
    ),
    (
        "algorithm",
        "Given a directed graph, describe an efficient algorithm for finding "
        "all vertices that belong to at least one cycle.",
    ),
    (
        "data_analysis",
        "A metric improved by 8 percent overall but declined in every customer "
        "segment. Explain how this can happen and how to investigate it.",
    ),
    (
        "writing",
        "Write the opening of a science-fiction story in which a lunar "
        "research station receives a message sent from Earth in the future.",
    ),
    (
        "translation",
        "Translate the following sentence into natural Chinese and explain two "
        "reasonable translation choices: 'The design is simple, but not naive.'",
    ),
    (
        "chinese_history",
        "请比较唐代和宋代城市经济的发展特点，并说明造成差异的主要因素。",
    ),
    (
        "chinese_math",
        "有三个盒子，标签分别写着苹果、橘子和混合，但所有标签都贴错了。"
        "说明如何只取出一个水果就确定三个盒子的内容。",
    ),
    (
        "chinese_code",
        "请设计一个支持并发读写、过期淘汰和容量限制的内存缓存，并说明关键"
        "数据结构与竞态条件。",
    ),
    (
        "critique",
        "Critique the claim that increasing model context length always "
        "improves answer quality.",
    ),
    (
        "planning",
        "Create an engineering plan for migrating a high-traffic API from a "
        "monolith to independently deployable services without downtime.",
    ),
    (
        "rl_training",
        "Explain why rollout logprobs and training-time teacher-forced "
        "logprobs can differ even when they use the same model checkpoint.",
    ),
]

PROMPT_VARIANTS = [
    "Give a direct answer with the essential reasoning.",
    "Reason step by step and include a concrete example.",
    "Discuss assumptions, edge cases, and possible failure modes.",
    "Present the answer as a concise technical note with recommendations.",
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        json.dump(payload, writer, ensure_ascii=False, indent=2)
        writer.write("\n")


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as reader:
        return json.load(reader)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def create_prompts(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    records = []
    for variant_index, variant in enumerate(PROMPT_VARIANTS):
        for base_name, base_prompt in BASE_PROMPTS:
            content = f"{base_prompt}\n\n{variant}"
            messages = [{"role": "user", "content": content}]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_token_ids = tokenizer.encode(
                prompt_text,
                add_special_tokens=False,
            )
            records.append(
                {
                    "name": f"{base_name}_v{variant_index}",
                    "messages": messages,
                    "prompt_text": prompt_text,
                    "prompt_token_ids": prompt_token_ids,
                    "prompt_tokens": len(prompt_token_ids),
                }
            )
    if args.num_prompts > len(records):
        raise ValueError(
            f"Requested {args.num_prompts} prompts, only {len(records)} are defined"
        )
    records = records[: args.num_prompts]
    lengths = [record["prompt_tokens"] for record in records]
    payload = {
        "model": args.model,
        "num_prompts": len(records),
        "prompt_tokens": {
            "total": sum(lengths),
            "min": min(lengths),
            "mean": sum(lengths) / len(lengths),
            "max": max(lengths),
        },
        "records": records,
    }
    _write_json(args.out, payload)
    print(
        f"[rl-rollout] saved {len(records)} prompts to {args.out}; "
        f"token lengths min/mean/max="
        f"{min(lengths)}/{sum(lengths) / len(lengths):.1f}/{max(lengths)}",
        flush=True,
    )


async def _post_rollout_batch(
    session: Any,
    args: argparse.Namespace,
    url: str,
    url_index: int,
    indexed_records: list[tuple[int, dict[str, Any]]],
) -> tuple[list[tuple[int, dict[str, Any]]], dict[str, Any]]:
    records = [record for _, record in indexed_records]
    request_payload = {
        "model": args.served_model_name,
        "prompt": [record["prompt_token_ids"] for record in records],
        "add_special_tokens": False,
        "max_tokens": args.max_tokens,
        "min_tokens": args.min_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "seed": args.seed,
        "logprobs": 1,
        "stream": False,
        "skip_special_tokens": False,
        "return_tokens_as_token_ids": True,
        "return_token_ids": True,
    }
    async with session.post(url, json=request_payload) as response:
        response_text = await response.text()
        if response.status != 200:
            raise RuntimeError(
                f"vLLM request to {url} failed with HTTP {response.status}: "
                f"{response_text[:2000]}"
            )
    response_payload = json.loads(response_text)
    choices = sorted(response_payload["choices"], key=lambda choice: choice["index"])
    if len(choices) != len(records):
        raise RuntimeError(
            f"Expected {len(records)} choices from {url}, received {len(choices)}"
        )

    rollout_records = []
    for (record_index, record), choice in zip(indexed_records, choices):
        token_ids = choice.get("token_ids")
        prompt_token_ids = choice.get("prompt_token_ids")
        logprobs = choice.get("logprobs") or {}
        token_logprobs = logprobs.get("token_logprobs")
        if token_ids is None or token_logprobs is None:
            raise RuntimeError(
                f"Response for {record['name']} omitted token IDs or logprobs"
            )
        if len(token_ids) != len(token_logprobs):
            raise RuntimeError(
                f"Token/logprob length mismatch for {record['name']}: "
                f"{len(token_ids)} != {len(token_logprobs)}"
            )
        if any(value is None or not math.isfinite(value) for value in token_logprobs):
            raise RuntimeError(f"Non-finite sampled logprob for {record['name']}")
        if (
            prompt_token_ids is not None
            and prompt_token_ids != record["prompt_token_ids"]
        ):
            raise RuntimeError(f"Server changed prompt token IDs for {record['name']}")
        rollout_records.append(
            (
                record_index,
                {
                    **record,
                    "server_url_index": url_index,
                    "output_text": choice["text"],
                    "output_token_ids": token_ids,
                    "output_token_logprobs": token_logprobs,
                    "output_tokens": len(token_ids),
                    "finish_reason": choice.get("finish_reason"),
                    "stop_reason": choice.get("stop_reason"),
                },
            )
        )
    return rollout_records, response_payload


async def _request_rollout(args: argparse.Namespace, prompts: dict[str, Any]) -> None:
    import aiohttp

    if args.temperature != 1.0 or args.top_p != 1.0:
        raise ValueError(
            "RL KL estimation requires temperature=1 and top_p=1 because vLLM "
            "returns raw pre-sampling logprobs"
        )
    if args.min_tokens != 0:
        raise ValueError(
            "RL KL estimation requires min_tokens=0 because vLLM returns raw "
            "logprobs before the minimum-length stop-token mask"
        )
    records = prompts["records"]
    urls = args.url or ["http://127.0.0.1:8001/v1/completions"]
    indexed_batches = [
        list(enumerate(records))[url_index :: len(urls)]
        for url_index in range(len(urls))
    ]
    if any(not batch for batch in indexed_batches):
        raise ValueError(f"Received {len(urls)} URLs for only {len(records)} prompts")

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        batch_results = await asyncio.gather(
            *(
                _post_rollout_batch(
                    session,
                    args,
                    url,
                    url_index,
                    indexed_records,
                )
                for url_index, (url, indexed_records) in enumerate(
                    zip(urls, indexed_batches)
                )
            )
        )
    elapsed = time.perf_counter() - start
    indexed_rollout_records = [
        record for batch_records, _ in batch_results for record in batch_records
    ]
    indexed_rollout_records.sort(key=lambda item: item[0])
    rollout_records = [record for _, record in indexed_rollout_records]
    response_payloads = [response for _, response in batch_results]

    output_lengths = [record["output_tokens"] for record in rollout_records]
    output_tokens = sum(output_lengths)
    prompt_tokens = sum(record["prompt_tokens"] for record in rollout_records)
    payload = {
        "setting": args.setting,
        "server": {
            "url": urls[0] if len(urls) == 1 else urls,
            "served_model_name": args.served_model_name,
            "requests_per_url": [len(batch) for batch in indexed_batches],
        },
        "sampling": {
            "max_tokens": args.max_tokens,
            "min_tokens": args.min_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "raw_sampled_token_logprobs": True,
        },
        "performance": {
            "wall_seconds": elapsed,
            "num_requests": len(rollout_records),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "aggregate_output_tokens_per_second": output_tokens / elapsed,
            "aggregate_total_tokens_per_second": (prompt_tokens + output_tokens)
            / elapsed,
            "requests_per_second": len(rollout_records) / elapsed,
        },
        "output_length": {
            "min": min(output_lengths),
            "mean": sum(output_lengths) / len(output_lengths),
            "p50": _percentile(output_lengths, 0.50),
            "p95": _percentile(output_lengths, 0.95),
            "max": max(output_lengths),
        },
        "usage": (
            response_payloads[0].get("usage")
            if len(response_payloads) == 1
            else [response.get("usage") for response in response_payloads]
        ),
        "records": rollout_records,
    }
    _write_json(args.out, payload)
    print(
        f"[rl-rollout] {args.setting}: {output_tokens} output tokens in "
        f"{elapsed:.3f}s = {output_tokens / elapsed:.2f} tok/s; "
        f"output length min/mean/max="
        f"{min(output_lengths)}/{sum(output_lengths) / len(output_lengths):.1f}/"
        f"{max(output_lengths)}",
        flush=True,
    )


def run_rollout(args: argparse.Namespace) -> None:
    prompts = _read_json(args.prompts)
    asyncio.run(_request_rollout(args, prompts))


def _metrics_url(completions_url: str) -> str:
    parsed = urlsplit(completions_url)
    return f"{parsed.scheme}://{parsed.netloc}/metrics"


def _prometheus_values(payload: str, metric_name: str) -> list[float]:
    values = []
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        name_and_labels, _, raw_value = line.partition(" ")
        if not raw_value:
            continue
        name = name_and_labels.split("{", 1)[0]
        if name != metric_name:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return values


async def _sample_server_metrics(
    session: Any,
    urls: list[str],
    stop_event: asyncio.Event,
    interval: float,
) -> tuple[list[dict[str, float]], int]:
    metric_urls = [_metrics_url(url) for url in urls]
    samples = []
    errors = 0
    start = time.perf_counter()
    while not stop_event.is_set():
        waiting = 0.0
        running = 0.0
        kv_values = []
        for metric_url in metric_urls:
            try:
                async with session.get(metric_url) as response:
                    payload = await response.text()
                    if response.status != 200:
                        errors += 1
                        continue
            except Exception:
                errors += 1
                continue
            waiting += sum(_prometheus_values(payload, "vllm:num_requests_waiting"))
            running += sum(_prometheus_values(payload, "vllm:num_requests_running"))
            kv_values.extend(_prometheus_values(payload, "vllm:kv_cache_usage_perc"))
        samples.append(
            {
                "seconds": time.perf_counter() - start,
                "waiting": waiting,
                "running": running,
                "kv_cache_mean": (
                    sum(kv_values) / len(kv_values) if kv_values else 0.0
                ),
                "kv_cache_max": max(kv_values, default=0.0),
            }
        )
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
    return samples, errors


async def _request_load_item(
    session: Any,
    args: argparse.Namespace,
    url: str,
    url_index: int,
    request_index: int,
    record: dict[str, Any],
    scheduled_at: float,
) -> dict[str, Any]:
    request_payload = {
        "model": args.served_model_name,
        "prompt": record["prompt_token_ids"],
        "add_special_tokens": False,
        "max_tokens": args.max_tokens,
        "min_tokens": args.min_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "seed": args.seed + request_index,
        "logprobs": 1,
        "stream": False,
        "skip_special_tokens": False,
        "return_tokens_as_token_ids": True,
        "return_token_ids": True,
    }
    dispatched_at = time.perf_counter()
    try:
        async with session.post(url, json=request_payload) as response:
            response_text = await response.text()
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {response_text[:1000]}")
        response_payload = json.loads(response_text)
        choice = response_payload["choices"][0]
        token_ids = choice.get("token_ids")
        token_logprobs = (choice.get("logprobs") or {}).get("token_logprobs")
        if token_ids is None or token_logprobs is None:
            raise RuntimeError("Response omitted token IDs or logprobs")
        if len(token_ids) != len(token_logprobs):
            raise RuntimeError(
                f"Token/logprob length mismatch: "
                f"{len(token_ids)} != {len(token_logprobs)}"
            )
        if any(value is None or not math.isfinite(value) for value in token_logprobs):
            raise RuntimeError("Response contained a non-finite sampled logprob")
        completed_at = time.perf_counter()
        return {
            **record,
            "request_index": request_index,
            "server_url_index": url_index,
            "scheduled_at": scheduled_at,
            "dispatched_at": dispatched_at,
            "completed_at": completed_at,
            "dispatch_lag_seconds": dispatched_at - scheduled_at,
            "latency_seconds": completed_at - dispatched_at,
            "output_text": choice["text"],
            "output_token_ids": token_ids,
            "output_token_logprobs": token_logprobs,
            "output_tokens": len(token_ids),
            "finish_reason": choice.get("finish_reason"),
            "stop_reason": choice.get("stop_reason"),
            "success": True,
            "error": None,
        }
    except Exception as error:
        completed_at = time.perf_counter()
        return {
            **record,
            "request_index": request_index,
            "server_url_index": url_index,
            "scheduled_at": scheduled_at,
            "dispatched_at": dispatched_at,
            "completed_at": completed_at,
            "dispatch_lag_seconds": dispatched_at - scheduled_at,
            "latency_seconds": completed_at - dispatched_at,
            "output_token_ids": [],
            "output_token_logprobs": [],
            "output_tokens": 0,
            "success": False,
            "error": f"{type(error).__name__}: {error}",
        }


def _max_in_flight(records: list[dict[str, Any]]) -> int:
    events = []
    for record in records:
        events.append((record["dispatched_at"], 1))
        events.append((record["completed_at"], -1))
    current = 0
    maximum = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        current += delta
        maximum = max(maximum, current)
    return maximum


async def _request_continuous_load(
    args: argparse.Namespace,
    prompts: dict[str, Any],
) -> None:
    import aiohttp

    if args.temperature != 1.0 or args.top_p != 1.0:
        raise ValueError(
            "RL KL estimation requires temperature=1 and top_p=1 because vLLM "
            "returns raw pre-sampling logprobs"
        )
    urls = args.url or ["http://127.0.0.1:8001/v1/completions"]
    prompt_records = prompts["records"]
    if not prompt_records:
        raise ValueError("Prompt dataset is empty")
    if args.request_rate <= 0 and not math.isinf(args.request_rate):
        raise ValueError("request_rate must be positive or inf")
    if args.burstiness <= 0:
        raise ValueError("burstiness must be positive")

    rng = random.Random(args.seed)
    arrival_offsets = [0.0]
    for _ in range(1, args.num_requests):
        if math.isinf(args.request_rate):
            interval = 0.0
        else:
            interval = rng.gammavariate(
                args.burstiness,
                1.0 / (args.request_rate * args.burstiness),
            )
        arrival_offsets.append(arrival_offsets[-1] + interval)

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    stop_metrics = asyncio.Event()
    tasks = []
    start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        metrics_task = asyncio.create_task(
            _sample_server_metrics(
                session,
                urls,
                stop_metrics,
                args.metrics_interval,
            )
        )
        for request_index, arrival_offset in enumerate(arrival_offsets):
            scheduled_at = start + arrival_offset
            delay = scheduled_at - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            url_index = request_index % len(urls)
            prompt_record = prompt_records[request_index % len(prompt_records)]
            tasks.append(
                asyncio.create_task(
                    _request_load_item(
                        session,
                        args,
                        urls[url_index],
                        url_index,
                        request_index,
                        prompt_record,
                        scheduled_at,
                    )
                )
            )
        records = await asyncio.gather(*tasks)
        stop_metrics.set()
        metric_samples, metric_errors = await metrics_task
    completed = time.perf_counter()

    successful_records = [record for record in records if record["success"]]
    failed_records = [record for record in records if not record["success"]]
    if not successful_records:
        raise RuntimeError(
            f"All {len(records)} load-test requests failed: "
            f"{failed_records[0]['error']}"
        )
    latencies = [record["latency_seconds"] for record in successful_records]
    dispatch_lags = [record["dispatch_lag_seconds"] for record in successful_records]
    output_lengths = [record["output_tokens"] for record in successful_records]
    output_tokens = sum(output_lengths)
    prompt_tokens = sum(record["prompt_tokens"] for record in successful_records)
    wall_seconds = completed - start
    waiting_values = [sample["waiting"] for sample in metric_samples]
    running_values = [sample["running"] for sample in metric_samples]
    kv_mean_values = [sample["kv_cache_mean"] for sample in metric_samples]
    kv_max_values = [sample["kv_cache_max"] for sample in metric_samples]
    payload = {
        "setting": args.setting,
        "server": {
            "url": urls[0] if len(urls) == 1 else urls,
            "served_model_name": args.served_model_name,
            "requests_per_url": [
                sum(
                    record["server_url_index"] == url_index
                    for record in successful_records
                )
                for url_index in range(len(urls))
            ],
        },
        "sampling": {
            "max_tokens": args.max_tokens,
            "min_tokens": args.min_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "raw_sampled_token_logprobs": True,
        },
        "load": {
            "request_rate": args.request_rate,
            "burstiness": args.burstiness,
            "arrival_span_seconds": arrival_offsets[-1],
            "metrics_interval_seconds": args.metrics_interval,
        },
        "performance": {
            "wall_seconds": wall_seconds,
            "num_requests": len(records),
            "successful_requests": len(successful_records),
            "failed_requests": len(failed_records),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "aggregate_output_tokens_per_second": output_tokens / wall_seconds,
            "aggregate_total_tokens_per_second": (prompt_tokens + output_tokens)
            / wall_seconds,
            "requests_per_second": len(successful_records) / wall_seconds,
            "max_in_flight": _max_in_flight(records),
        },
        "latency": {
            "mean": sum(latencies) / len(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "max": max(latencies),
            "dispatch_lag_p99": _percentile(dispatch_lags, 0.99),
        },
        "server_metrics": {
            "samples": len(metric_samples),
            "errors": metric_errors,
            "waiting_mean": (
                sum(waiting_values) / len(waiting_values) if waiting_values else 0.0
            ),
            "waiting_p95": (
                _percentile(waiting_values, 0.95) if waiting_values else 0.0
            ),
            "waiting_max": max(waiting_values, default=0.0),
            "running_mean": (
                sum(running_values) / len(running_values) if running_values else 0.0
            ),
            "running_max": max(running_values, default=0.0),
            "kv_cache_mean": (
                sum(kv_mean_values) / len(kv_mean_values) if kv_mean_values else 0.0
            ),
            "kv_cache_max": max(kv_max_values, default=0.0),
        },
        "output_length": {
            "min": min(output_lengths),
            "mean": sum(output_lengths) / len(output_lengths),
            "p50": _percentile(output_lengths, 0.50),
            "p95": _percentile(output_lengths, 0.95),
            "max": max(output_lengths),
        },
        "records": successful_records,
        "failures": [
            {
                "request_index": record["request_index"],
                "name": record["name"],
                "error": record["error"],
            }
            for record in failed_records
        ],
    }
    _write_json(args.out, payload)
    print(
        f"[rl-load] {args.setting}: requests={len(successful_records)}/"
        f"{len(records)} wall={wall_seconds:.3f}s "
        f"output={output_tokens / wall_seconds:.2f} tok/s "
        f"latency p50/p95={_percentile(latencies, 0.50):.3f}/"
        f"{_percentile(latencies, 0.95):.3f}s "
        f"queue max={max(waiting_values, default=0.0):.0f}",
        flush=True,
    )


def run_continuous_load(args: argparse.Namespace) -> None:
    prompts = _read_json(args.prompts)
    asyncio.run(_request_continuous_load(args, prompts))


def _install_native_fa4() -> list[int]:
    import importlib

    import arch.attention as native_attention
    from flash_attn.cute import flash_attn_varlen_func as native_cute_varlen

    call_count = [0]
    for module_name in (
        "nnscaler.customized_ops.ring_attention.sliding_window_attn",
        "nnscaler.customized_ops.ring_attention.ring_attn_varlen",
        "nnscaler.customized_ops.ring_attention.zigzag_allgather_attn_varlen",
    ):
        module = importlib.import_module(module_name)
        original = module.flash_attn_cute_varlen_func

        def counted_ring_cute(*call_args, _original=original, **call_kwargs):
            call_count[0] += 1
            return _original(*call_args, **call_kwargs)

        module.flash_attn_cute_varlen_func = counted_ring_cute

    def counted_native_cute(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
    ):
        if dropout_p:
            raise ValueError("Native FA4 scoring requires dropout_p=0")
        if alibi_slopes is not None:
            raise ValueError("Native FA4 scoring does not support ALiBi")
        call_count[0] += 1
        result = native_cute_varlen(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=tuple(None if size == -1 else size for size in window_size),
            deterministic=deterministic,
            return_lse=return_attn_probs,
        )
        if return_attn_probs:
            output, lse = result
            return output, lse, None
        return result[0] if isinstance(result, tuple) else result

    native_attention.flash_attn_varlen_func = counted_native_cute
    return call_count


def _init_native_distributed() -> tuple[int, int]:
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.accelerator.set_device_index(local_rank)
    return local_rank, world_size


def _load_native_model(args: argparse.Namespace):
    from torch.distributed.device_mesh import init_device_mesh

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from logprob_kl import (
        _device_mapping,
        _normalize_native_checkpoint_modelargs,
        _normalize_native_checkpoint_state,
    )

    llm_dir = Path(args.llm_train_dir).resolve() / "llm"
    sys.path.insert(0, str(llm_dir))
    from arch.model import Model, ModelArgs

    _, world_size = _init_native_distributed()
    if world_size != 1:
        raise ValueError("Native rollout scoring currently requires one GPU")
    init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["dp"])

    if args.quant_mode == "mxfp8" and args.use_torch_fp8_quant:
        import arch.linear as linear
        import kernel.moe_ffn as moe_ffn
        import kernel.quant as quant

        linear.per_token_cast_to_fp8 = quant._per_token_cast_to_fp8_torch
        moe_ffn.per_token_cast_to_fp8 = quant._per_token_cast_to_fp8_torch
        print("[rl-native] using torch FP8 activation quantization", flush=True)

    metadata_path = Path(args.native_checkpoint) / "metadata.json"
    with metadata_path.open(encoding="utf-8") as reader:
        metadata = json.load(reader)
    checkpoint_modelargs = _normalize_native_checkpoint_modelargs(metadata["modelargs"])
    modelargs = ModelArgs()
    for key, value in checkpoint_modelargs.items():
        setattr(modelargs, key, value)
    modelargs.quant_mode = args.quant_mode
    if args.quant_block_size is not None:
        modelargs.quant_block_size = args.quant_block_size
    modelargs.use_cute = True
    modelargs.moe_fwd_bwd_overlap = False
    modelargs.validate()

    device = torch.device("cuda")
    default_device = torch.get_default_device()
    default_dtype = torch.get_default_dtype()
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)
    model = Model(modelargs)
    torch.set_default_device(default_device)
    torch.set_default_dtype(default_dtype)
    model.eval()

    state = torch.load(
        Path(args.native_checkpoint) / "model_state_rank_0.pth",
        map_location=_device_mapping(-1),
        mmap=True,
    )
    state = _normalize_native_checkpoint_state(state, checkpoint_modelargs)
    model.load_state_dict(state)
    fa4_call_count = _install_native_fa4()
    print(
        f"[rl-native] model loaded quant_mode={modelargs.quant_mode} "
        f"batch_sizes={args.batch_size} use_cute={modelargs.use_cute}",
        flush=True,
    )
    return model, modelargs, fa4_call_count


@torch.no_grad()
def _score_rollout(
    model: Any,
    modelargs: Any,
    rollout: dict[str, Any],
    batch_size: int,
    max_model_len: int,
    fa4_call_count: list[int],
) -> dict[str, Any]:
    records = rollout["records"]
    server_urls = rollout.get("server", {}).get("url")
    if isinstance(server_urls, list):
        record_groups = [[] for _ in server_urls]
        for record_index, record in enumerate(records):
            url_index = record.get("server_url_index", record_index % len(server_urls))
            if not 0 <= url_index < len(server_urls):
                raise ValueError(
                    f"Invalid server_url_index={url_index} for {record['name']}"
                )
            record_groups[url_index].append(record)
    else:
        record_groups = [records]

    scored_records = []
    total_output_tokens = 0
    fa4_start = fa4_call_count[0]
    torch.cuda.synchronize()
    start = time.perf_counter()

    for group_index, record_group in enumerate(record_groups):
        for batch_start in range(0, len(record_group), batch_size):
            batch = record_group[batch_start : batch_start + batch_size]
            _score_native_batch(
                model,
                modelargs,
                batch,
                max_model_len,
                group_index,
                batch_start,
                scored_records,
            )
            total_output_tokens += sum(
                len(record["output_token_ids"]) for record in batch
            )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    fa4_calls = fa4_call_count[0] - fa4_start
    if fa4_calls <= 0:
        raise RuntimeError("Native rollout scoring did not execute FA4")
    return {
        "rollout_setting": rollout["setting"],
        "native": {
            "quant_mode": modelargs.quant_mode,
            "quant_block_size": modelargs.quant_block_size,
            "batch_size": batch_size,
            "server_groups": len(record_groups),
            "use_cute": True,
            "fa4_calls": fa4_calls,
            "scoring_seconds": elapsed,
            "output_tokens": total_output_tokens,
            "output_tokens_per_second": total_output_tokens / elapsed,
        },
        "records": scored_records,
    }


def _score_native_batch(
    model: Any,
    modelargs: Any,
    batch: list[dict[str, Any]],
    max_model_len: int,
    group_index: int,
    batch_start: int,
    scored_records: list[dict[str, Any]],
) -> None:
    device = torch.device("cuda")
    token_lists = [
        record["prompt_token_ids"] + record["output_token_ids"] for record in batch
    ]
    if any(len(token_ids) > max_model_len for token_ids in token_lists):
        raise ValueError(
            f"Rollout exceeds max_model_len={max_model_len} in server group "
            f"{group_index}, batch starting at record {batch_start}"
        )
    if any(not record["output_token_ids"] for record in batch):
        raise ValueError("Native scoring requires at least one output token")
    if any(max(token_ids) >= modelargs.vocab_size for token_ids in token_lists):
        raise ValueError("Rollout contains a token outside the native vocabulary")

    sequence_lengths = [len(token_ids) for token_ids in token_lists]
    seqlens = torch.tensor(sequence_lengths, device=device, dtype=torch.int32)
    tokens = torch.cat(
        [
            torch.tensor(token_ids, device=device, dtype=torch.long)
            for token_ids in token_lists
        ]
    )
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.int32),
            seqlens.cumsum(dim=0).to(torch.int32),
        ]
    )
    positions = torch.cat(
        [
            torch.arange(length, device=device, dtype=torch.int32)
            for length in sequence_lengths
        ]
    )
    context = {
        "cu_seqlens_q": cu_seqlens,
        "cu_seqlens_k": cu_seqlens,
        "max_seqlen_q": max(sequence_lengths),
        "max_seqlen_k": max(sequence_lengths),
        "positions": positions,
    }
    hidden, _, _ = model(tokens, context=context, last_hidden_only=True)
    logits = model.output(hidden).float()

    for index, record in enumerate(batch):
        sequence_start = int(cu_seqlens[index].item())
        prompt_len = len(record["prompt_token_ids"])
        output_ids = record["output_token_ids"]
        prediction_start = sequence_start + prompt_len - 1
        prediction_end = prediction_start + len(output_ids)
        prediction_logits = logits[prediction_start:prediction_end]
        target_ids = torch.tensor(
            output_ids,
            device=device,
            dtype=torch.long,
        )
        selected_logits = prediction_logits.gather(1, target_ids.unsqueeze(1)).squeeze(
            1
        )
        token_logprobs = (
            selected_logits - torch.logsumexp(prediction_logits, dim=-1)
        ).cpu()
        if not torch.isfinite(token_logprobs).all():
            raise RuntimeError(
                f"Native scoring produced non-finite logprobs for {record['name']}"
            )
        scored_records.append(
            {
                "name": record["name"],
                "prompt_tokens": len(record["prompt_token_ids"]),
                "output_tokens": len(output_ids),
                "output_token_ids": output_ids,
                "native_token_logprobs": token_logprobs.tolist(),
            }
        )

    del hidden, logits, tokens
    torch.cuda.empty_cache()


def run_native_score(args: argparse.Namespace) -> None:
    import torch.distributed as dist

    try:
        model, modelargs, fa4_call_count = _load_native_model(args)
        for rollout_path in args.rollout:
            rollout = _read_json(rollout_path)
            for batch_size in args.batch_size:
                result = _score_rollout(
                    model,
                    modelargs,
                    rollout,
                    batch_size,
                    args.max_model_len,
                    fa4_call_count,
                )
                output_path = args.out_dir / (
                    f"{rollout_path.stem}.native-{args.quant_mode}-b{batch_size}.json"
                )
                _write_json(output_path, result)
                native = result["native"]
                print(
                    f"[rl-native] {rollout['setting']}: "
                    f"{native['output_tokens']} output tokens in "
                    f"{native['scoring_seconds']:.3f}s = "
                    f"{native['output_tokens_per_second']:.2f} tok/s; "
                    f"FA4 calls={native['fa4_calls']}; saved {output_path}",
                    flush=True,
                )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def compare(args: argparse.Namespace) -> None:
    rollout = _read_json(args.rollout)
    native = _read_json(args.native)
    if rollout.get("sampling", {}).get("min_tokens", 0) != 0:
        raise ValueError(
            "Cannot estimate KL from a rollout with min_tokens != 0 and raw "
            "pre-sampling logprobs"
        )
    native_by_name = {record["name"]: record for record in native["records"]}

    token_rows = []
    sequence_rows = []
    for rollout_record in rollout["records"]:
        name = rollout_record["name"]
        native_record = native_by_name.get(name)
        if native_record is None:
            raise KeyError(f"Native score is missing rollout record {name}")
        if rollout_record["output_token_ids"] != native_record["output_token_ids"]:
            raise ValueError(f"Output token mismatch for {name}")
        rollout_lps = rollout_record["output_token_logprobs"]
        native_lps = native_record["native_token_logprobs"]
        if len(rollout_lps) != len(native_lps):
            raise ValueError(f"Logprob length mismatch for {name}")

        sequence_k3 = []
        sequence_sampled_kl = []
        for token_index, (rollout_lp, native_lp) in enumerate(
            zip(rollout_lps, native_lps)
        ):
            sampled_kl = rollout_lp - native_lp
            native_over_rollout = native_lp - rollout_lp
            ratio = math.exp(max(-30.0, min(30.0, native_over_rollout)))
            k3_kl = ratio - 1.0 - native_over_rollout
            row = {
                "name": name,
                "token_index": token_index,
                "token_id": rollout_record["output_token_ids"][token_index],
                "rollout_logprob": rollout_lp,
                "native_logprob": native_lp,
                "sampled_kl": sampled_kl,
                "k3_kl": k3_kl,
                "abs_logprob_diff": abs(sampled_kl),
            }
            token_rows.append(row)
            sequence_sampled_kl.append(sampled_kl)
            sequence_k3.append(k3_kl)
        sequence_rows.append(
            {
                "name": name,
                "output_tokens": len(rollout_lps),
                "mean_sampled_kl": sum(sequence_sampled_kl) / len(sequence_sampled_kl),
                "mean_k3_kl": sum(sequence_k3) / len(sequence_k3),
            }
        )

    sampled_kl_values = [row["sampled_kl"] for row in token_rows]
    k3_values = [row["k3_kl"] for row in token_rows]
    abs_diff_values = [row["abs_logprob_diff"] for row in token_rows]
    position_buckets = []
    for start, end in ((0, 64), (64, 128), (128, 256), (256, 512), (512, 1024)):
        rows = [row for row in token_rows if start <= row["token_index"] < end]
        if not rows:
            continue
        position_buckets.append(
            {
                "start": start,
                "end": end,
                "tokens": len(rows),
                "mean_sampled_kl": sum(row["sampled_kl"] for row in rows) / len(rows),
                "mean_k3_kl": sum(row["k3_kl"] for row in rows) / len(rows),
                "mean_abs_logprob_diff": sum(row["abs_logprob_diff"] for row in rows)
                / len(rows),
            }
        )

    mean_sampled_kl = sum(sampled_kl_values) / len(sampled_kl_values)
    mean_k3_kl = sum(k3_values) / len(k3_values)
    payload = {
        "setting": rollout["setting"],
        "metric_definition": {
            "sampled_kl": (
                "Mean log p_vllm(a|s) - log p_llm_train(a|s) for actions "
                "sampled from vLLM with temperature=1 and top_p=1. This is the "
                "on-policy Monte Carlo estimator of KL(vLLM || llm-train)."
            ),
            "k3_kl": (
                "Mean exp(log p_llm_train - log p_vllm) - 1 - "
                "(log p_llm_train - log p_vllm), the non-negative low-variance "
                "k3 estimator of the same KL."
            ),
        },
        "rollout_performance": rollout["performance"],
        "output_length": rollout["output_length"],
        "native": native["native"],
        "tokens": len(token_rows),
        "mean_sampled_kl": mean_sampled_kl,
        "mean_k3_kl": mean_k3_kl,
        "sampled_kl": _metric_summary(sampled_kl_values),
        "k3_kl": _metric_summary(k3_values),
        "abs_logprob_diff": _metric_summary(abs_diff_values),
        "passes_1e_2": mean_k3_kl < 1e-2,
        "passes_5e_3": mean_k3_kl < 5e-3,
        "position_buckets": position_buckets,
        "sequences": sequence_rows,
    }
    _write_json(args.out, payload)
    print(
        f"[rl-compare] {rollout['setting']}: tokens={len(token_rows)} "
        f"sampled_KL={mean_sampled_kl:.9g} k3_KL={mean_k3_kl:.9g} "
        f"abs_lp_diff={sum(abs_diff_values) / len(abs_diff_values):.9g} "
        f"rollout={rollout['performance']['aggregate_output_tokens_per_second']:.2f} "
        f"tok/s pass<1e-2={mean_k3_kl < 1e-2} "
        f"pass<5e-3={mean_k3_kl < 5e-3}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prompts = subparsers.add_parser("prompts")
    prompts.add_argument("--model", required=True)
    prompts.add_argument("--num-prompts", type=int, default=64)
    prompts.add_argument("--out", type=Path, required=True)
    prompts.set_defaults(func=create_prompts)

    rollout = subparsers.add_parser("rollout")
    rollout.add_argument("--prompts", type=Path, required=True)
    rollout.add_argument("--url", action="append")
    rollout.add_argument("--served-model-name", default="yoco")
    rollout.add_argument("--setting", required=True)
    rollout.add_argument("--max-tokens", type=int, default=512)
    rollout.add_argument("--min-tokens", type=int, default=0)
    rollout.add_argument("--temperature", type=float, default=1.0)
    rollout.add_argument("--top-p", type=float, default=1.0)
    rollout.add_argument("--seed", type=int, default=1234)
    rollout.add_argument("--timeout", type=float, default=1800)
    rollout.add_argument("--out", type=Path, required=True)
    rollout.set_defaults(func=run_rollout)

    load = subparsers.add_parser("load")
    load.add_argument("--prompts", type=Path, required=True)
    load.add_argument("--url", action="append")
    load.add_argument("--served-model-name", default="yoco")
    load.add_argument("--setting", required=True)
    load.add_argument("--num-requests", type=int, default=64)
    load.add_argument("--request-rate", type=float, required=True)
    load.add_argument("--burstiness", type=float, default=1.0)
    load.add_argument("--max-tokens", type=int, default=256)
    load.add_argument("--min-tokens", type=int, default=256)
    load.add_argument("--temperature", type=float, default=1.0)
    load.add_argument("--top-p", type=float, default=1.0)
    load.add_argument("--seed", type=int, default=1234)
    load.add_argument("--timeout", type=float, default=1800)
    load.add_argument("--metrics-interval", type=float, default=0.2)
    load.add_argument("--out", type=Path, required=True)
    load.set_defaults(func=run_continuous_load)

    native = subparsers.add_parser("native-score")
    native.add_argument("--rollout", type=Path, action="append", required=True)
    native.add_argument("--out-dir", type=Path, required=True)
    native.add_argument("--native-checkpoint", required=True)
    native.add_argument("--llm-train-dir", default="/workspace/shaohanh/llm-train")
    native.add_argument(
        "--quant-mode",
        choices=("bfloat16", "mxfp8"),
        required=True,
    )
    native.add_argument("--quant-block-size", type=int)
    native.add_argument("--batch-size", type=int, nargs="+", default=[16])
    native.add_argument("--max-model-len", type=int, default=2048)
    native.add_argument(
        "--use-torch-fp8-quant",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    native.set_defaults(func=run_native_score)

    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--rollout", type=Path, required=True)
    comparison.add_argument("--native", type=Path, required=True)
    comparison.add_argument("--out", type=Path, required=True)
    comparison.set_defaults(func=compare)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
