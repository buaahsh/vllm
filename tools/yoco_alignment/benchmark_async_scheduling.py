# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare fixed-K8 schedulers with staggered arrivals and client cancellation."""

import argparse
import asyncio
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tools.yoco_alignment.benchmark_mega_moe import PROMPTS
from tools.yoco_alignment.scheduling import SchedulingOptions


async def collect_request(engine, request, params, epoch):
    await asyncio.sleep(max(0.0, epoch + request["arrival_s"] - time.perf_counter()))
    submitted = time.perf_counter()
    stream = engine.generate(
        {"prompt_token_ids": request["prompt_token_ids"]},
        params,
        request["request_id"],
    )
    events = []
    token_ids = []
    finished = False
    cancelled = False
    finish_reason = None
    try:
        async for output in stream:
            now = time.perf_counter()
            completion = output.outputs[0]
            new_tokens = list(completion.token_ids)
            if new_tokens:
                token_ids.extend(new_tokens)
                events.append({"time_s": now - epoch, "token_ids": new_tokens})
            finished = output.finished
            finish_reason = completion.finish_reason
            limit = request.get("cancel_after_updates")
            if limit is not None and len(events) >= limit and not finished:
                cancelled = True
                break
    finally:
        # Closing AsyncLLM.generate aborts an unfinished request in the engine.
        await stream.aclose()
    ended = time.perf_counter()
    if len(token_ids) > request["max_tokens"]:
        raise RuntimeError("Request emitted tokens beyond its output budget")
    if not cancelled and not finished:
        raise RuntimeError("Request stream ended without a terminal result")
    if (
        not cancelled
        and request["ignore_eos"]
        and len(token_ids) != request["max_tokens"]
    ):
        raise RuntimeError("Fixed-output request did not reach its output budget")
    first = events[0]["time_s"] if events else None
    last = events[-1]["time_s"] if events else None
    return {
        "request_id": request["request_id"],
        "planned_arrival_s": request["arrival_s"],
        "submitted_s": submitted - epoch,
        "submission_lateness_s": submitted - epoch - request["arrival_s"],
        "ended_s": ended - epoch,
        "elapsed_s": ended - submitted,
        "ttft_ms": None if first is None else 1000 * (first - (submitted - epoch)),
        "mean_tpot_ms": (
            1000 * (last - first) / (len(token_ids) - 1)
            if first is not None and last is not None and len(token_ids) > 1
            else None
        ),
        "inter_update_ms": [
            1000 * (current["time_s"] - previous["time_s"])
            for previous, current in zip(events, events[1:])
        ],
        "events": events,
        "output_ids": token_ids,
        "finished": finished,
        "cancelled": cancelled,
        "finish_reason": finish_reason,
    }


def build_workload(tokenizer, count, interval, long_tokens):
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        for text in PROMPTS
    ]
    filler = tokenizer.encode(
        "This technical note describes model execution and cache reuse. ",
        add_special_tokens=False,
    )
    natural_stop_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with exactly one word: OK."}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    requests = []
    for index in range(count):
        tokens = list(
            natural_stop_prompt if index % 4 == 2 else prompts[index % len(prompts)]
        )
        if long_tokens and index % 3 == 1:
            needed = long_tokens - len(tokens)
            if needed < 0:
                raise ValueError("long-prompt-tokens is shorter than the chat prompt")
            tokens = (filler * (needed // len(filler) + 1))[:needed] + tokens
        request = {
            "request_id": f"request-{index:04d}",
            "prompt_token_ids": tokens,
            "arrival_s": index * interval,
            "max_tokens": [24, 40, 64, 128][index % 4],
            "seed": 20261022 + index,
            "ignore_eos": index % 4 != 2,
        }
        if index % 8 == 3:
            request["cancel_after_updates"] = 2
        requests.append(request)
    return requests


async def run(args, config, scheduling):
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    def save(name: str, data: Any):
        (args.output / name).write_text(json.dumps(data, indent=2) + "\n")

    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**config))
    profiling = False
    try:
        workers = await engine.collective_rpc("ab_bind_graphs")
        for worker in workers:
            scheduling.validate_runtime(worker)
        save("worker-before.json", workers)
        tokenizer = engine.get_tokenizer()
        requests = (
            json.loads(args.workload.read_text())
            if args.workload
            else build_workload(
                tokenizer, args.requests, args.arrival_interval, args.long_prompt_tokens
            )
        )
        if len({request["request_id"] for request in requests}) != len(requests):
            raise ValueError("Workload contains duplicate request IDs")
        for request in requests:
            if len(request["prompt_token_ids"]) + request["max_tokens"] > config.get(
                "max_model_len", 12288
            ):
                raise ValueError("Workload exceeds max_model_len")
        save("workload.json", requests)
        for warmup in range(2):
            jobs = []
            for index in range(min(8, len(requests))):
                request = dict(requests[index])
                request.update(request_id=f"warmup-{warmup}-{index}", arrival_s=0.0)
                request.pop("cancel_after_updates", None)
                params = SamplingParams(
                    temperature=1.0,
                    seed=request["seed"],
                    max_tokens=request["max_tokens"],
                    ignore_eos=request["ignore_eos"],
                    output_kind=RequestOutputKind.DELTA,
                )
                jobs.append(
                    collect_request(engine, request, params, time.perf_counter())
                )
            await asyncio.gather(*jobs)
        await engine.collective_rpc("ab_synchronize")
        if args.profile_dir is not None:
            await engine.collective_rpc(
                "ab_profile_start", kwargs={"annotate_scheduler": True}
            )
            profiling = True
        epoch = time.perf_counter()
        tasks = []
        async with asyncio.TaskGroup() as group:
            for request in requests:
                params = SamplingParams(
                    temperature=1.0,
                    seed=request["seed"],
                    max_tokens=request["max_tokens"],
                    ignore_eos=request["ignore_eos"],
                    output_kind=RequestOutputKind.DELTA,
                )
                tasks.append(
                    group.create_task(collect_request(engine, request, params, epoch))
                )
        rows = [task.result() for task in tasks]
        client_elapsed = time.perf_counter() - epoch
        await engine.collective_rpc("ab_synchronize")
        elapsed = time.perf_counter() - epoch
        save("requests.json", rows)
        if profiling:
            profile = await engine.collective_rpc(
                "ab_profile_stop", kwargs={"trace_dir": str(args.profile_dir)}
            )
            profiling = False
            save("profile.json", profile)
        final_workers = await engine.collective_rpc("ab_status")
        for worker in final_workers:
            scheduling.validate_runtime(worker)
        unfinished = engine.get_num_unfinished_requests()
        if unfinished:
            raise RuntimeError(f"Engine retained {unfinished} unfinished requests")
        completed_tokens = sum(
            len(row["output_ids"]) for row in rows if not row["cancelled"]
        )
        save(
            "complete.json",
            {
                "status": "PASS",
                "scheduling": asdict(scheduling),
                "engine_client": type(engine.engine_core).__qualname__,
                "elapsed_s": elapsed,
                "client_elapsed_s": client_elapsed,
                "drain_s": elapsed - client_elapsed,
                "profiled": args.profile_dir is not None,
                "completed_output_tokens": completed_tokens,
                "completed_output_tokens_per_s": completed_tokens / elapsed,
                "cancelled_requests": sum(row["cancelled"] for row in rows),
                "unfinished_requests": unfinished,
                "workers": final_workers,
                "timing_note": "Per-update intervals; speculative chunks can contain "
                "multiple tokens. These are not per-token ITL percentiles.",
            },
        )
    finally:
        try:
            if profiling:
                await engine.collective_rpc("ab_profile_stop")
        finally:
            engine.shutdown(timeout=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scheduler-mode", choices=["sync", "async"], required=True)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--arrival-interval", type=float, default=0.02)
    parser.add_argument("--long-prompt-tokens", type=int, default=0)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    if (
        args.requests < 4
        or not math.isfinite(args.arrival_interval)
        or args.arrival_interval < 0
        or args.long_prompt_tokens < 0
    ):
        parser.error(
            "Require at least 4 requests and nonnegative finite arrivals/lengths"
        )
    scheduling = SchedulingOptions("multiprocess", args.scheduler_mode)
    scheduling.apply_environment()
    config = json.loads(args.config.read_text())
    spec = config.get("speculative_config", {})
    if spec.get("method") != "dspark" or spec.get("num_speculative_tokens") != 8:
        parser.error("This benchmark requires the fixed DSpark K8 configuration")
    config.update(scheduling.llm_kwargs())
    config["worker_extension_cls"] = (
        "tools.yoco_alignment.benchmark_mega_moe.MegaMoEBenchmarkWorker"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    try:
        asyncio.run(run(args, config, scheduling))
    except BaseException as exc:
        (args.output / "failure.json").write_text(
            json.dumps({"error": type(exc).__name__, "message": str(exc)}, indent=2)
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()
