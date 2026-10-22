# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prevent scheduler comparisons from also changing engine process layout."""

import argparse
import asyncio
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.yoco_alignment.benchmark_async_scheduling import collect_request
from tools.yoco_alignment.benchmark_mega_moe import MegaMoEBenchmarkWorker
from tools.yoco_alignment.scheduling import (
    SchedulingOptions,
    add_scheduling_arguments,
    resolve_scheduling_options,
)


def parse_options(*argv):
    parser = argparse.ArgumentParser()
    add_scheduling_arguments(parser)
    return resolve_scheduling_options(parser.parse_args(argv))


@pytest.mark.parametrize("mode,enabled", [("sync", False), ("async", True)])
def test_scheduler_comparison_keeps_multiprocess_engine(mode, enabled, monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    options = parse_options("--engine-mode", "multiprocess", "--scheduler-mode", mode)
    options.apply_environment()
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "1"
    assert options.llm_kwargs() == {"async_scheduling": enabled}


@pytest.mark.parametrize("engine_value", [None, "0", "1"])
def test_auto_preserves_existing_environment_and_engine_defaults(
    engine_value, monkeypatch
):
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    if engine_value is not None:
        monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", engine_value)
    options = parse_options()
    options.apply_environment()
    assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == engine_value
    assert options.llm_kwargs() == {}


def test_legacy_option_preserves_in_process_synchronous_behavior(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    options = parse_options("--synchronous-engine")
    options.apply_environment()
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    assert options.llm_kwargs() == {"async_scheduling": False}


@pytest.mark.parametrize(
    "argv",
    [("--engine-mode", "auto"), ("--scheduler-mode", "async")],
)
def test_legacy_option_rejects_ambiguous_overrides(argv):
    with pytest.raises(ValueError, match="cannot be combined"):
        parse_options("--synchronous-engine", *argv)


def test_runtime_receipt_rejects_silently_disabled_async():
    options = SchedulingOptions("multiprocess", "async")
    with pytest.raises(RuntimeError, match="resolved async=False"):
        options.validate_runtime(
            {"async_scheduling": False, "max_concurrent_batches": 1}
        )


def test_runtime_receipt_requires_async_batch_capacity():
    options = SchedulingOptions("multiprocess", "async")
    with pytest.raises(RuntimeError, match="no concurrent batch capacity"):
        options.validate_runtime(
            {"async_scheduling": True, "max_concurrent_batches": 1}
        )
    options.validate_runtime({"async_scheduling": True, "max_concurrent_batches": 2})


class StreamingEngine:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False
        self.delivered = 0

    async def generate(self, prompt, params, request_id):
        try:
            for tokens, finished in self.chunks:
                self.delivered += 1
                yield SimpleNamespace(
                    finished=finished,
                    outputs=[SimpleNamespace(token_ids=tokens, finish_reason="length")],
                )
        finally:
            self.closed = True


def streaming_request(**kwargs):
    return {
        "request_id": "test",
        "prompt_token_ids": [1],
        "arrival_s": 0,
        "max_tokens": 3,
        "ignore_eos": True,
        **kwargs,
    }


def test_streaming_collector_preserves_multi_token_deltas():
    engine = StreamingEngine([([10, 11], False), ([12], True)])
    result = asyncio.run(
        collect_request(engine, streaming_request(), None, time.perf_counter())
    )
    assert result["output_ids"] == [10, 11, 12]
    assert result["finished"] and not result["cancelled"]
    assert engine.closed


def test_client_cancellation_closes_stream_before_later_chunks():
    engine = StreamingEngine([([10], False), ([11, 12], True)])
    result = asyncio.run(
        collect_request(
            engine,
            streaming_request(cancel_after_updates=1),
            None,
            time.perf_counter(),
        )
    )
    assert result["cancelled"] and result["output_ids"] == [10]
    assert engine.closed and engine.delivered == 1


def test_collector_rejects_missing_terminal_result():
    engine = StreamingEngine([([10], False)])
    with pytest.raises(RuntimeError, match="without a terminal result"):
        asyncio.run(
            collect_request(engine, streaming_request(), None, time.perf_counter())
        )
    assert engine.closed


def test_profile_stop_restores_scheduler_and_releases_trace(monkeypatch, tmp_path):
    import torch

    import vllm.v1.core.sched.scheduler as scheduler_module

    class FakeScheduler:
        def schedule(self):
            return "planned"

        def update_from_output(self):
            return "committed"

    class FakeProfiler:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def key_averages(self):
            return SimpleNamespace(table=lambda **kwargs: "summary")

        def export_chrome_trace(self, path):
            Path(path).write_text("{}")

    monkeypatch.setattr(scheduler_module, "Scheduler", FakeScheduler)
    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: FakeProfiler())
    schedule = FakeScheduler.schedule
    update = FakeScheduler.update_from_output
    worker = MegaMoEBenchmarkWorker()
    worker.ab_profile_start(annotate_scheduler=True)
    assert FakeScheduler.schedule is not schedule
    assert FakeScheduler().schedule() == "planned"
    result = worker.ab_profile_stop(trace_dir=tmp_path)
    assert FakeScheduler.schedule is schedule
    assert FakeScheduler.update_from_output is update
    assert worker._ab_profile is None
    assert Path(result["trace_path"]).read_text() == "{}"
