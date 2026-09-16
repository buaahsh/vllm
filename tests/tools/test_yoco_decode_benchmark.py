# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve the benchmark's timing boundary and release a paused engine on errors."""

from types import SimpleNamespace

import pytest

from tools.yoco_alignment import benchmark_fast_decode as bench


class FakeLLM:
    def __init__(self, failure=None):
        self.llm_engine = SimpleNamespace(engine_core=self)
        self.clock = 0.0
        self.paused = False
        self.profiling = False
        self.failure = failure
        self.outputs = [object()]

    def call_utility(self, method, *args):
        self.paused = method == "pause_scheduler"
        self.clock += 5 if self.paused else 1

    def enqueue(self, prompts, params, **kwargs):
        assert self.paused
        self.clock += 10
        if self.failure == "enqueue":
            raise RuntimeError("enqueue failed")

    def wait_for_completion(self, **kwargs):
        assert not self.paused
        self.clock += 2
        if self.failure == "drain":
            raise RuntimeError("drain failed")
        return self.outputs

    def start_profile(self, name):
        assert self.paused
        self.profiling = True
        self.clock += 20

    def stop_profile(self):
        self.profiling = False
        self.clock += 20


def test_timer_excludes_enqueue_and_profiler_but_includes_resume_and_drain(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(bench.time, "perf_counter", lambda: llm.clock)
    outputs, seconds = bench.generate_batch(llm, [], object(), profile="test")
    assert outputs is llm.outputs
    assert seconds == 3
    assert not llm.paused and not llm.profiling


@pytest.mark.parametrize("failure", ["enqueue", "drain"])
def test_failed_generation_leaves_scheduler_resumed_and_profiler_stopped(failure):
    llm = FakeLLM(failure)
    with pytest.raises(RuntimeError, match=failure):
        bench.generate_batch(llm, [], object(), profile="test")
    assert not llm.paused and not llm.profiling
