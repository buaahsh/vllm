# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep engine multiprocessing independent of benchmark scheduling mode."""

import argparse
import os
from dataclasses import dataclass
from typing import Any, Literal

EngineMode = Literal["auto", "in-process", "multiprocess"]
SchedulerMode = Literal["auto", "sync", "async"]


@dataclass(frozen=True)
class SchedulingOptions:
    engine_mode: EngineMode
    scheduler_mode: SchedulerMode

    def apply_environment(self) -> None:
        if self.engine_mode != "auto":
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = (
                "1" if self.engine_mode == "multiprocess" else "0"
            )

    def llm_kwargs(self) -> dict[str, bool]:
        if self.scheduler_mode == "auto":
            return {}
        return {"async_scheduling": self.scheduler_mode == "async"}

    def validate_runtime(self, status: dict[str, Any]) -> None:
        actual = status["async_scheduling"]
        if self.scheduler_mode != "auto":
            expected = self.scheduler_mode == "async"
            if actual is not expected:
                raise RuntimeError(
                    f"Requested scheduler={self.scheduler_mode}, "
                    f"resolved async={actual}"
                )
        if actual and status["max_concurrent_batches"] < 2:
            raise RuntimeError("Async scheduling has no concurrent batch capacity")


def add_scheduling_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine-mode", choices=["auto", "in-process", "multiprocess"])
    parser.add_argument("--scheduler-mode", choices=["auto", "sync", "async"])
    parser.add_argument(
        "--synchronous-engine",
        action="store_true",
        help="Legacy shorthand for in-process engine and synchronous scheduling",
    )


def resolve_scheduling_options(args: argparse.Namespace) -> SchedulingOptions:
    if args.synchronous_engine:
        if args.engine_mode is not None or args.scheduler_mode is not None:
            raise ValueError(
                "--synchronous-engine cannot be combined with --engine-mode "
                "or --scheduler-mode"
            )
        return SchedulingOptions("in-process", "sync")
    return SchedulingOptions(args.engine_mode or "auto", args.scheduler_mode or "auto")
