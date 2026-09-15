# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility for YOCO training adapters using the former module path."""

from vllm.model_executor.determinism import batch_invariant as _implementation


def __getattr__(name: str):
    return getattr(_implementation, name)
