# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-GPU Fast FP8 decode; see FAST_FP8_DECODE_BENCHMARK.md."""

from pathlib import Path

from benchmark_fast_decode import main

if __name__ == "__main__":
    main(precision="fp8", entrypoint=Path(__file__))
