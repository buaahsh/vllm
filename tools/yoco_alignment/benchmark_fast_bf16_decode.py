# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-GPU Fast BF16 decode with the same workload/timer as Fast FP8.

BF16 weights and KV, FA4, BF16 residual/router; no online FP8 quantization.
See docs/yoco/performance/FAST_BF16_DECODE_BENCHMARK.md.
"""

from pathlib import Path

from benchmark_fast_decode import main

if __name__ == "__main__":
    main(precision="bf16", entrypoint=Path(__file__))
