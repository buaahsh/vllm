#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the packaged YOCO N1280 MoE config with vLLM defaults."""

import importlib.util

BENCHMARK_PATH = "/workspace/vllm/benchmarks/kernels/benchmark_moe_defaults.py"
spec = importlib.util.spec_from_file_location("benchmark_moe_defaults", BENCHMARK_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {BENCHMARK_PATH}")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

from vllm.model_executor.layers.fused_moe.fused_moe import (  # noqa: E402
    get_default_config,
    get_moe_configs,
)

configs = get_moe_configs(128, 1280, None, None, None)
if configs is None:
    raise RuntimeError("The packaged E128/N1280 B200 MoE config was not found")
for batch in (1, 2, 4, 8):
    packaged = configs[min(configs, key=lambda value: abs(value - batch))]
    fallback = get_default_config(batch, 128, 1280, 3072, 8, None, None)
    if packaged != fallback:
        raise RuntimeError(
            f"Batch {batch} must use the runtime fallback config: "
            f"packaged={packaged}, fallback={fallback}"
        )
print("Verified: packaged M=1/2/4/8 entries equal the runtime fallback configs.")

benchmark.MODELS = [
    ("YOCO-v2 BF16", 128, 1280, 3072, 8, None, False, None),
]
benchmark.BATCH_SIZES = [1, 2, 4, 8, 16, 32, 128, 1024, 2843, 3899, 8192, 32768]
benchmark.main()
