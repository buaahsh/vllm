#!/usr/bin/env bash
set -euo pipefail

base_url="${BASE_URL:-http://127.0.0.1:8001/v1}"
model_name="${MODEL_NAME:-yoco-v2-long}"
tokenizer="${TOKENIZER:?Set TOKENIZER to the YOCO checkpoint path}"
dp_size="${DP_SIZE:-1}"
gpu_indices="${GPU_INDICES:-0}"
result_dir="${RESULT_DIR:-/tmp/yoco-long-context-warmup}"
repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

mkdir -p "${result_dir}"

# Warm the exact long-running agent trajectory, including the high-context
# YOCO RMSNorm and routing kernels that a two-turn warmup does not reach.
python "${repo_root}/benchmarks/multi_turn/benchmark_agent_trace.py" \
  --base-url "${base_url}" \
  --model "${model_name}" \
  --tokenizer "${tokenizer}" \
  --output "${result_dir}/warmup-only.json" \
  --concurrency "${dp_size}" \
  --dp-size "${dp_size}" \
  --trajectories "${dp_size}" \
  --turns 40 \
  --warmup-turns 40 \
  --warmup-only \
  --prefill-per-turn 2925 \
  --output-per-turn 325 \
  --cache-alignment 1056 \
  --gpu-indices "${gpu_indices}" \
  --timeout 7200
