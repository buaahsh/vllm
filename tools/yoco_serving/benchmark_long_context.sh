#!/usr/bin/env bash
set -euo pipefail

base_url="${BASE_URL:-http://127.0.0.1:8001}"
model_name="${MODEL_NAME:-yoco-v2-long}"
tokenizer="${TOKENIZER:?Set TOKENIZER to the YOCO checkpoint path}"
dp_size="${DP_SIZE:-1}"
gpu_indices="${GPU_INDICES:-0}"
batches="${BATCHES:-1 2 4 8}"
result_dir="${RESULT_DIR:-/tmp/yoco-long-context-bench}"
repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

mkdir -p "${result_dir}/single-turn" "${result_dir}/agentic"

run_single_turn() {
  local workload=$1
  local input_tokens=$2
  local output_tokens=$3
  local batch=$4
  local seed=$5

  vllm bench serve \
    --backend vllm \
    --base-url "${base_url}" \
    --endpoint /v1/completions \
    --model "${model_name}" \
    --tokenizer "${tokenizer}" \
    --trust-remote-code \
    --dataset-name random \
    --random-input-len "${input_tokens}" \
    --random-output-len "${output_tokens}" \
    --random-range-ratio 0.0 \
    --seed "${seed}" \
    --num-prompts "${batch}" \
    --max-concurrency "${batch}" \
    --temperature 0 \
    --ignore-eos \
    --disable-tqdm \
    --ready-check-timeout-sec 0 \
    --percentile-metrics ttft,tpot,itl \
    --metric-percentiles 50,95,99 \
    --save-result \
    --save-detailed \
    --result-dir "${result_dir}/single-turn" \
    --result-filename "${workload}-batch${batch}.json"
}

for batch in ${batches}; do
  run_single_turn workload-1 8192 65536 "${batch}" "$((106400 + batch))"
  run_single_turn workload-2 65536 16384 "${batch}" "$((206400 + batch))"

  python "${repo_root}/benchmarks/multi_turn/benchmark_agent_trace.py" \
    --base-url "${base_url}/v1" \
    --model "${model_name}" \
    --tokenizer "${tokenizer}" \
    --output "${result_dir}/agentic/workload-3-batch${batch}.json" \
    --concurrency "${batch}" \
    --dp-size "${dp_size}" \
    --trajectories "${batch}" \
    --turns 40 \
    --skip-warmup \
    --prefill-per-turn 2925 \
    --output-per-turn 325 \
    --cache-alignment 1056 \
    --gpu-indices "${gpu_indices}" \
    --seed "$((306400 + batch))" \
    --timeout 7200
done
