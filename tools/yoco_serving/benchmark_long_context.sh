#!/usr/bin/env bash
set -euo pipefail

base_url="${BASE_URL:-http://127.0.0.1:8001}"
model_name="${MODEL_NAME:-yoco-v2-long}"
tokenizer="${TOKENIZER:?Set TOKENIZER to the YOCO checkpoint path}"
dp_size="${DP_SIZE:-1}"
gpu_indices="${GPU_INDICES:-0}"
batches="${BATCHES:-1 2 4 8}"
workloads="${WORKLOADS:-1 2 3}"
skip_existing="${SKIP_EXISTING:-1}"
run_id="${RUN_ID:-$(date +%s)-$$}"
result_dir="${RESULT_DIR:-/tmp/yoco-long-context-bench}"
repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if [[ ! "${run_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID may contain only letters, numbers, dot, underscore, and dash" >&2
  exit 2
fi

mkdir -p "${result_dir}/single-turn" "${result_dir}/agentic"

has_workload() {
  [[ " ${workloads} " == *" $1 "* ]]
}

result_is_complete() {
  local result=$1
  local identity_key=$2
  local identity_value=$3
  [[ "${skip_existing}" == 1 ]] \
    && [[ -s "${result}" ]] \
    && python -c \
      'import json, sys; sys.exit(json.load(open(sys.argv[1])).get(sys.argv[2]) != sys.argv[3])' \
      "${result}" "${identity_key}" "${identity_value}" >/dev/null 2>&1
}

run_single_turn() {
  local workload=$1
  local input_tokens=$2
  local output_tokens=$3
  local batch=$4
  local seed=$5
  local result="${result_dir}/single-turn/${workload}-batch${batch}.json"

  if result_is_complete "${result}" benchmark_run_id "${run_id}"; then
    echo "SKIP completed ${workload} batch=${batch}: ${result}"
    return
  fi
  echo "RUN ${workload} batch=${batch} input=${input_tokens} output=${output_tokens}"

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
    --extra-body "{\"cache_salt\":\"${run_id}-${workload}-batch${batch}\"}" \
    --disable-tqdm \
    --ready-check-timeout-sec 0 \
    --percentile-metrics ttft,tpot,itl \
    --metric-percentiles 50,95,99 \
    --metadata \
    "benchmark_run_id=${run_id}" \
    "workload=${workload}" \
    "batch=${batch}" \
    --save-result \
    --save-detailed \
    --result-dir "${result_dir}/single-turn" \
    --result-filename "${workload}-batch${batch}.json"
}

for batch in ${batches}; do
  if has_workload 1; then
    run_single_turn workload-1 8192 65536 "${batch}" "$((106400 + batch))"
  fi
  if has_workload 2; then
    run_single_turn workload-2 65536 16384 "${batch}" "$((206400 + batch))"
  fi
  if has_workload 3; then
    agent_result="${result_dir}/agentic/workload-3-batch${batch}.json"
    if result_is_complete \
      "${agent_result}" cache_salt_prefix \
      "${run_id}-workload-3-batch${batch}"; then
      echo "SKIP completed workload-3 batch=${batch}: ${agent_result}"
      continue
    fi
    echo "RUN workload-3 batch=${batch} turns=40 final_tokens=130000"
    python "${repo_root}/benchmarks/multi_turn/benchmark_agent_trace.py" \
      --base-url "${base_url}/v1" \
      --model "${model_name}" \
      --tokenizer "${tokenizer}" \
      --output "${agent_result}" \
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
      --cache-salt-prefix "${run_id}-workload-3-batch${batch}" \
      --timeout 7200
  fi
done
