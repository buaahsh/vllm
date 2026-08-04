#!/usr/bin/env bash
set -euo pipefail

model="${MODEL:?Set MODEL to the YOCO checkpoint path}"
model_name="${MODEL_NAME:-yoco-v2-long}"
host="${HOST:-0.0.0.0}"
port="${PORT:-8001}"
dp_size="${DP_SIZE:-1}"
max_num_seqs="${MAX_NUM_SEQS:-128}"
async_scheduling="${ASYNC_SCHEDULING:-1}"

if [[ ! "${dp_size}" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "${max_num_seqs}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DP_SIZE and MAX_NUM_SEQS must be positive integers" >&2
  exit 2
fi

case "${async_scheduling}" in
  1 | true) scheduler_args=(--async-scheduling) ;;
  0 | false) scheduler_args=(--no-async-scheduling) ;;
  *)
    echo "ASYNC_SCHEDULING must be 0, 1, false, or true" >&2
    exit 2
    ;;
esac

exec vllm serve "${model}" \
  --served-model-name "${model_name}" \
  --host "${host}" \
  --port "${port}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --attention-backend FLASHINFER \
  --moe-backend triton \
  --tensor-parallel-size 1 \
  --data-parallel-size "${dp_size}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}" \
  --max-num-seqs "${max_num_seqs}" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --kv-sharing-fast-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser agens \
  --reasoning-parser agens \
  "${scheduler_args[@]}" \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
