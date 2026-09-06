#!/usr/bin/env bash
set -euo pipefail

model="${MODEL:?Set MODEL to the YOCO checkpoint path}"
model_name="${MODEL_NAME:-yoco-v2-long}"
host="${HOST:-0.0.0.0}"
port="${PORT:-8001}"
dp_size="${DP_SIZE:-1}"
max_num_seqs="${MAX_NUM_SEQS:-128}"
max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-auto}"
async_scheduling="${ASYNC_SCHEDULING:-1}"
attention_backend="${ATTENTION_BACKEND:-FLASHINFER}"
enable_expert_parallel="${ENABLE_EXPERT_PARALLEL:-auto}"
all2all_backend="${ALL2ALL_BACKEND:-allgather_reducescatter}"
nvshmem_qp_depth="${NVSHMEM_QP_DEPTH:-auto}"
# "auto" is a launcher sentinel, not a value understood by NVSHMEM.
unset NVSHMEM_QP_DEPTH

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

case "${enable_expert_parallel}" in
  auto)
    if ((dp_size > 1)); then
      enable_expert_parallel=1
    else
      enable_expert_parallel=0
    fi
    ;;
  1 | true) enable_expert_parallel=1 ;;
  0 | false) enable_expert_parallel=0 ;;
  *)
    echo "ENABLE_EXPERT_PARALLEL must be auto, 0, 1, false, or true" >&2
    exit 2
    ;;
esac

case "${max_num_batched_tokens}" in
  auto)
    if [[ "${enable_expert_parallel}" == 1 ]] \
      && [[ "${all2all_backend}" == deepep_low_latency ]]; then
      # DeepEP LL's RDMA buffer grows with this token budget. YOCO's
      # 32K budget exceeds DeepEP's int32 buffer-index limit; 8K is safe.
      max_num_batched_tokens=8192
    else
      max_num_batched_tokens=32768
    fi
    ;;
  *)
    if [[ ! "${max_num_batched_tokens}" =~ ^[1-9][0-9]*$ ]]; then
      echo "MAX_NUM_BATCHED_TOKENS must be auto or a positive integer" >&2
      exit 2
    fi
    ;;
esac

if [[ "${enable_expert_parallel}" == 1 ]] \
  && [[ "${all2all_backend}" == deepep_low_latency ]]; then
  required_qp_depth=$(((max_num_batched_tokens + 1) * 2))
  case "${nvshmem_qp_depth}" in
    auto)
      nvshmem_qp_depth=1
      while ((nvshmem_qp_depth < required_qp_depth)); do
        nvshmem_qp_depth=$((nvshmem_qp_depth * 2))
      done
      ;;
    *)
      if [[ ! "${nvshmem_qp_depth}" =~ ^[1-9][0-9]*$ ]] \
        || ((nvshmem_qp_depth < required_qp_depth)); then
        echo "NVSHMEM_QP_DEPTH must be at least ${required_qp_depth}" >&2
        exit 2
      fi
      ;;
  esac
  export NVSHMEM_QP_DEPTH="${nvshmem_qp_depth}"
  echo "DeepEP low-latency NVSHMEM_QP_DEPTH=${NVSHMEM_QP_DEPTH}"
fi

parallel_args=()
if [[ "${enable_expert_parallel}" == 1 ]]; then
  if ((dp_size == 1)); then
    echo "Expert parallelism requires DP_SIZE greater than one" >&2
    exit 2
  fi

  case "${all2all_backend}" in
    deepep_low_latency | deepep_high_throughput)
      deep_ep_so="$(python - <<'PY'
import importlib.util

spec = importlib.util.find_spec("deep_ep_cpp")
if spec is None or spec.origin is None:
    raise SystemExit("deep_ep_cpp extension was not found")
print(spec.origin)
PY
)"
      resolved_nvshmem="$(ldd "${deep_ep_so}" \
        | awk '$1 == "libnvshmem_host.so.3" {print $3}')"
      case "${resolved_nvshmem}" in
        */site-packages/nvidia/nvshmem/lib/libnvshmem_host.so.3) ;;
        */dist-packages/nvidia/nvshmem/lib/libnvshmem_host.so.3) ;;
        *)
          echo "DeepEP resolves an incompatible NVSHMEM: ${resolved_nvshmem:-not found}" >&2
          exit 2
          ;;
      esac
      python - <<'PY'
import importlib.metadata as metadata
import deep_ep

print(
    "DeepEP runtime verified:",
    metadata.version("deep_ep"),
    "NVSHMEM",
    metadata.version("nvidia-nvshmem-cu13"),
)
PY
      if [[ "${all2all_backend}" == deepep_low_latency ]]; then
        if ! compgen -G '/dev/infiniband/uverbs*' >/dev/null; then
          echo "DeepEP low latency requires visible /dev/infiniband/uverbs devices" >&2
          exit 2
        fi
        if ! grep -Eq '^EnableStreamMemOPs:[[:space:]]+1$' \
          /proc/driver/nvidia/params 2>/dev/null \
          && [[ ! -c /dev/gdrdrv ]]; then
          echo "DeepEP low latency requires IBGDA: set NVIDIA" \
            "EnableStreamMemOPs=1 and reboot, or load gdrdrv" >&2
          exit 2
        fi
        python - "${model}/config.json" \
          "${max_num_batched_tokens}" "${dp_size}" <<'PY'
import json
import sys

import deep_ep

config_path, max_tokens, ep_size = sys.argv[1:]
with open(config_path) as config_file:
    config = json.load(config_file)
rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
    int(max_tokens),
    int(config["d_model"]),
    int(ep_size),
    int(config["moe_expert_num"]),
)
if rdma_bytes // 16 >= 2**31 - 1:
    raise SystemExit(
        "DeepEP low-latency RDMA buffer exceeds its int32 index limit: "
        f"{rdma_bytes / (1 << 30):.2f} GiB for {max_tokens} tokens; "
        "use MAX_NUM_BATCHED_TOKENS=8192 or deepep_high_throughput"
    )
print(f"DeepEP low-latency RDMA buffer: {rdma_bytes / (1 << 30):.2f} GiB")
PY
      fi
      ;;
  esac
  parallel_args=(--enable-expert-parallel --all2all-backend "${all2all_backend}")
fi

exec vllm serve "${model}" \
  --served-model-name "${model_name}" \
  --host "${host}" \
  --port "${port}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --attention-backend "${attention_backend}" \
  --moe-backend triton \
  --tensor-parallel-size 1 \
  --data-parallel-size "${dp_size}" \
  "${parallel_args[@]}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --max-num-batched-tokens "${max_num_batched_tokens}" \
  --max-num-seqs "${max_num_seqs}" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --kv-sharing-fast-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser agens \
  --reasoning-parser agens \
  "${scheduler_args[@]}" \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
