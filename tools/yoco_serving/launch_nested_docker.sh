#!/usr/bin/env bash
set -euo pipefail

image="${IMAGE:-buaahsh/pytorch:26.02-b200-vllm-yoco-longctx-multigpu-20260804}"
model="${MODEL:?Set MODEL to the host checkpoint path}"
dp_size="${DP_SIZE:-1}"
gpu_list="${GPU_LIST:-0}"
max_num_seqs="${MAX_NUM_SEQS:-128}"
port="${PORT:-8001}"
name="${NAME:-yoco-longctx-dp${dp_size}}"
result_dir="${RESULT_DIR:-/data/yoco-longctx-results/dp${dp_size}}"
allow_busy_gpus="${ALLOW_BUSY_GPUS:-0}"

IFS=, read -r -a selected_gpus <<<"${gpu_list}"
if [[ ! "${dp_size}" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "${max_num_seqs}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DP_SIZE and MAX_NUM_SEQS must be positive integers" >&2
  exit 2
fi
if [[ "${#selected_gpus[@]}" -ne "${dp_size}" ]]; then
  echo "GPU_LIST contains ${#selected_gpus[@]} GPUs but DP_SIZE=${dp_size}" >&2
  exit 2
fi
if [[ ! -d "${model}" ]]; then
  echo "Checkpoint does not exist: ${model}" >&2
  exit 2
fi

for index in "${selected_gpus[@]}"; do
  if [[ ! "${index}" =~ ^[0-9]+$ ]] || [[ ! -e "/dev/nvidia${index}" ]]; then
    echo "Invalid or unavailable GPU index: ${index}" >&2
    exit 2
  fi
  used_mib=$(
    nvidia-smi --id="${index}" --query-gpu=memory.used \
      --format=csv,noheader,nounits | tr -d ' '
  )
  if [[ "${allow_busy_gpus}" != 1 ]] && ((used_mib > 1024)); then
    echo "GPU ${index} already uses ${used_mib} MiB; refusing to overlap" >&2
    exit 2
  fi
done

mkdir -p "${result_dir}"
docker rm -f "${name}" >/dev/null 2>&1 || true

# Nested Docker does not have the NVIDIA container runtime. Preserve physical
# GPU numbering by mapping every device node, then restrict CUDA explicitly.
device_args=()
for device in /dev/nvidia[0-9]*; do
  device_args+=(--device "${device}")
done
for device in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools \
  /dev/nvidia-modeset; do
  [[ -e "${device}" ]] && device_args+=(--device "${device}")
done

# DeepEP needs the RDMA character devices in addition to the GPU devices.
# The outer Pod may expose only the HCAs granted by its RDMA resource, so map
# exactly the device nodes that are visible here instead of using --privileged.
rdma_args=()
if [[ -d /dev/infiniband ]]; then
  while IFS= read -r -d '' device; do
    rdma_args+=(--device "${device}")
  done < <(find /dev/infiniband -maxdepth 1 -type c -print0)
fi

library_args=()
nvml_lib=$(readlink -f /lib/x86_64-linux-gnu/libnvidia-ml.so.1 || true)
if [[ -n "${nvml_lib}" ]] && [[ -f "${nvml_lib}" ]]; then
  # The image already searches /usr/local/nvidia/lib64. Mount NVML there so
  # we do not need to replace its LD_LIBRARY_PATH.
  library_args+=(-v "${nvml_lib}:/usr/local/nvidia/lib64/libnvidia-ml.so.1:ro")
fi
nvidia_smi=$(command -v nvidia-smi || true)
if [[ -n "${nvidia_smi}" ]] && [[ -x "${nvidia_smi}" ]]; then
  library_args+=(-v "${nvidia_smi}:/usr/local/bin/nvidia-smi:ro")
fi

# Do not pass LD_LIBRARY_PATH here. The image-provided value keeps the
# DeepEP-matched pip NVSHMEM ahead of the older CUDA toolkit copy.
docker run -d \
  --name "${name}" \
  --network host \
  --ipc host \
  --ulimit memlock=-1:-1 \
  "${device_args[@]}" \
  "${rdma_args[@]}" \
  "${library_args[@]}" \
  -e NVIDIA_VISIBLE_DEVICES="${gpu_list}" \
  -e CUDA_VISIBLE_DEVICES="${gpu_list}" \
  -e LD_PRELOAD=/usr/local/cuda/compat/lib.real/libcuda.so.1 \
  -e MODEL="${model}" \
  -e MODEL_NAME=yoco-v2-long \
  -e DP_SIZE="${dp_size}" \
  -e ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-auto}" \
  -e ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-allgather_reducescatter}" \
  -e VLLM_DEEPEP_BUFFER_SIZE_MB="${VLLM_DEEPEP_BUFFER_SIZE_MB:-1024}" \
  -e NVSHMEM_QP_DEPTH="${NVSHMEM_QP_DEPTH:-auto}" \
  -e PORT="${port}" \
  -e ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASHINFER}" \
  -e ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}" \
  -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-auto}" \
  -e MAX_NUM_SEQS="${max_num_seqs}" \
  -v "${model}:${model}:ro" \
  -v "${result_dir}:/results" \
  "${image}" \
  bash tools/yoco_serving/serve_long_context.sh

for _ in $(seq 1 "${HEALTH_RETRIES:-240}"); do
  if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null; then
    echo "${name} healthy: GPUs=${gpu_list}, DP=${dp_size}, port=${port}"
    exit 0
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "${name}")" != true ]]; then
    docker logs "${name}"
    exit 1
  fi
  sleep 5
done

echo "Timed out waiting for ${name}" >&2
docker logs --tail 200 "${name}" >&2
exit 1
