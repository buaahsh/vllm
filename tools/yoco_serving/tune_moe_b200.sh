#!/usr/bin/env bash
set -euo pipefail

model="${MODEL:?Set MODEL to the YOCO checkpoint path}"
repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
config_dir="${repo_root}/vllm/model_executor/layers/fused_moe/configs"
config_name="E=128,N=1280,device_name=NVIDIA_B200.json"
batch_sizes=(1 2 4 8 16 32 128 1024 2843 3899 8192 32768)

cd "${repo_root}"

python benchmarks/kernels/benchmark_moe.py \
  --model "${model}" \
  --tp-size 1 \
  --trust-remote-code \
  --batch-size "${batch_sizes[@]}"

python benchmarks/kernels/benchmark_moe.py \
  --model "${model}" \
  --tp-size 1 \
  --trust-remote-code \
  --batch-size "${batch_sizes[@]}" \
  --tune \
  --tune-neighbor-configs \
  --tune-neighbor-count 3 \
  --save-dir "${config_dir}"

# Benchmark the generated file using the installed runtime path that vLLM's
# config loader resolves. The repository copy remains the source artifact.
runtime_config="/workspace/vllm/vllm/model_executor/layers/fused_moe/configs/${config_name}"
if [[ "$(readlink -f "${config_dir}/${config_name}")" != "$(readlink -f "${runtime_config}")" ]]; then
  cp "${config_dir}/${config_name}" "${runtime_config}"
fi

python benchmarks/kernels/benchmark_moe.py \
  --model "${model}" \
  --tp-size 1 \
  --trust-remote-code \
  --batch-size "${batch_sizes[@]}"
