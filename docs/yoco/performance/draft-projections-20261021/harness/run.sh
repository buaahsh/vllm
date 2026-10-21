#!/bin/bash
set -euo pipefail
cd /home/fhb/yoco-draft-projections-20261021
old=/home/fhb/yoco-mega-moe-20261020
runtime=/home/fhb/yoco-fast-prefill-overhead-20261021/integration-venv/bin/python
export CUDA_VISIBLE_DEVICES=GPU-ef7d6ee6-0fce-4e0b-b14c-87bd8a36c5fe
exec 9>"$old/gpu6.lock"
flock -n 9
used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits)
[[ $used -lt 100 ]]
benchmark_status=0
trap 'benchmark_status=$?; echo "$benchmark_status" > run.rc; nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > gpu-after.csv' EXIT
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$old/fullmodel-venv/lib/python3.12/site-packages/torch/lib:$old/fullmodel-lib:$old/build-deps/prefix/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export CPATH=$old/build-deps/prefix/usr/include:/usr/local/cuda/include
export OMP_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=2
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_BATCH_INVARIANT=0
export VLLM_YOCO_BF16_SAMPLING=0
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_CACHE_ROOT=$old/fullmodel-cache
export TRITON_CACHE_DIR=$old/fullmodel-triton-cache
export TORCHINDUCTOR_CACHE_DIR=$old/fullmodel-inductor-cache
export DG_JIT_CACHE_DIR=$old/fullmodel-dg-cache
export NCCL_IB_DISABLE=1
for version in rope gate-up; do
 export PYTHONPATH=$PWD/$version:$PWD
 "$runtime" -m pytest --confcutdir="$version/tests/model_executor" -q "$version/tests/model_executor/test_yoco_prefill.py" "$version/tests/model_executor/test_yoco_shared_kv.py" > "results/b200-model-$version.log" 2>&1
done
export PYTHONPATH=$PWD/parent:$PWD
timeout --signal=TERM --kill-after=30 2400 "$runtime" -u paired.py > results/paired.log 2>&1
for name in parent-a rope-r gate-up-g parent-b; do
 case "$name" in
  parent-*) version=parent;;
  rope-*) version=rope;;
  gate-up-*) version=gate-up;;
 esac
 export PYTHONPATH=$PWD/$version:$PWD
 timeout --signal=TERM --kill-after=30 1800 "$runtime" -u -m tools.yoco_alignment.benchmark_mega_moe --backend hybrid --output "results/$name" --max-model-len 12288 --tokens 128 --quality-tokens 64 --trials 3 --extra-prompt-tokens 8193 --concurrencies 1 4 8 --synchronous-engine --fast-prefill > "results/$name.log" 2>&1
done
