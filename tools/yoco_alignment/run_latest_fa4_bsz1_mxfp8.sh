#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PROBE="${SCRIPT_DIR}/logprob_kl.py"

MODEL=""
NATIVE_CHECKPOINT=""
OVERLAY_DIR=${YOCO_LATEST_FA4_OVERLAY:-"${REPO_ROOT}/../.latest-fa4-4.0.0b23"}
OUTPUT_DIR="${REPO_ROOT}/../logs/yoco_alignment_results/latest_fa4_4.0.0b23/bsz1_mxfp8"
LLM_TRAIN_DIR="${REPO_ROOT}/../llm-train"
NATIVE_PYTHON_BIN=${NATIVE_PYTHON_BIN:-"${REPO_ROOT}/../../.venv-yoco-native/bin/python"}
VLLM_PYTHON_BIN=${VLLM_PYTHON_BIN:-"${REPO_ROOT}/../../.venv-yoco-mxfp8/bin/python"}
COMPARE_PYTHON_BIN=${COMPARE_PYTHON_BIN:-"${VLLM_PYTHON_BIN}"}
NATIVE_GPU=0
VLLM_GPU=1
DRY_RUN=0
COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}'

usage() {
    cat <<'EOF'
Run batch-size-1 all-MXFP8 alignment with both Native and vLLM using the same
latest FA4 b23 source/runtime overlay. The five mixed5 prompts execute as five
separate one-request forwards. Native and vLLM run concurrently on two GPUs.

Usage:
  run_latest_fa4_bsz1_mxfp8.sh \
    --model MODEL --native-checkpoint DIR [options]

Options:
  --overlay-dir PATH       Output of setup_latest_fa4_overlay.sh.
  --output-dir PATH        Result directory.
  --llm-train-dir PATH     llm-train checkout.
  --native-python PATH     Native NVIDIA-stack Python.
  --vllm-python PATH       vLLM Python.
  --compare-python PATH    Comparison Python.
  --native-gpu ID          Physical GPU for Native (default: 0).
  --vllm-gpu ID            Physical GPU for vLLM (default: 1).
  --dry-run                Print commands without model execution.
  -h, --help               Show this help.
EOF
}

while (($#)); do
    case "$1" in
        --model) MODEL=${2:?"--model requires a value"}; shift 2 ;;
        --native-checkpoint) NATIVE_CHECKPOINT=${2:?"--native-checkpoint requires a value"}; shift 2 ;;
        --overlay-dir) OVERLAY_DIR=${2:?"--overlay-dir requires a value"}; shift 2 ;;
        --output-dir) OUTPUT_DIR=${2:?"--output-dir requires a value"}; shift 2 ;;
        --llm-train-dir) LLM_TRAIN_DIR=${2:?"--llm-train-dir requires a value"}; shift 2 ;;
        --native-python) NATIVE_PYTHON_BIN=${2:?"--native-python requires a value"}; shift 2 ;;
        --vllm-python) VLLM_PYTHON_BIN=${2:?"--vllm-python requires a value"}; shift 2 ;;
        --compare-python) COMPARE_PYTHON_BIN=${2:?"--compare-python requires a value"}; shift 2 ;;
        --native-gpu) NATIVE_GPU=${2:?"--native-gpu requires a value"}; shift 2 ;;
        --vllm-gpu) VLLM_GPU=${2:?"--vllm-gpu requires a value"}; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${MODEL}" || -z "${NATIVE_CHECKPOINT}" ]]; then
    echo "--model and --native-checkpoint are required." >&2
    exit 2
fi
if [[ "${NATIVE_GPU}" == "${VLLM_GPU}" ]]; then
    echo "Native and vLLM require different GPUs for concurrent execution." >&2
    exit 2
fi

SITE_PACKAGES="${OVERLAY_DIR}/site-packages"
CUTLASS_PACKAGES="${SITE_PACKAGES}/nvidia_cutlass_dsl/python_packages"
for path in \
    "${OVERLAY_DIR}/manifest.json" \
    "${CUTLASS_PACKAGES}/cutlass/__init__.py" \
    "${SITE_PACKAGES}/flash_attn/cute/interface.py"; do
    if [[ ! -f "${path}" ]]; then
        echo "Latest FA4 overlay is incomplete: ${path}" >&2
        exit 2
    fi
done
for python_bin in "${NATIVE_PYTHON_BIN}" "${VLLM_PYTHON_BIN}" "${COMPARE_PYTHON_BIN}"; do
    if [[ ! -x "${python_bin}" ]]; then
        echo "Python is not executable: ${python_bin}" >&2
        exit 2
    fi
done

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR=$(cd -- "${OUTPUT_DIR}" && pwd)
NATIVE_OUT="${OUTPUT_DIR}/native_mxfp8_fa4.pt"
VLLM_OUT="${OUTPUT_DIR}/vllm_fp8_per_block_fa4.pt"
COMPARE_OUT="${OUTPUT_DIR}/native_mxfp8_fa4_vs_vllm_fp8_per_block.json"
EXACTNESS_OUT="${OUTPUT_DIR}/exactness.json"
COMMON_PYTHONPATH="${CUTLASS_PACKAGES}:${SITE_PACKAGES}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

NATIVE_CMD=(
    "${NATIVE_PYTHON_BIN}" -m torch.distributed.run
    --standalone --nproc-per-node 1
    "${PROBE}" native
    --model "${MODEL}"
    --native-checkpoint "${NATIVE_CHECKPOINT}"
    --llm-train-dir "${LLM_TRAIN_DIR}"
    --native-dtype bfloat16
    --native-quant-mode mxfp8
    --native-quant-block-size 128
    --native-use-torch-fp8-quant
    --native-local-attention
    --native-require-transformer-engine
    --native-use-cute
    --native-no-kv-cache
    --force-fa-num-splits-one
    --fa4-source-root "${SITE_PACKAGES}"
    --fa4-profile default
    --fa4-pack-gqa auto
    --fa4-tile-mn 128x128
    --prompt-suite mixed5
    --batch-size 1
    --max-model-len 8192
    --seed 0
    --out "${NATIVE_OUT}"
)
VLLM_CMD=(
    "${VLLM_PYTHON_BIN}" "${PROBE}" vllm
    --model "${MODEL}"
    --out "${VLLM_OUT}"
    --prompt-suite mixed5
    --batch-size 1
    --dtype bfloat16
    --tensor-parallel-size 1
    --max-num-seqs 1
    --max-model-len 8192
    --max-num-batched-tokens 8192
    --gpu-memory-utilization 0.9
    --max-logprobs -1
    --quantization fp8_per_block
    --moe-backend deep_gemm
    --attention-backend FLASH_ATTN
    --flash-attn-version 4
    --force-fa-num-splits-one
    --no-v1-multiprocessing
    --no-enable-prefix-caching
    --log-iteration-details
    --compilation-config-json "${COMPILATION_CONFIG}"
    --fa4-source-root "${SITE_PACKAGES}"
    --fa4-profile default
    --fa4-pack-gqa auto
    --fa4-tile-mn 128x128
    --seed 0
)

print_command() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
}

echo "Latest FA4 b23 batch-1 output: ${OUTPUT_DIR}"
echo "Native GPU: ${NATIVE_GPU}; vLLM GPU: ${VLLM_GPU}"
print_command env CUDA_VISIBLE_DEVICES="${NATIVE_GPU}" PYTHONPATH="${COMMON_PYTHONPATH}" "${NATIVE_CMD[@]}"
print_command env CUDA_VISIBLE_DEVICES="${VLLM_GPU}" PYTHONPATH="${COMMON_PYTHONPATH}" "${VLLM_CMD[@]}"
if ((DRY_RUN)); then
    exit 0
fi

env CUDA_VISIBLE_DEVICES="${NATIVE_GPU}" PYTHONPATH="${COMMON_PYTHONPATH}" \
    "${NATIVE_CMD[@]}" >"${OUTPUT_DIR}/native.log" 2>&1 &
NATIVE_PID=$!
env CUDA_VISIBLE_DEVICES="${VLLM_GPU}" PYTHONPATH="${COMMON_PYTHONPATH}" \
    "${VLLM_CMD[@]}" >"${OUTPUT_DIR}/vllm.log" 2>&1 &
VLLM_PID=$!

STATUS=0
if ! wait "${NATIVE_PID}"; then
    echo "Native run failed; see ${OUTPUT_DIR}/native.log" >&2
    STATUS=1
fi
if ! wait "${VLLM_PID}"; then
    echo "vLLM run failed; see ${OUTPUT_DIR}/vllm.log" >&2
    STATUS=1
fi
if ((STATUS)); then
    exit "${STATUS}"
fi

"${COMPARE_PYTHON_BIN}" "${PROBE}" compare \
    --reference "${NATIVE_OUT}" \
    --candidate "${VLLM_OUT}" \
    --model "${MODEL}" \
    --out-json "${COMPARE_OUT}" \
    --top-k 20

"${COMPARE_PYTHON_BIN}" - "${NATIVE_OUT}" "${VLLM_OUT}" "${EXACTNESS_OUT}" <<'PY'
import json
import sys
from pathlib import Path

import torch

native = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
vllm = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
native_rows = {row["prompt"]["name"]: row for row in native["results"]}
vllm_rows = {row["prompt"]["name"]: row for row in vllm["results"]}
if native_rows.keys() != vllm_rows.keys():
    raise RuntimeError("Native/vLLM prompt sets differ")

rows = []
for name in native_rows:
    reference = native_rows[name]["logprobs"]
    candidate = vllm_rows[name]["logprobs"]
    difference = (candidate.float() - reference.float()).abs()
    rows.append(
        {
            "prompt": name,
            "exact": bool(torch.equal(reference, candidate)),
            "max_abs_logprob_diff": float(difference.max()),
            "mean_abs_logprob_diff": float(difference.mean()),
        }
    )
report = {
    "all_prompts_exact": all(row["exact"] for row in rows),
    "num_exact_prompts": sum(row["exact"] for row in rows),
    "num_prompts": len(rows),
    "prompts": rows,
}
Path(sys.argv[3]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY

echo "Latest FA4 b23 batch-1 experiment complete: ${OUTPUT_DIR}"