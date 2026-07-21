#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

COMPARE_PYTHON_BIN=${COMPARE_PYTHON_BIN:-${VLLM_PYTHON_BIN:-${PYTHON_BIN:-"${REPO_ROOT}/../../.venv-yoco-mxfp8/bin/python"}}}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/../logs/yoco_alignment_results"}
ATTENTION_VERSION=${ATTENTION_VERSION:-2}
GPU_COUNTS=${GPU_COUNTS:-"1 4 8"}
TOP_K=${TOP_K:-20}

case "${ATTENTION_VERSION,,}" in
    2|fa2)
        ATTENTION_TAG=fa2
        ;;
    4|fa4)
        ATTENTION_TAG=fa4
        ;;
    *)
        echo "ATTENTION_VERSION must be 2, 4, fa2, or fa4." >&2
        exit 2
        ;;
esac

NATIVE_OUT=${NATIVE_OUT:-"${OUTPUT_DIR}/native_mxfp8_${ATTENTION_TAG}.pt"}

if [[ ! -x "${COMPARE_PYTHON_BIN}" ]]; then
    echo "Comparison Python is not executable: ${COMPARE_PYTHON_BIN}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

for NUM_GPUS in ${GPU_COUNTS}; do
    FP8_OUT="${OUTPUT_DIR}/vllm_fp8_per_block_${ATTENTION_TAG}_dp${NUM_GPUS}_ep${NUM_GPUS}.pt"
    FP8_COMPARE_OUT="${OUTPUT_DIR}/native_mxfp8_${ATTENTION_TAG}_vs_vllm_fp8_per_block_dp${NUM_GPUS}_ep${NUM_GPUS}.json"
    "${COMPARE_PYTHON_BIN}" "${SCRIPT_DIR}/logprob_kl.py" compare \
        --reference "${NATIVE_OUT}" \
        --candidate "${FP8_OUT}" \
        --out-json "${FP8_COMPARE_OUT}" \
        --top-k "${TOP_K}"
    echo "Native-MXFP8-vs-FP8-per-block ${ATTENTION_TAG} comparison saved to ${FP8_COMPARE_OUT}"
done