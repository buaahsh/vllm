#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"/data/yanqi/model_ckpt"}

VLLM_PYTHON_BIN=${VLLM_PYTHON_BIN:-${PYTHON_BIN:-"${REPO_ROOT}/../../.venv-yoco-mxfp8/bin/python"}}
MODEL=${MODEL:-"${CHECKPOINT_ROOT}/0000-6000-hf"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/../logs/yoco_alignment_results"}

ATTENTION_VERSION=${ATTENTION_VERSION:-2}
GPU_COUNTS=${GPU_COUNTS:-"1 4 8"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
PROMPT_SUITE=${PROMPT_SUITE:-mixed5}
PROMPT_LIMIT=${PROMPT_LIMIT:-5}
MOE_BACKEND=${MOE_BACKEND:-triton}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
SEED=${SEED:-0}

case "${ATTENTION_VERSION,,}" in
    2|fa2)
        ATTENTION_VERSION=2
        ATTENTION_TAG=fa2
        ;;
    4|fa4)
        ATTENTION_VERSION=4
        ATTENTION_TAG=fa4
        ;;
    *)
        echo "ATTENTION_VERSION must be 2, 4, fa2, or fa4." >&2
        exit 2
        ;;
esac

if [[ ! -x "${VLLM_PYTHON_BIN}" ]]; then
    echo "vLLM Python is not executable: ${VLLM_PYTHON_BIN}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

VLLM_EXTRA_ARGS=()
VLLM_EXTRA_ARGS+=(--attention-backend FLASH_ATTN)
VLLM_EXTRA_ARGS+=(--flash-attn-version "${ATTENTION_VERSION}")
if ((ENFORCE_EAGER)); then
    VLLM_EXTRA_ARGS+=(--enforce-eager)
fi

for NUM_GPUS in ${GPU_COUNTS}; do
    BF16_OUT="${OUTPUT_DIR}/vllm_bf16_${ATTENTION_TAG}_dp${NUM_GPUS}_ep${NUM_GPUS}.pt"
    echo "Running BF16 YOCO ${ATTENTION_TAG} with DP=${NUM_GPUS}, EP=${NUM_GPUS}, TP=1"
    "${VLLM_PYTHON_BIN}" -m torch.distributed.run \
        --standalone \
        --nproc-per-node "${NUM_GPUS}" \
        "${SCRIPT_DIR}/logprob_kl.py" vllm \
        --model "${MODEL}" \
        --dtype bfloat16 \
        --tensor-parallel-size 1 \
        --data-parallel-size "${NUM_GPUS}" \
        --enable-expert-parallel \
        --distributed-executor-backend external_launcher \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-logprobs -1 \
        --seed "${SEED}" \
        --out "${BF16_OUT}" \
        --prompt-suite "${PROMPT_SUITE}" \
        --prompt-limit "${PROMPT_LIMIT}" \
        --moe-backend "${MOE_BACKEND}" \
        "${VLLM_EXTRA_ARGS[@]}"
    echo "BF16 YOCO ${ATTENTION_TAG} logprobs saved to ${BF16_OUT}"
done