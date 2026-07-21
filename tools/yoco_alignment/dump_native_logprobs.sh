#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"/data/yanqi/model_ckpt"}

NATIVE_PYTHON_BIN=${NATIVE_PYTHON_BIN:-${PYTHON_BIN:-"${REPO_ROOT}/../../.venv-yoco-native/bin/python"}}
MODEL=${MODEL:-"${CHECKPOINT_ROOT}/0000-6000-hf"}
NATIVE_CHECKPOINT=${NATIVE_CHECKPOINT:-"${CHECKPOINT_ROOT}/0000-6000-merged"}
LLM_TRAIN_DIR=${LLM_TRAIN_DIR:-"/data/yanqi/yanqi/yoco_mxfp8/llm-train"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/../logs/yoco_alignment_results"}

ATTENTION_VERSION=${ATTENTION_VERSION:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
PROMPT_SUITE=${PROMPT_SUITE:-mixed5}
PROMPT_LIMIT=${PROMPT_LIMIT:-5}
NATIVE_QUANT_MODE=${NATIVE_QUANT_MODE:-bfloat16}
NATIVE_QUANT_BLOCK_SIZE=${NATIVE_QUANT_BLOCK_SIZE:-}
NATIVE_USE_TORCH_FP8_QUANT=${NATIVE_USE_TORCH_FP8_QUANT:-}
SEED=${SEED:-0}

case "${ATTENTION_VERSION,,}" in
    2|fa2)
        ATTENTION_VERSION=2
        ATTENTION_TAG=fa2
        NATIVE_ATTENTION_ARGS=(
            --native-local-attention
            --native-require-transformer-engine
        )
        ;;
    4|fa4)
        ATTENTION_VERSION=4
        ATTENTION_TAG=fa4
        NATIVE_ATTENTION_ARGS=(
            --native-local-attention
            --native-require-transformer-engine
            --native-use-cute
            --native-no-kv-cache
        )
        ;;
    *)
        echo "ATTENTION_VERSION must be 2, 4, fa2, or fa4." >&2
        exit 2
        ;;
esac

if [[ ${NATIVE_NPROC:-1} != 1 ]]; then
    echo "Native alignment uses local attention and requires NATIVE_NPROC=1." >&2
    exit 2
fi

case "${NATIVE_QUANT_MODE}" in
    bfloat16)
        NATIVE_USE_TORCH_FP8_QUANT=${NATIVE_USE_TORCH_FP8_QUANT:-0}
        NATIVE_OUT=${NATIVE_OUT:-"${OUTPUT_DIR}/native_bf16_${ATTENTION_TAG}.pt"}
        ;;
    mxfp8)
        NATIVE_QUANT_BLOCK_SIZE=${NATIVE_QUANT_BLOCK_SIZE:-128}
        NATIVE_USE_TORCH_FP8_QUANT=${NATIVE_USE_TORCH_FP8_QUANT:-1}
        NATIVE_OUT=${NATIVE_OUT:-"${OUTPUT_DIR}/native_mxfp8_${ATTENTION_TAG}.pt"}
        ;;
    *)
        echo "NATIVE_QUANT_MODE must be bfloat16 or mxfp8." >&2
        exit 2
        ;;
esac

mkdir -p "${OUTPUT_DIR}"

if [[ ! -x "${NATIVE_PYTHON_BIN}" ]]; then
    echo "Native Python is not executable: ${NATIVE_PYTHON_BIN}" >&2
    exit 2
fi
if ! "${NATIVE_PYTHON_BIN}" -c '
import deep_gemm
for name in (
    "fp8_fp4_gemm_nt",
    "m_grouped_fp8_fp4_gemm_nt_contiguous",
    "m_grouped_bf16_gemm_nt_contiguous",
):
    getattr(deep_gemm, name)
' >/dev/null 2>&1; then
    echo "Native alignment requires the pinned external DeepGEMM package in ${NATIVE_PYTHON_BIN}." >&2
    exit 2
fi
if ! "${NATIVE_PYTHON_BIN}" -c '
from transformer_engine.pytorch.permutation import (
    moe_permute_and_pad_with_probs,
    moe_sort_chunks_by_index,
    moe_sort_chunks_by_index_with_probs,
    moe_unpermute,
)
' >/dev/null 2>&1; then
    echo "Native alignment requires NVIDIA TransformerEngine MoE APIs in ${NATIVE_PYTHON_BIN}." >&2
    exit 2
fi
if [[ "${ATTENTION_VERSION}" == 2 ]] && ! "${NATIVE_PYTHON_BIN}" -c '
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
' >/dev/null 2>&1; then
    echo "Native FA2 requires flash-attn in ${NATIVE_PYTHON_BIN}." >&2
    exit 2
fi
if [[ "${ATTENTION_VERSION}" == 4 ]] && ! "${NATIVE_PYTHON_BIN}" -c '
import runpy
namespace = runpy.run_path("'"${SCRIPT_DIR}/logprob_kl.py"'")
namespace["_import_fa4_varlen_func"]()
' >/dev/null 2>&1; then
    echo "Native FA4 requires flash-attn-4 in ${NATIVE_PYTHON_BIN}." >&2
    exit 2
fi
if [[ ! -d "${NATIVE_CHECKPOINT}" ]]; then
    echo "Native checkpoint directory is unavailable: ${NATIVE_CHECKPOINT}" >&2
    echo "Mount the checkpoint or set NATIVE_CHECKPOINT to a local merged checkpoint." >&2
    exit 1
fi
if [[ ! -f "${NATIVE_CHECKPOINT}/metadata.json" ]]; then
    echo "Missing native checkpoint metadata: ${NATIVE_CHECKPOINT}/metadata.json" >&2
    exit 1
fi
if [[ ! -f "${NATIVE_CHECKPOINT}/model_state_rank_0.pth" ]]; then
    echo "Missing native checkpoint weights: ${NATIVE_CHECKPOINT}/model_state_rank_0.pth" >&2
    exit 1
fi
if [[ ! -f "${LLM_TRAIN_DIR}/llm/arch/model.py" ]]; then
    echo "Missing llm-train model source: ${LLM_TRAIN_DIR}/llm/arch/model.py" >&2
    exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f "${NATIVE_CHECKPOINT}/metadata.json" ]]; then
    echo "Native checkpoint metadata not found: ${NATIVE_CHECKPOINT}/metadata.json" >&2
    echo "Set NATIVE_CHECKPOINT to an accessible merged llm-train checkpoint." >&2
    exit 2
fi

if [[ ! -f "${NATIVE_CHECKPOINT}/model_state_rank_0.pth" ]]; then
    echo "Native checkpoint weights not found: ${NATIVE_CHECKPOINT}/model_state_rank_0.pth" >&2
    echo "Set NATIVE_CHECKPOINT to an accessible merged llm-train checkpoint." >&2
    exit 2
fi

NATIVE_QUANT_ARGS=(--native-quant-mode "${NATIVE_QUANT_MODE}")
if [[ -n "${NATIVE_QUANT_BLOCK_SIZE}" ]]; then
    NATIVE_QUANT_ARGS+=(--native-quant-block-size "${NATIVE_QUANT_BLOCK_SIZE}")
fi
if ((NATIVE_USE_TORCH_FP8_QUANT)); then
    NATIVE_QUANT_ARGS+=(--native-use-torch-fp8-quant)
fi

"${NATIVE_PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc-per-node 1 \
    "${SCRIPT_DIR}/logprob_kl.py" native \
    --model "${MODEL}" \
    --native-checkpoint "${NATIVE_CHECKPOINT}" \
    --llm-train-dir "${LLM_TRAIN_DIR}" \
    --native-dtype bfloat16 \
    --out "${NATIVE_OUT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --seed "${SEED}" \
    --prompt-suite "${PROMPT_SUITE}" \
    --prompt-limit "${PROMPT_LIMIT}" \
    "${NATIVE_ATTENTION_ARGS[@]}" \
    "${NATIVE_QUANT_ARGS[@]}"

echo "Native ${NATIVE_QUANT_MODE} ${ATTENTION_TAG} logprobs saved to ${NATIVE_OUT}"