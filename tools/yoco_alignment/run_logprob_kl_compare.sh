#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PROBE="${SCRIPT_DIR}/logprob_kl.py"

usage() {
    cat <<'EOF'
Run YOCO native-vs-vLLM full-vocabulary alignment for BF16 and FP8-per-block.

Usage:
  run_logprob_kl_compare.sh --model MODEL --native-checkpoint DIR [options]

Required:
  --model PATH                 Converted HF/vLLM model and tokenizer path.
  --native-checkpoint PATH     Merged llm-train native checkpoint directory.

Options:
    --output-dir PATH            Output directory (default: ../logs/yoco_alignment_results).
    --llm-train-dir PATH         llm-train checkout (default: /data/yanqi/yanqi/yoco_mxfp8/llm-train).
    --native-python PATH         Native Python (default: .venv-yoco-native).
    --native-fa4-source NAME     installed or vllm-vendored (default: installed).
    --native-fa4-overlay PATH    Isolated dependency overlay directory.
    --vllm-python PATH           vLLM Python (default: .venv-yoco-mxfp8).
    --compare-python PATH        Comparison Python (default: vLLM Python).
    --python PATH                Use one Python for all stages (compatibility alias).
    --attention-version VERSION  2, 4, fa2, or fa4 (default: 2).
    --compilation-config-json J  vLLM compilation config (default: FULL_DECODE_ONLY).
    --prompt-suite NAME          mixed5, mixed16, or default (default: mixed5).
  --prompt-index N             Select one prompt when supported (default: 0).
  --prompt-limit N             Limit the number of prompts.
    --batch-size N               Prompts per Native/vLLM forward (default: 1).
  --max-model-len N            Maximum model length (default: 8192).
  --tensor-parallel-size N     vLLM tensor parallel size (default: 1).
  --gpu-memory-utilization F   vLLM GPU memory utilization (default: 0.9).
    --bf16-moe-backend NAME      BF16 vLLM MoE backend (default: triton).
    --fp8-moe-backend NAME       FP8 vLLM MoE backend (default: deep_gemm).
    --moe-backend NAME           Set both backends to NAME (compatibility alias).
    --variants LIST              Variants: bf16, fp8, or bf16,fp8
                                                             (default: bf16,fp8).
  --seed N                     Random seed (default: 0).
  --top-k N                    Tokens included in comparison JSON (default: 20).
  --kv-sharing-fast-prefill    Enable fast prefill in both vLLM runs.
    --enforce-eager              Use eager execution instead of the vLLM default.
    --compiled                   Use the default non-eager path (compatibility alias).
    --dry-run                    Print commands without executing them.
  -h, --help                   Show this help.

Outputs (subject to --variants):
    native_bf16_fa{2,4}.pt
    native_mxfp8_fa{2,4}.pt
    vllm_bf16_fa{2,4}.pt
    vllm_fp8_per_block_fa{2,4}.pt
    native_bf16_fa{2,4}_vs_vllm.json
    native_mxfp8_fa{2,4}_vs_vllm_fp8_per_block.json

Native execution is single-rank and bypasses NNScaler ring attention.
When TransformerEngine is unavailable, the probe pads MoE rows to DeepGEMM's
128-row alignment and unpermutes them deterministically.
The vLLM runner uses FULL_DECODE_ONLY by default: prefill is eager and
single-token decode is graph captured. FA4 runs require the vendored CuTe
interface to import successfully. Use --enforce-eager for a separate fully
eager validation.
EOF
}

MODEL=""
NATIVE_CHECKPOINT=""
OUTPUT_DIR="${REPO_ROOT}/../logs/yoco_alignment_results"
LLM_TRAIN_DIR="/data/yanqi/yanqi/yoco_mxfp8/llm-train"
NATIVE_PYTHON_BIN=${NATIVE_PYTHON_BIN:-"${REPO_ROOT}/../../.venv-yoco-native/bin/python"}
NATIVE_FA4_SOURCE="installed"
NATIVE_FA4_OVERLAY=${NATIVE_VLLM_FA4_OVERLAY:-"${REPO_ROOT}/../.native-vllm-fa4-overlay"}
VLLM_PYTHON_BIN=${VLLM_PYTHON_BIN:-"${REPO_ROOT}/../../.venv-yoco-mxfp8/bin/python"}
COMPARE_PYTHON_BIN=${COMPARE_PYTHON_BIN:-}
ATTENTION_VERSION=2
COMPILATION_CONFIG_JSON='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}'
PROMPT_SUITE="mixed5"
PROMPT_INDEX=0
PROMPT_LIMIT=""
BATCH_SIZE=1
MAX_MODEL_LEN=8192
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9
BF16_MOE_BACKEND="triton"
FP8_MOE_BACKEND="deep_gemm"
VARIANTS="bf16,fp8"
SEED=0
TOP_K=20
KV_SHARING_FAST_PREFILL=0
ENFORCE_EAGER=0
DRY_RUN=0

while (($#)); do
    case "$1" in
        --model)
            MODEL=${2:?"--model requires a value"}
            shift 2
            ;;
        --native-checkpoint)
            NATIVE_CHECKPOINT=${2:?"--native-checkpoint requires a value"}
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR=${2:?"--output-dir requires a value"}
            shift 2
            ;;
        --llm-train-dir)
            LLM_TRAIN_DIR=${2:?"--llm-train-dir requires a value"}
            shift 2
            ;;
        --native-python)
            NATIVE_PYTHON_BIN=${2:?"--native-python requires a value"}
            shift 2
            ;;
        --native-fa4-source)
            NATIVE_FA4_SOURCE=${2:?"--native-fa4-source requires a value"}
            shift 2
            ;;
        --native-fa4-overlay)
            NATIVE_FA4_OVERLAY=${2:?"--native-fa4-overlay requires a value"}
            shift 2
            ;;
        --vllm-python)
            VLLM_PYTHON_BIN=${2:?"--vllm-python requires a value"}
            shift 2
            ;;
        --compare-python)
            COMPARE_PYTHON_BIN=${2:?"--compare-python requires a value"}
            shift 2
            ;;
        --python)
            NATIVE_PYTHON_BIN=${2:?"--python requires a value"}
            VLLM_PYTHON_BIN=${NATIVE_PYTHON_BIN}
            COMPARE_PYTHON_BIN=${NATIVE_PYTHON_BIN}
            shift 2
            ;;
        --attention-version)
            ATTENTION_VERSION=${2:?"--attention-version requires a value"}
            shift 2
            ;;
        --compilation-config-json)
            COMPILATION_CONFIG_JSON=${2:?"--compilation-config-json requires a value"}
            shift 2
            ;;
        --prompt-suite)
            PROMPT_SUITE=${2:?"--prompt-suite requires a value"}
            shift 2
            ;;
        --prompt-index)
            PROMPT_INDEX=${2:?"--prompt-index requires a value"}
            shift 2
            ;;
        --prompt-limit)
            PROMPT_LIMIT=${2:?"--prompt-limit requires a value"}
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE=${2:?"--batch-size requires a value"}
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN=${2:?"--max-model-len requires a value"}
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE=${2:?"--tensor-parallel-size requires a value"}
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION=${2:?"--gpu-memory-utilization requires a value"}
            shift 2
            ;;
        --moe-backend)
            BF16_MOE_BACKEND=${2:?"--moe-backend requires a value"}
            FP8_MOE_BACKEND=${BF16_MOE_BACKEND}
            shift 2
            ;;
        --bf16-moe-backend)
            BF16_MOE_BACKEND=${2:?"--bf16-moe-backend requires a value"}
            shift 2
            ;;
        --fp8-moe-backend)
            FP8_MOE_BACKEND=${2:?"--fp8-moe-backend requires a value"}
            shift 2
            ;;
        --variants)
            VARIANTS=${2:?"--variants requires a value"}
            shift 2
            ;;
        --seed)
            SEED=${2:?"--seed requires a value"}
            shift 2
            ;;
        --top-k)
            TOP_K=${2:?"--top-k requires a value"}
            shift 2
            ;;
        --kv-sharing-fast-prefill)
            KV_SHARING_FAST_PREFILL=1
            shift
            ;;
        --enforce-eager)
            ENFORCE_EAGER=1
            shift
            ;;
        --compiled)
            ENFORCE_EAGER=0
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${MODEL}" || -z "${NATIVE_CHECKPOINT}" ]]; then
    echo "Both --model and --native-checkpoint are required." >&2
    usage >&2
    exit 2
fi

if [[ "${PROMPT_SUITE}" != "mixed5" && "${PROMPT_SUITE}" != "mixed16" && "${PROMPT_SUITE}" != "default" ]]; then
    echo "--prompt-suite must be 'mixed5', 'mixed16', or 'default'." >&2
    exit 2
fi
if ((BATCH_SIZE < 1)); then
    echo "--batch-size must be positive." >&2
    exit 2
fi
if [[ "${NATIVE_FA4_SOURCE}" != "installed" && "${NATIVE_FA4_SOURCE}" != "vllm-vendored" ]]; then
    echo "--native-fa4-source must be 'installed' or 'vllm-vendored'." >&2
    exit 2
fi

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
            --native-fa4-source "${NATIVE_FA4_SOURCE}"
        )
        ;;
    *)
        echo "--attention-version must be 2, 4, fa2, or fa4." >&2
        exit 2
        ;;
esac

NATIVE_ENV=(env)
if [[ "${NATIVE_FA4_SOURCE}" == "vllm-vendored" ]]; then
    if [[ "${ATTENTION_VERSION}" != 4 ]]; then
        echo "--native-fa4-source vllm-vendored requires --attention-version 4." >&2
        exit 2
    fi
    NATIVE_FA4_SITE_PACKAGES="${NATIVE_FA4_OVERLAY}/site-packages"
    if [[ ! -d "${NATIVE_FA4_SITE_PACKAGES}" ]]; then
        echo "Native FA4 overlay not found: ${NATIVE_FA4_SITE_PACKAGES}" >&2
        echo "Run tools/yoco_alignment/setup_native_vllm_fa4_overlay.sh first." >&2
        exit 2
    fi
    NATIVE_ENV+=(
        "PYTHONPATH=${NATIVE_FA4_SITE_PACKAGES}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    )
fi

VLLM_ATTENTION_ARGS=(
    --attention-backend FLASH_ATTN
    --flash-attn-version "${ATTENTION_VERSION}"
    --force-fa-num-splits-one
)
NATIVE_ATTENTION_ARGS+=(--force-fa-num-splits-one)

RUN_BF16=0
RUN_FP8=0
case "${VARIANTS}" in
    bf16)
        RUN_BF16=1
        ;;
    fp8|fp8_per_block|mxfp8)
        RUN_FP8=1
        ;;
    bf16,fp8|fp8,bf16|bf16,fp8_per_block|fp8_per_block,bf16|\
    bf16,mxfp8|mxfp8,bf16|both)
        RUN_BF16=1
        RUN_FP8=1
        ;;
    *)
        echo "--variants must be bf16, fp8, bf16,fp8, or both." >&2
        exit 2
        ;;
esac

if [[ -z "${COMPARE_PYTHON_BIN}" ]]; then
    COMPARE_PYTHON_BIN=${VLLM_PYTHON_BIN}
fi

for PYTHON_ROLE in NATIVE VLLM COMPARE; do
    PYTHON_VARIABLE="${PYTHON_ROLE}_PYTHON_BIN"
    PYTHON_VALUE=${!PYTHON_VARIABLE}
    if [[ ! -x "${PYTHON_VALUE}" ]]; then
        echo "${PYTHON_ROLE,,} Python is not executable: ${PYTHON_VALUE}" >&2
        exit 2
    fi
done
if ! "${NATIVE_ENV[@]}" "${NATIVE_PYTHON_BIN}" -c '
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
if ! "${NATIVE_ENV[@]}" "${NATIVE_PYTHON_BIN}" -c '
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
if [[ "${ATTENTION_VERSION}" == 2 ]] && ! "${NATIVE_ENV[@]}" "${NATIVE_PYTHON_BIN}" -c '
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
' >/dev/null 2>&1; then
    echo "Native FA2 requires flash-attn in ${NATIVE_PYTHON_BIN}." >&2
    exit 2
fi
if [[ "${ATTENTION_VERSION}" == 4 ]] && ! "${NATIVE_ENV[@]}" "${NATIVE_PYTHON_BIN}" -c '
import runpy
namespace = runpy.run_path("'"${PROBE}"'")
namespace["_import_fa4_varlen_func"]("'"${NATIVE_FA4_SOURCE}"'")
' >/dev/null 2>&1; then
    echo "Native FA4 source '${NATIVE_FA4_SOURCE}' failed its import preflight." >&2
    exit 2
fi
if [[ "${ATTENTION_VERSION}" == 4 ]] && ! PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${VLLM_PYTHON_BIN}" -c '
from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported
assert callable(_flash_attn_fwd)
assert is_fa_version_supported(4)
' >/dev/null 2>&1; then
    echo "vLLM vendored FA4 is unavailable in ${VLLM_PYTHON_BIN}." >&2
    echo "Install requirements/cuda.txt (CUTLASS DSL 4.4.2 for this checkout)." >&2
    exit 2
fi
if [[ ! -f "${PROBE}" ]]; then
    echo "Probe script not found: ${PROBE}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR=$(cd -- "${OUTPUT_DIR}" && pwd)

NATIVE_OUT="${OUTPUT_DIR}/native_bf16_${ATTENTION_TAG}.pt"
NATIVE_MXFP8_OUT="${OUTPUT_DIR}/native_mxfp8_${ATTENTION_TAG}.pt"
BF16_OUT="${OUTPUT_DIR}/vllm_bf16_${ATTENTION_TAG}.pt"
FP8_OUT="${OUTPUT_DIR}/vllm_fp8_per_block_${ATTENTION_TAG}.pt"
BF16_COMPARE_OUT="${OUTPUT_DIR}/native_bf16_${ATTENTION_TAG}_vs_vllm.json"
FP8_COMPARE_OUT="${OUTPUT_DIR}/native_mxfp8_${ATTENTION_TAG}_vs_vllm_fp8_per_block.json"

run() {
    printf '\n+'
    printf ' %q' "$@"
    printf '\n'
    if ((DRY_RUN)); then
        return
    fi
    "$@"
}

PROMPT_ARGS=(
    --prompt-suite "${PROMPT_SUITE}"
    --prompt-index "${PROMPT_INDEX}"
    --batch-size "${BATCH_SIZE}"
)
if [[ -n "${PROMPT_LIMIT}" ]]; then
    PROMPT_ARGS+=(--prompt-limit "${PROMPT_LIMIT}")
fi

VLLM_EXTRA_ARGS=()
VLLM_EXTRA_ARGS+=("${VLLM_ATTENTION_ARGS[@]}")
if ((ENFORCE_EAGER)); then
    VLLM_EXTRA_ARGS+=(--enforce-eager)
fi
if ((KV_SHARING_FAST_PREFILL)); then
    VLLM_EXTRA_ARGS+=(--kv-sharing-fast-prefill)
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Writing YOCO ${ATTENTION_TAG} alignment artifacts to ${OUTPUT_DIR}"
echo "Native Python: ${NATIVE_PYTHON_BIN}"
echo "Native FA4 source: ${NATIVE_FA4_SOURCE}"
if [[ "${NATIVE_FA4_SOURCE}" == "vllm-vendored" ]]; then
    echo "Native FA4 overlay: ${NATIVE_FA4_OVERLAY}"
fi
echo "vLLM Python: ${VLLM_PYTHON_BIN}"
echo "Comparison Python: ${COMPARE_PYTHON_BIN}"

GENERATED_FILES=()

if ((RUN_BF16)); then
    run "${NATIVE_ENV[@]}" "${NATIVE_PYTHON_BIN}" -m torch.distributed.run \
        --standalone \
        --nproc-per-node 1 \
        "${PROBE}" native \
        --model "${MODEL}" \
        --native-checkpoint "${NATIVE_CHECKPOINT}" \
        --llm-train-dir "${LLM_TRAIN_DIR}" \
        --native-dtype bfloat16 \
        --native-quant-mode bfloat16 \
        --out "${NATIVE_OUT}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --seed "${SEED}" \
        "${NATIVE_ATTENTION_ARGS[@]}" \
        "${PROMPT_ARGS[@]}"
    GENERATED_FILES+=("${NATIVE_OUT}")
fi

if ((RUN_FP8)); then
    run "${NATIVE_ENV[@]}" "${NATIVE_PYTHON_BIN}" -m torch.distributed.run \
        --standalone \
        --nproc-per-node 1 \
        "${PROBE}" native \
        --model "${MODEL}" \
        --native-checkpoint "${NATIVE_CHECKPOINT}" \
        --llm-train-dir "${LLM_TRAIN_DIR}" \
        --native-dtype bfloat16 \
        --native-quant-mode mxfp8 \
        --native-quant-block-size 128 \
        --native-use-torch-fp8-quant \
        --out "${NATIVE_MXFP8_OUT}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --seed "${SEED}" \
        "${NATIVE_ATTENTION_ARGS[@]}" \
        "${PROMPT_ARGS[@]}"
    GENERATED_FILES+=("${NATIVE_MXFP8_OUT}")
fi

VLLM_COMMON_ARGS=(
    "${PROBE}" vllm
    --model "${MODEL}"
    --dtype bfloat16
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --max-num-seqs "${BATCH_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-logprobs -1
    --seed "${SEED}"
    --no-v1-multiprocessing
    --no-enable-prefix-caching
    --log-iteration-details
    --compilation-config-json "${COMPILATION_CONFIG_JSON}"
    "${PROMPT_ARGS[@]}"
    "${VLLM_EXTRA_ARGS[@]}"
)

if ((RUN_BF16)); then
    run "${VLLM_PYTHON_BIN}" "${VLLM_COMMON_ARGS[@]}" \
        --moe-backend "${BF16_MOE_BACKEND}" \
        --out "${BF16_OUT}"
    run "${COMPARE_PYTHON_BIN}" "${PROBE}" compare \
        --reference "${NATIVE_OUT}" \
        --candidate "${BF16_OUT}" \
        --out-json "${BF16_COMPARE_OUT}" \
        --model "${MODEL}" \
        --top-k "${TOP_K}"
    GENERATED_FILES+=("${BF16_OUT}" "${BF16_COMPARE_OUT}")
fi

if ((RUN_FP8)); then
    run "${VLLM_PYTHON_BIN}" "${VLLM_COMMON_ARGS[@]}" \
        --moe-backend "${FP8_MOE_BACKEND}" \
        --quantization fp8_per_block \
        --out "${FP8_OUT}"
    run "${COMPARE_PYTHON_BIN}" "${PROBE}" compare \
        --reference "${NATIVE_MXFP8_OUT}" \
        --candidate "${FP8_OUT}" \
        --out-json "${FP8_COMPARE_OUT}" \
        --model "${MODEL}" \
        --top-k "${TOP_K}"
    GENERATED_FILES+=("${FP8_OUT}" "${FP8_COMPARE_OUT}")
fi

echo
echo "Alignment run complete:"
printf '  %s\n' "${GENERATED_FILES[@]}"