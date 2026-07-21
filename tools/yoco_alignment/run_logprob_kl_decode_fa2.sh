#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PROBE="${SCRIPT_DIR}/logprob_kl_decode.py"

usage() {
    cat <<'EOF'
Run the YOCO FA2 decode-vs-teacher-forced-prefill alignment matrix.

Usage:
  run_logprob_kl_decode_fa2.sh --model MODEL --native-checkpoint DIR [options]

Required:
  --model PATH                 Converted HF/vLLM model and tokenizer.
  --native-checkpoint PATH     Merged llm-train checkpoint.

Options:
  --output-dir PATH            Default: ../logs/yoco_alignment_results/fa2_decode_mixed16
  --llm-train-dir PATH         Default: ../llm-train
  --native-python PATH         Default: .venv-yoco-native
  --vllm-python PATH           Default: .venv-yoco-mxfp8
  --compare-python PATH        Default: vLLM Python
  --variants LIST              bf16, mxfp8, or bf16,mxfp8 (default: both)
  --lengths LIST               Comma-separated decode lengths (default: 16,128)
  --stages LIST                vllm,native,compare (default: all)
  --batch-size N               Default: 16
  --prompt-suite NAME          Default: mixed16
  --max-model-len N            Default: 8192
  --tensor-parallel-size N     Default: 1
  --gpu-memory-utilization F   Default: 0.9
  --bf16-moe-backend NAME      Default: triton
  --mxfp8-moe-backend NAME     Default: deep_gemm
  --seed N                     Default: 0
  --top-k N                    Comparison top rows (default: 20)
  --dry-run                    Print commands without executing them
  -h, --help                   Show this help

Each length stores full-vocabulary rows at positions 1,2,4,8 and every 16th
position through the requested length. If the length is not a multiple of 16,
the final position is included as well. EOS may end individual responses early.

Run the 16-token smoke stage before the 128-token acceptance stage. The vLLM
stage uses one in-process EngineCore, prefix caching disabled, FA2 num_splits=1,
and FULL_DECODE_ONLY. Native replays the exact rollout token IDs in one packed
teacher-forced prefill and requires real TransformerEngine MoE APIs.
EOF
}

MODEL=""
NATIVE_CHECKPOINT=""
OUTPUT_DIR="${REPO_ROOT}/../logs/yoco_alignment_results/fa2_decode_mixed16"
LLM_TRAIN_DIR="${REPO_ROOT}/../llm-train"
NATIVE_PYTHON_BIN=${NATIVE_PYTHON_BIN:-"${REPO_ROOT}/../../.venv-yoco-native/bin/python"}
VLLM_PYTHON_BIN=${VLLM_PYTHON_BIN:-"${REPO_ROOT}/../../.venv-yoco-mxfp8/bin/python"}
COMPARE_PYTHON_BIN=${COMPARE_PYTHON_BIN:-}
VARIANTS="bf16,mxfp8"
LENGTHS="16,128"
STAGES="vllm,native,compare"
BATCH_SIZE=16
PROMPT_SUITE="mixed16"
MAX_MODEL_LEN=8192
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9
BF16_MOE_BACKEND="triton"
MXFP8_MOE_BACKEND="deep_gemm"
SEED=0
TOP_K=20
DRY_RUN=0
COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}'

while (($#)); do
    case "$1" in
        --model) MODEL=${2:?"--model requires a value"}; shift 2 ;;
        --native-checkpoint) NATIVE_CHECKPOINT=${2:?"--native-checkpoint requires a value"}; shift 2 ;;
        --output-dir) OUTPUT_DIR=${2:?"--output-dir requires a value"}; shift 2 ;;
        --llm-train-dir) LLM_TRAIN_DIR=${2:?"--llm-train-dir requires a value"}; shift 2 ;;
        --native-python) NATIVE_PYTHON_BIN=${2:?"--native-python requires a value"}; shift 2 ;;
        --vllm-python) VLLM_PYTHON_BIN=${2:?"--vllm-python requires a value"}; shift 2 ;;
        --compare-python) COMPARE_PYTHON_BIN=${2:?"--compare-python requires a value"}; shift 2 ;;
        --variants) VARIANTS=${2:?"--variants requires a value"}; shift 2 ;;
        --lengths) LENGTHS=${2:?"--lengths requires a value"}; shift 2 ;;
        --stages) STAGES=${2:?"--stages requires a value"}; shift 2 ;;
        --batch-size) BATCH_SIZE=${2:?"--batch-size requires a value"}; shift 2 ;;
        --prompt-suite) PROMPT_SUITE=${2:?"--prompt-suite requires a value"}; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN=${2:?"--max-model-len requires a value"}; shift 2 ;;
        --tensor-parallel-size) TENSOR_PARALLEL_SIZE=${2:?"--tensor-parallel-size requires a value"}; shift 2 ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION=${2:?"--gpu-memory-utilization requires a value"}; shift 2 ;;
        --bf16-moe-backend) BF16_MOE_BACKEND=${2:?"--bf16-moe-backend requires a value"}; shift 2 ;;
        --mxfp8-moe-backend) MXFP8_MOE_BACKEND=${2:?"--mxfp8-moe-backend requires a value"}; shift 2 ;;
        --seed) SEED=${2:?"--seed requires a value"}; shift 2 ;;
        --top-k) TOP_K=${2:?"--top-k requires a value"}; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${MODEL}" || -z "${NATIVE_CHECKPOINT}" ]]; then
    echo "--model and --native-checkpoint are required." >&2
    exit 2
fi
if [[ -z "${COMPARE_PYTHON_BIN}" ]]; then
    COMPARE_PYTHON_BIN=${VLLM_PYTHON_BIN}
fi
if ((BATCH_SIZE < 1 || MAX_MODEL_LEN < 1 || TENSOR_PARALLEL_SIZE < 1)); then
    echo "Batch size, model length, and tensor parallel size must be positive." >&2
    exit 2
fi

contains_csv() {
    local list=",${1},"
    local item=$2
    [[ "${list}" == *",${item},"* ]]
}

case "${VARIANTS}" in
    bf16|mxfp8|bf16,mxfp8|mxfp8,bf16|both) ;;
    *) echo "--variants must be bf16, mxfp8, or both." >&2; exit 2 ;;
esac
[[ "${VARIANTS}" == "both" ]] && VARIANTS="bf16,mxfp8"

for stage in vllm native compare; do
    if ! contains_csv "${STAGES}" "${stage}" && [[ "${STAGES}" != "all" ]]; then
        continue
    fi
done
for requested_stage in ${STAGES//,/ }; do
    case "${requested_stage}" in
        all|vllm|native|compare) ;;
        *) echo "Unknown stage: ${requested_stage}" >&2; exit 2 ;;
    esac
done

for python_bin in "${NATIVE_PYTHON_BIN}" "${VLLM_PYTHON_BIN}" "${COMPARE_PYTHON_BIN}"; do
    if [[ ! -x "${python_bin}" ]]; then
        echo "Python is not executable: ${python_bin}" >&2
        exit 2
    fi
done
if [[ ! -f "${PROBE}" ]]; then
    echo "Decode probe not found: ${PROBE}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR=$(cd -- "${OUTPUT_DIR}" && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_logged() {
    local log_path=$1
    shift
    printf '\n+'
    printf ' %q' "$@"
    printf ' 2>&1 | tee %q\n' "${log_path}"
    if ((DRY_RUN)); then
        return
    fi
    "$@" 2>&1 | tee "${log_path}"
}

run_plain() {
    printf '\n+'
    printf ' %q' "$@"
    printf '\n'
    if ((DRY_RUN)); then
        return
    fi
    "$@"
}

checkpoint_positions() {
    local length=$1
    local positions=(1 2 4 8)
    local position
    for ((position = 16; position <= length; position += 16)); do
        positions+=("${position}")
    done
    if ((length < 8)); then
        positions=()
        for ((position = 1; position <= length; position *= 2)); do
            positions+=("${position}")
        done
    fi
    local last_index=$((${#positions[@]} - 1))
    if ((last_index < 0)) || [[ "${positions[${last_index}]}" != "${length}" ]]; then
        positions+=("${length}")
    fi
    local joined
    joined=$(IFS=,; echo "${positions[*]}")
    printf '%s' "${joined}"
}

run_variant_length() {
    local variant=$1
    local length=$2
    local tag="${variant}_decode${length}_fa2"
    local rollout="${OUTPUT_DIR}/${tag}_vllm.pt"
    local native="${OUTPUT_DIR}/${tag}_native.pt"
    local comparison="${OUTPUT_DIR}/${tag}_compare.json"
    local positions
    positions=$(checkpoint_positions "${length}")

    local vllm_extra=()
    local native_extra=()
    if [[ "${variant}" == "bf16" ]]; then
        vllm_extra+=(--moe-backend "${BF16_MOE_BACKEND}")
        native_extra+=(--native-quant-mode bfloat16)
    else
        vllm_extra+=(--quantization fp8_per_block --moe-backend "${MXFP8_MOE_BACKEND}")
        native_extra+=(
            --native-quant-mode mxfp8
            --native-quant-block-size 128
            --native-use-torch-fp8-quant
        )
    fi

    if [[ "${STAGES}" == "all" ]] || contains_csv "${STAGES}" vllm; then
        run_logged "${OUTPUT_DIR}/${tag}_vllm.log" \
            "${VLLM_PYTHON_BIN}" "${PROBE}" vllm \
            --model "${MODEL}" \
            --out "${rollout}" \
            --prompt-suite "${PROMPT_SUITE}" \
            --batch-size "${BATCH_SIZE}" \
            --decode-length "${length}" \
            --vocab-logprob-positions "${positions}" \
            --artifact-top-k "${TOP_K}" \
            --dtype bfloat16 \
            --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
            --max-num-seqs "${BATCH_SIZE}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --max-num-batched-tokens "${MAX_MODEL_LEN}" \
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
            --no-enable-prefix-caching \
            --attention-backend FLASH_ATTN \
            --flash-attn-version 2 \
            --force-fa-num-splits-one \
            --compilation-config-json "${COMPILATION_CONFIG}" \
            --seed "${SEED}" \
            "${vllm_extra[@]}"
    fi

    if [[ "${STAGES}" == "all" ]] || contains_csv "${STAGES}" native; then
        run_logged "${OUTPUT_DIR}/${tag}_native.log" \
            "${NATIVE_PYTHON_BIN}" -m torch.distributed.run \
            --standalone --nproc-per-node 1 \
            "${PROBE}" native \
            --rollout "${rollout}" \
            --out "${native}" \
            --native-checkpoint "${NATIVE_CHECKPOINT}" \
            --llm-train-dir "${LLM_TRAIN_DIR}" \
            --native-dtype bfloat16 \
            --native-local-attention \
            --native-no-kv-cache \
            --native-require-transformer-engine \
            --force-fa-num-splits-one \
            --max-model-len "${MAX_MODEL_LEN}" \
            --seed "${SEED}" \
            "${native_extra[@]}"
    fi

    if [[ "${STAGES}" == "all" ]] || contains_csv "${STAGES}" compare; then
        run_plain "${COMPARE_PYTHON_BIN}" "${PROBE}" compare \
            --vllm "${rollout}" \
            --native "${native}" \
            --model "${MODEL}" \
            --out-json "${comparison}" \
            --top-k "${TOP_K}"
    fi
}

echo "FA2 decode alignment output: ${OUTPUT_DIR}"
echo "Variants: ${VARIANTS}; lengths: ${LENGTHS}; stages: ${STAGES}"

for variant in ${VARIANTS//,/ }; do
    for length in ${LENGTHS//,/ }; do
        if ! [[ "${length}" =~ ^[1-9][0-9]*$ ]]; then
            echo "Invalid decode length: ${length}" >&2
            exit 2
        fi
        run_variant_length "${variant}" "${length}"
    done
done

echo
echo "FA2 decode alignment matrix complete: ${OUTPUT_DIR}"