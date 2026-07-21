#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PROBE="${SCRIPT_DIR}/logprob_kl.py"
VERIFY="${SCRIPT_DIR}/verify_v1_batch_shape.py"

usage() {
    cat <<'EOF'
Run the YOCO FA2 prefill alignment ablation for BF16 and MXFP8.

Usage:
  run_logprob_kl_prefill_ablation_fa2.sh \
    --model MODEL --native-checkpoint DIR [options]

Arms:
  A  Historical: V1 multiprocessing on, prefix cache on, heuristic splits;
     compare with Native 1+(batch-1).
  B  Controlled legacy: multiprocessing on, prefix cache off, heuristic splits;
     compare with Native 1+(batch-1).
  C  Shape change: multiprocessing off, prefix cache off, heuristic splits;
     compare with one Native batch.
  D  Canonical: multiprocessing off, prefix cache off, num_splits=1;
     compare with one Native batch.

Required:
  --model PATH                 Converted HF/vLLM model and tokenizer.
  --native-checkpoint PATH     Merged llm-train checkpoint.

Options:
  --output-dir PATH            Default: ../logs/yoco_alignment_results/fa2_prefill_ablation_mixed16
  --llm-train-dir PATH         Default: ../llm-train
  --native-python PATH         Default: .venv-yoco-native
  --vllm-python PATH           Default: .venv-yoco-mxfp8
  --compare-python PATH        Default: vLLM Python
  --variants LIST              bf16, mxfp8, or bf16,mxfp8 (default: both)
  --arms LIST                  A,B,C,D subset (default: all)
  --stages LIST                native,vllm,compare,verify (default: all)
  --batch-size N               Default: 16
  --prompt-suite NAME          Default: mixed16
  --max-model-len N            Default: 8192
  --tensor-parallel-size N     Default: 1
  --gpu-memory-utilization F   Default: 0.9
  --bf16-moe-backend NAME      Default: triton
  --mxfp8-moe-backend NAME     Default: deep_gemm
  --seed N                     Default: 0
  --top-k N                    Default: 20
  --dry-run                    Print commands without executing them
  -h, --help                   Show this help

The B-vs-C verifier cross-checks actual scheduler iteration request/token counts
against both artifacts and the one-batch Native model_forwards record.
EOF
}

MODEL=""
NATIVE_CHECKPOINT=""
OUTPUT_DIR="${REPO_ROOT}/../logs/yoco_alignment_results/fa2_prefill_ablation_mixed16"
LLM_TRAIN_DIR="${REPO_ROOT}/../llm-train"
NATIVE_PYTHON_BIN=${NATIVE_PYTHON_BIN:-"${REPO_ROOT}/../../.venv-yoco-native/bin/python"}
VLLM_PYTHON_BIN=${VLLM_PYTHON_BIN:-"${REPO_ROOT}/../../.venv-yoco-mxfp8/bin/python"}
COMPARE_PYTHON_BIN=${COMPARE_PYTHON_BIN:-}
VARIANTS="bf16,mxfp8"
ARMS="A,B,C,D"
STAGES="native,vllm,compare,verify"
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
        --arms) ARMS=${2:?"--arms requires a value"}; shift 2 ;;
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
if ((BATCH_SIZE < 2 || MAX_MODEL_LEN < 1 || TENSOR_PARALLEL_SIZE < 1)); then
    echo "Ablation requires batch-size >= 2 and positive model/parallel sizes." >&2
    exit 2
fi

contains_csv() {
    local list=",${1},"
    local item=$2
    [[ "${list}" == *",${item},"* ]]
}

[[ "${VARIANTS}" == "both" ]] && VARIANTS="bf16,mxfp8"
case "${VARIANTS}" in
    bf16|mxfp8|bf16,mxfp8|mxfp8,bf16) ;;
    *) echo "--variants must be bf16, mxfp8, or both." >&2; exit 2 ;;
esac
ARMS=${ARMS^^}
[[ "${ARMS}" == "ALL" ]] && ARMS="A,B,C,D"
for arm in ${ARMS//,/ }; do
    case "${arm}" in A|B|C|D) ;; *) echo "Unknown arm: ${arm}" >&2; exit 2 ;; esac
done
[[ "${STAGES}" == "all" ]] && STAGES="native,vllm,compare,verify"
for stage in ${STAGES//,/ }; do
    case "${stage}" in native|vllm|compare|verify) ;; *) echo "Unknown stage: ${stage}" >&2; exit 2 ;; esac
done

for python_bin in "${NATIVE_PYTHON_BIN}" "${VLLM_PYTHON_BIN}" "${COMPARE_PYTHON_BIN}"; do
    if [[ ! -x "${python_bin}" ]]; then
        echo "Python is not executable: ${python_bin}" >&2
        exit 2
    fi
done
for path in "${PROBE}" "${VERIFY}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Required tool not found: ${path}" >&2
        exit 2
    fi
done

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
    if ((DRY_RUN)); then return; fi
    "$@" 2>&1 | tee "${log_path}"
}

run_plain() {
    printf '\n+'
    printf ' %q' "$@"
    printf '\n'
    if ((DRY_RUN)); then return; fi
    "$@"
}

native_extra_args() {
    local variant=$1
    if [[ "${variant}" == "bf16" ]]; then
        printf '%s\0' --native-quant-mode bfloat16
    else
        printf '%s\0' \
            --native-quant-mode mxfp8 \
            --native-quant-block-size 128 \
            --native-use-torch-fp8-quant
    fi
}

run_native_reference() {
    local variant=$1
    local shape=$2
    local output="${OUTPUT_DIR}/${variant}_native_${shape}_fa2.pt"
    local log="${OUTPUT_DIR}/${variant}_native_${shape}_fa2.log"
    local shape_args=()
    if [[ "${shape}" == "1plus$((BATCH_SIZE - 1))" ]]; then
        shape_args+=(--first-batch-size 1 --batch-size "$((BATCH_SIZE - 1))")
    else
        shape_args+=(--batch-size "${BATCH_SIZE}" --force-fa-num-splits-one)
    fi
    local native_extra=()
    while IFS= read -r -d '' item; do native_extra+=("${item}"); done < <(native_extra_args "${variant}")

    run_logged "${log}" \
        "${NATIVE_PYTHON_BIN}" -m torch.distributed.run \
        --standalone --nproc-per-node 1 \
        "${PROBE}" native \
        --model "${MODEL}" \
        --native-checkpoint "${NATIVE_CHECKPOINT}" \
        --llm-train-dir "${LLM_TRAIN_DIR}" \
        --native-dtype bfloat16 \
        --native-local-attention \
        --native-require-transformer-engine \
        --prompt-suite "${PROMPT_SUITE}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --seed "${SEED}" \
        --out "${output}" \
        "${shape_args[@]}" \
        "${native_extra[@]}"
}

arm_native_shape() {
    case "$1" in
        A|B) printf '1plus%s' "$((BATCH_SIZE - 1))" ;;
        C|D) printf 'batch%s' "${BATCH_SIZE}" ;;
    esac
}

run_vllm_arm() {
    local variant=$1
    local arm=$2
    local tag="${variant}_arm${arm}_fa2"
    local output="${OUTPUT_DIR}/${tag}_vllm.pt"
    local log="${OUTPUT_DIR}/${tag}_vllm.log"
    local arm_args=()
    case "${arm}" in
        A) arm_args+=(--v1-multiprocessing --enable-prefix-caching) ;;
        B) arm_args+=(--v1-multiprocessing --no-enable-prefix-caching) ;;
        C) arm_args+=(--no-v1-multiprocessing --no-enable-prefix-caching) ;;
        D) arm_args+=(--no-v1-multiprocessing --no-enable-prefix-caching --force-fa-num-splits-one) ;;
    esac
    local variant_args=()
    if [[ "${variant}" == "bf16" ]]; then
        variant_args+=(--moe-backend "${BF16_MOE_BACKEND}")
    else
        variant_args+=(--quantization fp8_per_block --moe-backend "${MXFP8_MOE_BACKEND}")
    fi

    run_logged "${log}" \
        "${VLLM_PYTHON_BIN}" "${PROBE}" vllm \
        --model "${MODEL}" \
        --out "${output}" \
        --prompt-suite "${PROMPT_SUITE}" \
        --batch-size "${BATCH_SIZE}" \
        --dtype bfloat16 \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --max-num-seqs "${BATCH_SIZE}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-batched-tokens "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-logprobs -1 \
        --attention-backend FLASH_ATTN \
        --flash-attn-version 2 \
        --compilation-config-json "${COMPILATION_CONFIG}" \
        --log-iteration-details \
        --seed "${SEED}" \
        "${arm_args[@]}" \
        "${variant_args[@]}"
}

compare_arm() {
    local variant=$1
    local arm=$2
    local shape
    shape=$(arm_native_shape "${arm}")
    run_plain "${COMPARE_PYTHON_BIN}" "${PROBE}" compare \
        --reference "${OUTPUT_DIR}/${variant}_native_${shape}_fa2.pt" \
        --candidate "${OUTPUT_DIR}/${variant}_arm${arm}_fa2_vllm.pt" \
        --model "${MODEL}" \
        --out-json "${OUTPUT_DIR}/${variant}_arm${arm}_fa2_compare.json" \
        --top-k "${TOP_K}"
}

verify_shape_change() {
    local variant=$1
    run_plain "${COMPARE_PYTHON_BIN}" "${VERIFY}" \
        --multiprocessing-log "${OUTPUT_DIR}/${variant}_armB_fa2_vllm.log" \
        --multiprocessing-artifact "${OUTPUT_DIR}/${variant}_armB_fa2_vllm.pt" \
        --in-process-log "${OUTPUT_DIR}/${variant}_armC_fa2_vllm.log" \
        --in-process-artifact "${OUTPUT_DIR}/${variant}_armC_fa2_vllm.pt" \
        --native-artifact "${OUTPUT_DIR}/${variant}_native_batch${BATCH_SIZE}_fa2.pt" \
        --out-json "${OUTPUT_DIR}/${variant}_armB_vs_C_batch_shape.json"
}

echo "FA2 prefill ablation output: ${OUTPUT_DIR}"
echo "Variants: ${VARIANTS}; arms: ${ARMS}; stages: ${STAGES}"

for variant in ${VARIANTS//,/ }; do
    if contains_csv "${STAGES}" native; then
        if contains_csv "${ARMS}" A || contains_csv "${ARMS}" B; then
            run_native_reference "${variant}" "1plus$((BATCH_SIZE - 1))"
        fi
        if contains_csv "${ARMS}" C || contains_csv "${ARMS}" D; then
            run_native_reference "${variant}" "batch${BATCH_SIZE}"
        fi
    fi

    for arm in ${ARMS//,/ }; do
        if contains_csv "${STAGES}" vllm; then
            run_vllm_arm "${variant}" "${arm}"
        fi
        if contains_csv "${STAGES}" compare; then
            compare_arm "${variant}" "${arm}"
        fi
    done

    if contains_csv "${STAGES}" verify; then
        if contains_csv "${ARMS}" B && contains_csv "${ARMS}" C; then
            verify_shape_change "${variant}"
        else
            echo "Skipping ${variant} B-vs-C shape verification: arms B and C are required."
        fi
    fi
done

echo
echo "FA2 prefill ablation complete: ${OUTPUT_DIR}"