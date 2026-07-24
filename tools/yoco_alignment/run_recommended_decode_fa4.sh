#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PROBE="${SCRIPT_DIR}/logprob_kl_decode.py"

MODEL=""
NATIVE_CHECKPOINT=""
OUTPUT_ROOT="${REPO_ROOT}/../logs/yoco_alignment_results/fa4_decode_recommended_b13"
OVERLAY_DIR=${YOCO_IMAGE_FA4_OVERLAY:-"${REPO_ROOT}/../.image-fa4-4.0.0b13"}
LLM_TRAIN_DIR="${REPO_ROOT}/../llm-train"
NATIVE_PYTHON_BIN=${NATIVE_PYTHON_BIN:-"${REPO_ROOT}/../../.venv-yoco-native/bin/python"}
VLLM_PYTHON_BIN=${VLLM_PYTHON_BIN:-"${REPO_ROOT}/../../.venv-yoco-mxfp8/bin/python"}
COMPARE_PYTHON_BIN=${COMPARE_PYTHON_BIN:-"${VLLM_PYTHON_BIN}"}
PROFILES="fa4-bf16,fa4-mxfp8-qkv-bf16"
GPUS=${CUDA_VISIBLE_DEVICES:-0}
LENGTHS="16,128"
STAGES="vllm,native,compare"
PROMPT_SUITE="mixed16"
BATCH_SIZE=16
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.9
SEED=0
TOP_K=20
DRY_RUN=0
COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}'

usage() {
    cat <<'EOF'
Run exact-image b13 FA4 decode alignment for the recommended YOCO profiles.

Usage:
  run_recommended_decode_fa4.sh \
    --model MODEL --native-checkpoint DIR [options]

Profiles:
  fa4-bf16                 BF16 with Triton MoE.
  fa4-mxfp8-qkv-bf16       MXFP8/DeepGEMM with attention-input QKV in BF16.

Options:
  --profiles LIST          Comma-separated profile subset (default: both).
  --gpus LIST              Comma-separated GPU IDs. Profiles are assigned
                           round-robin; each GPU serializes its profiles while
                           distinct GPUs run concurrently.
  --lengths LIST           Decode lengths (default: 16,128).
  --stages LIST            vllm,native,compare subset (default: all).
  --output-root PATH       Parent output directory.
  --overlay-dir PATH       Exact-image b13 FA4 overlay.
  --llm-train-dir PATH     llm-train checkout.
  --native-python PATH     Native NVIDIA-stack Python.
  --vllm-python PATH       vLLM Python.
  --compare-python PATH    Comparison Python.
  --prompt-suite NAME      Prompt suite (default: mixed16).
  --batch-size N           Matched batch size (default: 16).
  --max-model-len N        Maximum model length (default: 8192).
  --gpu-memory-utilization F
                           vLLM GPU memory fraction (default: 0.9).
  --seed N                 Random seed (default: 0).
  --top-k N                Comparison top rows (default: 20).
  --dry-run                Print commands without model execution.
  -h, --help               Show this help.

The launcher uses mixed16 in one in-process 16-request batch, prefix cache off,
num_splits=1, FULL_DECODE_ONLY, default exp2, explicit tile_mn=(128,128), and
auto q-stage. Native replays the exact rollout in one packed no-cache prefill.
EOF
}

while (($#)); do
    case "$1" in
        --model) MODEL=${2:?"--model requires a value"}; shift 2 ;;
        --native-checkpoint) NATIVE_CHECKPOINT=${2:?"--native-checkpoint requires a value"}; shift 2 ;;
        --profiles) PROFILES=${2:?"--profiles requires a value"}; shift 2 ;;
        --gpus) GPUS=${2:?"--gpus requires a value"}; shift 2 ;;
        --lengths) LENGTHS=${2:?"--lengths requires a value"}; shift 2 ;;
        --stages) STAGES=${2:?"--stages requires a value"}; shift 2 ;;
        --output-root) OUTPUT_ROOT=${2:?"--output-root requires a value"}; shift 2 ;;
        --overlay-dir) OVERLAY_DIR=${2:?"--overlay-dir requires a value"}; shift 2 ;;
        --llm-train-dir) LLM_TRAIN_DIR=${2:?"--llm-train-dir requires a value"}; shift 2 ;;
        --native-python) NATIVE_PYTHON_BIN=${2:?"--native-python requires a value"}; shift 2 ;;
        --vllm-python) VLLM_PYTHON_BIN=${2:?"--vllm-python requires a value"}; shift 2 ;;
        --compare-python) COMPARE_PYTHON_BIN=${2:?"--compare-python requires a value"}; shift 2 ;;
        --prompt-suite) PROMPT_SUITE=${2:?"--prompt-suite requires a value"}; shift 2 ;;
        --batch-size) BATCH_SIZE=${2:?"--batch-size requires a value"}; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN=${2:?"--max-model-len requires a value"}; shift 2 ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION=${2:?"--gpu-memory-utilization requires a value"}; shift 2 ;;
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
if ((BATCH_SIZE < 1 || MAX_MODEL_LEN < 1 || TOP_K < 1)); then
    echo "Batch size, model length, and top-k must be positive." >&2
    exit 2
fi
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

IFS=',' read -r -a PROFILE_ARRAY <<< "${PROFILES}"
declare -A SEEN_PROFILES=()
for index in "${!PROFILE_ARRAY[@]}"; do
    profile=${PROFILE_ARRAY[index]//[[:space:]]/}
    PROFILE_ARRAY[index]=${profile}
    case "${profile}" in
        fa4-bf16|fa4-mxfp8-qkv-bf16) ;;
        *) echo "Unknown profile: ${profile}" >&2; exit 2 ;;
    esac
    if [[ -n "${SEEN_PROFILES[${profile}]:-}" ]]; then
        echo "Duplicate profile: ${profile}" >&2
        exit 2
    fi
    SEEN_PROFILES[${profile}]=1
done

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
declare -A SEEN_GPUS=()
for index in "${!GPU_ARRAY[@]}"; do
    gpu=${GPU_ARRAY[index]//[[:space:]]/}
    GPU_ARRAY[index]=${gpu}
    if [[ -z "${gpu}" || -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
        echo "GPU IDs must be non-empty and unique: ${gpu}" >&2
        exit 2
    fi
    SEEN_GPUS[${gpu}]=1
done

IFS=',' read -r -a LENGTH_ARRAY <<< "${LENGTHS}"
for index in "${!LENGTH_ARRAY[@]}"; do
    length=${LENGTH_ARRAY[index]//[[:space:]]/}
    LENGTH_ARRAY[index]=${length}
    if ! [[ "${length}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid decode length: ${length}" >&2
        exit 2
    fi
done

[[ "${STAGES}" == "all" ]] && STAGES="vllm,native,compare"
for stage in ${STAGES//,/ }; do
    case "${stage}" in
        vllm|native|compare) ;;
        *) echo "Unknown stage: ${stage}" >&2; exit 2 ;;
    esac
done

contains_csv() {
    local list=",${1},"
    local item=$2
    [[ "${list}" == *",${item},"* ]]
}

SITE_PACKAGES="${OVERLAY_DIR}/site-packages"
CUTLASS_PACKAGES="${SITE_PACKAGES}/nvidia_cutlass_dsl/python_packages"
FA4_SOURCE="${SITE_PACKAGES}"
for path in \
    "${CUTLASS_PACKAGES}/cutlass/__init__.py" \
    "${FA4_SOURCE}/flash_attn/cute/interface.py"; do
    if [[ ! -f "${path}" ]]; then
        echo "Exact-image FA4 overlay is incomplete: ${path}" >&2
        echo "Run tools/yoco_alignment/setup_image_fa4_profiles.sh first." >&2
        exit 2
    fi
done

mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT=$(cd -- "${OUTPUT_ROOT}" && pwd)

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

run_command() {
    printf '\n+'
    printf ' %q' "$@"
    printf '\n'
    if ((DRY_RUN)); then
        return
    fi
    "$@"
}

run_profile_length() {
    local profile=$1
    local gpu=$2
    local length=$3
    local output_dir="${OUTPUT_ROOT}/${profile}/decode${length}"
    local rollout="${output_dir}/vllm.pt"
    local native="${output_dir}/native.pt"
    local comparison="${output_dir}/compare.json"
    local positions
    positions=$(checkpoint_positions "${length}")
    mkdir -p "${output_dir}"

    local vllm_precision_args=()
    local native_precision_args=()
    if [[ "${profile}" == "fa4-bf16" ]]; then
        vllm_precision_args+=(--moe-backend triton)
        native_precision_args+=(--native-quant-mode bfloat16)
    else
        vllm_precision_args+=(
            --quantization fp8_per_block
            --moe-backend deep_gemm
            --keep-attention-qkv-bf16
        )
        native_precision_args+=(
            --native-quant-mode mxfp8
            --native-quant-block-size 128
            --native-use-torch-fp8-quant
            --keep-attention-qkv-bf16
        )
    fi

    local common_fa4_args=(
        --fa4-source-root "${FA4_SOURCE}"
        --fa4-profile default
        --fa4-pack-gqa auto
        --fa4-tile-mn 128x128
    )

    echo "[${profile}] decode${length} on GPU ${gpu}"
    if contains_csv "${STAGES}" vllm; then
        run_command "${VLLM_PYTHON_BIN}" "${PROBE}" vllm \
            --model "${MODEL}" \
            --out "${rollout}" \
            --prompt-suite "${PROMPT_SUITE}" \
            --batch-size "${BATCH_SIZE}" \
            --decode-length "${length}" \
            --vocab-logprob-positions "${positions}" \
            --artifact-top-k "${TOP_K}" \
            --dtype bfloat16 \
            --tensor-parallel-size 1 \
            --max-num-seqs "${BATCH_SIZE}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --max-num-batched-tokens "${MAX_MODEL_LEN}" \
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
            --no-enable-prefix-caching \
            --attention-backend FLASH_ATTN \
            --flash-attn-version 4 \
            --force-fa-num-splits-one \
            --compilation-config-json "${COMPILATION_CONFIG}" \
            --seed "${SEED}" \
            "${common_fa4_args[@]}" \
            "${vllm_precision_args[@]}"
    fi

    if contains_csv "${STAGES}" native; then
        run_command "${NATIVE_PYTHON_BIN}" -m torch.distributed.run \
            --standalone --nproc-per-node 1 \
            "${PROBE}" native \
            --rollout "${rollout}" \
            --out "${native}" \
            --native-checkpoint "${NATIVE_CHECKPOINT}" \
            --llm-train-dir "${LLM_TRAIN_DIR}" \
            --native-dtype bfloat16 \
            --native-use-cute \
            --native-local-attention \
            --native-no-kv-cache \
            --native-require-transformer-engine \
            --force-fa-num-splits-one \
            --max-model-len "${MAX_MODEL_LEN}" \
            --seed "${SEED}" \
            "${common_fa4_args[@]}" \
            "${native_precision_args[@]}"
    fi

    if contains_csv "${STAGES}" compare; then
        run_command "${COMPARE_PYTHON_BIN}" "${PROBE}" compare \
            --vllm "${rollout}" \
            --native "${native}" \
            --model "${MODEL}" \
            --out-json "${comparison}" \
            --top-k "${TOP_K}"
    fi
}

run_profile_worker() {
    local profile=$1
    local gpu=$2
    local status=0
    local length output_dir log_path
    for length in "${LENGTH_ARRAY[@]}"; do
        output_dir="${OUTPUT_ROOT}/${profile}/decode${length}"
        log_path="${output_dir}/run.log"
        mkdir -p "${output_dir}"
        echo "[START] ${profile} decode${length} on GPU ${gpu}"
        if ((DRY_RUN)); then
            run_profile_length "${profile}" "${gpu}" "${length}"
        elif CUDA_VISIBLE_DEVICES="${gpu}" \
            PYTHONPATH="${CUTLASS_PACKAGES}:${FA4_SOURCE}:${SITE_PACKAGES}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
            run_profile_length "${profile}" "${gpu}" "${length}" \
            >"${log_path}" 2>&1; then
            echo "[PASS] ${profile} decode${length} -> ${output_dir}"
        else
            echo "[FAIL] ${profile} decode${length}; see ${log_path}" >&2
            status=1
            break
        fi
    done
    return "${status}"
}

declare -A GPU_PROFILES=()
declare -A PROFILE_GPU=()
for index in "${!PROFILE_ARRAY[@]}"; do
    profile=${PROFILE_ARRAY[index]}
    gpu=${GPU_ARRAY[index % ${#GPU_ARRAY[@]}]}
    GPU_PROFILES[${gpu}]="${GPU_PROFILES[${gpu}]:-}${GPU_PROFILES[${gpu}]:+ }${profile}"
    PROFILE_GPU[${profile}]=${gpu}
done

echo "Recommended FA4 decode output: ${OUTPUT_ROOT}"
echo "Lengths: ${LENGTHS}; stages: ${STAGES}"
for profile in "${PROFILE_ARRAY[@]}"; do
    echo "  ${profile}: GPU ${PROFILE_GPU[${profile}]}"
done

run_gpu_worker() {
    local gpu=$1
    local status=0
    local profile
    for profile in ${GPU_PROFILES[${gpu}]}; do
        if ! run_profile_worker "${profile}" "${gpu}"; then
            status=1
        fi
    done
    return "${status}"
}

if ((DRY_RUN)); then
    for profile in "${PROFILE_ARRAY[@]}"; do
        run_profile_worker "${profile}" "${PROFILE_GPU[${profile}]}"
    done
    echo "Recommended FA4 decode dry-run complete."
    exit 0
fi

PIDS=()
for gpu in "${GPU_ARRAY[@]}"; do
    if [[ -n "${GPU_PROFILES[${gpu}]:-}" ]]; then
        run_gpu_worker "${gpu}" &
        PIDS+=("$!")
    fi
done

STATUS=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        STATUS=1
    fi
done
if ((STATUS)); then
    echo "One or more recommended FA4 decode profiles failed." >&2
    exit "${STATUS}"
fi
echo "Recommended FA4 decode matrix complete: ${OUTPUT_ROOT}"