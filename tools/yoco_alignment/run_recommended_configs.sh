#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
RUNNER="${SCRIPT_DIR}/run_logprob_kl_compare.sh"
FA4_SETUP="${SCRIPT_DIR}/setup_image_fa4_profiles.sh"

MODEL=""
NATIVE_CHECKPOINT=""
OUTPUT_ROOT="${REPO_ROOT}/../logs/yoco_alignment_results/recommended_configs"
OVERLAY_DIR=${YOCO_IMAGE_FA4_OVERLAY:-"${REPO_ROOT}/../.image-fa4-4.0.0b13"}
LLM_TRAIN_DIR="${REPO_ROOT}/../llm-train"
NATIVE_PYTHON_BIN=${NATIVE_PYTHON_BIN:-"${REPO_ROOT}/../../.venv-yoco-native/bin/python"}
VLLM_PYTHON_BIN=${VLLM_PYTHON_BIN:-"${REPO_ROOT}/../../.venv-yoco-mxfp8/bin/python"}
COMPARE_PYTHON_BIN=${COMPARE_PYTHON_BIN:-"${VLLM_PYTHON_BIN}"}
PROFILES="fa2-bf16,fa2-mxfp8,fa4-bf16,fa4-mxfp8-qkv-bf16"
GPUS=${CUDA_VISIBLE_DEVICES:-0}
PROMPT_SUITE="mixed16"
BATCH_SIZE=16
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.9
SEED=0
TOP_K=20
KL_THRESHOLD=0.01
SETUP_FA4=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Reproduce the recommended YOCO FA2 and FA4 full-vocabulary alignment configs.

Usage:
  run_recommended_configs.sh \
    --model MODEL --native-checkpoint DIR [options]

Recommended profiles:
  fa2-bf16                 FA2 BF16 reference with Triton MoE.
  fa2-mxfp8                Production FA2 MXFP8 with DeepGEMM MoE.
  fa4-bf16                 Exact-image b13 FA4 BF16 reference.
  fa4-mxfp8-qkv-bf16       Exact-image b13 FA4 MXFP8 with attention QKV BF16.

Optional control:
  fa4-mxfp8                Exact-image b13 all-MXFP8 baseline.

Options:
  --profiles LIST          Comma-separated profiles. "recommended" selects the
                           four profiles above; "all" also adds the FA4 control.
  --gpus LIST              Comma-separated physical GPU IDs (default:
                           CUDA_VISIBLE_DEVICES, or 0). Profiles are assigned
                           round-robin. Each GPU runs its profiles serially;
                           distinct GPUs run in parallel.
  --output-root PATH       Parent output directory.
  --overlay-dir PATH       Exact-image b13 FA4 overlay directory.
  --setup-fa4              Create the exact FA4 overlay when it is absent.
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
  --top-k N                Tokens retained in reports (default: 20).
    --kl-threshold F         Required Native-to-vLLM mean-KL upper bound for
                                                     recommended profiles (default: 0.01).
  --dry-run                Print the resolved plan and child commands.
  -h, --help               Show this help.

Examples:
  # Four profiles on four GPUs.
  run_recommended_configs.sh --model MODEL --native-checkpoint DIR \
    --gpus 0,1,2,3

  # The two MXFP8 recommendations in parallel.
  run_recommended_configs.sh --model MODEL --native-checkpoint DIR \
    --profiles fa2-mxfp8,fa4-mxfp8-qkv-bf16 --gpus 0,1

  # All profiles serially on one GPU.
  run_recommended_configs.sh --model MODEL --native-checkpoint DIR \
    --profiles all --gpus 0
EOF
}

while (($#)); do
    case "$1" in
        --model) MODEL=${2:?"--model requires a value"}; shift 2 ;;
        --native-checkpoint) NATIVE_CHECKPOINT=${2:?"--native-checkpoint requires a value"}; shift 2 ;;
        --profiles) PROFILES=${2:?"--profiles requires a value"}; shift 2 ;;
        --gpus) GPUS=${2:?"--gpus requires a value"}; shift 2 ;;
        --output-root) OUTPUT_ROOT=${2:?"--output-root requires a value"}; shift 2 ;;
        --overlay-dir) OVERLAY_DIR=${2:?"--overlay-dir requires a value"}; shift 2 ;;
        --setup-fa4) SETUP_FA4=1; shift ;;
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
        --kl-threshold) KL_THRESHOLD=${2:?"--kl-threshold requires a value"}; shift 2 ;;
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
if [[ ! "${KL_THRESHOLD}" =~ ^[0-9]+([.][0-9]+)?([eE]-?[0-9]+)?$ ]]; then
    echo "--kl-threshold must be a non-negative number." >&2
    exit 2
fi
for path in "${RUNNER}" "${FA4_SETUP}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Required tool not found: ${path}" >&2
        exit 2
    fi
done

case "${PROFILES}" in
    recommended)
        PROFILES="fa2-bf16,fa2-mxfp8,fa4-bf16,fa4-mxfp8-qkv-bf16"
        ;;
    all)
        PROFILES="fa2-bf16,fa2-mxfp8,fa4-bf16,fa4-mxfp8-qkv-bf16,fa4-mxfp8"
        ;;
esac

IFS=',' read -r -a PROFILE_ARRAY <<< "${PROFILES}"
declare -A SEEN_PROFILES=()
NEEDS_FA4=0
for index in "${!PROFILE_ARRAY[@]}"; do
    profile=${PROFILE_ARRAY[index]//[[:space:]]/}
    PROFILE_ARRAY[index]=${profile}
    case "${profile}" in
        fa2-bf16|fa2-mxfp8) ;;
        fa4-bf16|fa4-mxfp8-qkv-bf16|fa4-mxfp8) NEEDS_FA4=1 ;;
        *) echo "Unknown profile: ${profile}" >&2; exit 2 ;;
    esac
    if [[ -n "${SEEN_PROFILES[${profile}]:-}" ]]; then
        echo "Duplicate profile: ${profile}" >&2
        exit 2
    fi
    SEEN_PROFILES[${profile}]=1
done
if ((${#PROFILE_ARRAY[@]} == 0)); then
    echo "At least one profile is required." >&2
    exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
declare -A SEEN_GPUS=()
for index in "${!GPU_ARRAY[@]}"; do
    gpu=${GPU_ARRAY[index]//[[:space:]]/}
    GPU_ARRAY[index]=${gpu}
    if [[ -z "${gpu}" ]]; then
        echo "GPU IDs must not be empty." >&2
        exit 2
    fi
    if [[ -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
        echo "Duplicate GPU ID: ${gpu}" >&2
        exit 2
    fi
    SEEN_GPUS[${gpu}]=1
done
if ((${#GPU_ARRAY[@]} == 0)); then
    echo "At least one GPU is required." >&2
    exit 2
fi

SITE_PACKAGES="${OVERLAY_DIR}/site-packages"
CUTLASS_PACKAGES="${SITE_PACKAGES}/nvidia_cutlass_dsl/python_packages"
DEFAULT_FA4_SOURCE="${SITE_PACKAGES}"

fa4_overlay_complete() {
    [[ -f "${CUTLASS_PACKAGES}/cutlass/__init__.py" ]] &&
        [[ -f "${DEFAULT_FA4_SOURCE}/flash_attn/cute/interface.py" ]]
}

if ((NEEDS_FA4)) && ! fa4_overlay_complete; then
    if ((SETUP_FA4)); then
        if [[ -e "${OVERLAY_DIR}" ]]; then
            echo "FA4 overlay exists but is incomplete: ${OVERLAY_DIR}" >&2
            echo "Move it aside or recreate it explicitly before retrying." >&2
            exit 2
        fi
        if ((DRY_RUN)); then
            echo "Dry-run cannot prepare a missing FA4 overlay." >&2
            echo "Run ${FA4_SETUP} --overlay-dir ${OVERLAY_DIR} first." >&2
            exit 2
        fi
        bash "${FA4_SETUP}" \
            --python "${VLLM_PYTHON_BIN}" \
            --overlay-dir "${OVERLAY_DIR}"
    else
        echo "Exact-image FA4 overlay is missing or incomplete: ${OVERLAY_DIR}" >&2
        echo "Rerun with --setup-fa4, or run ${FA4_SETUP} first." >&2
        exit 2
    fi
fi

mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT=$(cd -- "${OUTPUT_ROOT}" && pwd)

declare -A GPU_PROFILES=()
declare -A PROFILE_GPU=()
for index in "${!PROFILE_ARRAY[@]}"; do
    profile=${PROFILE_ARRAY[index]}
    gpu=${GPU_ARRAY[index % ${#GPU_ARRAY[@]}]}
    GPU_PROFILES[${gpu}]="${GPU_PROFILES[${gpu}]:-}${GPU_PROFILES[${gpu}]:+ }${profile}"
    PROFILE_GPU[${profile}]=${gpu}
done

echo "Recommended-config output: ${OUTPUT_ROOT}"
echo "Prompt suite: ${PROMPT_SUITE}; batch size: ${BATCH_SIZE}; seed: ${SEED}"
echo "Resolved profile plan:"
for profile in "${PROFILE_ARRAY[@]}"; do
    printf '  %-28s GPU %s -> %s/%s\n' \
        "${profile}" "${PROFILE_GPU[${profile}]}" "${OUTPUT_ROOT}" "${profile}"
done

run_profile() {
    local profile=$1
    local gpu=$2
    local output_dir="${OUTPUT_ROOT}/${profile}"
    local attention_version variants
    local extra_args=()

    case "${profile}" in
        fa2-bf16)
            attention_version=2
            variants=bf16
            ;;
        fa2-mxfp8)
            attention_version=2
            variants=fp8
            ;;
        fa4-bf16)
            attention_version=4
            variants=bf16
            ;;
        fa4-mxfp8-qkv-bf16)
            attention_version=4
            variants=fp8
            extra_args+=(--keep-attention-qkv-bf16)
            ;;
        fa4-mxfp8)
            attention_version=4
            variants=fp8
            ;;
    esac

    local args=(
        --model "${MODEL}"
        --native-checkpoint "${NATIVE_CHECKPOINT}"
        --output-dir "${output_dir}"
        --llm-train-dir "${LLM_TRAIN_DIR}"
        --native-python "${NATIVE_PYTHON_BIN}"
        --vllm-python "${VLLM_PYTHON_BIN}"
        --compare-python "${COMPARE_PYTHON_BIN}"
        --attention-version "${attention_version}"
        --prompt-suite "${PROMPT_SUITE}"
        --batch-size "${BATCH_SIZE}"
        --max-model-len "${MAX_MODEL_LEN}"
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
        --bf16-moe-backend triton
        --fp8-moe-backend deep_gemm
        --variants "${variants}"
        --seed "${SEED}"
        --top-k "${TOP_K}"
        "${extra_args[@]}"
    )
    if ((DRY_RUN)); then
        args+=(--dry-run)
    fi

    if [[ "${attention_version}" == 4 ]]; then
        CUDA_VISIBLE_DEVICES="${gpu}" \
        YOCO_FA4_SOURCE_ROOT="${DEFAULT_FA4_SOURCE}" \
        YOCO_FA4_PROFILE=default \
        YOCO_FA4_PACK_GQA=auto \
        YOCO_FA4_TILE_MN=128x128 \
        PYTHONPATH="${CUTLASS_PACKAGES}:${DEFAULT_FA4_SOURCE}:${SITE_PACKAGES}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
            bash "${RUNNER}" "${args[@]}"
    else
        CUDA_VISIBLE_DEVICES="${gpu}" \
        PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
            bash "${RUNNER}" "${args[@]}"
    fi
}

run_profile_logged() {
    local profile=$1
    local gpu=$2
    local output_dir="${OUTPUT_ROOT}/${profile}"
    local log_path="${output_dir}/run.log"
    mkdir -p "${output_dir}"
    echo "[START] ${profile} on GPU ${gpu}"
    if ((DRY_RUN)); then
        run_profile "${profile}" "${gpu}"
        echo "[DRY-RUN] ${profile}"
        return
    fi
    if run_profile "${profile}" "${gpu}" >"${log_path}" 2>&1; then
        echo "[PASS] ${profile} -> ${output_dir}"
        return
    else
        local status=$?
        echo "[FAIL] ${profile} (exit ${status}); see ${log_path}" >&2
        return "${status}"
    fi
}

run_gpu_worker() {
    local gpu=$1
    local status=0
    local profile
    for profile in ${GPU_PROFILES[${gpu}]}; do
        if ! run_profile_logged "${profile}" "${gpu}"; then
            status=1
        fi
    done
    return "${status}"
}

summarize_results() {
    local profile_gpu_args=()
    local profile
    for profile in "${PROFILE_ARRAY[@]}"; do
        profile_gpu_args+=("${profile}" "${PROFILE_GPU[${profile}]}")
    done
    "${COMPARE_PYTHON_BIN}" - \
        "${OUTPUT_ROOT}" "${KL_THRESHOLD}" "${profile_gpu_args[@]}" <<'PY'
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
threshold = float(sys.argv[2])
profile_gpu_args = sys.argv[3:]
if len(profile_gpu_args) % 2:
    raise RuntimeError("Expected profile/GPU argument pairs")
profile_gpus = dict(zip(profile_gpu_args[::2], profile_gpu_args[1::2]))
profiles = list(profile_gpus)

comparison_paths = {
    "fa2-bf16": "native_bf16_fa2_vs_vllm.json",
    "fa2-mxfp8": "native_mxfp8_fa2_vs_vllm_fp8_per_block.json",
    "fa4-bf16": "native_bf16_fa4_vs_vllm.json",
    "fa4-mxfp8-qkv-bf16": (
        "native_mxfp8_qkv_bf16_fa4_vs_vllm_fp8_per_block_qkv_bf16.json"
    ),
    "fa4-mxfp8": "native_mxfp8_fa4_vs_vllm_fp8_per_block.json",
}
documented_kl = {
    "fa2-bf16": 0.0064852691066334955,
    "fa2-mxfp8": 0.008401697690715082,
    "fa4-bf16": 0.00341708,
    "fa4-mxfp8-qkv-bf16": 0.00565650,
    "fa4-mxfp8": 0.0187185,
}

summary_rows = []
failed_profiles = []
for profile in profiles:
    comparison_path = output_root / profile / comparison_paths[profile]
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    observed_kl = float(comparison["aggregate"]["mean_kl_native_to_vllm"])
    required_pass = profile != "fa4-mxfp8"
    accepted = observed_kl < threshold
    if required_pass and not accepted:
        failed_profiles.append(profile)
    summary_rows.append(
        {
            "profile": profile,
            "gpu": profile_gpus[profile],
            "comparison": str(comparison_path.relative_to(output_root)),
            "mean_kl_native_to_vllm": observed_kl,
            "documented_mean_kl": documented_kl[profile],
            "delta_from_documented": observed_kl - documented_kl[profile],
            "kl_threshold": threshold,
            "required_pass": required_pass,
            "accepted": accepted,
        }
    )

summary = {
    "kl_threshold": threshold,
    "all_recommended_profiles_accepted": not failed_profiles,
    "failed_recommended_profiles": failed_profiles,
    "profiles": summary_rows,
}
summary_path = output_root / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

print("\nCombined recommended-config summary:")
print(f"{'profile':28} {'observed KL':>13} {'recorded KL':>13}  result")
for row in summary_rows:
    if not row["required_pass"]:
        result = "control"
    else:
        result = "pass" if row["accepted"] else "FAIL"
    print(
        f"{row['profile']:28} "
        f"{row['mean_kl_native_to_vllm']:13.8f} "
        f"{row['documented_mean_kl']:13.8f}  {result}"
    )
print(f"Summary JSON: {summary_path}")
if failed_profiles:
    raise SystemExit(
        "Recommended profiles above KL threshold: " + ", ".join(failed_profiles)
    )
PY
}

if ((DRY_RUN)); then
    for profile in "${PROFILE_ARRAY[@]}"; do
        run_profile_logged "${profile}" "${PROFILE_GPU[${profile}]}"
    done
    echo
    echo "Recommended-config dry-run complete."
    exit 0
fi

PIDS=()
for gpu in "${GPU_ARRAY[@]}"; do
    if [[ -z "${GPU_PROFILES[${gpu}]:-}" ]]; then
        continue
    fi
    run_gpu_worker "${gpu}" &
    PIDS+=("$!")
done

STATUS=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        STATUS=1
    fi
done

echo
if ((STATUS)); then
    echo "One or more recommended-config profiles failed." >&2
    exit "${STATUS}"
fi
summarize_results
echo "Recommended-config reproduction complete: ${OUTPUT_ROOT}"