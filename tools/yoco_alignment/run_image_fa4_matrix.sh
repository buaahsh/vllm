#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
RUNNER="${SCRIPT_DIR}/run_logprob_kl_compare.sh"

MODEL=""
NATIVE_CHECKPOINT=""
OVERLAY_DIR=${YOCO_IMAGE_FA4_OVERLAY:-"${REPO_ROOT}/../.image-fa4-4.0.0b13"}
OUTPUT_ROOT="${REPO_ROOT}/../logs/yoco_alignment_results/image_fa4_4.0.0b13"
PROMPT_SUITE="mixed16"
BATCH_SIZE=16
ARMS="default,qkv-bf16,no-ex2,pack-gqa-off,tile-n64,tile-n64-qkv-bf16,tile-n128,tile-n128-qkv-bf16,q-stage1,q-stage2"
DRY_RUN=0

usage() {
    cat <<'EOF'
Run matched YOCO prefill matrices with the FA4 stack installed by the 26.02
B200 project image setup.

Usage:
  run_image_fa4_matrix.sh --model MODEL --native-checkpoint DIR [options]

Options:
  --overlay-dir PATH   Output of setup_image_fa4_profiles.sh.
  --output-root PATH   Parent result directory.
  --prompt-suite NAME  Prompt suite (default: mixed16).
  --batch-size N       Matched batch size (default: 16).
    --arms LIST          default,qkv-bf16,no-ex2,pack-gqa-off,tile-n64,
                           tile-n64-qkv-bf16,tile-n128,tile-n128-qkv-bf16,
                           q-stage1,q-stage2.
  --dry-run            Print commands without model execution.
  -h, --help           Show this help.

Arms:
  default    BF16 and all-MXFP8 with target FA4 defaults.
  qkv-bf16  MXFP8 with every attention-input projection kept BF16.
  no-ex2     All-MXFP8 with target FA4 exp2 emulation disabled.
    pack-gqa-off  All-MXFP8 with FA4 packed-GQA disabled.
    tile-n64      All-MXFP8 with FA4 tile_mn forced to 128x64.
    tile-n64-qkv-bf16  QKV-BF16 MXFP8 with tile_mn forced to 128x64.
    tile-n128     All-MXFP8 with FA4 tile_mn forced to 128x128.
    tile-n128-qkv-bf16 QKV-BF16 MXFP8 with tile_mn forced to 128x128.
    q-stage1      All-MXFP8 with q_stage forced to 1 and tile_mn=128x128.
    q-stage2      All-MXFP8 with q_stage forced to 2 and tile_mn=128x128.
EOF
}

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
        --overlay-dir)
            OVERLAY_DIR=${2:?"--overlay-dir requires a value"}
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT=${2:?"--output-root requires a value"}
            shift 2
            ;;
        --prompt-suite)
            PROMPT_SUITE=${2:?"--prompt-suite requires a value"}
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE=${2:?"--batch-size requires a value"}
            shift 2
            ;;
        --arms)
            ARMS=${2:?"--arms requires a value"}
            shift 2
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
    exit 2
fi
if ((BATCH_SIZE < 1)); then
    echo "--batch-size must be positive." >&2
    exit 2
fi

SITE_PACKAGES="${OVERLAY_DIR}/site-packages"
CUTLASS_PACKAGES="${SITE_PACKAGES}/nvidia_cutlass_dsl/python_packages"
DEFAULT_SOURCE="${SITE_PACKAGES}"
NO_EX2_SOURCE="${OVERLAY_DIR}/profiles/no-ex2"
for path in \
    "${CUTLASS_PACKAGES}/cutlass/__init__.py" \
    "${DEFAULT_SOURCE}/flash_attn/cute/interface.py" \
    "${NO_EX2_SOURCE}/flash_attn/cute/interface.py"; do
    if [[ ! -f "${path}" ]]; then
        echo "FA4 overlay is incomplete: ${path}" >&2
        echo "Run tools/yoco_alignment/setup_image_fa4_profiles.sh first." >&2
        exit 2
    fi
done

has_arm() {
    [[ ",${ARMS}," == *",$1,"* ]]
}

for arm in ${ARMS//,/ }; do
    case "${arm}" in
        default|qkv-bf16|no-ex2|pack-gqa-off|tile-n64|tile-n64-qkv-bf16|tile-n128|tile-n128-qkv-bf16|q-stage1|q-stage2) ;;
        *)
            echo "Unknown arm in --arms: ${arm}" >&2
            exit 2
            ;;
    esac
done

run_arm() {
    local profile=$1
    local source_root=$2
    local output_dir=$3
    local pack_gqa=$4
    local tile_mn=$5
    shift 5
    local args=(
        --model "${MODEL}"
        --native-checkpoint "${NATIVE_CHECKPOINT}"
        --attention-version 4
        --prompt-suite "${PROMPT_SUITE}"
        --batch-size "${BATCH_SIZE}"
        --output-dir "${output_dir}"
        "$@"
    )
    if ((DRY_RUN)); then
        args+=(--dry-run)
    fi
    YOCO_FA4_SOURCE_ROOT="${source_root}" \
    YOCO_FA4_PROFILE="${profile}" \
    YOCO_FA4_PACK_GQA="${pack_gqa}" \
    YOCO_FA4_TILE_MN="${tile_mn}" \
    PYTHONPATH="${CUTLASS_PACKAGES}:${source_root}:${SITE_PACKAGES}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        bash "${RUNNER}" "${args[@]}"
}

if has_arm default; then
    run_arm default "${DEFAULT_SOURCE}" "${OUTPUT_ROOT}/default" auto default \
        --variants bf16,fp8
fi
if has_arm qkv-bf16; then
    run_arm default "${DEFAULT_SOURCE}" "${OUTPUT_ROOT}/qkv_bf16" auto default \
        --variants fp8 \
        --keep-attention-qkv-bf16
fi
if has_arm no-ex2; then
    run_arm no-ex2 "${NO_EX2_SOURCE}" "${OUTPUT_ROOT}/no_ex2" auto default \
        --variants fp8
fi
if has_arm pack-gqa-off; then
    run_arm default "${DEFAULT_SOURCE}" "${OUTPUT_ROOT}/pack_gqa_off" off default \
        --variants fp8
fi
if has_arm tile-n64; then
    run_arm default "${DEFAULT_SOURCE}" "${OUTPUT_ROOT}/tile_n64" auto 128x64 \
        --variants fp8
fi
if has_arm tile-n64-qkv-bf16; then
    run_arm default "${DEFAULT_SOURCE}" "${OUTPUT_ROOT}/tile_n64_qkv_bf16" auto 128x64 \
        --variants fp8 \
        --keep-attention-qkv-bf16
fi
if has_arm tile-n128; then
    run_arm default "${DEFAULT_SOURCE}" "${OUTPUT_ROOT}/tile_n128" auto 128x128 \
        --variants fp8
fi
if has_arm tile-n128-qkv-bf16; then
    run_arm default "${DEFAULT_SOURCE}" "${OUTPUT_ROOT}/tile_n128_qkv_bf16" auto 128x128 \
        --variants fp8 \
        --keep-attention-qkv-bf16
fi
if has_arm q-stage1; then
    run_arm q-stage1 "${OVERLAY_DIR}/profiles/q-stage1" "${OUTPUT_ROOT}/q_stage1" auto 128x128 \
        --variants fp8
fi
if has_arm q-stage2; then
    run_arm q-stage2 "${OVERLAY_DIR}/profiles/q-stage2" "${OUTPUT_ROOT}/q_stage2" auto 128x128 \
        --variants fp8
fi

echo
echo "Image-derived FA4 matrix complete: ${OUTPUT_ROOT}"