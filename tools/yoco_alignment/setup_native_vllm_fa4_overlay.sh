#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PROBE="${SCRIPT_DIR}/logprob_kl.py"

NATIVE_PYTHON_BIN=${NATIVE_PYTHON_BIN:-"${REPO_ROOT}/../../.venv-yoco-native/bin/python"}
OVERLAY_DIR=${NATIVE_VLLM_FA4_OVERLAY:-"${REPO_ROOT}/../.native-vllm-fa4-overlay"}

usage() {
    cat <<'EOF'
Install the exact compiler/runtime dependency overlay for vLLM's vendored FA4.

Usage:
  setup_native_vllm_fa4_overlay.sh [options]

Options:
  --native-python PATH  Native Python executable.
  --overlay-dir PATH    New isolated overlay directory.
  -h, --help            Show this help.

The installer uses --no-deps so Native Torch, TransformerEngine, and DeepGEMM
remain untouched. It refuses to overwrite an existing overlay.
EOF
}

while (($#)); do
    case "$1" in
        --native-python)
            NATIVE_PYTHON_BIN=${2:?"--native-python requires a value"}
            shift 2
            ;;
        --overlay-dir)
            OVERLAY_DIR=${2:?"--overlay-dir requires a value"}
            shift 2
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

if [[ ! -x "${NATIVE_PYTHON_BIN}" ]]; then
    echo "Native Python is not executable: ${NATIVE_PYTHON_BIN}" >&2
    exit 2
fi
if [[ -e "${OVERLAY_DIR}" ]]; then
    echo "Overlay already exists; refusing to overwrite: ${OVERLAY_DIR}" >&2
    exit 2
fi

mkdir -p "$(dirname -- "${OVERLAY_DIR}")"
TEMP_DIR=$(mktemp -d "${OVERLAY_DIR}.tmp.XXXXXX")
trap 'rm -rf -- "${TEMP_DIR}"' EXIT
SITE_PACKAGES="${TEMP_DIR}/site-packages"
mkdir -p "${SITE_PACKAGES}"

"${NATIVE_PYTHON_BIN}" -m pip install \
    --disable-pip-version-check \
    --no-deps \
    --target "${SITE_PACKAGES}" \
    'apache-tvm-ffi==0.1.9' \
    'nvidia-cutlass-dsl==4.4.2' \
    'nvidia-cutlass-dsl-libs-base==4.4.2' \
    'nvidia-cutlass-dsl-libs-core==4.6.0' \
    'nvidia-cutlass-dsl-libs-cu13==4.4.2' \
    'quack-kernels==0.4.1'

PYTHONPATH="${SITE_PACKAGES}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
PROBE="${PROBE}" \
"${NATIVE_PYTHON_BIN}" - <<'PY'
import os
import runpy

import cutlass
import torch
import tvm_ffi
from importlib.metadata import version

assert cutlass.__version__ == "4.4.2", cutlass.__version__
assert tvm_ffi.__version__ == "0.1.9", tvm_ffi.__version__
assert version("quack-kernels") == "0.4.1"
namespace = runpy.run_path(os.environ["PROBE"])
function = namespace["_import_fa4_varlen_func"]("vllm-vendored")
assert callable(function)
print(
    "Validated Native vendored-FA4 overlay: "
    f"torch={torch.__version__}, cutlass={cutlass.__version__}, "
    f"tvm_ffi={tvm_ffi.__version__}, quack={version('quack-kernels')}"
)
PY

mv -- "${TEMP_DIR}" "${OVERLAY_DIR}"
trap - EXIT
echo "Native vendored-FA4 overlay installed at ${OVERLAY_DIR}"