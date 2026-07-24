#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"${REPO_ROOT}/../../.venv-yoco-mxfp8/bin/python"}
OVERLAY_DIR=${YOCO_LATEST_FA4_OVERLAY:-"${REPO_ROOT}/../.latest-fa4-4.0.0b23"}
RECREATE=0

usage() {
    cat <<'EOF'
Build an isolated overlay for the latest compatible FA4 stack:
flash-attn-4 4.0.0b23, CUTLASS DSL 4.6.0.dev0, TVM FFI 0.1.12,
and Quack 0.5.3.

Usage:
  setup_latest_fa4_overlay.sh [options]

Options:
  --python PATH       Python used to install the overlay packages.
  --overlay-dir PATH  Destination directory.
  --recreate          Replace an existing overlay.
  -h, --help          Show this help.
EOF
}

while (($#)); do
    case "$1" in
        --python) PYTHON_BIN=${2:?"--python requires a value"}; shift 2 ;;
        --overlay-dir) OVERLAY_DIR=${2:?"--overlay-dir requires a value"}; shift 2 ;;
        --recreate) RECREATE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python is not executable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ -e "${OVERLAY_DIR}" ]]; then
    if ((RECREATE)); then
        rm -rf -- "${OVERLAY_DIR}"
    else
        echo "Overlay already exists: ${OVERLAY_DIR}" >&2
        echo "Use --recreate to replace it." >&2
        exit 2
    fi
fi

mkdir -p "$(dirname -- "${OVERLAY_DIR}")"
TEMP_DIR=$(mktemp -d "${OVERLAY_DIR}.tmp.XXXXXX")
trap 'rm -rf -- "${TEMP_DIR}"' EXIT
SITE_PACKAGES="${TEMP_DIR}/site-packages"
mkdir -p "${SITE_PACKAGES}"

PIP_CONSTRAINT= "${PYTHON_BIN}" -m pip install \
    --disable-pip-version-check \
    --no-deps \
    --target "${SITE_PACKAGES}" \
    'cuda-bindings==13.3.1' \
    'cuda-core==1.0.1' \
    'cuda-pathfinder==1.6.0' \
    'cuda-python==13.3.1' \
    'apache-tvm-ffi==0.1.12' \
    'nvidia-cutlass-dsl-libs-base==4.6.0.dev0' \
    'nvidia-cutlass-dsl-libs-cu13==4.6.0.dev0' \
    'nvidia-cutlass-dsl==4.6.0.dev0' \
    'torch-c-dlpack-ext==0.1.5' \
    'quack-kernels==0.5.3' \
    'flash-attn-4==4.0.0b23'

SITE_PACKAGES="${SITE_PACKAGES}" \
PYTHONPATH="${SITE_PACKAGES}/nvidia_cutlass_dsl/python_packages:${SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON_BIN}" - <<'PY'
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path

import cutlass
import tvm_ffi

site_packages = Path(os.environ["SITE_PACKAGES"]).resolve()
expected = {
    "flash-attn-4": "4.0.0b23",
    "nvidia-cutlass-dsl": "4.6.0.dev0",
    "nvidia-cutlass-dsl-libs-base": "4.6.0.dev0",
    "nvidia-cutlass-dsl-libs-cu13": "4.6.0.dev0",
    "apache-tvm-ffi": "0.1.12",
    "quack-kernels": "0.5.3",
    "cuda-bindings": "13.3.1",
    "cuda-core": "1.0.1",
    "cuda-pathfinder": "1.6.0",
    "cuda-python": "13.3.1",
    "torch-c-dlpack-ext": "0.1.5",
}
actual = {name: metadata.version(name) for name in expected}
if actual != expected:
    raise RuntimeError(f"FA4 b23 overlay mismatch: expected {expected}, got {actual}")
for module_path in (Path(cutlass.__file__).resolve(), Path(tvm_ffi.__file__).resolve()):
    if not module_path.is_relative_to(site_packages):
        raise RuntimeError(f"Overlay module imported from outside target: {module_path}")

cute_root = site_packages / "flash_attn" / "cute"
if not (cute_root / "interface.py").is_file():
    raise RuntimeError(f"FA4 b23 source is missing: {cute_root}")
hasher = hashlib.sha256()
for path in sorted(cute_root.rglob("*.py")):
    hasher.update(path.relative_to(cute_root).as_posix().encode())
    hasher.update(b"\0")
    hasher.update(path.read_bytes())

manifest = {
    "packages": actual,
    "source_root": str(cute_root.relative_to(site_packages.parent)),
    "source_sha256": hasher.hexdigest(),
    "cutlass_module": str(
        Path(cutlass.__file__).resolve().relative_to(site_packages.parent)
    ),
    "tvm_ffi_module": str(
        Path(tvm_ffi.__file__).resolve().relative_to(site_packages.parent)
    ),
}
(site_packages.parent / "manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, indent=2))
PY

mv -- "${TEMP_DIR}" "${OVERLAY_DIR}"
trap - EXIT
echo "Latest FA4 overlay installed at ${OVERLAY_DIR}"
