#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"${REPO_ROOT}/../../.venv-yoco-mxfp8/bin/python"}
OVERLAY_DIR=${YOCO_IMAGE_FA4_OVERLAY:-"${REPO_ROOT}/../.image-fa4-4.0.0b13"}
IMAGE_MANIFEST_SHA256="cdbdc71b773142a98a303d488816d2faf6b9d85d179a4639b92263fca6da4769"
DEFAULT_SOURCE_SHA256="edbe1f46fcd2ac531be02900ef6caf7269e449279a0dd23509f8cb47420cf369"
NO_EX2_SOURCE_SHA256="18f2bf4ffe0c4c6122fefe55d31e7ab9433aded50383d1532a0d2e6e9253709b"
Q_STAGE1_SOURCE_SHA256="ce1b26290d2a7587d6e99f35d413fe812ec25fb94aab66d3b40accb4483c842d"
Q_STAGE2_SOURCE_SHA256="e12768d636fa215c9ed91c131c83ca5f4c6e70d7b96d0dabacfae9a4f68cf99d"
RECREATE=0

usage() {
    cat <<'EOF'
Build isolated source profiles for the FA4 stack baked into
donglixp/pytorch:26.02-b200: flash-attn-4 4.0.0b13, CUTLASS DSL 4.5.1,
TVM FFI 0.1.11, and Quack 0.4.1.

Usage:
  setup_image_fa4_profiles.sh [options]

Options:
  --python PATH       Python used to download/install pure Python wheels.
  --overlay-dir PATH  Destination directory.
  --recreate          Replace an existing overlay.
  -h, --help          Show this help.
EOF
}

while (($#)); do
    case "$1" in
        --python)
            PYTHON_BIN=${2:?"--python requires a value"}
            shift 2
            ;;
        --overlay-dir)
            OVERLAY_DIR=${2:?"--overlay-dir requires a value"}
            shift 2
            ;;
        --recreate)
            RECREATE=1
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
CUTLASS_PACKAGES="${SITE_PACKAGES}/nvidia_cutlass_dsl/python_packages"
NO_EX2_ROOT="${TEMP_DIR}/profiles/no-ex2"
Q_STAGE1_ROOT="${TEMP_DIR}/profiles/q-stage1"
Q_STAGE2_ROOT="${TEMP_DIR}/profiles/q-stage2"
mkdir -p \
    "${SITE_PACKAGES}" \
    "${NO_EX2_ROOT}/flash_attn" \
    "${Q_STAGE1_ROOT}/flash_attn" \
    "${Q_STAGE2_ROOT}/flash_attn"

"${PYTHON_BIN}" -m pip install \
    --disable-pip-version-check \
    --no-deps \
    --target "${SITE_PACKAGES}" \
    'cuda-bindings==13.3.1' \
    'cuda-core==1.0.1' \
    'cuda-pathfinder==1.5.6' \
    'cuda-python==13.3.1' \
    'apache-tvm-ffi==0.1.11' \
    'nvidia-cutlass-dsl-libs-base==4.5.1' \
    'nvidia-cutlass-dsl==4.5.1' \
    'torch-c-dlpack-ext==0.1.5' \
    'quack-kernels==0.4.1' \
    'flash-attn-4==4.0.0b13'

cp -R --no-preserve=mode,ownership,timestamps \
    "${SITE_PACKAGES}/flash_attn/cute" \
    "${NO_EX2_ROOT}/flash_attn/cute"
cp -R --no-preserve=mode,ownership,timestamps \
    "${SITE_PACKAGES}/flash_attn/cute" \
    "${Q_STAGE1_ROOT}/flash_attn/cute"
cp -R --no-preserve=mode,ownership,timestamps \
    "${SITE_PACKAGES}/flash_attn/cute" \
    "${Q_STAGE2_ROOT}/flash_attn/cute"

NO_EX2_SOURCE="${NO_EX2_ROOT}/flash_attn/cute/flash_fwd_sm100.py" \
"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["NO_EX2_SOURCE"])
text = path.read_text(encoding="utf-8")
old = "        cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE\n"
new = (
    "        self.enable_ex2_emu = False\n"
    "        self.ex2_emu_freq = 0\n"
    + old
)
if text.count(old) != 1:
    raise RuntimeError(
        f"Expected one FA4 exp2 override insertion point, found {text.count(old)}"
    )
path.write_text(text.replace(old, new), encoding="utf-8")
PY

Q_STAGE1_SOURCE="${Q_STAGE1_ROOT}/flash_attn/cute/interface.py" \
Q_STAGE2_SOURCE="${Q_STAGE2_ROOT}/flash_attn/cute/interface.py" \
"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

old = "        q_stage = 2 if seqlen_q_packgqa > tile_m else 1\n"
for stage, env_name in ((1, "Q_STAGE1_SOURCE"), (2, "Q_STAGE2_SOURCE")):
    path = Path(os.environ[env_name])
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(
            f"Expected one q_stage heuristic in {path}, found {text.count(old)}"
        )
    path.write_text(
        text.replace(old, f"        q_stage = {stage}\n"),
        encoding="utf-8",
    )
PY

SITE_PACKAGES="${SITE_PACKAGES}" NO_EX2_ROOT="${NO_EX2_ROOT}" \
Q_STAGE1_ROOT="${Q_STAGE1_ROOT}" Q_STAGE2_ROOT="${Q_STAGE2_ROOT}" \
DEFAULT_SOURCE_SHA256="${DEFAULT_SOURCE_SHA256}" \
NO_EX2_SOURCE_SHA256="${NO_EX2_SOURCE_SHA256}" \
Q_STAGE1_SOURCE_SHA256="${Q_STAGE1_SOURCE_SHA256}" \
Q_STAGE2_SOURCE_SHA256="${Q_STAGE2_SOURCE_SHA256}" \
PYTHONPATH="${CUTLASS_PACKAGES}:${SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON_BIN}" - <<'PY'
import hashlib
import importlib.metadata as metadata
import os
from pathlib import Path

import cutlass

expected = {
    "flash-attn-4": "4.0.0b13",
    "nvidia-cutlass-dsl": "4.5.1",
    "apache-tvm-ffi": "0.1.11",
    "quack-kernels": "0.4.1",
}
actual = {name: metadata.version(name) for name in expected}
if actual != expected:
    raise RuntimeError(f"FA4 overlay mismatch: expected {expected}, got {actual}")

site_packages = Path(os.environ["SITE_PACKAGES"]).resolve()
if not Path(cutlass.__file__).resolve().is_relative_to(site_packages):
    raise RuntimeError(f"CUTLASS imported outside the overlay: {cutlass.__file__}")
default_root = site_packages / "flash_attn" / "cute"
no_ex2_root = Path(os.environ["NO_EX2_ROOT"]) / "flash_attn" / "cute"
q_stage1_root = Path(os.environ["Q_STAGE1_ROOT"]) / "flash_attn" / "cute"
q_stage2_root = Path(os.environ["Q_STAGE2_ROOT"]) / "flash_attn" / "cute"

def digest(root: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        hasher.update(path.relative_to(root).as_posix().encode())
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
    return hasher.hexdigest()

default_files = {p.relative_to(default_root) for p in default_root.rglob("*.py")}
for profile_name, profile_root in (
    ("no-ex2", no_ex2_root),
    ("q-stage1", q_stage1_root),
    ("q-stage2", q_stage2_root),
):
    profile_files = {p.relative_to(profile_root) for p in profile_root.rglob("*.py")}
    if default_files != profile_files:
        raise RuntimeError(
            f"Default and {profile_name} FA4 source profiles have different files"
        )

changed = []
for relative_path in sorted(default_files):
    if (default_root / relative_path).read_bytes() != (no_ex2_root / relative_path).read_bytes():
        changed.append(relative_path.as_posix())
if changed != ["flash_fwd_sm100.py"]:
    raise RuntimeError(f"Unexpected no-ex2 source changes: {changed}")

for profile_name, profile_root in (
    ("q-stage1", q_stage1_root),
    ("q-stage2", q_stage2_root),
):
    changed = [
        relative_path.as_posix()
        for relative_path in sorted(default_files)
        if (default_root / relative_path).read_bytes()
        != (profile_root / relative_path).read_bytes()
    ]
    if changed != ["interface.py"]:
        raise RuntimeError(f"Unexpected {profile_name} source changes: {changed}")

no_ex2_text = (no_ex2_root / "flash_fwd_sm100.py").read_text(encoding="utf-8")
needle = "self.enable_ex2_emu = False\n        self.ex2_emu_freq = 0"
if no_ex2_text.count(needle) != 1:
    raise RuntimeError("No-ex2 profile does not contain the expected override")

default_digest = digest(default_root)
no_ex2_digest = digest(no_ex2_root)
q_stage1_digest = digest(q_stage1_root)
q_stage2_digest = digest(q_stage2_root)
expected_default = os.environ["DEFAULT_SOURCE_SHA256"]
expected_no_ex2 = os.environ["NO_EX2_SOURCE_SHA256"]
expected_q_stage1 = os.environ["Q_STAGE1_SOURCE_SHA256"]
expected_q_stage2 = os.environ["Q_STAGE2_SOURCE_SHA256"]
if (
    default_digest != expected_default
    or no_ex2_digest != expected_no_ex2
    or q_stage1_digest != expected_q_stage1
    or q_stage2_digest != expected_q_stage2
):
    raise RuntimeError(
        "FA4 source hash mismatch: "
        f"default={default_digest} expected={expected_default}, "
        f"no-ex2={no_ex2_digest} expected={expected_no_ex2}, "
        f"q-stage1={q_stage1_digest} expected={expected_q_stage1}, "
        f"q-stage2={q_stage2_digest} expected={expected_q_stage2}"
    )

print(f"Target FA4 packages: {actual}")
print(f"Default source SHA256: {default_digest}")
print(f"No-exp2 source SHA256: {no_ex2_digest}")
print(f"Q-stage1 source SHA256: {q_stage1_digest}")
print(f"Q-stage2 source SHA256: {q_stage2_digest}")
PY

mv -- "${TEMP_DIR}" "${OVERLAY_DIR}"
trap - EXIT
echo "Image-derived FA4 profiles installed at ${OVERLAY_DIR}"
echo "  image manifest: sha256:${IMAGE_MANIFEST_SHA256}"
echo "  default source: ${OVERLAY_DIR}/site-packages"
echo "  no-exp2 source: ${OVERLAY_DIR}/profiles/no-ex2"
echo "  q-stage1 source: ${OVERLAY_DIR}/profiles/q-stage1"
echo "  q-stage2 source: ${OVERLAY_DIR}/profiles/q-stage2"
echo "  CUTLASS path: ${OVERLAY_DIR}/site-packages/nvidia_cutlass_dsl/python_packages"