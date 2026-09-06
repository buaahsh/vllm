#!/usr/bin/env bash
set -euo pipefail

target=${1:?usage: install_decode_flashinfer.sh /absolute/overlay/path}
python_bin=${PYTHON_BIN:-python}

mkdir -p "${target}"
"${python_bin}" -m pip install --no-deps --target "${target}" \
  flashinfer-python==0.6.18

PYTHONPATH="${target}${PYTHONPATH:+:${PYTHONPATH}}" "${python_bin}" - <<'PY'
import inspect

import flashinfer
from flashinfer.fused_moe import trtllm_bf16_routed_moe

required = {
    "gemm1_alpha",
    "gemm1_beta",
    "gemm1_clamp_limit",
    "output",
}
parameters = inspect.signature(trtllm_bf16_routed_moe).parameters
missing = required.difference(parameters)
if missing:
    raise RuntimeError(f"FlashInfer {flashinfer.__version__} misses {sorted(missing)}")
print(f"YOCO Decode FlashInfer overlay ready: {flashinfer.__version__}")
PY
