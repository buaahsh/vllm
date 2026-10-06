# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve an exported SWA8K draft with consistent CUDA graph capture settings."""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

DEFAULT_TARGET = "/mnt/pvc/shaohanh/exp/agens/30A3B/merged/balanced-b040-c035-d025-hf"
DEFAULT_DRAFT = "/mnt/pvc/lidong1/sharedkv-swa8k-serving-20261006/draft-step-007750-pvc"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", default=os.environ.get("TARGET_MODEL", DEFAULT_TARGET)
    )
    parser.add_argument("--draft", default=os.environ.get("SWA_DRAFT", DEFAULT_DRAFT))
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--spec-tokens", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default="yoco-swa8k")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without loading models",
    )
    args = parser.parse_args()
    if not 1 <= args.max_num_seqs <= 128:
        parser.error("--max-num-seqs must be between 1 and 128")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    # Match the regular capture ladder and include the exact verification cap.
    # The CLI may otherwise truncate a cap such as 36 to a default list ending at 32.
    cap = args.max_num_seqs * (args.spec_tokens + 1)
    sizes = sorted(
        {n for n in [1, 2, 4] if n <= cap}
        | set(range(8, min(cap + 1, 256), 8))
        | set(range(256, cap + 1, 16))
        | {cap}
    )
    compilation = {
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes": sizes,
    }
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.target,
        "--served-model-name",
        args.served_model_name,
        "--spec-model",
        args.draft,
        "--spec-method",
        "dspark",
        "--spec-tokens",
        str(args.spec_tokens),
        "--speculative-config",
        json.dumps({"draft_sample_method": "probabilistic"}),
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "131072",
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        "8192",
        "--gpu-memory-utilization",
        "0.75",
        "--no-enable-prefix-caching",
        "--attention-config",
        json.dumps({"backend": "FLASHINFER"}),
        "--compilation-config",
        json.dumps(compilation),
        "--max-cudagraph-capture-size",
        str(cap),
        "--shutdown-timeout",
        "90",
        "--renderer-num-workers",
        "1",
        "--kernel-config",
        json.dumps(
            {
                "enable_jit_warmup": False,
                "enable_cutedsl_warmup": False,
                "enable_flashinfer_autotune": False,
            }
        ),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    print(
        "VLLM_USE_V2_MODEL_RUNNER=1 VLLM_BATCH_INVARIANT=0 " + shlex.join(command),
        flush=True,
    )
    if args.dry_run:
        return

    config_path = Path(args.draft) / "config.json"
    if (
        not config_path.is_file()
        or not (Path(args.draft) / "model.safetensors").is_file()
    ):
        parser.error("--draft must be an exported directory with config and weights")
    config = json.loads(config_path.read_text())
    if config.get("shared_kv_config", {}).get("draft_kv_window") != 8192:
        parser.error("--draft is not an exported SWA8192 checkpoint")
    if config.get("shared_kv_target") != str(Path(args.target).resolve()):
        parser.error("Target path differs from export; re-export with this --target")
    env = os.environ | {"VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_BATCH_INVARIANT": "0"}
    env.setdefault("OMP_NUM_THREADS", "4")
    os.execvpe(sys.executable, command, env)


if __name__ == "__main__":
    main()
