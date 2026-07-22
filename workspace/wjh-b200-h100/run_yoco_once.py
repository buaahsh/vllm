#!/usr/bin/env python
"""Run one local YOCO inference with vLLM.

Execute this file from:
    /home/v-jiahaowang/workspace/vllm/workspace/wjh-b200-h100
"""

from __future__ import annotations

import inspect
import gc
import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = Path("/data/wjh/0000-6000-hf-gpu")


def _resolve_model_path() -> Path:
    override = os.environ.get("YOCO_MODEL_PATH")
    if override:
        return Path(override)
    if (DEFAULT_MODEL_ROOT / "config.json").exists():
        return DEFAULT_MODEL_ROOT
    nested = DEFAULT_MODEL_ROOT / DEFAULT_MODEL_ROOT.name
    if (nested / "config.json").exists():
        return nested
    return DEFAULT_MODEL_ROOT


def _add_repo_to_path() -> None:
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _configure_cuda_toolkit() -> None:
    if shutil.which("nvcc"):
        return

    for cuda_home in (Path("/usr/local/cuda-13.3"), Path("/usr/local/cuda-13")):
        nvcc = cuda_home / "bin" / "nvcc"
        if nvcc.exists():
            os.environ.setdefault("CUDA_HOME", str(cuda_home))
            os.environ["PATH"] = f"{cuda_home / 'bin'}:{os.environ.get('PATH', '')}"
            return


def _supported_kwargs(cls, kwargs: dict) -> dict:
    signature = inspect.signature(cls.__init__)
    params = signature.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in params}


def _shutdown_llm(llm: object | None) -> None:
    if llm is None:
        return
    llm_engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(llm_engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if shutdown is not None:
        shutdown(timeout=5)
    gc.collect()


def main() -> int:
    if Path.cwd().name != "wjh-b200-h100":
        raise RuntimeError(f"Run from wjh-b200-h100, current cwd is {Path.cwd()}")

    model_path = _resolve_model_path()
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(
            f"Model config not found under {model_path}. "
            "Set YOCO_MODEL_PATH to the downloaded HF checkpoint directory."
        )

    _add_repo_to_path()
    _configure_cuda_toolkit()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")

    from vllm import LLM, SamplingParams

    llm_kwargs = {
        "model": str(model_path),
        "trust_remote_code": True,
        "tensor_parallel_size": int(os.environ.get("YOCO_TP", "1")),
        "max_model_len": int(os.environ.get("YOCO_MAX_MODEL_LEN", "1024")),
        "max_num_seqs": int(os.environ.get("YOCO_MAX_NUM_SEQS", "1")),
        "max_num_batched_tokens": int(os.environ.get("YOCO_MAX_BATCHED_TOKENS", "1024")),
        "gpu_memory_utilization": float(os.environ.get("YOCO_GPU_MEMORY_UTIL", "0.90")),
        "quantization": os.environ.get("YOCO_QUANTIZATION", "fp8_per_block"),
        "moe_backend": os.environ.get("YOCO_MOE_BACKEND", "deep_gemm"),
        "enforce_eager": os.environ.get("YOCO_ENFORCE_EAGER", "1") != "0",
        "enable_chunked_prefill": False,
    }
    llm = None
    try:
        llm = LLM(**_supported_kwargs(LLM, llm_kwargs))

        prompt = os.environ.get(
            "YOCO_PROMPT",
            "用一句话介绍你自己，并说明你正在本地通过 vLLM 运行。",
        )
        sampling = SamplingParams(
            temperature=float(os.environ.get("YOCO_TEMPERATURE", "0.7")),
            top_p=float(os.environ.get("YOCO_TOP_P", "0.95")),
            max_tokens=int(os.environ.get("YOCO_MAX_TOKENS", "32")),
        )

        outputs = llm.generate([prompt], sampling)
        text = outputs[0].outputs[0].text
        print("PROMPT:")
        print(prompt)
        print("\nOUTPUT:")
        print(text)
        return 0
    finally:
        _shutdown_llm(llm)


if __name__ == "__main__":
    raise SystemExit(main())
