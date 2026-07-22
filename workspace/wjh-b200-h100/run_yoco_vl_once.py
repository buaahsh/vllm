#!/usr/bin/env python
"""Run one local YOCO-VL inference with vLLM.

Execute this file from:
    /home/v-jiahaowang/workspace/vllm/workspace/wjh-b200-h100
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import random
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = Path("/data/wjh/updates_3000-hf-vl")
DEFAULT_TOKENIZER_PATH = Path("/mnt/msranlp/yutao/hf_cache/agens_vl_tokenizer_0622")
DEFAULT_SYSTEM_PROMPT = "You are a helpful and friendly AI assistant."
DEFAULT_PROMPT = "Describe this image in detail."
IMAGE_PLACEHOLDER = "<image>"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "no", "off", "")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value is not None else None


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("YOCO_VL_MODEL_PATH", str(DEFAULT_MODEL_PATH)),
        help="HF/vLLM checkpoint directory.",
    )
    parser.add_argument(
        "--image",
        default=os.environ.get("YOCO_VL_IMAGE"),
        help="Local image path. If omitted, use a generated smoke-test image.",
    )
    parser.add_argument(
        "--prompt",
        default=os.environ.get("YOCO_VL_PROMPT", DEFAULT_PROMPT),
    )
    parser.add_argument(
        "--system_prompt",
        default=os.environ.get("YOCO_VL_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
    )
    parser.add_argument(
        "--tokenizer_path",
        default=os.environ.get("YOCO_VL_TOKENIZER_PATH", str(DEFAULT_TOKENIZER_PATH)),
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=_env_int("YOCO_MAX_NEW_TOKENS", _env_int("YOCO_MAX_TOKENS", 64)),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=_env_float("YOCO_TEMPERATURE", 0.0),
    )
    parser.add_argument("--top_p", type=float, default=_env_float("YOCO_TOP_P", 0.9))
    parser.add_argument("--seed", type=int, default=_env_int("YOCO_SEED", 1))
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=_env_flag("YOCO_ENABLE_THINKING", False),
    )
    parser.add_argument(
        "--quant_mode",
        choices=("checkpoint", "bfloat16", "mxfp8"),
        default=os.environ.get("YOCO_QUANT_MODE", "bfloat16"),
        help=(
            "Matches llm/vl_infer.py. bfloat16 disables vLLM quantization; "
            "mxfp8 maps to vLLM fp8_per_block."
        ),
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=_env_optional_int("YOCO_MAX_MODEL_LEN"),
        help=(
            "Defaults to the actual prompt token count plus max_new_tokens, "
            "matching llm/vl_infer.py's KV-cache length."
        ),
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=_env_int("YOCO_MAX_NUM_SEQS", 1),
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=_env_optional_int("YOCO_MAX_BATCHED_TOKENS"),
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=_env_float("YOCO_GPU_MEMORY_UTIL", 0.90),
    )
    parser.add_argument(
        "--vllm_flash_attn_version",
        type=int,
        choices=(2, 3, 4),
        default=_env_optional_int("YOCO_VLLM_FLASH_ATTN_VERSION"),
        help="Optional vLLM FlashAttention version override.",
    )
    return parser.parse_args()


def _add_repo_to_path() -> None:
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    pythonpath = [
        entry for entry in os.environ.get("PYTHONPATH", "").split(":") if entry
    ]
    if repo not in pythonpath:
        pythonpath.insert(0, repo)
    os.environ["PYTHONPATH"] = ":".join(pythonpath)


def _configure_cuda_toolkit() -> None:
    if (
        "TRITON_PTXAS_PATH" not in os.environ
        and Path("/usr/local/cuda/bin/ptxas").is_file()
    ):
        os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"

    cuda_include = Path("/usr/local/cuda/targets/x86_64-linux/include")
    if cuda_include.is_dir():
        cpath_entries = [
            entry for entry in os.environ.get("CPATH", "").split(":") if entry
        ]
        if str(cuda_include) not in cpath_entries:
            os.environ["CPATH"] = ":".join([str(cuda_include), *cpath_entries])

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


def _make_test_image() -> Image.Image:
    image = Image.new("RGB", (224, 224), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 36, 200, 188), outline=(20, 80, 180), width=8)
    draw.ellipse((76, 70, 148, 142), fill=(220, 40, 40))
    draw.line((36, 184, 188, 48), fill=(30, 150, 90), width=6)
    return image


def _load_image(image_path: str | None) -> tuple[Image.Image, str]:
    if image_path is None:
        return _make_test_image(), "<generated>"
    with Image.open(image_path) as image:
        return image.convert("RGB"), str(image_path)


def _shutdown_llm(llm: object | None) -> None:
    if llm is None:
        return
    llm_engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(llm_engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if shutdown is not None:
        shutdown(timeout=5)
    gc.collect()


def _load_tokenizer(tokenizer_path: Path):
    return AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)


def _render_prompt(
    tokenizer,
    prompt: str,
    system_prompt: str | None,
    enable_thinking: bool,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": f"{IMAGE_PLACEHOLDER}\n{prompt}"})

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        clear_thinking=not enable_thinking,
    )
    if not enable_thinking:
        rendered = rendered.replace(
            "<|assistant|><think></think>", "<|assistant|></think>"
        )
    if rendered.count(IMAGE_PLACEHOLDER) != 1:
        raise RuntimeError(
            f"Expected exactly one {IMAGE_PLACEHOLDER}, got: {rendered!r}"
        )
    return rendered


def _load_model_config(model_path: Path) -> dict:
    with (model_path / "config.json").open("r", encoding="utf-8") as reader:
        return json.load(reader)


def _checkpoint_quant_mode(model_path: Path) -> str:
    config = _load_model_config(model_path)
    return str(config.get("text_config", {}).get("quant_mode", "bfloat16"))


def _quantization_for_mode(model_path: Path, quant_mode: str) -> str | None:
    override = os.environ.get("YOCO_QUANTIZATION")
    if override is not None:
        return None if override.lower() in ("", "none", "false", "0") else override

    resolved = (
        _checkpoint_quant_mode(model_path)
        if quant_mode == "checkpoint"
        else quant_mode
    )
    if resolved == "bfloat16":
        return None
    if resolved == "mxfp8":
        return "fp8_per_block"
    raise ValueError(f"Unsupported quant mode for vLLM: {resolved!r}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _num_image_tokens(config: dict, image: Image.Image) -> int:
    width, height = image.size
    vision_config = config["vision_config"]
    patch_size = int(vision_config["patch_size"])
    merge_kernel_size = int(vision_config["merge_kernel_size"][0])
    patch_limit = int(config.get("vision_patch_limit_on_one_side", 512))
    max_image_tokens = config.get("vision_max_image_tokens")
    align_mode = config.get("vision_align_mode", "resize")
    if align_mode not in ("pad", "resize"):
        raise ValueError(f"Unsupported vision align_mode={align_mode!r}")

    patch_limit_total = 16384
    if max_image_tokens is not None:
        patch_limit_total = min(
            patch_limit_total, int(max_image_tokens) * merge_kernel_size**2
        )

    scale_by_area = math.sqrt(
        patch_limit_total
        / (
            max(1.0, width // patch_size)
            * max(1.0, height // patch_size)
        )
    )
    scale_by_width = patch_limit * patch_size / width
    scale_by_height = patch_limit * patch_size / height
    scale = min(1.0, scale_by_area, scale_by_width, scale_by_height)
    new_width = min(max(1, int(width * scale)), patch_limit * patch_size)
    new_height = min(max(1, int(height * scale)), patch_limit * patch_size)

    factor = merge_kernel_size * patch_size
    pad_height = (factor - new_height % factor) % factor
    pad_width = (factor - new_width % factor) % factor
    if align_mode == "resize":
        new_height += pad_height
        new_width += pad_width
        pad_height = 0
        pad_width = 0

    token_height = (new_height + pad_height) // factor
    token_width = (new_width + pad_width) // factor
    return token_height * token_width


def _estimate_prompt_tokens(
    tokenizer,
    config: dict,
    rendered_prompt: str,
    image: Image.Image,
) -> int:
    parts = rendered_prompt.split(IMAGE_PLACEHOLDER)
    if len(parts) != 2:
        raise ValueError(
            f"Expected one {IMAGE_PLACEHOLDER} placeholder, got {len(parts) - 1}"
        )

    bos_token = getattr(tokenizer, "bos_token", None)
    token_count = 0
    if not (
        isinstance(bos_token, str) and rendered_prompt.lstrip().startswith(bos_token)
    ):
        token_count += 1

    token_count += len(tokenizer.encode(parts[0], add_special_tokens=False))
    token_count += 1 + _num_image_tokens(config, image) + 1
    token_count += len(tokenizer.encode(parts[1], add_special_tokens=False))
    return token_count


def main() -> int:
    if Path.cwd().name != "wjh-b200-h100":
        raise RuntimeError(f"Run from wjh-b200-h100, current cwd is {Path.cwd()}")

    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top_p must be in (0, 1]")

    model_path = Path(args.model)
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"Model config not found under {model_path}")
    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")
    model_config = _load_model_config(model_path)

    _add_repo_to_path()
    _configure_cuda_toolkit()
    warnings.filterwarnings(
        "ignore",
        message=r"Use explicit `struct\.scalar\.ptr` for pointer instead\.",
        category=DeprecationWarning,
    )

    os.environ.setdefault(
        "CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "1")
    )
    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    _seed_everything(args.seed)
    tokenizer = _load_tokenizer(tokenizer_path)
    prompt = _render_prompt(
        tokenizer,
        args.prompt,
        args.system_prompt,
        args.enable_thinking,
    )
    image, image_source = _load_image(args.image)
    prompt_tokens = _estimate_prompt_tokens(tokenizer, model_config, prompt, image)
    max_model_len = args.max_model_len or (prompt_tokens + args.max_new_tokens)
    max_num_batched_tokens = args.max_num_batched_tokens or max_model_len

    from vllm import LLM, SamplingParams

    quantization = _quantization_for_mode(model_path, args.quant_mode)
    llm_kwargs = {
        "model": str(model_path),
        "tokenizer": str(tokenizer_path),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "seed": args.seed,
        "tensor_parallel_size": int(os.environ.get("YOCO_TP", "1")),
        "max_model_len": max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "quantization": quantization,
        "enforce_eager": os.environ.get("YOCO_ENFORCE_EAGER", "1") != "0",
        "enable_chunked_prefill": False,
        "limit_mm_per_prompt": {"image": 1},
    }
    moe_backend = os.environ.get("YOCO_MOE_BACKEND")
    if moe_backend is None and quantization is None:
        moe_backend = "triton"
    if moe_backend:
        llm_kwargs["moe_backend"] = moe_backend
    if args.vllm_flash_attn_version is not None:
        llm_kwargs["attention_config"] = {
            "flash_attn_version": args.vllm_flash_attn_version,
        }

    llm = None
    try:
        llm = LLM(**_supported_kwargs(LLM, llm_kwargs))

        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )

        request = {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }
        outputs = llm.generate([request], sampling)
        generated_text = outputs[0].outputs[0].text
        print("SETTINGS:")
        print(
            json.dumps(
                {
                    "model": str(model_path),
                    "tokenizer_path": str(tokenizer_path),
                    "image": image_source,
                    "image_size": list(image.size),
                    "prompt": args.prompt,
                    "system_prompt": args.system_prompt,
                    "max_new_tokens": args.max_new_tokens,
                    "prompt_tokens": prompt_tokens,
                    "max_model_len": max_model_len,
                    "max_num_batched_tokens": max_num_batched_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.seed,
                    "enable_thinking": args.enable_thinking,
                    "quant_mode": args.quant_mode,
                    "vllm_quantization": quantization,
                    "vllm_flash_attn_version": args.vllm_flash_attn_version,
                    "moe_backend": moe_backend,
                    "dtype": "bfloat16",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print("PROMPT:")
        print(prompt)
        print("OUTPUT:")
        print(generated_text)
        return 0
    finally:
        _shutdown_llm(llm)


if __name__ == "__main__":
    raise SystemExit(main())
