#!/usr/bin/env python
"""Run one local batch YOCO-VL inference with vLLM.

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
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = Path("/data/wjh/updates_3000-hf-vl")
DEFAULT_TOKENIZER_PATH = Path("/mnt/msranlp/yutao/hf_cache/agens_vl_tokenizer_0622")
DEFAULT_IMAGES = [
    "/home/v-jiahaowang/workspace/llm-train/workspace/dog1.jpeg",
    "/home/v-jiahaowang/workspace/llm-train/workspace/dog2.jpeg",
    "/home/v-jiahaowang/workspace/llm-train/workspace/dog3.jpeg",
]
DEFAULT_PROMPTS = [
    "Describe this dog and the surrounding scene in one concise sentence.",
    "What is the dog doing? Mention its pose and expression.",
    "List the most noticeable visual details in this image.",
]
DEFAULT_SYSTEM_PROMPT = "You are a helpful and friendly AI assistant."
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
        "--images",
        nargs="+",
        default=os.environ.get("YOCO_VL_BATCH_IMAGES", "").split()
        or DEFAULT_IMAGES,
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=os.environ.get("YOCO_VL_BATCH_PROMPTS", "").split("\t")
        if os.environ.get("YOCO_VL_BATCH_PROMPTS")
        else DEFAULT_PROMPTS,
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
            "Matches llm/vl_batch_infer.py. bfloat16 disables vLLM "
            "quantization; mxfp8 maps to vLLM fp8_per_block."
        ),
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=_env_optional_int("YOCO_MAX_MODEL_LEN"),
        help=(
            "Defaults to max prompt token count plus max_new_tokens, matching "
            "llm/vl_batch_infer.py's per-sample KV-cache length."
        ),
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=_env_optional_int("YOCO_MAX_BATCHED_TOKENS"),
        help=(
            "Defaults to sum(prompt_tokens) + batch_size * max_new_tokens, "
            "so the full batch can be scheduled together."
        ),
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=_env_int("YOCO_MAX_NUM_SEQS", 0),
        help="Defaults to the batch size.",
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
    parser.add_argument(
        "--enable_prefix_caching",
        action="store_true",
        default=_env_flag("YOCO_ENABLE_PREFIX_CACHING", False),
        help="Default is false to match the native batch script more closely.",
    )
    parser.add_argument("--output_json", default=os.environ.get("YOCO_OUTPUT_JSON"))
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


def _load_images(image_paths: list[str]) -> list[tuple[Image.Image, str]]:
    images: list[tuple[Image.Image, str]] = []
    for image_path in image_paths:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            images.append((image.convert("RGB"), str(path)))
    return images


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


def _default_stop_token_ids(tokenizer, config: dict) -> list[int]:
    stop_ids: set[int] = set()
    eos_from_config = config.get("eos_token_id") or config.get(
        "text_config", {}
    ).get("eos_token_id")
    if isinstance(eos_from_config, int) and eos_from_config >= 0:
        stop_ids.add(eos_from_config)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int) and eos_token_id >= 0:
        stop_ids.add(eos_token_id)

    for token in ("<|user|>", "<|assistant|>", "<|system|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            stop_ids.add(token_id)
    return sorted(stop_ids)


def main() -> int:
    if Path.cwd().name != "wjh-b200-h100":
        raise RuntimeError(f"Run from wjh-b200-h100, current cwd is {Path.cwd()}")

    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top_p must be in (0, 1]")
    if len(args.images) != len(args.prompts):
        raise ValueError(
            f"--images and --prompts must have the same length; got "
            f"{len(args.images)} images and {len(args.prompts)} prompts"
        )

    model_path = Path(args.model)
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"Model config not found under {model_path}")
    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")

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

    model_config = _load_model_config(model_path)
    tokenizer = _load_tokenizer(tokenizer_path)
    loaded_images = _load_images(args.images)
    prompts = [
        _render_prompt(
            tokenizer,
            prompt,
            args.system_prompt,
            args.enable_thinking,
        )
        for prompt in args.prompts
    ]
    prompt_tokens = [
        _estimate_prompt_tokens(tokenizer, model_config, prompt, image)
        for prompt, (image, _) in zip(prompts, loaded_images)
    ]
    image_tokens = [
        _num_image_tokens(model_config, image) for image, _ in loaded_images
    ]
    max_prompt_tokens = max(prompt_tokens)
    batch_size = len(prompt_tokens)
    max_model_len = args.max_model_len or (max_prompt_tokens + args.max_new_tokens)
    max_num_batched_tokens = args.max_num_batched_tokens or (
        sum(prompt_tokens) + batch_size * args.max_new_tokens
    )
    max_num_seqs = args.max_num_seqs or batch_size

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
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "quantization": quantization,
        "enforce_eager": os.environ.get("YOCO_ENFORCE_EAGER", "1") != "0",
        "enable_chunked_prefill": False,
        "enable_prefix_caching": args.enable_prefix_caching,
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

    stop_token_ids = _default_stop_token_ids(tokenizer, model_config)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=True,
    )

    llm = None
    start = time.perf_counter()
    try:
        llm = LLM(**_supported_kwargs(LLM, llm_kwargs))

        requests = [
            {
                "prompt": rendered_prompt,
                "multi_modal_data": {"image": image},
            }
            for rendered_prompt, (image, _) in zip(prompts, loaded_images)
        ]
        outputs = llm.generate(requests, sampling)
        elapsed = time.perf_counter() - start

        results = []
        for index, (sample_output, prompt, (image, image_source)) in enumerate(
            zip(outputs, args.prompts, loaded_images)
        ):
            completion = sample_output.outputs[0]
            results.append(
                {
                    "index": index,
                    "image": image_source,
                    "image_size": list(image.size),
                    "image_tokens": image_tokens[index],
                    "prompt": prompt,
                    "rendered_prompt": prompts[index],
                    "prompt_tokens": prompt_tokens[index],
                    "generated_ids": list(completion.token_ids),
                    "generated_text": completion.text,
                    "finish_reason": completion.finish_reason,
                    "stop_reason": completion.stop_reason,
                }
            )

        payload = {
            "model": str(model_path),
            "tokenizer_path": str(tokenizer_path),
            "batch_size": batch_size,
            "max_prompt_tokens": max_prompt_tokens,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_num_seqs": max_num_seqs,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "enable_thinking": args.enable_thinking,
            "quant_mode": args.quant_mode,
            "vllm_quantization": quantization,
            "vllm_flash_attn_version": args.vllm_flash_attn_version,
            "moe_backend": moe_backend,
            "dtype": "bfloat16",
            "enable_prefix_caching": args.enable_prefix_caching,
            "stop_token_ids": stop_token_ids,
            "elapsed_seconds": elapsed,
            "results": results,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as writer:
                json.dump(payload, writer, ensure_ascii=False, indent=2)
                writer.write("\n")
        return 0
    finally:
        _shutdown_llm(llm)


if __name__ == "__main__":
    raise SystemExit(main())
