#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-vocab next-token logprob/KL probes for YOCO/Qwen alignment.

Typical YOCO acceptance flow:

1. Generate native reference logits from a merged llm-train checkpoint.
2. Generate vLLM logits from a converted HF/vLLM checkpoint.
3. Compare full-vocab logprob distributions.

The script is intentionally self-contained so the prompt suite and comparison
logic stay identical across native, HF Transformers, and vLLM runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class PromptSpec:
    name: str
    kind: str
    text: str | None = None
    messages: list[dict[str, str]] | None = None


DEFAULT_PROMPTS = [
    PromptSpec(name="hello_name", kind="completion", text="Hello, my name is"),
    PromptSpec(
        name="france_capital", kind="completion", text="The capital of France is"
    ),
    PromptSpec(
        name="harry_potter",
        kind="completion",
        text=(
            "Harry Potter and the Philosopher's Stone is a fantasy novel "
            "written by J.K. Rowling and the first book in the Harry Potter "
            "series. The story follows an"
        ),
    ),
    PromptSpec(
        name="zh_intro",
        kind="chat",
        messages=[{"role": "user", "content": "请用三句话介绍一下你自己。"}],
    ),
    PromptSpec(
        name="zh_reasoning",
        kind="chat",
        messages=[
            {"role": "user", "content": "如果一个数的两倍加三等于十一，这个数是多少？"}
        ],
    ),
    PromptSpec(
        name="en_recipe",
        kind="chat",
        messages=[
            {"role": "user", "content": "Give me a concise recipe for tomato soup."}
        ],
    ),
]


MIXED5_PROMPTS = [
    ("short_hello", "Hello,"),
    ("short_fact", "The capital of France is"),
    (
        "medium_english",
        "Harry Potter and the Philosopher's Stone is a fantasy novel written "
        "by J.K. Rowling and the first book in the Harry Potter series. The "
        "story follows an orphaned boy who learns on his eleventh birthday "
        "that he is a wizard, then leaves his ordinary life behind to attend "
        "Hogwarts School of Witchcraft and Wizardry.",
    ),
    ("short_zh", "请用三句话介绍一下你自己。"),
    (
        "long_zh",
        "在一个多语言模型的评测任务中，我们希望同时观察短问题、事实补全、长段落续写和中文对话对模型输出分布的影响。"
        "请注意，这段输入故意包含较长的上下文、多个并列要求以及一些容易让模型在推理时改变语气的提示。"
        "评测时不要只看生成文本是否通顺，还要比较下一 token 的完整概率分布，"
        "因为很小的数值差异可能会改变 top-k 排序，"
        "尤其是在多个候选 token 概率接近的时候。现在，请继续这段说明：",
    ),
]

MIXED16_PROMPTS = [
    (f"{name}_{repeat}", text) for repeat in range(4) for name, text in MIXED5_PROMPTS
][:16]

CHUNK4K_PROMPTS = [
    (
        "long_english_4k",
        (
            "Harry Potter and the Philosopher's Stone is a fantasy novel "
            "written by J.K. Rowling and the first book in the Harry Potter "
            "series. The story follows an orphaned boy who learns on his "
            "eleventh birthday that he is a wizard, then leaves his ordinary "
            "life behind to attend Hogwarts School of Witchcraft and Wizardry. "
        )
        * 72,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-vocab next-token KL alignment probe"
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    for name in ("native", "hf", "vllm"):
        sub = subparsers.add_parser(name)
        sub.add_argument(
            "--model", required=True, help="HF/vLLM model path or Hugging Face repo id"
        )
        sub.add_argument("--out", required=True, help="Output .pt path")
        sub.add_argument(
            "--prompt-suite",
            choices=("default", "mixed5", "mixed16", "chunk4k"),
            default="mixed5",
        )
        sub.add_argument("--prompt-index", type=int, default=0)
        sub.add_argument("--prompt-start", type=int, default=0)
        sub.add_argument("--prompt-limit", type=int)
        sub.add_argument(
            "--batch-size",
            type=int,
            default=1,
            help="Number of prompts to execute together in each model forward.",
        )
        sub.add_argument(
            "--first-batch-size",
            type=int,
            help=(
                "Optional size of the first forward before using --batch-size "
                "for the remaining prompts."
            ),
        )
        sub.add_argument("--max-model-len", type=int, default=8192)
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument(
            "--trace-out",
            help="Optional output path for layer-by-layer hidden-state traces.",
        )
        if name != "hf":
            sub.add_argument(
                "--force-fa-num-splits-one",
                action="store_true",
                help=(
                    "Force one FlashAttention split. Native FA2 varlen prefill "
                    "has no split API and already uses its non-split path."
                ),
            )
            sub.add_argument(
                "--keep-attention-qkv-bf16",
                action="store_true",
                help=(
                    "Keep every projection feeding attention in BF16 while "
                    "leaving output, FFN, and MoE projections quantized."
                ),
            )
            sub.add_argument("--fa4-source-root", help=argparse.SUPPRESS)
            sub.add_argument(
                "--fa4-profile",
                choices=("default", "no-ex2", "q-stage1", "q-stage2"),
                default="default",
                help=argparse.SUPPRESS,
            )
            sub.add_argument(
                "--fa4-pack-gqa",
                choices=("auto", "on", "off"),
                default="auto",
                help=argparse.SUPPRESS,
            )
            sub.add_argument(
                "--fa4-tile-mn",
                choices=("default", "128x64", "128x128"),
                default="default",
                help=argparse.SUPPRESS,
            )

        if name == "native":
            sub.add_argument("--native-checkpoint", required=True)
            sub.add_argument(
                "--llm-train-dir", default="/data/users/shaohanh/llm-train"
            )
            sub.add_argument(
                "--native-dtype", choices=("bfloat16",), default="bfloat16"
            )
            sub.add_argument("--native-quant-mode", choices=("bfloat16", "mxfp8"))
            sub.add_argument("--native-quant-block-size", type=int)
            sub.add_argument("--native-use-cute", action="store_true")
            sub.add_argument(
                "--native-fa4-source",
                choices=("installed", "vllm-vendored"),
                default="installed",
                help=argparse.SUPPRESS,
            )
            sub.add_argument(
                "--native-local-attention",
                action="store_true",
                help=(
                    "Bypass NNScaler ring wrappers and execute FA2/FA4 "
                    "directly. This requires WORLD_SIZE=1."
                ),
            )
            sub.add_argument(
                "--native-require-transformer-engine",
                action="store_true",
                help=(
                    "Require llm-train to use TransformerEngine's padded MoE "
                    "permutation instead of the compatibility implementation."
                ),
            )
            sub.add_argument("--native-no-kv-cache", action="store_true")
            sub.add_argument("--native-prefill-chunk-size", type=int)
            sub.add_argument(
                "--native-use-torch-fp8-quant",
                action="store_true",
                help=(
                    "Use llm-train's torch activation quantization reference "
                    "instead of its Triton kernel."
                ),
            )
            sub.add_argument(
                "--native-forward-repeats",
                type=int,
                default=1,
                help=(
                    "Repeat each Native forward in the same process and report "
                    "the maximum logprob drift."
                ),
            )
        elif name == "hf":
            sub.add_argument("--device", default="cuda:0")
            sub.add_argument(
                "--dtype",
                choices=("bfloat16", "float16", "float32"),
                default="bfloat16",
            )
            sub.add_argument("--attn-implementation")
        else:
            sub.add_argument(
                "--dtype",
                choices=("auto", "bfloat16", "float16", "float32"),
                default="auto",
            )
            sub.add_argument("--tensor-parallel-size", type=int, default=1)
            sub.add_argument("--data-parallel-size", type=int, default=1)
            sub.add_argument("--enable-expert-parallel", action="store_true")
            sub.add_argument("--distributed-executor-backend")
            sub.add_argument("--gpu-memory-utilization", type=float, default=0.9)
            sub.add_argument("--max-num-seqs", type=int)
            sub.add_argument("--max-num-batched-tokens", type=int)
            sub.add_argument(
                "--enable-chunked-prefill",
                action=argparse.BooleanOptionalAction,
                default=None,
            )
            sub.add_argument("--kv-sharing-fast-prefill", action="store_true")
            sub.add_argument("--enforce-eager", action="store_true")
            sub.add_argument(
                "--v1-multiprocessing",
                action=argparse.BooleanOptionalAction,
                default=False,
                help=(
                    "Run the EngineCore in a child process. Disabled by default "
                    "so all requests are queued before scheduling."
                ),
            )
            sub.add_argument(
                "--log-iteration-details",
                action="store_true",
                help="Log actual context/generation request counts per forward.",
            )
            sub.add_argument(
                "--enable-prefix-caching",
                action=argparse.BooleanOptionalAction,
                default=False,
                help=(
                    "Enable prefix caching. Disabled by default so vLLM and "
                    "Native execute the same prompt-token rows."
                ),
            )
            sub.add_argument("--compilation-config-json")
            sub.add_argument("--quantization", default=None)
            sub.add_argument("--quantization-config-json")
            sub.add_argument("--quantization-ignore", action="append", default=[])
            sub.add_argument("--attention-backend")
            sub.add_argument("--flash-attn-version", type=int)
            sub.add_argument("--moe-backend", default=None)
            sub.add_argument("--max-logprobs", type=int, default=-1)

    cmp_parser = subparsers.add_parser("compare")
    cmp_parser.add_argument("--reference", "--native", dest="reference", required=True)
    cmp_parser.add_argument("--candidate", "--vllm", dest="candidate", required=True)
    cmp_parser.add_argument("--out-json", required=True)
    cmp_parser.add_argument("--top-k", type=int, default=20)
    cmp_parser.add_argument(
        "--model",
        help=(
            "Tokenizer path for comparison. Use this when artifacts were "
            "created with container-local model paths."
        ),
    )
    return parser.parse_args()


def _save(path: str, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _device_mapping(cuda_device: int):
    def inner_device_mapping(storage: torch.Storage, location) -> torch.Storage:
        if cuda_device >= 0:
            return storage.cuda(cuda_device)
        return storage

    return inner_device_mapping


def _tokenizer_encode(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return tokenizer.encode(text, add_special_tokens=False)
    raise TypeError(f"Unsupported tokenizer: {type(tokenizer)!r}")


def _bos_id(tokenizer: Any) -> int | None:
    for attr in ("bos_token_id", "bos_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        for token in ("<sop>", "<|startoftext|>"):
            value = tokenizer.convert_tokens_to_ids(token)
            unk = getattr(tokenizer, "unk_token_id", None)
            if value is not None and value != unk:
                return int(value)
    return None


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _default_prompt_records(
    tokenizer: Any,
    prompt_limit: int | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    specs = (
        DEFAULT_PROMPTS[:prompt_limit] if prompt_limit is not None else DEFAULT_PROMPTS
    )
    bos_id = _bos_id(tokenizer)
    for spec in specs:
        if spec.kind == "completion":
            assert spec.text is not None
            prompt_text = spec.text
            token_ids = _tokenizer_encode(tokenizer, prompt_text)
            if bos_id is not None:
                token_ids = [bos_id] + token_ids
        elif spec.kind == "chat":
            assert spec.messages is not None
            prompt_text = _apply_chat_template(tokenizer, spec.messages)
            token_ids = _tokenizer_encode(tokenizer, prompt_text)
        else:
            raise ValueError(f"Unknown prompt kind: {spec.kind}")
        records.append(
            {
                "name": spec.name,
                "kind": spec.kind,
                "prompt_text": prompt_text,
                "prompt_token_ids": token_ids,
                "prompt_len": len(token_ids),
            }
        )
    return records


def _mixed_prompt_records(
    tokenizer: Any,
    prompt_specs: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    bos_id = _bos_id(tokenizer)
    records = []
    for name, text in prompt_specs:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if bos_id is not None:
            token_ids = [bos_id] + token_ids
        records.append(
            {
                "name": name,
                "kind": "completion",
                "prompt_text": text,
                "prompt_token_ids": token_ids,
                "prompt_len": len(token_ids),
            }
        )
    return records


def _prompt_records(
    model_dir: str,
    prompt_suite: str,
    prompt_index: int,
    prompt_start: int,
    prompt_limit: int | None,
) -> tuple[Any, list[dict[str, Any]]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if prompt_suite in ("mixed5", "mixed16", "chunk4k"):
        prompt_specs = {
            "mixed5": MIXED5_PROMPTS,
            "mixed16": MIXED16_PROMPTS,
            "chunk4k": CHUNK4K_PROMPTS,
        }[prompt_suite]
        records = _mixed_prompt_records(tokenizer, prompt_specs)
        if prompt_limit is not None:
            records = records[prompt_start : prompt_start + prompt_limit]
        elif prompt_start:
            records = records[prompt_start:]
        elif prompt_index:
            records = [records[prompt_index]]
        return tokenizer, records
    if prompt_start:
        raise ValueError("--prompt-start is only supported for mixed prompt suites")

    if prompt_limit is not None:
        records = _default_prompt_records(tokenizer, prompt_limit)
    else:
        records = _default_prompt_records(tokenizer, prompt_index + 1)
        if prompt_index >= len(records):
            raise IndexError(
                f"prompt_index={prompt_index} out of range ({len(records)})"
            )
        records = [records[prompt_index]]
    return tokenizer, records


def _top_payload(logprobs: torch.Tensor, k: int = 20) -> dict[str, torch.Tensor]:
    top = torch.topk(logprobs, k=min(k, logprobs.numel()))
    return {"top_ids": top.indices.cpu(), "top_logprobs": top.values.cpu()}


def _result_payload(
    *,
    backend: str,
    model_dir: str,
    record: dict[str, Any],
    logprobs: torch.Tensor,
    chosen_token_id: int | None = None,
) -> dict[str, Any]:
    if not torch.isfinite(logprobs).all():
        finite = int(torch.isfinite(logprobs).sum())
        raise RuntimeError(
            f"Non-finite logprobs for {record['name']}: "
            f"{finite}/{logprobs.numel()} finite"
        )
    if logprobs.numel() > 1 and torch.all(logprobs == logprobs[0]):
        raise RuntimeError(
            f"Collapsed uniform logprobs for {record['name']}: "
            f"all {logprobs.numel()} values are {float(logprobs[0])}"
        )
    payload: dict[str, Any] = {
        "backend": backend,
        "model": model_dir,
        "prompt": record,
        "vocab_size": int(logprobs.numel()),
        "logprobs": logprobs.cpu(),
    }
    if chosen_token_id is not None:
        payload["chosen_token_id"] = int(chosen_token_id)
    payload.update(_top_payload(logprobs))
    return payload


def _payload_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return payload.get("results", [payload])


def _comparison_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "backend",
        "model",
        "native_checkpoint",
        "attention_backend",
        "attention_version",
        "native_fa4_source",
        "native_fa4_runtime",
        "vllm_fa4_runtime",
        "local_attention",
        "kv_cache_enabled",
        "quant_mode",
        "quant_block_size",
        "quantization",
        "moe_backend",
        "torch_fp8_quant_fallback",
        "transformer_engine_enabled",
        "force_fa_num_splits_one",
        "keep_attention_qkv_bf16",
        "attention_qkv_bf16_modules",
        "quantization_ignore",
        "batch_size",
        "first_batch_size",
        "data_parallel_size",
        "expert_parallel_enabled",
        "v1_multiprocessing_enabled",
        "prefix_caching_enabled",
        "compilation_config",
    )
    return {key: payload[key] for key in keys if key in payload}


def _record_batches(
    records: list[dict[str, Any]],
    batch_size: int,
    first_batch_size: int | None = None,
) -> list[list[dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if first_batch_size is not None and first_batch_size < 1:
        raise ValueError(f"first_batch_size must be positive, got {first_batch_size}")
    batches = []
    start = 0
    if first_batch_size is not None:
        batches.append(records[:first_batch_size])
        start = first_batch_size
    batches.extend(
        records[offset : offset + batch_size]
        for offset in range(start, len(records), batch_size)
    )
    return [batch for batch in batches if batch]


_NATIVE_FA4_FUNCTIONS: dict[str, Any] = {}
_NATIVE_FA4_RUNTIME: dict[str, Any] = {}

ATTENTION_QKV_BF16_IGNORE = (
    r"re:^model\.layers\.[0-9]+\.self_attn\.(q|k|v)_proj$",
    r"re:^model\.yoco_(k|v)_proj$",
)


def _keep_native_attention_qkv_bf16(model: torch.nn.Module) -> list[str]:
    modules: list[str] = []
    for layer_index, layer in enumerate(model.layers):
        attention = layer.self_attn
        for projection_name in ("q_proj", "k_proj", "v_proj"):
            projection = getattr(attention, projection_name, None)
            if projection is None:
                continue
            projection.quant_mode = "bfloat16"
            modules.append(f"layers.{layer_index}.self_attn.{projection_name}")
    for projection_name in ("k_proj", "v_proj"):
        projection = getattr(model, projection_name, None)
        if projection is None:
            continue
        projection.quant_mode = "bfloat16"
        modules.append(projection_name)
    if not modules:
        raise RuntimeError("No Native attention QKV projections were found")
    return modules


def _fa4_source_runtime(source_root: str, profile: str) -> dict[str, Any]:
    import hashlib
    import importlib.metadata as metadata

    import cutlass
    import tvm_ffi

    root = Path(source_root).resolve()
    cute_root = root / "flash_attn" / "cute"
    interface_path = cute_root / "interface.py"
    forward_path = cute_root / "flash_fwd_sm100.py"
    if not interface_path.is_file() or not forward_path.is_file():
        raise RuntimeError(f"Invalid FA4 source profile: {cute_root}")

    expected_versions = {
        "flash-attn-4": "4.0.0b13",
        "nvidia-cutlass-dsl": "4.5.1",
        "apache-tvm-ffi": "0.1.11",
        "quack-kernels": "0.4.1",
    }
    runtime_versions = {
        "flash-attn-4": metadata.version("flash-attn-4"),
        "nvidia-cutlass-dsl": cutlass.__version__,
        "apache-tvm-ffi": tvm_ffi.__version__,
        "quack-kernels": metadata.version("quack-kernels"),
    }
    if runtime_versions != expected_versions:
        raise RuntimeError(
            "Image-derived FA4 runtime mismatch: "
            f"expected {expected_versions}, found {runtime_versions}"
        )

    source_hasher = hashlib.sha256()
    for source_path in sorted(cute_root.rglob("*.py")):
        relative_path = source_path.relative_to(cute_root).as_posix()
        source_hasher.update(relative_path.encode("utf-8"))
        source_hasher.update(b"\0")
        source_hasher.update(source_path.read_bytes())

    forward_text = forward_path.read_text(encoding="utf-8")
    interface_text = interface_path.read_text(encoding="utf-8")
    override = "self.enable_ex2_emu = False\n        self.ex2_emu_freq = 0"
    exp2_emu_disabled = override in forward_text
    if exp2_emu_disabled != (profile == "no-ex2"):
        raise RuntimeError(
            f"FA4 profile {profile!r} does not match source override state "
            f"exp2_emu_disabled={exp2_emu_disabled}"
        )

    q_stage_override = None
    for stage in (1, 2):
        snippet = (
            "    if arch // 10 == 10:\n"
            f"        q_stage = {stage}\n"
            "    else:\n"
            "        q_stage = 1\n"
        )
        if snippet in interface_text:
            q_stage_override = stage
    expected_q_stage = {
        "default": None,
        "no-ex2": None,
        "q-stage1": 1,
        "q-stage2": 2,
    }[profile]
    if q_stage_override != expected_q_stage:
        raise RuntimeError(
            f"FA4 profile {profile!r} does not match q_stage override "
            f"state {q_stage_override!r}"
        )

    return {
        "source": "donglixp/pytorch:26.02-b200",
        "profile": profile,
        "source_root": str(cute_root),
        "source_sha256": source_hasher.hexdigest(),
        "runtime_versions": runtime_versions,
        "cutlass_module": cutlass.__file__,
        "tvm_ffi_module": tvm_ffi.__file__,
        "exp2_emu_disabled": exp2_emu_disabled,
        "q_stage_override": q_stage_override,
    }


def _import_external_fa4_interface(source_root: str, profile: str):
    import importlib

    root = Path(source_root).resolve()
    package_path = root / "flash_attn"
    if not (package_path / "cute" / "interface.py").is_file():
        raise RuntimeError(f"FA4 source package not found: {package_path}")

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import flash_attn
    except ImportError:
        flash_attn = None
    if flash_attn is not None and hasattr(flash_attn, "__path__"):
        package_path_text = str(package_path)
        if package_path_text not in flash_attn.__path__:
            flash_attn.__path__.insert(0, package_path_text)

    existing = sys.modules.get("flash_attn.cute.interface")
    if existing is not None:
        existing_path = Path(existing.__file__).resolve()
        if not existing_path.is_relative_to(package_path):
            raise RuntimeError(
                "flash_attn.cute.interface was imported before the requested "
                f"FA4 source profile: {existing_path}"
            )
        interface = existing
    else:
        interface = importlib.import_module("flash_attn.cute.interface")

    runtime = _fa4_source_runtime(str(root), profile)
    return interface, runtime


def _install_external_fa4_tuning(
    interface: Any,
    *,
    pack_gqa: str = "auto",
    tile_mn: str = "default",
) -> dict[str, Any]:
    pack_gqa_override = {"auto": None, "on": True, "off": False}[pack_gqa]
    tile_mn_override = {
        "default": None,
        "128x64": (128, 64),
        "128x128": (128, 128),
    }[tile_mn]
    overrides = {
        key: value
        for key, value in (
            ("pack_gqa", pack_gqa_override),
            ("tile_mn", tile_mn_override),
        )
        if value is not None
    }
    if not overrides:
        return {}

    current = interface._flash_attn_fwd
    if hasattr(current, "__yoco_fa4_tuning_overrides__"):
        if current.__yoco_fa4_tuning_overrides__ != overrides:
            raise RuntimeError(
                "External FA4 tuning was already installed with different "
                f"overrides: {current.__yoco_fa4_tuning_overrides__}"
            )
        return overrides

    def tuned_flash_attn_fwd(*args, **kwargs):
        kwargs.update(overrides)
        return current(*args, **kwargs)

    tuned_flash_attn_fwd.compile_cache = current.compile_cache
    tuned_flash_attn_fwd.__yoco_fa4_tuning_overrides__ = overrides
    interface._flash_attn_fwd = tuned_flash_attn_fwd
    return overrides


def _import_fa4_varlen_func(
    source: str = "installed",
    *,
    source_root: str | None = None,
    profile: str = "default",
    pack_gqa: str = "auto",
    tile_mn: str = "default",
):
    """Load either installed FA4 or this checkout's exact vendored FA4 source."""
    cache_key = f"{source}:{source_root or ''}:{profile}:{pack_gqa}:{tile_mn}"
    cached = _NATIVE_FA4_FUNCTIONS.get(cache_key)
    if cached is not None:
        return cached

    if source_root is not None:
        interface, runtime = _import_external_fa4_interface(source_root, profile)
        tuning_overrides = _install_external_fa4_tuning(
            interface,
            pack_gqa=pack_gqa,
            tile_mn=tile_mn,
        )
        flash_attn_varlen_func = interface.flash_attn_varlen_func
        _NATIVE_FA4_FUNCTIONS[cache_key] = flash_attn_varlen_func
        _NATIVE_FA4_RUNTIME.update(
            {**runtime, "tuning_overrides": tuning_overrides}
        )
        print(
            "[native-kl] loaded image-derived FA4 "
            f"profile={profile} sha256={runtime['source_sha256']}",
            flush=True,
        )
        return flash_attn_varlen_func

    if source == "vllm-vendored":
        import hashlib
        import importlib
        import types

        import cutlass
        import tvm_ffi

        expected_versions = {
            "nvidia-cutlass-dsl": "4.4.2",
            "apache-tvm-ffi": "0.1.9",
            "quack-kernels": "0.4.1",
        }
        runtime_versions = {
            "nvidia-cutlass-dsl": cutlass.__version__,
            "apache-tvm-ffi": tvm_ffi.__version__,
        }

        import importlib.metadata as metadata

        runtime_versions["quack-kernels"] = metadata.version("quack-kernels")
        mismatches = {
            name: (expected_versions[name], runtime_versions[name])
            for name in expected_versions
            if runtime_versions[name] != expected_versions[name]
        }
        if mismatches:
            details = ", ".join(
                f"{name}: expected {expected}, found {actual}"
                for name, (expected, actual) in mismatches.items()
            )
            raise RuntimeError(
                "vLLM vendored FA4 runtime mismatch; activate the Native "
                f"FA4 overlay ({details})"
            )

        repo_root = Path(__file__).resolve().parents[2]
        vllm_package = repo_root / "vllm"
        vendored_package = vllm_package / "vllm_flash_attn"
        interface_path = vendored_package / "cute" / "interface.py"
        if not interface_path.is_file():
            raise RuntimeError(f"vLLM vendored FA4 source not found: {interface_path}")

        for module_name, package_path in (
            ("vllm", vllm_package),
            ("vllm.vllm_flash_attn", vendored_package),
        ):
            module = sys.modules.get(module_name)
            if module is None:
                module = types.ModuleType(module_name)
                module.__package__ = module_name
                module.__path__ = [str(package_path)]
                sys.modules[module_name] = module
            elif str(package_path) not in getattr(module, "__path__", ()):
                raise RuntimeError(
                    f"Cannot load vendored FA4 because {module_name} is already "
                    "imported from another location"
                )

        interface = importlib.import_module(
            "vllm.vllm_flash_attn.cute.interface"
        )
        source_root = vendored_package / "cute"
        source_hasher = hashlib.sha256()
        for source_path in sorted(source_root.rglob("*.py")):
            relative_path = source_path.relative_to(source_root).as_posix()
            source_hasher.update(relative_path.encode("utf-8"))
            source_hasher.update(b"\0")
            source_hasher.update(source_path.read_bytes())

        flash_attn_varlen_func = interface.flash_attn_varlen_func
        _NATIVE_FA4_FUNCTIONS[cache_key] = flash_attn_varlen_func
        _NATIVE_FA4_RUNTIME.update(
            {
                "source": source,
                "source_root": str(source_root),
                "source_sha256": source_hasher.hexdigest(),
                "runtime_versions": runtime_versions,
            }
        )
        print(
            "[native-kl] loaded vLLM vendored FA4 "
            f"sha256={source_hasher.hexdigest()}",
            flush=True,
        )
        return flash_attn_varlen_func

    if source != "installed":
        raise ValueError(f"Unsupported Native FA4 source: {source}")

    import importlib.metadata as metadata

    import flash_attn

    try:
        distribution = metadata.distribution("flash-attn-4")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("Native FA4 requires the flash-attn-4 package") from exc

    for relative_path in distribution.files or ():
        if str(relative_path).endswith("flash_attn/cute/__init__.py"):
            package_path = str(distribution.locate_file(relative_path).parents[1])
            if package_path not in flash_attn.__path__:
                flash_attn.__path__.append(package_path)
            break

    try:
        from flash_attn.cute import flash_attn_varlen_func
    except ImportError as exc:
        raise RuntimeError(
            "flash-attn-4 is installed but flash_attn.cute is unavailable"
        ) from exc
    _NATIVE_FA4_FUNCTIONS[cache_key] = flash_attn_varlen_func
    _NATIVE_FA4_RUNTIME.update(
        {
            "source": source,
            "distribution": distribution.metadata["Name"],
            "distribution_version": distribution.version,
            "source_file": sys.modules[flash_attn_varlen_func.__module__].__file__,
        }
    )
    return flash_attn_varlen_func


def _install_native_inference_compat(
    *,
    local_attention: bool = False,
    force_fa_num_splits_one: bool = False,
    native_fa4_source: str = "installed",
    fa4_source_root: str | None = None,
    fa4_profile: str = "default",
    fa4_pack_gqa: str = "auto",
    fa4_tile_mn: str = "default",
) -> None:
    """Provide inference-only fallbacks for optional llm-train dependencies.

    The native alignment probe is intentionally single-GPU and prefill-only.
    NNScaler is only needed by this path for import-time operator registration
    and distributed ring/EP helpers. On one GPU, execute the selected attention
    kernel directly instead of entering ring collectives. If upstream
    ``flash_attn`` is absent, adapt vLLM's bundled varlen prefill kernel to its
    argument order.
    """
    import importlib.util
    import types

    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if local_attention and world_size != 1:
        raise RuntimeError("--native-local-attention requires WORLD_SIZE=1")

    def identity_single_gpu(value, **kwargs):
        return value

    def unsupported_distributed_op(*args, **kwargs):
        raise RuntimeError(
            "The nnscaler-free native probe reached a distributed "
            "NNScaler operation; use --native-local-attention with "
            "WORLD_SIZE=1 or install NNScaler"
        )

    def flash_attn_cute_varlen_func(*args, **kwargs):
        flash_attn_varlen_func = _import_fa4_varlen_func(
            native_fa4_source,
            source_root=fa4_source_root,
            profile=fa4_profile,
            pack_gqa=fa4_pack_gqa,
            tile_mn=fa4_tile_mn,
        )
        return flash_attn_varlen_func(*args, **kwargs)

    def make_single_gpu_attention_wrapper(cute_module):
        def wrapper(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            process_group=None,
            **kwargs,
        ):
            keyword_process_group = kwargs.pop("process_group", None)
            if process_group is not None or keyword_process_group is not None:
                unsupported_distributed_op()

            kwargs.pop("enable_ring", None)
            use_cute = kwargs.pop("use_cute", False)
            return_lse = kwargs.pop("return_lse", False)
            max_seqlen_q = kwargs.pop("max_seqlen_q", None)
            max_seqlen_k = kwargs.pop("max_seqlen_k", None)
            if max_seqlen_q is None:
                max_seqlen_q = int(
                    (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
                )
            if max_seqlen_k is None:
                max_seqlen_k = int(
                    (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
                )

            if use_cute:
                if force_fa_num_splits_one:
                    kwargs["num_splits"] = 1
                return cute_module.flash_attn_cute_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    return_lse=return_lse,
                    **kwargs,
                )

            from flash_attn import flash_attn_varlen_func

            return flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                return_attn_probs=return_lse,
                **kwargs,
            )

        return wrapper

    if importlib.util.find_spec("nnscaler") is None:
        if world_size != 1:
            raise RuntimeError(
                "Native inference without nnscaler only supports WORLD_SIZE=1"
            )

        def register_op(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

        class DimopSplit:
            D = staticmethod(lambda dim: ("D", dim))
            R = staticmethod(lambda: ("R",))
            V = staticmethod(lambda: ("V",))

        class TransformRule:
            def __init__(self, *args, **kwargs):
                pass

        class IRDimops:
            pass

        class DeviceGroup:
            def get_group(self, process_group):
                unsupported_distributed_op()

        module_names = (
            "nnscaler",
            "nnscaler.customized_ops",
            "nnscaler.customized_ops.ring_attention",
            "nnscaler.customized_ops.ring_attention.sliding_window_attn",
            "nnscaler.customized_ops.ring_attention.ring_attn_varlen",
            "nnscaler.customized_ops.ring_attention.zigzag_allgather_attn_varlen",
            "nnscaler.runtime",
            "nnscaler.runtime.device",
            "nnscaler.graph",
            "nnscaler.graph.function",
            "nnscaler.graph.function.dimops",
        )
        modules = {name: types.ModuleType(name) for name in module_names}
        modules["nnscaler"].register_op = register_op

        ring_attention = modules["nnscaler.customized_ops.ring_attention"]
        sliding_window_attention = modules[
            "nnscaler.customized_ops.ring_attention.sliding_window_attn"
        ]
        ring_varlen_attention = modules[
            "nnscaler.customized_ops.ring_attention.ring_attn_varlen"
        ]
        zigzag_attention = modules[
            "nnscaler.customized_ops.ring_attention.zigzag_allgather_attn_varlen"
        ]
        for attention_module in (
            sliding_window_attention,
            ring_varlen_attention,
            zigzag_attention,
        ):
            attention_module.flash_attn_cute_varlen_func = (
                flash_attn_cute_varlen_func
            )

        ring_attention.wrap_maybe_shuffle = identity_single_gpu
        ring_attention.wrap_maybe_unshuffle = identity_single_gpu
        if local_attention:
            ring_attention.wrap_ring_attn_varlen_func = (
                make_single_gpu_attention_wrapper(ring_varlen_attention)
            )
            ring_attention.wrap_sliding_window_attn_func = (
                make_single_gpu_attention_wrapper(sliding_window_attention)
            )
            ring_attention.wrap_zigzag_allgather_attn_varlen_func = (
                make_single_gpu_attention_wrapper(zigzag_attention)
            )
        else:
            ring_attention.wrap_ring_attn_varlen_func = unsupported_distributed_op
            ring_attention.wrap_sliding_window_attn_func = unsupported_distributed_op
            ring_attention.wrap_zigzag_allgather_attn_varlen_func = (
                unsupported_distributed_op
            )

        modules["nnscaler.runtime.device"].DeviceGroup = DeviceGroup
        dimops = modules["nnscaler.graph.function.dimops"]
        dimops.DimopSplit = DimopSplit
        dimops.TransformRule = TransformRule
        dimops.IRDimops = IRDimops
        sys.modules.update(modules)
        print(
            "[native-kl] nnscaler not installed; using single-GPU "
            f"{'direct-attention' if local_attention else 'inference'} "
            "compatibility shim",
            flush=True,
        )
    elif local_attention:
        import importlib

        ring_attention = importlib.import_module(
            "nnscaler.customized_ops.ring_attention"
        )
        sliding_window_attention = importlib.import_module(
            "nnscaler.customized_ops.ring_attention.sliding_window_attn"
        )
        ring_varlen_attention = importlib.import_module(
            "nnscaler.customized_ops.ring_attention.ring_attn_varlen"
        )
        zigzag_attention = importlib.import_module(
            "nnscaler.customized_ops.ring_attention.zigzag_allgather_attn_varlen"
        )
        ring_attention.wrap_maybe_shuffle = identity_single_gpu
        ring_attention.wrap_maybe_unshuffle = identity_single_gpu
        ring_attention.wrap_ring_attn_varlen_func = (
            make_single_gpu_attention_wrapper(ring_varlen_attention)
        )
        ring_attention.wrap_sliding_window_attn_func = (
            make_single_gpu_attention_wrapper(sliding_window_attention)
        )
        ring_attention.wrap_zigzag_allgather_attn_varlen_func = (
            make_single_gpu_attention_wrapper(zigzag_attention)
        )
        print(
            "[native-kl] bypassing NNScaler ring attention with local kernels",
            flush=True,
        )

    if importlib.util.find_spec("flash_attn") is None:
        from vllm.vllm_flash_attn import (
            flash_attn_varlen_func as vllm_flash_attn_varlen_func,
        )

        def flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            *args,
            **kwargs,
        ):
            return vllm_flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q,
                cu_seqlens_q,
                max_seqlen_k,
                cu_seqlens_k,
                *args,
                **kwargs,
            )

        def flash_attn_with_kvcache(*args, **kwargs):
            raise RuntimeError(
                "The native compatibility path supports prefill only; "
                "install flash-attn for incremental decoding"
            )

        flash_attn = types.ModuleType("flash_attn")
        flash_attn.flash_attn_varlen_func = flash_attn_varlen_func
        flash_attn.flash_attn_with_kvcache = flash_attn_with_kvcache
        sys.modules["flash_attn"] = flash_attn
        print(
            "[native-kl] flash-attn not installed; using vLLM's bundled "
            "varlen prefill kernel",
            flush=True,
        )


def _install_native_moe_padding_compat() -> bool:
    """Pad eager single-rank MoE rows when TransformerEngine is unavailable."""
    import arch.all2all_moe as native_all2all_moe
    import arch.moe_utils_v2 as native_moe_utils

    if native_moe_utils.HAVE_TE:
        print(
            "[native-kl] using TransformerEngine padded MoE permutation",
            flush=True,
        )
        return True

    alignment = 128
    original_permute = native_moe_utils.permute

    def padded_permute(
        tokens,
        routing_map,
        probs,
        tokens_per_expert=None,
    ):
        if tokens_per_expert is None:
            return original_permute(
                tokens,
                routing_map,
                probs,
                tokens_per_expert=tokens_per_expert,
            )

        tokens_per_expert = tokens_per_expert.to(
            device=tokens.device,
            dtype=torch.long,
        )
        target_tokens_per_expert = (
            (tokens_per_expert + alignment - 1) // alignment * alignment
        )
        total_padded_tokens = int(target_tokens_per_expert.sum().item())
        padded_tokens = tokens.new_zeros((total_padded_tokens, tokens.shape[-1]))
        padded_probs = probs.new_zeros((total_padded_tokens,))
        top_k = int(routing_map.sum(dim=1).max().item())
        row_map = torch.full(
            (tokens.shape[0], top_k),
            -1,
            device=tokens.device,
            dtype=torch.long,
        )
        token_slots = torch.zeros(
            tokens.shape[0],
            device=tokens.device,
            dtype=torch.long,
        )

        target_offset = 0
        for expert_index, (token_count, target_count) in enumerate(
            zip(tokens_per_expert.tolist(), target_tokens_per_expert.tolist())
        ):
            if token_count:
                token_indices = torch.nonzero(
                    routing_map[:, expert_index],
                    as_tuple=False,
                ).flatten()
                if token_indices.numel() != token_count:
                    raise RuntimeError(
                        "MoE routing count mismatch for expert "
                        f"{expert_index}: expected {token_count}, got "
                        f"{token_indices.numel()}"
                    )
                end = target_offset + token_count
                padded_tokens[target_offset:end] = tokens.index_select(
                    0, token_indices
                )
                padded_probs[target_offset:end] = probs[
                    token_indices, expert_index
                ]
                slots = token_slots.index_select(0, token_indices)
                row_map[token_indices, slots] = torch.arange(
                    target_offset,
                    end,
                    device=tokens.device,
                )
                token_slots[token_indices] += 1
            target_offset += target_count

        pad_offsets = torch.cumsum(
            target_tokens_per_expert - tokens_per_expert,
            dim=0,
        )
        pad_offsets = torch.cat(
            [pad_offsets.new_zeros(1), pad_offsets[:-1]],
        )
        return (
            padded_tokens,
            padded_probs,
            row_map,
            pad_offsets,
            target_tokens_per_expert,
        )

    def padded_unpermute(
        permuted_tokens,
        row_map,
        restore_shape,
        probs=None,
        routing_map=None,
        pad_offsets=None,
    ):
        del routing_map, pad_offsets
        valid_rows = row_map >= 0
        safe_rows = row_map.clamp_min(0)
        token_rows = permuted_tokens.index_select(0, safe_rows.flatten()).view(
            row_map.shape[0],
            row_map.shape[1],
            permuted_tokens.shape[-1],
        )
        token_rows = token_rows.masked_fill(~valid_rows.unsqueeze(-1), 0)
        if probs is not None:
            token_rows = token_rows * probs.unsqueeze(-1)
        return token_rows.sum(dim=1).view(restore_shape)

    native_moe_utils.permute = padded_permute
    native_moe_utils.unpermute = padded_unpermute
    native_all2all_moe.permute = padded_permute
    native_all2all_moe.unpermute = padded_unpermute
    print(
        "[native-kl] TransformerEngine unavailable; using deterministic "
        "128-row MoE padding compatibility path",
        flush=True,
    )
    return False


@torch.no_grad()
def run_native(args: argparse.Namespace) -> None:
    import importlib

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    llm_dir = Path(args.llm_train_dir).resolve() / "llm"
    sys.path.insert(0, str(llm_dir))
    _install_native_inference_compat(
        local_attention=args.native_local_attention,
        force_fa_num_splits_one=args.force_fa_num_splits_one,
        native_fa4_source=args.native_fa4_source,
        fa4_source_root=args.fa4_source_root,
        fa4_profile=args.fa4_profile,
        fa4_pack_gqa=args.fa4_pack_gqa,
        fa4_tile_mn=args.fa4_tile_mn,
    )
    from arch.model import Model, ModelArgs, create_kv_cache

    using_transformer_engine = _install_native_moe_padding_compat()
    if args.native_require_transformer_engine and not using_transformer_engine:
        raise RuntimeError(
            "--native-require-transformer-engine was set, but llm-train "
            "could not import TransformerEngine's MoE permutation APIs"
        )

    checkpoint_dir = Path(args.native_checkpoint)
    metadata_path = checkpoint_dir / "metadata.json"
    state_path = checkpoint_dir / "model_state_rank_0.pth"
    for path, description in (
        (metadata_path, "metadata"),
        (state_path, "rank-0 model state"),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Native checkpoint {description} not found: {path}. "
                "Use --native-checkpoint with an accessible merged "
                "llm-train checkpoint."
            )

    if args.native_prefill_chunk_size is not None:
        import arch.attention as native_attention

        native_flash_attn_with_kvcache = native_attention.flash_attn_with_kvcache

        def flash_attn_with_chunked_kvcache(
            query: torch.Tensor,
            *flash_args,
            **flash_kwargs,
        ) -> torch.Tensor:
            if query.shape[1] == 1 and query.shape[0] > 1:
                output = native_flash_attn_with_kvcache(
                    query.transpose(0, 1),
                    *flash_args,
                    **flash_kwargs,
                )
                return output.transpose(0, 1)
            return native_flash_attn_with_kvcache(
                query,
                *flash_args,
                **flash_kwargs,
            )

        native_attention.flash_attn_with_kvcache = flash_attn_with_chunked_kvcache

    if args.native_use_torch_fp8_quant:
        import arch.linear as linear
        import kernel.moe_ffn as moe_ffn
        import kernel.quant as quant

        linear.per_token_cast_to_fp8 = quant._per_token_cast_to_fp8_torch
        linear.per_block_cast_to_fp8 = quant._per_block_cast_to_fp8_torch
        moe_ffn.per_token_cast_to_fp8 = quant._per_token_cast_to_fp8_torch
        print("[native-kl] using torch FP8 activation quantization", flush=True)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.accelerator.set_device_index(local_rank)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    with metadata_path.open(encoding="utf-8") as reader:
        metadata = json.load(reader)
    modelargs = ModelArgs()
    for key, value in metadata["modelargs"].items():
        setattr(modelargs, key, value)
    if args.native_quant_mode is not None:
        modelargs.quant_mode = args.native_quant_mode
    if args.native_quant_block_size is not None:
        modelargs.quant_block_size = args.native_quant_block_size
    modelargs.use_cute = args.native_use_cute
    if args.native_no_kv_cache:
        modelargs.moe_fwd_bwd_overlap = False

    init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["dp"])
    default_device = torch.get_default_device()
    default_dtype = torch.get_default_dtype()
    torch.set_default_device(device)
    torch.set_default_dtype(_dtype(args.native_dtype))
    model = Model(modelargs)
    torch.set_default_device(default_device)
    torch.set_default_dtype(default_dtype)
    model.eval()

    attention_qkv_bf16_modules: list[str] = []
    if args.keep_attention_qkv_bf16:
        if args.native_quant_mode != "mxfp8":
            raise ValueError(
                "--keep-attention-qkv-bf16 requires --native-quant-mode mxfp8"
            )
        attention_qkv_bf16_modules = _keep_native_attention_qkv_bf16(model)
        print(
            "[native-kl] keeping attention QKV projections BF16: "
            f"{len(attention_qkv_bf16_modules)} logical projections",
            flush=True,
        )

    state = torch.load(
        state_path,
        map_location=_device_mapping(-1),
        mmap=True,
    )
    state = {
        key: value for key, value in state.items() if not key.startswith("moe_loss.")
    }
    model.load_state_dict(state)
    print(
        f"[native-kl] model loaded use_cute={modelargs.use_cute} "
        f"kv_cache={not args.native_no_kv_cache}",
        flush=True,
    )

    trace_records: list[dict[str, Any]] = []
    trace_handles = []
    trace_counts: dict[str, int] = {}
    native_moe_patches = []
    if args.trace_out:
        trace_prompt_records = _prompt_records(
            args.model,
            args.prompt_suite,
            args.prompt_index,
            args.prompt_start,
            args.prompt_limit,
        )[1]
        if (
            len(trace_prompt_records) != 1
            or args.batch_size != 1
            or args.first_batch_size not in (None, 1)
        ):
            raise ValueError("--trace-out requires exactly one prompt and batch-size=1")
        if args.native_forward_repeats != 1:
            raise ValueError("--trace-out requires --native-forward-repeats=1")

        def record_trace(name: str, tensor: torch.Tensor) -> None:
            count = trace_counts.get(name, 0)
            trace_counts[name] = count + 1
            key = name if count == 0 else f"{name}#{count}"
            trace_records.append(
                {
                    "name": key,
                    "tensor": tensor.detach().float().cpu(),
                }
            )

        def trace_hook(name: str):
            def hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if not isinstance(tensor, torch.Tensor):
                    return
                record_trace(name, tensor)

            return hook

        def trace_rope_hook(name: str):
            def hook(_module, _inputs, output):
                if not isinstance(output, tuple) or len(output) != 2:
                    return
                record_trace(f"{name}.q_rotary", output[0])
                record_trace(f"{name}.k_rotary", output[1])

            return hook

        active_mlp = [""]
        active_attn = [""]
        trace_handles.append(
            model.tok_embeddings.register_forward_hook(trace_hook("embed"))
        )
        for layer_idx, layer in enumerate(model.layers):
            prefix = f"layer.{layer_idx}"
            mlp_prefix = f"{prefix}.mlp"
            attn_prefix = f"{prefix}.self_attn"

            def set_active_mlp(_module, _inputs, name=mlp_prefix):
                active_mlp[0] = name

            def clear_active_mlp(_module, _inputs, _output):
                active_mlp[0] = ""

            def set_active_attn(_module, _inputs, name=attn_prefix):
                active_attn[0] = name

            def clear_active_attn(_module, _inputs, _output):
                active_attn[0] = ""

            trace_handles.extend(
                [
                    layer.input_layernorm.register_forward_hook(
                        trace_hook(f"{prefix}.input_norm")
                    ),
                    layer.self_attn.register_forward_pre_hook(set_active_attn),
                    layer.self_attn.register_forward_hook(
                        trace_hook(f"{prefix}.attn_out")
                    ),
                    layer.self_attn.register_forward_hook(clear_active_attn),
                    layer.self_attn.o_proj.register_forward_hook(
                        trace_hook(f"{attn_prefix}.output")
                    ),
                    layer.post_attention_layernorm.register_forward_hook(
                        trace_hook(f"{prefix}.post_attn_norm")
                    ),
                    layer.mlp.register_forward_pre_hook(set_active_mlp),
                    layer.mlp.register_forward_hook(trace_hook(f"{prefix}.mlp_out")),
                    layer.mlp.register_forward_hook(clear_active_mlp),
                    layer.register_forward_hook(trace_hook(f"{prefix}.output")),
                    layer.mlp.gate.register_forward_hook(
                        trace_hook(f"{mlp_prefix}.router_logits")
                    ),
                    layer.mlp.shared_gate.register_forward_hook(
                        trace_hook(f"{mlp_prefix}.shared_gate_linear")
                    ),
                    layer.mlp.shared.up_proj.register_forward_hook(
                        trace_hook(f"{mlp_prefix}.shared_up")
                    ),
                    layer.mlp.shared.gate_proj.register_forward_hook(
                        trace_hook(f"{mlp_prefix}.shared_gate")
                    ),
                    layer.mlp.shared.down_proj.register_forward_hook(
                        trace_hook(f"{mlp_prefix}.shared_down")
                    ),
                    layer.mlp.shared.register_forward_hook(
                        trace_hook(f"{mlp_prefix}.shared_out_unscaled")
                    ),
                ]
            )
            trace_handles.append(
                layer.self_attn.q_norm.register_forward_hook(
                    trace_hook(f"{attn_prefix}.q_norm")
                )
            )
            if hasattr(layer.self_attn, "k_norm"):
                trace_handles.append(
                    layer.self_attn.k_norm.register_forward_hook(
                        trace_hook(f"{attn_prefix}.k_norm")
                    )
                )
            if layer.self_attn.rope is not None:
                trace_handles.append(
                    layer.self_attn.rope.register_forward_hook(
                        trace_rope_hook(attn_prefix)
                    )
                )
            if layer.self_attn.lambda_proj is not None:
                trace_handles.append(
                    layer.self_attn.lambda_proj.register_forward_hook(
                        trace_hook(f"{attn_prefix}.lambda")
                    )
                )
        if model.args.yoco_cross_layers > 0:
            trace_handles.extend(
                [
                    model.yoco_norm.register_forward_hook(trace_hook("yoco_norm")),
                    model.k_proj.register_forward_hook(trace_hook("yoco_key")),
                    model.v_proj.register_forward_hook(trace_hook("yoco_value")),
                    model.k_norm.register_forward_hook(trace_hook("yoco_key_norm")),
                ]
            )
        trace_handles.append(model.norm.register_forward_hook(trace_hook("norm")))
        trace_handles.append(
            model.output.register_forward_hook(trace_hook("lm_head"))
        )

        import arch.attention as native_attention
        import arch.moe as native_moe
        import arch.moe_utils_v2 as native_moe_utils
        import kernel.moe_ffn as native_moe_ffn

        original_topk_routing = native_moe.topk_routing
        original_qkv_linear = native_attention.qkv_mix_precision_mxfp8_linear
        original_sliding_attention = native_attention.wrap_sliding_window_attn_func
        original_cached_attention = native_attention.flash_attn_with_kvcache
        original_varlen_attention = native_attention.flash_attn_varlen_func
        original_routed_moe = native_moe.nnscaler_all2all_moe_gmm
        original_te_unpermute = native_moe_utils.unpermute
        original_fused_silu = native_moe_ffn.fused_silu
        original_per_block_cast_to_fp8 = native_moe_ffn.per_block_cast_to_fp8

        def traced_topk_routing(*call_args, **call_kwargs):
            outputs = original_topk_routing(*call_args, **call_kwargs)
            if active_mlp[0]:
                record_trace(f"{active_mlp[0]}.routing_probs", outputs[0])
                record_trace(f"{active_mlp[0]}.routing_map", outputs[1])
            return outputs

        def traced_qkv_linear(*call_args, **call_kwargs):
            outputs = original_qkv_linear(*call_args, **call_kwargs)
            if active_attn[0]:
                record_trace(f"{active_attn[0]}.q_pre_norm", outputs[0])
                record_trace(f"{active_attn[0]}.k_pre_norm", outputs[1])
                record_trace(f"{active_attn[0]}.value", outputs[2])
            return outputs

        def traced_sliding_attention(*call_args, **call_kwargs):
            output = original_sliding_attention(*call_args, **call_kwargs)
            tensor = output[0] if isinstance(output, tuple) else output
            if active_attn[0]:
                record_trace(f"{active_attn[0]}.raw_attn", tensor)
            return output

        def traced_cached_attention(*call_args, **call_kwargs):
            output = original_cached_attention(*call_args, **call_kwargs)
            if active_attn[0]:
                record_trace(f"{active_attn[0]}.raw_attn", output)
            return output

        def traced_varlen_attention(*call_args, **call_kwargs):
            output = original_varlen_attention(*call_args, **call_kwargs)
            if active_attn[0]:
                record_trace(f"{active_attn[0]}.raw_attn", output)
            return output

        def traced_routed_moe(*call_args, **call_kwargs):
            output = original_routed_moe(*call_args, **call_kwargs)
            if active_mlp[0]:
                record_trace(f"{active_mlp[0]}.routed_out", output)
            return output

        def traced_te_unpermute(
            inp,
            row_id_map,
            merging_probs=None,
            restore_shape=None,
            pad_offsets=None,
            **call_kwargs,
        ):
            if active_mlp[0]:
                record_trace(f"{active_mlp[0]}.expert_out_rows", inp)
                record_trace(f"{active_mlp[0]}.row_id_map", row_id_map)
                if pad_offsets is not None:
                    record_trace(f"{active_mlp[0]}.pad_offsets", pad_offsets)
            return original_te_unpermute(
                inp,
                row_id_map,
                merging_probs=merging_probs,
                restore_shape=restore_shape,
                pad_offsets=pad_offsets,
                **call_kwargs,
            )

        def traced_fused_silu(y13, routing_weights, *call_args, **call_kwargs):
            if active_mlp[0]:
                record_trace(f"{active_mlp[0]}.mm1_rows", y13)
                record_trace(
                    f"{active_mlp[0]}.permuted_routing_weights",
                    routing_weights,
                )
            output = original_fused_silu(
                y13, routing_weights, *call_args, **call_kwargs
            )
            if active_mlp[0]:
                record_trace(f"{active_mlp[0]}.w2_input_rows", output)
            return output

        def traced_per_block_cast_to_fp8(weight, *call_args, **call_kwargs):
            weight_quant, weight_scale = original_per_block_cast_to_fp8(
                weight, *call_args, **call_kwargs
            )
            if (
                active_mlp[0] == "layer.1.mlp"
                and weight.ndim == 3
                and weight.shape[-2:] == (3072, 1280)
            ):
                record_trace(
                    f"{active_mlp[0]}.expert10_w2_quant",
                    weight_quant[10],
                )
                record_trace(
                    f"{active_mlp[0]}.expert10_w2_scale",
                    weight_scale[10],
                )
            return weight_quant, weight_scale

        native_attention.qkv_mix_precision_mxfp8_linear = traced_qkv_linear
        native_attention.wrap_sliding_window_attn_func = traced_sliding_attention
        native_attention.flash_attn_with_kvcache = traced_cached_attention
        native_attention.flash_attn_varlen_func = traced_varlen_attention
        native_moe.topk_routing = traced_topk_routing
        native_moe.nnscaler_all2all_moe_gmm = traced_routed_moe
        native_moe_utils.unpermute = traced_te_unpermute
        native_moe_ffn.fused_silu = traced_fused_silu
        native_moe_ffn.per_block_cast_to_fp8 = traced_per_block_cast_to_fp8
        native_moe_patches.extend(
            [
                (
                    native_attention,
                    "qkv_mix_precision_mxfp8_linear",
                    original_qkv_linear,
                ),
                (
                    native_attention,
                    "wrap_sliding_window_attn_func",
                    original_sliding_attention,
                ),
                (
                    native_attention,
                    "flash_attn_with_kvcache",
                    original_cached_attention,
                ),
                (
                    native_attention,
                    "flash_attn_varlen_func",
                    original_varlen_attention,
                ),
                (native_moe, "topk_routing", original_topk_routing),
                (native_moe, "nnscaler_all2all_moe_gmm", original_routed_moe),
                (native_moe_utils, "unpermute", original_te_unpermute),
                (native_moe_ffn, "fused_silu", original_fused_silu),
                (
                    native_moe_ffn,
                    "per_block_cast_to_fp8",
                    original_per_block_cast_to_fp8,
                ),
            ]
        )

    cute_call_count = [0]
    patched_cute_funcs = []
    if args.native_use_cute:
        for module_name in (
            "nnscaler.customized_ops.ring_attention.sliding_window_attn",
            "nnscaler.customized_ops.ring_attention.ring_attn_varlen",
            "nnscaler.customized_ops.ring_attention.zigzag_allgather_attn_varlen",
        ):
            module = importlib.import_module(module_name)
            original = module.flash_attn_cute_varlen_func

            def counted_cute(*call_args, _original=original, **call_kwargs):
                cute_call_count[0] += 1
                return _original(*call_args, **call_kwargs)

            patched_cute_funcs.append((module, original))
            module.flash_attn_cute_varlen_func = counted_cute

    tokenizer, records = _prompt_records(
        args.model,
        args.prompt_suite,
        args.prompt_index,
        args.prompt_start,
        args.prompt_limit,
    )
    results = []
    model_forwards = []
    for logical_batch_index, batch in enumerate(
        _record_batches(records, args.batch_size, args.first_batch_size)
    ):
        token_lists = [record["prompt_token_ids"] for record in batch]
        for record, token_ids in zip(batch, token_lists):
            if max(token_ids) >= model.args.vocab_size:
                raise ValueError(
                    f"Prompt {record['name']} contains token id {max(token_ids)} "
                    f">= model vocab_size {model.args.vocab_size}"
                )

        if args.native_prefill_chunk_size is not None:
            if len(batch) != 1:
                raise ValueError("--native-prefill-chunk-size requires batch-size=1")
            if args.native_no_kv_cache:
                raise ValueError(
                    "--native-prefill-chunk-size requires the Native KV cache"
                )
            if args.native_prefill_chunk_size <= 0:
                raise ValueError("--native-prefill-chunk-size must be positive")
            if args.native_forward_repeats != 1:
                raise ValueError(
                    "--native-prefill-chunk-size requires --native-forward-repeats=1"
                )

            record = batch[0]
            token_ids = token_lists[0]
            kv_cache = create_kv_cache(
                model.args,
                1,
                args.max_model_len,
                _dtype(args.native_dtype),
                device,
            )
            hidden = None
            for chunk_start in range(0, len(token_ids), args.native_prefill_chunk_size):
                chunk_end = min(
                    chunk_start + args.native_prefill_chunk_size,
                    len(token_ids),
                )
                chunk_tokens = torch.tensor(
                    token_ids[chunk_start:chunk_end],
                    dtype=torch.long,
                    device=device,
                )
                positions = torch.arange(
                    chunk_start,
                    chunk_end,
                    device=device,
                    dtype=torch.int32,
                )
                chunk_len = chunk_end - chunk_start
                context = {
                    "cu_seqlens_q": torch.tensor(
                        [0, chunk_len], device=device, dtype=torch.int32
                    ),
                    "cu_seqlens_k": torch.tensor(
                        [0, chunk_end], device=device, dtype=torch.int32
                    ),
                    "max_seqlen_q": chunk_len,
                    "max_seqlen_k": chunk_end,
                    "positions": positions,
                    "kv_cache": kv_cache,
                    "slot_mapping": positions,
                    "layer_index": 0,
                }
                if chunk_start:
                    context["cache_seqlens"] = torch.tensor(
                        [chunk_end], device=device, dtype=torch.int32
                    )
                model_forwards.append(
                    {
                        "logical_batch_index": logical_batch_index,
                        "phase": "chunked_prefill",
                        "num_requests": 1,
                        "num_tokens": chunk_len,
                        "prompt_names": [record["name"]],
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                    }
                )
                hidden, _, _ = model(
                    chunk_tokens,
                    context=context,
                    last_hidden_only=True,
                )

            assert hidden is not None
            logprobs = torch.log_softmax(model.output(hidden[-1:]).float(), dim=-1)[
                0
            ].cpu()
            payload = _result_payload(
                backend="native",
                model_dir=args.model,
                record=record,
                logprobs=logprobs,
            )
            results.append(payload)
            top1 = int(payload["top_ids"][0])
            print(
                f"[native-kl] {record['name']}: top1={top1} "
                f"{tokenizer.decode([top1], skip_special_tokens=False)!r} "
                f"{float(payload['top_logprobs'][0]):.6f}",
                flush=True,
            )
            continue

        seqlens = torch.tensor(
            [len(token_ids) for token_ids in token_lists],
            device=device,
            dtype=torch.int32,
        )
        prefill_tokens = torch.cat(
            [
                torch.tensor(token_ids, dtype=torch.long, device=device)
                for token_ids in token_lists
            ]
        )
        cu_seqlens = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.int32),
                seqlens.cumsum(dim=0).to(torch.int32),
            ]
        )
        positions = torch.cat(
            [
                torch.arange(len(token_ids), device=device, dtype=torch.int32)
                for token_ids in token_lists
            ]
        )
        max_seqlen = int(seqlens.max())
        context = {
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens,
            "max_seqlen_q": max_seqlen,
            "max_seqlen_k": max_seqlen,
            "positions": positions,
        }
        if not args.native_no_kv_cache:
            batch_indices = torch.cat(
                [
                    torch.full(
                        (len(token_ids),),
                        batch_index,
                        device=device,
                        dtype=torch.int32,
                    )
                    for batch_index, token_ids in enumerate(token_lists)
                ]
            )
            context.update(
                {
                    "kv_cache": create_kv_cache(
                        model.args,
                        len(batch),
                        args.max_model_len,
                        _dtype(args.native_dtype),
                        device,
                    ),
                    "slot_mapping": batch_indices * args.max_model_len + positions,
                    "layer_index": 0,
                }
            )
        if args.native_forward_repeats < 1:
            raise ValueError(
                "native_forward_repeats must be positive, got "
                f"{args.native_forward_repeats}"
            )
        repeated_logprobs = []
        for repeat_index in range(args.native_forward_repeats):
            if "layer_index" in context:
                context["layer_index"] = 0
            model_forwards.append(
                {
                    "logical_batch_index": logical_batch_index,
                    "phase": "prefill",
                    "repeat_index": repeat_index,
                    "num_requests": len(batch),
                    "num_tokens": len(prefill_tokens),
                    "prompt_names": [record["name"] for record in batch],
                    "sequence_lengths": [len(token_ids) for token_ids in token_lists],
                }
            )
            hidden, _, _ = model(prefill_tokens, context=context, last_hidden_only=True)
            last_hidden = hidden[cu_seqlens[1:].long() - 1]
            repeated_logprobs.append(
                torch.log_softmax(model.output(last_hidden).float(), dim=-1).cpu()
            )
        batch_logprobs = repeated_logprobs[-1]
        if len(repeated_logprobs) > 1:
            max_repeat_diff = max(
                float((repeat - repeated_logprobs[0]).abs().max())
                for repeat in repeated_logprobs[1:]
            )
            print(
                f"[native-kl] same-process repeat max logprob diff="
                f"{max_repeat_diff:.9g}",
                flush=True,
            )
        for record, logprobs in zip(batch, batch_logprobs):
            payload = _result_payload(
                backend="native",
                model_dir=args.model,
                record=record,
                logprobs=logprobs,
            )
            results.append(payload)
            top1 = int(payload["top_ids"][0])
            print(
                f"[native-kl] {record['name']}: top1={top1} "
                f"{tokenizer.decode([top1], skip_special_tokens=False)!r} "
                f"{float(payload['top_logprobs'][0]):.6f}",
                flush=True,
            )

    _save(
        args.out,
        {
            "backend": "native",
            "model": args.model,
            "native_checkpoint": args.native_checkpoint,
            "attention_version": 4 if args.native_use_cute else 2,
            "native_fa4_source": args.native_fa4_source,
            "native_fa4_runtime": dict(_NATIVE_FA4_RUNTIME),
            "local_attention": args.native_local_attention,
            "kv_cache_enabled": not args.native_no_kv_cache,
            "quant_mode": args.native_quant_mode,
            "quant_block_size": args.native_quant_block_size,
            "torch_fp8_quant_fallback": args.native_use_torch_fp8_quant,
            "transformer_engine_enabled": using_transformer_engine,
            "force_fa_num_splits_one": args.force_fa_num_splits_one,
            "keep_attention_qkv_bf16": args.keep_attention_qkv_bf16,
            "attention_qkv_bf16_modules": attention_qkv_bf16_modules,
            "batch_size": args.batch_size,
            "first_batch_size": args.first_batch_size,
            "model_forwards": model_forwards,
            "results": results,
        },
    )
    if args.trace_out:
        _save(
            args.trace_out,
            {
                "backend": "native",
                "model": args.model,
                "records": trace_records,
            },
        )
        print(f"[native-kl] saved trace {args.trace_out}", flush=True)
        for handle in trace_handles:
            handle.remove()
        for module, name, original in native_moe_patches:
            setattr(module, name, original)
    print(f"[native-kl] saved {args.out}", flush=True)
    if args.native_use_cute:
        print(f"[native-kl] flash_attn.cute calls={cute_call_count[0]}", flush=True)
        for module, original in patched_cute_funcs:
            module.flash_attn_cute_varlen_func = original

    if dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def run_hf(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM

    if args.batch_size != 1 or args.first_batch_size is not None:
        raise ValueError(
            "The HF backend supports only --batch-size=1 without --first-batch-size"
        )
    tokenizer, records = _prompt_records(
        args.model,
        args.prompt_suite,
        args.prompt_index,
        args.prompt_start,
        args.prompt_limit,
    )
    kwargs: dict[str, Any] = {
        "dtype": _dtype(args.dtype),
        "device_map": {"": args.device},
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()

    results = []
    with torch.inference_mode():
        for record in records:
            input_ids = torch.tensor(
                [record["prompt_token_ids"]], dtype=torch.long, device=args.device
            )
            output = model(input_ids=input_ids, use_cache=False)
            logprobs = torch.log_softmax(output.logits[0, -1].float(), dim=-1).cpu()
            payload = _result_payload(
                backend="hf",
                model_dir=args.model,
                record=record,
                logprobs=logprobs,
                chosen_token_id=int(torch.argmax(logprobs)),
            )
            results.append(payload)
            top1 = int(payload["top_ids"][0])
            print(
                f"[hf-kl] {record['name']}: top1={top1} "
                f"{tokenizer.decode([top1], skip_special_tokens=False)!r} "
                f"{float(payload['top_logprobs'][0]):.6f}",
                flush=True,
            )

    _save(args.out, {"backend": "hf", "model": args.model, "results": results})
    print(f"[hf-kl] saved {args.out}", flush=True)


def _vllm_logprob_tensor(step_logprobs: Any, vocab_size: int) -> torch.Tensor:
    values = torch.full((vocab_size,), float("-inf"), dtype=torch.float32)
    if hasattr(step_logprobs, "token_ids") and hasattr(step_logprobs, "logprobs"):
        values[torch.tensor(step_logprobs.token_ids, dtype=torch.long)] = torch.tensor(
            step_logprobs.logprobs, dtype=torch.float32
        )
        return values
    for token_id, logprob in step_logprobs.items():
        values[int(token_id)] = float(logprob.logprob)
    return values


def _disable_transformers_torchvision() -> None:
    site_dir = Path(tempfile.mkdtemp(prefix="logprob_kl_no_torchvision_"))
    (site_dir / "sitecustomize.py").write_text(
        textwrap.dedent(
            """
            def _false(*args, **kwargs):
                return False

            try:
                import transformers.utils as _transformers_utils
                import transformers.utils.import_utils as _import_utils
                for _module in (_transformers_utils, _import_utils):
                    _module.is_torchvision_available = _false
                    _module.is_torchvision_v2_available = _false
                    _module.is_torchvision_greater_or_equal = _false
            except Exception:
                pass
            """
        ),
        encoding="utf-8",
    )
    os.environ["PYTHONPATH"] = (
        str(site_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as import_utils
    except Exception:
        return
    for module in (transformers_utils, import_utils):
        module.is_torchvision_available = lambda: False
        module.is_torchvision_v2_available = lambda: False
        module.is_torchvision_greater_or_equal = lambda *_args, **_kwargs: False


def _patch_local_vllm_metadata() -> None:
    import importlib.metadata as metadata

    original_version = metadata.version

    def version(package: str) -> str:
        return "0.0.0+local" if package == "vllm" else original_version(package)

    metadata.version = version


def _configure_vllm_alignment_env(*, enabled: bool = False) -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1" if enabled else "0"


def _install_vllm_external_fa4(
    source_root: str,
    profile: str,
    *,
    pack_gqa: str = "auto",
    tile_mn: str = "default",
) -> dict[str, Any]:
    import types

    interface, runtime = _import_external_fa4_interface(source_root, profile)
    tuning_overrides = _install_external_fa4_tuning(
        interface,
        pack_gqa=pack_gqa,
        tile_mn=tile_mn,
    )
    adapter_name = "vllm.vllm_flash_attn.cute.interface"
    package_name = "vllm.vllm_flash_attn.cute"

    adapter = types.ModuleType(adapter_name)
    adapter.__file__ = interface.__file__

    def _flash_attn_fwd(*args, **kwargs):
        result = interface._flash_attn_fwd(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) not in (2, 4):
            raise RuntimeError(
                "Unexpected external FA4 private return shape: "
                f"{type(result).__name__}, len={getattr(result, '__len__', lambda: '?')()}"
            )
        return result[0], result[1]

    adapter._flash_attn_fwd = _flash_attn_fwd
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(source_root).resolve() / "flash_attn" / "cute")]
    package.interface = adapter
    sys.modules[package_name] = package
    sys.modules[adapter_name] = adapter

    runtime = {
        **runtime,
        "tuning_overrides": tuning_overrides,
        "vllm_private_return_adapter": "4-to-2",
    }
    print(
        "[vllm-kl] installed external FA4 adapter "
        f"profile={profile} sha256={runtime['source_sha256']}",
        flush=True,
    )
    return runtime


def run_vllm(args: argparse.Namespace) -> None:
    _configure_vllm_alignment_env(enabled=args.v1_multiprocessing)
    _disable_transformers_torchvision()
    _patch_local_vllm_metadata()
    external_fa4_runtime: dict[str, Any] = {}
    if args.fa4_source_root:
        if args.flash_attn_version != 4:
            raise ValueError("--fa4-source-root requires --flash-attn-version 4")
        external_fa4_runtime = _install_vllm_external_fa4(
            args.fa4_source_root,
            args.fa4_profile,
            pack_gqa=args.fa4_pack_gqa,
            tile_mn=args.fa4_tile_mn,
        )
    from vllm import LLM, SamplingParams

    tokenizer, records = _prompt_records(
        args.model,
        args.prompt_suite,
        args.prompt_index,
        args.prompt_start,
        args.prompt_limit,
    )
    if args.trace_out:
        raise ValueError("--trace-out is only supported by the Native backend")
    if args.keep_attention_qkv_bf16:
        if args.quantization not in ("fp8_per_block", "mxfp8"):
            raise ValueError(
                "--keep-attention-qkv-bf16 requires online fp8_per_block or "
                "mxfp8 quantization"
            )
        for pattern in ATTENTION_QKV_BF16_IGNORE:
            if pattern not in args.quantization_ignore:
                args.quantization_ignore.append(pattern)
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": (args.max_num_batched_tokens or args.max_model_len),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "max_logprobs": args.max_logprobs,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    if args.log_iteration_details:
        llm_kwargs["enable_logging_iteration_details"] = True
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.enable_expert_parallel:
        llm_kwargs["enable_expert_parallel"] = True
    if args.distributed_executor_backend:
        llm_kwargs["distributed_executor_backend"] = (
            args.distributed_executor_backend
        )
    if args.enable_chunked_prefill is not None:
        llm_kwargs["enable_chunked_prefill"] = args.enable_chunked_prefill
    if args.kv_sharing_fast_prefill:
        llm_kwargs["kv_sharing_fast_prefill"] = True
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    quantization_config = (
        json.loads(args.quantization_config_json)
        if args.quantization_config_json
        else None
    )
    if args.quantization_ignore:
        quantization_config = quantization_config or {}
        quantization_config["ignore"] = args.quantization_ignore
    if quantization_config is not None:
        llm_kwargs["quantization_config"] = quantization_config
    if (
        args.attention_backend
        or args.flash_attn_version is not None
        or args.force_fa_num_splits_one
    ):
        attention_config = {}
        if args.attention_backend:
            attention_config["backend"] = args.attention_backend
        if args.flash_attn_version is not None:
            attention_config["flash_attn_version"] = args.flash_attn_version
        if args.force_fa_num_splits_one:
            attention_config["flash_attn_force_num_splits_one"] = True
        llm_kwargs["attention_config"] = attention_config
    if args.moe_backend:
        llm_kwargs["moe_backend"] = args.moe_backend
    if args.compilation_config_json:
        llm_kwargs["compilation_config"] = json.loads(args.compilation_config_json)

    llm = LLM(**llm_kwargs)
    params = SamplingParams(
        temperature=0.0, max_tokens=1, logprobs=args.max_logprobs, seed=args.seed
    )
    vocab_size = int(llm.llm_engine.model_config.get_vocab_size())
    parallel_config = llm.llm_engine.vllm_config.parallel_config
    dp_rank = int(parallel_config.data_parallel_rank or 0)
    results = []
    for batch in _record_batches(records, args.batch_size, args.first_batch_size):
        outputs = llm.generate(
            [
                {
                    "prompt_token_ids": record["prompt_token_ids"],
                    "prompt": record["prompt_text"],
                }
                for record in batch
            ],
            sampling_params=params,
            use_tqdm=False,
        )
        for record, request_output in zip(batch, outputs):
            output = request_output.outputs[0]
            logprobs = _vllm_logprob_tensor(output.logprobs[0], vocab_size).cpu()
            finite = torch.isfinite(logprobs).sum().item()
            if finite != vocab_size:
                raise RuntimeError(
                    f"Expected full vocab logprobs for {record['name']}, "
                    f"got {finite}/{vocab_size}"
                )
            payload = _result_payload(
                backend="vllm",
                model_dir=args.model,
                record=record,
                logprobs=logprobs,
                chosen_token_id=int(output.token_ids[0]),
            )
            results.append(payload)
            top1 = int(payload["top_ids"][0])
            if dp_rank == 0:
                print(
                    f"[vllm-kl] {record['name']}: top1={top1} "
                    f"{tokenizer.decode([top1], skip_special_tokens=False)!r} "
                    f"{float(payload['top_logprobs'][0]):.6f}",
                    flush=True,
                )
    if dp_rank == 0:
        vllm_fa4_runtime = external_fa4_runtime
        if args.flash_attn_version == 4 and not vllm_fa4_runtime:
            _import_fa4_varlen_func("vllm-vendored")
            vllm_fa4_runtime = dict(_NATIVE_FA4_RUNTIME)
        _save(
            args.out,
            {
                "backend": "vllm",
                "model": args.model,
                "batch_size": args.batch_size,
                "first_batch_size": args.first_batch_size,
                "data_parallel_size": args.data_parallel_size,
                "expert_parallel_enabled": args.enable_expert_parallel,
                "quantization": args.quantization,
                "moe_backend": args.moe_backend,
                "attention_backend": args.attention_backend,
                "attention_version": args.flash_attn_version,
                "vllm_fa4_runtime": vllm_fa4_runtime,
                "compilation_config": (
                    json.loads(args.compilation_config_json)
                    if args.compilation_config_json
                    else None
                ),
                "force_fa_num_splits_one": args.force_fa_num_splits_one,
                "keep_attention_qkv_bf16": args.keep_attention_qkv_bf16,
                "quantization_ignore": args.quantization_ignore,
                "v1_multiprocessing_enabled": args.v1_multiprocessing,
                "iteration_details_logged": args.log_iteration_details,
                "prefix_caching_enabled": args.enable_prefix_caching,
                "results": results,
            },
        )
        print(f"[vllm-kl] saved {args.out}", flush=True)


def _top_rows(
    tokenizer: Any, logprobs: torch.Tensor, top_k: int
) -> list[dict[str, Any]]:
    top = torch.topk(logprobs, k=top_k)
    rows = []
    for rank, (token_id, logprob) in enumerate(
        zip(top.indices.tolist(), top.values.tolist()), start=1
    ):
        rows.append(
            {
                "rank": rank,
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                "logprob": float(logprob),
            }
        )
    return rows


def _metrics(
    reference_lp: torch.Tensor, candidate_lp: torch.Tensor
) -> dict[str, float]:
    p = reference_lp.exp()
    q = candidate_lp.exp()
    m = 0.5 * (p + q)
    eps = torch.finfo(torch.float32).tiny
    p_safe = p.clamp_min(eps)
    q_safe = q.clamp_min(eps)
    m_safe = m.clamp_min(eps)
    return {
        "kl_reference_to_candidate": torch.sum(
            p * (p_safe.log() - q_safe.log())
        ).item(),
        "kl_candidate_to_reference": torch.sum(
            q * (q_safe.log() - p_safe.log())
        ).item(),
        # Backward-compatible aliases used by existing YOCO notes/scripts.
        "kl_native_to_vllm": torch.sum(p * (p_safe.log() - q_safe.log())).item(),
        "kl_vllm_to_native": torch.sum(q * (q_safe.log() - p_safe.log())).item(),
        "js_divergence": (
            0.5 * torch.sum(p * (p_safe.log() - m_safe.log())).item()
            + 0.5 * torch.sum(q * (q_safe.log() - m_safe.log())).item()
        ),
        "max_abs_logprob_diff": torch.max(
            torch.abs(reference_lp - candidate_lp)
        ).item(),
        "mean_abs_logprob_diff": torch.mean(
            torch.abs(reference_lp - candidate_lp)
        ).item(),
    }


def compare(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    reference_payload = torch.load(args.reference, map_location="cpu")
    candidate_payload = torch.load(args.candidate, map_location="cpu")
    reference_results = {
        row["prompt"]["name"]: row for row in _payload_results(reference_payload)
    }
    candidate_results = {
        row["prompt"]["name"]: row for row in _payload_results(candidate_payload)
    }
    names = [name for name in reference_results if name in candidate_results]
    if not names:
        raise ValueError("No overlapping prompt names in compared payloads")

    model_for_tokenizer = args.model or candidate_payload.get(
        "model", reference_payload.get("model")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_for_tokenizer, trust_remote_code=True
    )
    prompt_summaries = []
    for name in names:
        reference = reference_results[name]
        candidate = candidate_results[name]
        reference_lp = reference["logprobs"].float()
        candidate_lp = candidate["logprobs"].float()
        if reference_lp.shape != candidate_lp.shape:
            raise ValueError(
                f"Shape mismatch for {name}: "
                f"reference={tuple(reference_lp.shape)} "
                f"candidate={tuple(candidate_lp.shape)}"
            )
        summary = {
            "prompt_name": name,
            "prompt_kind": reference["prompt"]["kind"],
            "prompt_len": reference["prompt"]["prompt_len"],
            **_metrics(reference_lp, candidate_lp),
            "native_top": _top_rows(tokenizer, reference_lp, args.top_k),
            "vllm_top": _top_rows(tokenizer, candidate_lp, args.top_k),
            "reference_top": _top_rows(tokenizer, reference_lp, args.top_k),
            "candidate_top": _top_rows(tokenizer, candidate_lp, args.top_k),
        }
        prompt_summaries.append(summary)

    metric_names = (
        "kl_reference_to_candidate",
        "kl_candidate_to_reference",
        "kl_native_to_vllm",
        "kl_vllm_to_native",
        "js_divergence",
        "max_abs_logprob_diff",
        "mean_abs_logprob_diff",
    )
    aggregate = {
        f"mean_{metric}": sum(row[metric] for row in prompt_summaries)
        / len(prompt_summaries)
        for metric in metric_names
    }
    aggregate["num_prompts"] = len(prompt_summaries)
    result = {
        "reference_config": _comparison_metadata(reference_payload),
        "candidate_config": _comparison_metadata(candidate_payload),
        "aggregate": aggregate,
        "prompts": prompt_summaries,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as writer:
        json.dump(result, writer, ensure_ascii=False, indent=2)
        writer.write("\n")
    print(json.dumps(aggregate, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.cmd == "native":
        run_native(args)
    elif args.cmd == "hf":
        run_hf(args)
    elif args.cmd == "vllm":
        run_vllm(args)
    elif args.cmd == "compare":
        compare(args)
    else:
        raise AssertionError(args.cmd)


if __name__ == "__main__":
    main()
