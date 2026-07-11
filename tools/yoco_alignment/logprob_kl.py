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
import re
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
    PromptSpec(name="france_capital", kind="completion",
               text="The capital of France is"),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-vocab next-token KL alignment probe"
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    for name in ("native", "hf", "vllm"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--model", required=True,
                         help="HF/vLLM model path or Hugging Face repo id")
        sub.add_argument("--out", required=True, help="Output .pt path")
        sub.add_argument("--prompt-suite", choices=("default", "mixed5"),
                         default="mixed5")
        sub.add_argument("--prompt-index", type=int, default=0)
        sub.add_argument("--prompt-limit", type=int)
        sub.add_argument("--max-model-len", type=int, default=8192)
        sub.add_argument("--seed", type=int, default=0)
        if name in ("native", "vllm"):
            sub.add_argument(
                "--dump-hidden-dir",
                help=(
                    "Optional directory for per-prompt hidden-state dumps. "
                    "Only records selected modules and does not affect the "
                    "default KL payload."
                ),
            )
            sub.add_argument(
                "--dump-hidden-level",
                choices=("block", "operator"),
                default="block",
                help="Granularity for --dump-hidden-dir.",
            )
            sub.add_argument(
                "--dump-hidden-filter",
                action="append",
                default=[],
                help=(
                    "Regex for module names to dump. Can be passed multiple "
                    "times. When omitted, built-in block/operator filters are "
                    "used."
                ),
            )
            sub.add_argument(
                "--dump-hidden-tensors",
                action="store_true",
                help=(
                    "Also save full tensors. By default only last-token "
                    "vectors and summary stats are saved."
                ),
            )

        if name == "native":
            sub.add_argument("--native-checkpoint", required=True)
            sub.add_argument("--llm-train-dir",
                             default="/data/users/shaohanh/llm-train")
            sub.add_argument("--native-dtype", choices=("bfloat16",),
                             default="bfloat16")
            sub.add_argument(
                "--training-yaml",
                help=(
                    "Optional llm-train yaml. When set, native ModelArgs "
                    "inherits --quant_mode/--quant_block_size from the "
                    "torchrun command in that file."
                ),
            )
            sub.add_argument(
                "--native-quant-mode",
                choices=("bfloat16", "mxfp8"),
                help="Override native ModelArgs.quant_mode for parity probes.",
            )
            sub.add_argument(
                "--native-quant-block-size",
                type=int,
                help="Override native ModelArgs.quant_block_size.",
            )
        elif name == "hf":
            sub.add_argument("--device", default="cuda:0")
            sub.add_argument("--dtype", choices=("bfloat16", "float16",
                                                 "float32"),
                             default="bfloat16")
            sub.add_argument("--attn-implementation")
        else:
            sub.add_argument("--dtype", choices=("auto", "bfloat16",
                                                 "float16", "float32"),
                             default="auto")
            sub.add_argument("--tensor-parallel-size", type=int, default=1)
            sub.add_argument("--gpu-memory-utilization", type=float, default=0.9)
            sub.add_argument("--kv-sharing-fast-prefill", action="store_true")
            sub.add_argument("--enforce-eager", action="store_true")
            sub.add_argument(
                "--compilation-config-json",
                help="Raw JSON object passed as vLLM compilation_config.",
            )
            sub.add_argument("--quantization", default=None)
            sub.add_argument(
                "--quantization-config-json",
                help=(
                    "Raw JSON object passed as vLLM quantization_config. "
                    "Useful for explicit online quant specs such as "
                    "'{\"moe\":{\"weight\":\"mxfp8\"}}'."
                ),
            )
            sub.add_argument(
                "--quantization-ignore",
                action="append",
                default=[],
                help=(
                    "Layer prefix to exclude from vLLM online quantization. "
                    "Can be passed multiple times."
                ),
            )
            sub.add_argument(
                "--attention-backend",
                default=None,
                help=(
                    "vLLM attention backend override, e.g. FLASH_ATTN, "
                    "FLASHINFER, or TRITON_ATTN."
                ),
            )
            sub.add_argument("--moe-backend", default=None)
            sub.add_argument("--max-logprobs", type=int, default=-1)

    cmp_parser = subparsers.add_parser("compare")
    cmp_parser.add_argument("--reference", "--native", dest="reference",
                            required=True)
    cmp_parser.add_argument("--candidate", "--vllm", dest="candidate",
                            required=True)
    cmp_parser.add_argument("--out-json", required=True)
    cmp_parser.add_argument("--top-k", type=int, default=20)
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


def _training_yaml_flag(path: str | None, flag: str) -> str | None:
    if path is None:
        return None
    text = Path(path).read_text(encoding="utf-8")
    spellings = {flag, flag.replace("_", "-")}
    for spelling in spellings:
        marker = f"--{spelling}"
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line.startswith(marker):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0] == marker:
                return parts[1]
    return None


def _training_yaml_bool_flag(path: str | None, flag: str) -> bool | None:
    if path is None:
        return None
    text = Path(path).read_text(encoding="utf-8")
    spellings = {flag, flag.replace("_", "-")}
    for spelling in spellings:
        marker = f"--{spelling}"
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line.startswith(marker):
                continue
            parts = line.split()
            if not parts or parts[0] != marker:
                continue
            if len(parts) == 1:
                return True
            value = parts[1].lower()
            if value in {"1", "true", "yes", "on"}:
                return True
            if value in {"0", "false", "no", "off"}:
                return False
    return None


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


def _apply_chat_template(tokenizer: Any,
                         messages: list[dict[str, str]]) -> str:
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
        DEFAULT_PROMPTS[:prompt_limit]
        if prompt_limit is not None
        else DEFAULT_PROMPTS
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
        records.append({
            "name": spec.name,
            "kind": spec.kind,
            "prompt_text": prompt_text,
            "prompt_token_ids": token_ids,
            "prompt_len": len(token_ids),
        })
    return records


def _mixed5_prompt_records(tokenizer: Any) -> list[dict[str, Any]]:
    bos_id = _bos_id(tokenizer)
    records = []
    for name, text in MIXED5_PROMPTS:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if bos_id is not None:
            token_ids = [bos_id] + token_ids
        records.append({
            "name": name,
            "kind": "completion",
            "prompt_text": text,
            "prompt_token_ids": token_ids,
            "prompt_len": len(token_ids),
        })
    return records


def _prompt_records(
    model_dir: str,
    prompt_suite: str,
    prompt_index: int,
    prompt_limit: int | None,
) -> tuple[Any, list[dict[str, Any]]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if prompt_suite == "mixed5":
        records = _mixed5_prompt_records(tokenizer)
        if prompt_limit is not None:
            records = records[:prompt_limit]
        elif prompt_index:
            records = [records[prompt_index]]
        return tokenizer, records

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
    top = torch.topk(logprobs, k=k)
    return {"top_ids": top.indices.cpu(), "top_logprobs": top.values.cpu()}


def _result_payload(
    *,
    backend: str,
    model_dir: str,
    record: dict[str, Any],
    logprobs: torch.Tensor,
    chosen_token_id: int | None = None,
) -> dict[str, Any]:
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


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


class _HiddenDumper:
    _NATIVE_BLOCK_RE = re.compile(r"^layers\.\d+$")
    _VLLM_BLOCK_RE = re.compile(r"^model\.layers\.\d+$")
    _OPERATOR_SUFFIXES = (
        ".input_layernorm",
        ".post_attention_layernorm",
        ".self_attn",
        ".self_attn.qkv_proj",
        ".self_attn.q_proj",
        ".self_attn.k_proj",
        ".self_attn.v_proj",
        ".self_attn.q_norm",
        ".self_attn.k_norm",
        ".self_attn.rope",
        ".self_attn.rotary_emb",
        ".self_attn.o_proj",
        ".self_attn.lambda_proj",
        ".self_attn.attn",
        ".mlp",
        ".mlp.gate",
        ".mlp.experts",
        ".mlp.shared",
        ".mlp.shared.up_proj",
        ".mlp.shared.gate_proj",
        ".mlp.shared.down_proj",
        ".mlp.shared_gate",
        ".mlp.shared_experts",
        ".mlp.shared_experts.gate_up_proj",
        ".mlp.shared_experts.down_proj",
    )
    _INPUT_OPERATOR_SUFFIXES = (
        ".self_attn.o_proj",
        ".mlp",
        ".mlp.shared.down_proj",
        ".mlp.shared_experts.down_proj",
    )

    def __init__(
        self,
        *,
        dump_dir: str | None,
        backend: str,
        level: str,
        filters: list[str],
        save_tensors: bool,
    ) -> None:
        self.enabled = dump_dir is not None
        self.dump_dir = Path(dump_dir) if dump_dir is not None else None
        self.backend = backend
        self.level = level
        self.filters = [re.compile(pattern) for pattern in filters]
        self.save_tensors = save_tensors
        self.prompt: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []

    def should_capture(self, name: str) -> bool:
        if not self.enabled:
            return False
        if self.filters:
            return any(pattern.search(name) for pattern in self.filters)
        if self.level == "block":
            if self.backend == "native":
                return (
                    name == "tok_embeddings"
                    or name == "norm"
                    or name in {"yoco_norm", "k_proj", "v_proj"}
                    or self._NATIVE_BLOCK_RE.fullmatch(name) is not None
                )
            return (
                name == "model.embed_tokens"
                or name == "model.norm"
                or name in {"model.yoco_norm", "model.yoco_k_proj", "model.yoco_v_proj"}
                or self._VLLM_BLOCK_RE.fullmatch(name) is not None
            )
        return self._should_capture_operator(name)

    def _should_capture_operator(self, name: str) -> bool:
        if self.backend == "native":
            if name in {"tok_embeddings", "norm", "yoco_norm", "k_proj", "v_proj"}:
                return True
            return any(name.endswith(suffix) for suffix in self._OPERATOR_SUFFIXES)
        if name in {
            "model.embed_tokens",
            "model.norm",
            "model.yoco_norm",
            "model.yoco_k_proj",
            "model.yoco_v_proj",
        }:
            return True
        if re.search(r"\.self_attn\.attn\.\d+$", name):
            return True
        return any(name.endswith(suffix) for suffix in self._OPERATOR_SUFFIXES)

    def set_prompt(self, prompt: dict[str, Any]) -> None:
        self.prompt = prompt
        self.records = []

    def clear_prompt(self) -> None:
        self.prompt = None

    def hook(self, name: str):
        def inner(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            if self.prompt is None:
                return
            if isinstance(output, (tuple, list)):
                tensor_outputs = [
                    item for item in output if isinstance(item, torch.Tensor)
                ]
                if tensor_outputs:
                    for idx, tensor in enumerate(tensor_outputs):
                        self._append(f"{name}.{idx}", tensor)
                    return
            tensor = _first_tensor(output)
            if tensor is None:
                return
            self._append(name, tensor)

        return inner

    def pre_hook(self, name: str):
        def inner(_module: torch.nn.Module, inputs: Any) -> None:
            if self.prompt is None:
                return
            tensor = _first_tensor(inputs)
            if tensor is None:
                return
            self._append(f"{name}.input", tensor)

        return inner

    def should_capture_input(self, name: str) -> bool:
        if not self.enabled or self.level != "operator":
            return False
        if self.filters and not any(pattern.search(name) for pattern in self.filters):
            return False
        return any(name.endswith(suffix) for suffix in self._INPUT_OPERATOR_SUFFIXES)

    def append_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if self.prompt is None:
            return
        self._append(name, tensor)

    def _append(self, name: str, tensor: torch.Tensor) -> None:
        detached = tensor.detach()
        flat = detached.reshape(detached.shape[0], -1) if detached.ndim > 1 else detached.reshape(1, -1)
        last = flat[-1].float().cpu()
        stats = detached.float()
        record: dict[str, Any] = {
            "index": len(self.records),
            "name": name,
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "last_token": last,
            "mean": float(stats.mean().item()),
            "std": float(stats.std(unbiased=False).item()),
            "absmax": float(stats.abs().max().item()),
            "l2": float(torch.linalg.vector_norm(stats).item()),
        }
        if self.save_tensors:
            record["tensor"] = detached.cpu()
        self.records.append(record)

    def save(self) -> None:
        if not self.enabled or self.prompt is None:
            return
        assert self.dump_dir is not None
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        name = self.prompt["name"]
        path = self.dump_dir / f"{self.backend}_{name}_{self.level}.pt"
        torch.save(
            {
                "backend": self.backend,
                "level": self.level,
                "prompt": self.prompt,
                "records": self.records,
            },
            path,
        )
        print(
            f"[hidden-dump] saved {path} ({len(self.records)} records)",
            flush=True,
        )


def _install_hidden_hooks(
    model: torch.nn.Module,
    dumper: _HiddenDumper,
) -> list[torch.utils.hooks.RemovableHandle]:
    if not dumper.enabled:
        return []
    handles = []
    for name, module in model.named_modules():
        if dumper.should_capture_input(name):
            handles.append(module.register_forward_pre_hook(dumper.pre_hook(name)))
        if dumper.should_capture(name):
            handles.append(module.register_forward_hook(dumper.hook(name)))
    print(
        f"[hidden-dump] installed {len(handles)} {dumper.level} hooks "
        f"for {dumper.backend}",
        flush=True,
    )
    return handles


def _payload_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return payload.get("results", [payload])


@torch.no_grad()
def run_native(args: argparse.Namespace) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    llm_dir = Path(args.llm_train_dir).resolve() / "llm"
    sys.path.insert(0, str(llm_dir))
    from arch.model import Model, ModelArgs, create_kv_cache

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda")

    metadata_path = Path(args.native_checkpoint) / "metadata.json"
    with metadata_path.open(encoding="utf-8") as reader:
        metadata = json.load(reader)
    modelargs = ModelArgs()
    for key, value in metadata["modelargs"].items():
        setattr(modelargs, key, value)
    yaml_quant_mode = _training_yaml_flag(args.training_yaml, "quant_mode")
    yaml_quant_block_size = _training_yaml_flag(
        args.training_yaml, "quant_block_size"
    )
    native_quant_mode = args.native_quant_mode or yaml_quant_mode
    native_quant_block_size = args.native_quant_block_size
    if native_quant_block_size is None and yaml_quant_block_size is not None:
        native_quant_block_size = int(yaml_quant_block_size)
    if native_quant_mode is not None:
        modelargs.quant_mode = native_quant_mode
    if native_quant_block_size is not None:
        modelargs.quant_block_size = native_quant_block_size
    yaml_use_cute = _training_yaml_bool_flag(args.training_yaml, "use_cute")
    if yaml_use_cute is not None:
        modelargs.use_cute = yaml_use_cute
    print(
        "[native-kl] ModelArgs "
        f"quant_mode={getattr(modelargs, 'quant_mode', None)} "
        f"quant_block_size={getattr(modelargs, 'quant_block_size', None)} "
        f"use_cute={getattr(modelargs, 'use_cute', None)}",
        flush=True,
    )

    init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["dp"])
    default_device = torch.get_default_device()
    default_dtype = torch.get_default_dtype()
    torch.set_default_device(device)
    torch.set_default_dtype(_dtype(args.native_dtype))
    model = Model(modelargs)
    torch.set_default_device(default_device)
    torch.set_default_dtype(default_dtype)
    model.eval()

    state = torch.load(
        Path(args.native_checkpoint) / "model_state_rank_0.pth",
        map_location=_device_mapping(-1),
        mmap=True,
    )
    state = {
        key: value for key, value in state.items()
        if not key.startswith("moe_loss.")
    }
    model.load_state_dict(state)
    print("[native-kl] model loaded", flush=True)

    dumper = _HiddenDumper(
        dump_dir=args.dump_hidden_dir,
        backend="native",
        level=args.dump_hidden_level,
        filters=args.dump_hidden_filter,
        save_tensors=args.dump_hidden_tensors,
    )
    hook_handles = _install_hidden_hooks(model, dumper)
    native_attention = None
    original_qkv_mxfp8 = None
    original_flash_attn_varlen = None
    if dumper.enabled and args.dump_hidden_level == "operator":
        import arch.attention as native_attention

        original_qkv_mxfp8 = native_attention.qkv_mix_precision_mxfp8_linear
        original_flash_attn_varlen = native_attention.flash_attn_varlen_func

        def wrapped_qkv_mix_precision_mxfp8_linear(
            input: torch.Tensor,
            wq: torch.Tensor,
            wk: torch.Tensor,
            wv: torch.Tensor,
            quant_block_size: int = 128,
        ):
            q, k, v = original_qkv_mxfp8(
                input, wq, wk, wv, quant_block_size=quant_block_size
            )
            dumper.append_tensor("native.self_attn.qkv_mxfp8.q", q)
            dumper.append_tensor("native.self_attn.qkv_mxfp8.k", k)
            dumper.append_tensor("native.self_attn.qkv_mxfp8.v", v)
            return q, k, v

        native_attention.qkv_mix_precision_mxfp8_linear = (
            wrapped_qkv_mix_precision_mxfp8_linear
        )

        def wrapped_flash_attn_varlen_func(*func_args: Any, **func_kwargs: Any):
            q = func_kwargs.get("q", func_args[0] if len(func_args) > 0 else None)
            k = func_kwargs.get("k", func_args[1] if len(func_args) > 1 else None)
            v = func_kwargs.get("v", func_args[2] if len(func_args) > 2 else None)
            if isinstance(q, torch.Tensor):
                dumper.append_tensor("native.self_attn.flash_attn.q", q)
            if isinstance(k, torch.Tensor):
                dumper.append_tensor("native.self_attn.flash_attn.k", k)
            if isinstance(v, torch.Tensor):
                dumper.append_tensor("native.self_attn.flash_attn.v", v)
            output = original_flash_attn_varlen(*func_args, **func_kwargs)
            tensor = output[0] if isinstance(output, tuple) else output
            dumper.append_tensor("native.self_attn.flash_attn", tensor)
            return output

        native_attention.flash_attn_varlen_func = wrapped_flash_attn_varlen_func

    tokenizer, records = _prompt_records(
        args.model, args.prompt_suite, args.prompt_index, args.prompt_limit
    )
    results = []
    for record in records:
        token_ids = record["prompt_token_ids"]
        if max(token_ids) >= model.args.vocab_size:
            raise ValueError(
                f"Prompt {record['name']} contains token id {max(token_ids)} "
                f">= model vocab_size {model.args.vocab_size}"
            )

        prefill_tokens = torch.tensor(token_ids, dtype=torch.long, device=device)
        seqlen = len(token_ids)
        cu_seqlens = torch.tensor([0, seqlen], device=device, dtype=torch.int32)
        positions = torch.arange(0, seqlen, device=device, dtype=torch.int32)
        kv_cache = create_kv_cache(
            model.args, 1, args.max_model_len, _dtype(args.native_dtype), device
        )
        context = {
            "kv_cache": kv_cache,
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens,
            "max_seqlen_q": seqlen,
            "max_seqlen_k": seqlen,
            "positions": positions,
            "slot_mapping": positions,
            "layer_index": 0,
        }
        dumper.set_prompt(record)
        hidden, _, _ = model(prefill_tokens, context=context, last_hidden_only=True)
        dumper.save()
        dumper.clear_prompt()
        logits = model.output(hidden[-1]).float()
        logprobs = torch.log_softmax(logits, dim=-1).cpu()
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

    _save(args.out, {"backend": "native", "model": args.model, "results": results})
    print(f"[native-kl] saved {args.out}", flush=True)

    for handle in hook_handles:
        handle.remove()
    if native_attention is not None and original_qkv_mxfp8 is not None:
        native_attention.qkv_mix_precision_mxfp8_linear = original_qkv_mxfp8
    if native_attention is not None and original_flash_attn_varlen is not None:
        native_attention.flash_attn_varlen_func = original_flash_attn_varlen

    if dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def run_hf(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM

    tokenizer, records = _prompt_records(
        args.model, args.prompt_suite, args.prompt_index, args.prompt_limit
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
    # This probe is text-only. Some local CUDA images have torchvision wheels
    # that are importable but ABI-incompatible with the installed torch; avoid
    # importing torchvision through transformers' optional image utilities.
    # vLLM also inspects model classes in a fresh Python subprocess, so install
    # a tiny sitecustomize module that applies the same monkeypatch there.
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
    # Running this script directly from a source tree means there may be no
    # installed `vllm` distribution metadata. Platform detection only needs to
    # distinguish CPU wheels from CUDA builds, so provide a local non-CPU
    # version string for this process.
    import importlib.metadata as metadata

    original_version = metadata.version

    def version(package: str) -> str:
        if package == "vllm":
            return "0.0.0+local"
        return original_version(package)

    metadata.version = version


def _get_vllm_worker_model(llm: Any) -> torch.nn.Module:
    model_executor = getattr(llm.llm_engine, "model_executor", None)
    if model_executor is None:
        engine_core_client = getattr(llm.llm_engine, "engine_core", None)
        engine_core = getattr(engine_core_client, "engine_core", None)
        model_executor = getattr(engine_core, "model_executor", None)
    if model_executor is None:
        raise RuntimeError(
            "Cannot access vLLM model for hidden dump. Set "
            "VLLM_ENABLE_V1_MULTIPROCESSING=0 before constructing LLM."
        )

    driver_worker = getattr(model_executor, "driver_worker", None)
    worker = getattr(driver_worker, "worker", None)
    if worker is not None and hasattr(worker, "get_model"):
        return worker.get_model()

    model_runner = getattr(worker, "model_runner", None)
    if model_runner is not None and hasattr(model_runner, "get_model"):
        return model_runner.get_model()

    raise RuntimeError("Cannot locate vLLM worker model for hidden dump")


def run_vllm(args: argparse.Namespace) -> None:
    _disable_transformers_torchvision()
    _patch_local_vllm_metadata()
    if args.dump_hidden_dir:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams

    tokenizer, records = _prompt_records(
        args.model, args.prompt_suite, args.prompt_index, args.prompt_limit
    )
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "max_logprobs": args.max_logprobs,
        "enforce_eager": args.enforce_eager,
    }
    if args.kv_sharing_fast_prefill:
        llm_kwargs["kv_sharing_fast_prefill"] = True
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    quantization_config: dict[str, Any] | None = None
    if args.quantization_config_json:
        quantization_config = json.loads(args.quantization_config_json)
    if args.quantization_ignore:
        if quantization_config is None:
            quantization_config = {}
        quantization_config["ignore"] = args.quantization_ignore
    if quantization_config is not None:
        llm_kwargs["quantization_config"] = quantization_config
    if args.attention_backend:
        llm_kwargs["attention_config"] = {"backend": args.attention_backend}
    if args.moe_backend:
        llm_kwargs["moe_backend"] = args.moe_backend
    if args.compilation_config_json:
        llm_kwargs["compilation_config"] = json.loads(
            args.compilation_config_json
        )

    llm = LLM(**llm_kwargs)
    dumper = _HiddenDumper(
        dump_dir=args.dump_hidden_dir,
        backend="vllm",
        level=args.dump_hidden_level,
        filters=args.dump_hidden_filter,
        save_tensors=args.dump_hidden_tensors,
    )
    hook_handles = (
        _install_hidden_hooks(_get_vllm_worker_model(llm), dumper)
        if dumper.enabled
        else []
    )
    params = SamplingParams(
        temperature=0.0, max_tokens=1, logprobs=args.max_logprobs, seed=args.seed
    )
    vocab_size = int(llm.llm_engine.model_config.get_vocab_size())
    results = []
    for record in records:
        # Keep KL probes schedule-matched with native/HF: those paths run one
        # prompt per forward, while a single batched vLLM generate can change
        # numerics through chunked prefill and grouped MoE dispatch order.
        dumper.set_prompt(record)
        outputs = llm.generate(
            [{
                "prompt_token_ids": record["prompt_token_ids"],
                "prompt": record["prompt_text"],
            }],
            sampling_params=params,
            use_tqdm=False,
        )
        dumper.save()
        dumper.clear_prompt()
        output = outputs[0].outputs[0]
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
        print(
            f"[vllm-kl] {record['name']}: top1={top1} "
            f"{tokenizer.decode([top1], skip_special_tokens=False)!r} "
            f"{float(payload['top_logprobs'][0]):.6f}",
            flush=True,
        )
    for handle in hook_handles:
        handle.remove()
    _save(args.out, {"backend": "vllm", "model": args.model, "results": results})
    print(f"[vllm-kl] saved {args.out}", flush=True)


def _top_rows(
    tokenizer: Any, logprobs: torch.Tensor, top_k: int
) -> list[dict[str, Any]]:
    top = torch.topk(logprobs, k=top_k)
    rows = []
    for rank, (token_id, logprob) in enumerate(
        zip(top.indices.tolist(), top.values.tolist()), start=1
    ):
        rows.append({
            "rank": rank,
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)], skip_special_tokens=False),
            "logprob": float(logprob),
        })
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

    model_for_tokenizer = candidate_payload.get("model", reference_payload.get("model"))
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
    result = {"aggregate": aggregate, "prompts": prompt_summaries}

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
