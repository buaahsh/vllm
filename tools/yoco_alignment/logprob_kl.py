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

        if name == "native":
            sub.add_argument("--native-checkpoint", required=True)
            sub.add_argument("--llm-train-dir",
                             default="/data/users/shaohanh/llm-train")
            sub.add_argument("--native-dtype", choices=("bfloat16",),
                             default="bfloat16")
        elif name == "hf":
            sub.add_argument("--device", default="cuda:0")
            sub.add_argument("--dtype", choices=("bfloat16", "float16",
                                                 "float32"),
                             default="bfloat16")
            sub.add_argument("--attn-implementation")
        else:
            sub.add_argument("--tensor-parallel-size", type=int, default=1)
            sub.add_argument("--gpu-memory-utilization", type=float, default=0.9)
            sub.add_argument("--kv-sharing-fast-prefill", action="store_true")
            sub.add_argument("--enforce-eager", action="store_true")
            sub.add_argument("--quantization", default=None)
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
    modelargs.use_cute = False

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
        hidden, _, _ = model(prefill_tokens, context=context, last_hidden_only=True)
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


def run_vllm(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    tokenizer, records = _prompt_records(
        args.model, args.prompt_suite, args.prompt_index, args.prompt_limit
    )
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
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
    if args.moe_backend:
        llm_kwargs["moe_backend"] = args.moe_backend

    llm = LLM(**llm_kwargs)
    params = SamplingParams(
        temperature=0.0, max_tokens=1, logprobs=args.max_logprobs, seed=args.seed
    )
    outputs = llm.generate(
        [
            {"prompt_token_ids": record["prompt_token_ids"],
             "prompt": record["prompt_text"]}
            for record in records
        ],
        sampling_params=params,
        use_tqdm=False,
    )
    by_prompt = {output.prompt: output for output in outputs}
    vocab_size = int(llm.llm_engine.model_config.get_vocab_size())
    results = []
    for record in records:
        output = by_prompt[record["prompt_text"]].outputs[0]
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
