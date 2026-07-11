#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay native and vLLM FA4 kernels with identical YOCO Q/K/V inputs."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MIXED5_PROMPTS = {
    "short_hello": "Hello,",
    "short_fact": "The capital of France is",
    "medium_english": (
        "Harry Potter and the Philosopher's Stone is a fantasy novel written "
        "by J.K. Rowling and the first book in the Harry Potter series. The "
        "story follows an orphaned boy who learns on his eleventh birthday "
        "that he is a wizard, then leaves his ordinary life behind to attend "
        "Hogwarts School of Witchcraft and Wizardry."
    ),
    "short_zh": "请用三句话介绍一下你自己。",
    "long_zh": (
        "在一个多语言模型的评测任务中，我们希望同时观察短问题、事实补全、长段落续写和中文对话对模型输出分布的影响。"
        "请注意，这段输入故意包含较长的上下文、多个并列要求以及一些容易让模型在推理时改变语气的提示。"
        "评测时不要只看生成文本是否通顺，还要比较下一 token 的完整概率分布，"
        "因为很小的数值差异可能会改变 top-k 排序，"
        "尤其是在多个候选 token 概率接近的时候。现在，请继续这段说明："
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Converted HF model")
    parser.add_argument("--native-checkpoint", required=True)
    parser.add_argument("--llm-train-dir", default="/root/code2/llm-train")
    parser.add_argument("--prompt-name", default="long_zh")
    parser.add_argument("--native-quant-mode", default="mxfp8")
    parser.add_argument("--native-quant-block-size", type=int, default=128)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def tensor_diff(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float()
    cand = candidate.float()
    diff = ref - cand
    denom = torch.linalg.vector_norm(ref).clamp_min(1e-12)
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rel_l2": float((torch.linalg.vector_norm(diff) / denom).item()),
        "last_token_max_abs": float(diff[-1].abs().max().item()),
        "last_token_mean_abs": float(diff[-1].abs().mean().item()),
    }


def error_payload(exc: Exception) -> dict[str, str]:
    return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    args = parse_args()
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoTokenizer

    llm_dir = Path(args.llm_train_dir).resolve() / "llm"
    sys.path.insert(0, str(llm_dir))
    from arch.model import Model, ModelArgs
    from flash_attn.cute import flash_attn_varlen_func as native_fa4

    from vllm.vllm_flash_attn import flash_attn_varlen_func as vllm_fa4
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd as vendored_fa4

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(torch.distributed.get_rank())
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda")
    init_device_mesh("cuda", mesh_shape=(dist.get_world_size(),), mesh_dim_names=["dp"])

    metadata_path = Path(args.native_checkpoint) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_args = ModelArgs()
    for key, value in metadata["modelargs"].items():
        setattr(model_args, key, value)
    model_args.quant_mode = args.native_quant_mode
    model_args.quant_block_size = args.native_quant_block_size
    model_args.use_cute = True
    # This is a single-rank inference replay. Training overlap injects
    # max_seqlen kwargs into the ring wrapper and is unrelated to FA4 parity.
    model_args.moe_fwd_bwd_overlap = False

    default_device = torch.get_default_device()
    default_dtype = torch.get_default_dtype()
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)
    model = Model(model_args)
    torch.set_default_device(default_device)
    torch.set_default_dtype(default_dtype)
    state = torch.load(
        Path(args.native_checkpoint) / "model_state_rank_0.pth",
        map_location={"cuda:0": f"cuda:{local_rank}"},
        mmap=True,
    )
    model.load_state_dict(
        {k: v for k, v in state.items() if not k.startswith("moe_loss.")}
    )
    model.eval()
    print("[fa4-replay] native model loaded", flush=True)

    if args.prompt_name not in MIXED5_PROMPTS:
        raise ValueError(
            f"Unknown prompt {args.prompt_name!r}; choices={sorted(MIXED5_PROMPTS)}"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    token_ids = tokenizer.encode(
        MIXED5_PROMPTS[args.prompt_name], add_special_tokens=False
    )
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device)
    seqlen = len(token_ids)
    cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    positions = torch.arange(seqlen, dtype=torch.int32, device=device)
    context = {
        "cu_seqlens_q": cu_seqlens,
        "cu_seqlens_k": cu_seqlens,
        "max_seqlen_q": seqlen,
        "max_seqlen_k": seqlen,
        "positions": positions,
    }

    replay_rows: list[dict[str, Any]] = []
    call_index = 0
    self_layers = model_args.n_layers - model_args.yoco_cross_layers
    self_calls = self_layers * model_args.universal_loop

    def layer_label(index: int) -> dict[str, int | str]:
        if index < self_calls:
            return {
                "attention_type": "self",
                "layer": index % self_layers,
                "universal_loop": index // self_layers,
            }
        return {
            "attention_type": "cross",
            "layer": self_layers + index - self_calls,
            "universal_loop": model_args.universal_loop,
        }

    def replay_wrapper(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *pos: Any, **kw: Any
    ):
        nonlocal call_index
        reference_out, reference_lse = native_fa4(q, k, v, *pos, **kw, return_lse=True)

        bound_cuq = kw.get("cu_seqlens_q")
        bound_cuk = kw.get("cu_seqlens_k")
        max_q = int(kw.get("max_seqlen_q"))
        max_k = int(kw.get("max_seqlen_k"))
        scale = kw.get("softmax_scale")
        causal = bool(kw.get("causal", False))
        native_window = tuple(kw.get("window_size", (None, None)))
        vllm_window = [(-1 if x is None else int(x)) for x in native_window]
        deterministic = bool(kw.get("deterministic", False))

        row: dict[str, Any] = {
            "call_index": call_index,
            **layer_label(call_index),
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "dtype": str(q.dtype),
            "max_seqlen_q": max_q,
            "max_seqlen_k": max_k,
            "softmax_scale": (
                float(scale) if scale is not None else 1.0 / math.sqrt(q.shape[-1])
            ),
            "causal": causal,
            "native_window_size": list(native_window),
        }

        # KV-sharing fast prefill runs later cross layers only for the logits
        # query while reading the complete shared K/V cache. Compare that
        # shape against the last row of full-query attention.
        try:
            one_cu_q = torch.tensor([0, 1], dtype=torch.int32, device=q.device)
            one_query_out = native_fa4(
                q[-1:],
                k,
                v,
                cu_seqlens_q=one_cu_q,
                cu_seqlens_k=bound_cuk,
                max_seqlen_q=1,
                max_seqlen_k=max_k,
                softmax_scale=scale,
                causal=causal,
                window_size=native_window,
                deterministic=deterministic,
            )
            if isinstance(one_query_out, tuple):
                one_query_out = one_query_out[0]
            row["native_last_query_vs_full"] = tensor_diff(
                reference_out[-1:], one_query_out
            )
        except Exception as exc:
            row["native_last_query_vs_full"] = error_payload(exc)

        for splits in (0, 1):
            name = f"vllm_public_splits_{splits}"
            try:
                candidate, candidate_lse = vllm_fa4(
                    q=q,
                    k=k,
                    v=v,
                    max_seqlen_q=max_q,
                    cu_seqlens_q=bound_cuq,
                    max_seqlen_k=max_k,
                    cu_seqlens_k=bound_cuk,
                    softmax_scale=scale,
                    causal=causal,
                    window_size=vllm_window,
                    deterministic=deterministic,
                    return_softmax_lse=True,
                    num_splits=splits,
                    fa_version=4,
                )
                row[name] = tensor_diff(reference_out, candidate)
                row[name]["lse"] = tensor_diff(reference_lse.T, candidate_lse.T)
            except Exception as exc:
                row[name] = error_payload(exc)

        for pack_gqa in (None, False, True):
            name = f"vllm_vendored_pack_gqa_{pack_gqa}"
            try:
                candidate, candidate_lse = vendored_fa4(
                    q,
                    k,
                    v,
                    cu_seqlens_q=bound_cuq,
                    cu_seqlens_k=bound_cuk,
                    max_seqlen_q=max_q,
                    max_seqlen_k=max_k,
                    softmax_scale=scale,
                    causal=causal,
                    window_size_left=None if vllm_window[0] < 0 else vllm_window[0],
                    window_size_right=None if vllm_window[1] < 0 else vllm_window[1],
                    num_splits=1,
                    pack_gqa=pack_gqa,
                    return_lse=True,
                )
                row[name] = tensor_diff(reference_out, candidate)
                row[name]["lse"] = tensor_diff(reference_lse.T, candidate_lse.T)
            except Exception as exc:
                row[name] = error_payload(exc)

        replay_rows.append(row)
        best = row["vllm_public_splits_1"]
        print(
            f"[fa4-replay] call={call_index} {row['attention_type']} "
            f"layer={row['layer']} loop={row['universal_loop']} "
            f"max_abs={best.get('max_abs', best.get('error'))}",
            flush=True,
        )
        call_index += 1
        return reference_out

    module_names = (
        "nnscaler.customized_ops.ring_attention.sliding_window_attn",
        "nnscaler.customized_ops.ring_attention.ring_attn_varlen",
        "nnscaler.customized_ops.ring_attention.zigzag_allgather_attn_varlen",
    )
    patched: list[tuple[Any, Any]] = []
    for module_name in module_names:
        module = importlib.import_module(module_name)
        if hasattr(module, "flash_attn_cute_varlen_func"):
            patched.append((module, module.flash_attn_cute_varlen_func))
            module.flash_attn_cute_varlen_func = replay_wrapper

    try:
        with torch.inference_mode():
            model(tokens, context=context, last_hidden_only=True)
    finally:
        for module, original in patched:
            module.flash_attn_cute_varlen_func = original

    metric_names = [
        "native_last_query_vs_full",
        "vllm_public_splits_0",
        "vllm_public_splits_1",
        "vllm_vendored_pack_gqa_None",
        "vllm_vendored_pack_gqa_False",
        "vllm_vendored_pack_gqa_True",
    ]
    aggregate: dict[str, Any] = {"num_attention_calls": len(replay_rows)}
    for name in metric_names:
        valid = [row[name] for row in replay_rows if "error" not in row[name]]
        aggregate[name] = {
            "max_abs": max(x["max_abs"] for x in valid) if valid else None,
            "mean_rel_l2": (
                sum(x["rel_l2"] for x in valid) / len(valid) if valid else None
            ),
            "lse_max_abs": max(x["lse"]["max_abs"] for x in valid if "lse" in x)
            if any("lse" in x for x in valid)
            else None,
            "lse_mean_rel_l2": sum(x["lse"]["rel_l2"] for x in valid if "lse" in x)
            / sum("lse" in x for x in valid)
            if any("lse" in x for x in valid)
            else None,
            "valid_calls": len(valid),
        }

    payload = {
        "prompt_name": args.prompt_name,
        "prompt_tokens": seqlen,
        "native_fa4_module": native_fa4.__module__,
        "vllm_fa4_module": vllm_fa4.__module__,
        "aggregate": aggregate,
        "calls": replay_rows,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)
    print(f"[fa4-replay] saved {out_path}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
