#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay YOCO layer0 attention o_proj and complete MLP on identical inputs."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Converted HF checkpoint dir")
    parser.add_argument("--native-attn-dump", required=True)
    parser.add_argument("--vllm-attn-dump", required=True)
    parser.add_argument("--native-mlp-dump", required=True)
    parser.add_argument("--vllm-mlp-dump", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--occurrence", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--llm-train-dir", default="/root/code2/llm-train")
    parser.add_argument("--quant-block-size", type=int, default=128)
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    return value


def load_pt(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def record_tensor(
    dump: dict[str, Any],
    names: list[str],
    *,
    occurrence: int = 0,
) -> torch.Tensor:
    hits: list[torch.Tensor] = []
    for record in dump["records"]:
        if record.get("name") in names and "tensor" in record:
            hits.append(record["tensor"])
    if occurrence >= len(hits):
        raise KeyError(f"Cannot find occurrence {occurrence} for names={names}")
    return hits[occurrence]


def record_optional(
    dump: dict[str, Any],
    names: list[str],
    *,
    occurrence: int = 0,
) -> torch.Tensor | None:
    try:
        return record_tensor(dump, names, occurrence=occurrence)
    except KeyError:
        return None


def stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a_f = a.float()
    b_f = b.float()
    diff = a_f - b_f
    abs_diff = diff.abs()
    denom = torch.linalg.vector_norm(b_f).clamp_min(1e-12)
    return {
        "max_abs": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "mean_abs": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
        "rel_l2": float((torch.linalg.vector_norm(diff) / denom).item()),
    }


def load_model_tensor(model_dir: str, name: str, device: torch.device) -> torch.Tensor:
    model_path = Path(model_dir)
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    shard_name = index["weight_map"][name]
    shard = load_file(str(model_path / shard_name), device=str(device))
    return shard[name]


def load_layer_weights(
    model_dir: str,
    layer: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer}"
    names = {
        "o_proj": f"{prefix}.self_attn.o_proj.weight",
        "mlp_gate": f"{prefix}.mlp.gate.weight",
        "shared_gate": f"{prefix}.mlp.shared_gate.weight",
        "shared_gate_up": f"{prefix}.mlp.shared_experts.gate_up_proj.weight",
        "shared_down": f"{prefix}.mlp.shared_experts.down_proj.weight",
        "w13": f"{prefix}.mlp.experts.w13_weight",
        "w2": f"{prefix}.mlp.experts.w2_weight",
    }
    weights = {
        key: load_model_tensor(model_dir, name, device).contiguous()
        for key, name in names.items()
    }
    num_experts = int(weights["mlp_gate"].shape[0])
    hidden = int(weights["mlp_gate"].shape[1])
    ffn = int(weights["w2"].shape[1])
    weights["w13"] = weights["w13"].view(num_experts, 2 * ffn, hidden).contiguous()
    weights["w2"] = weights["w2"].view(num_experts, hidden, ffn).contiguous()
    return weights


def add_llm_train_path(llm_train_dir: str) -> None:
    llm_path = str(Path(llm_train_dir) / "llm")
    if llm_path not in sys.path:
        sys.path.insert(0, llm_path)


def native_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    block: int,
    llm_train_dir: str,
) -> torch.Tensor:
    add_llm_train_path(llm_train_dir)
    import deep_gemm
    from kernel.quant import per_block_cast_to_fp8, per_token_cast_to_fp8

    x2d = x.reshape(-1, x.shape[-1]).contiguous().to(torch.bfloat16)
    weight = weight.contiguous().to(torch.bfloat16)
    xq, xs = per_token_cast_to_fp8(x2d, gran_k=block)
    wq, ws = per_block_cast_to_fp8(weight, gran_k=block)
    out = torch.empty((xq.shape[0], wq.shape[0]), device=xq.device, dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt(
        (xq, xs),
        (wq, ws),
        out,
        recipe_a=(1, block),
        recipe_b=(block, block),
    )
    return out.view(*x.shape[:-1], weight.shape[0])


def native_bf16_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(x.to(torch.bfloat16), weight.to(torch.bfloat16))


def vllm_fp8_linear_raw(
    x: torch.Tensor,
    weight: torch.Tensor,
    block: int,
) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )
    from vllm.utils.deep_gemm import fp8_gemm_nt, per_block_cast_to_fp8

    x2d = x.reshape(-1, x.shape[-1]).contiguous().to(torch.bfloat16)
    weight = weight.contiguous().to(torch.bfloat16)
    xq, xs = per_token_group_quant_fp8(
        x2d,
        group_size=block,
        eps=1e-4,
        column_major_scales=False,
        use_ue8m0=True,
    )
    wq, ws = per_block_cast_to_fp8(weight, [block, block], use_ue8m0=True)
    out = torch.empty((xq.shape[0], wq.shape[0]), device=xq.device, dtype=torch.bfloat16)
    fp8_gemm_nt(
        (xq, xs),
        (wq, ws),
        out,
        recipe_a=(1, block),
        recipe_b=(block, block),
        is_deep_gemm_e8m0_used=True,
    )
    return out.view(*x.shape[:-1], weight.shape[0])


def vllm_fp8_linear_packed(
    x: torch.Tensor,
    weight: torch.Tensor,
    block: int,
) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
        per_token_group_quant_fp8_packed_for_deepgemm,
    )
    from vllm.utils.deep_gemm import fp8_gemm_nt, per_block_cast_to_fp8

    x2d = x.reshape(-1, x.shape[-1]).contiguous().to(torch.bfloat16)
    weight = weight.contiguous().to(torch.bfloat16)
    xq, xs = per_token_group_quant_fp8_packed_for_deepgemm(
        x2d,
        group_size=block,
        use_ue8m0=True,
    )
    wq, ws = per_block_cast_to_fp8(weight, [block, block], use_ue8m0=True)
    wq, ws = deepgemm_post_process_fp8_weight_block(
        wq=wq,
        ws=ws,
        quant_block_shape=(block, block),
        use_e8m0=True,
    )
    out = torch.empty((xq.shape[0], wq.shape[0]), device=xq.device, dtype=torch.bfloat16)
    fp8_gemm_nt(
        (xq, xs),
        (wq, ws),
        out,
        is_deep_gemm_e8m0_used=True,
    )
    return out.view(*x.shape[:-1], weight.shape[0])


def quantize_grouped_weight_native(
    weight: torch.Tensor,
    block: int,
    llm_train_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    add_llm_train_path(llm_train_dir)
    from kernel.quant import per_block_cast_to_fp8

    return per_block_cast_to_fp8(weight.contiguous().to(torch.bfloat16), gran_k=block)


def quantize_grouped_weight_vllm(
    weight: torch.Tensor,
    block: int,
    *,
    packed: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    weight = weight.contiguous().to(torch.bfloat16)
    q_chunks = []
    s_chunks = []
    for expert in range(weight.shape[0]):
        q, s = per_block_cast_to_fp8(
            weight[expert],
            [block, block],
            use_ue8m0=True,
        )
        q_chunks.append(q)
        s_chunks.append(s)
    wq = torch.stack(q_chunks, dim=0).contiguous()
    ws = torch.stack(s_chunks, dim=0).contiguous()
    if not packed:
        return wq, ws
    return deepgemm_post_process_fp8_weight_block(
        wq=wq,
        ws=ws,
        quant_block_shape=(block, block),
        use_e8m0=True,
    )


def topk_routing(
    logits: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_scores = F.softmax(logits, dim=-1, dtype=torch.float32)
    scores, top_indices = torch.topk(gate_scores, k=top_k, dim=-1)
    probs = scores / scores.sum(dim=-1, keepdim=True)
    routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs.to(logits.dtype))
    routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter(1, top_indices, True)
    return routing_probs, routing_map, gate_scores, top_indices, probs


def build_grouped_rows(
    x: torch.Tensor,
    routing_probs: torch.Tensor,
    routing_map: torch.Tensor,
    *,
    align_size: int,
) -> dict[str, torch.Tensor]:
    rows = []
    probs = []
    tokens = []
    valid = []
    counts = []
    target_counts = []
    num_experts = routing_map.shape[1]
    for expert in range(num_experts):
        token_idx = torch.nonzero(routing_map[:, expert], as_tuple=False).flatten()
        count = int(token_idx.numel())
        target = ((count + align_size - 1) // align_size) * align_size if count else 0
        counts.append(count)
        target_counts.append(target)
        if target == 0:
            continue
        expert_rows = x.new_zeros((target, x.shape[-1]))
        expert_probs = torch.zeros((target,), device=x.device, dtype=routing_probs.dtype)
        expert_tokens = torch.zeros((target,), device=x.device, dtype=torch.long)
        expert_valid = torch.zeros((target,), device=x.device, dtype=torch.bool)
        if count:
            expert_rows[:count] = x[token_idx]
            expert_probs[:count] = routing_probs[token_idx, expert]
            expert_tokens[:count] = token_idx
            expert_valid[:count] = True
        rows.append(expert_rows)
        probs.append(expert_probs)
        tokens.append(expert_tokens)
        valid.append(expert_valid)
    if rows:
        sorted_x = torch.cat(rows, dim=0).contiguous()
        row_prob = torch.cat(probs, dim=0).contiguous().float()
        row_token = torch.cat(tokens, dim=0).contiguous().long()
        row_valid = torch.cat(valid, dim=0).contiguous().bool()
    else:
        sorted_x = x.new_empty((0, x.shape[-1]))
        row_prob = torch.empty((0,), device=x.device, dtype=torch.float32)
        row_token = torch.empty((0,), device=x.device, dtype=torch.long)
        row_valid = torch.empty((0,), device=x.device, dtype=torch.bool)
    cnt = torch.tensor(counts, device=x.device, dtype=torch.int32)
    target_cnt = torch.tensor(target_counts, device=x.device, dtype=torch.int32)
    cnt_tma = torch.cumsum(target_cnt, dim=0).to(torch.int32)
    return {
        "x": sorted_x,
        "row_prob": row_prob,
        "row_token": row_token,
        "row_valid": row_valid,
        "cnt": cnt,
        "target_cnt": target_cnt,
        "cnt_tma": cnt_tma,
    }


def swiglu_from_gate_up(
    gate: torch.Tensor,
    up: torch.Tensor,
    limit: float,
) -> torch.Tensor:
    gate_f = gate.float().clamp(max=limit)
    up_f = up.float().clamp(min=-limit, max=limit)
    return (F.silu(gate_f) * up_f).to(torch.bfloat16)


def vllm_swiglu_from_packed(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    d = gate_up.shape[-1] // 2
    out = torch.empty(gate_up.shape[:-1] + (d,), dtype=gate_up.dtype, device=gate_up.device)
    if hasattr(torch.ops._C, "silu_and_mul_with_clamp"):
        torch.ops._C.silu_and_mul_with_clamp(out, gate_up, float(limit))
        return out
    gate, up = torch.chunk(gate_up, 2, dim=-1)
    return swiglu_from_gate_up(gate, up, limit)


def reduce_rows(
    y: torch.Tensor,
    row_token: torch.Tensor,
    row_valid: torch.Tensor,
    num_tokens: int,
    row_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    out = torch.zeros((num_tokens, y.shape[-1]), device=y.device, dtype=y.dtype)
    values = y[row_valid]
    if row_weight is not None:
        values = values * row_weight[row_valid].to(y.dtype).unsqueeze(-1)
    out.index_add_(0, row_token[row_valid], values)
    return out


def native_routed_moe(
    grouped: dict[str, torch.Tensor],
    w13: torch.Tensor,
    w2: torch.Tensor,
    block: int,
    llm_train_dir: str,
    swiglu_limit: float,
    num_tokens: int,
) -> dict[str, torch.Tensor]:
    add_llm_train_path(llm_train_dir)
    import deep_gemm
    from kernel.moe_ffn import fused_silu
    from kernel.quant import per_token_cast_to_fp8

    x = grouped["x"].to(torch.bfloat16).contiguous()
    cnt_tma = grouped["cnt_tma"].to(torch.int32).contiguous()
    rw = grouped["row_prob"].float().contiguous()
    w13q, w13s = quantize_grouped_weight_native(w13, block, llm_train_dir)
    xq, xs = per_token_cast_to_fp8(x, gran_k=block)
    y13 = torch.empty((x.shape[0], w13q.shape[1]), device=x.device, dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (xq, xs),
        (w13q, w13s),
        y13,
        cnt_tma,
        disable_ue8m0_cast=False,
        use_psum_layout=True,
        recipe_a=(1, block),
        recipe_b=(block, block),
    )
    x2 = fused_silu(y13, rw, swiglu_limit=swiglu_limit)
    x2q, x2s = per_token_cast_to_fp8(x2, gran_k=block)
    w2q, w2s = quantize_grouped_weight_native(w2, block, llm_train_dir)
    y2 = torch.empty((x.shape[0], w2q.shape[1]), device=x.device, dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (x2q, x2s),
        (w2q, w2s),
        y2,
        cnt_tma,
        disable_ue8m0_cast=False,
        use_psum_layout=True,
        recipe_a=(1, block),
        recipe_b=(block, block),
    )
    return {
        "y13": y13,
        "x2": x2,
        "y2": y2,
        "out": reduce_rows(y2, grouped["row_token"], grouped["row_valid"], num_tokens),
    }


def vllm_activation_quant(
    y13: torch.Tensor,
    row_weights: torch.Tensor | None,
    block: int,
    swiglu_limit: float,
    *,
    packed: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
        silu_mul_quant_fp8_packed_triton,
    )

    if packed:
        return silu_mul_quant_fp8_packed_triton(
            input=y13.contiguous(),
            group_size=block,
            clamp_limit=swiglu_limit,
            row_weights=row_weights.contiguous() if row_weights is not None else None,
        )
    gate, up = torch.chunk(y13.float(), 2, dim=-1)
    x2 = (F.silu(gate.clamp(max=swiglu_limit)) * up.clamp(-swiglu_limit, swiglu_limit))
    x2 = x2.to(torch.bfloat16)
    if row_weights is not None:
        x2 = x2 * row_weights.to(torch.bfloat16).unsqueeze(-1)
    x2 = x2.contiguous()
    return per_token_group_quant_fp8(
        x2,
        group_size=block,
        eps=1e-4,
        column_major_scales=False,
        use_ue8m0=True,
    )


def vllm_routed_moe(
    grouped: dict[str, torch.Tensor],
    w13: torch.Tensor,
    w2: torch.Tensor,
    block: int,
    swiglu_limit: float,
    num_tokens: int,
    *,
    packed: bool,
    probs_before_w2: bool = False,
) -> dict[str, torch.Tensor]:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
        per_token_group_quant_fp8_packed_for_deepgemm,
    )
    from vllm.utils.deep_gemm import m_grouped_fp8_gemm_nt_contiguous

    x = grouped["x"].to(torch.bfloat16).contiguous()
    cnt_tma = grouped["cnt_tma"].to(torch.int32).contiguous()
    rw = grouped["row_prob"].float().contiguous()
    if packed:
        xq, xs = per_token_group_quant_fp8_packed_for_deepgemm(
            x,
            group_size=block,
            use_ue8m0=True,
        )
        grouped_kwargs: dict[str, Any] = {"use_psum_layout": True}
    else:
        xq, xs = per_token_group_quant_fp8(
            x,
            group_size=block,
            eps=1e-4,
            column_major_scales=False,
            use_ue8m0=True,
        )
        grouped_kwargs = {
            "use_psum_layout": True,
            "recipe_a": (1, block),
            "recipe_b": (block, block),
        }
    w13q, w13s = quantize_grouped_weight_vllm(w13, block, packed=packed)
    y13 = torch.empty((x.shape[0], w13q.shape[1]), device=x.device, dtype=torch.bfloat16)
    m_grouped_fp8_gemm_nt_contiguous(
        (xq, xs),
        (w13q, w13s),
        y13,
        cnt_tma,
        **grouped_kwargs,
    )
    x2q, x2s = vllm_activation_quant(
        y13,
        rw if probs_before_w2 else None,
        block,
        swiglu_limit,
        packed=packed,
    )
    w2q, w2s = quantize_grouped_weight_vllm(w2, block, packed=packed)
    y2 = torch.empty((x.shape[0], w2q.shape[1]), device=x.device, dtype=torch.bfloat16)
    m_grouped_fp8_gemm_nt_contiguous(
        (x2q, x2s),
        (w2q, w2s),
        y2,
        cnt_tma,
        **grouped_kwargs,
    )
    return {
        "y13": y13,
        "y2": y2,
        "out": reduce_rows(
            y2,
            grouped["row_token"],
            grouped["row_valid"],
            num_tokens,
            row_weight=None if probs_before_w2 else rw,
        ),
    }


def replay_attention(
    weights: dict[str, torch.Tensor],
    native_dump: dict[str, Any],
    vllm_dump: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    occ = args.occurrence
    n_in = record_tensor(
        native_dump,
        [f"layers.{args.layer}.self_attn.o_proj.input"],
        occurrence=occ,
    ).to(device)
    n_out = record_tensor(
        native_dump,
        [f"layers.{args.layer}.self_attn.o_proj"],
        occurrence=occ,
    ).to(device)
    v_in = record_tensor(
        vllm_dump,
        [f"model.layers.{args.layer}.self_attn.o_proj.input"],
        occurrence=occ,
    ).to(device)
    v_out = record_tensor(
        vllm_dump,
        [f"model.layers.{args.layer}.self_attn.o_proj.0"],
        occurrence=occ,
    ).to(device)
    weight = weights["o_proj"]

    summary: dict[str, Any] = {
        "actual_dump": {
            "input_vllm_vs_native": stats(v_in, n_in),
            "output_vllm_vs_native": stats(v_out, n_out),
        },
        "same_input": {},
    }
    for label, x in (("native_input", n_in), ("vllm_input", v_in)):
        ref = native_bf16_linear(x, weight)
        native = native_fp8_linear(x, weight, args.quant_block_size, args.llm_train_dir)
        vllm_raw = vllm_fp8_linear_raw(x, weight, args.quant_block_size)
        vllm_packed = vllm_fp8_linear_packed(x, weight, args.quant_block_size)
        item: dict[str, Any] = {
            "native_mxfp8_vs_bf16": stats(native, ref),
            "vllm_raw_vs_bf16": stats(vllm_raw, ref),
            "vllm_packed_vs_bf16": stats(vllm_packed, ref),
            "vllm_raw_vs_native_mxfp8": stats(vllm_raw, native),
            "vllm_packed_vs_native_mxfp8": stats(vllm_packed, native),
            "vllm_packed_vs_vllm_raw": stats(vllm_packed, vllm_raw),
        }
        if label == "native_input":
            item["native_replay_vs_native_dump"] = stats(native, n_out)
        else:
            item["vllm_raw_vs_vllm_dump"] = stats(vllm_raw, v_out)
            item["vllm_packed_vs_vllm_dump"] = stats(vllm_packed, v_out)
        summary["same_input"][label] = item
        del ref, native, vllm_raw, vllm_packed
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
    return summary


def replay_shared_expert(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    args: argparse.Namespace,
    *,
    variant: str,
) -> dict[str, torch.Tensor]:
    gate_w, up_w = torch.chunk(weights["shared_gate_up"], 2, dim=0)
    if variant == "native":
        gate = native_fp8_linear(x, gate_w, args.quant_block_size, args.llm_train_dir)
        up = native_fp8_linear(x, up_w, args.quant_block_size, args.llm_train_dir)
        act = swiglu_from_gate_up(gate, up, args.swiglu_limit)
        down = native_fp8_linear(
            act,
            weights["shared_down"],
            args.quant_block_size,
            args.llm_train_dir,
        )
    elif variant == "vllm_raw":
        gate_up = vllm_fp8_linear_raw(x, weights["shared_gate_up"], args.quant_block_size)
        gate, up = torch.chunk(gate_up, 2, dim=-1)
        act = vllm_swiglu_from_packed(gate_up, args.swiglu_limit)
        down = vllm_fp8_linear_raw(act, weights["shared_down"], args.quant_block_size)
    elif variant == "vllm_packed":
        gate_up = vllm_fp8_linear_packed(x, weights["shared_gate_up"], args.quant_block_size)
        gate, up = torch.chunk(gate_up, 2, dim=-1)
        act = vllm_swiglu_from_packed(gate_up, args.swiglu_limit)
        down = vllm_fp8_linear_packed(act, weights["shared_down"], args.quant_block_size)
    elif variant == "vllm_packed_fp32_act":
        gate_up = vllm_fp8_linear_packed(x, weights["shared_gate_up"], args.quant_block_size)
        gate, up = torch.chunk(gate_up, 2, dim=-1)
        act = swiglu_from_gate_up(gate, up, args.swiglu_limit)
        down = vllm_fp8_linear_packed(act, weights["shared_down"], args.quant_block_size)
    else:
        raise ValueError(variant)
    shared_gate = native_bf16_linear(x, weights["shared_gate"])
    shared_scaled = torch.sigmoid(shared_gate) * down
    return {
        "gate": gate,
        "up": up,
        "act": act,
        "down": down,
        "shared_gate": shared_gate,
        "scaled": shared_scaled,
    }


def replay_mlp_once(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    logits = F.linear(x.float(), weights["mlp_gate"].float())
    routing_probs, routing_map, gate_scores, topk_ids, topk_weights = topk_routing(
        logits,
        top_k=8,
    )
    grouped = build_grouped_rows(
        x,
        routing_probs,
        routing_map,
        align_size=args.quant_block_size,
    )
    routed_native = native_routed_moe(
        grouped,
        weights["w13"],
        weights["w2"],
        args.quant_block_size,
        args.llm_train_dir,
        args.swiglu_limit,
        x.shape[0],
    )
    routed_vllm_raw = vllm_routed_moe(
        grouped,
        weights["w13"],
        weights["w2"],
        args.quant_block_size,
        args.swiglu_limit,
        x.shape[0],
        packed=False,
    )
    routed_vllm_packed = vllm_routed_moe(
        grouped,
        weights["w13"],
        weights["w2"],
        args.quant_block_size,
        args.swiglu_limit,
        x.shape[0],
        packed=True,
    )
    routed_vllm_packed_pre_w2 = vllm_routed_moe(
        grouped,
        weights["w13"],
        weights["w2"],
        args.quant_block_size,
        args.swiglu_limit,
        x.shape[0],
        packed=True,
        probs_before_w2=True,
    )
    shared_native = replay_shared_expert(x, weights, args, variant="native")
    shared_vllm_raw = replay_shared_expert(x, weights, args, variant="vllm_raw")
    shared_vllm_packed = replay_shared_expert(x, weights, args, variant="vllm_packed")
    shared_vllm_packed_fp32_act = replay_shared_expert(
        x, weights, args, variant="vllm_packed_fp32_act"
    )

    final_native = routed_native["out"] + shared_native["scaled"]
    final_vllm_raw = routed_vllm_raw["out"] + shared_vllm_raw["scaled"]
    final_vllm_packed = routed_vllm_packed["out"] + shared_vllm_packed["scaled"]
    final_vllm_packed_pre_w2 = (
        routed_vllm_packed_pre_w2["out"] + shared_vllm_packed["scaled"]
    )
    final_vllm_aligned = (
        routed_vllm_packed_pre_w2["out"] + shared_vllm_packed_fp32_act["scaled"]
    )

    return {
        "logits": logits,
        "routing_probs": routing_probs,
        "routing_map": routing_map,
        "gate_scores": gate_scores,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "grouped_counts": grouped["cnt"],
        "routed_native": routed_native,
        "routed_vllm_raw": routed_vllm_raw,
        "routed_vllm_packed": routed_vllm_packed,
        "routed_vllm_packed_pre_w2": routed_vllm_packed_pre_w2,
        "shared_native": shared_native,
        "shared_vllm_raw": shared_vllm_raw,
        "shared_vllm_packed": shared_vllm_packed,
        "shared_vllm_packed_fp32_act": shared_vllm_packed_fp32_act,
        "final_native": final_native,
        "final_vllm_raw": final_vllm_raw,
        "final_vllm_packed": final_vllm_packed,
        "final_vllm_packed_pre_w2": final_vllm_packed_pre_w2,
        "final_vllm_aligned": final_vllm_aligned,
    }


def replay_mlp(
    weights: dict[str, torch.Tensor],
    native_dump: dict[str, Any],
    vllm_dump: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    occ = args.occurrence
    n_prefix = f"layers.{args.layer}.mlp"
    v_prefix = f"model.layers.{args.layer}.mlp"
    n_in = record_tensor(native_dump, [f"{n_prefix}.input"], occurrence=occ).to(device)
    n_final = record_tensor(native_dump, [f"{n_prefix}.0"], occurrence=occ).to(device)
    n_routing_map = record_tensor(native_dump, [f"{n_prefix}.1"], occurrence=occ).to(device)
    n_gate_scores = record_tensor(native_dump, [f"{n_prefix}.2"], occurrence=occ).to(device)
    n_router = record_tensor(native_dump, [f"{n_prefix}.gate"], occurrence=occ).to(device)
    n_shared_gate = record_tensor(native_dump, [f"{n_prefix}.shared_gate"], occurrence=occ).to(device)
    n_shared_up = record_tensor(native_dump, [f"{n_prefix}.shared.up_proj"], occurrence=occ).to(device)
    n_shared_gate_proj = record_tensor(
        native_dump,
        [f"{n_prefix}.shared.gate_proj"],
        occurrence=occ,
    ).to(device)
    n_shared_act = record_tensor(
        native_dump,
        [f"{n_prefix}.shared.down_proj.input"],
        occurrence=occ,
    ).to(device)
    n_shared_down = record_tensor(
        native_dump,
        [f"{n_prefix}.shared.down_proj", f"{n_prefix}.shared"],
        occurrence=occ,
    ).to(device)

    v_in = record_tensor(vllm_dump, [f"{v_prefix}.input"], occurrence=occ).to(device)
    v_final = record_tensor(vllm_dump, [f"{v_prefix}"], occurrence=occ).to(device)
    v_router = record_tensor(vllm_dump, [f"{v_prefix}.gate.0"], occurrence=occ).to(device)
    v_routed = record_tensor(vllm_dump, [f"{v_prefix}.experts"], occurrence=occ).to(device)
    v_gate_up = record_tensor(
        vllm_dump,
        [f"{v_prefix}.shared_experts.gate_up_proj.0"],
        occurrence=occ,
    ).to(device)
    v_shared_act = record_tensor(
        vllm_dump,
        [f"{v_prefix}.shared_experts.act_fn"],
        occurrence=occ,
    ).to(device)
    v_shared_down = record_tensor(
        vllm_dump,
        [f"{v_prefix}.shared_experts.down_proj.0", f"{v_prefix}.shared_experts"],
        occurrence=occ,
    ).to(device)
    n_shared_scaled = torch.sigmoid(n_shared_gate) * n_shared_down
    n_routed_derived = n_final - n_shared_scaled
    v_shared_gate = native_bf16_linear(v_in, weights["shared_gate"])
    v_shared_scaled = torch.sigmoid(v_shared_gate) * v_shared_down
    v_final_reconstructed = v_routed + v_shared_scaled

    summary: dict[str, Any] = {
        "actual_dump": {
            "input_vllm_vs_native": stats(v_in, n_in),
            "router_logits_vllm_vs_native": stats(v_router, n_router),
            "final_vllm_vs_native": stats(v_final, n_final),
            "shared_act_vllm_vs_native": stats(v_shared_act, n_shared_act),
            "shared_down_unscaled_vllm_vs_native": stats(v_shared_down, n_shared_down),
            "shared_scaled_vllm_vs_native": stats(v_shared_scaled, n_shared_scaled),
            "routed_vllm_vs_native_derived": stats(v_routed, n_routed_derived),
            "vllm_final_reconstructed_vs_dump": stats(v_final_reconstructed, v_final),
        },
        "same_input": {},
    }

    for label, x in (("native_input", n_in), ("vllm_input", v_in)):
        replay = replay_mlp_once(x, weights, args)
        native = {
            "router_logits": replay["logits"],
            "gate_scores": replay["gate_scores"],
            "routing_map": replay["routing_map"],
            "shared_gate": replay["shared_native"]["shared_gate"],
            "shared_gate_proj": replay["shared_native"]["gate"],
            "shared_up_proj": replay["shared_native"]["up"],
            "shared_act": replay["shared_native"]["act"],
            "shared_down_unscaled": replay["shared_native"]["down"],
            "shared_scaled": replay["shared_native"]["scaled"],
            "routed": replay["routed_native"]["out"],
            "final": replay["final_native"],
        }
        vllm_raw = {
            "shared_gate": replay["shared_vllm_raw"]["shared_gate"],
            "shared_gate_proj": replay["shared_vllm_raw"]["gate"],
            "shared_up_proj": replay["shared_vllm_raw"]["up"],
            "shared_act": replay["shared_vllm_raw"]["act"],
            "shared_down_unscaled": replay["shared_vllm_raw"]["down"],
            "shared_scaled": replay["shared_vllm_raw"]["scaled"],
            "routed": replay["routed_vllm_raw"]["out"],
            "final": replay["final_vllm_raw"],
        }
        vllm_packed = {
            "shared_gate": replay["shared_vllm_packed"]["shared_gate"],
            "shared_gate_proj": replay["shared_vllm_packed"]["gate"],
            "shared_up_proj": replay["shared_vllm_packed"]["up"],
            "shared_act": replay["shared_vllm_packed"]["act"],
            "shared_down_unscaled": replay["shared_vllm_packed"]["down"],
            "shared_scaled": replay["shared_vllm_packed"]["scaled"],
            "routed": replay["routed_vllm_packed"]["out"],
            "final": replay["final_vllm_packed"],
        }
        vllm_packed_pre_w2 = {
            **vllm_packed,
            "routed": replay["routed_vllm_packed_pre_w2"]["out"],
            "final": replay["final_vllm_packed_pre_w2"],
        }
        vllm_aligned = {
            "shared_gate": replay["shared_vllm_packed_fp32_act"]["shared_gate"],
            "shared_gate_proj": replay["shared_vllm_packed_fp32_act"]["gate"],
            "shared_up_proj": replay["shared_vllm_packed_fp32_act"]["up"],
            "shared_act": replay["shared_vllm_packed_fp32_act"]["act"],
            "shared_down_unscaled": replay["shared_vllm_packed_fp32_act"]["down"],
            "shared_scaled": replay["shared_vllm_packed_fp32_act"]["scaled"],
            "routed": replay["routed_vllm_packed_pre_w2"]["out"],
            "final": replay["final_vllm_aligned"],
        }
        item: dict[str, Any] = {
            "route_counts": {
                "valid_rows": int(replay["grouped_counts"].sum().item()),
                "nonzero_experts": int((replay["grouped_counts"] > 0).sum().item()),
                "max_tokens_per_expert": int(replay["grouped_counts"].max().item()),
                "target_rows_aligned": int(
                    (((replay["grouped_counts"] + args.quant_block_size - 1)
                      // args.quant_block_size)
                     * args.quant_block_size).sum().item()
                ),
            },
            "vllm_raw_vs_native": {
                key: stats(vllm_raw[key], native[key])
                for key in vllm_raw
                if key in native
            },
            "vllm_packed_vs_native": {
                key: stats(vllm_packed[key], native[key])
                for key in vllm_packed
                if key in native
            },
            "vllm_packed_pre_w2_vs_native": {
                key: stats(vllm_packed_pre_w2[key], native[key])
                for key in vllm_packed_pre_w2
                if key in native
            },
            "vllm_aligned_vs_native": {
                key: stats(vllm_aligned[key], native[key])
                for key in vllm_aligned
                if key in native
            },
            "vllm_packed_vs_vllm_raw": {
                key: stats(vllm_packed[key], vllm_raw[key])
                for key in vllm_raw
            },
        }
        if label == "native_input":
            item["native_replay_vs_native_dump"] = {
                "router_logits": stats(native["router_logits"], n_router),
                "gate_scores": stats(native["gate_scores"], n_gate_scores),
                "routing_map_equal": bool(torch.equal(native["routing_map"], n_routing_map.bool())),
                "shared_gate": stats(native["shared_gate"], n_shared_gate),
                "shared_gate_proj": stats(native["shared_gate_proj"], n_shared_gate_proj),
                "shared_up_proj": stats(native["shared_up_proj"], n_shared_up),
                "shared_act": stats(native["shared_act"], n_shared_act),
                "shared_down_unscaled": stats(native["shared_down_unscaled"], n_shared_down),
                "final": stats(native["final"], n_final),
            }
        else:
            item["vllm_raw_vs_vllm_dump"] = {
                "router_logits": stats(native["router_logits"], v_router),
                "shared_gate_up": stats(
                    torch.cat([vllm_raw["shared_gate_proj"], vllm_raw["shared_up_proj"]], dim=-1),
                    v_gate_up,
                ),
                "shared_act": stats(vllm_raw["shared_act"], v_shared_act),
                "shared_down_unscaled": stats(vllm_raw["shared_down_unscaled"], v_shared_down),
                "routed": stats(vllm_raw["routed"], v_routed),
                "final": stats(vllm_raw["final"], v_final),
            }
            item["vllm_packed_vs_vllm_dump"] = {
                "router_logits": stats(native["router_logits"], v_router),
                "shared_gate_up": stats(
                    torch.cat([vllm_packed["shared_gate_proj"], vllm_packed["shared_up_proj"]], dim=-1),
                    v_gate_up,
                ),
                "shared_act": stats(vllm_packed["shared_act"], v_shared_act),
                "shared_down_unscaled": stats(vllm_packed["shared_down_unscaled"], v_shared_down),
                "routed": stats(vllm_packed["routed"], v_routed),
                "final": stats(vllm_packed["final"], v_final),
            }
        summary["same_input"][label] = item

        del replay, native, vllm_raw, vllm_packed, vllm_packed_pre_w2, vllm_aligned
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("DeepGEMM replay requires CUDA.")
    add_llm_train_path(args.llm_train_dir)

    weights = load_layer_weights(args.model, args.layer, device)
    native_attn = load_pt(args.native_attn_dump)
    vllm_attn = load_pt(args.vllm_attn_dump)
    native_mlp = load_pt(args.native_mlp_dump)
    vllm_mlp = load_pt(args.vllm_mlp_dump)

    summary = {
        "model": args.model,
        "layer": args.layer,
        "occurrence": args.occurrence,
        "quant_block_size": args.quant_block_size,
        "swiglu_limit": args.swiglu_limit,
        "dumps": {
            "native_attn": args.native_attn_dump,
            "vllm_attn": args.vllm_attn_dump,
            "native_mlp": args.native_mlp_dump,
            "vllm_mlp": args.vllm_mlp_dump,
        },
        "attention_o_proj": replay_attention(
            weights,
            native_attn,
            vllm_attn,
            args,
            device,
        ),
        "layer0_mlp": replay_mlp(
            weights,
            native_mlp,
            vllm_mlp,
            args,
            device,
        ),
    }

    printable = json.dumps(_jsonable(summary), indent=2, ensure_ascii=False)
    print(printable)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(printable + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
