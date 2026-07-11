#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay YOCO routed MoE W13/W2 on aligned native/vLLM dispatch rows."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Converted HF checkpoint dir")
    parser.add_argument("--native-dump", required=True, help="native_callXXX.pt")
    parser.add_argument(
        "--native-dispatch-dump",
        required=True,
        help="native_dispatch_callXXX.pt from all2all_moe.py",
    )
    parser.add_argument("--vllm-dump", required=True, help="vllm_callXXX_*.pt")
    parser.add_argument(
        "--vllm-prepare-dump",
        help="Optional vllm_prepare_callXXX_*.pt from modular prepare",
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--llm-train-dir", default="/root/code2/llm-train")
    parser.add_argument("--quant-block-size", type=int, default=128)
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument(
        "--variants",
        default="native,vllm_raw_recipe,vllm_packed",
        help="Comma-separated: native,vllm_raw_recipe,vllm_packed",
    )
    parser.add_argument("--out-json", help="Optional JSON summary path")
    return parser.parse_args()


def load_pt(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def load_model_tensor(model_dir: str, name: str, device: torch.device) -> torch.Tensor:
    model_path = Path(model_dir)
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shard_name = index["weight_map"][name]
    shard = load_file(str(model_path / shard_name), device=str(device))
    return shard[name]


def stats(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float]:
    if mask is not None:
        a = a[mask]
        b = b[mask]
    a_f = a.float()
    b_f = b.float()
    diff = a_f - b_f
    diff_abs = diff.abs()
    denom = torch.linalg.vector_norm(b_f).clamp_min(1e-12)
    return {
        "max_abs": float(diff_abs.max().item()) if diff_abs.numel() else 0.0,
        "mean_abs": float(diff_abs.mean().item()) if diff_abs.numel() else 0.0,
        "rel_l2": float((torch.linalg.vector_norm(diff) / denom).item()),
    }


def build_match_summary(native_dispatch: dict[str, Any], vllm_dump: dict[str, Any]) -> dict[str, Any]:
    nt = native_dispatch["tensors"]
    vt = vllm_dump["tensors"]

    n_valid = nt["row_valid"].bool()
    v_valid = vt.get("row_valid")
    if v_valid is None:
        v_valid = torch.zeros(vt["a1q"].shape[0], dtype=torch.bool)
        topk_ids = vt["topk_ids"].long()
        inv_perm = vt["inv_perm"].long()
        valid = topk_ids >= 0
        rows = inv_perm[valid]
        rows = rows[(rows >= 0) & (rows < v_valid.numel())]
        v_valid[rows] = True
    else:
        v_valid = v_valid.bool()

    def key_rows(tensors: dict[str, torch.Tensor], valid: torch.Tensor) -> dict[tuple[int, int], int]:
        row_token = tensors["row_token"].long()
        row_expert = tensors["row_expert"].long()
        rows = torch.nonzero(valid, as_tuple=False).flatten().tolist()
        out: dict[tuple[int, int], int] = {}
        for row in rows:
            out[(int(row_token[row]), int(row_expert[row]))] = int(row)
        return out

    n_map = key_rows(nt, n_valid)
    v_map = key_rows(vt, v_valid)
    common = sorted(set(n_map) & set(v_map))
    n_rows = torch.tensor([n_map[k] for k in common], dtype=torch.long)
    v_rows = torch.tensor([v_map[k] for k in common], dtype=torch.long)

    prob_stats = {}
    if common:
        prob_stats = stats(vt["row_prob"][v_rows], nt["row_prob"][n_rows])

    return {
        "native_valid_rows": int(n_valid.sum().item()),
        "vllm_valid_rows": int(v_valid.sum().item()),
        "matched_rows": len(common),
        "native_only_rows": len(set(n_map) - set(v_map)),
        "vllm_only_rows": len(set(v_map) - set(n_map)),
        "native_rows": n_rows,
        "vllm_rows": v_rows,
        "prob_diff": prob_stats,
    }


def element_stats(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    a_f = a.float()
    b_f = b.float()
    diff = a_f[mask] - b_f[mask]
    diff_abs = diff.abs()
    denom = torch.linalg.vector_norm(b_f[mask]).clamp_min(1e-12)
    return {
        "num_elements": int(diff.numel()),
        "max_abs": float(diff_abs.max().item()) if diff_abs.numel() else 0.0,
        "mean_abs": float(diff_abs.mean().item()) if diff_abs.numel() else 0.0,
        "rel_l2": float((torch.linalg.vector_norm(diff) / denom).item()),
    }


def compare_actual_vllm_dump(
    native: dict[str, Any],
    native_dispatch: dict[str, Any],
    vllm_dump: dict[str, Any],
    native_rows: torch.Tensor,
    vllm_rows: torch.Tensor,
    block_size: int,
) -> dict[str, Any]:
    nt = native["tensors"]
    ndt = native_dispatch["tensors"]
    vt = vllm_dump["tensors"]
    out: dict[str, Any] = {
        "row_prob": stats(vt["row_prob"][vllm_rows], ndt["row_prob"][native_rows]),
        "mm1_out_vs_native_y13": stats(
            vt["mm1_out"][vllm_rows], nt["y13"][native_rows]
        ),
        "mm2_out_vs_native_y2": stats(
            vt["mm2_out"][vllm_rows], nt["y2"][native_rows]
        ),
        "a1q_scale": stats(
            vt["a1q_scale"][vllm_rows], nt["x13_scale"][native_rows]
        ),
    }

    vq = vt["a1q"][vllm_rows].float()
    nq = nt["x13_quant"][native_rows].float()
    non_nan = ~torch.isnan(vq)
    out["a1q_payload_non_nan"] = element_stats(vq, nq, non_nan)
    out["a1q_nan_elements"] = int(torch.isnan(vq).sum().item())
    out["a1q_nan_rows"] = int(torch.isnan(vq).any(dim=1).sum().item())

    vs = vt["a1q_scale"][vllm_rows].repeat_interleave(block_size, dim=1)
    ns = nt["x13_scale"][native_rows].repeat_interleave(block_size, dim=1)
    out["a1_dequant_non_nan"] = element_stats(vq * vs, nq * ns, non_nan)
    return out


def compare_prepare_input_quant(
    native: dict[str, Any],
    vllm_dump: dict[str, Any],
    vllm_prepare_dump: dict[str, Any],
    native_rows: torch.Tensor,
    vllm_rows: torch.Tensor,
    block_size: int,
) -> dict[str, Any]:
    nt = native["tensors"]
    vt = vllm_dump["tensors"]
    pt = vllm_prepare_dump["tensors"]

    token_rows = vt["row_token"][vllm_rows].long()
    out: dict[str, Any] = {
        "prepare_call_index": int(vllm_prepare_dump.get("call_index", -1)),
        "experts_call_index": int(vllm_dump.get("call_index", -1)),
        "token_rows_min": int(token_rows.min().item()) if token_rows.numel() else -1,
        "token_rows_max": int(token_rows.max().item()) if token_rows.numel() else -1,
    }

    hidden = pt.get("hidden_states_pre_quant")
    if hidden is not None:
        out["hidden_pre_quant_vs_native_x13"] = stats(
            hidden[token_rows], nt["x13"][native_rows]
        )

    prep_topk_ids = pt.get("topk_ids_prepared")
    prep_topk_weights = pt.get("topk_weights_prepared")
    if prep_topk_ids is not None:
        out["topk_ids_prepare_vs_experts_equal"] = bool(
            torch.equal(prep_topk_ids, vt["topk_ids"])
        )
    if prep_topk_weights is not None:
        out["topk_weights_prepare_vs_experts"] = stats(
            prep_topk_weights, vt["topk_weights_original"]
        )

    a1q_unpermuted = pt.get("a1q_unpermuted")
    a1q_scale_unpermuted = pt.get("a1q_scale_unpermuted")
    if a1q_unpermuted is not None:
        p_q = a1q_unpermuted[token_rows].float()
        n_q = nt["x13_quant"][native_rows].float()
        non_nan = ~torch.isnan(p_q)
        out["a1q_unpermuted_payload_vs_native_x13_quant"] = element_stats(
            p_q, n_q, non_nan
        )
        out["a1q_unpermuted_nan_elements"] = int(torch.isnan(p_q).sum().item())
        out["a1q_unpermuted_nan_rows"] = int(torch.isnan(p_q).any(dim=1).sum().item())
        out["a1q_permuted_snapshot_vs_prepare_unpermuted"] = element_stats(
            vt["a1q"][vllm_rows].float(), p_q, non_nan
        )

    if a1q_scale_unpermuted is not None:
        p_s = a1q_scale_unpermuted[token_rows]
        n_s = nt["x13_scale"][native_rows]
        out["a1q_scale_unpermuted_vs_native_x13_scale"] = stats(p_s, n_s)
        out["a1q_scale_permuted_snapshot_vs_prepare_unpermuted"] = stats(
            vt["a1q_scale"][vllm_rows], p_s
        )

        if a1q_unpermuted is not None:
            p_q = a1q_unpermuted[token_rows].float()
            n_q = nt["x13_quant"][native_rows].float()
            p_s_full = p_s.repeat_interleave(block_size, dim=1)
            n_s_full = n_s.repeat_interleave(block_size, dim=1)
            non_nan = ~torch.isnan(p_q)
            out["a1_dequant_unpermuted_vs_native"] = element_stats(
                p_q * p_s_full, n_q * n_s_full, non_nan
            )
            if hidden is not None:
                out["vllm_input_quant_dequant_vs_hidden"] = element_stats(
                    p_q * p_s_full,
                    hidden[token_rows].float(),
                    non_nan,
                )

    return out


def reduce_valid_rows(y2: torch.Tensor, dispatch_tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    valid = dispatch_tensors["row_valid"].to(y2.device).bool()
    row_token = dispatch_tensors["row_token"].to(y2.device).long()
    num_tokens = int(dispatch_tensors["routing_map"].shape[0])
    output = torch.zeros((num_tokens, y2.shape[1]), dtype=torch.float32, device=y2.device)
    output.index_add_(0, row_token[valid], y2[valid].float())
    return output


def quantize_weight_native(weight: torch.Tensor, block_size: int, llm_train_dir: str):
    llm_path = str(Path(llm_train_dir) / "llm")
    if llm_path not in sys.path:
        sys.path.insert(0, llm_path)
    from kernel.quant import per_block_cast_to_fp8

    return per_block_cast_to_fp8(weight, gran_k=block_size)


def quantize_weight_vllm(weight: torch.Tensor, block_size: int, *, packed: bool):
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    q_chunks = []
    s_chunks = []
    for expert in range(weight.shape[0]):
        q, s = per_block_cast_to_fp8(
            weight[expert], [block_size, block_size], use_ue8m0=True
        )
        q_chunks.append(q)
        s_chunks.append(s)
    q_weight = torch.stack(q_chunks, dim=0).contiguous()
    q_scale = torch.stack(s_chunks, dim=0).contiguous()
    if not packed:
        return q_weight, q_scale
    return deepgemm_post_process_fp8_weight_block(
        q_weight,
        q_scale,
        quant_block_shape=(block_size, block_size),
        use_e8m0=True,
    )


def replay_native(
    native: dict[str, Any],
    dispatch: dict[str, Any],
    w13: torch.Tensor,
    w2: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    sys.path.insert(0, str(Path(args.llm_train_dir) / "llm"))
    import deep_gemm
    from kernel.moe_ffn import fused_silu
    from kernel.quant import per_token_cast_to_fp8

    t = native["tensors"]
    block = args.quant_block_size
    cnt_tma = t["cnt_tma"].to(device=device, dtype=torch.int32)
    rw = t["rw"].to(device=device, dtype=torch.float32)
    x13q = t["x13_quant"].to(device=device)
    x13s = t["x13_scale"].to(device=device)

    w13q, w13s = quantize_weight_native(w13, block, args.llm_train_dir)
    y13 = torch.empty((x13q.shape[0], w13q.shape[1]), device=device, dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (x13q, x13s),
        (w13q, w13s),
        y13,
        cnt_tma,
        disable_ue8m0_cast=False,
        use_psum_layout=True,
        recipe_a=(1, block),
        recipe_b=(block, block),
    )

    x2 = fused_silu(y13, rw, swiglu_limit=args.swiglu_limit)
    x2q, x2s = per_token_cast_to_fp8(x2, gran_k=block)
    del x2
    w2q, w2s = quantize_weight_native(w2, block, args.llm_train_dir)
    y2 = torch.empty((x2q.shape[0], w2q.shape[1]), device=device, dtype=torch.bfloat16)
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

    valid = dispatch["tensors"]["row_valid"].to(device).bool()
    native_y13 = t["y13"].to(device)
    native_y2 = t["y2"].to(device)
    result = {
        "y13_vs_dump_valid": stats(y13, native_y13, valid),
        "y2_vs_dump_valid": stats(y2, native_y2, valid),
        "reduced_y2_vs_dump_valid": stats(
            reduce_valid_rows(y2, dispatch["tensors"]),
            reduce_valid_rows(native_y2, dispatch["tensors"]),
        ),
    }
    del w13q, w13s, w2q, w2s, y13, y2, x2q, x2s
    return result


def vllm_activation_quant(
    y13: torch.Tensor,
    row_weights: torch.Tensor,
    block: int,
    swiglu_limit: float,
    *,
    packed: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
        per_token_group_quant_fp8_packed_for_deepgemm,
        silu_mul_quant_fp8_packed_triton,
    )

    if packed:
        return silu_mul_quant_fp8_packed_triton(
            input=y13,
            group_size=block,
            clamp_limit=swiglu_limit,
            row_weights=row_weights.contiguous(),
        )

    gate, up = torch.chunk(y13.float(), 2, dim=-1)
    gate = torch.minimum(gate, torch.tensor(swiglu_limit, device=y13.device))
    up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    x2 = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    x2.mul_(row_weights.to(torch.bfloat16).unsqueeze(-1))
    return per_token_group_quant_fp8(
        x2,
        block,
        eps=1e-4,
        column_major_scales=False,
        use_ue8m0=True,
    )


def replay_vllm_same_input(
    native: dict[str, Any],
    dispatch: dict[str, Any],
    w13: torch.Tensor,
    w2: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    *,
    packed: bool,
) -> dict[str, Any]:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
        per_token_group_quant_fp8_packed_for_deepgemm,
    )
    from vllm.utils.deep_gemm import m_grouped_fp8_gemm_nt_contiguous

    t = native["tensors"]
    block = args.quant_block_size
    cnt_tma = t["cnt_tma"].to(device=device, dtype=torch.int32)
    x13 = t["x13"].to(device=device).contiguous()
    rw = t["rw"].to(device=device, dtype=torch.float32)

    if packed:
        x13q, x13s = per_token_group_quant_fp8_packed_for_deepgemm(x13, block)
        grouped_kwargs: dict[str, Any] = {"use_psum_layout": True}
    else:
        x13q, x13s = per_token_group_quant_fp8(
            x13,
            block,
            eps=1e-4,
            column_major_scales=False,
            use_ue8m0=True,
        )
        grouped_kwargs = {
            "use_psum_layout": True,
            "recipe_a": (1, block),
            "recipe_b": (block, block),
        }

    w13q, w13s = quantize_weight_vllm(w13, block, packed=packed)
    y13 = torch.empty((x13q.shape[0], w13q.shape[1]), device=device, dtype=torch.bfloat16)
    m_grouped_fp8_gemm_nt_contiguous(
        (x13q, x13s),
        (w13q, w13s),
        y13,
        cnt_tma,
        **grouped_kwargs,
    )

    x2q, x2s = vllm_activation_quant(
        y13,
        rw,
        block,
        args.swiglu_limit,
        packed=packed,
    )
    w2q, w2s = quantize_weight_vllm(w2, block, packed=packed)
    y2 = torch.empty((x2q.shape[0], w2q.shape[1]), device=device, dtype=torch.bfloat16)
    m_grouped_fp8_gemm_nt_contiguous(
        (x2q, x2s),
        (w2q, w2s),
        y2,
        cnt_tma,
        **grouped_kwargs,
    )

    valid = dispatch["tensors"]["row_valid"].to(device).bool()
    native_y13 = t["y13"].to(device)
    native_y2 = t["y2"].to(device)
    result = {
        "y13_vs_native_valid": stats(y13, native_y13, valid),
        "y2_vs_native_valid": stats(y2, native_y2, valid),
        "reduced_y2_vs_native": stats(
            reduce_valid_rows(y2, dispatch["tensors"]),
            reduce_valid_rows(native_y2, dispatch["tensors"]),
        ),
    }
    del x13q, x13s, w13q, w13s, x2q, x2s, w2q, w2s, y13, y2
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("DeepGEMM replay requires CUDA.")

    llm_path = str(Path(args.llm_train_dir) / "llm")
    if llm_path not in sys.path:
        sys.path.insert(0, llm_path)

    native = load_pt(args.native_dump)
    native_dispatch = load_pt(args.native_dispatch_dump)
    vllm_dump = load_pt(args.vllm_dump)
    vllm_prepare_dump = (
        load_pt(args.vllm_prepare_dump) if args.vllm_prepare_dump else None
    )

    alignment = build_match_summary(native_dispatch, vllm_dump)
    native_rows = alignment.pop("native_rows")
    vllm_rows = alignment.pop("vllm_rows")

    summary: dict[str, Any] = {
        "native_dump": args.native_dump,
        "native_dispatch_dump": args.native_dispatch_dump,
        "vllm_dump": args.vllm_dump,
        "vllm_prepare_dump": args.vllm_prepare_dump,
        "layer": args.layer,
        "dispatch_alignment": alignment,
        "actual_vllm_vs_native": compare_actual_vllm_dump(
            native,
            native_dispatch,
            vllm_dump,
            native_rows,
            vllm_rows,
            args.quant_block_size,
        ),
        "replay": {},
    }
    if vllm_prepare_dump is not None:
        summary["prepare_input_quant_alignment"] = compare_prepare_input_quant(
            native,
            vllm_dump,
            vllm_prepare_dump,
            native_rows,
            vllm_rows,
            args.quant_block_size,
        )

    w13_name = f"model.layers.{args.layer}.mlp.experts.w13_weight"
    w2_name = f"model.layers.{args.layer}.mlp.experts.w2_weight"
    w13 = load_model_tensor(args.model, w13_name, device).contiguous()
    w2 = load_model_tensor(args.model, w2_name, device).contiguous()
    num_experts = int(native.get("num_local_experts", native["tensors"]["cnt"].numel()))
    hidden_dim = int(native.get("hidden_dim", native["tensors"]["x13"].shape[1]))
    ffn_dim = int(native.get("ffn_dim", w2.shape[-1]))
    if w13.dim() == 2:
        w13 = w13.view(num_experts, 2 * ffn_dim, hidden_dim).contiguous()
    if w2.dim() == 2:
        w2 = w2.view(num_experts, hidden_dim, ffn_dim).contiguous()

    variants = {item.strip() for item in args.variants.split(",") if item.strip()}
    if "native" in variants:
        summary["replay"]["native"] = replay_native(
            native, native_dispatch, w13, w2, args, device
        )
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    if "vllm_raw_recipe" in variants:
        summary["replay"]["vllm_raw_recipe"] = replay_vllm_same_input(
            native, native_dispatch, w13, w2, args, device, packed=False
        )
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    if "vllm_packed" in variants:
        summary["replay"]["vllm_packed"] = replay_vllm_same_input(
            native, native_dispatch, w13, w2, args, device, packed=True
        )
        torch.cuda.synchronize()

    printable = json.dumps(summary, indent=2, ensure_ascii=False)
    print(printable)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(printable + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
