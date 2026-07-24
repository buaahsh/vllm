#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run and compare a common-source FA4 BF16 CUDA smoke case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from logprob_kl import _NATIVE_FA4_RUNTIME, _import_fa4_varlen_func


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the vendored FA4 kernel")
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--fa4-source-root")
    run_parser.add_argument(
        "--fa4-profile",
        choices=("default", "no-ex2", "q-stage1", "q-stage2"),
        default="default",
    )

    compare_parser = subparsers.add_parser("compare", help="Compare two runs")
    compare_parser.add_argument("--reference", required=True)
    compare_parser.add_argument("--candidate", required=True)
    compare_parser.add_argument("--out-json", required=True)
    compare_parser.add_argument("--require-exact", action="store_true")
    return parser.parse_args()


def deterministic_tensor(
    shape: tuple[int, ...], *, multiplier: int, offset: int, device: torch.device
) -> torch.Tensor:
    count = 1
    for size in shape:
        count *= size
    values = torch.arange(count, dtype=torch.int32)
    values = ((values * multiplier + offset) % 251) - 125
    return (values.to(torch.float32) / 128.0).to(torch.bfloat16).reshape(shape).to(device)


def run_kernel(
    out_path: str,
    device_name: str,
    fa4_source_root: str | None,
    fa4_profile: str,
) -> None:
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    sequence_lengths = (17, 129)
    total_tokens = sum(sequence_lengths)
    num_query_heads = 8
    num_kv_heads = 2
    head_dim = 128

    q = deterministic_tensor(
        (total_tokens, num_query_heads, head_dim),
        multiplier=17,
        offset=3,
        device=device,
    )
    k = deterministic_tensor(
        (total_tokens, num_kv_heads, head_dim),
        multiplier=29,
        offset=5,
        device=device,
    )
    v = deterministic_tensor(
        (total_tokens, num_kv_heads, head_dim),
        multiplier=43,
        offset=7,
        device=device,
    )
    cu_seqlens = torch.tensor(
        [0, sequence_lengths[0], total_tokens], dtype=torch.int32, device=device
    )

    flash_attn_varlen_func = _import_fa4_varlen_func(
        "vllm-vendored" if fa4_source_root is None else "installed",
        source_root=fa4_source_root,
        profile=fa4_profile,
    )
    output, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(sequence_lengths),
        max_seqlen_k=max(sequence_lengths),
        causal=True,
        num_splits=1,
        return_lse=True,
    )
    torch.cuda.synchronize(device)
    if not torch.isfinite(output).all() or not torch.isfinite(lse).all():
        raise RuntimeError("FA4 smoke output contains non-finite values")

    artifact = {
        "torch_version": str(torch.__version__),
        "fa4_runtime": dict(_NATIVE_FA4_RUNTIME),
        "sequence_lengths": list(sequence_lengths),
        "q_shape": list(q.shape),
        "k_shape": list(k.shape),
        "v_shape": list(v.shape),
        "dtype": str(q.dtype),
        "output": output.cpu(),
        "lse": lse.cpu(),
    }
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, destination)
    print(f"Saved common-source FA4 smoke artifact: {destination}")


def tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference_float = reference.float()
    candidate_float = candidate.float()
    difference = candidate_float - reference_float
    denominator = torch.linalg.vector_norm(reference_float).clamp_min(1e-12)
    return {
        "exact": bool(torch.equal(reference, candidate)),
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "relative_l2": float(torch.linalg.vector_norm(difference) / denominator),
    }


def compare_runs(
    reference_path: str,
    candidate_path: str,
    out_json: str,
    require_exact: bool,
) -> None:
    reference = torch.load(reference_path, map_location="cpu")
    candidate = torch.load(candidate_path, map_location="cpu")
    for key in ("sequence_lengths", "q_shape", "k_shape", "v_shape", "dtype"):
        if reference[key] != candidate[key]:
            raise RuntimeError(
                f"FA4 smoke metadata mismatch for {key}: "
                f"{reference[key]!r} != {candidate[key]!r}"
            )

    report = {
        "reference": reference_path,
        "candidate": candidate_path,
        "reference_torch_version": reference["torch_version"],
        "candidate_torch_version": candidate["torch_version"],
        "reference_fa4_runtime": reference["fa4_runtime"],
        "candidate_fa4_runtime": candidate["fa4_runtime"],
        "output": tensor_metrics(reference["output"], candidate["output"]),
        "lse": tensor_metrics(reference["lse"], candidate["lse"]),
    }
    destination = Path(out_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if require_exact and not (report["output"]["exact"] and report["lse"]["exact"]):
        raise RuntimeError("Common-source FA4 outputs are not bitwise identical")


def main() -> None:
    args = parse_args()
    if args.command == "run":
        run_kernel(
            args.out,
            args.device,
            args.fa4_source_root,
            args.fa4_profile,
        )
    else:
        compare_runs(
            args.reference,
            args.candidate,
            args.out_json,
            args.require_exact,
        )


if __name__ == "__main__":
    main()