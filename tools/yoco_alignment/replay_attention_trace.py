# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay trained first-layer Q/K/V across FA2 decode launch contexts."""

import argparse
import json
from pathlib import Path

import torch

from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = torch.load(args.trace, weights_only=True)
    starts = [i for i, r in enumerate(data) if r["name"] == "embed_tokens"]
    prefill, decode = data[: starts[1]], data[starts[1] : starts[2]]
    rows = []
    for loop in range(3):
        name = f"layers.0.self_attn.attn.{loop}"
        p = next(r for r in prefill if r["name"] == name)
        d = next(r for r in decode if r["name"] == name)
        q = d["inputs"][0].cuda().reshape(1, 64, 128)
        k = torch.zeros(7, 16, 8, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.zeros_like(k)
        k.view(-1, 8, 128)[:67] = (
            torch.cat([p["inputs"][1], d["inputs"][1]]).cuda().reshape(67, 8, 128)
        )
        v.view(-1, 8, 128)[:67] = (
            torch.cat([p["inputs"][2], d["inputs"][2]]).cuda().reshape(67, 8, 128)
        )
        k[5:] = k[:2]
        v[5:] = v[:2]
        expected = d["output"].cuda().reshape(1, 64, 128)
        for mixed in (False, True):
            query = (
                torch.cat([q, p["inputs"][0][:22].cuda().reshape(22, 64, 128)])
                if mixed
                else q
            )
            cuq = torch.tensor(
                [0, 1, 23] if mixed else [0, 1], device="cuda", dtype=torch.int32
            )
            lens = torch.tensor(
                [67, 22] if mixed else [67], device="cuda", dtype=torch.int32
            )
            blocks = torch.tensor(
                [[0, 1, 2, 3, 4], [5, 6, 0, 0, 0]] if mixed else [[0, 1, 2, 3, 4]],
                device="cuda",
                dtype=torch.int32,
            )
            for maxq in (22,) if mixed else (1, 2, 22):
                for maxk in (67, 4104):
                    out = flash_attn_varlen_func(
                        q=query,
                        k=k,
                        v=v,
                        cu_seqlens_q=cuq,
                        seqused_k=lens,
                        max_seqlen_q=maxq,
                        max_seqlen_k=maxk,
                        block_table=blocks,
                        softmax_scale=128**-0.5,
                        causal=True,
                        window_size=(512, 0),
                        fa_version=2,
                        num_splits=1,
                    )[:1]
                    row = {
                        "loop": loop,
                        "mixed": mixed,
                        "maxq": maxq,
                        "maxk": maxk,
                        "equal": torch.equal(out, expected),
                        "different": int((out != expected).sum()),
                        "max_abs": float((out.float() - expected.float()).abs().max()),
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
