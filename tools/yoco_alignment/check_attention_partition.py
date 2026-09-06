# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe FA2 paged-KV query-length dispatch with identical Q/K/V."""

import argparse
import json
from pathlib import Path

import torch

from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mixed", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(190)
    length = 1024
    q = torch.randn(length, 64, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(66, 16, 8, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    table = torch.arange(64, device="cuda", dtype=torch.int32)[None]

    def run(query, size, maxq, window):
        query_len = query.shape[0]
        cuq = [0, query_len]
        seq_lens = [size]
        blocks = table
        maxk = size
        if args.mixed and query_len == 1:
            query = torch.cat([query, q[:22]])
            cuq = [0, 1, 23]
            seq_lens = [size, 22]
            filler_blocks = table.clone()
            filler_blocks[0, :2] = torch.tensor([64, 65], device="cuda")
            blocks = torch.cat([table, filler_blocks])
            maxq = max(maxq, 22)
            maxk = max(size, 22)
        result = flash_attn_varlen_func(
            q=query,
            k=k,
            v=v,
            cu_seqlens_q=torch.tensor(cuq, device="cuda", dtype=torch.int32),
            seqused_k=torch.tensor(seq_lens, device="cuda", dtype=torch.int32),
            max_seqlen_q=maxq,
            max_seqlen_k=maxk,
            block_table=blocks,
            softmax_scale=128**-0.5,
            causal=True,
            window_size=window,
            fa_version=2,
            num_splits=1,
        )
        return result[:query_len]

    records = []
    for window in ((-1, -1), (512, 0)):
        full = run(q, length, length, window)
        for pos in (0, 1, 15, 16, 31, 32, 63, 64, 69, 127, 511, 512, 1023):
            for maxq in (1, 2, 128):
                out = run(q[pos : pos + 1], pos + 1, maxq, window)
                expected = full[pos : pos + 1]
                row = {
                    "window": window,
                    "position": pos,
                    "maxq": maxq,
                    "equal": torch.equal(out, expected),
                    "different": int((out != expected).sum()),
                    "max_abs": float((out.float() - expected.float()).abs().max()),
                }
                records.append(row)
                print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
