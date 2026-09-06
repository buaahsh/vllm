# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure smaller M tiles without changing the invariant GEMM's K traversal."""

import argparse
import json
from pathlib import Path

import torch

from vllm.model_executor.layers.batch_invariant import (
    linear_batch_invariant,
    matmul_kernel_persistent,
    num_compute_units,
)
from vllm.triton_utils import triton


def linear(x, weight, block_m):
    m, k = x.shape
    n = weight.shape[0]
    out = x.new_empty((m, n))
    sms = num_compute_units(x.device.index)
    matmul_kernel_persistent[
        (min(sms, triton.cdiv(m, block_m) * triton.cdiv(n, 128)),)
    ](
        x,
        weight.t(),
        out,
        None,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        weight.stride(1),
        weight.stride(0),
        out.stride(0),
        out.stride(1),
        NUM_SMS=sms,
        A_LARGE=False,
        B_LARGE=False,
        C_LARGE=False,
        HAS_BIAS=False,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
        num_stages=3,
        num_warps=4 if block_m <= 32 else 8,
    )
    return out


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(24)
    rows = []
    for n in (64, 1024, 8192, 154880):
        w = torch.randn(n, 3072, device="cuda", dtype=torch.bfloat16)
        for m in (1, 8, 17, 128, 2048):
            x = torch.randn(m, 3072, device="cuda", dtype=torch.bfloat16)
            expected = linear_batch_invariant(x, w)
            for bm in (16, 32, 128):
                actual = linear(x, w, bm)
                equal = torch.equal(actual, expected)
                us = 1000 * triton.testing.do_bench_cudagraph(
                    lambda x=x, w=w, bm=bm: linear(x, w, bm), rep=60
                )
                row = {"m": m, "n": n, "bm": bm, "equal": equal, "us": us}
                rows.append(row)
                print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    if not all(r["equal"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
