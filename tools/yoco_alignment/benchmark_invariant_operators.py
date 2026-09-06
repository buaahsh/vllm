# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA Graph kernel latency for the former and invariant Align operators."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm.model_executor.models.yoco import (
    RMSClip,
    RMSNorm,
    _yoco_align_linear,
    _yoco_align_rms_clip,
    _yoco_align_rms_norm,
    _yoco_align_router_linear,
    _yoco_weighted_rms_clip_kernel,
)
from vllm.triton_utils import triton


def previous_clip(x, module):
    if x.shape[0] < 128:
        return _yoco_align_rms_clip(x, module.weight, module.eps, module.limit)
    output = torch.empty_like(x)
    tokens, heads, _ = x.shape
    rows = tokens * heads
    block_rows = 16 if rows < 12288 else 32
    _yoco_weighted_rms_clip_kernel[((rows + block_rows - 1) // block_rows,)](
        x,
        module.weight,
        output,
        tokens,
        heads,
        x.stride(0),
        x.stride(1),
        eps=module.eps,
        limit=module.limit,
        HEAD_DIM=128,
        BLOCK_ROWS=block_rows,
        ROUND_BEFORE_WEIGHT=False,
        num_warps=8,
        num_stages=1,
    )
    return output


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20260905)
    results = []
    for batch in (1, 8, 128, 1024, 2048):
        pairs = []
        for hidden in (1024, 3072):
            norm = RMSNorm(hidden, execution_mode="align").cuda()
            x = torch.randn(batch, hidden, device="cuda", dtype=torch.float32)
            if hidden == 3072:
                previous = lambda x=x, m=norm: torch.ops.vllm.yoco_align_rms_norm(
                    x, m.weight, m.eps
                )
            else:
                previous = lambda x=x, m=norm: _yoco_align_rms_norm(x, m.weight, m.eps)
            pairs.append((f"norm_{hidden}", previous, lambda x=x, m=norm: m(x)))
        for heads in (8, 64):
            clip = (
                RMSClip(128, has_weight=True, execution_mode="align").cuda().bfloat16()
            )
            x = 4 * torch.randn(batch, heads, 128, device="cuda", dtype=torch.bfloat16)
            previous = lambda x=x, m=clip: previous_clip(x, m)
            pairs.append((f"clip_{heads}", previous, lambda x=x, m=clip: m(x)))
        x = torch.randn(batch, 3072, device="cuda")
        w = torch.randn(128, 3072, device="cuda")
        pairs.append(
            (
                "router_3072",
                lambda x=x, w=w: torch.ops.vllm.yoco_router_linear_tf32(x, w, False),
                lambda x=x, w=w: _yoco_align_router_linear(x, w, False),
            )
        )
        x = torch.randn(batch, 3072, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(8192, 3072, device="cuda", dtype=torch.bfloat16)
        pairs.append(
            (
                "q_proj",
                lambda x=x, w=w: F.linear(x, w),
                lambda x=x, w=w: _yoco_align_linear(x, w),
            )
        )
        for name, previous, candidate in pairs:
            previous()
            candidate()
            old_us = 1000 * triton.testing.do_bench_cudagraph(previous, rep=100)
            new_us = 1000 * triton.testing.do_bench_cudagraph(candidate, rep=100)
            row = {
                "operator": name,
                "batch": batch,
                "previous_us": old_us,
                "invariant_us": new_us,
                "ratio": new_us / old_us,
            }
            results.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
