# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare compiled latent norm + Mega staging with direct fused staging."""

import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.yoco_ops.mega_moe import (
    prepare_yoco_latent_megamoe_inputs,
)
from vllm.model_executor.layers.yoco_ops.norm import RMSNorm
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for rows in [9, 18, 36, 72, 128, 512, 2048]:
        torch.manual_seed(20261021)
        x = torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16)
        norm = RMSNorm(1024, dtype=torch.bfloat16).cuda()
        ids = torch.randint(128, (rows, 8), device="cuda", dtype=torch.int32)
        weights = torch.rand(rows, 8, device="cuda")
        buffer = SimpleNamespace(
            x=torch.empty_like(x, dtype=torch.float8_e4m3fn),
            x_sf=torch.empty(rows, 8, device="cuda", dtype=torch.int32),
            topk_idx=torch.empty_like(ids, dtype=torch.int64),
            topk_weights=torch.empty_like(weights),
        )

        def original(x=x, norm=norm, weights=weights, ids=ids, buffer=buffer):
            prepare_megamoe_inputs(
                norm(x),
                weights,
                ids,
                buffer.x,
                buffer.x_sf,
                buffer.topk_idx,
                buffer.topk_weights,
            )

        def fused(x=x, norm=norm, weights=weights, ids=ids, buffer=buffer):
            prepare_yoco_latent_megamoe_inputs(x, norm, weights, ids, buffer)

        original()
        expected = {name: t.clone() for name, t in vars(buffer).items()}
        fused()
        for name, value in vars(buffer).items():
            reference = expected[name]
            if name == "x":
                value, reference = value.view(torch.uint8), reference.view(torch.uint8)
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        graphs = {}
        for mode, fn in [("original", original), ("fused", fused)]:
            for _ in range(5):
                fn()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(100):
                    fn()
            graphs[mode] = graph
        samples = {mode: [] for mode in graphs}
        for trial in range(8):
            order = ["original", "fused", "fused", "original"]
            if trial % 2:
                order = ["fused", "original", "original", "fused"]
            for mode in order:
                graph = graphs[mode]
                for _ in range(5):
                    graph.replay()
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                for _ in range(20):
                    graph.replay()
                end.record()
                end.synchronize()
                samples[mode].append(start.elapsed_time(end) * 1000 / 2000)
        medians = {mode: statistics.median(data) for mode, data in samples.items()}
        row = dict(
            rows=rows,
            samples_us=samples,
            median_us=medians,
            reduction_pct=(1 - medians["fused"] / medians["original"]) * 100,
        )
        results.append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
