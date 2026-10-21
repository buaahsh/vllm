# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ABBA microbenchmark for the fast-prefill boundary gather only.

Does not measure full prefill, attention, clone/scatter/norm, or model TTFT.
CUDA Graph batches isolate device work; eager wall timing includes Python.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from vllm.model_executor.layers.yoco_ops.prefill import gather_prefill_inputs


@torch.inference_mode()
def run_case(rows, selected, hidden_width, kv_width, kv_dtype, rounds, iterations):
    sources = [
        torch.arange(rows, device="cuda"),
        torch.randn(rows, hidden_width, device="cuda", dtype=torch.float32),
        torch.empty(rows, kv_width, device="cuda", dtype=kv_dtype),
        torch.empty(rows, kv_width, device="cuda", dtype=kv_dtype),
    ]
    for tensor in sources[2:]:
        raw = tensor.view(torch.uint8)
        raw.copy_(torch.randint(256, raw.shape, device="cuda", dtype=torch.uint8))
    indices = torch.randint(rows, (selected,), device="cuda")
    outputs = [torch.empty_like(x[:selected]) for x in sources]

    def baseline():
        outputs[0].copy_(sources[0][indices])
        outputs[1].copy_(sources[1][indices])
        for source, output in zip(sources[2:], outputs[2:]):
            if kv_dtype == torch.float8_e4m3fn:
                output.view(torch.uint8).copy_(source.view(torch.uint8)[indices])
            else:
                output.copy_(source[indices])

    def fused():
        gather_prefill_inputs(indices, *sources, *outputs)

    baseline()
    reference = [x.clone() for x in outputs]
    fused()
    for actual, expected in zip(outputs, reference):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))

    functions = {"baseline": baseline, "fused": fused}
    graphs = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for fn in functions.values():
            for _ in range(10):
                fn()
    torch.cuda.current_stream().wait_stream(stream)
    for name, fn in functions.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(iterations):
                fn()
        graphs[name] = graph
        for _ in range(3):
            graph.replay()
    torch.accelerator.synchronize()

    samples = []
    for mode in ("graph_device", "eager_wall"):
        for repeat in range(rounds):
            for name in ("baseline", "fused", "fused", "baseline"):
                torch.accelerator.synchronize()
                if mode == "graph_device":
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    us = start.elapsed_time(end) * 1000 / iterations
                else:
                    start_time = time.perf_counter()
                    for _ in range(iterations):
                        functions[name]()
                    torch.accelerator.synchronize()
                    us = (time.perf_counter() - start_time) * 1e6 / iterations
                samples.append(dict(mode=mode, repeat=repeat, variant=name, us=us))
    summary = {}
    for mode in ("graph_device", "eager_wall"):
        medians = {
            name: statistics.median(
                s["us"] for s in samples if s["mode"] == mode and s["variant"] == name
            )
            for name in functions
        }
        summary[mode] = dict(
            **medians, ratio_fused_over_baseline=medians["fused"] / medians["baseline"]
        )
    return dict(
        rows=rows,
        selected=selected,
        hidden_width=hidden_width,
        kv_width=kv_width,
        hidden_dtype="float32",
        kv_dtype=str(kv_dtype),
        byte_equality=True,
        summary=summary,
        samples=samples,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1024, 8192, 32768])
    parser.add_argument("--selected", type=int, nargs="+", default=[1, 32, 256])
    parser.add_argument("--hidden-width", type=int, default=5120)
    parser.add_argument("--kv-width", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(
            *args.rows,
            *args.selected,
            args.rounds,
            args.iterations,
            args.hidden_width,
            args.kv_width,
        )
        < 1
    ):
        parser.error("all dimensions and repetition counts must be positive")
    if max(args.selected) > min(args.rows):
        parser.error("selected must not exceed source rows")
    torch.manual_seed(20261020)
    props = torch.cuda.get_device_properties(0)
    result = dict(
        scope="gather only, warmed inputs and graph replay, no model or TTFT",
        device=props.name,
        uuid=str(props.uuid),
        sm_count=props.multi_processor_count,
        torch=torch.__version__,
        seed=20261020,
        rounds=args.rounds,
        iterations=args.iterations,
        cases=[],
    )
    for rows in args.rows:
        for selected in args.selected:
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                case = run_case(
                    rows,
                    selected,
                    args.hidden_width,
                    args.kv_width,
                    dtype,
                    args.rounds,
                    args.iterations,
                )
                result["cases"].append(case)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(
                    json.dumps({k: v for k, v in case.items() if k != "samples"}),
                    flush=True,
                )


if __name__ == "__main__":
    main()
