# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen explicit shared/latent FP8 candidates on real checkpoint weights.

Keeps the native DeepGEMM library and quantizers. Measures hot layer-0 operands
and then a rotating 80-matrix pool (20 physical layers x four projections).
The pool is an operator benchmark, not end-to-end model inference.
"""

import argparse
import hashlib
import json
import random
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors import safe_open

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
    per_token_group_quant_fp8_packed_for_deepgemm,
)
from vllm.model_executor.layers.yoco_ops.small_fp8 import SmallFP8Config, small_fp8_mm
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_gemm_nt, per_block_cast_to_fp8

PROJECTIONS = {
    "shared_w13": "shared_experts.gate_up_proj",
    "shared_w2": "shared_experts.down_proj",
    "latent_in": "fc1_latent_proj",
    "latent_out": "fc2_latent_proj",
}


def quantize(x):
    return per_token_group_quant_fp8_packed_for_deepgemm(x, 128, eps=1e-4)


def native(a, weight, a_scale, weight_scale):
    output = torch.empty(
        (a.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=a.device
    )
    fp8_gemm_nt(
        (a, a_scale), (weight, weight_scale), output, is_deep_gemm_e8m0_used=True
    )
    return output


def unpack(scales, k):
    groups = torch.arange(k // 128, device=scales.device)
    exponents = (scales[:, groups // 4].to(torch.int64) >> ((groups % 4) * 8)) & 255
    return (
        (exponents.to(torch.int32) << 23).view(torch.float32).repeat_interleave(128, 1)
    )


def reference(a, weight, a_scale, weight_scale):
    x = a.double() * unpack(a_scale, a.shape[1]).double()
    w = weight.double() * unpack(weight_scale, weight.shape[1]).double()
    return (x @ w.T).to(torch.bfloat16)


def error(actual, expected):
    delta = actual.float() - expected.float()
    relative = delta.norm() / expected.float().norm().clamp_min(1e-30)
    return {
        "relative_l2": relative.item(),
        "max_abs": delta.abs().max().item(),
        "changed_elements": (actual != expected).sum().item(),
        "finite": bool(torch.isfinite(actual).all()),
    }


def measure(functions, *, nodes=16, calls=1, rounds=5):
    graphs = {}
    for label, fn in functions.items():
        for _ in range(3):
            fn()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(nodes):
                fn()
        graphs[label] = graph
    samples = {label: [] for label in graphs}
    order = list(graphs)
    rng = random.Random(918)
    for _ in range(rounds):
        rng.shuffle(order)
        for label in order:
            graph = graphs[label]
            for _ in range(3):
                graph.replay()
            begin, end = (
                torch.Event(enable_timing=True),
                torch.Event(enable_timing=True),
            )
            begin.record()
            for _ in range(30):
                graph.replay()
            end.record()
            end.synchronize()
            samples[label].append(begin.elapsed_time(end) * 1000 / (30 * nodes * calls))
    return {
        label: {"us": statistics.median(values), "samples_us": values}
        for label, values in samples.items()
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite existing results")
    if any(m not in [1, 2, 4, 8] for m in args.batches):
        raise ValueError("Supported benchmark batches: 1, 2, 4, 8")
    torch.manual_seed(918)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    result = {
        "completed": False,
        "gpu": current_platform.get_device_name(),
        "gpu_uuid": current_platform.get_device_uuid(),
        "torch": str(torch.__version__),
        "candidate_sha256": hashlib.sha256(
            Path(sys.modules[small_fp8_mm.__module__].__file__).read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": (
            "Real checkpoint weights; random BF16 activations; pure GEMM and "
            "input-quant+GEMM; not model quality or end-to-end timing"
        ),
        "weights": [],
        "screen": [],
        "rotating": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    weights = []
    for layer in range(20):
        for category, suffix in PROJECTIONS.items():
            key = f"model.layers.{layer}.mlp.{suffix}.weight"
            with safe_open(
                args.model / index[key], framework="pt", device="cpu"
            ) as handle:
                bf16 = handle.get_tensor(key).to(device="cuda", dtype=torch.bfloat16)
            qweight, scale = per_block_cast_to_fp8(bf16, [128, 128], use_ue8m0=True)
            qweight, scale = deepgemm_post_process_fp8_weight_block(
                qweight, scale, (128, 128), use_e8m0=True
            )
            weights.append((layer, category, qweight, scale))
            result["weights"].append(
                {
                    "key": key,
                    "shape": list(qweight.shape),
                    "scale_shape": list(scale.shape),
                    "scale_stride": list(scale.stride()),
                }
            )
    del bf16
    result["weight_payload_bytes"] = sum(w.numel() for _, _, w, _ in weights)
    configs = [
        SmallFP8Config("tensor", n, 4, stages)
        for n in [16, 32, 64]
        for stages in [2, 4]
    ]
    configs += [
        SmallFP8Config("direct", n, warps, 1) for n in [1, 2, 4] for warps in [4, 8]
    ]
    choices = {
        f"{c.backend}-n{c.block_n}-w{c.num_warps}-s{c.num_stages}": c for c in configs
    }
    save()
    for m in args.batches:
        selected = {}
        for _, category, w, ws in weights[:4]:
            x = torch.randn(m, w.shape[1], device="cuda", dtype=torch.bfloat16)
            a, sa = quantize(x)
            operands = (a, w, sa, ws)
            expected = reference(*operands)
            base = native(*operands)
            functions = {"native": lambda operands=operands: native(*operands)}
            errors = {"native": error(base, expected)}
            for label, cfg in choices.items():
                actual = small_fp8_mm(*operands, cfg)
                errors[label] = {
                    **error(actual, expected),
                    "versus_native": error(actual, base),
                }
                if not errors[label]["finite"] or errors[label]["relative_l2"] > 1e-3:
                    raise AssertionError((category, m, label, errors[label]))
                functions[label] = lambda cfg=cfg, operands=operands: small_fp8_mm(
                    *operands, cfg
                )
            timings = measure(functions)
            best = min(choices, key=lambda label: timings[label]["us"])
            selected[category] = choices[best]
            row = {
                "batch": m,
                "category": category,
                "timings": timings,
                "errors": errors,
                "selected": best,
            }
            result["screen"].append(row)
            save()
            print(
                "SCREEN",
                m,
                category,
                "native",
                round(timings["native"]["us"], 3),
                "best",
                best,
                round(timings[best]["us"], 3),
                flush=True,
            )
        cases, validation = [], []
        for layer, category, w, ws in weights:
            x = torch.randn(m, w.shape[1], device="cuda", dtype=torch.bfloat16)
            a, sa = quantize(x)
            cfg = selected[category]
            expected = reference(a, w, sa, ws)
            base, candidate = native(a, w, sa, ws), small_fp8_mm(a, w, sa, ws, cfg)
            metrics = {
                "layer": layer,
                "category": category,
                "versus_fp64": error(candidate, expected),
                "versus_native": error(candidate, base),
            }
            assert (
                metrics["versus_fp64"]["finite"]
                and metrics["versus_fp64"]["relative_l2"] < 1e-3
            )
            validation.append(metrics)
            cases.append((category, x, a, w, sa, ws, cfg))

        def pool(candidate, with_quant=False, cases=cases):
            for _, x, a, w, sa, ws, cfg in cases:
                if with_quant:
                    a, sa = quantize(x)
                if candidate:
                    small_fp8_mm(a, w, sa, ws, cfg)
                else:
                    native(a, w, sa, ws)

        timings = measure(
            {
                "native_gemm": lambda: pool(False),
                "candidate_gemm": lambda: pool(True),
                "native_quant_gemm": lambda: pool(False, True),
                "candidate_quant_gemm": lambda: pool(True, True),
            },
            nodes=1,
            calls=len(cases),
            rounds=7,
        )
        result["rotating"].append(
            {
                "batch": m,
                "matrices": len(cases),
                "choices": {k: asdict(v) for k, v in selected.items()},
                "timings_per_projection": timings,
                "validation": validation,
            }
        )
        save()
        print(
            "ROTATING",
            m,
            {k: round(v["us"], 3) for k, v in timings.items()},
            flush=True,
        )
    dg = next(
        sys.modules[name]
        for name in ("deep_gemm", "vllm.third_party.deep_gemm")
        if name in sys.modules
    )
    native_path = Path(dg._C.__file__)
    result["native_deep_gemm"] = {
        "module": str(Path(dg.__file__).resolve()),
        "library": str(native_path.resolve()),
        "sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
    }
    result["completed"] = True
    save()


if __name__ == "__main__":
    main()
