# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check YOCO operator invariance using unrelated rows and CUDA Graph replay.

Run in a fresh process/cache for each candidate. This is an operator gate,
not evidence of whole-model or train/inference equivalence.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from vllm.model_executor.models.yoco import (
    RMSClip,
    RMSNorm,
    YOCOCombinedOutputTransform,
    _yoco_align_linear,
    _yoco_align_router_linear,
    _yoco_align_shared_expert_swiglu,
    _yoco_align_topk_routing,
    _yoco_diff_attention_v3_dispatch,
)


def differences(actual, expected):
    if isinstance(actual, torch.Tensor):
        actual, expected = (actual,), (expected,)
    return {
        "equal": all(torch.equal(a, b) for a, b in zip(actual, expected)),
        "different_elements": sum(
            int(torch.count_nonzero(a != b)) for a, b in zip(actual, expected)
        ),
        "max_abs_diff": max(
            float((a.float() - b.float()).abs().max()) for a, b in zip(actual, expected)
        ),
    }


def select(output, indices):
    if isinstance(output, torch.Tensor):
        return output[indices]
    return tuple(t[indices] for t in output)


def join(outputs):
    if isinstance(outputs[0], torch.Tensor):
        return torch.cat(outputs)
    return tuple(torch.cat(parts) for parts in zip(*outputs))


def cases():
    for hidden in (1024, 3072):
        for dtype in (torch.bfloat16, torch.float32):
            for residual in (False, True):
                module = RMSNorm(hidden, execution_mode="align").cuda()
                module.weight.data.uniform_(-2, 2)
                yield (
                    f"norm_{hidden}_{dtype}_{residual=}",
                    module,
                    (hidden,),
                    dtype,
                    residual,
                )
    for heads in (8, 64):
        for weighted in (False, True):
            module = (
                RMSClip(128, has_weight=weighted, execution_mode="align")
                .cuda()
                .to(torch.bfloat16)
            )
            if weighted:
                module.weight.data.uniform_(-2, 2)
            yield (
                f"clip_{heads}_{weighted=}",
                module,
                (heads, 128),
                torch.bfloat16,
                False,
            )
    for n in (64, 1024, 8192, 154880):
        weight = torch.randn(n, 3072, device="cuda", dtype=torch.bfloat16) * 0.02
        yield (
            f"linear_3072_{n}",
            lambda x, w=weight: _yoco_align_linear(x, w),
            (3072,),
            torch.bfloat16,
            False,
        )
    for normalized in (False, True):
        weight = torch.randn(128, 1024, device="cuda", dtype=torch.float32) * 0.02
        yield (
            f"router_{normalized=}",
            lambda x, w=weight, n=normalized: _yoco_align_router_linear(x, w, n),
            (1024,),
            torch.float32,
            False,
        )
    yield (
        "router_topk",
        lambda x: _yoco_align_topk_routing(x, x, 8, True),
        (128,),
        torch.float32,
        False,
    )
    weight = torch.randn(1, 3072, device="cuda", dtype=torch.bfloat16) * 0.02
    yield (
        "pointwise_scalar_gate",
        lambda x: _yoco_align_linear(x, weight),
        (3072,),
        torch.bfloat16,
        False,
    )
    yield (
        "pointwise_shared_swiglu",
        lambda x: _yoco_align_shared_expert_swiglu(
            x[:, 0].contiguous(), x[:, 1].contiguous(), 10.0
        ),
        (2, 1280),
        torch.bfloat16,
        False,
    )
    yield (
        "pointwise_diff",
        lambda x: _yoco_diff_attention_v3_dispatch(
            x[:, :, :128].contiguous(), x[:, :, 128].contiguous(), False
        ),
        (64, 129),
        torch.bfloat16,
        False,
    )
    gate = torch.nn.Linear(3072, 1, bias=False, device="cuda", dtype=torch.bfloat16)
    combined = YOCOCombinedOutputTransform(gate, execution_mode="align")
    yield (
        "pointwise_combined",
        lambda x: combined(x[:, 0], x[:, 1], x[:, 2]),
        (3, 3072),
        torch.bfloat16,
        False,
    )
    for normalized in (False, True):
        weight = torch.randn(128, 3072, device="cuda", dtype=torch.float32) * 0.02
        yield (
            f"router_l3_{normalized=}",
            lambda x, w=weight, n=normalized: _yoco_align_router_linear(x, w, n),
            (3072,),
            torch.float32,
            False,
        )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 2, 4, 100, 127, 128, 129, 1024, 2048],
    )
    parser.add_argument("--filter", default="")
    parser.add_argument("--init-runtime", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--warmup-batches", type=int, nargs="*", default=[])
    args = parser.parse_args()
    if args.init_runtime:
        from vllm.model_executor.layers.batch_invariant import init_batch_invariance

        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
        init_batch_invariance()
    reference = {}
    if args.reference:
        reference = {
            (r["operator"], r["batch"]): r["target_sha256"]
            for r in json.loads(args.reference.read_text())["cases"]
        }
    torch.manual_seed(20260905)
    rows = []
    for name, module, shape, dtype, has_residual in cases():
        if args.filter not in name:
            continue
        target = 4 * torch.randn(1, *shape, device="cuda", dtype=dtype)
        target_inputs = [target]
        if has_residual:
            target_inputs.append(torch.randn_like(target, dtype=torch.float32))
        for warmup_batch in args.warmup_batches:
            module(
                *(t.expand(warmup_batch, *shape).contiguous() for t in target_inputs)
            )
        expected = module(*target_inputs)
        for batch in args.batches:
            positions = sorted({0, batch // 3, batch // 2, batch - 1})
            inputs = [
                4 * torch.randn(batch, *shape, device="cuda", dtype=t.dtype)
                for t in target_inputs
            ]
            for x, t in zip(inputs, target_inputs):
                x[positions] = t
            actual = module(*inputs)
            selected = select(actual, positions)
            tensors = (selected,) if isinstance(selected, torch.Tensor) else selected
            digest = hashlib.sha256()
            for tensor in tensors:
                digest.update(
                    tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                )
            wanted = select(expected, [0] * len(positions))
            checks = {"positions": differences(select(actual, positions), wanted)}

            # Ragged partitions exercise a new last-tile and reduction context.
            chunks = [
                module(*(x[i : i + 17] for x in inputs)) for i in range(0, batch, 17)
            ]
            checks["microbatches"] = differences(join(chunks), actual)
            for _ in range(3):
                module(*inputs)
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = module(*inputs)
            graph.replay()
            checks["graph"] = differences(captured, actual)
            # Change unrelated rows at stable addresses, then replay again.
            for x, t in zip(inputs, target_inputs):
                x.mul_(-0.75)
                x[positions] = t
            graph.replay()
            checks["graph_changed_fillers"] = differences(
                select(captured, positions), wanted
            )
            if args.reference:
                matches = digest.hexdigest() == reference[(name, batch)]
                checks["fresh_process"] = {"equal": matches}
            row = {
                "operator": name,
                "batch": batch,
                "checks": checks,
                "target_sha256": digest.hexdigest(),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            del graph, captured, chunks, actual, inputs
    if not rows:
        raise ValueError("No operator matched --filter")
    report = {
        "configuration": {
            "filter": args.filter,
            "batches": args.batches,
            "warmup_batches": args.warmup_batches,
            "init_runtime": args.init_runtime,
        },
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "all_equal": all(c["equal"] for r in rows for c in r["checks"].values()),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
