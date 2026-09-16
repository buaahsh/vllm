# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline B1 operand accounting and audited YOCO CUDA Graph trace summary.

Operand bytes are a once-per-invocation reference, not measured DRAM traffic.
Kernel duration sums include stream overlap; graph spans measure elapsed time.
This tool neither imports vLLM nor starts GPU work.
"""

import argparse
import gzip
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def weight_reference(config):
    """Count L3 Fast FP8 matrix operands, including repeated self layers."""
    expected = {
        "d_model": 3072,
        "moe_latent_dim": 1024,
        "moe_ffn_dim": 3840,
        "d_shared_expert": 1280,
        "moe_top_k": 8,
        "moe_expert_num": 128,
        "head": 32,
        "cross_head": 32,
        "kv_head": 8,
        "cross_kv_head": 8,
        "head_dim": 128,
        "cross_head_dim": 128,
        "n_layers": 20,
        "yoco_cross_layers": 10,
        "universal_loop": 3,
        "vocab_size": 154880,
        "diff_v3": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unvalidated L3 shape: {key}={config.get(key)!r}")
    hidden, latent, ffn, shared = 3072, 1024, 3840, 1280
    # diff_v3 has twice the query heads, followed by a 4096-wide diff output.
    self_calls, cross_calls, calls = 30, 10, 40
    matrices = {
        "routed_w13": (calls * 8 * 2 * latent * ffn, 1),
        "routed_w2": (calls * 8 * latent * ffn, 1),
        "shared_w13": (calls * 2 * hidden * shared, 1),
        "shared_w2": (calls * hidden * shared, 1),
        "self_qkv": (self_calls * hidden * (64 + 2 * 8) * 128, 1),
        "cross_q": (cross_calls * hidden * 64 * 128, 1),
        "attention_o": (calls * hidden * 32 * 128, 1),
        "shared_kv_proj": (hidden * 2 * 8 * 128, 1),
        "latent_in": (calls * hidden * latent, 1),
        "latent_out": (calls * hidden * latent, 1),
        "router": (calls * hidden * 128, 2),
        "lambda": (calls * hidden * 64, 2),
        "shared_gate": (calls * hidden, 2),
        "lm_head": (hidden * 154880, 2),
    }
    weights = {name: elements * size for name, (elements, size) in matrices.items()}
    total = sum(weights.values())
    return {
        "scope": "B1 FP8 weights, BF16 router/lambda/head, FP8 KV; not DRAM bytes",
        "self_calls": self_calls,
        "cross_calls": cross_calls,
        "weight_bytes": weights,
        "weight_bytes_total": total,
        "executed_weight_elements": sum(elements for elements, _ in matrices.values()),
        "kv_scenarios": [
            {
                "context": context,
                "kv_read_bytes": (
                    self_calls * min(context, 513) + cross_calls * context
                )
                * 2
                * 8
                * 128,
                "weight_plus_kv_ms_at_8TBps": (
                    total
                    + (self_calls * min(context, 513) + cross_calls * context)
                    * 2
                    * 8
                    * 128
                )
                / 8e9,
            }
            for context in [512, 576, 640, 8192, 32768, 80000]
        ],
        "excludes": [
            "scales, indices, norm parameters, activations and scratch",
            "cache hits, CTA reloads, transaction rounding and spills",
        ],
    }


def interval_union(events):
    """GPU busy time, counting concurrent kernels only once."""
    total, end = 0.0, float("-inf")
    for event in sorted(events, key=lambda e: e["ts"]):
        stop = event["ts"] + event["dur"]
        total += max(0, stop - max(end, event["ts"]))
        end = max(end, stop)
    return total


def dense_category(name):
    if "deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl" not in name:
        return None
    marker = "128u, 128u, 0u, "
    if marker not in name:
        return None
    fields = name.split(marker, 1)[1].split(", ", 2)[:2]
    if len(fields) != 2 or any(
        not f.endswith("u") or not f[:-1].isdigit() for f in fields
    ):
        return None
    return {
        (10240, 3072): "self_qkv",
        (8192, 3072): "cross_q",
        (3072, 4096): "attention_o",
        (2048, 3072): "shared_kv_proj",
        (1024, 3072): "latent_in",
        (3072, 1024): "latent_out",
        (2560, 3072): "shared_w13",
        (3072, 1280): "shared_w2",
    }.get(tuple(int(field[:-1]) for field in fields))


def coarse_category(category):
    if category.startswith("routed_"):
        return "Routed experts"
    groups = {
        "Shared expert": {
            "shared_w13",
            "shared_w2",
            "quant_shared_w13",
            "shared_activation",
            "scale_zero",
        },
        "Latent projections / norm": {
            "latent_in",
            "latent_out",
            "quant_latent_in",
            "quant_latent_out",
            "triton_per_fused__fused_rms_norm_per_token_group_fp8_quant_packed_1",
            "triton_per_fused__fused_rms_norm_moe_forward_shared_0",
        },
        "Attention projections": {
            "self_qkv",
            "cross_q",
            "attention_o",
            "shared_kv_proj",
            "quant_self_qkv",
            "quant_cross_q",
            "quant_shared_kv_proj",
        },
        "Router / gates": {
            "router",
            "lambda",
            "_yoco_fused_topk_routing_kernel",
            "_yoco_fused_shared_gate_moe_output_kernel",
        },
        "FA4 attention / combine": {"attention_main", "attention_combine"},
        "Attention preparation": {
            "attention_scheduler",
            "attention_descale",
            "_cache_fp8_kernel",
            "_yoco_qk_rms_clip_rotary_kernel",
            "_yoco_weighted_rms_clip_kernel",
            "_diff_quant_kernel",
        },
        "Residual / norm": {"_yoco_fused_add_rms_norm_kernel", "_yoco_rms_norm_kernel"},
        "LM head": {"lm_head"},
    }
    return next(
        (label for label, categories in groups.items() if category in categories),
        "Sampling / layout / other",
    )


def summarize_trace(path, batch):
    with gzip.open(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    annotations = [
        e
        for e in events
        if e.get("cat") == "user_annotation" and e["name"].startswith("execute_context")
    ]
    expected = f"execute_context_0(0)_generation_{batch}({batch})"
    if len(annotations) != 12 or {e["name"] for e in annotations} != {expected}:
        raise ValueError("Trace must contain 12 observed pure-generation steps")
    kernels = [e for e in events if e.get("cat") == "kernel"]
    graphs, streams = defaultdict(list), defaultdict(list)
    for event in kernels:
        graphs[event["args"]["correlation"]].append(event)
        streams[event["args"]["stream"]].append(event)
    categories, audited_graphs = {}, []
    for correlation, graph in graphs.items():
        attention = sorted(
            [e for e in graph if "FlashAttentionForwardSm100" in e["name"]],
            key=lambda e: e["ts"],
        )
        if not attention:
            continue
        combine = [e for e in graph if "FlashAttentionForwardCombine" in e["name"]]
        launches = [
            e
            for e in events
            if e.get("cat") == "cuda_runtime"
            and e["name"].startswith("cudaGraphLaunch")
            and e.get("args", {}).get("correlation") == correlation
        ]
        if len(attention) != 40 or len(combine) != 40 or len(launches) != 1:
            raise ValueError("Expected 40 attention calls and one actual graph replay")
        routed = sorted(
            [e for e in graph if e["name"] == "fused_moe_kernel"],
            key=lambda e: e["ts"],
        )
        if len(routed) != 80 or len({e["args"]["stream"] for e in routed}) != 1:
            raise ValueError("Expected 40 same-stream W13/W2 pairs")
        for i, event in enumerate(routed):
            categories[id(event)] = "routed_w13" if i % 2 == 0 else "routed_w2"
        for event in attention:
            stream = sorted(
                [e for e in graph if e["args"]["stream"] == event["args"]["stream"]],
                key=lambda e: e["ts"],
            )
            index = next(i for i, e in enumerate(stream) if e is event)
            # v0.29 inserts its FA4 scheduler preparation after the three copies.
            start = index - 1
            if "FlashPrepareScheduler" in stream[start]["name"]:
                categories[id(stream[start])] = "attention_scheduler"
                start -= 1
            copies = stream[start - 2 : start + 1]
            if len(copies) != 3 or any(
                "direct_copy_kernel_cuda" not in e["name"]
                or "lambda(float)" not in e["name"]
                for e in copies
            ):
                raise ValueError("FA4 descale triplet not proven by stream adjacency")
            for copy in copies:
                categories[id(copy)] = "attention_descale"
        begin = min(e["ts"] for e in graph)
        span = max(e["ts"] + e["dur"] for e in graph) - begin
        busy = interval_union(graph)
        audited_graphs.append(
            {
                "correlation": correlation,
                "kernels": len(graph),
                "span_us": span,
                "busy_union_us": busy,
                "no_kernel_us": span - busy,
                "kernel_sum_us": sum(e["dur"] for e in graph),
                "attention_calls": len(attention),
                "routed_calls": len(routed),
            }
        )
    if len(audited_graphs) != 12:
        raise ValueError("Expected 12 complete CUDA Graph replays")
    for stream in streams.values():
        stream.sort(key=lambda e: e["ts"])
        for i, event in enumerate(stream):
            name = event["name"]
            previous = stream[i - 1]["name"] if i else ""
            following = stream[i + 1]["name"] if i + 1 < len(stream) else ""
            category = categories.get(id(event))
            if category:
                continue
            dense = dense_category(name)
            if dense:
                category = dense
            elif "packed_register_kernel" in name and dense_category(following):
                category = "quant_" + dense_category(following)
            elif name == "_silu_mul_quant_fp8_packed_kernel":
                if following == "fused_moe_kernel":
                    category = "routed_activation"
                elif dense_category(following) == "shared_w2":
                    category = "shared_activation"
            elif "FlashAttentionForwardSm100" in name:
                category = "attention_main"
            elif "FlashAttentionForwardCombine" in name:
                category = "attention_combine"
            elif "nvjet_" in name or "cublasLt::splitKreduce_kernel" in name:
                gemm = previous if "splitKreduce" in name else name
                if "nvjet_sm100_tst_32x64_64x16_4x1_" in gemm:
                    category = "router"
                elif (
                    "nvjet_sm100_tst_32x64_64x16_2x1_" in gemm
                    or "cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8"
                    in gemm
                ):
                    category = "lambda"
            elif "cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8" in name:
                category = "lambda"
            elif "per_token_group_quant_8bit_kernel<" in name:
                category = "routed_quant"
            elif "moe_sum_vec_kernel<c10::BFloat16" in name:
                category = "routed_reduce"
            elif "moe_align_block_size_e128_small_kernel" in name:
                category = "routed_layout"
            elif name == "_yoco_lm_head_kernel":
                category = "lm_head"
            elif "FillFunctor<int>" in name and following == (
                "_silu_mul_quant_fp8_packed_kernel"
            ):
                category = "scale_zero"
            categories[id(event)] = category or name
    rows = defaultdict(list)
    for event in kernels:
        rows[categories[id(event)]].append(event)
    expected_counts = {
        "routed_w13": 40,
        "routed_w2": 40,
        "routed_activation": 40,
        "routed_reduce": 40,
        "routed_quant": 40,
        "self_qkv": 30,
        "cross_q": 10,
        "attention_o": 40,
        "shared_kv_proj": 1,
        "latent_in": 40,
        "latent_out": 40,
        "shared_w13": 40,
        "shared_w2": 40,
        "shared_activation": 40,
        "scale_zero": 40,
        "router": 80,
        "lambda": 80,
        "attention_descale": 120,
        "attention_scheduler": 40,
        "attention_main": 40,
        "attention_combine": 40,
        "lm_head": 1,
    }
    for category, count in expected_counts.items():
        if len(rows[category]) != count * 12:
            raise ValueError(f"Unexpected {category} count: {len(rows[category])}")
    for category in [
        "self_qkv",
        "cross_q",
        "shared_kv_proj",
        "latent_in",
        "latent_out",
        "shared_w13",
    ]:
        if len(rows["quant_" + category]) != len(rows[category]):
            raise ValueError(f"Input quantizer adjacency not proven for {category}")
    operators = []
    coarse_us = defaultdict(float)
    for name, selected in rows.items():
        coarse_us[coarse_category(name)] += sum(e["dur"] for e in selected) / 12
        operators.append(
            {
                "category": name,
                "kernels_per_step": len(selected) / 12,
                "us_per_step": sum(e["dur"] for e in selected) / 12,
                "mean_kernel_us": statistics.mean(e["dur"] for e in selected),
                "grids": dict(Counter(str(e["args"].get("grid")) for e in selected)),
                "kernel_names": sorted({e["name"] for e in selected}),
            }
        )
    graph_correlations = {g["correlation"] for g in audited_graphs}
    outside = [e for e in kernels if e["args"]["correlation"] not in graph_correlations]
    return {
        "batch": batch,
        "trace": path.name,
        "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "annotations": expected,
        "steps": 12,
        "kernels_per_step": len(kernels) / 12,
        "kernel_sum_us_per_step": sum(e["dur"] for e in kernels) / 12,
        "median_graph_span_us": statistics.median(g["span_us"] for g in audited_graphs),
        "median_graph_busy_union_us": statistics.median(
            g["busy_union_us"] for g in audited_graphs
        ),
        "outside_model_graph_kernels_per_step": len(outside) / 12,
        "outside_model_graph_kernel_sum_us_per_step": sum(e["dur"] for e in outside)
        / 12,
        "lm_head_is_outside_model_graph": all(
            e["args"]["correlation"] not in graph_correlations for e in rows["lm_head"]
        ),
        "graphs": audited_graphs,
        "groups": [
            {
                "label": name,
                "us_per_step": duration,
                "percent": 100 * duration * 12 / sum(e["dur"] for e in kernels),
            }
            for name, duration in sorted(coarse_us.items(), key=lambda row: -row[1])
        ],
        "operators": sorted(operators, key=lambda row: -row["us_per_step"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark = json.loads(args.benchmark.read_text())
    if not benchmark["completed"] or benchmark["precision"] != "fp8":
        raise ValueError("Require completed FP8 benchmark")
    runtime = benchmark["runtime"]
    if runtime["tp"] != 1 or runtime["dp"] != 1 or runtime["enable_expert_parallel"]:
        raise ValueError("This analysis requires TP1/DP1/EP1")
    if benchmark["env"].get("VLLM_YOCO_BF16_CHAIN") != "1":
        raise ValueError("Weight reference requires BF16 router cache")
    if not runtime["latent"] or any(
        e["dtype"] != "torch.float8_e4m3fn" for e in runtime["latent"]
    ):
        raise ValueError("Weight reference requires FP8 latent projections")
    if not runtime["experts"] or any(
        e["w13_dtype"] != "torch.float8_e4m3fn"
        or e["w2_dtype"] != "torch.float8_e4m3fn"
        or e["small_m_limit"] != 16
        for e in runtime["experts"]
    ):
        raise ValueError("Require audited small-M FP8 experts")
    if not runtime["attention"] or any(
        e["kv_cache_dtype"] != "fp8" or e["fa_version"] != 4
        for e in runtime["attention"]
    ):
        raise ValueError("KV reference requires FP8 FA4")
    result = {
        "method": "logical operand bytes and kernel timings, not hardware counters",
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "model_config_sha256": hashlib.sha256(
            args.model_config.read_bytes()
        ).hexdigest(),
        "reference": weight_reference(json.loads(args.model_config.read_text())),
        "model_config": json.loads(args.model_config.read_text()),
        "benchmark_settings": {
            key: benchmark[key]
            for key in (
                "config",
                "env",
                "torch",
                "cuda",
                "warmups",
                "repeats",
                "source_revision",
                "source_bundle_sha256",
                "script_sha256",
                "implementation_sha256",
                "tokens_sha256",
                "scope",
            )
        },
        "runtime": runtime,
        "benchmark": benchmark["benchmark"],
        "profiles": [
            summarize_trace(args.trace_dir / Path(path).name, profile["batch"])
            for profile in benchmark["profiles"]
            for path in profile["files"]
            if path.endswith(".gz")
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for profile in result["profiles"]:
        print(
            f"B{profile['batch']}: {profile['kernels_per_step']:.0f} kernels/step; "
            f"graph span {profile['median_graph_span_us'] / 1000:.3f} ms; "
            f"kernel sum {profile['kernel_sum_us_per_step'] / 1000:.3f} ms"
        )


if __name__ == "__main__":
    main()
