# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared single-GPU YOCO Fast BF16/FP8 warm-prefix decode measurement.

See docs/yoco/performance/FAST_FP8_DECODE_BENCHMARK.md for the timing scope.
The two precision entrypoints share the workload, timer and runtime audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

DEFAULT_ENV = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "MAX_JOBS": "4",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_DP_SIZE": "1",
    "VLLM_DP_RANK": "0",
    "VLLM_DP_RANK_LOCAL": "0",
    "VLLM_USE_DEEP_GEMM": "1",
    "VLLM_USE_DEEP_GEMM_E8M0": "1",
    "VLLM_YOCO_FP8_ATTENTION_FUSION": "1",
    "VLLM_YOCO_FP8_LATENT_NORM_FUSION": "0",
    "VLLM_YOCO_BF16_RESIDUAL": "0",
    "VLLM_YOCO_BF16_CHAIN": "1",
    "VLLM_YOCO_BF16_REDUCTIONS": "0",
    "VLLM_YOCO_BF16_SAMPLING": "0",
    "VLLM_YOCO_FP8_W2_TUNING": "1",
}
DEFAULT_TOKENS = Path(__file__).with_name("data") / "fast_fp8_decode_tokens.json"


def runtime_manifest(worker: Any) -> dict[str, Any]:
    """Read actual loaded weights and backend selection in the GPU worker."""
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.platforms import current_platform

    config = worker.vllm_config
    parallel = config.parallel_config
    experts, latent, small_fp8_linears = [], [], []
    for name, layer in worker.model_runner.get_model().named_modules():
        if isinstance(layer, RoutedExperts):
            impl = layer.quant_method.moe_kernel.fused_experts
            experts.append(
                {
                    "name": name,
                    "local_experts": layer.local_num_experts,
                    "w13_dtype": str(layer.w13_weight.dtype),
                    "w2_dtype": str(layer.w2_weight.dtype),
                    "backend": type(impl).__name__,
                    "small_m_limit": getattr(impl, "_yoco_fp8_decode_limit", None),
                }
            )
        if name.endswith(("fc1_latent_proj", "fc2_latent_proj")):
            latent.append({"name": name, "dtype": str(layer.weight.dtype)})
        if name.endswith(
            (
                "fc1_latent_proj",
                "fc2_latent_proj",
                "shared_experts.gate_up_proj",
                "shared_experts.down_proj",
            )
        ):
            kernel = getattr(getattr(layer, "quant_method", None), "fp8_linear", None)
            small_fp8_linears.append(
                {
                    "name": name,
                    "backend": type(kernel).__name__,
                    "weight_shape": list(layer.weight.shape),
                }
            )
    attention = [
        {
            "name": name,
            "fa_version": layer.impl.vllm_flash_attn_version,
            "kv_cache_dtype": layer.kv_cache_dtype,
            "cache_tensor_dtype": str(layer.kv_cache.dtype),
            "k_scale": layer._k_scale.item(),
            "v_scale": layer._v_scale.item(),
        }
        for name, layer in config.compilation_config.static_forward_context.items()
        if isinstance(layer, Attention)
    ]
    deep_gemm = next(
        (
            sys.modules[n]
            for n in ("deep_gemm", "vllm.third_party.deep_gemm")
            if n in sys.modules
        ),
        None,
    )
    native_library = getattr(getattr(deep_gemm, "_C", None), "__file__", None)
    return {
        "gpu": current_platform.get_device_name(),
        "gpu_uuid": current_platform.get_device_uuid(),
        "tp": parallel.tensor_parallel_size,
        "dp": parallel.data_parallel_size,
        "enable_expert_parallel": parallel.enable_expert_parallel,
        "cudagraph_mode": str(config.compilation_config.cudagraph_mode),
        "runner": type(worker.model_runner).__name__,
        "deep_gemm": (str(Path(deep_gemm.__file__).resolve()) if deep_gemm else None),
        "deep_gemm_library": (
            str(Path(native_library).resolve()) if native_library else None
        ),
        "deep_gemm_library_sha256": (
            hashlib.sha256(Path(native_library).read_bytes()).hexdigest()
            if native_library
            else None
        ),
        "experts": experts,
        "latent": latent,
        "small_fp8_linears": small_fp8_linears,
        "attention": attention,
    }


class FastDecodeWorker:
    """Named worker RPC avoids enabling arbitrary-object serialization."""

    def fast_decode_manifest(self) -> dict[str, Any]:
        return runtime_manifest(self)


def generate_batch(llm: Any, prompts: Any, params: Any, profile: str | None = None):
    """Time resume-to-drain, excluding whole-batch enqueue and profiler setup."""
    core = llm.llm_engine.engine_core
    core.call_utility("pause_scheduler", "keep", False)
    profiling = False
    try:
        try:
            llm.enqueue(prompts, params, use_tqdm=False)
            if profile:
                llm.start_profile(profile)
                profiling = True
            started = time.perf_counter()
        finally:
            core.call_utility("resume_scheduler")
        outputs = llm.wait_for_completion(use_tqdm=False)
        return outputs, time.perf_counter() - started
    finally:
        if profiling:
            llm.stop_profile()


def parse_args(precision: str = "fp8") -> argparse.Namespace:
    description = f"Single-GPU YOCO Fast {precision.upper()} warm-prefix decode"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", required=True, help="YOCO HF checkpoint directory")
    parser.add_argument("--gpu", default="0", help="One CUDA device index or UUID")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--tokens-json", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument(
        "--output", type=Path, required=True, help="New result JSON path"
    )
    parser.add_argument(
        "--profile-dir", type=Path, help="Also profile B1/B8 after timing"
    )
    parser.add_argument(
        "--container-compat",
        action="store_true",
        help="Use the existing metadata/torchvision workaround for the lab image",
    )
    args = parser.parse_args()
    if any(b not in [1, 2, 4, 8, 16] for b in args.batches):
        parser.error("--batches must be drawn from 1 2 4 8 16")
    if min(args.prompt_tokens, args.output_tokens, args.warmups, args.repeats) < 1:
        parser.error("Lengths, warmups, and repeats must be positive")
    if args.stride < 1 or args.prompt_tokens + args.output_tokens > 8192:
        parser.error("Require stride > 0 and prompt + output <= 8192")
    if "," in args.gpu or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Run as one ordinary process on one GPU, without torchrun")
    if args.output.exists():
        parser.error(f"Result already exists: {args.output}")
    return args


def main(
    *, precision: Literal["fp8", "bf16"] = "fp8", entrypoint: Path | None = None
) -> None:
    args = parse_args(precision)
    if precision not in ("fp8", "bf16"):
        raise ValueError(f"Unsupported precision: {precision}")
    use_fp8 = precision == "fp8"
    env = dict(DEFAULT_ENV)
    # Compilation concurrency affects cold startup, outside the timed runs.
    env["MAX_JOBS"] = os.environ.get("MAX_JOBS", env["MAX_JOBS"])
    env["VLLM_YOCO_FP8_SMALL_M"] = os.environ.get("VLLM_YOCO_FP8_SMALL_M", "1")
    if not use_fp8:
        env.update(
            VLLM_YOCO_FP8_ATTENTION_FUSION="0",
            VLLM_YOCO_FP8_LATENT_NORM_FUSION="0",
            VLLM_YOCO_FP8_W2_TUNING="0",
            VLLM_YOCO_FP8_SMALL_M="0",
        )
    # Set before importing Torch/vLLM; this is the measured single-GPU preset.
    os.environ.update(env)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", "")]
    )
    if args.container_compat:
        from logprob_kl import (
            _disable_transformers_torchvision,
            _patch_local_vllm_metadata,
        )

        _disable_transformers_torchvision()
        _patch_local_vllm_metadata()

    import torch

    from vllm import LLM, SamplingParams

    data = json.loads(args.tokens_json.read_text())
    tokens = data["tokens"]
    required = args.stride * (max(args.batches) - 1) + args.prompt_tokens
    if len(tokens) < required or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError(f"Need at least {required} nonnegative integer token IDs")
    config: dict[str, Any] = {
        "model": args.model,
        "additional_config": {"yoco_execution_mode": "fast"},
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "enable_expert_parallel": False,
        "gpu_memory_utilization": 0.65,
        "max_model_len": 8192,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 16,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "kv_sharing_fast_prefill": True,
        "quantization": "fp8_per_block" if use_fp8 else None,
        "kv_cache_dtype": "fp8" if use_fp8 else "auto",
        "attention_config": {"backend": "FLASH_ATTN", "flash_attn_version": 4},
        "kernel_config": {"enable_flashinfer_autotune": False},
        "compilation_config": {
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "custom_ops": [],
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
        },
        "logprobs_mode": "raw_logprobs",
        "seed": 918,
        "worker_extension_cls": "benchmark_fast_decode.FastDecodeWorker",
    }
    if args.profile_dir:
        args.profile_dir = args.profile_dir.resolve()
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        config["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.profile_dir),
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": True,
            "ignore_frontend": True,
            "delay_iterations": 4,
            "max_iterations": 12,
        }
    repo = Path(__file__).resolve().parents[2]
    try:
        revision = (
            subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                text=True,
                capture_output=True,
            ).stdout.strip()
            or None
        )
    except FileNotFoundError:
        revision = None  # Source-only runtime bundles may not include Git.
    result: dict[str, Any] = {
        "completed": False,
        "precision": precision,
        "config": config,
        "env": {**env, "CUDA_VISIBLE_DEVICES": args.gpu},
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "source_revision": revision,
        "source_bundle_sha256": os.environ.get("YOCO_BENCH_SOURCE_BUNDLE_SHA256"),
        "script_sha256": hashlib.sha256(
            (entrypoint or Path(__file__)).read_bytes()
        ).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "tokens_sha256": hashlib.sha256(args.tokens_json.read_bytes()).hexdigest(),
        "scope": "Warm-prefix offline generation; resume to drained outputs",
        "warmups": args.warmups,
        "repeats": args.repeats,
        "benchmark": [],
        "profiles": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    llm = LLM(**config)
    try:
        manifest = llm.collective_rpc("fast_decode_manifest")[0]
        result["runtime"] = manifest
        assert manifest["tp"] == manifest["dp"] == 1
        assert not manifest["enable_expert_parallel"]
        expected_weight = "torch.float8_e4m3fn" if use_fp8 else "torch.bfloat16"
        assert manifest["experts"] and all(
            e["w13_dtype"] == e["w2_dtype"] == expected_weight
            for e in manifest["experts"]
        )
        assert manifest["latent"] and all(
            e["dtype"] == expected_weight for e in manifest["latent"]
        )
        assert manifest["attention"] and all(
            a["fa_version"] == 4
            and (
                a["kv_cache_dtype"] == "fp8"
                if use_fp8
                else a["kv_cache_dtype"] == "auto"
                and a["cache_tensor_dtype"] == "torch.bfloat16"
            )
            for a in manifest["attention"]
        )
        save()

        params = SamplingParams(
            temperature=0,
            ignore_eos=True,
            min_tokens=args.output_tokens,
            max_tokens=args.output_tokens,
        )
        print(f"Fast {precision.upper()} · TP1/DP1/EP1 · FA4")
        print("Batch   Total tok/s   Mean step ms   Cached prompt tokens")
        for batch in args.batches:
            prompts = [
                {
                    "prompt_token_ids": tokens[
                        i * args.stride : i * args.stride + args.prompt_tokens
                    ]
                }
                for i in range(batch)
            ]
            for _ in range(args.warmups):
                generate_batch(llm, prompts, params)
            elapsed, cache_hits = [], []
            for _ in range(args.repeats):
                outputs, seconds = generate_batch(llm, prompts, params)
                assert len(outputs) == batch
                assert all(
                    len(o.outputs[0].token_ids) == args.output_tokens for o in outputs
                )
                hits = [o.num_cached_tokens for o in outputs]
                assert all(hit is not None and hit > 0 for hit in hits), hits
                elapsed.append(seconds)
                cache_hits.append(hits)
            median = statistics.median(elapsed)
            row = {
                "batch": batch,
                "prompt_tokens": args.prompt_tokens,
                "output_tokens_per_request": args.output_tokens,
                "seconds": elapsed,
                "median_seconds": median,
                "total_tokens_per_second": batch * args.output_tokens / median,
                "mean_output_step_ms": 1000 * median / args.output_tokens,
                "cached_prompt_tokens": cache_hits,
                "last_generated_token_ids": [o.outputs[0].token_ids for o in outputs],
            }
            result["benchmark"].append(row)
            save()
            print(
                f"{batch:5d} {row['total_tokens_per_second']:13.2f} "
                f"{row['mean_output_step_ms']:14.3f}   {cache_hits[-1]}",
                flush=True,
            )
            if args.profile_dir and batch in [1, 8]:
                profile_params = SamplingParams(
                    temperature=0,
                    ignore_eos=True,
                    min_tokens=32,
                    max_tokens=32,
                )
                for _ in range(3):
                    generate_batch(llm, prompts, profile_params)
                previous = set(args.profile_dir.glob("*.gz"))
                generate_batch(
                    llm, prompts, profile_params, profile=f"fast-{precision}-b{batch}"
                )
                result["profiles"].append(
                    {
                        "batch": batch,
                        "files": [
                            str(p)
                            for p in sorted(
                                set(args.profile_dir.glob("*.gz")) - previous
                            )
                        ],
                    }
                )
                save()
        result["completed"] = True
        save()
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
