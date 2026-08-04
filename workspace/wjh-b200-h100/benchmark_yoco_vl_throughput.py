#!/usr/bin/env python
"""Benchmark high-batch YOCO-VL throughput on one GPU.

This is a local diagnostic script. It deliberately creates a unique image
hash for every request so vLLM cannot reuse one vision-encoder result across
the repeated dog images in the synthetic batch.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoTokenizer


MODEL = Path("/mnt/nvme/wjh/updates_3000-hf-vl")
TOKENIZER = Path("/mnt/msranlp/yutao/hf_cache/agens_vl_tokenizer_0622")
IMAGE_PATHS = (
    Path("/root/workspace/llm-train/workspace/dog1.jpeg"),
    Path("/root/workspace/llm-train/workspace/dog2.jpeg"),
    Path("/root/workspace/llm-train/workspace/dog3.jpeg"),
    Path("/root/workspace/llm-train/workspace/dog1.jpeg"),
)
PROMPTS = (
    "Describe this dog and the surrounding scene in one concise sentence.",
    "What is the dog doing? Mention its pose and expression.",
    "What animal is shown, and what is visible in the background?",
    "What colors are most prominent in this image?",
)
SYSTEM_PROMPT = "You are a helpful and friendly AI assistant."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bf16", "fp8"), required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--batches", type=int, nargs="+", default=(32, 64))
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


class GPUMonitor:
    def __init__(self, gpu: int, interval: float = 0.2):
        self.gpu = gpu
        self.interval = interval
        self.samples: list[tuple[float, float, float]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def reset(self) -> None:
        with self._lock:
            self.samples.clear()

    def summary(self) -> dict[str, float | int]:
        with self._lock:
            samples = list(self.samples)
        if not samples:
            return {"samples": 0}
        memory = [sample[0] for sample in samples]
        utilization = [sample[1] for sample in samples]
        power = [sample[2] for sample in samples]
        return {
            "samples": len(samples),
            "gpu_memory_used_mib_peak": max(memory),
            "gpu_utilization_percent_mean": statistics.fmean(utilization),
            "gpu_utilization_percent_peak": max(utilization),
            "power_watts_mean": statistics.fmean(power),
            "power_watts_peak": max(power),
        }

    def _run(self) -> None:
        command = [
            "nvidia-smi",
            "-i",
            str(self.gpu),
            "--query-gpu=memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                line = subprocess.check_output(
                    command, text=True, stderr=subprocess.DEVNULL, timeout=2
                ).strip()
                memory, utilization, power = (
                    float(value.strip()) for value in line.split(",")
                )
                with self._lock:
                    self.samples.append((memory, utilization, power))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval)


def render_prompt(tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<image>\n{prompt}"},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        clear_thinking=True,
    )
    return rendered.replace("<|assistant|><think></think>", "<|assistant|></think>")


def make_unique_image(base: Image.Image, unique_id: int) -> Image.Image:
    array = np.asarray(base, dtype=np.uint8).copy()
    # Encode a unique ID in two corner pixels. This preserves image dimensions
    # and workload shape while avoiding multimodal hash/encoder-cache reuse.
    array[0, 0] = (
        unique_id & 0xFF,
        (unique_id >> 8) & 0xFF,
        (unique_id >> 16) & 0xFF,
    )
    array[0, 1] = (
        (unique_id * 17) & 0xFF,
        (unique_id * 31) & 0xFF,
        (unique_id * 47) & 0xFF,
    )
    return Image.fromarray(array, mode="RGB")


def build_requests(
    base_images: tuple[Image.Image, ...],
    rendered_prompts: tuple[str, ...],
    batch_size: int,
    run_id: int,
) -> list[dict[str, object]]:
    requests = []
    for index in range(batch_size):
        pattern = index % len(base_images)
        unique_id = run_id * 1000 + index + 1
        requests.append(
            {
                "prompt": rendered_prompts[pattern],
                "multi_modal_data": {
                    "image": make_unique_image(base_images[pattern], unique_id)
                },
            }
        )
    return requests


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_run(outputs, elapsed: float, monitor: GPUMonitor) -> dict[str, object]:
    input_tokens = sum(len(output.prompt_token_ids or ()) for output in outputs)
    output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    ttfts = [
        float(output.metrics.first_token_latency)
        for output in outputs
        if output.metrics is not None
    ]
    tpots = []
    for output in outputs:
        metrics = output.metrics
        num_tokens = len(output.outputs[0].token_ids)
        if metrics is not None and num_tokens > 1:
            tpots.append(
                float(metrics.last_token_ts - metrics.first_token_ts)
                / (num_tokens - 1)
            )
    batch_size = len(outputs)
    return {
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "requests_per_second": batch_size / elapsed,
        "input_tokens_per_second": input_tokens / elapsed,
        "output_tokens_per_second": output_tokens / elapsed,
        "total_tokens_per_second": (input_tokens + output_tokens) / elapsed,
        "ttft_seconds_mean": statistics.fmean(ttfts) if ttfts else 0.0,
        "ttft_seconds_p50": percentile(ttfts, 50),
        "ttft_seconds_p95": percentile(ttfts, 95),
        "tpot_ms_mean": 1000 * statistics.fmean(tpots) if tpots else 0.0,
        "tpot_ms_p95": 1000 * percentile(tpots, 95),
        **monitor.summary(),
    }


def shutdown(llm) -> None:
    engine = getattr(llm, "llm_engine", None)
    core = getattr(engine, "engine_core", None)
    if core is not None and hasattr(core, "shutdown"):
        core.shutdown(timeout=10)
    gc.collect()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), trust_remote_code=True)
    rendered_prompts = tuple(render_prompt(tokenizer, prompt) for prompt in PROMPTS)
    base_images_list = []
    for path in IMAGE_PATHS:
        with Image.open(path) as image:
            base_images_list.append(image.convert("RGB"))
    base_images = tuple(base_images_list)

    max_batch = max(args.batches)
    max_num_batched_tokens = args.max_num_batched_tokens
    quantization = "fp8_per_block" if args.mode == "fp8" else None
    moe_backend = "deep_gemm" if args.mode == "fp8" else "triton"
    llm_kwargs = {
        "model": str(MODEL),
        "tokenizer": str(TOKENIZER),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": 3072,
        "max_num_seqs": max_batch,
        "max_num_batched_tokens": max_num_batched_tokens,
        "gpu_memory_utilization": 0.90,
        "quantization": quantization,
        "moe_backend": moe_backend,
        "enforce_eager": args.enforce_eager,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "mm_processor_cache_gb": 0,
        "limit_mm_per_prompt": {"image": 1},
        "attention_config": {"flash_attn_version": 2},
        "disable_log_stats": False,
        "seed": 1,
    }
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_tokens,
        ignore_eos=True,
        detokenize=False,
    )
    warmup_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        ignore_eos=True,
        detokenize=False,
    )

    monitor = GPUMonitor(args.gpu)
    monitor.start()
    load_start = time.perf_counter()
    llm = None
    try:
        llm = LLM(**llm_kwargs)
        load_seconds = time.perf_counter() - load_start
        print(f"MODEL_READY mode={args.mode} load_seconds={load_seconds:.3f}", flush=True)

        # Small functional warmup before exercising high-batch shapes.
        llm.generate(
            build_requests(base_images, rendered_prompts, 4, run_id=1),
            warmup_sampling,
            use_tqdm=False,
        )

        all_results = []
        next_run_id = 10
        for batch_size in args.batches:
            print(f"SHAPE_WARMUP mode={args.mode} batch={batch_size}", flush=True)
            llm.generate(
                build_requests(
                    base_images, rendered_prompts, batch_size, run_id=next_run_id
                ),
                warmup_sampling,
                use_tqdm=False,
            )
            next_run_id += 1

            for repeat in range(args.repeats):
                requests = build_requests(
                    base_images, rendered_prompts, batch_size, run_id=next_run_id
                )
                next_run_id += 1
                monitor.reset()
                start = time.perf_counter()
                outputs = llm.generate(requests, sampling, use_tqdm=False)
                elapsed = time.perf_counter() - start
                result = summarize_run(outputs, elapsed, monitor)
                result["repeat"] = repeat + 1
                all_results.append(result)
                print(
                    "RESULT "
                    f"mode={args.mode} batch={batch_size} repeat={repeat + 1} "
                    f"seconds={elapsed:.3f} "
                    f"req_s={result['requests_per_second']:.3f} "
                    f"out_tok_s={result['output_tokens_per_second']:.3f}",
                    flush=True,
                )

        payload = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": args.mode,
            "quantization": quantization,
            "moe_backend": moe_backend,
            "dtype": "bfloat16",
            "vision_dtype": "bfloat16",
            "projector_dtype": "float32",
            "flash_attn_version": 2,
            "gpu_physical_index": args.gpu,
            "gpu_name": torch.cuda.get_device_name(0),
            "vllm_version": vllm.__version__,
            "model": str(MODEL),
            "tokenizer": str(TOKENIZER),
            "batches": list(args.batches),
            "output_tokens_per_request": args.output_tokens,
            "repeats": args.repeats,
            "enforce_eager": args.enforce_eager,
            "max_model_len": 3072,
            "max_num_seqs": max_batch,
            "max_num_batched_tokens": max_num_batched_tokens,
            "gpu_memory_utilization": 0.90,
            "unique_image_hash_per_request": True,
            "model_load_seconds": load_seconds,
            "runs": all_results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"WROTE {args.output_json}", flush=True)
        return 0
    finally:
        monitor.stop()
        if llm is not None:
            shutdown(llm)


if __name__ == "__main__":
    raise SystemExit(main())
