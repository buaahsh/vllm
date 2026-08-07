#!/usr/bin/env python
"""Benchmark pure-text YOCO throughput on one GPU."""

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
from transformers import AutoTokenizer


MODEL = Path("/data/wjh/0000-6000-hf-gpu")
PROMPTS = (
    "Explain how large language model inference works in comprehensive detail.",
    "Write a detailed technical overview of mixture-of-experts transformer models.",
    "Describe the major considerations when deploying an AI model in production.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--batches", type=int, nargs="+", default=(32, 64, 128, 192))
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-json", type=Path, required=True)
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
                values = tuple(float(value.strip()) for value in line.split(","))
                with self._lock:
                    self.samples.append(values)
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval)


def render_prompt(tokenizer, prompt: str, unique_id: int) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                f"Request ID {unique_id}. You are a helpful and knowledgeable "
                "AI assistant."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        clear_thinking=True,
    )
    return rendered.replace("<|assistant|><think></think>", "<|assistant|></think>")


def build_requests(tokenizer, batch_size: int, run_id: int) -> list[str]:
    return [
        render_prompt(
            tokenizer,
            PROMPTS[index % len(PROMPTS)],
            unique_id=run_id * 1000 + index,
        )
        for index in range(batch_size)
    ]


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
    return {
        "batch_size": len(outputs),
        "elapsed_seconds": elapsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "requests_per_second": len(outputs) / elapsed,
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

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    llm_kwargs = {
        "model": str(MODEL),
        "tokenizer": str(MODEL),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "max_model_len": 8192,
        "max_num_seqs": 192,
        "max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.85,
        "quantization": None,
        "moe_backend": "triton",
        "enforce_eager": False,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "kv_sharing_fast_prefill": True,
        "attention_config": {"flash_attn_version": 4},
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
        print(f"MODEL_READY load_seconds={load_seconds:.3f}", flush=True)

        llm.generate(build_requests(tokenizer, 4, run_id=1), warmup_sampling, use_tqdm=False)
        results = []
        next_run_id = 10
        for batch_size in args.batches:
            print(f"SHAPE_WARMUP batch={batch_size}", flush=True)
            llm.generate(
                build_requests(tokenizer, batch_size, run_id=next_run_id),
                warmup_sampling,
                use_tqdm=False,
            )
            next_run_id += 1
            for repeat in range(args.repeats):
                requests = build_requests(tokenizer, batch_size, run_id=next_run_id)
                next_run_id += 1
                monitor.reset()
                start = time.perf_counter()
                outputs = llm.generate(requests, sampling, use_tqdm=False)
                elapsed = time.perf_counter() - start
                result = summarize_run(outputs, elapsed, monitor)
                result["repeat"] = repeat + 1
                results.append(result)
                print(
                    "RESULT "
                    f"batch={batch_size} repeat={repeat + 1} "
                    f"seconds={elapsed:.3f} "
                    f"out_tok_s={result['output_tokens_per_second']:.3f}",
                    flush=True,
                )

        payload = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": str(MODEL),
            "architecture": "YOCOForCausalLM",
            "dtype": "bfloat16",
            "quantization": None,
            "flash_attn_version": 4,
            "moe_backend": "triton",
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "gpu_memory_utilization": 0.85,
            "max_model_len": 8192,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 192,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "kv_sharing_fast_prefill": True,
            "enforce_eager": False,
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "gpu_physical_index": args.gpu,
            "gpu_name": torch.cuda.get_device_name(0),
            "vllm_version": vllm.__version__,
            "batches": list(args.batches),
            "output_tokens_per_request": args.output_tokens,
            "repeats": args.repeats,
            "unique_request_id": True,
            "model_load_seconds": load_seconds,
            "runs": results,
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
