# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K8 native target+draft A/B with fixed input and sampling settings."""

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

from tools.yoco_alignment.scheduling import (
    add_scheduling_arguments,
    resolve_scheduling_options,
)

PROMPTS = [
    "What is 17 times 23? Explain the calculation step by step.",
    "Write a Python function that merges two sorted lists. Explain its complexity.",
    "请用中文解释二分查找，并给出一个带边界处理的 Python 示例。",
    "Explain how a hash table works and when its worst-case lookup is slow.",
    "Describe how to debug a deadlock between two threads using concrete examples.",
    "Design an LRU cache in Python and explain the data structures you use.",
    (
        "Explain attention key-value caching and its memory cost "
        "during autoregressive decoding."
    ),
    "Write a clear proof that the sum of the first n odd integers is n squared.",
]
CONTINUATIONS = [
    (
        "17 times 23 is 391. We can calculate 17 times 20, which is 340, "
        "and 17 times 3, which is 51. Adding these gives 340 + 51 = 391."
    ),
    (
        "Maintain an index into each list. Repeatedly append the smaller current "
        "element and advance its index. Finally append the unconsumed suffix. "
        "This takes O(n + m) time and O(n + m) space for the returned list."
    ),
    (
        "二分查找要求数组有序。维护左闭右开的区间 [left, right)，每次检查中点。"
        "如果中点元素小于目标，就令 left = mid + 1；否则令 right = mid。"
        "循环结束后检查 left 是否在数组内且对应元素等于目标。时间复杂度为 O(log n)。"
    ),
    (
        "A hash function maps a key to a bucket. Collisions can be handled with "
        "chaining or open addressing. With a suitable hash function and bounded "
        "load factor, lookup takes expected constant time. If many keys collide, "
        "a lookup can take linear time in the number of entries."
    ),
]
ROOT = Path(__file__).resolve().parent
TARGET = "/mnt/pvc/shaohanh/exp/agens/30A3B/merged/balanced-b040-c035-d025-hf"
DRAFT = "/mnt/pvc/lidong1/opd-fast-rollout-20261016/draft-parent-30000"


def build_benchmark_prompts(count):
    """Keep the original panel while providing distinct requests above C8."""
    return [
        PROMPTS[index]
        if index < len(PROMPTS)
        else (
            f"{PROMPTS[index % len(PROMPTS)]}\n"
            f"Include a worked example numbered {index + 1}."
        )
        for index in range(count)
    ]


def validate_fixed_outputs(outputs, concurrency, token_count):
    if len(outputs) != concurrency:
        raise RuntimeError(f"Expected {concurrency} requests, got {len(outputs)}")
    if any(len(o.outputs[0].token_ids) != token_count for o in outputs):
        raise RuntimeError("A request did not complete its fixed output budget")


class MegaMoEBenchmarkWorker:
    def ab_profile_start(self, annotate_scheduler=False):
        import torch

        self._ab_profile = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        )
        self._ab_profile.__enter__()
        self._ab_profile_originals = []
        if annotate_scheduler:
            from vllm.v1.core.sched.scheduler import Scheduler

            for name in ("schedule", "update_from_output"):
                original = getattr(Scheduler, name)

                def annotated(*args, original=original, name=name, **kwargs):
                    with torch.profiler.record_function(f"yoco_scheduler::{name}"):
                        return original(*args, **kwargs)

                self._ab_profile_originals.append((Scheduler, name, original))
                setattr(Scheduler, name, annotated)

    def ab_profile_stop(self, trace_dir=None):
        for owner, name, original in self._ab_profile_originals:
            setattr(owner, name, original)
        self._ab_profile_originals = []
        profiler = self._ab_profile
        self._ab_profile = None
        profiler.__exit__(None, None, None)
        trace_path = None
        if trace_dir is not None:
            directory = Path(trace_dir)
            directory.mkdir(parents=True, exist_ok=True)
            trace_path = directory / f"worker-{os.getpid()}-{time.time_ns()}.json"
            profiler.export_chrome_trace(str(trace_path))
        return dict(
            trace_path=str(trace_path) if trace_path is not None else None,
            cpu=profiler.key_averages().table(
                sort_by="self_cpu_time_total", row_limit=40
            ),
            gpu=profiler.key_averages().table(
                sort_by="self_device_time_total", row_limit=40
            ),
        )

    def ab_bind_graphs(self):
        self._ab_graph_counts = {"target": {}, "draft": {}}
        for name, manager in (
            ("target", self.model_runner.cudagraph_manager),
            ("draft", self.model_runner.speculator.shared_kv_cudagraph_manager),
        ):
            original = manager.run_fullgraph
            counts = self._ab_graph_counts[name]

            def replay(desc, original=original, counts=counts):
                key = f"{desc.num_reqs}x{desc.uniform_token_count}/{desc.num_tokens}"
                counts[key] = counts.get(key, 0) + 1
                return original(desc)

            manager.run_fullgraph = replay
        return self.ab_status()

    def ab_status(self):
        import torch

        from vllm.model_executor.layers.yoco_moe import YOCOMoE
        from vllm.utils.deep_gemm import _import_deep_gemm
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

        runner = self.model_runner
        layers = [
            (name, block.experts.routed_experts)
            for name, block in runner.model.named_modules()
            if isinstance(block, YOCOMoE)
        ]
        cores = [getattr(layer, "_yoco_mega_moe", None) for _, layer in layers]
        cores = [core for core in cores if core is not None]
        workspaces = {id(core.workspace): core.workspace for core in cores}
        dg = _import_deep_gemm()
        config = self.vllm_config
        scheduler_cls = config.scheduler_config.get_scheduler_cls()
        owner = runner.speculator._cache_owner
        status = dict(
            backend="mega" if cores else "current",
            layers=[name for name, _ in layers],
            mega_layers=len(cores),
            workspace_count=len(workspaces),
            shared_workspace_bytes=sum(
                w.buffer.buffer.nbytes for w in workspaces.values()
            ),
            torch=str(torch.__version__),
            deep_gemm=dg.__version__,
            deep_gemm_file=dg.__file__,
            allocated=torch.accelerator.memory_allocated(),
            draft_type=type(runner.speculator).__name__,
            runner=type(runner).__module__,
            fast_prefill=runner.model.fast_prefill_enabled,
            aux_hidden_layers=list(runner.model.model.aux_hidden_state_layers),
            draft_logits_dtype=str(runner.speculator.draft_logits.dtype),
            additional_config=self.vllm_config.additional_config,
            async_scheduling=config.scheduler_config.async_scheduling,
            scheduler_class=f"{scheduler_cls.__module__}.{scheduler_cls.__qualname__}",
            max_concurrent_batches=config.max_concurrent_batches,
            speculative_tokens=runner.speculator.num_speculative_steps,
            executor_backend=str(config.parallel_config.distributed_executor_backend),
            worker_pid=os.getpid(),
            engine_multiprocessing=os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
            max_num_seqs=config.scheduler_config.max_num_seqs,
            max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
            kv_cache_memory_bytes=config.cache_config.kv_cache_memory_bytes,
            kv_cache_dtype=config.cache_config.cache_dtype,
            shared_kv_cache={
                "storage_dtype": str(owner.kv_cache.dtype),
                "shape": list(owner.kv_cache.shape),
                "strides": list(owner.kv_cache.stride()),
                "elements": owner.kv_cache.numel(),
                "bytes": owner.kv_cache.numel() * owner.kv_cache.element_size(),
                "k_scale": owner._k_scale.item(),
                "v_scale": owner._v_scale.item(),
            },
            shared_kv_attention_fa_version=get_flash_attn_version(head_size=128),
            dispatch=asdict(layers[0][1]._yoco_moe_dispatch),
            mega_capture_decisions=layers[0][1]._yoco_moe_capture_backends,
            triton_max_tokens=getattr(
                layers[0][1].quant_method.moe_kernel.fused_experts,
                "_yoco_fp8_decode_limit",
                None,
            ),
        )
        status["graphs"] = {}
        for name, manager, width in (
            ("target", runner.cudagraph_manager, 9),
            ("draft", runner.speculator.shared_kv_cudagraph_manager, 8),
        ):
            requests = config.scheduler_config.max_num_seqs
            desc = manager.dispatch(
                requests, requests * width, width, 0, max_query_len=width
            )
            status["graphs"][name] = dict(
                widest_mode=str(desc.cg_mode),
                widest_captured=desc in manager.graphs,
                captured_tokens=manager.captured_token_counts(),
                replays=getattr(self, "_ab_graph_counts", {}).get(name, {}),
            )
        status["target_graph_backends"] = {
            f"{desc.num_reqs}x{desc.uniform_token_count}/{desc.num_tokens}": (
                "mega"
                if status["mega_capture_decisions"].get(desc.num_tokens, False)
                else (
                    "triton"
                    if 0 < desc.num_tokens <= (status["triton_max_tokens"] or 0)
                    else "deep_gemm"
                )
            )
            for desc in runner.cudagraph_manager.graphs
        }
        if cores:
            assert all(
                layer._yoco_moe_capture_backends == status["mega_capture_decisions"]
                for _, layer in layers
            )
            assert all(
                desc.num_tokens in status["mega_capture_decisions"]
                for desc in runner.cudagraph_manager.graphs
            )
        return status

    def ab_synchronize(self):
        import torch

        torch.accelerator.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default=TARGET)
    p.add_argument("--draft", default=DRAFT)
    p.add_argument("--backend", choices=["current", "mega", "hybrid"], required=True)
    p.add_argument("--dispatch-config", type=Path)
    p.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--prompt-tokens", type=int, default=0)
    p.add_argument("--quality-tokens", type=int, default=512)
    p.add_argument("--extra-prompt-tokens", type=int, default=0)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--fast-prefill", action="store_true")
    p.add_argument("--profile-prefill", action="store_true")
    add_scheduling_arguments(p)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--token-counts", type=int, nargs="+")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--kv-cache-gib", type=float, default=8)
    p.add_argument(
        "--kv-cache-dtype",
        choices=["auto", "bfloat16", "fp8", "fp8_e4m3"],
        default="auto",
    )
    p.add_argument(
        "--cudagraph-capture-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 9, 16, 18, 32, 36, 64, 72],
    )
    args = p.parse_args()
    try:
        scheduling = resolve_scheduling_options(args)
    except ValueError as exc:
        p.error(str(exc))
    scheduling.apply_environment()
    from vllm import LLM, SamplingParams

    if (
        min(args.token_counts or [args.tokens]) < 1
        or args.trials < 1
        or args.quality_tokens < 1
        or args.max_num_seqs < 1
        or any(not 1 <= c <= args.max_num_seqs for c in args.concurrencies)
        or args.max_num_batched_tokens < 9 * args.max_num_seqs
        or not math.isfinite(args.kv_cache_gib)
        or args.kv_cache_gib <= 0
        or min(args.cudagraph_capture_sizes) < 1
        or max(args.cudagraph_capture_sizes) < 9 * args.max_num_seqs
        or min(args.prompt_tokens, args.extra_prompt_tokens) < 0
    ):
        p.error(
            "Require positive token/trial/KV counts, concurrency <= max-num-seqs, "
            "nonnegative prompt-tokens, and batch/graph capacity >= 9*max-num-seqs"
        )
    token_counts = args.token_counts or [args.tokens]
    if (
        max(token_counts) + max(args.prompt_tokens, args.extra_prompt_tokens)
        > args.max_model_len
    ):
        p.error("Prompt and output must fit --max-model-len")
    args.output.mkdir(exist_ok=False, parents=True)

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False))

    def event(stage, **fields):
        r = dict(stage=stage, unix=time.time(), **fields)
        save("progress.json", r)
        print(json.dumps(r), flush=True)

    config = {
        "model": args.model,
        "dtype": "bfloat16",
        "quantization": "fp8_per_block",
        "tensor_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": 0.75,
        "kv_cache_memory_bytes": int(args.kv_cache_gib * 1024**3),
        "kv_cache_dtype": args.kv_cache_dtype,
        "enable_prefix_caching": False,
        "kv_sharing_fast_prefill": args.fast_prefill,
        "disable_log_stats": False,
        "worker_extension_cls": (
            "tools.yoco_alignment.benchmark_mega_moe.MegaMoEBenchmarkWorker"
        ),
        "speculative_config": {
            "model": args.draft,
            "method": "dspark",
            "num_speculative_tokens": 8,
            "draft_sample_method": "probabilistic",
            "rejection_sample_method": "standard",
        },
        "compilation_config": {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "max_cudagraph_capture_size": max(args.cudagraph_capture_sizes),
            "cudagraph_capture_sizes": sorted(set(args.cudagraph_capture_sizes)),
        },
        "additional_config": {
            "yoco_execution_mode": "fast",
            "yoco_mega_moe": "off" if args.backend == "current" else "fp8_group32",
        },
    }
    if args.backend == "mega":
        config["additional_config"]["yoco_moe_dispatch"] = {
            "mode": "manual",
            "mega_min_tokens": 1,
            "mega_prefill": True,
        }
    config.update(scheduling.llm_kwargs())
    if args.dispatch_config:
        import yaml

        extra = yaml.safe_load(args.dispatch_config.read_text())["additional-config"]
        config["additional_config"].update(extra)
    if args.pilot:
        config["enforce_eager"] = True
    prompt_texts = build_benchmark_prompts(max(8, max(args.concurrencies)))
    save(
        "manifest.json",
        {
            "config": config,
            "backend": args.backend,
            "prompts": prompt_texts,
            "prompt_sha256": hashlib.sha256(
                json.dumps(prompt_texts, ensure_ascii=False).encode()
            ).hexdigest(),
            "draft_config": json.loads((Path(args.draft) / "config.json").read_text()),
            "script_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in [Path(__file__).name]
            },
            "scope": (
                "native target+draft K8; fixed token budget and numerical "
                "trajectories; no task-quality claim"
            ),
            "prompt_tokens": args.prompt_tokens,
            "extra_prompt_tokens": args.extra_prompt_tokens,
            "quality_tokens_per_prompt": args.quality_tokens,
            "concurrencies": args.concurrencies,
            "token_counts": token_counts,
            "synchronous_engine": args.synchronous_engine,
            "scheduling": asdict(scheduling),
            "runtime_env": {
                key: os.environ.get(key)
                for key in (
                    "VLLM_USE_V2_MODEL_RUNNER",
                    "VLLM_ENABLE_V1_MULTIPROCESSING",
                    "VLLM_YOCO_BF16_SAMPLING",
                )
            },
        },
    )
    event("loading")
    begin = time.perf_counter()
    llm = LLM(**config)
    event("loaded", startup_s=time.perf_counter() - begin)
    status = llm.collective_rpc("ab_bind_graphs")
    save("worker.json", status)
    for worker in status:
        scheduling.validate_runtime(worker)
    client_cls = type(llm.llm_engine.engine_core)
    save(
        "scheduling.json",
        {
            "requested": asdict(scheduling),
            "engine_client": f"{client_cls.__module__}.{client_cls.__qualname__}",
            "frontend_pid": os.getpid(),
            "workers": status,
        },
    )
    assert status[0]["draft_type"] == "SharedKVSpeculator", status
    assert status[0]["runner"] == "vllm.v1.worker.gpu.model_runner", status
    assert status[0]["fast_prefill"] == args.fast_prefill, status
    assert status[0]["draft_logits_dtype"] == "torch.float32", status
    assert len(status[0]["layers"]) == 20
    expect_mega = status[0]["dispatch"]["mega_min_tokens"] is not None
    assert status[0]["mega_layers"] == (20 if expect_mega else 0)
    assert status[0]["workspace_count"] == (1 if expect_mega else 0)
    if args.backend == "hybrid" and not args.pilot and not args.dispatch_config:
        assert status[0]["dispatch"]["profile"] == "b200-l3-fp8-tp1-k8", status
    if not args.pilot:
        assert all(g["widest_captured"] for g in status[0]["graphs"].values()), status
    tokenizer = llm.get_tokenizer()
    prompts = [
        {
            "prompt_token_ids": tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        }
        for text in prompt_texts
    ]
    if args.prompt_tokens:
        filler = tokenizer.encode(
            "This technical note describes cache reuse, model execution, "
            "request scheduling, and numerical accuracy. ",
            add_special_tokens=False,
        )
        for prompt in prompts:
            short = prompt["prompt_token_ids"]
            needed = args.prompt_tokens - len(short)
            if needed < 0:
                p.error("prompt-tokens is shorter than an original chat prompt")
            prompt["prompt_token_ids"] = (filler * (needed // len(filler) + 1))[
                :needed
            ] + short
    profiles = [("short", prompts)]
    if args.extra_prompt_tokens:
        filler = tokenizer.encode(
            "This technical note describes cache reuse, model execution, "
            "request scheduling, and numerical accuracy. ",
            add_special_tokens=False,
        )
        longer = []
        for prompt in prompts:
            short = prompt["prompt_token_ids"]
            needed = args.extra_prompt_tokens - len(short)
            if needed < 0:
                p.error("extra-prompt-tokens must exceed the original prompt length")
            longer.append(
                {
                    "prompt_token_ids": (filler * (needed // len(filler) + 1))[:needed]
                    + short
                }
            )
        profiles.append((f"context{args.extra_prompt_tokens}", longer))
    save("prompt-ids.json", dict(profiles))
    if args.profile_prefill:
        warmup_params = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        llm.generate(prompts[:1], warmup_params, use_tqdm=False)
        llm.generate(prompts[:1], warmup_params, use_tqdm=False)
        llm.collective_rpc("ab_profile_start")
        llm.generate(prompts[:1], warmup_params, use_tqdm=False)
        save("prefill-profile.json", llm.collective_rpc("ab_profile_stop"))

    # Same input prefixes expose end-of-prompt full-model logprob differences.
    quality = llm.generate(
        prompts[:4],
        SamplingParams(
            temperature=1.0, max_tokens=1, seed=20261020, logprobs=20, ignore_eos=True
        ),
        use_tqdm=False,
    )
    save(
        "first-token.json",
        [
            {
                "token_ids": list(o.outputs[0].token_ids),
                "logprobs": [
                    {str(k): asdict(v) for k, v in d.items()}
                    for d in o.outputs[0].logprobs
                ],
            }
            for o in quality
        ],
    )
    teacher_inputs = [
        {
            "prompt_token_ids": prompt["prompt_token_ids"]
            + (
                tokenizer.encode(text, add_special_tokens=False)
                * (args.quality_tokens + 1)
            )[: args.quality_tokens]
        }
        for prompt, text in zip(prompts, CONTINUATIONS)
    ]
    teacher = llm.generate(
        teacher_inputs,
        SamplingParams(
            temperature=1.0,
            max_tokens=1,
            seed=20261020,
            ignore_eos=True,
            prompt_logprobs=0,
        ),
        use_tqdm=False,
    )
    teacher_rows = []
    for prompt, inp, output in zip(prompts, teacher_inputs, teacher):
        start = len(prompt["prompt_token_ids"])
        tokens = inp["prompt_token_ids"]
        values = [
            float(output.prompt_logprobs[j][tokens[j]].logprob)
            for j in range(start, len(tokens))
        ]
        assert values and all(math.isfinite(v) for v in values)
        teacher_rows.append(
            {
                "input_ids": tokens,
                "continuation_start": start,
                "logprobs": values,
                "mean_nll": -sum(values) / len(values),
            }
        )
    save("teacher-forced.json", teacher_rows)
    rows = []
    for profile, workload, token_count in (
        (profile, workload, token_count)
        for profile, workload in profiles
        for token_count in token_counts
    ):
        for concurrency in args.concurrencies:
            count = 1 if args.pilot else args.trials + 2
            for trial in range(count):
                scored = trial - 2
                seed = 20261020 + (max(scored, 0) if trial >= 2 else trial)
                params = SamplingParams(
                    temperature=1.0,
                    top_p=1.0,
                    top_k=-1,
                    max_tokens=16 if args.pilot else token_count,
                    seed=seed,
                    ignore_eos=True,
                )
                before = [asdict(m) for m in llm.get_metrics()]
                llm.collective_rpc("ab_synchronize")
                begin = time.perf_counter()
                outputs = llm.generate(workload[:concurrency], params, use_tqdm=False)
                llm.collective_rpc("ab_synchronize")
                elapsed = time.perf_counter() - begin
                validate_fixed_outputs(outputs, concurrency, params.max_tokens)
                after = [asdict(m) for m in llm.get_metrics()]
                tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
                row = {
                    "profile": profile,
                    "concurrency": concurrency,
                    "output_tokens_per_request": params.max_tokens,
                    "trial": trial,
                    "warmup": trial < 2,
                    "seed": seed,
                    "elapsed_s": elapsed,
                    "tokens": tokens,
                    "tokens_per_s": tokens / elapsed,
                    "metrics_before": before,
                    "metrics_after": after,
                    "output_ids": [list(o.outputs[0].token_ids) for o in outputs],
                    "text": [o.outputs[0].text for o in outputs],
                    "request_metrics": [
                        asdict(o.metrics) if o.metrics is not None else None
                        for o in outputs
                    ],
                }
                rows.append(row)
                save("trials.json", rows)
                event(
                    "trial",
                    profile=profile,
                    concurrency=concurrency,
                    output_tokens_per_request=params.max_tokens,
                    trial=trial,
                    tokens=tokens,
                    elapsed_s=elapsed,
                    tokens_per_s=tokens / elapsed,
                )
    final_status = llm.collective_rpc("ab_status")
    for worker in final_status:
        scheduling.validate_runtime(worker)
    save(
        "complete.json",
        {
            "status": "PASS",
            "rows": len(rows),
            "worker": final_status,
        },
    )
    event("complete")
    llm.llm_engine.engine_core.shutdown(timeout=60)


if __name__ == "__main__":
    main()
