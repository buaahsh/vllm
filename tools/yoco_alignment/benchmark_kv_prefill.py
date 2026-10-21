# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native, non-speculative YOCO fast-prefill data-movement validation."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from tools.yoco_alignment.benchmark_mega_moe import PROMPTS, TARGET


class PrefillBenchmarkWorker:
    def prefill_status(self):
        model = self.model_runner.get_model()
        core = model.model
        return dict(
            runner=type(self.model_runner).__module__,
            fast_prefill=model.fast_prefill_enabled,
            split_blocks=core.self_block is not None and core.cross_block is not None,
            hidden_buffer=(
                list(core.fp_hidden_states.shape),
                str(core.fp_hidden_states.dtype),
            ),
            key_buffer=(list(core.fp_yoco_key.shape), str(core.fp_yoco_key.dtype)),
            additional_config=self.vllm_config.additional_config,
        )

    def prefill_profile_start(self):
        import torch

        self._prefill_profile = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        self._prefill_profile.__enter__()

    def prefill_profile_stop(self):
        profile = self._prefill_profile
        profile.__exit__(None, None, None)
        names = [e.name for e in profile.events() if "gather_prefill" in e.name]
        self._prefill_profile = None
        return names

    def prefill_synchronize(self):
        import torch

        torch.accelerator.synchronize()


def main():
    from vllm import LLM, SamplingParams

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", choices=["baseline", "fused"], required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[1024, 8192, 32768])
    p.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4])
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--tokens", type=int, default=64)
    args = p.parse_args()
    if min(*args.lengths, args.trials, args.tokens) < 1 or any(
        c not in range(1, 5) for c in args.concurrencies
    ):
        p.error("Require positive lengths/trials/tokens and C in [1,4]")
    args.output.mkdir(exist_ok=False, parents=True)

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False))

    def event(stage, **fields):
        record = dict(stage=stage, unix=time.time(), **fields)
        save("progress.json", record)
        print(json.dumps(record), flush=True)

    config = dict(
        model=TARGET,
        dtype="bfloat16",
        quantization="fp8_per_block",
        tensor_parallel_size=1,
        max_model_len=max(args.lengths) + args.tokens + 512,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        kv_cache_memory_bytes=8 * 1024**3,
        gpu_memory_utilization=0.75,
        enable_prefix_caching=False,
        kv_sharing_fast_prefill=True,
        async_scheduling=False,
        disable_log_stats=False,
        additional_config={"yoco_execution_mode": "fast", "yoco_mega_moe": "off"},
        compilation_config=dict(
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32],
            max_cudagraph_capture_size=32,
        ),
        worker_extension_cls=(
            "tools.yoco_alignment.benchmark_kv_prefill.PrefillBenchmarkWorker"
        ),
    )
    save(
        "manifest.json",
        dict(
            config=config,
            variant=args.variant,
            lengths=args.lengths,
            concurrencies=args.concurrencies,
            trials=args.trials,
            tokens=args.tokens,
        ),
    )
    event("loading")
    llm = LLM(**config)
    status = llm.collective_rpc("prefill_status")
    save("worker.json", status)
    assert status[0]["fast_prefill"] and status[0]["split_blocks"], status
    event("loaded")
    tokenizer = llm.get_tokenizer()
    filler = tokenizer.encode(
        "Cache reuse and model execution require careful scheduling. ",
        add_special_tokens=False,
    )
    chats = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        for text in PROMPTS[:4]
    ]

    def prompts(length, n):
        result = []
        for chat in chats[:n]:
            assert len(chat) <= length
            count = length - len(chat)
            result.append(
                {
                    "prompt_token_ids": (filler * (count // len(filler) + 1))[:count]
                    + chat
                }
            )
        return result

    llm.collective_rpc("prefill_profile_start")
    llm.generate(
        prompts(8192, 1),
        SamplingParams(temperature=0, max_tokens=2, ignore_eos=True),
        use_tqdm=False,
    )
    profile = llm.collective_rpc("prefill_profile_stop")
    save("profile-kernels.json", profile)
    assert bool(profile[0]) == (args.variant == "fused"), profile

    quality = []
    # Fast prefill intentionally rejects prompt logprobs in the retained V1 runner.
    for length in [1024, 8192]:
        output = llm.generate(
            prompts(length, 1),
            SamplingParams(
                temperature=0,
                max_tokens=8,
                ignore_eos=True,
                logprobs=5,
            ),
            use_tqdm=False,
        )[0]
        quality.append(
            dict(
                input_ids=output.prompt_token_ids,
                output_ids=list(output.outputs[0].token_ids),
                logprobs=[
                    {str(k): asdict(v) for k, v in row.items()}
                    for row in output.outputs[0].logprobs
                ],
                prompt_logprobs=[
                    None if row is None else {str(k): asdict(v) for k, v in row.items()}
                    for row in (output.prompt_logprobs or [])
                ],
            )
        )
    save("quality.json", quality)
    rows = []
    for length in args.lengths:
        for concurrency in args.concurrencies:
            workload = prompts(length, concurrency)
            for trial in range(args.trials + 2):
                params = SamplingParams(
                    temperature=0,
                    max_tokens=args.tokens,
                    ignore_eos=True,
                    seed=20261021,
                )
                llm.collective_rpc("prefill_synchronize")
                start = time.perf_counter()
                output = llm.generate(workload, params, use_tqdm=False)
                llm.collective_rpc("prefill_synchronize")
                elapsed = time.perf_counter() - start
                assert all(len(o.outputs[0].token_ids) == args.tokens for o in output)
                row = dict(
                    length=length,
                    concurrency=concurrency,
                    trial=trial,
                    warmup=trial < 2,
                    elapsed_s=elapsed,
                    tokens=args.tokens * concurrency,
                    output_ids=[list(o.outputs[0].token_ids) for o in output],
                    request_metrics=[asdict(o.metrics) for o in output],
                )
                rows.append(row)
                save("trials.json", rows)
                event(
                    "trial",
                    length=length,
                    concurrency=concurrency,
                    trial=trial,
                    elapsed_s=elapsed,
                )
    save(
        "complete.json",
        dict(
            status="PASS", rows=len(rows), worker=llm.collective_rpc("prefill_status")
        ),
    )
    event("complete")


if __name__ == "__main__":
    main()
