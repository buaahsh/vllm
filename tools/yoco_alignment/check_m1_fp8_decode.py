# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare real decode raw logprobs using V2's native forced-token replay.

Run separate engines with VLLM_YOCO_FP8_SMALL_M=0/1 and compare their JSON.
The sampler forces known tokens after sampling, before raw logprob gathering;
the actual prefill and autoregressive decode model computations remain active.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

from benchmark_fast_decode import DEFAULT_ENV, DEFAULT_TOKENS, generate_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens-json", type=Path, default=DEFAULT_TOKENS)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite existing results")
    mode = os.environ.get("VLLM_YOCO_FP8_SMALL_M", "1")
    os.environ.update(DEFAULT_ENV)
    os.environ["VLLM_YOCO_FP8_SMALL_M"] = mode
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", "")]
    )
    from vllm import LLM, SamplingParams

    tokens = json.loads(args.tokens_json.read_text())["tokens"]
    assert len(tokens) >= 1152
    config = dict(
        model=args.model,
        additional_config={"yoco_execution_mode": "fast"},
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=False,
        gpu_memory_utilization=0.65,
        max_model_len=8192,
        max_num_batched_tokens=4096,
        max_num_seqs=16,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        kv_sharing_fast_prefill=True,
        quantization="fp8_per_block",
        kv_cache_dtype="fp8",
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 4},
        kernel_config={"enable_flashinfer_autotune": False},
        compilation_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "custom_ops": [],
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
        },
        logprobs_mode="raw_logprobs",
        enable_trace_replay=True,
        seed=918,
        worker_extension_cls="benchmark_fast_decode.FastDecodeWorker",
    )
    result = {
        "completed": False,
        "small_m": mode,
        "config": config,
        "tokens_sha256": hashlib.sha256(args.tokens_json.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "b1": [],
        "b8": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def scores(output, expected=None):
        completion = output.outputs[0]
        if expected is not None:
            assert list(completion.token_ids) == expected
        values = [
            step[token].logprob
            for token, step in zip(completion.token_ids, completion.logprobs)
        ]
        assert len(values) == len(completion.token_ids)
        assert all(math.isfinite(value) and value <= 0 for value in values)
        return {
            "tokens": list(completion.token_ids),
            "raw_logprobs": values,
            "cached_prompt_tokens": output.num_cached_tokens,
        }

    llm = LLM(**config)
    try:
        result["runtime"] = llm.collective_rpc("fast_decode_manifest")[0]
        linears = result["runtime"]["small_fp8_linears"]
        assert len(linears) == 80
        expected_backend = (
            "YocoM1Fp8LinearKernel" if mode == "1" else "DeepGemmFp8BlockScaledMMKernel"
        )
        assert {row["backend"] for row in linears} == {expected_backend}
        prompt = {"prompt_token_ids": tokens[:512]}
        assert llm.reset_prefix_cache()
        outputs, _ = generate_batch(
            llm,
            [prompt],
            SamplingParams(
                temperature=0, max_tokens=8, min_tokens=8, ignore_eos=True, logprobs=1
            ),
        )
        greedy = scores(outputs[0])
        assert llm.reset_prefix_cache()
        outputs, _ = generate_batch(
            llm,
            [prompt],
            SamplingParams(
                temperature=0,
                max_tokens=8,
                ignore_eos=True,
                logprobs=1,
                trace_decode_token_ids=greedy["tokens"],
            ),
        )
        replay = scores(outputs[0], greedy["tokens"])
        drift = max(
            abs(a - b) for a, b in zip(greedy["raw_logprobs"], replay["raw_logprobs"])
        )
        assert drift < 1e-5 and min(replay["raw_logprobs"]) < -1e-6
        result["raw_logprob_anchor"] = {
            "greedy": greedy,
            "replay": replay,
            "max_abs": drift,
        }
        save()
        for prefix in range(128, 1025, 128):
            assert llm.reset_prefix_cache()
            target = tokens[prefix : prefix + 128]
            outputs, _ = generate_batch(
                llm,
                [{"prompt_token_ids": tokens[:prefix]}],
                SamplingParams(
                    temperature=0,
                    max_tokens=128,
                    ignore_eos=True,
                    logprobs=1,
                    trace_decode_token_ids=target,
                ),
            )
            result["b1"].append({"prompt_tokens": prefix, **scores(outputs[0], target)})
            save()
            print("B1 teacher forced", prefix, flush=True)
        assert llm.reset_prefix_cache()
        prompts = [
            {"prompt_token_ids": tokens[64 * i : 64 * i + 512]} for i in range(8)
        ]
        targets = [tokens[64 * i + 512 : 64 * i + 640] for i in range(8)]
        params = [
            SamplingParams(
                temperature=0,
                max_tokens=128,
                ignore_eos=True,
                logprobs=1,
                trace_decode_token_ids=target,
            )
            for target in targets
        ]
        outputs, _ = generate_batch(llm, prompts, params)
        result["b8"] = [scores(out, target) for out, target in zip(outputs, targets)]
        # Real larger-prefill fallback and generation, outside quality scoring.
        assert llm.reset_prefix_cache()
        long_prompt = (tokens * 4)[:4096]
        outputs, _ = generate_batch(
            llm,
            [{"prompt_token_ids": long_prompt}],
            SamplingParams(
                temperature=0, max_tokens=16, min_tokens=16, ignore_eos=True
            ),
        )
        assert len(outputs[0].outputs[0].token_ids) == 16
        result["long_smoke_tokens"] = list(outputs[0].outputs[0].token_ids)
        for key in ("b1", "b8"):
            values = [lp for row in result[key] for lp in row["raw_logprobs"]]
            assert len(values) == 1024
            result[key + "_nll"] = -statistics.mean(values)
        result["completed"] = True
        save()
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
