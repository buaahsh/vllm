# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Insert a prefill after a fixed target decode step using an in-process engine."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from logprob_kl import (
    _disable_transformers_torchvision,
    _patch_local_vllm_metadata,
    _vllm_logprob_tensor,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--multiprocess", action="store_true")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--insertion-delay-ms", type=float, default=5)
    args = parser.parse_args()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1" if args.multiprocess else "0"
    if not args.multiprocess:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    _disable_transformers_torchvision()
    _patch_local_vllm_metadata()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        align=True,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_batched_tokens=4096,
        max_num_seqs=2,
        gpu_memory_utilization=0.65,
        max_logprobs=-1,
        enable_prefix_caching=False,
        async_scheduling=args.async_scheduling,
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 4},
        kernel_config={"moe_backend": "triton"},
        compilation_config={"cudagraph_capture_sizes": [1, 2]},
        worker_cls="graph_trace_worker.GraphTraceWorker" if args.trace_dir else "auto",
    )
    engine = llm.llm_engine
    tokenizer = llm.get_tokenizer()
    vocab = engine.model_config.get_vocab_size()
    target = tokenizer.encode(
        "Explain why the sky is blue, with a clear physical explanation. "
    )
    target = (target * 8)[:66]
    filler = (
        tokenizer.encode("Write a short recipe for soup with fresh vegetables. ") * 4
    )[:22]
    params = SamplingParams(
        temperature=0, max_tokens=args.steps, logprobs=-1, seed=0, ignore_eos=True
    )
    results = []
    for inject in (False, True):
        request_id = f"target-{inject}"
        engine.add_request(request_id, {"prompt_token_ids": target}, params)
        inserted = False
        snapshot = False
        final = None
        if inject and args.multiprocess:
            time.sleep(args.insertion_delay_ms / 1000)
            engine.add_request(
                "filler",
                {"prompt_token_ids": filler},
                SamplingParams(temperature=0, max_tokens=1, seed=19),
            )
            inserted = True
        while engine.has_unfinished_requests():
            outputs = engine.step()
            for output in outputs:
                if output.request_id != request_id:
                    continue
                final = output.outputs[0]
                print(
                    json.dumps({"inject": inject, "tokens": len(final.token_ids)}),
                    flush=True,
                )
                if inject and not inserted and len(final.token_ids) >= 1:
                    engine.add_request(
                        "filler",
                        {"prompt_token_ids": filler},
                        SamplingParams(temperature=0, max_tokens=1, seed=19),
                    )
                    inserted = True
                if args.trace_dir and not snapshot and len(final.token_ids) >= 2:
                    llm.collective_rpc(
                        "save_graph_trace",
                        args=(str(args.trace_dir / f"inject-{inject}.pt"),),
                    )
                    snapshot = True
        logprobs = torch.stack([_vllm_logprob_tensor(s, vocab) for s in final.logprobs])
        results.append(
            {"inject": inject, "tokens": list(final.token_ids), "logprobs": logprobs}
        )
    diff = (
        (results[0]["logprobs"] - results[1]["logprobs"]).abs().max(-1).values.tolist()
    )
    report = {
        "equal": torch.equal(results[0]["logprobs"], results[1]["logprobs"]),
        "per_step_max": diff,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
