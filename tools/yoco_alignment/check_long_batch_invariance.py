# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare a seeded trajectory and complete-logit hashes alone and with fillers."""

import argparse
import json
from pathlib import Path

from logprob_kl import _disable_transformers_torchvision, _patch_local_vllm_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    args = parser.parse_args()
    _disable_transformers_torchvision()
    _patch_local_vllm_metadata()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        align=True,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        gpu_memory_utilization=0.65,
        enable_prefix_caching=False,
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 4},
        kernel_config={"moe_backend": "triton"},
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4]},
        worker_cls="logit_hash_worker.LogitHashWorker",
    )
    target = (
        "Write an extended tutorial on memory management in operating systems. "
        "Explain virtual memory, page tables, allocation, "
        "and reclamation with examples."
    )
    fillers = [
        "Explain how stars form.",
        "Write a tutorial on sorting algorithms.",
        "Describe the water cycle in detail.",
    ]
    target_params = SamplingParams(
        temperature=0.8, top_p=0.95, seed=42, max_tokens=args.tokens, ignore_eos=True
    )
    filler_params = SamplingParams(
        temperature=0.8, seed=19, max_tokens=args.tokens + 32, ignore_eos=True
    )
    results = []
    for batch in (1, 4):
        llm.collective_rpc("reset_logit_hashes")
        prompts = (
            [target] if batch == 1 else [fillers[0], fillers[1], target, fillers[2]]
        )
        params = (
            [target_params]
            if batch == 1
            else [filler_params, filler_params, target_params, filler_params]
        )
        outputs = llm.generate(prompts, params, use_tqdm=False)
        target_output = outputs[0 if batch == 1 else 2].outputs[0]
        hashes = llm.collective_rpc("get_logit_hashes")[0]
        assert sum(map(len, hashes.values())) == sum(
            len(o.outputs[0].token_ids) for o in outputs
        )
        candidates = [
            values for values in hashes.values() if len(values) == args.tokens
        ]
        assert len(candidates) == 1, {key: len(value) for key, value in hashes.items()}
        results.append(
            {
                "batch": batch,
                "tokens": list(target_output.token_ids),
                "hashes": candidates[0],
                "text": target_output.text,
            }
        )
        print(
            f"completed batch={batch}, target_tokens={len(target_output.token_ids)}",
            flush=True,
        )
    report = {
        "tokens_equal": results[0]["tokens"] == results[1]["tokens"],
        "all_logits_equal": results[0]["hashes"] == results[1]["hashes"],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)
    if not report["tokens_equal"] or not report["all_logits_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
