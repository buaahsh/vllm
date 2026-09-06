# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare complete target-request distributions in ragged YOCO batches."""

import argparse
import json
from pathlib import Path

import torch
from logprob_kl import (
    _disable_transformers_torchvision,
    _patch_local_vllm_metadata,
    _vllm_logprob_tensor,
)


def configure_trace(root, path):
    """Capture first-layer boundaries for the next prefill, outside graphs."""
    handles = getattr(root, "_invariance_trace_handles", [])
    for handle in handles:
        handle.remove()
    records = []
    active = [True]

    def clone(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, (tuple, list)):
            return [clone(x) for x in value if isinstance(x, torch.Tensor)]
        return None

    def hook(name):
        def capture(module, inputs, output):
            if active[0]:
                records.append(
                    {"name": name, "inputs": clone(inputs), "output": clone(output)}
                )

        return capture

    handles = []
    for name, module in root.model.named_modules():
        if name == "embed_tokens" or name == "layers.0" or name.startswith("layers.0."):
            handles.append(module.register_forward_hook(hook(name)))

    def finish(module, inputs, output):
        if active[0]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(records, path)
            active[0] = False

    handles.append(root.model.register_forward_hook(finish))
    root._invariance_trace_handles = handles
    root._invariance_trace_records = records


class AlignTraceWorker:
    def configure_invariance_trace(self, path: str):
        configure_trace(self.get_model(), path)

    def save_invariance_trace(self, path: str):
        root = self.get_model()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(root._invariance_trace_records, path)
        for handle in root._invariance_trace_handles:
            handle.remove()
        root._invariance_trace_handles = []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--capture-batches", type=int, nargs="+")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--prefill-tokens", type=int, default=4096)
    parser.add_argument("--prompt-length", type=int, default=66)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--short-fillers", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--graph-trace-dir", type=Path)
    args = parser.parse_args()
    _disable_transformers_torchvision()
    _patch_local_vllm_metadata()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        align=True,
        enforce_eager=args.enforce_eager,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=max(4096, args.prompt_length + args.steps),
        max_num_batched_tokens=args.prefill_tokens,
        max_num_seqs=max(args.batches + (args.capture_batches or [])),
        gpu_memory_utilization=0.65,
        max_logprobs=-1,
        enable_prefix_caching=False,
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 4},
        kernel_config={"moe_backend": "triton"},
        compilation_config={
            "cudagraph_capture_sizes": sorted(set(args.capture_batches or args.batches))
        },
        seed=0,
        worker_extension_cls=(
            "check_model_batch_invariance.AlignTraceWorker" if args.trace_dir else ""
        ),
        worker_cls="graph_trace_worker.GraphTraceWorker"
        if args.graph_trace_dir
        else "auto",
    )
    tokenizer = llm.get_tokenizer()
    vocab_size = llm.llm_engine.model_config.get_vocab_size()
    target = tokenizer.encode(
        "Explain why the sky is blue, with a clear physical explanation. "
    )
    target = (target * (args.prompt_length // len(target) + 1))[: args.prompt_length]
    filler = tokenizer.encode("Write a short recipe for soup with fresh vegetables. ")
    target_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.steps,
        logprobs=-1,
        seed=0,
        ignore_eos=True,
    )
    filler_params = SamplingParams(temperature=0, max_tokens=args.steps + 2, seed=19)
    results = []
    expected = None
    for batch in args.batches:
        for position in sorted({0, batch // 2, batch - 1}):
            prompts = [
                {"prompt_token_ids": (filler * 16)[: 3 + (i * 19) % 110]}
                for i in range(batch)
            ]
            prompts[position] = {"prompt_token_ids": target}
            params = [filler_params] * batch
            if args.short_fillers:
                params = [
                    SamplingParams(temperature=0, max_tokens=1 + i % 3, seed=19)
                    for i in range(batch)
                ]
            params[position] = target_params
            if args.trace_dir:
                llm.collective_rpc(
                    "configure_invariance_trace",
                    args=(str(args.trace_dir / f"b{batch}-p{position}.pt"),),
                )
            output = llm.generate(prompts, params, use_tqdm=False)[position].outputs[0]
            if args.trace_dir:
                llm.collective_rpc(
                    "save_invariance_trace",
                    args=(str(args.trace_dir / f"b{batch}-p{position}.pt"),),
                )
            if args.graph_trace_dir:
                llm.collective_rpc(
                    "save_graph_trace",
                    args=(str(args.graph_trace_dir / f"b{batch}-p{position}.pt"),),
                )
            logprobs = torch.stack(
                [_vllm_logprob_tensor(step, vocab_size) for step in output.logprobs]
            )
            item = {
                "batch": batch,
                "position": position,
                "tokens": list(output.token_ids),
                "logprobs": logprobs,
            }
            if expected is None:
                expected = item
            item["tokens_equal"] = item["tokens"] == expected["tokens"]
            item["logprobs_equal"] = torch.equal(logprobs, expected["logprobs"])
            item["max_abs_diff"] = float((logprobs - expected["logprobs"]).abs().max())
            results.append(item)
            print(
                json.dumps({k: v for k, v in item.items() if k != "logprobs"}),
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, args.output.with_suffix(".pt"))
    report = {
        "prompt_length": args.prompt_length,
        "steps": args.steps,
        "max_num_batched_tokens": args.prefill_tokens,
        "temperature": args.temperature,
        "short_fillers": args.short_fillers,
        "all_equal": all(r["tokens_equal"] and r["logprobs_equal"] for r in results),
        "results": [{k: v for k, v in r.items() if k != "logprobs"} for r in results],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
