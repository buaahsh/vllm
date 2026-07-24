#!/usr/bin/env python3
"""Decode-path logprob alignment for vLLM rollouts and Native replay.

The vLLM stage generates continuations through the real KV-cached decode path.
It records the sampled-token logprob at every response position and the full
vocabulary distribution at sparse, one-based response positions. The Native
stage teacher-forces the exact saved token IDs in packed prefill forwards.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

try:
    from . import logprob_kl as prefill_probe
except ImportError:
    import logprob_kl as prefill_probe


ARTIFACT_FORMAT = "yoco_decode_logprob_kl"
SCHEMA_VERSION = 1


def _positive_int_csv(value: str) -> list[int]:
    try:
        positions = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "positions must be comma-separated integers"
        ) from error
    if not positions or any(position < 1 for position in positions):
        raise argparse.ArgumentTypeError("positions must contain positive integers")
    if len(set(positions)) != len(positions):
        raise argparse.ArgumentTypeError("positions must not contain duplicates")
    return sorted(positions)


def _add_prompt_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prompt-suite",
        choices=("default", "mixed5", "mixed16", "chunk4k"),
        default="mixed5",
    )
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--prompt-start", type=int, default=0)
    parser.add_argument("--prompt-limit", type=int)


def _add_vllm_engine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--distributed-executor-backend")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument(
        "--enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--kv-sharing-fast-prefill", action="store_true")
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--compilation-config-json")
    parser.add_argument("--quantization")
    parser.add_argument("--quantization-config-json")
    parser.add_argument("--quantization-ignore", action="append", default=[])
    parser.add_argument("--attention-backend")
    parser.add_argument("--flash-attn-version", type=int)
    parser.add_argument("--force-fa-num-splits-one", action="store_true")
    parser.add_argument("--keep-attention-qkv-bf16", action="store_true")
    parser.add_argument("--moe-backend")
    parser.add_argument("--fa4-source-root", help=argparse.SUPPRESS)
    parser.add_argument(
        "--fa4-profile",
        choices=("default", "no-ex2", "q-stage1", "q-stage2"),
        default="default",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fa4-pack-gqa",
        choices=("auto", "on", "off"),
        default="auto",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fa4-tile-mn",
        choices=("default", "128x64", "128x128"),
        default="default",
        help=argparse.SUPPRESS,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare vLLM decode logprobs with Native teacher forcing"
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    vllm_parser = subparsers.add_parser(
        "vllm",
        help="Generate and save vLLM rollouts and decode logprobs",
    )
    vllm_parser.add_argument("--model", required=True)
    vllm_parser.add_argument("--out", required=True)
    _add_prompt_args(vllm_parser)
    _add_vllm_engine_args(vllm_parser)
    vllm_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Number of prompts submitted together to the vLLM engine. "
            "--max-num-seqs defaults to this value."
        ),
    )
    vllm_parser.add_argument(
        "--first-batch-size",
        type=int,
        help=(
            "Optional first submission size before using --batch-size; "
            "use 1 with batch-size 15 for a 16-prompt 1+15 split."
        ),
    )
    vllm_parser.add_argument("--decode-length", type=int, default=64)
    vllm_parser.add_argument(
        "--vocab-logprob-stride",
        type=int,
        default=16,
        help=(
            "Save full-vocabulary logprobs at one-based response positions "
            "stride, 2*stride, and so on."
        ),
    )
    vllm_parser.add_argument(
        "--vocab-logprob-positions",
        type=_positive_int_csv,
        help=(
            "Comma-separated one-based response positions at which to save "
            "full-vocabulary logprobs. Overrides --vocab-logprob-stride."
        ),
    )
    vllm_parser.add_argument(
        "--include-prefill-vocab-logprob",
        action="store_true",
        help="Also save the full distribution for response position 1.",
    )
    vllm_parser.add_argument("--artifact-top-k", type=int, default=20)
    vllm_parser.add_argument("--temperature", type=float, default=0.0)
    vllm_parser.add_argument("--top-p", type=float, default=1.0)
    vllm_parser.add_argument("--top-k", type=int, default=0)
    vllm_parser.add_argument("--min-p", type=float, default=0.0)
    vllm_parser.add_argument("--presence-penalty", type=float, default=0.0)
    vllm_parser.add_argument("--frequency-penalty", type=float, default=0.0)
    vllm_parser.add_argument("--repetition-penalty", type=float, default=1.0)
    vllm_parser.add_argument("--seed", type=int, default=0)

    native_parser = subparsers.add_parser(
        "native",
        help="Teacher-force a saved rollout through the Native model",
    )
    native_parser.add_argument("--rollout", required=True)
    native_parser.add_argument("--out", required=True)
    native_parser.add_argument("--native-checkpoint", required=True)
    native_parser.add_argument(
        "--llm-train-dir", default="/data/yanqi/yanqi/yoco_mxfp8/llm-train"
    )
    native_parser.add_argument(
        "--native-dtype", choices=("bfloat16",), default="bfloat16"
    )
    native_parser.add_argument(
        "--native-quant-mode", choices=("bfloat16", "mxfp8")
    )
    native_parser.add_argument("--native-quant-block-size", type=int)
    native_parser.add_argument("--native-use-cute", action="store_true")
    native_parser.add_argument("--native-local-attention", action="store_true")
    native_parser.add_argument(
        "--native-require-transformer-engine",
        action="store_true",
        help="Fail unless llm-train imports TransformerEngine MoE permutation APIs.",
    )
    native_parser.add_argument("--native-no-kv-cache", action="store_true")
    native_parser.add_argument("--force-fa-num-splits-one", action="store_true")
    native_parser.add_argument("--keep-attention-qkv-bf16", action="store_true")
    native_parser.add_argument("--fa4-source-root", help=argparse.SUPPRESS)
    native_parser.add_argument(
        "--fa4-profile",
        choices=("default", "no-ex2", "q-stage1", "q-stage2"),
        default="default",
        help=argparse.SUPPRESS,
    )
    native_parser.add_argument(
        "--fa4-pack-gqa",
        choices=("auto", "on", "off"),
        default="auto",
        help=argparse.SUPPRESS,
    )
    native_parser.add_argument(
        "--fa4-tile-mn",
        choices=("default", "128x64", "128x128"),
        default="default",
        help=argparse.SUPPRESS,
    )
    native_parser.add_argument(
        "--native-use-torch-fp8-quant", action="store_true"
    )
    native_parser.add_argument("--max-model-len", type=int)
    native_parser.add_argument("--seed", type=int)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare a vLLM rollout artifact with Native replay",
    )
    compare_parser.add_argument("--vllm", required=True)
    compare_parser.add_argument("--native", required=True)
    compare_parser.add_argument("--out-json", required=True)
    compare_parser.add_argument("--model")
    compare_parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")


def _as_json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _checkpoint_positions(
    response_length: int,
    stride: int,
    include_prefill: bool,
    explicit_positions: list[int] | None = None,
) -> list[int]:
    positions = (
        [position for position in explicit_positions if position <= response_length]
        if explicit_positions is not None
        else list(range(stride, response_length + 1, stride))
    )
    if include_prefill and response_length and 1 not in positions:
        positions.insert(0, 1)
    return positions


def _step_logprob_data(
    all_logprobs: Any,
    step: int,
    selected_token_id: int,
    include_full: bool,
) -> tuple[float, list[int] | None, list[float] | None]:
    if all(
        hasattr(all_logprobs, name)
        for name in ("start_indices", "end_indices", "token_ids", "logprobs")
    ):
        start = int(all_logprobs.start_indices[step])
        end = int(all_logprobs.end_indices[step])
        try:
            selected_index = all_logprobs.token_ids.index(
                selected_token_id, start, end
            )
        except ValueError as error:
            raise RuntimeError(
                f"Selected token {selected_token_id} missing from logprobs"
            ) from error
        if not include_full:
            return float(all_logprobs.logprobs[selected_index]), None, None
        return (
            float(all_logprobs.logprobs[selected_index]),
            all_logprobs.token_ids[start:end],
            all_logprobs.logprobs[start:end],
        )

    step_logprobs = all_logprobs[step]
    selected = step_logprobs.get(selected_token_id)
    if selected is None:
        raise RuntimeError(f"Selected token {selected_token_id} missing from logprobs")
    selected_logprob = float(selected.logprob)
    if not include_full:
        return selected_logprob, None, None
    token_ids = []
    values = []
    for token_id, logprob in step_logprobs.items():
        token_ids.append(int(token_id))
        values.append(float(logprob.logprob))
    return selected_logprob, token_ids, values


def _dense_logprobs(
    token_ids: list[int],
    logprobs: list[float],
    vocab_size: int,
    *,
    prompt_name: str,
    decode_position: int,
) -> torch.Tensor:
    values = torch.full((vocab_size,), float("-inf"), dtype=torch.float32)
    values[torch.tensor(token_ids, dtype=torch.long)] = torch.tensor(
        logprobs, dtype=torch.float32
    )
    finite = int(torch.isfinite(values).sum())
    if finite != vocab_size:
        raise RuntimeError(
            f"Expected full vocab logprobs for {prompt_name} at response "
            f"position {decode_position}, got {finite}/{vocab_size}"
        )
    return values


def _rollout_fingerprint(payload: dict[str, Any]) -> str:
    canonical = {
        "format": payload["format"],
        "schema_version": payload["schema_version"],
        "model": payload["model"],
        "engine_config": payload["engine_config"],
        "sampling_config": payload["sampling_config"],
        "prompt_config": payload.get("prompt_config"),
        "batch_config": payload["batch_config"],
        "vocab_logprob_config": payload["vocab_logprob_config"],
        "responses": [
            {
                "prompt_name": row["prompt"]["name"],
                "prompt_token_ids": row["prompt"]["prompt_token_ids"],
                "generated_token_ids": row["generated_token_ids"],
                "submission_batch_index": row["submission_batch_index"],
                "submission_batch_slot": row["submission_batch_slot"],
            }
            for row in payload["results"]
        ],
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_artifact(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"Unsupported artifact format in {path}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version in {path}: "
            f"{payload.get('schema_version')}"
        )
    return payload


def _vllm_engine_config(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    requested_batch_capacity = max(
        args.batch_size,
        args.first_batch_size or 0,
    )
    effective_max_num_seqs = args.max_num_seqs or requested_batch_capacity
    config: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": (
            args.max_num_batched_tokens or args.max_model_len
        ),
        "max_num_seqs": effective_max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "max_logprobs": -1,
        "logprobs_mode": "raw_logprobs",
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    if args.enable_expert_parallel:
        config["enable_expert_parallel"] = True
    if args.distributed_executor_backend:
        config["distributed_executor_backend"] = args.distributed_executor_backend
    if args.enable_chunked_prefill is not None:
        config["enable_chunked_prefill"] = args.enable_chunked_prefill
    if args.kv_sharing_fast_prefill:
        config["kv_sharing_fast_prefill"] = True
    if args.quantization:
        config["quantization"] = args.quantization

    quantization_config = (
        json.loads(args.quantization_config_json)
        if args.quantization_config_json
        else None
    )
    if args.quantization_ignore:
        quantization_config = quantization_config or {}
        quantization_config["ignore"] = args.quantization_ignore
    if quantization_config is not None:
        config["quantization_config"] = quantization_config

    if (
        args.attention_backend
        or args.flash_attn_version is not None
        or args.force_fa_num_splits_one
    ):
        attention_config = {}
        if args.attention_backend:
            attention_config["backend"] = args.attention_backend
        if args.flash_attn_version is not None:
            attention_config["flash_attn_version"] = args.flash_attn_version
        if args.force_fa_num_splits_one:
            attention_config["flash_attn_force_num_splits_one"] = True
        config["attention_config"] = attention_config
    if args.moe_backend:
        config["moe_backend"] = args.moe_backend
    if args.compilation_config_json:
        config["compilation_config"] = json.loads(args.compilation_config_json)
    return config, effective_max_num_seqs


def run_vllm(args: argparse.Namespace) -> None:
    _validate_positive("batch_size", args.batch_size)
    if args.first_batch_size is not None:
        _validate_positive("first_batch_size", args.first_batch_size)
    _validate_positive("decode_length", args.decode_length)
    _validate_positive("vocab_logprob_stride", args.vocab_logprob_stride)
    _validate_positive("artifact_top_k", args.artifact_top_k)
    if args.vocab_logprob_positions and max(args.vocab_logprob_positions) > args.decode_length:
        raise ValueError(
            "vocab_logprob_positions cannot exceed decode_length: "
            f"{max(args.vocab_logprob_positions)} > {args.decode_length}"
        )

    prefill_probe._configure_vllm_alignment_env()
    prefill_probe._disable_transformers_torchvision()
    prefill_probe._patch_local_vllm_metadata()
    if args.keep_attention_qkv_bf16:
        if args.quantization not in ("fp8_per_block", "mxfp8"):
            raise ValueError(
                "--keep-attention-qkv-bf16 requires online fp8_per_block or "
                "mxfp8 quantization"
            )
        for pattern in prefill_probe.ATTENTION_QKV_BF16_IGNORE:
            if pattern not in args.quantization_ignore:
                args.quantization_ignore.append(pattern)
    fa4_runtime: dict[str, Any] = {}
    if args.fa4_source_root:
        if args.flash_attn_version != 4:
            raise ValueError("--fa4-source-root requires --flash-attn-version 4")
        fa4_runtime = prefill_probe._install_vllm_external_fa4(
            args.fa4_source_root,
            args.fa4_profile,
            pack_gqa=args.fa4_pack_gqa,
            tile_mn=args.fa4_tile_mn,
        )
    from vllm import LLM, SamplingParams

    tokenizer, records = prefill_probe._prompt_records(
        args.model,
        args.prompt_suite,
        args.prompt_index,
        args.prompt_start,
        args.prompt_limit,
    )
    engine_config, effective_max_num_seqs = _vllm_engine_config(args)
    llm = LLM(**engine_config)
    sampling_config = {
        "n": 1,
        "max_tokens": args.decode_length,
        "min_tokens": 0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "stop": None,
        "stop_token_ids": None,
        "ignore_eos": False,
        "logprobs": -1,
        "flat_logprobs": True,
        "detokenize": False,
        "output_kind": "DELTA",
        "returned_logprobs_mode": "raw_logprobs_before_sampling_transforms",
        "eos_token_id": _as_json_scalar(tokenizer.eos_token_id),
    }
    from vllm.sampling_params import RequestOutputKind

    def make_sampling_params() -> Any:
        return SamplingParams(
            max_tokens=args.decode_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            ignore_eos=False,
            logprobs=-1,
            flat_logprobs=True,
            detokenize=False,
            output_kind=RequestOutputKind.DELTA,
        )

    vocab_size = int(llm.llm_engine.model_config.get_vocab_size())
    parallel_config = llm.llm_engine.vllm_config.parallel_config
    dp_rank = int(parallel_config.data_parallel_rank or 0)
    results = []
    submitted_batches = []
    batches = prefill_probe._record_batches(
        records,
        args.batch_size,
        args.first_batch_size,
    )
    for batch_index, batch in enumerate(batches):
        submitted_batches.append(
            {
                "batch_index": batch_index,
                "size": len(batch),
                "prompt_names": [record["name"] for record in batch],
            }
        )
        states: dict[str, dict[str, Any]] = {}
        state_order = []
        configured_checkpoint_positions = set(
            _checkpoint_positions(
                args.decode_length,
                args.vocab_logprob_stride,
                args.include_prefill_vocab_logprob,
                args.vocab_logprob_positions,
            )
        )
        for batch_slot, record in enumerate(batch):
            request_id = f"decode-kl-{batch_index}-{batch_slot}"
            state = {
                "record": record,
                "batch_slot": batch_slot,
                "generated_token_ids": [],
                "selected_token_logprobs": [],
                "vocab_logprobs": [],
                "finish_reason": None,
                "stop_reason": None,
                "cumulative_logprob": None,
                "finished": False,
            }
            states[request_id] = state
            state_order.append(state)
            llm.llm_engine.add_request(
                request_id,
                {
                    "prompt_token_ids": record["prompt_token_ids"],
                    "prompt": record["prompt_text"],
                },
                make_sampling_params(),
            )

        while llm.llm_engine.has_unfinished_requests():
            step_outputs = llm.llm_engine.step()
            for request_output in step_outputs:
                state = states.get(request_output.request_id)
                if state is None:
                    raise RuntimeError(
                        f"Unexpected vLLM request ID: {request_output.request_id}"
                    )
                if len(request_output.outputs) != 1:
                    raise RuntimeError(
                        f"Expected one completion for {request_output.request_id}, "
                        f"got {len(request_output.outputs)}"
                    )
                output = request_output.outputs[0]
                delta_token_ids = [int(token_id) for token_id in output.token_ids]
                if output.logprobs is None:
                    raise RuntimeError(
                        f"vLLM returned no logprobs for "
                        f"{state['record']['name']}"
                    )
                if len(output.logprobs) != len(delta_token_ids):
                    raise RuntimeError(
                        f"Delta token/logprob length mismatch for "
                        f"{state['record']['name']}: {len(delta_token_ids)} vs "
                        f"{len(output.logprobs)}"
                    )

                for delta_step, selected_token_id in enumerate(delta_token_ids):
                    decode_step = len(state["generated_token_ids"])
                    decode_position = decode_step + 1
                    keep_full_logprobs = (
                        decode_position in configured_checkpoint_positions
                    )
                    selected_logprob, token_ids, step_logprobs = (
                        _step_logprob_data(
                            output.logprobs,
                            delta_step,
                            selected_token_id,
                            keep_full_logprobs,
                        )
                    )
                    state["generated_token_ids"].append(selected_token_id)
                    state["selected_token_logprobs"].append(selected_logprob)
                    if keep_full_logprobs:
                        assert token_ids is not None and step_logprobs is not None
                        full_logprobs = _dense_logprobs(
                            token_ids,
                            step_logprobs,
                            vocab_size,
                            prompt_name=state["record"]["name"],
                            decode_position=decode_position,
                        )
                        state["vocab_logprobs"].append(
                            {
                                "decode_position": decode_position,
                                "decode_step": decode_step,
                                "selected_token_id": selected_token_id,
                                "selected_token_logprob": selected_logprob,
                                "logprobs": full_logprobs,
                                **prefill_probe._top_payload(
                                    full_logprobs, args.artifact_top_k
                                ),
                            }
                        )
                if output.finish_reason is not None:
                    state["finish_reason"] = _as_json_scalar(output.finish_reason)
                    state["stop_reason"] = _as_json_scalar(output.stop_reason)
                    state["cumulative_logprob"] = _as_json_scalar(
                        output.cumulative_logprob
                    )
                if request_output.finished:
                    state["finished"] = True
            del step_outputs

        for state in state_order:
            record = state["record"]
            generated_token_ids = state["generated_token_ids"]
            if not state["finished"]:
                raise RuntimeError(f"vLLM did not finish request {record['name']}")
            if not generated_token_ids:
                raise RuntimeError(f"vLLM generated no tokens for {record['name']}")
            if len(generated_token_ids) > args.decode_length:
                raise RuntimeError(
                    f"vLLM exceeded decode length for {record['name']}: "
                    f"{len(generated_token_ids)} > {args.decode_length}"
                )
            response_text = tokenizer.decode(
                generated_token_ids, skip_special_tokens=True
            )
            response_text_with_special_tokens = tokenizer.decode(
                generated_token_ids, skip_special_tokens=False
            )
            result = {
                "prompt": record,
                "submission_batch_index": batch_index,
                "submission_batch_slot": state["batch_slot"],
                "submission_batch_size": len(batch),
                "response_text": response_text,
                "response_text_with_special_tokens": (
                    response_text_with_special_tokens
                ),
                "generated_token_ids": generated_token_ids,
                "num_generated_tokens": len(generated_token_ids),
                "finish_reason": state["finish_reason"],
                "stop_reason": state["stop_reason"],
                "cumulative_logprob": state["cumulative_logprob"],
                "selected_token_logprobs": torch.tensor(
                    state["selected_token_logprobs"], dtype=torch.float32
                ),
                "vocab_logprobs": state["vocab_logprobs"],
            }
            results.append(result)
            if dp_rank == 0:
                print(
                    f"[vllm-decode-kl] {record['name']}: "
                    f"tokens={len(generated_token_ids)} "
                    f"finish={state['finish_reason']} "
                    f"vocab_positions="
                    f"{[row['decode_position'] for row in state['vocab_logprobs']]}",
                    flush=True,
                )
        gc.collect()

    if dp_rank != 0:
        return
    payload = {
        "format": ARTIFACT_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "stage": "rollout",
        "backend": "vllm",
        "model": args.model,
        "engine_config": {
            **engine_config,
            "fa4_runtime": fa4_runtime,
            "keep_attention_qkv_bf16": args.keep_attention_qkv_bf16,
            "quantization_ignore": list(args.quantization_ignore),
        },
        "sampling_config": sampling_config,
        "prompt_config": {
            "prompt_suite": args.prompt_suite,
            "prompt_index": args.prompt_index,
            "prompt_start": args.prompt_start,
            "prompt_limit": args.prompt_limit,
        },
        "batch_config": {
            "submission_batch_size": args.batch_size,
            "first_submission_batch_size": args.first_batch_size,
            "max_num_seqs": effective_max_num_seqs,
            "v1_multiprocessing_enabled": False,
            "submitted_batches": submitted_batches,
        },
        "vocab_logprob_config": {
            "position_indexing": "one-based response token position",
            "decode_step_indexing": "zero-based; step 0 is produced by prefill",
            "stride": args.vocab_logprob_stride,
            "requested_positions": args.vocab_logprob_positions,
            "include_prefill_position": args.include_prefill_vocab_logprob,
            "transient_capture": (
                "streamed full vocabulary at every vLLM decode step; "
                "non-checkpoint rows are discarded immediately"
            ),
            "artifact_storage": "full vocabulary only at configured positions",
        },
        "vocab_size": vocab_size,
        "results": results,
    }
    payload["rollout_fingerprint"] = _rollout_fingerprint(payload)
    prefill_probe._save(args.out, payload)
    print(
        f"[vllm-decode-kl] saved {args.out} "
        f"fingerprint={payload['rollout_fingerprint']}",
        flush=True,
    )


def _validated_rollout(path: str) -> dict[str, Any]:
    payload = _load_artifact(path)
    if payload.get("stage") != "rollout" or payload.get("backend") != "vllm":
        raise ValueError(f"Expected a vLLM rollout artifact: {path}")
    expected_fingerprint = _rollout_fingerprint(payload)
    if payload.get("rollout_fingerprint") != expected_fingerprint:
        raise ValueError(
            f"Rollout fingerprint mismatch in {path}: expected "
            f"{expected_fingerprint}, got {payload.get('rollout_fingerprint')}"
        )
    return payload


def _results_by_submission_batch(
    results: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(int(result["submission_batch_index"]), []).append(result)

    batches = []
    for batch_index in sorted(grouped):
        batch = sorted(
            grouped[batch_index], key=lambda row: int(row["submission_batch_slot"])
        )
        expected_slots = list(range(len(batch)))
        actual_slots = [int(row["submission_batch_slot"]) for row in batch]
        if actual_slots != expected_slots:
            raise ValueError(
                f"Non-contiguous slots in rollout batch {batch_index}: {actual_slots}"
            )
        if any(int(row["submission_batch_size"]) != len(batch) for row in batch):
            raise ValueError(f"Recorded size mismatch in rollout batch {batch_index}")
        batches.append(batch)
    return batches


def _load_native_model(
    args: argparse.Namespace,
    seed: int,
) -> tuple[Any, Any, torch.device, Any, bool, list[str]]:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    llm_dir = Path(args.llm_train_dir).resolve() / "llm"
    if str(llm_dir) not in sys.path:
        sys.path.insert(0, str(llm_dir))
    prefill_probe._install_native_inference_compat(
        local_attention=args.native_local_attention,
        force_fa_num_splits_one=args.force_fa_num_splits_one,
        fa4_source_root=args.fa4_source_root,
        fa4_profile=args.fa4_profile,
        fa4_pack_gqa=args.fa4_pack_gqa,
        fa4_tile_mn=args.fa4_tile_mn,
    )
    from arch.model import Model, ModelArgs, create_kv_cache

    using_transformer_engine = prefill_probe._install_native_moe_padding_compat()
    if args.native_require_transformer_engine and not using_transformer_engine:
        raise RuntimeError(
            "--native-require-transformer-engine was set, but llm-train "
            "could not import TransformerEngine's MoE permutation APIs"
        )

    checkpoint_dir = Path(args.native_checkpoint)
    metadata_path = checkpoint_dir / "metadata.json"
    state_path = checkpoint_dir / "model_state_rank_0.pth"
    for path, description in (
        (metadata_path, "metadata"),
        (state_path, "rank-0 model state"),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Native checkpoint {description} not found: {path}"
            )

    if args.native_use_torch_fp8_quant:
        import arch.linear as linear
        import kernel.moe_ffn as moe_ffn
        import kernel.quant as quant

        linear.per_token_cast_to_fp8 = quant._per_token_cast_to_fp8_torch
        linear.per_block_cast_to_fp8 = quant._per_block_cast_to_fp8_torch
        moe_ffn.per_token_cast_to_fp8 = quant._per_token_cast_to_fp8_torch
        print(
            "[native-decode-kl] using torch FP8 activation quantization",
            flush=True,
        )

    if not dist.is_initialized():
        if "RANK" not in os.environ:
            raise RuntimeError(
                "Run the Native stage with torch.distributed.run/torchrun, "
                "including for a one-GPU standalone run."
            )
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.accelerator.set_device_index(local_rank)
    device = torch.device("cuda")
    torch.manual_seed(seed)

    with metadata_path.open(encoding="utf-8") as reader:
        metadata = json.load(reader)
    modelargs = ModelArgs()
    for key, value in metadata["modelargs"].items():
        setattr(modelargs, key, value)
    if args.native_quant_mode is not None:
        modelargs.quant_mode = args.native_quant_mode
    if args.native_quant_block_size is not None:
        modelargs.quant_block_size = args.native_quant_block_size
    modelargs.use_cute = args.native_use_cute
    if args.native_no_kv_cache:
        modelargs.moe_fwd_bwd_overlap = False

    init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["dp"])
    default_device = torch.get_default_device()
    default_dtype = torch.get_default_dtype()
    torch.set_default_device(device)
    torch.set_default_dtype(prefill_probe._dtype(args.native_dtype))
    model = Model(modelargs)
    torch.set_default_device(default_device)
    torch.set_default_dtype(default_dtype)
    model.eval()

    attention_qkv_bf16_modules: list[str] = []
    if args.keep_attention_qkv_bf16:
        if args.native_quant_mode != "mxfp8":
            raise ValueError(
                "--keep-attention-qkv-bf16 requires --native-quant-mode mxfp8"
            )
        attention_qkv_bf16_modules = (
            prefill_probe._keep_native_attention_qkv_bf16(model)
        )
        print(
            "[native-decode-kl] keeping attention QKV projections BF16: "
            f"{len(attention_qkv_bf16_modules)} logical projections",
            flush=True,
        )

    state = torch.load(
        state_path,
        map_location=prefill_probe._device_mapping(-1),
        mmap=True,
        weights_only=False,
    )
    state = {
        key: value for key, value in state.items() if not key.startswith("moe_loss.")
    }
    model.load_state_dict(state)
    print(
        f"[native-decode-kl] model loaded use_cute={modelargs.use_cute} "
        f"kv_cache={not args.native_no_kv_cache}",
        flush=True,
    )
    return (
        model,
        create_kv_cache,
        device,
        dist,
        using_transformer_engine,
        attention_qkv_bf16_modules,
    )


def _native_result_shell(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": source["prompt"],
        "submission_batch_index": source["submission_batch_index"],
        "submission_batch_slot": source["submission_batch_slot"],
        "submission_batch_size": source["submission_batch_size"],
        "response_text": source["response_text"],
        "response_text_with_special_tokens": source[
            "response_text_with_special_tokens"
        ],
        "generated_token_ids": source["generated_token_ids"],
        "num_generated_tokens": source["num_generated_tokens"],
        "finish_reason": source["finish_reason"],
        "stop_reason": source["stop_reason"],
        "selected_token_logprobs": [],
        "vocab_logprobs": [],
    }


@torch.no_grad()
def _run_native_batch(
    *,
    model: Any,
    create_kv_cache: Any,
    device: torch.device,
    source_batch: list[dict[str, Any]],
    max_model_len: int,
    native_dtype: str,
    use_kv_cache: bool,
    artifact_top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    token_lists = []
    for source in source_batch:
        prompt_ids = [
            int(token_id)
            for token_id in source["prompt"]["prompt_token_ids"]
        ]
        generated_ids = [int(token_id) for token_id in source["generated_token_ids"]]
        if not generated_ids:
            raise ValueError(f"Empty rollout for {source['prompt']['name']}")
        token_ids = prompt_ids + generated_ids
        if len(token_ids) > max_model_len:
            raise ValueError(
                f"Sequence {source['prompt']['name']} has {len(token_ids)} tokens, "
                f"exceeding max_model_len={max_model_len}"
            )
        if max(token_ids) >= model.args.vocab_size:
            raise ValueError(
                f"Sequence {source['prompt']['name']} contains token id "
                f"{max(token_ids)} >= vocab_size={model.args.vocab_size}"
            )
        token_lists.append(token_ids)

    seqlens = torch.tensor(
        [len(token_ids) for token_ids in token_lists],
        device=device,
        dtype=torch.int32,
    )
    prefill_tokens = torch.cat(
        [
            torch.tensor(token_ids, dtype=torch.long, device=device)
            for token_ids in token_lists
        ]
    )
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.int32),
            seqlens.cumsum(dim=0).to(torch.int32),
        ]
    )
    positions = torch.cat(
        [
            torch.arange(len(token_ids), device=device, dtype=torch.int32)
            for token_ids in token_lists
        ]
    )
    context: dict[str, Any] = {
        "cu_seqlens_q": cu_seqlens,
        "cu_seqlens_k": cu_seqlens,
        "max_seqlen_q": int(seqlens.max()),
        "max_seqlen_k": int(seqlens.max()),
        "positions": positions,
    }
    if use_kv_cache:
        batch_indices = torch.cat(
            [
                torch.full(
                    (len(token_ids),),
                    batch_index,
                    device=device,
                    dtype=torch.int32,
                )
                for batch_index, token_ids in enumerate(token_lists)
            ]
        )
        context.update(
            {
                "kv_cache": create_kv_cache(
                    model.args,
                    len(source_batch),
                    max_model_len,
                    prefill_probe._dtype(native_dtype),
                    device,
                ),
                "slot_mapping": batch_indices * max_model_len + positions,
                "layer_index": 0,
            }
        )

    hidden, _, _ = model(prefill_tokens, context=context, last_hidden_only=True)
    starts = [0] + seqlens.cumsum(dim=0)[:-1].long().cpu().tolist()
    native_results = [_native_result_shell(source) for source in source_batch]
    checkpoint_positions = [
        {
            int(row["decode_position"])
            for row in source["vocab_logprobs"]
        }
        for source in source_batch
    ]
    response_lengths = [
        len(source["generated_token_ids"]) for source in source_batch
    ]
    logit_batch_sizes = []
    for decode_position in range(1, max(response_lengths) + 1):
        active_indices = [
            index
            for index, response_length in enumerate(response_lengths)
            if decode_position <= response_length
        ]
        hidden_indices = []
        selected_token_ids = []
        for result_index in active_indices:
            prompt_len = len(
                source_batch[result_index]["prompt"]["prompt_token_ids"]
            )
            hidden_indices.append(
                starts[result_index] + prompt_len + decode_position - 2
            )
            selected_token_ids.append(
                int(
                    source_batch[result_index]["generated_token_ids"][
                        decode_position - 1
                    ]
                )
            )

        step_hidden = hidden.index_select(
            0, torch.tensor(hidden_indices, device=device, dtype=torch.long)
        )
        step_logprobs = torch.log_softmax(
            model.output(step_hidden).float(), dim=-1
        )
        selected = step_logprobs.gather(
            1,
            torch.tensor(
                selected_token_ids, device=device, dtype=torch.long
            ).unsqueeze(1),
        ).squeeze(1)
        if not torch.isfinite(selected).all():
            raise RuntimeError(
                f"Native produced non-finite selected-token logprobs at "
                f"response position {decode_position}"
            )

        logit_batch_sizes.append(
            {
                "decode_position": decode_position,
                "active_sequences": len(active_indices),
            }
        )
        for active_row, result_index in enumerate(active_indices):
            selected_logprob = float(selected[active_row].cpu())
            native_results[result_index]["selected_token_logprobs"].append(
                selected_logprob
            )
            if decode_position in checkpoint_positions[result_index]:
                full_logprobs = step_logprobs[active_row].cpu()
                if not torch.isfinite(full_logprobs).all():
                    raise RuntimeError(
                        f"Native produced non-finite full-vocab logprobs for "
                        f"{source_batch[result_index]['prompt']['name']} at "
                        f"response position {decode_position}"
                    )
                native_results[result_index]["vocab_logprobs"].append(
                    {
                        "decode_position": decode_position,
                        "decode_step": decode_position - 1,
                        "selected_token_id": selected_token_ids[active_row],
                        "selected_token_logprob": selected_logprob,
                        "logprobs": full_logprobs,
                        **prefill_probe._top_payload(
                            full_logprobs, artifact_top_k
                        ),
                    }
                )
        del step_logprobs

    for result in native_results:
        result["selected_token_logprobs"] = torch.tensor(
            result["selected_token_logprobs"], dtype=torch.float32
        )
    batch_metadata = {
        "submission_batch_index": int(source_batch[0]["submission_batch_index"]),
        "size": len(source_batch),
        "packed_sequence_lengths": seqlens.cpu().tolist(),
        "packed_num_tokens": int(seqlens.sum()),
        "logit_batch_sizes": logit_batch_sizes,
    }
    return native_results, batch_metadata


def run_native(args: argparse.Namespace) -> None:
    rollout = _validated_rollout(args.rollout)
    seed = (
        args.seed
        if args.seed is not None
        else int(rollout["sampling_config"].get("seed") or 0)
    )
    max_model_len = args.max_model_len or int(
        rollout["engine_config"]["max_model_len"]
    )
    _validate_positive("max_model_len", max_model_len)
    artifact_top_k = max(
        (
            int(row["top_ids"].numel())
            for result in rollout["results"]
            for row in result["vocab_logprobs"]
        ),
        default=20,
    )
    (
        model,
        create_kv_cache,
        device,
        dist,
        using_transformer_engine,
        attention_qkv_bf16_modules,
    ) = _load_native_model(args, seed)
    results = []
    replay_batches = []
    try:
        for source_batch in _results_by_submission_batch(rollout["results"]):
            batch_results, batch_metadata = _run_native_batch(
                model=model,
                create_kv_cache=create_kv_cache,
                device=device,
                source_batch=source_batch,
                max_model_len=max_model_len,
                native_dtype=args.native_dtype,
                use_kv_cache=not args.native_no_kv_cache,
                artifact_top_k=artifact_top_k,
            )
            results.extend(batch_results)
            replay_batches.append(batch_metadata)
            print(
                f"[native-decode-kl] batch "
                f"{batch_metadata['submission_batch_index']}: "
                f"sequences={batch_metadata['size']} "
                f"tokens={batch_metadata['packed_num_tokens']}",
                flush=True,
            )

        payload = {
            "format": ARTIFACT_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "stage": "teacher_forced_replay",
            "backend": "native",
            "model": rollout["model"],
            "native_checkpoint": args.native_checkpoint,
            "rollout_source": str(Path(args.rollout).resolve()),
            "rollout_fingerprint": rollout["rollout_fingerprint"],
            "source_engine_config": rollout["engine_config"],
            "source_sampling_config": rollout["sampling_config"],
            "source_batch_config": rollout["batch_config"],
            "vocab_logprob_config": rollout["vocab_logprob_config"],
            "vocab_size": rollout["vocab_size"],
            "native_config": {
                "native_checkpoint": args.native_checkpoint,
                "llm_train_dir": str(Path(args.llm_train_dir).resolve()),
                "dtype": args.native_dtype,
                "quant_mode": args.native_quant_mode,
                "quant_block_size": args.native_quant_block_size,
                "use_cute": args.native_use_cute,
                "attention_version": 4 if args.native_use_cute else 2,
                "local_attention": args.native_local_attention,
                "kv_cache_enabled": not args.native_no_kv_cache,
                "torch_fp8_quant_fallback": args.native_use_torch_fp8_quant,
                "transformer_engine_enabled": using_transformer_engine,
                "force_fa_num_splits_one": args.force_fa_num_splits_one,
                "keep_attention_qkv_bf16": args.keep_attention_qkv_bf16,
                "attention_qkv_bf16_modules": attention_qkv_bf16_modules,
                "fa4_runtime": dict(prefill_probe._NATIVE_FA4_RUNTIME),
                "max_model_len": max_model_len,
                "seed": seed,
                "execution": "packed teacher-forced prefill",
                "logit_projection": "one response position at a time",
            },
            "replay_batches": replay_batches,
            "results": results,
        }
        prefill_probe._save(args.out, payload)
        print(f"[native-decode-kl] saved {args.out}", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _result_key(result: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(result["submission_batch_index"]),
        int(result["submission_batch_slot"]),
        str(result["prompt"]["name"]),
    )


def _diff_summary(diffs: list[float]) -> dict[str, float | int | None]:
    if not diffs:
        return {
            "count": 0,
            "mean_diff": None,
            "mean_abs_diff": None,
            "p50_abs_diff": None,
            "p95_abs_diff": None,
            "p99_abs_diff": None,
            "max_abs_diff": None,
            "rms_diff": None,
            "mean_abs_ratio_minus_one": None,
            "p95_abs_ratio_minus_one": None,
            "p99_abs_ratio_minus_one": None,
            "max_abs_ratio_minus_one": None,
        }
    values = torch.tensor(diffs, dtype=torch.float64)
    absolute = values.abs()
    ratio_distortion = (values.exp() - 1).abs()
    return {
        "count": len(diffs),
        "mean_diff": float(values.mean()),
        "mean_abs_diff": float(absolute.mean()),
        "p50_abs_diff": float(torch.quantile(absolute, 0.50)),
        "p95_abs_diff": float(torch.quantile(absolute, 0.95)),
        "p99_abs_diff": float(torch.quantile(absolute, 0.99)),
        "max_abs_diff": float(absolute.max()),
        "rms_diff": float(values.square().mean().sqrt()),
        "mean_abs_ratio_minus_one": float(ratio_distortion.mean()),
        "p95_abs_ratio_minus_one": float(torch.quantile(ratio_distortion, 0.95)),
        "p99_abs_ratio_minus_one": float(torch.quantile(ratio_distortion, 0.99)),
        "max_abs_ratio_minus_one": float(ratio_distortion.max()),
    }


def _response_length_summary(
    results: list[dict[str, Any]], decode_length: int
) -> dict[str, Any]:
    lengths = torch.tensor(
        [len(result["generated_token_ids"]) for result in results],
        dtype=torch.float64,
    )
    finish_reason_counts: dict[str, int] = {}
    for result in results:
        reason = str(result.get("finish_reason"))
        finish_reason_counts[reason] = finish_reason_counts.get(reason, 0) + 1
    return {
        "count": len(results),
        "min": int(lengths.min()),
        "mean": float(lengths.mean()),
        "median": float(torch.quantile(lengths, 0.50)),
        "max": int(lengths.max()),
        "reached_decode_length": int((lengths == decode_length).sum()),
        "finish_reason_counts": finish_reason_counts,
    }


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float | int]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        **{
            f"mean_{name}": sum(row[name] for row in rows) / len(rows)
            for name in rows[0]
        },
    }


def _top_rows_or_empty(
    tokenizer: Any | None,
    logprobs: torch.Tensor,
    top_k: int,
) -> list[dict[str, Any]]:
    if tokenizer is None or top_k == 0:
        return []
    return prefill_probe._top_rows(
        tokenizer, logprobs, min(top_k, int(logprobs.numel()))
    )


def _native_replay(path: str, rollout_fingerprint: str) -> dict[str, Any]:
    payload = _load_artifact(path)
    if (
        payload.get("stage") != "teacher_forced_replay"
        or payload.get("backend") != "native"
    ):
        raise ValueError(f"Expected a Native replay artifact: {path}")
    if payload.get("rollout_fingerprint") != rollout_fingerprint:
        raise ValueError(
            "Native replay was produced from a different rollout: "
            f"{payload.get('rollout_fingerprint')} != {rollout_fingerprint}"
        )
    return payload


def compare(args: argparse.Namespace) -> None:
    if args.top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {args.top_k}")
    rollout = _validated_rollout(args.vllm)
    native = _native_replay(args.native, rollout["rollout_fingerprint"])
    if int(rollout["vocab_size"]) != int(native["vocab_size"]):
        raise ValueError(
            f"Vocabulary mismatch: vLLM={rollout['vocab_size']} "
            f"Native={native['vocab_size']}"
        )

    tokenizer = None
    if args.top_k:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model or rollout["model"], trust_remote_code=True
        )

    rollout_results = {_result_key(row): row for row in rollout["results"]}
    native_results = {_result_key(row): row for row in native["results"]}
    if rollout_results.keys() != native_results.keys():
        missing_native = sorted(rollout_results.keys() - native_results.keys())
        missing_vllm = sorted(native_results.keys() - rollout_results.keys())
        raise ValueError(
            "Result identity mismatch: "
            f"missing_native={missing_native}, missing_vllm={missing_vllm}"
        )

    all_selected_diffs = []
    decode_only_diffs = []
    selected_by_position: dict[int, list[float]] = {}
    all_vocab_metrics = []
    vocab_metrics_by_position: dict[int, list[dict[str, float]]] = {}
    prompt_reports = []
    for key, vllm_result in rollout_results.items():
        native_result = native_results[key]
        generated_token_ids = [
            int(token_id) for token_id in vllm_result["generated_token_ids"]
        ]
        if generated_token_ids != [
            int(token_id) for token_id in native_result["generated_token_ids"]
        ]:
            raise ValueError(f"Generated token mismatch for {key}")

        vllm_selected = vllm_result["selected_token_logprobs"].float()
        native_selected = native_result["selected_token_logprobs"].float()
        if vllm_selected.shape != native_selected.shape:
            raise ValueError(
                f"Selected-logprob shape mismatch for {key}: "
                f"vLLM={tuple(vllm_selected.shape)} "
                f"Native={tuple(native_selected.shape)}"
            )
        if vllm_selected.numel() != len(generated_token_ids):
            raise ValueError(f"Selected-logprob/token length mismatch for {key}")

        selected_rows = []
        prompt_diffs = []
        for decode_step, token_id in enumerate(generated_token_ids):
            decode_position = decode_step + 1
            vllm_logprob = float(vllm_selected[decode_step])
            native_logprob = float(native_selected[decode_step])
            diff = vllm_logprob - native_logprob
            prompt_diffs.append(diff)
            all_selected_diffs.append(diff)
            if decode_position > 1:
                decode_only_diffs.append(diff)
            selected_by_position.setdefault(decode_position, []).append(diff)
            selected_rows.append(
                {
                    "decode_position": decode_position,
                    "decode_step": decode_step,
                    "execution_path": (
                        "prefill_boundary" if decode_position == 1 else "decode"
                    ),
                    "token_id": token_id,
                    "vllm_logprob": vllm_logprob,
                    "native_logprob": native_logprob,
                    "logprob_diff_vllm_minus_native": diff,
                    "abs_logprob_diff": abs(diff),
                }
            )

        vllm_vocab_rows = {
            int(row["decode_position"]): row
            for row in vllm_result["vocab_logprobs"]
        }
        native_vocab_rows = {
            int(row["decode_position"]): row
            for row in native_result["vocab_logprobs"]
        }
        if vllm_vocab_rows.keys() != native_vocab_rows.keys():
            raise ValueError(
                f"Full-vocab checkpoint mismatch for {key}: "
                f"vLLM={sorted(vllm_vocab_rows)} "
                f"Native={sorted(native_vocab_rows)}"
            )

        vocab_reports = []
        prompt_vocab_metrics = []
        for decode_position in sorted(vllm_vocab_rows):
            vllm_row = vllm_vocab_rows[decode_position]
            native_row = native_vocab_rows[decode_position]
            expected_token_id = generated_token_ids[decode_position - 1]
            if (
                int(vllm_row["selected_token_id"]) != expected_token_id
                or int(native_row["selected_token_id"]) != expected_token_id
            ):
                raise ValueError(
                    f"Checkpoint token mismatch for {key} at "
                    f"response position {decode_position}"
                )
            vllm_logprobs = vllm_row["logprobs"].float()
            native_logprobs = native_row["logprobs"].float()
            if vllm_logprobs.shape != native_logprobs.shape:
                raise ValueError(
                    f"Full-vocab shape mismatch for {key} at response "
                    f"position {decode_position}: "
                    f"vLLM={tuple(vllm_logprobs.shape)} "
                    f"Native={tuple(native_logprobs.shape)}"
                )
            metrics = prefill_probe._metrics(native_logprobs, vllm_logprobs)
            all_vocab_metrics.append(metrics)
            prompt_vocab_metrics.append(metrics)
            vocab_metrics_by_position.setdefault(decode_position, []).append(metrics)
            vocab_reports.append(
                {
                    "decode_position": decode_position,
                    "decode_step": decode_position - 1,
                    "execution_path": (
                        "prefill_boundary" if decode_position == 1 else "decode"
                    ),
                    "selected_token_id": expected_token_id,
                    **metrics,
                    "native_top": _top_rows_or_empty(
                        tokenizer, native_logprobs, args.top_k
                    ),
                    "vllm_top": _top_rows_or_empty(
                        tokenizer, vllm_logprobs, args.top_k
                    ),
                }
            )

        prompt_reports.append(
            {
                "prompt_name": vllm_result["prompt"]["name"],
                "prompt_kind": vllm_result["prompt"]["kind"],
                "prompt_len": vllm_result["prompt"]["prompt_len"],
                "submission_batch_index": key[0],
                "submission_batch_slot": key[1],
                "response_text": vllm_result["response_text"],
                "generated_token_ids": generated_token_ids,
                "num_generated_tokens": len(generated_token_ids),
                "finish_reason": vllm_result["finish_reason"],
                "selected_token_summary": _diff_summary(prompt_diffs),
                "selected_tokens": selected_rows,
                "full_vocab_summary": _mean_metrics(prompt_vocab_metrics),
                "full_vocab_positions": vocab_reports,
            }
        )

    selected_position_reports = [
        {
            "decode_position": decode_position,
            "decode_step": decode_position - 1,
            "execution_path": (
                "prefill_boundary" if decode_position == 1 else "decode"
            ),
            **_diff_summary(selected_by_position[decode_position]),
        }
        for decode_position in sorted(selected_by_position)
    ]
    vocab_position_reports = [
        {
            "decode_position": decode_position,
            "decode_step": decode_position - 1,
            "execution_path": (
                "prefill_boundary" if decode_position == 1 else "decode"
            ),
            **_mean_metrics(vocab_metrics_by_position[decode_position]),
        }
        for decode_position in sorted(vocab_metrics_by_position)
    ]
    aggregate = {
        "selected_token_logprob_diff_vllm_minus_native": {
            "all_response_positions": _diff_summary(all_selected_diffs),
            "decode_only_positions": _diff_summary(decode_only_diffs),
        },
        "full_vocab": _mean_metrics(all_vocab_metrics),
        "response_lengths": _response_length_summary(
            list(rollout_results.values()),
            int(rollout["sampling_config"]["max_tokens"]),
        ),
        "num_prompts": len(prompt_reports),
    }
    report = {
        "format": ARTIFACT_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "comparison": "vllm_decode_vs_native_teacher_forced_prefill",
        "comparison_direction": (
            "selected diff is vLLM - Native; full-vocab KL uses Native as "
            "reference and vLLM as candidate"
        ),
        "rollout_fingerprint": rollout["rollout_fingerprint"],
        "vllm_artifact": str(Path(args.vllm).resolve()),
        "native_artifact": str(Path(args.native).resolve()),
        "sampling_config": rollout["sampling_config"],
        "batch_config": rollout["batch_config"],
        "vocab_logprob_config": rollout["vocab_logprob_config"],
        "aggregate": aggregate,
        "selected_token_by_position": selected_position_reports,
        "full_vocab_by_position": vocab_position_reports,
        "prompts": prompt_reports,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as writer:
        json.dump(report, writer, ensure_ascii=False, indent=2, allow_nan=False)
        writer.write("\n")
    print(json.dumps(aggregate, indent=2, allow_nan=False), flush=True)
    print(f"[compare-decode-kl] saved {args.out_json}", flush=True)


def main() -> None:
    args = parse_args()
    if args.cmd == "vllm":
        run_vllm(args)
    elif args.cmd == "native":
        run_native(args)
    elif args.cmd == "compare":
        compare(args)
    else:
        raise AssertionError(args.cmd)


if __name__ == "__main__":
    main()