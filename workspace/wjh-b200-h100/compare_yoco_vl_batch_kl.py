#!/usr/bin/env python
"""Compute batch next-token KL between llm-train YOCO-VL and vLLM.

Execute this file from:
    /home/v-jiahaowang/workspace/vllm/workspace/wjh-b200-h100
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import random
import shutil
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
LLM_TRAIN_LLM_ROOT = Path(
    os.environ.get("LLM_TRAIN_LLM_ROOT", "/root/workspace/llm-train/llm")
)
DEFAULT_CHECKPOINT_DIR = Path("/data/wjh/updates_3000")
DEFAULT_VLLM_MODEL = Path("/data/wjh/updates_3000-hf-vl")
DEFAULT_TOKENIZER = Path("/mnt/msranlp/yutao/hf_cache/agens_vl_tokenizer_0622")
DEFAULT_IMAGES = [
    "/home/v-jiahaowang/workspace/llm-train/workspace/dog1.jpeg",
    "/home/v-jiahaowang/workspace/llm-train/workspace/dog2.jpeg",
    "/home/v-jiahaowang/workspace/llm-train/workspace/dog3.jpeg",
]
DEFAULT_PROMPTS = [
    "Describe this dog and the surrounding scene in one concise sentence.",
    "What is the dog doing? Mention its pose and expression.",
    "List the most noticeable visual details in this image.",
]
DEFAULT_SYSTEM_PROMPT = "You are a helpful and friendly AI assistant."


def _add_paths() -> None:
    paths = (str(REPO_ROOT), str(LLM_TRAIN_LLM_ROOT))
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)
    system_dist_packages = "/usr/local/lib/python3.12/dist-packages"
    if system_dist_packages not in sys.path and Path(system_dist_packages).is_dir():
        # The container image provides the native flash-attn build here. Append
        # it after the venv so the venv's PyTorch and vLLM remain authoritative.
        sys.path.append(system_dist_packages)
    pythonpath = [
        entry for entry in os.environ.get("PYTHONPATH", "").split(":") if entry
    ]
    for path in reversed(paths):
        if path not in pythonpath:
            pythonpath.insert(0, path)
    os.environ["PYTHONPATH"] = ":".join(pythonpath)


def _configure_cuda_toolkit() -> None:
    if (
        "TRITON_PTXAS_PATH" not in os.environ
        and Path("/usr/local/cuda/bin/ptxas").is_file()
    ):
        os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"

    cuda_include = Path("/usr/local/cuda/targets/x86_64-linux/include")
    if cuda_include.is_dir():
        cpath_entries = [
            entry for entry in os.environ.get("CPATH", "").split(":") if entry
        ]
        if str(cuda_include) not in cpath_entries:
            os.environ["CPATH"] = ":".join([str(cuda_include), *cpath_entries])

    if shutil.which("nvcc"):
        return

    for cuda_home in (Path("/usr/local/cuda-13.3"), Path("/usr/local/cuda-13")):
        nvcc = cuda_home / "bin" / "nvcc"
        if nvcc.exists():
            os.environ.setdefault("CUDA_HOME", str(cuda_home))
            os.environ["PATH"] = f"{cuda_home / 'bin'}:{os.environ.get('PATH', '')}"
            return


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _supported_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(cls.__init__)
    params = signature.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in params}


def _shutdown_llm(llm: object | None) -> None:
    if llm is None:
        return
    llm_engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(llm_engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if shutdown is not None:
        shutdown(timeout=5)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


def _default_stop_token_ids(tokenizer) -> set[int]:
    stop_ids = {int(tokenizer.eos_id)}
    for token in ("<|user|>", "<|assistant|>", "<|system|>"):
        token_id = tokenizer.tok.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            stop_ids.add(token_id)
    return stop_ids


@torch.inference_mode()
def _reference_batch_next_logits(
    *,
    model,
    prompt_ids: list[torch.Tensor],
    image_masks: list[torch.Tensor],
    projected_image_features: list[torch.Tensor],
    max_new_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    from arch.model import create_kv_cache

    batch_size = len(prompt_ids)
    prompt_lengths = torch.tensor(
        [int(ids.numel()) for ids in prompt_ids],
        device=device,
        dtype=torch.int32,
    )
    max_prompt_length = int(prompt_lengths.max().item())
    max_length = max_prompt_length + max_new_tokens
    for index, length in enumerate(prompt_lengths.tolist()):
        if length + max_new_tokens > model.args.max_seq_len:
            raise ValueError(
                f"sample {index} prompt ({length}) + max_new_tokens "
                f"({max_new_tokens}) exceeds model max_seq_len ({model.args.max_seq_len})"
            )

    flat_prompt_ids = torch.cat(prompt_ids, dim=0).to(device=device)
    flat_image_mask = torch.cat(image_masks, dim=0).to(device=device)
    image_features = torch.zeros(
        flat_prompt_ids.numel(),
        model.args.d_model,
        dtype=projected_image_features[0].dtype,
        device=device,
    )
    image_offsets = flat_image_mask.nonzero(as_tuple=False).flatten()
    image_features[image_offsets] = torch.cat(projected_image_features, dim=0)

    kv_cache = create_kv_cache(
        model.args,
        batch_size=batch_size,
        max_seq_len=max_length,
        dtype=torch.bfloat16,
        device=device,
    )
    cu_seqlens = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(prompt_lengths, dim=0)
    positions = torch.cat(
        [
            torch.arange(length, device=device, dtype=torch.int32)
            for length in prompt_lengths.tolist()
        ],
        dim=0,
    )
    slot_mapping = torch.cat(
        [
            batch_index * max_length
            + torch.arange(length, device=device, dtype=torch.int32)
            for batch_index, length in enumerate(prompt_lengths.tolist())
        ],
        dim=0,
    )
    prefill_context = {
        "kv_cache": kv_cache,
        "cu_seqlens_q": cu_seqlens,
        "cu_seqlens_k": cu_seqlens,
        "max_seqlen_q": max_prompt_length,
        "max_seqlen_k": max_prompt_length,
        "positions": positions,
        "slot_mapping": slot_mapping,
        "layer_index": 0,
    }
    hidden, _, _ = model(
        flat_prompt_ids,
        context=prefill_context,
        last_hidden_only=True,
        image_features=image_features,
        image_input_mask=flat_image_mask,
    )
    last_indices = cu_seqlens[1:].long() - 1
    logits = model.output(hidden[last_indices]).float().cpu().clone()
    del kv_cache, flat_prompt_ids, flat_image_mask, image_features
    del positions, slot_mapping, prefill_context, hidden
    torch.cuda.empty_cache()
    return logits


def _matrix_stats(name: str, ref: torch.Tensor, cand: torch.Tensor) -> dict[str, Any]:
    ref_f = ref.detach().float()
    cand_f = cand.detach().float()
    diff = cand_f - ref_f
    ref_norm = torch.linalg.vector_norm(ref_f.reshape(-1))
    cand_norm = torch.linalg.vector_norm(cand_f.reshape(-1))
    diff_norm = torch.linalg.vector_norm(diff.reshape(-1))
    cosine = torch.dot(ref_f.reshape(-1), cand_f.reshape(-1)) / torch.clamp(
        ref_norm * cand_norm,
        min=1e-12,
    )
    return {
        "name": name,
        "shape": list(ref.shape),
        "ref_frobenius_norm": float(ref_norm.item()),
        "candidate_frobenius_norm": float(cand_norm.item()),
        "diff_frobenius_norm": float(diff_norm.item()),
        "relative_frobenius": float((diff_norm / torch.clamp(ref_norm, min=1e-12)).item()),
        "global_cosine": float(cosine.item()),
        "mean_abs": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
    }


def _distribution_divergence(ref_logits: torch.Tensor, cand_logits: torch.Tensor) -> dict[str, Any]:
    ref = ref_logits.detach().float()
    cand = cand_logits.detach().float()
    ref_logp = torch.log_softmax(ref, dim=-1)
    cand_logp = torch.log_softmax(cand, dim=-1)
    ref_p = ref_logp.exp()
    cand_p = cand_logp.exp()
    midpoint = 0.5 * (ref_p + cand_p)
    midpoint_logp = torch.log(torch.clamp(midpoint, min=torch.finfo(torch.float32).tiny))
    kl_ref_to_cand = torch.sum(ref_p * (ref_logp - cand_logp), dim=-1)
    kl_cand_to_ref = torch.sum(cand_p * (cand_logp - ref_logp), dim=-1)
    js = 0.5 * torch.sum(ref_p * (ref_logp - midpoint_logp), dim=-1) + 0.5 * torch.sum(
        cand_p * (cand_logp - midpoint_logp),
        dim=-1,
    )
    prob_abs_diff = (cand_p - ref_p).abs()
    ref_top1 = torch.argmax(ref, dim=-1)
    cand_top1 = torch.argmax(cand, dim=-1)
    return {
        "rows": int(ref.shape[0]),
        "vocab": int(ref.shape[-1]),
        "kl_ref_to_candidate": [float(x) for x in kl_ref_to_cand.tolist()],
        "kl_candidate_to_ref": [float(x) for x in kl_cand_to_ref.tolist()],
        "js": [float(x) for x in js.tolist()],
        "prob_l1": [float(x) for x in prob_abs_diff.sum(dim=-1).tolist()],
        "prob_linf": [float(x) for x in prob_abs_diff.max(dim=-1).values.tolist()],
        "kl_ref_to_candidate_mean": float(kl_ref_to_cand.mean().item()),
        "kl_candidate_to_ref_mean": float(kl_cand_to_ref.mean().item()),
        "js_mean": float(js.mean().item()),
        "prob_l1_mean": float(prob_abs_diff.sum(dim=-1).mean().item()),
        "prob_linf_max": float(prob_abs_diff.max().item()),
        "top1_agreement": float((ref_top1 == cand_top1).float().mean().item()),
        "ref_top1": [int(x) for x in ref_top1.tolist()],
        "candidate_top1": [int(x) for x in cand_top1.tolist()],
        "ref_entropy": [float(x) for x in (-(ref_p * ref_logp).sum(dim=-1)).tolist()],
        "candidate_entropy": [
            float(x) for x in (-(cand_p * cand_logp).sum(dim=-1)).tolist()
        ],
    }


def _pairwise_distribution_divergence(
    ref_logits: torch.Tensor,
    cand_logits: torch.Tensor,
) -> dict[str, Any]:
    ref = ref_logits.detach().float()
    cand = cand_logits.detach().float()
    ref_logp = torch.log_softmax(ref, dim=-1)
    cand_logp = torch.log_softmax(cand, dim=-1)
    ref_p = ref_logp.exp()
    cand_p = cand_logp.exp()

    kl_ref_to_cand = torch.empty(
        (ref.shape[0], cand.shape[0]),
        dtype=torch.float32,
        device=ref.device,
    )
    kl_cand_to_ref = torch.empty_like(kl_ref_to_cand)
    js = torch.empty_like(kl_ref_to_cand)
    prob_l1 = torch.empty_like(kl_ref_to_cand)
    for ref_idx in range(ref.shape[0]):
        midpoint = 0.5 * (ref_p[ref_idx : ref_idx + 1] + cand_p)
        midpoint_logp = torch.log(
            torch.clamp(midpoint, min=torch.finfo(torch.float32).tiny)
        )
        kl_ref_to_cand[ref_idx] = torch.sum(
            ref_p[ref_idx] * (ref_logp[ref_idx] - cand_logp),
            dim=-1,
        )
        kl_cand_to_ref[ref_idx] = torch.sum(
            cand_p * (cand_logp - ref_logp[ref_idx]),
            dim=-1,
        )
        js[ref_idx] = 0.5 * torch.sum(
            ref_p[ref_idx] * (ref_logp[ref_idx] - midpoint_logp),
            dim=-1,
        ) + 0.5 * torch.sum(cand_p * (cand_logp - midpoint_logp), dim=-1)
        prob_l1[ref_idx] = (cand_p - ref_p[ref_idx]).abs().sum(dim=-1)

    def minimum_cost_assignment(cost: torch.Tensor) -> list[int]:
        """Return the minimum-cost one-to-one row/column assignment in O(n^3)."""
        matrix = cost.detach().float().cpu().numpy()
        rows, columns = matrix.shape
        if rows != columns:
            raise ValueError(
                "Pairwise assignment requires a square cost matrix; "
                f"got {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("Pairwise assignment cost matrix contains NaN or inf")

        # Hungarian algorithm using 1-based indexing for the matching arrays.
        # ``p[column]`` is the row currently matched to that column.
        u = np.zeros(rows + 1, dtype=np.float64)
        v = np.zeros(columns + 1, dtype=np.float64)
        p = np.zeros(columns + 1, dtype=np.int64)
        way = np.zeros(columns + 1, dtype=np.int64)
        for row in range(1, rows + 1):
            p[0] = row
            column0 = 0
            min_value = np.full(columns + 1, np.inf, dtype=np.float64)
            used = np.zeros(columns + 1, dtype=np.bool_)
            while True:
                used[column0] = True
                row0 = int(p[column0])
                delta = np.inf
                column1 = 0
                for column in range(1, columns + 1):
                    if used[column]:
                        continue
                    current = matrix[row0 - 1, column - 1] - u[row0] - v[column]
                    if current < min_value[column]:
                        min_value[column] = current
                        way[column] = column0
                    if min_value[column] < delta:
                        delta = min_value[column]
                        column1 = column
                for column in range(columns + 1):
                    if used[column]:
                        u[p[column]] += delta
                        v[column] -= delta
                    else:
                        min_value[column] -= delta
                column0 = column1
                if p[column0] == 0:
                    break
            while True:
                column1 = int(way[column0])
                p[column0] = p[column1]
                column0 = column1
                if column0 == 0:
                    break

        assignment = [-1] * rows
        for column in range(1, columns + 1):
            if p[column] != 0:
                assignment[int(p[column]) - 1] = column - 1
        if any(column < 0 for column in assignment):
            raise RuntimeError(f"Hungarian assignment is incomplete: {assignment}")
        return assignment

    rows = ref.shape[0]

    def assignment_entry(assignment: list[int]) -> dict[str, Any]:
        perm_tensor = torch.tensor(assignment, device=ref.device, dtype=torch.long)
        row_indices = torch.arange(rows, device=ref.device)
        kl_values = kl_ref_to_cand[row_indices, perm_tensor]
        js_values = js[row_indices, perm_tensor]
        return {
            "candidate_row_for_reference_row": [int(x) for x in assignment],
            "kl_ref_to_candidate": [float(x) for x in kl_values.tolist()],
            "kl_ref_to_candidate_mean": float(kl_values.mean().item()),
            "js": [float(x) for x in js_values.tolist()],
            "js_mean": float(js_values.mean().item()),
        }

    best_by_kl = assignment_entry(minimum_cost_assignment(kl_ref_to_cand))
    best_by_js = assignment_entry(minimum_cost_assignment(js))

    return {
        "kl_ref_to_candidate": kl_ref_to_cand.cpu().tolist(),
        "kl_candidate_to_ref": kl_cand_to_ref.cpu().tolist(),
        "js": js.cpu().tolist(),
        "prob_l1": prob_l1.cpu().tolist(),
        "best_permutation_by_kl": best_by_kl,
        "best_permutation_by_js": best_by_js,
    }


def _capture_vllm_batch_logits(
    model: torch.nn.Module,
    *,
    ref_logits_cpu: torch.Tensor,
) -> dict[str, Any]:
    import types

    device = next(model.parameters()).device
    model._yoco_batch_ref_logits = ref_logits_cpu.to(device=device)
    model._yoco_batch_logits_capture = None
    model._yoco_batch_logits_calls = []
    model._yoco_batch_logits_chunks = []

    if not hasattr(model, "_yoco_batch_orig_compute_logits"):
        model._yoco_batch_orig_compute_logits = model.compute_logits

        def wrapped_compute_logits(self, hidden_states, *args, **kwargs):
            logits = self._yoco_batch_orig_compute_logits(
                hidden_states,
                *args,
                **kwargs,
            )
            if isinstance(logits, torch.Tensor):
                call = {
                    "hidden_shape": list(hidden_states.shape),
                    "logits_shape": list(logits.shape),
                    "hidden_dtype": str(hidden_states.dtype),
                    "logits_dtype": str(logits.dtype),
                }
                self._yoco_batch_logits_calls.append(call)
                ref_logits = self._yoco_batch_ref_logits
                if self._yoco_batch_logits_capture is None and logits.dim() == 2:
                    self._yoco_batch_logits_chunks.append(logits.float().detach().clone())
                    cand_logits = torch.cat(self._yoco_batch_logits_chunks, dim=0)
                    if cand_logits.shape[0] > ref_logits.shape[0]:
                        cand_logits = cand_logits[: ref_logits.shape[0]]
                    if list(cand_logits.shape) != list(ref_logits.shape):
                        return logits
                    self._yoco_batch_logits_capture = {
                        "matrix": _matrix_stats(
                            "batch_next_token_logits",
                            ref_logits,
                            cand_logits,
                        ),
                        "distribution": _distribution_divergence(
                            ref_logits,
                            cand_logits,
                        ),
                        "pairwise_distribution": _pairwise_distribution_divergence(
                            ref_logits,
                            cand_logits,
                        ),
                    }
            return logits

        model.compute_logits = types.MethodType(wrapped_compute_logits, model)

    return {"capture_installed": True, "ref_shape": list(ref_logits_cpu.shape)}


def _get_vllm_batch_logits_capture(model: torch.nn.Module) -> dict[str, Any]:
    capture = getattr(model, "_yoco_batch_logits_capture", None)
    if capture is None:
        raise RuntimeError(
            "vLLM batch logits were not captured; "
            f"calls={getattr(model, '_yoco_batch_logits_calls', None)}"
        )
    return {
        **capture,
        "vllm_logits_calls": getattr(model, "_yoco_batch_logits_calls", []),
    }


def _get_vllm_projector_dtypes(model: torch.nn.Module) -> list[str]:
    projector = getattr(model, "vision_projector", None)
    if projector is None:
        raise RuntimeError("vLLM model has no vision_projector")
    return sorted({str(parameter.dtype) for parameter in projector.parameters()})


@torch.inference_mode()
def _run_vllm_capture(
    *,
    model_path: Path,
    tokenizer_path: Path,
    rendered_prompts: list[str],
    image_paths: list[str],
    ref_logits: torch.Tensor,
    max_model_len: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    enforce_eager: bool,
    enable_chunked_prefill: bool,
    enable_prefix_caching: bool,
    kv_sharing_fast_prefill: bool,
    seed: int,
    gpu_memory_utilization: float,
    flash_attn_version: int | None,
    quantization: str | None,
    dtype: str,
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    llm = None
    try:
        if quantization == "mxfp8":
            moe_backend = "auto"
        elif quantization == "fp8_per_block":
            moe_backend = "deep_gemm"
        else:
            moe_backend = "triton"
        llm_kwargs = {
            "model": str(model_path),
            "tokenizer": str(tokenizer_path),
            "trust_remote_code": True,
            "dtype": dtype,
            "seed": seed,
            "tensor_parallel_size": 1,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "gpu_memory_utilization": gpu_memory_utilization,
            "quantization": quantization,
            "moe_backend": moe_backend,
            "enforce_eager": enforce_eager,
            "enable_chunked_prefill": enable_chunked_prefill,
            "enable_prefix_caching": enable_prefix_caching,
            "kv_sharing_fast_prefill": kv_sharing_fast_prefill,
            "limit_mm_per_prompt": {"image": 1},
        }
        if flash_attn_version is not None:
            llm_kwargs["attention_config"] = {
                "flash_attn_version": flash_attn_version,
            }
        llm = LLM(**_supported_kwargs(LLM, llm_kwargs))
        llm.apply_model(
            partial(
                _capture_vllm_batch_logits,
                ref_logits_cpu=ref_logits,
            )
        )
        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))
        requests = [
            {"prompt": prompt, "multi_modal_data": {"image": image}}
            for prompt, image in zip(rendered_prompts, images)
        ]
        sampling = SamplingParams(
            temperature=0.0,
            top_p=0.9,
            max_tokens=1,
            seed=seed,
        )
        outputs = llm.generate(requests, sampling, use_tqdm=False)
        capture = llm.apply_model(_get_vllm_batch_logits_capture)[0]
        projector_dtypes = llm.apply_model(_get_vllm_projector_dtypes)[0]
        return {
            **capture,
            "vllm_projector_parameter_dtypes": projector_dtypes,
            "vllm_moe_backend": moe_backend,
            "generated_ids": [list(output.outputs[0].token_ids) for output in outputs],
            "generated_text": [output.outputs[0].text for output in outputs],
        }
    finally:
        _shutdown_llm(llm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--vllm_model", default=str(DEFAULT_VLLM_MODEL))
    parser.add_argument("--tokenizer_path", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--images", nargs="+", default=DEFAULT_IMAGES)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=0,
        help="Defaults to the longest reference prompt plus max_new_tokens.",
    )
    parser.add_argument(
        "--vllm_max_num_seqs",
        type=int,
        default=0,
        help="Defaults to the batch size; set to 1 to force serial vLLM scheduling.",
    )
    parser.add_argument(
        "--vllm_max_num_batched_tokens",
        type=int,
        default=0,
        help=(
            "Defaults to the full batch token count. Set a smaller value together "
            "with --enable_chunked_prefill to avoid pathological oversized kernels."
        ),
    )
    parser.add_argument(
        "--enable_chunked_prefill",
        action="store_true",
        help="Allow vLLM to split a logical request batch into bounded prefill chunks.",
    )
    parser.add_argument(
        "--enable_prefix_caching",
        action="store_true",
        help="Enable vLLM prefix caching for runtime-configuration comparisons.",
    )
    parser.add_argument(
        "--kv_sharing_fast_prefill",
        action="store_true",
        help="Enable vLLM KV-sharing fast prefill for runtime-configuration comparisons.",
    )
    parser.add_argument(
        "--vllm_execution_mode",
        choices=("eager", "cuda_graph"),
        default="eager",
        help="Use eager execution or FULL_AND_PIECEWISE CUDA graphs.",
    )
    parser.add_argument(
        "--reference_prompt_variant",
        choices=("with_bos", "no_bos"),
        default="no_bos",
        help=(
            "Use no_bos by default because vLLM receives rendered string prompts "
            "and does not include llm-train's explicit leading <sop> token."
        ),
    )
    parser.add_argument(
        "--reference_quant_mode",
        choices=("bfloat16", "mxfp8"),
        default="bfloat16",
    )
    parser.add_argument(
        "--vllm_quantization",
        choices=("none", "mxfp8", "fp8_per_block"),
        default="none",
    )
    parser.add_argument(
        "--vllm_dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--reference_llm_flash_attn_version",
        type=int,
        choices=(2, 4),
        default=2,
    )
    parser.add_argument(
        "--reference_vision_flash_attn_version",
        type=int,
        choices=(2, 4),
        default=None,
        help="Defaults to the checkpoint's saved use_cute setting.",
    )
    parser.add_argument(
        "--vllm_flash_attn_version",
        type=int,
        choices=(2, 3, 4),
        default=2,
    )
    parser.add_argument("--output_json", default="/data/wjh/yoco_vl_batch_kl.json")
    return parser.parse_args()


def main() -> int:
    if Path.cwd().name != "wjh-b200-h100":
        raise RuntimeError(f"Run from wjh-b200-h100, current cwd is {Path.cwd()}")

    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    if len(args.images) != len(args.prompts):
        raise ValueError(
            f"--images and --prompts must have the same length; got "
            f"{len(args.images)} images and {len(args.prompts)} prompts"
        )

    _add_paths()
    _configure_cuda_toolkit()
    warnings.filterwarnings(
        "ignore",
        message=r"Use explicit `struct\.scalar\.ptr` for pointer instead\.",
        category=DeprecationWarning,
    )
    os.environ.setdefault(
        "CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "1")
    )
    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    _seed_everything(args.seed)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(device)

    checkpoint_dir = Path(args.checkpoint_dir)
    vllm_model = Path(args.vllm_model)
    tokenizer_path = Path(args.tokenizer_path)
    for path in [checkpoint_dir, vllm_model, tokenizer_path, *map(Path, args.images)]:
        if not path.exists():
            raise FileNotFoundError(path)

    from vl_batch_infer import build_samples
    from vl_infer import (
        Tokenizer,
        load_language_model_and_projector,
        load_metadata,
        load_vision_tower,
        model_args_from_metadata,
    )

    metadata = load_metadata(checkpoint_dir)
    model_args = model_args_from_metadata(metadata, args.reference_quant_mode)
    model_args.use_cute = args.reference_llm_flash_attn_version == 4
    if args.reference_vision_flash_attn_version is None:
        reference_vision_fa_version = (
            4 if metadata["modelargs"].get("use_cute", False) else 2
        )
    else:
        reference_vision_fa_version = args.reference_vision_flash_attn_version
    vision_attn_implementation = (
        "flash_attention_cute"
        if reference_vision_fa_version == 4
        else "flash_attention_2"
    )
    tokenizer = Tokenizer(str(tokenizer_path))
    model, projector, vision_tower_state = load_language_model_and_projector(
        checkpoint_dir,
        model_args,
        device,
        checkpoint_load_mode="auto",
    )
    reference_projector_parameter_dtypes = sorted(
        {str(parameter.dtype) for parameter in projector.parameters()}
    )
    vision_tower = load_vision_tower(
        model_args.vision_encoder_path,
        device,
        vision_attn_implementation,
        vision_state=vision_tower_state,
    )
    del vision_tower_state

    sample_args = argparse.Namespace(
        images=args.images,
        prompts=args.prompts,
        system_prompt=args.system_prompt,
        enable_thinking=False,
    )
    samples, vision_seconds = build_samples(
        sample_args,
        tokenizer,
        model_args,
        vision_tower,
        projector,
        device,
    )
    native_prompt_ids = [sample.prompt_ids for sample in samples]
    native_image_masks = [sample.image_mask for sample in samples]
    reference_prompt_ids = native_prompt_ids
    reference_image_masks = native_image_masks
    stripped_leading_bos = [False for _ in samples]
    if args.reference_prompt_variant == "no_bos":
        reference_prompt_ids = []
        reference_image_masks = []
        stripped_leading_bos = []
        for sample in samples:
            prompt_ids = sample.prompt_ids
            image_mask = sample.image_mask
            if prompt_ids.numel() > 0 and int(prompt_ids[0].item()) == int(
                tokenizer.bos_id
            ):
                prompt_ids = prompt_ids[1:].contiguous()
                image_mask = image_mask[1:].contiguous()
                stripped_leading_bos.append(True)
            else:
                stripped_leading_bos.append(False)
            reference_prompt_ids.append(prompt_ids)
            reference_image_masks.append(image_mask)
    ref_logits = _reference_batch_next_logits(
        model=model,
        prompt_ids=reference_prompt_ids,
        image_masks=reference_image_masks,
        projected_image_features=[sample.projected_image_features for sample in samples],
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    ref_top1 = torch.argmax(ref_logits, dim=-1).tolist()
    del model, projector, vision_tower
    gc.collect()
    torch.cuda.empty_cache()

    max_prompt_tokens = max(int(ids.numel()) for ids in reference_prompt_ids)
    required_max_model_len = max_prompt_tokens + args.max_new_tokens
    max_model_len = args.vllm_max_model_len or required_max_model_len
    if max_model_len < required_max_model_len:
        raise ValueError(
            "--vllm_max_model_len must be at least the longest prompt plus "
            f"max_new_tokens ({required_max_model_len}); got {max_model_len}"
        )
    full_batch_tokens = (
        sum(int(ids.numel()) for ids in reference_prompt_ids)
        + len(samples) * args.max_new_tokens
    )
    max_num_batched_tokens = args.vllm_max_num_batched_tokens or full_batch_tokens
    if max_num_batched_tokens < max_model_len:
        raise ValueError(
            "--vllm_max_num_batched_tokens must be at least max_model_len "
            f"({max_model_len}); got {max_num_batched_tokens}"
        )
    max_num_seqs = args.vllm_max_num_seqs or len(samples)
    vllm_result = _run_vllm_capture(
        model_path=vllm_model,
        tokenizer_path=tokenizer_path,
        rendered_prompts=[sample.rendered_prompt for sample in samples],
        image_paths=args.images,
        ref_logits=ref_logits,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        enforce_eager=args.vllm_execution_mode == "eager",
        enable_chunked_prefill=args.enable_chunked_prefill,
        enable_prefix_caching=args.enable_prefix_caching,
        kv_sharing_fast_prefill=args.kv_sharing_fast_prefill,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        flash_attn_version=args.vllm_flash_attn_version,
        quantization=(
            None if args.vllm_quantization == "none" else args.vllm_quantization
        ),
        dtype=args.vllm_dtype,
    )

    result = {
        "settings": {
            "checkpoint_dir": str(checkpoint_dir),
            "vllm_model": str(vllm_model),
            "tokenizer_path": str(tokenizer_path),
            "images": args.images,
            "prompts": args.prompts,
            "system_prompt": args.system_prompt,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": len(samples),
            "native_prompt_tokens_with_bos": [
                int(sample.prompt_ids.numel()) for sample in samples
            ],
            "reference_prompt_variant": args.reference_prompt_variant,
            "reference_prompt_tokens": [
                int(ids.numel()) for ids in reference_prompt_ids
            ],
            "reference_stripped_leading_bos": stripped_leading_bos,
            "image_tokens": [int(sample.image_tokens) for sample in samples],
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "full_batch_tokens": full_batch_tokens,
            "vllm_max_num_seqs": max_num_seqs,
            "vllm_execution_mode": args.vllm_execution_mode,
            "vllm_enable_chunked_prefill": args.enable_chunked_prefill,
            "vllm_enable_prefix_caching": args.enable_prefix_caching,
            "vllm_kv_sharing_fast_prefill": args.kv_sharing_fast_prefill,
            "reference_quant_mode": model_args.quant_mode,
            "reference_llm_attn": (
                "flash_attention_cute"
                if args.reference_llm_flash_attn_version == 4
                else "flash_attention_2"
            ),
            "reference_vision_attn": vision_attn_implementation,
            "vllm_dtype": args.vllm_dtype,
            "reference_projector_parameter_dtypes": (
                reference_projector_parameter_dtypes
            ),
            "vllm_projector_parameter_dtypes": vllm_result[
                "vllm_projector_parameter_dtypes"
            ],
            "vllm_quantization": (
                None
                if args.vllm_quantization == "none"
                else args.vllm_quantization
            ),
            "vllm_moe_backend": vllm_result["vllm_moe_backend"],
            "vllm_flash_attn_version": args.vllm_flash_attn_version,
            "vision_seconds": vision_seconds,
            "vllm_logits_calls": vllm_result["vllm_logits_calls"],
        },
        "reference_top1": [int(x) for x in ref_top1],
        "vllm_generated_ids": vllm_result["generated_ids"],
        "vllm_generated_text": vllm_result["generated_text"],
        "logits_matrix": vllm_result["matrix"],
        "distribution": vllm_result["distribution"],
        "pairwise_distribution": vllm_result["pairwise_distribution"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as writer:
            json.dump(result, writer, ensure_ascii=False, indent=2)
            writer.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
