#!/usr/bin/env python
"""Compare llm-train YOCO-VL tensors against the vLLM implementation.

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
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
LLM_TRAIN_LLM_ROOT = Path(
    os.environ.get("LLM_TRAIN_LLM_ROOT", "/root/workspace/llm-train/llm")
)
DEFAULT_NATIVE_CHECKPOINT = Path("/data/wjh/updates_3000")
DEFAULT_VLLM_MODEL = Path("/data/wjh/updates_3000-hf-vl")
DEFAULT_TOKENIZER = Path("/mnt/msranlp/yutao/hf_cache/agens_vl_tokenizer_0622")
DEFAULT_IMAGE = Path("/home/v-jiahaowang/workspace/llm-train/workspace/dog.jpeg")
DEFAULT_SYSTEM_PROMPT = "You are a helpful and friendly AI assistant."
DEFAULT_PROMPT = "Describe this image in detail."
IMAGE_PLACEHOLDER = "<image>"
VISION_PROJECTOR_PREFIX = "vision_projector."
VISION_TOWER_PREFIX = "vision_tower."
TRAINING_ONLY_STATE_KEYS = {"moe_loss.accum_expert_cnt"}


@dataclass
class TensorStats:
    name: str
    shape: list[int]
    ref_dtype: str
    cand_dtype: str
    max_abs: float
    mean_abs: float
    rms_abs: float
    ref_rms: float
    rel_rms: float
    max_rel: float
    cosine: float
    allclose_atol_1e_3_rtol_1e_3: bool
    allclose_atol_1e_2_rtol_1e_2: bool


def _add_paths() -> None:
    paths = (str(REPO_ROOT), str(LLM_TRAIN_LLM_ROOT))
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)
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


@contextmanager
def _default_device_and_dtype(device: torch.device, dtype: torch.dtype):
    old_device = torch.get_default_device()
    old_dtype = torch.get_default_dtype()
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_device(old_device)
        torch.set_default_dtype(old_dtype)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


def _render_prompt(
    tokenizer,
    prompt: str,
    system_prompt: str | None,
    enable_thinking: bool,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": f"{IMAGE_PLACEHOLDER}\n{prompt}"})
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        clear_thinking=not enable_thinking,
    )
    if not enable_thinking:
        rendered = rendered.replace(
            "<|assistant|><think></think>", "<|assistant|></think>"
        )
    if rendered.count(IMAGE_PLACEHOLDER) != 1:
        raise ValueError(
            f"Expected one {IMAGE_PLACEHOLDER}, got {rendered.count(IMAGE_PLACEHOLDER)}"
        )
    return rendered


def _encode_no_special(tokenizer, text: str) -> list[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text, bos=False, eos=False)


def _pack_prompt_with_image(
    tokenizer,
    rendered_prompt: str,
    num_image_tokens: int,
    *,
    bos_id: int,
    image_start_id: int,
    image_end_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    before, after = rendered_prompt.split(IMAGE_PLACEHOLDER)
    before_ids = _encode_no_special(tokenizer, before)
    after_ids = _encode_no_special(tokenizer, after)
    token_ids = (
        [bos_id]
        + before_ids
        + [image_start_id]
        + [0] * num_image_tokens
        + [image_end_id]
        + after_ids
    )
    image_mask = (
        [False] * (1 + len(before_ids) + 1)
        + [True] * num_image_tokens
        + [False] * (1 + len(after_ids))
    )
    return torch.tensor(token_ids, dtype=torch.long), torch.tensor(
        image_mask, dtype=torch.bool
    )


def _model_args_from_metadata(metadata: dict[str, Any], quant_mode: str):
    from arch.config import ModelArgs

    args = ModelArgs()
    for key, value in metadata["modelargs"].items():
        if hasattr(args, key):
            setattr(args, key, value)
    if quant_mode != "checkpoint":
        args.quant_mode = quant_mode
    args.use_cute = False
    args.moe_fwd_bwd_overlap = False
    args.__post_init__()
    return args


def _normalize_merged_key(key: str) -> str:
    if key.startswith("backbone."):
        key = key[len("backbone.") :]
    if key.startswith("model."):
        key = key[len("model.") :]
    return key


def _extract_reference_state(
    state: dict[str, torch.Tensor],
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    tok_embeddings: torch.Tensor | None = None
    language_state: dict[str, torch.Tensor] = {}
    projector_state: dict[str, torch.Tensor] = {}
    vision_state: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        key = _normalize_merged_key(raw_key)
        if key in TRAINING_ONLY_STATE_KEYS:
            continue
        if key == "tok_embeddings.weight":
            tok_embeddings = value
        if key.startswith(VISION_PROJECTOR_PREFIX):
            projector_state[key[len(VISION_PROJECTOR_PREFIX) :]] = value
        elif key.startswith(VISION_TOWER_PREFIX):
            vision_state[key[len(VISION_TOWER_PREFIX) :]] = value
        else:
            language_state[key] = value
    if tok_embeddings is None:
        raise KeyError("Missing tok_embeddings.weight in native checkpoint")
    if not language_state:
        raise KeyError("Missing language model weights in native checkpoint")
    if not projector_state:
        raise KeyError("Missing vision_projector weights in native checkpoint")
    if not vision_state:
        raise KeyError("Missing vision_tower weights in native checkpoint")
    return tok_embeddings, language_state, projector_state, vision_state


def _adapt_vision_tower_state(
    vision_state: dict[str, torch.Tensor],
    tower: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    vt_state = dict(vision_state)
    proj_key = "patch_embed.proj.weight"
    if (
        proj_key in vt_state
        and isinstance(tower.patch_embed.proj, torch.nn.Linear)
        and vt_state[proj_key].dim() == 4
    ):
        weight = vt_state[proj_key]
        vt_state[proj_key] = weight.reshape(weight.shape[0], -1)
    elif (
        proj_key in vt_state
        and isinstance(tower.patch_embed.proj, torch.nn.Conv2d)
        and vt_state[proj_key].dim() == 2
    ):
        weight = vt_state[proj_key]
        out_dim, flat_dim = weight.shape
        in_dim = tower.patch_embed.proj.in_channels
        kernel_h, kernel_w = tower.patch_embed.patch_size
        expected = in_dim * kernel_h * kernel_w
        if flat_dim != expected:
            raise ValueError(
                f"Cannot reshape {proj_key}: flat_dim={flat_dim}, expected={expected}"
            )
        vt_state[proj_key] = weight.reshape(out_dim, in_dim, kernel_h, kernel_w)
    return vt_state


def _resolve_reference_vision_attn(
    metadata: dict[str, Any],
    requested: str,
) -> str:
    if requested != "checkpoint":
        return requested
    return (
        "flash_attention_cute"
        if metadata["modelargs"].get("use_cute", False)
        else "flash_attention_2"
    )


@torch.inference_mode()
def _run_reference_language_prefill(
    *,
    language_model: torch.nn.Module,
    model_args: Any,
    input_ids: torch.Tensor,
    image_mask: torch.Tensor,
    projected: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    from arch.model import create_kv_cache

    prompt_length = int(input_ids.numel())
    max_length = prompt_length + max_new_tokens
    if max_length > model_args.max_seq_len:
        raise ValueError(
            f"prompt ({prompt_length}) + max_new_tokens ({max_new_tokens}) exceeds "
            f"model max_seq_len ({model_args.max_seq_len})"
        )

    input_ids_device = input_ids.to(device=device)
    image_mask_device = image_mask.to(device=device)
    projected_device = projected.to(device=device)
    image_features = torch.zeros(
        prompt_length,
        model_args.d_model,
        dtype=projected_device.dtype,
        device=device,
    )
    image_features[image_mask_device] = projected_device
    kv_cache = create_kv_cache(
        model_args,
        batch_size=1,
        max_seq_len=max_length,
        dtype=torch.bfloat16,
        device=device,
    )
    positions = torch.arange(prompt_length, device=device, dtype=torch.int32)
    prefill_context = {
        "kv_cache": kv_cache,
        "cu_seqlens_q": torch.tensor(
            [0, prompt_length], device=device, dtype=torch.int32
        ),
        "cu_seqlens_k": torch.tensor(
            [0, prompt_length], device=device, dtype=torch.int32
        ),
        "max_seqlen_q": prompt_length,
        "max_seqlen_k": prompt_length,
        "positions": positions,
        "slot_mapping": positions,
        "layer_index": 0,
    }
    final_hidden, _, _ = language_model(
        input_ids_device,
        context=prefill_context,
        last_hidden_only=True,
        image_features=image_features,
        image_input_mask=image_mask_device,
    )
    next_token_logits = language_model.output(final_hidden[-1:]).float().cpu().clone()
    final_hidden = final_hidden.float().cpu().clone()
    del kv_cache, input_ids_device, image_mask_device
    del projected_device, image_features, positions, prefill_context
    torch.cuda.empty_cache()
    return final_hidden, next_token_logits


@torch.inference_mode()
def _collect_reference_tensors(
    *,
    checkpoint_dir: Path,
    tokenizer_path: Path,
    image_path: Path,
    prompt: str,
    system_prompt: str | None,
    enable_thinking: bool,
    max_new_tokens: int,
    quant_mode: str,
    vision_attn: str,
    device: torch.device,
) -> dict[str, Any]:
    from arch.model import Model
    from arch.projector import PatchMergerMLP
    from arch.vision_encoder import MoonViT3dVisionTower, tpool_patch_merger
    from data.tokenizer import Tokenizer
    from data.vision_processing import process_image

    metadata = _load_json(checkpoint_dir / "metadata.json")
    model_args = _model_args_from_metadata(metadata, quant_mode)
    resolved_vision_attn = _resolve_reference_vision_attn(metadata, vision_attn)

    tokenizer = Tokenizer(str(tokenizer_path))
    rendered_prompt = _render_prompt(
        tokenizer.tok,
        prompt,
        system_prompt,
        enable_thinking,
    )

    with Image.open(image_path) as image_reader:
        image = image_reader.convert("RGB")
    processed = process_image(
        image,
        patch_size=model_args.vision_patch_size,
        merge_kernel_size=model_args.vision_merge_kernel_size,
        max_image_tokens=model_args.vision_max_image_tokens,
        align_mode=model_args.vision_align_mode,
        normalize_in_patch_layout=True,
    )
    pixel_values = torch.from_numpy(processed["pixel_values"])
    grid_thw = torch.from_numpy(processed["grid_thw"]).reshape(1, 3)
    num_image_tokens = int(processed["num_tokens"])

    input_ids, image_mask = _pack_prompt_with_image(
        tokenizer,
        rendered_prompt,
        num_image_tokens,
        bos_id=tokenizer.bos_id,
        image_start_id=tokenizer.image_start_id,
        image_end_id=tokenizer.image_end_id,
    )

    state = torch.load(
        checkpoint_dir / "model_state_rank_0.pth",
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    tok_embeddings, language_state, projector_state, vision_state = (
        _extract_reference_state(state)
    )
    token_embeddings = tok_embeddings[input_ids].float().cpu().clone()

    projector = PatchMergerMLP(
        vit_hidden_dim=model_args.vision_encoder_hidden_size,
        merge_kernel_size=model_args.vision_merge_kernel_size,
        target_dim=model_args.d_model,
        ln_eps=model_args.norm_eps,
    )
    projector.load_state_dict(projector_state, strict=True)
    projector.to(device=device)
    projector.eval().requires_grad_(False)

    tower = MoonViT3dVisionTower(
        attn_implementation=resolved_vision_attn,
        patch_embed_impl="conv2d",
    )
    vt_state = _adapt_vision_tower_state(vision_state, tower)
    missing, unexpected = tower.load_state_dict(vt_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Vision tower checkpoint mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    tower.to(device=device, dtype=torch.bfloat16)
    tower.eval().requires_grad_(False)

    pixel_values_bf16 = pixel_values.to(device=device, dtype=torch.bfloat16)
    grid_thw_device = grid_thw.to(device=device)
    vision_patch_embed = tower.patch_embed(pixel_values_bf16, grid_thw_device)
    vision_encoder = tower.encoder(vision_patch_embed, grid_thw_device)
    reference_lengths = torch.cat(
        (
            torch.zeros(1, dtype=grid_thw_device.dtype, device=grid_thw_device.device),
            grid_thw_device[:, 0] * grid_thw_device[:, 1] * grid_thw_device[:, 2],
        )
    )
    reference_max_seqlen = int(reference_lengths.max().item())
    reference_cu_seqlens = reference_lengths.cumsum(dim=0, dtype=torch.int32)
    reference_rope = tower.encoder.rope_2d.get_freqs_cis(
        grid_thws=grid_thw_device,
        device=vision_patch_embed.device,
    )
    reference_block0 = tower.encoder.blocks[0]
    reference_block0_norm0 = reference_block0.norm0(vision_patch_embed)
    reference_block0_attn = reference_block0.attention_qkvpacked(
        reference_block0_norm0,
        reference_cu_seqlens,
        reference_max_seqlen,
        reference_rope,
    )
    reference_block0_after_attn = vision_patch_embed + reference_block0_attn
    reference_block0_norm1 = reference_block0.norm1(reference_block0_after_attn)
    reference_block0_mlp = reference_block0.mlp(reference_block0_norm1)
    reference_block0_out = reference_block0_after_attn + reference_block0_mlp
    raw_list = tpool_patch_merger(
        vision_encoder,
        grid_thw_device,
        merge_kernel_size=tower.merge_kernel_size,
    )
    if len(raw_list) != 1:
        raise RuntimeError(f"Expected one reference image tensor, got {len(raw_list)}")
    vision_raw = raw_list[0].float().cpu().clone()
    projected = projector.forward_flat(raw_list[0].float()).float().cpu().clone()
    combined = token_embeddings.clone()
    combined[image_mask] = projected
    vision_patch_embed_cpu = vision_patch_embed.float().cpu().clone()
    vision_block0_norm0_cpu = reference_block0_norm0.float().cpu().clone()
    vision_block0_attn_cpu = reference_block0_attn.float().cpu().clone()
    vision_block0_after_attn_cpu = reference_block0_after_attn.float().cpu().clone()
    vision_block0_norm1_cpu = reference_block0_norm1.float().cpu().clone()
    vision_block0_mlp_cpu = reference_block0_mlp.float().cpu().clone()
    vision_block0_out_cpu = reference_block0_out.float().cpu().clone()
    vision_encoder_cpu = vision_encoder.float().cpu().clone()

    del projector, tower, raw_list
    del vision_patch_embed, vision_encoder, reference_rope, reference_block0
    del reference_block0_norm0, reference_block0_attn, reference_block0_after_attn
    del reference_block0_norm1, reference_block0_mlp, reference_block0_out
    del pixel_values_bf16, grid_thw_device, reference_lengths, reference_cu_seqlens
    gc.collect()
    torch.cuda.empty_cache()

    prompt_length = int(input_ids.numel())
    max_length = prompt_length + max_new_tokens
    if max_length > model_args.max_seq_len:
        raise ValueError(
            f"prompt ({prompt_length}) + max_new_tokens ({max_new_tokens}) exceeds "
            f"model max_seq_len ({model_args.max_seq_len})"
        )

    with _default_device_and_dtype(device, torch.bfloat16):
        language_model = Model(model_args)
    language_model.load_state_dict(language_state, strict=True)
    language_model.eval().requires_grad_(False)

    final_hidden, next_token_logits = _run_reference_language_prefill(
        language_model=language_model,
        model_args=model_args,
        input_ids=input_ids,
        image_mask=image_mask,
        projected=projected,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    final_hidden_no_bos = None
    next_token_logits_no_bos = None
    if input_ids.numel() > 1 and int(input_ids[0].item()) == int(tokenizer.bos_id):
        final_hidden_no_bos, next_token_logits_no_bos = (
            _run_reference_language_prefill(
                language_model=language_model,
                model_args=model_args,
                input_ids=input_ids[1:].contiguous(),
                image_mask=image_mask[1:].contiguous(),
                projected=projected,
                max_new_tokens=max_new_tokens,
                device=device,
            )
        )

    del state, tok_embeddings, language_state, projector_state, vision_state, vt_state
    del language_model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "metadata": metadata,
        "model_args": model_args,
        "resolved_vision_attn": resolved_vision_attn,
        "rendered_prompt": rendered_prompt,
        "image_size": image.size,
        "pixel_values": pixel_values.contiguous().cpu(),
        "grid_thw": grid_thw.contiguous().cpu(),
        "num_image_tokens": num_image_tokens,
        "input_ids": input_ids,
        "image_mask": image_mask,
        "token_embeddings": token_embeddings,
        "vision_patch_embed": vision_patch_embed_cpu,
        "vision_block0_norm0": vision_block0_norm0_cpu,
        "vision_block0_attn": vision_block0_attn_cpu,
        "vision_block0_after_attn": vision_block0_after_attn_cpu,
        "vision_block0_norm1": vision_block0_norm1_cpu,
        "vision_block0_mlp": vision_block0_mlp_cpu,
        "vision_block0_out": vision_block0_out_cpu,
        "vision_encoder": vision_encoder_cpu,
        "vision_raw": vision_raw,
        "projected_image_features": projected,
        "combined_input_embeddings": combined,
        "final_hidden_matrix": final_hidden,
        "next_token_logits": next_token_logits,
        "final_hidden_matrix_no_bos": final_hidden_no_bos,
        "next_token_logits_no_bos": next_token_logits_no_bos,
    }


def _collect_vllm_model_tensors(
    model: torch.nn.Module,
    *,
    input_ids_cpu: torch.Tensor,
    image_mask_cpu: torch.Tensor,
    pixel_values_cpu: torch.Tensor,
    image_grid_thws_cpu: torch.Tensor,
) -> dict[str, torch.Tensor]:
    from vllm.model_executor.models.yoco_vl import YOCOVLImagePixelInputs

    device = next(model.parameters()).device
    input_ids = input_ids_cpu.to(device=device)
    image_mask = image_mask_cpu.to(device=device)
    pixel_values = pixel_values_cpu.to(device=device)
    image_grid_thws = image_grid_thws_cpu.to(device=device)

    with torch.inference_mode():
        image_input = YOCOVLImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            image_grid_thws=image_grid_thws,
        )
        target_dtype = model.vision_tower.patch_embed.proj.weight.dtype
        vision_pixel_values = pixel_values.to(dtype=target_dtype)
        vision_patch_embed = model.vision_tower.patch_embed(
            vision_pixel_values,
            image_grid_thws,
        )
        vision_encoder = model.vision_tower.encoder(
            vision_patch_embed,
            image_grid_thws,
        )
        vllm_lengths = torch.cat(
            (
                torch.zeros(
                    1,
                    device=image_grid_thws.device,
                    dtype=image_grid_thws.dtype,
                ),
                image_grid_thws[:, 0]
                * image_grid_thws[:, 1]
                * image_grid_thws[:, 2],
            )
        )
        vllm_cu_seqlens = vllm_lengths.cumsum(dim=0, dtype=torch.int32)
        vllm_rope = model.vision_tower.encoder.rope_2d.get_freqs_cis(
            grid_thws=image_grid_thws,
            device=vision_patch_embed.device,
        )
        vllm_block0 = model.vision_tower.encoder.blocks[0]
        vllm_block0_norm0 = vllm_block0.norm0(vision_patch_embed)
        vllm_block0_attn = vllm_block0.attention_qkvpacked(
            vllm_block0_norm0,
            vllm_cu_seqlens,
            rope_freqs_cis=vllm_rope,
        )
        vllm_block0_after_attn = vision_patch_embed + vllm_block0_attn
        vllm_block0_norm1 = vllm_block0.norm1(vllm_block0_after_attn)
        vllm_block0_mlp = vllm_block0.mlp(vllm_block0_norm1)
        vllm_block0_out = vllm_block0_after_attn + vllm_block0_mlp
        raw_features = model._process_image_pixels(image_input)
        if not isinstance(raw_features, (list, tuple)) or len(raw_features) != 1:
            raise RuntimeError(
                "Expected one vLLM raw image tensor, got "
                f"{type(raw_features).__name__}"
            )
        vision_raw = raw_features[0].float()
        projected = model.vision_projector(vision_raw).float()
        token_embeddings = model.language_model.get_input_embeddings(input_ids).float()
        combined = token_embeddings.clone()
        combined[image_mask] = projected

    return {
        "token_embeddings": token_embeddings.cpu().clone(),
        "vision_patch_embed": vision_patch_embed.float().cpu().clone(),
        "vision_block0_norm0": vllm_block0_norm0.float().cpu().clone(),
        "vision_block0_attn": vllm_block0_attn.float().cpu().clone(),
        "vision_block0_after_attn": vllm_block0_after_attn.float().cpu().clone(),
        "vision_block0_norm1": vllm_block0_norm1.float().cpu().clone(),
        "vision_block0_mlp": vllm_block0_mlp.float().cpu().clone(),
        "vision_block0_out": vllm_block0_out.float().cpu().clone(),
        "vision_encoder": vision_encoder.float().cpu().clone(),
        "vision_raw": vision_raw.cpu().clone(),
        "projected_image_features": projected.cpu().clone(),
        "combined_input_embeddings": combined.cpu().clone(),
    }


def _install_vllm_final_capture(
    model: torch.nn.Module,
    *,
    prompt_len: int,
    ref_final_hidden_cpu: torch.Tensor,
    ref_next_token_logits_cpu: torch.Tensor,
) -> dict[str, Any]:
    import types

    device = next(model.parameters()).device
    model._yoco_capture_prompt_len = int(prompt_len)
    model._yoco_capture_forward_calls = []
    model._yoco_capture_logits_calls = []
    model._yoco_ref_final_hidden = ref_final_hidden_cpu.to(device=device)
    model._yoco_ref_next_token_logits = ref_next_token_logits_cpu.to(device=device)
    model._yoco_final_hidden_stats = None
    model._yoco_next_token_hidden_stats = None
    model._yoco_next_token_logits_stats = None
    model._yoco_final_hidden_divergence = None
    model._yoco_next_token_logits_matrix_divergence = None
    model._yoco_next_token_logits_distribution = None

    if not hasattr(model, "_yoco_orig_forward"):
        model._yoco_orig_forward = model.forward

        def wrapped_forward(self, *args, **kwargs):
            output = self._yoco_orig_forward(*args, **kwargs)
            if isinstance(output, torch.Tensor):
                self._yoco_capture_forward_calls.append(
                    {
                        "shape": list(output.shape),
                        "dtype": str(output.dtype),
                        "device": str(output.device),
                    }
                )
                prompt_length = int(self._yoco_capture_prompt_len)
                if (
                    self._yoco_final_hidden_stats is None
                    and output.dim() >= 2
                    and output.shape[0] >= prompt_length
                ):
                    ref_hidden = self._yoco_ref_final_hidden
                    cand_hidden = output[:prompt_length].float()
                    self._yoco_final_hidden_stats = asdict(
                        _tensor_stats(
                            "final_hidden_matrix",
                            ref_hidden,
                            cand_hidden,
                        )
                    )
                    self._yoco_final_hidden_divergence = _matrix_divergence(
                        "final_hidden_matrix",
                        ref_hidden,
                        cand_hidden,
                    )
            return output

        model.forward = types.MethodType(wrapped_forward, model)

    if not hasattr(model, "_yoco_orig_compute_logits"):
        model._yoco_orig_compute_logits = model.compute_logits

        def wrapped_compute_logits(self, hidden_states, *args, **kwargs):
            logits = self._yoco_orig_compute_logits(hidden_states, *args, **kwargs)
            if isinstance(logits, torch.Tensor):
                self._yoco_capture_logits_calls.append(
                    {
                        "hidden_shape": list(hidden_states.shape),
                        "logits_shape": list(logits.shape),
                        "hidden_dtype": str(hidden_states.dtype),
                        "logits_dtype": str(logits.dtype),
                        "device": str(logits.device),
                    }
                )
                if self._yoco_next_token_logits_stats is None:
                    ref_hidden = self._yoco_ref_final_hidden[-hidden_states.shape[0] :]
                    ref_logits = self._yoco_ref_next_token_logits
                    cand_hidden = hidden_states.float()
                    cand_logits = logits.float()
                    self._yoco_next_token_hidden_stats = asdict(
                        _tensor_stats(
                            "next_token_hidden",
                            ref_hidden,
                            cand_hidden,
                        )
                    )
                    self._yoco_next_token_logits_stats = asdict(
                        _tensor_stats(
                            "next_token_logits",
                            ref_logits,
                            cand_logits,
                        )
                    )
                    self._yoco_next_token_logits_matrix_divergence = (
                        _matrix_divergence(
                            "next_token_logits",
                            ref_logits,
                            cand_logits,
                        )
                    )
                    self._yoco_next_token_logits_distribution = (
                        _distribution_divergence(
                            "next_token_logits",
                            ref_logits,
                            cand_logits,
                        )
                    )
            return logits

        model.compute_logits = types.MethodType(wrapped_compute_logits, model)

    return {"capture_installed": True, "prompt_len": int(prompt_len)}


def _get_vllm_final_capture(model: torch.nn.Module) -> dict[str, Any]:
    final_hidden_stats = getattr(model, "_yoco_final_hidden_stats", None)
    next_token_hidden_stats = getattr(model, "_yoco_next_token_hidden_stats", None)
    next_token_logits_stats = getattr(model, "_yoco_next_token_logits_stats", None)
    if final_hidden_stats is None:
        raise RuntimeError(
            "vLLM final hidden matrix stats were not captured; "
            f"forward_calls={getattr(model, '_yoco_capture_forward_calls', None)}"
        )
    if next_token_logits_stats is None:
        raise RuntimeError(
            "vLLM next-token logits stats were not captured; "
            f"logits_calls={getattr(model, '_yoco_capture_logits_calls', None)}"
        )
    return {
        "final_tensor_diffs": [
            final_hidden_stats,
            next_token_hidden_stats,
            next_token_logits_stats,
        ],
        "final_divergence": {
            "final_hidden_matrix": getattr(
                model, "_yoco_final_hidden_divergence", None
            ),
            "next_token_logits_matrix": getattr(
                model, "_yoco_next_token_logits_matrix_divergence", None
            ),
            "next_token_logits_distribution": getattr(
                model, "_yoco_next_token_logits_distribution", None
            ),
        },
        "vllm_forward_calls": getattr(model, "_yoco_capture_forward_calls", []),
        "vllm_logits_calls": getattr(model, "_yoco_capture_logits_calls", []),
    }


@torch.inference_mode()
def _collect_vllm_tensors(
    *,
    model_path: Path,
    tokenizer_path: Path,
    image_path: Path,
    prompt: str,
    ref_input_ids: torch.Tensor,
    capture_prompt_len: int,
    ref_final_hidden: torch.Tensor,
    ref_next_token_logits: torch.Tensor,
    ref_image_mask: torch.Tensor,
    max_model_len: int,
    vllm_pixel_values: torch.Tensor,
    vllm_image_grid_thws: torch.Tensor,
    seed: int,
    gpu_memory_utilization: float,
    flash_attn_version: int | None,
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    llm = None
    try:
        llm_kwargs = {
            "model": str(model_path),
            "tokenizer": str(tokenizer_path),
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "seed": seed,
            "tensor_parallel_size": 1,
            "max_model_len": max_model_len,
            "max_num_seqs": 1,
            "max_num_batched_tokens": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "quantization": None,
            "moe_backend": "triton",
            "enforce_eager": True,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "limit_mm_per_prompt": {"image": 1},
        }
        if flash_attn_version is not None:
            llm_kwargs["attention_config"] = {
                "flash_attn_version": flash_attn_version,
            }
        llm = LLM(**_supported_kwargs(LLM, llm_kwargs))
        fn = partial(
            _collect_vllm_model_tensors,
            input_ids_cpu=ref_input_ids,
            image_mask_cpu=ref_image_mask,
            pixel_values_cpu=vllm_pixel_values,
            image_grid_thws_cpu=vllm_image_grid_thws,
        )
        tensors = llm.apply_model(fn)[0]
        llm.apply_model(
            partial(
                _install_vllm_final_capture,
                prompt_len=int(capture_prompt_len),
                ref_final_hidden_cpu=ref_final_hidden,
                ref_next_token_logits_cpu=ref_next_token_logits,
            )
        )
        sampling = SamplingParams(
            temperature=0.0,
            top_p=0.9,
            max_tokens=1,
        )
        with Image.open(image_path) as image_reader:
            image = image_reader.convert("RGB")
        generation_prompt = prompt
        outputs = llm.generate(
            [{"prompt": generation_prompt, "multi_modal_data": {"image": image}}],
            sampling,
            use_tqdm=False,
        )
        final_tensors = llm.apply_model(_get_vllm_final_capture)[0]
        completion = outputs[0].outputs[0]
        return {
            "prompt": prompt,
            "generation_prompt": generation_prompt,
            "generated_ids": list(completion.token_ids),
            "generated_text": completion.text,
            **tensors,
            **final_tensors,
        }
    finally:
        _shutdown_llm(llm)


def _tensor_stats(name: str, ref: torch.Tensor, cand: torch.Tensor) -> TensorStats:
    if tuple(ref.shape) != tuple(cand.shape):
        raise ValueError(
            f"{name}: shape mismatch ref={tuple(ref.shape)} cand={tuple(cand.shape)}"
        )
    ref_f = ref.detach().float()
    cand_f = cand.detach().float()
    diff = cand_f - ref_f
    abs_diff = diff.abs()
    ref_norm = torch.linalg.vector_norm(ref_f)
    cand_norm = torch.linalg.vector_norm(cand_f)
    denom = torch.clamp(ref_f.abs(), min=1e-6)
    cosine = torch.dot(ref_f.reshape(-1), cand_f.reshape(-1)) / torch.clamp(
        ref_norm * cand_norm, min=1e-12
    )
    return TensorStats(
        name=name,
        shape=list(ref.shape),
        ref_dtype=str(ref.dtype),
        cand_dtype=str(cand.dtype),
        max_abs=float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        mean_abs=float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
        rms_abs=float(torch.sqrt(torch.mean(diff.square())).item())
        if diff.numel()
        else 0.0,
        ref_rms=float(torch.sqrt(torch.mean(ref_f.square())).item())
        if ref_f.numel()
        else 0.0,
        rel_rms=float(
            (
                torch.sqrt(torch.mean(diff.square()))
                / torch.clamp(torch.sqrt(torch.mean(ref_f.square())), min=1e-12)
            ).item()
        )
        if diff.numel()
        else 0.0,
        max_rel=float((abs_diff / denom).max().item()) if abs_diff.numel() else 0.0,
        cosine=float(cosine.item()) if ref_f.numel() else 1.0,
        allclose_atol_1e_3_rtol_1e_3=bool(
            torch.allclose(ref_f, cand_f, atol=1e-3, rtol=1e-3)
        ),
        allclose_atol_1e_2_rtol_1e_2=bool(
            torch.allclose(ref_f, cand_f, atol=1e-2, rtol=1e-2)
        ),
    )


def _matrix_divergence(
    name: str,
    ref: torch.Tensor,
    cand: torch.Tensor,
) -> dict[str, Any]:
    if tuple(ref.shape) != tuple(cand.shape):
        raise ValueError(
            f"{name}: shape mismatch ref={tuple(ref.shape)} cand={tuple(cand.shape)}"
        )
    ref_f = ref.detach().float()
    cand_f = cand.detach().float()
    diff = cand_f - ref_f
    ref_norm = torch.linalg.vector_norm(ref_f.reshape(-1))
    cand_norm = torch.linalg.vector_norm(cand_f.reshape(-1))
    diff_norm = torch.linalg.vector_norm(diff.reshape(-1))
    cosine = torch.dot(ref_f.reshape(-1), cand_f.reshape(-1)) / torch.clamp(
        ref_norm * cand_norm, min=1e-12
    )
    result: dict[str, Any] = {
        "name": name,
        "shape": list(ref.shape),
        "ref_frobenius_norm": float(ref_norm.item()),
        "cand_frobenius_norm": float(cand_norm.item()),
        "diff_frobenius_norm": float(diff_norm.item()),
        "relative_frobenius": float((diff_norm / torch.clamp(ref_norm, min=1e-12)).item()),
        "global_cosine": float(cosine.item()) if ref_f.numel() else 1.0,
    }
    if ref_f.dim() >= 2 and ref_f.numel():
        ref_rows = ref_f.reshape(-1, ref_f.shape[-1])
        cand_rows = cand_f.reshape(-1, cand_f.shape[-1])
        row_cosine = (ref_rows * cand_rows).sum(dim=-1) / torch.clamp(
            torch.linalg.vector_norm(ref_rows, dim=-1)
            * torch.linalg.vector_norm(cand_rows, dim=-1),
            min=1e-12,
        )
        result.update(
            {
                "mean_row_cosine": float(row_cosine.mean().item()),
                "min_row_cosine": float(row_cosine.min().item()),
                "max_row_cosine": float(row_cosine.max().item()),
            }
        )
    return result


def _distribution_divergence(
    name: str,
    ref_logits: torch.Tensor,
    cand_logits: torch.Tensor,
) -> dict[str, Any]:
    if tuple(ref_logits.shape) != tuple(cand_logits.shape):
        raise ValueError(
            f"{name}: shape mismatch ref={tuple(ref_logits.shape)} "
            f"cand={tuple(cand_logits.shape)}"
        )
    ref = ref_logits.detach().float().reshape(-1, ref_logits.shape[-1])
    cand = cand_logits.detach().float().reshape(-1, cand_logits.shape[-1])
    ref_logp = torch.log_softmax(ref, dim=-1)
    cand_logp = torch.log_softmax(cand, dim=-1)
    ref_p = ref_logp.exp()
    cand_p = cand_logp.exp()
    midpoint = 0.5 * (ref_p + cand_p)
    midpoint_logp = torch.log(torch.clamp(midpoint, min=torch.finfo(torch.float32).tiny))
    kl_ref_to_cand = torch.sum(ref_p * (ref_logp - cand_logp), dim=-1)
    kl_cand_to_ref = torch.sum(cand_p * (cand_logp - ref_logp), dim=-1)
    js = 0.5 * torch.sum(ref_p * (ref_logp - midpoint_logp), dim=-1) + 0.5 * torch.sum(
        cand_p * (cand_logp - midpoint_logp), dim=-1
    )
    prob_abs_diff = (cand_p - ref_p).abs()
    ref_top1 = torch.argmax(ref, dim=-1)
    cand_top1 = torch.argmax(cand, dim=-1)
    result: dict[str, Any] = {
        "name": name,
        "shape": list(ref_logits.shape),
        "rows": int(ref.shape[0]),
        "vocab": int(ref.shape[-1]),
        "kl_ref_to_candidate_mean": float(kl_ref_to_cand.mean().item()),
        "kl_ref_to_candidate_max": float(kl_ref_to_cand.max().item()),
        "kl_candidate_to_ref_mean": float(kl_cand_to_ref.mean().item()),
        "kl_candidate_to_ref_max": float(kl_cand_to_ref.max().item()),
        "js_mean": float(js.mean().item()),
        "js_max": float(js.max().item()),
        "prob_l1_mean": float(prob_abs_diff.sum(dim=-1).mean().item()),
        "prob_linf_max": float(prob_abs_diff.max().item()),
        "top1_agreement": float((ref_top1 == cand_top1).float().mean().item()),
        "ref_entropy_mean": float(-(ref_p * ref_logp).sum(dim=-1).mean().item()),
        "candidate_entropy_mean": float(
            -(cand_p * cand_logp).sum(dim=-1).mean().item()
        ),
    }
    if ref.shape[0] <= 16:
        result["ref_top1"] = [int(x) for x in ref_top1.tolist()]
        result["candidate_top1"] = [int(x) for x in cand_top1.tolist()]
    return result


def _print_stats(stats: list[TensorStats]) -> None:
    print("\nTENSOR_DIFFS")
    for item in stats:
        print(
            f"{item.name}: shape={item.shape} "
            f"max_abs={item.max_abs:.6g} mean_abs={item.mean_abs:.6g} "
            f"rms_abs={item.rms_abs:.6g} rel_rms={item.rel_rms:.6g} "
            f"cos={item.cosine:.9f} "
            f"allclose(1e-3)={item.allclose_atol_1e_3_rtol_1e_3} "
            f"allclose(1e-2)={item.allclose_atol_1e_2_rtol_1e_2}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", default=str(DEFAULT_NATIVE_CHECKPOINT))
    parser.add_argument("--vllm_model", default=str(DEFAULT_VLLM_MODEL))
    parser.add_argument("--tokenizer_path", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--quant_mode",
        choices=("checkpoint", "bfloat16", "mxfp8"),
        default="bfloat16",
    )
    parser.add_argument(
        "--reference_vision_attn",
        choices=("checkpoint", "flash_attention_2", "flash_attention_cute"),
        default="checkpoint",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument(
        "--vllm_flash_attn_version",
        type=int,
        choices=(2, 3, 4),
        default=None,
    )
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def main() -> int:
    if Path.cwd().name != "wjh-b200-h100":
        raise RuntimeError(f"Run from wjh-b200-h100, current cwd is {Path.cwd()}")

    args = parse_args()
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
        raise RuntimeError("CUDA is required for this comparison")
    torch.cuda.set_device(device)

    checkpoint_dir = Path(args.checkpoint_dir)
    vllm_model = Path(args.vllm_model)
    tokenizer_path = Path(args.tokenizer_path)
    image_path = Path(args.image)
    for path in (checkpoint_dir, vllm_model, tokenizer_path, image_path):
        if not path.exists():
            raise FileNotFoundError(path)

    print("Loading reference tensors...")
    reference = _collect_reference_tensors(
        checkpoint_dir=checkpoint_dir,
        tokenizer_path=tokenizer_path,
        image_path=image_path,
        prompt=args.prompt,
        system_prompt=args.system_prompt,
        enable_thinking=False,
        max_new_tokens=args.max_new_tokens,
        quant_mode=args.quant_mode,
        vision_attn=args.reference_vision_attn,
        device=device,
    )

    from vllm.model_executor.models.yoco_vl import _process_image
    from vllm.transformers_utils.configs.yoco_vl import YOCOVLConfig

    hf_config = YOCOVLConfig(**_load_json(vllm_model / "config.json"))
    with Image.open(image_path) as image_reader:
        image = image_reader.convert("RGB")
    vllm_patches, vllm_grid_hw, vllm_num_tokens = _process_image(image, hf_config)
    vllm_pixel_values = torch.from_numpy(vllm_patches).contiguous()
    vllm_image_grid_thws = torch.from_numpy(
        np.concatenate(([1], vllm_grid_hw))
    ).reshape(1, 3).contiguous()

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    vllm_prompt = _render_prompt(tokenizer, args.prompt, args.system_prompt, False)

    checks = {
        "rendered_prompt_equal": reference["rendered_prompt"] == vllm_prompt,
        "input_ids_exact": True,
        "image_mask_exact": True,
        "grid_thw_exact": reference["grid_thw"][0].tolist()
        == vllm_image_grid_thws[0].tolist(),
        "num_image_tokens_equal": reference["num_image_tokens"] == vllm_num_tokens,
    }

    final_reference_variant = "with_bos"
    final_reference_hidden = reference["final_hidden_matrix"]
    final_reference_logits = reference["next_token_logits"]
    if reference["final_hidden_matrix_no_bos"] is not None:
        final_reference_variant = "no_bos_matches_vllm_string_prefill"
        final_reference_hidden = reference["final_hidden_matrix_no_bos"]
        final_reference_logits = reference["next_token_logits_no_bos"]
    max_model_len = int(reference["input_ids"].numel()) + args.max_new_tokens
    vllm_capture_prompt_len = int(final_reference_hidden.shape[0])
    print("Loading vLLM tensors...")
    vllm_tensors = _collect_vllm_tensors(
        model_path=vllm_model,
        tokenizer_path=tokenizer_path,
        image_path=image_path,
        prompt=vllm_prompt,
        ref_input_ids=reference["input_ids"],
        capture_prompt_len=vllm_capture_prompt_len,
        ref_final_hidden=final_reference_hidden,
        ref_next_token_logits=final_reference_logits,
        ref_image_mask=reference["image_mask"],
        max_model_len=max_model_len,
        vllm_pixel_values=vllm_pixel_values,
        vllm_image_grid_thws=vllm_image_grid_thws,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        flash_attn_version=args.vllm_flash_attn_version,
    )

    stats = [
        _tensor_stats("pixel_values", reference["pixel_values"], vllm_pixel_values),
        _tensor_stats(
            "token_embeddings",
            reference["token_embeddings"],
            vllm_tensors["token_embeddings"],
        ),
        _tensor_stats(
            "vision_patch_embed",
            reference["vision_patch_embed"],
            vllm_tensors["vision_patch_embed"],
        ),
        _tensor_stats(
            "vision_block0_norm0",
            reference["vision_block0_norm0"],
            vllm_tensors["vision_block0_norm0"],
        ),
        _tensor_stats(
            "vision_block0_attn",
            reference["vision_block0_attn"],
            vllm_tensors["vision_block0_attn"],
        ),
        _tensor_stats(
            "vision_block0_after_attn",
            reference["vision_block0_after_attn"],
            vllm_tensors["vision_block0_after_attn"],
        ),
        _tensor_stats(
            "vision_block0_norm1",
            reference["vision_block0_norm1"],
            vllm_tensors["vision_block0_norm1"],
        ),
        _tensor_stats(
            "vision_block0_mlp",
            reference["vision_block0_mlp"],
            vllm_tensors["vision_block0_mlp"],
        ),
        _tensor_stats(
            "vision_block0_out",
            reference["vision_block0_out"],
            vllm_tensors["vision_block0_out"],
        ),
        _tensor_stats(
            "vision_encoder",
            reference["vision_encoder"],
            vllm_tensors["vision_encoder"],
        ),
        _tensor_stats("vision_raw", reference["vision_raw"], vllm_tensors["vision_raw"]),
        _tensor_stats(
            "projected_image_features",
            reference["projected_image_features"],
            vllm_tensors["projected_image_features"],
        ),
        _tensor_stats(
            "combined_input_embeddings",
            reference["combined_input_embeddings"],
            vllm_tensors["combined_input_embeddings"],
        ),
    ]
    stats.extend(TensorStats(**item) for item in vllm_tensors["final_tensor_diffs"])
    divergence = vllm_tensors["final_divergence"]

    result = {
        "settings": {
            "checkpoint_dir": str(checkpoint_dir),
            "vllm_model": str(vllm_model),
            "tokenizer_path": str(tokenizer_path),
            "image": str(image_path),
            "image_size": list(reference["image_size"]),
            "prompt": args.prompt,
            "system_prompt": args.system_prompt,
            "max_new_tokens": args.max_new_tokens,
            "prompt_tokens": int(reference["input_ids"].numel()),
            "max_model_len": max_model_len,
            "vllm_capture_prompt_len": vllm_capture_prompt_len,
            "final_reference_variant": final_reference_variant,
            "quant_mode": args.quant_mode,
            "reference_vision_attn": reference["resolved_vision_attn"],
            "vllm_dtype": "bfloat16",
            "vllm_quantization": None,
            "vllm_moe_backend": "triton",
            "vllm_enable_prefix_caching": False,
            "vllm_flash_attn_version": args.vllm_flash_attn_version,
            "vllm_generation_prompt": vllm_tensors["generation_prompt"],
            "vllm_capture_forward_calls": vllm_tensors["vllm_forward_calls"],
            "vllm_capture_logits_calls": vllm_tensors["vllm_logits_calls"],
            "vllm_generated_ids": vllm_tensors["generated_ids"],
            "vllm_generated_text": vllm_tensors["generated_text"],
        },
        "checks": checks,
        "tensor_diffs": [asdict(item) for item in stats],
        "divergence": divergence,
    }

    print("\nCHECKS")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    _print_stats(stats)
    print("\nDIVERGENCE")
    print(json.dumps(divergence, ensure_ascii=False, indent=2))
    print("\nSUMMARY_JSON")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as writer:
            json.dump(result, writer, ensure_ascii=False, indent=2)
            writer.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
