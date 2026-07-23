from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from convert_to_hf import (  # noqa: E402
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    EOS_TOKEN_IDS,
    MIN_MODEL_MAX_LENGTH,
    PAD_TOKEN_ID,
    UNK_TOKEN_ID,
    add_bos_post_processor,
    convert_state_dict,
    ensure_chat_template_has_bos,
    save_sharded,
)


IMAGE_START_TOKEN_ID = 154830
IMAGE_END_TOKEN_ID = 154831
IMAGE_PLACEHOLDER_TOKEN_ID = 0
IMAGE_PLACEHOLDER = "<image>"


def load_metadata(input_dir: Path) -> dict:
    with (input_dir / "metadata.json").open(encoding="utf-8") as reader:
        return json.load(reader)


def load_model_state(input_dir: Path) -> dict[str, torch.Tensor]:
    model_path = input_dir / "model_state_rank_0.pth"
    print(f"Loading checkpoint from {model_path} ...", flush=True)
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def build_text_config(
    metadata: dict,
    *,
    quant_mode: str | None,
    quant_block_size: int | None,
    router_weights_normalized: bool,
) -> dict:
    ma = metadata.get("modelargs", metadata)

    head_dim = ma.get("head_dim") or ma["d_model"] // ma["head"]
    cross_kv_head = ma.get("cross_kv_head", ma["kv_head"])
    cross_head = ma.get("cross_head", ma["head"])
    qk_rms_clip = ma.get("qk_rms_clip", False)
    max_seq_len = max(MIN_MODEL_MAX_LENGTH, ma["max_seq_len"])

    quant_mode = (
        quant_mode if quant_mode is not None else ma.get("quant_mode", "bfloat16")
    )
    quant_block_size = (
        quant_block_size
        if quant_block_size is not None
        else ma.get("quant_block_size", 128)
    )

    return {
        "architectures": ["YOCOForCausalLM"],
        "model_type": "yoco",
        "torch_dtype": "bfloat16",
        "d_model": ma["d_model"],
        "d_ffn": ma["d_ffn"],
        "head": ma["head"],
        "cross_head": cross_head,
        "kv_head": ma["kv_head"],
        "cross_kv_head": cross_kv_head,
        "head_dim": head_dim,
        "cross_head_dim": head_dim,
        "n_layers": ma["n_layers"],
        "vocab_size": ma["vocab_size"],
        "max_seq_len": max_seq_len,
        "norm_eps": ma["norm_eps"],
        "rope_theta": ma["rope_theta"],
        "qk_norm": False if qk_rms_clip else ma.get("qk_norm", False),
        "qk_rms_clip": qk_rms_clip,
        "qk_rms_limit": ma.get("qk_rms_limit", 3.0),
        "attention_bias": ma.get("attention_bias", False),
        "weight_tying": ma.get("weight_tying", False),
        "gated_attention": ma.get("gated_attention", False),
        "diff_attention": ma.get("diff_attention", False),
        "yoco_cross_layers": ma.get("yoco_cross_layers", 0),
        "yoco_window_size": ma.get("yoco_window_size", 512),
        "universal_loop": ma.get("universal_loop", 1),
        "moe": ma.get("moe", False),
        "moe_expert_num": ma.get("moe_expert_num", 0),
        "moe_top_k": ma.get("moe_top_k", 0),
        "moe_ffn_dim": ma.get("moe_ffn_dim", 0),
        "d_shared_expert": ma.get("d_shared_expert", 0),
        "dense_layers": ma.get("dense_layers", 0),
        "swiglu_limit": ma.get("swiglu_limit", 10.0),
        "router_weights_normalized": router_weights_normalized,
        "quant_mode": quant_mode,
        "quant_block_size": quant_block_size,
        "hidden_size": ma["d_model"],
        "intermediate_size": ma["d_ffn"],
        "num_experts": ma.get("moe_expert_num", 0),
        "num_experts_per_tok": ma.get("moe_top_k", 0),
        "moe_intermediate_size": ma.get("moe_ffn_dim", 0),
        "shared_expert_intermediate_size": ma.get("d_shared_expert", 0),
        "num_attention_heads": ma["head"],
        "num_key_value_heads": ma["kv_head"],
        "num_hidden_layers": ma["n_layers"],
        "max_position_embeddings": max_seq_len,
        "rms_norm_eps": ma["norm_eps"],
        "tie_word_embeddings": ma.get("weight_tying", False),
        "bos_token_id": BOS_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "unk_token_id": UNK_TOKEN_ID,
        "transformers_version": "4.36.0",
        "use_cache": True,
    }


def infer_vision_num_layers(state_dict: dict[str, torch.Tensor]) -> int:
    layer_indices = set[int]()
    prefix = "vision_tower.encoder.blocks."
    for key in state_dict:
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        layer_idx = rest.split(".", 1)[0]
        if layer_idx.isdigit():
            layer_indices.add(int(layer_idx))
    if not layer_indices:
        return 10
    return max(layer_indices) + 1


def build_vl_config(
    metadata: dict,
    state_dict: dict[str, torch.Tensor],
    *,
    quant_mode: str | None,
    quant_block_size: int | None,
    router_weights_normalized: bool,
) -> dict:
    ma = metadata["modelargs"]
    merge_kernel_size = ma.get("vision_merge_kernel_size", 2)

    text_config = build_text_config(
        metadata,
        quant_mode=quant_mode,
        quant_block_size=quant_block_size,
        router_weights_normalized=router_weights_normalized,
    )

    return {
        "architectures": ["YOCOVLForConditionalGeneration"],
        "model_type": "yoco_vl",
        "torch_dtype": "bfloat16",
        "text_config": text_config,
        "vision_config": {
            "model_type": "moonvit",
            "patch_size": ma.get("vision_patch_size", 14),
            "init_pos_emb_height": 64,
            "init_pos_emb_width": 64,
            "init_pos_emb_time": 4,
            "pos_emb_type": "divided_fixed",
            "num_attention_heads": 16,
            "num_hidden_layers": infer_vision_num_layers(state_dict),
            "hidden_size": ma.get("vision_encoder_hidden_size", 1152),
            "intermediate_size": 4304,
            "merge_kernel_size": [merge_kernel_size, merge_kernel_size],
            "video_attn_type": "spatial_temporal",
            "merge_type": "sd2_tpool",
        },
        "image_start_token_id": IMAGE_START_TOKEN_ID,
        "image_end_token_id": IMAGE_END_TOKEN_ID,
        "image_placeholder_token_id": IMAGE_PLACEHOLDER_TOKEN_ID,
        "image_placeholder": IMAGE_PLACEHOLDER,
        "video_placeholder": "<video>",
        "vision_max_image_tokens": ma.get("vision_max_image_tokens", 4096),
        "vision_align_mode": ma.get("vision_align_mode", "resize"),
        "vision_patch_limit_on_one_side": 512,
        "vision_image_mean": [0.5, 0.5, 0.5],
        "vision_image_std": [0.5, 0.5, 0.5],
        "bos_token_id": BOS_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "unk_token_id": UNK_TOKEN_ID,
        "transformers_version": "4.36.0",
    }


def write_generation_config(output_dir: Path) -> None:
    generation_config = {
        "bos_token_id": BOS_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_IDS,
        "pad_token_id": PAD_TOKEN_ID,
        "do_sample": True,
        "transformers_version": "4.36.0",
    }
    with (output_dir / "generation_config.json").open("w", encoding="utf-8") as writer:
        json.dump(generation_config, writer, indent=2)


def copy_tokenizer_files(tokenizer_dir: Path, output_dir: Path) -> None:
    tokenizer_json_path = output_dir / "tokenizer.json"
    shutil.copy2(tokenizer_dir / "tokenizer.json", tokenizer_json_path)
    add_bos_post_processor(str(tokenizer_json_path))

    chat_template = None
    chat_template_path = tokenizer_dir / "chat_template.jinja"
    if chat_template_path.is_file():
        chat_template = ensure_chat_template_has_bos(
            chat_template_path.read_text(encoding="utf-8")
        )
        (output_dir / "chat_template.jinja").write_text(
            chat_template, encoding="utf-8"
        )

    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "clean_up_tokenization_spaces": False,
        "do_lower_case": False,
        "remove_space": False,
        "padding_side": "left",
        "model_max_length": 202752,
        "bos_token": "<sop>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|reserved_154856|>",
        "unk_token": "<|reserved_154857|>",
        "additional_special_tokens": [
            "<|begin_of_image|>",
            "<|end_of_image|>",
        ],
    }
    if chat_template is not None:
        tokenizer_config["chat_template"] = chat_template

    with (output_dir / "tokenizer_config.json").open("w", encoding="utf-8") as writer:
        json.dump(tokenizer_config, writer, indent=2, ensure_ascii=False)

    special_tokens_map = {
        "bos_token": "<sop>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|reserved_154856|>",
        "unk_token": "<|reserved_154857|>",
        "additional_special_tokens": [
            "<|begin_of_image|>",
            "<|end_of_image|>",
        ],
    }
    with (output_dir / "special_tokens_map.json").open("w", encoding="utf-8") as writer:
        json.dump(special_tokens_map, writer, indent=2, ensure_ascii=False)


def split_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    language_state = {}
    vl_state = {}
    for key, value in state_dict.items():
        if key.startswith(("mtp_block.", "mtp_norm.", "moe_loss.")):
            continue
        if key.startswith(("vision_tower.", "vision_projector.")):
            vl_state[key] = value
        else:
            language_state[key] = value
    return language_state, vl_state


def convert_vl_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    verbose: bool,
    router_normalization: str,
) -> dict[str, torch.Tensor]:
    language_state, vl_state = split_state_dict(state_dict)
    print(
        "Split state: "
        f"language={len(language_state):,}, vl={len(vl_state):,}",
        flush=True,
    )

    converted_language = convert_state_dict(
        language_state,
        verbose=verbose,
        router_normalization=router_normalization,
    )
    converted = {
        f"language_model.{key}": value
        for key, value in converted_language.items()
    }
    converted.update(vl_state)
    return converted


def convert_checkpoint(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    tokenizer_dir = Path(args.tokenizer_dir)

    if args.router_normalization == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--router-normalization cuda requires CUDA. Use runtime on CPU."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("1. Loading metadata ...", flush=True)
    metadata = load_metadata(input_dir)
    ma = metadata["modelargs"]
    print(
        f"   d_model={ma['d_model']} n_layers={ma['n_layers']} "
        f"vision_hidden={ma.get('vision_encoder_hidden_size')} "
        f"updates={metadata.get('updates')}",
        flush=True,
    )

    print("2. Loading model state ...", flush=True)
    state_dict = load_model_state(input_dir)
    print(f"   Loaded {len(state_dict):,} tensors", flush=True)

    print("3. Writing config files ...", flush=True)
    config = build_vl_config(
        metadata,
        state_dict,
        quant_mode=args.quant_mode,
        quant_block_size=args.quant_block_size,
        router_weights_normalized=args.router_normalization == "cuda",
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as writer:
        json.dump(config, writer, indent=2)
    write_generation_config(output_dir)

    print("4. Copying tokenizer files ...", flush=True)
    copy_tokenizer_files(tokenizer_dir, output_dir)

    print("5. Converting weights ...", flush=True)
    converted = convert_vl_state_dict(
        state_dict,
        verbose=args.verbose,
        router_normalization=args.router_normalization,
    )
    del state_dict
    print(f"   Produced {len(converted):,} tensors", flush=True)

    print("6. Saving sharded safetensors ...", flush=True)
    save_sharded(
        converted,
        str(output_dir),
        max_shard_size=args.max_shard_size_gb * 1024 * 1024 * 1024,
    )

    print(f"Done: {output_dir}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert native YOCO-VL checkpoint to HF/vLLM format"
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quant-mode", choices=("bfloat16", "mxfp8"), default=None)
    parser.add_argument("--quant-block-size", type=int, default=None)
    parser.add_argument(
        "--router-normalization",
        choices=("cuda", "runtime"),
        default="cuda",
    )
    parser.add_argument("--max-shard-size-gb", type=int, default=5)
    args = parser.parse_args()

    try:
        convert_checkpoint(args)
        return 0
    except Exception as exc:
        print(f"Conversion failed: {exc}", flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
