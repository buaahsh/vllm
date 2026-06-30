#!/usr/bin/env python3
"""Convert YOCO checkpoint from training format to HuggingFace format.

Designed for the *new* nnscaler training code in ``llm-train`` that uses the
GLM-5.1 (``agens_tokenizer``) tokenizer and the updated YOCO architecture
(diff-attention, all-layer MoE with shared expert, etc.).

The script consumes:
- ``model_state_rank_0.pth`` + ``metadata.json`` from ``--input_dir``
- ``agens_tokenizer/`` (tokenizer.json, tokenizer_config.json, chat_template.jinja)
  and ``config.json`` / ``generation_config.json`` from this directory

It produces a sharded safetensors checkpoint compatible with the vLLM model
implementation in ``yoco_vllm.py``.

Usage::

    python convert_to_hf.py \
        --input_dir /mnt/pvc/shaohanh/exp/agens/30A3B-72M/0000-3125-merged \
        --output_dir /path/to/hf_output
"""

import argparse
import json
import os
import shutil
from typing import Dict

import torch
from safetensors.torch import save_file


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_DIR_NAME = "agens_tokenizer_0622"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_metadata(checkpoint_dir: str) -> dict:
    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found at {metadata_path}")
    with open(metadata_path, "r") as f:
        return json.load(f)


def load_model_state(checkpoint_dir: str) -> dict:
    model_path = os.path.join(checkpoint_dir, "model_state_rank_0.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model_state_rank_0.pth not found at {model_path}")
    print(f"Loading checkpoint from {model_path} ...")
    checkpoint = torch.load(model_path, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


# ---------------------------------------------------------------------------
# State dict conversion (training -> HF/vLLM)
# ---------------------------------------------------------------------------
def _flatten_proj(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten 3D per-head projection (num_heads, head_dim, in) to 2D
    (num_heads * head_dim, in) so it matches HF/vLLM linear layouts."""
    if tensor.dim() == 3:
        return tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2])
    return tensor


def convert_state_dict(state_dict: Dict[str, torch.Tensor], verbose: bool = False) -> Dict[str, torch.Tensor]:
    """Rename training keys to HF keys and reshape per-head projections."""
    new_state: Dict[str, torch.Tensor] = {}

    def add(new_key: str, tensor: torch.Tensor, orig: str):
        if new_key in new_state:
            raise RuntimeError(f"Duplicate target key {new_key} (from {orig})")
        new_state[new_key] = tensor
        if verbose:
            print(f"  {orig:65s} -> {new_key}  {tuple(tensor.shape)}")

    # ----- top-level (non-layer) parameters
    top_level_map = {
        "tok_embeddings.weight": "model.embed_tokens.weight",
        "output.weight": "lm_head.weight",
        "norm.weight": "model.norm.weight",
        "yoco_norm.weight": "model.yoco_norm.weight",
        "k_proj.weight": "model.k_proj.weight",
        "v_proj.weight": "model.v_proj.weight",
    }

    for key, value in state_dict.items():
        if key in top_level_map:
            add(top_level_map[key], _flatten_proj(value), key)
            continue

        if not key.startswith("layers."):
            print(f"  [WARN] unknown top-level key skipped: {key}  {tuple(value.shape)}")
            continue

        # layers.{idx}.<rest>
        _, layer_idx, rest = key.split(".", 2)
        prefix = f"model.layers.{layer_idx}"

        # --- self-attention block ---
        if rest.startswith("self_attn."):
            sub = rest[len("self_attn."):]
            # Per-head 3D projections that need flattening
            if sub in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
                add(f"{prefix}.self_attn.{sub}", _flatten_proj(value), key)
            elif sub in ("o_proj.weight", "lambda_proj.weight",
                         "q_norm.weight", "k_norm.weight"):
                add(f"{prefix}.self_attn.{sub}", value, key)
            else:
                # gate_proj.weight (when gated_attention=True), etc.
                add(f"{prefix}.self_attn.{sub}", _flatten_proj(value), key)
            continue

        # --- MLP / MoE block ---
        if rest.startswith("mlp."):
            sub = rest[len("mlp."):]

            # MoE routing gate (small Linear).
            # Training's GateLinear applies row-wise L2 normalization to the
            # weight before the matmul (see llm-train/llm/arch/linear.py:
            # norm_linear -> weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-6)).
            # vLLM's gate is a plain ReplicatedLinear with no such normalization,
            # so bake the per-expert-row normalization into the exported weight
            # to preserve router-logit parity. Done in fp32 (the gate runs in
            # fp32 during training: self.gate(x.float())).
            if sub == "gate.weight":
                gate_w = value.float()
                gate_w = gate_w / gate_w.norm(dim=1, keepdim=True).clamp_min(1e-6)
                add(f"{prefix}.mlp.gate.weight", gate_w, key)
                continue

            # MoE experts (3D tensors flattened on dim 0 for vLLM FusedMoE)
            if sub == "w13":
                tensor = value
                if tensor.dim() == 3:
                    tensor = tensor.reshape(-1, tensor.shape[-1])
                add(f"{prefix}.mlp.experts.w13_weight", tensor, key)
                continue
            if sub == "w2":
                tensor = value
                if tensor.dim() == 3:
                    tensor = tensor.reshape(-1, tensor.shape[-1])
                add(f"{prefix}.mlp.experts.w2_weight", tensor, key)
                continue

            # Shared expert: training stores separate gate/up; HF/vLLM uses fused
            # ``shared_experts.gate_up_proj`` (MergedColumnParallelLinear).
            if sub == "shared.gate_proj.weight":
                up_key = key.replace(".shared.gate_proj.", ".shared.up_proj.")
                if up_key not in state_dict:
                    raise RuntimeError(f"Missing paired up_proj for {key}")
                gate = value
                up = state_dict[up_key]
                fused = torch.cat([gate, up], dim=0)
                add(f"{prefix}.mlp.shared_experts.gate_up_proj.weight", fused, key)
                continue
            if sub == "shared.up_proj.weight":
                continue  # handled when we see gate_proj
            if sub == "shared.down_proj.weight":
                add(f"{prefix}.mlp.shared_experts.down_proj.weight", value, key)
                continue

            # Shared-expert gating scalar
            if sub == "shared_gate.weight":
                add(f"{prefix}.mlp.shared_gate.weight", value, key)
                continue

            # Dense MLP fallback (no MoE)
            if sub in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
                add(f"{prefix}.mlp.{sub}", value, key)
                continue

            print(f"  [WARN] unknown mlp sub-key skipped: {key}  {tuple(value.shape)}")
            continue

        # --- layer norms ---
        if rest in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            add(f"{prefix}.{rest}", value, key)
            continue

        print(f"  [WARN] unhandled layer key skipped: {key}  {tuple(value.shape)}")

    return new_state


# ---------------------------------------------------------------------------
# Config / tokenizer files
# ---------------------------------------------------------------------------
# Token IDs for the GLM-5.1 ``agens_tokenizer`` (must match llm-train/llm/data/tokenizer.py)
BOS_TOKEN_ID = 154824  # <sop>
EOS_TOKEN_ID = 154820  # <|endoftext|>
PAD_TOKEN_ID = 154856  # <|reserved_154856|>
UNK_TOKEN_ID = 154857  # <|reserved_154857|>


def add_bos_post_processor(tokenizer_json_path: str) -> None:
    """Make HF fast-tokenizer add <sop> when add_special_tokens=True."""
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)

    bos_template = {
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "<sop>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
        ],
        "pair": [
            {"SpecialToken": {"id": "<sop>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"Sequence": {"id": "B", "type_id": 0}},
        ],
        "special_tokens": {
            "<sop>": {
                "id": "<sop>",
                "ids": [BOS_TOKEN_ID],
                "tokens": ["<sop>"],
            }
        },
    }

    post_processor = tokenizer_json.get("post_processor")
    if post_processor and post_processor.get("type") == "Sequence":
        processors = post_processor.get("processors", [])
    else:
        processors = [post_processor] if post_processor else []

    for processor in processors:
        if processor and processor.get("type") == "TemplateProcessing":
            special_tokens = processor.get("special_tokens", {})
            if "<sop>" in special_tokens:
                print("   tokenizer.json already adds <sop>")
                return

    tokenizer_json["post_processor"] = {
        "type": "Sequence",
        "processors": [*processors, bos_template],
    }
    with open(tokenizer_json_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)
    print("   Patched tokenizer.json to add <sop> when add_special_tokens=True")


def ensure_chat_template_has_bos(chat_template: str) -> str:
    """Make chat-template rendering start with the GLM BOS token."""
    stripped = chat_template.lstrip()
    if stripped.startswith("<sop>") or stripped.startswith("{{ bos_token") or stripped.startswith("{{- bos_token"):
        return chat_template
    return "<sop>\n" + chat_template


def create_hf_config(metadata: dict, output_dir: str) -> dict:
    ma = metadata.get("modelargs", metadata)

    head_dim = ma.get("head_dim") or ma["d_model"] // ma["head"]
    cross_kv_head = ma.get("cross_kv_head", ma["kv_head"])
    cross_head = ma.get("cross_head", ma["head"])

    config = {
        "architectures": ["YOCOForCausalLM"],
        "model_type": "yoco",
        "torch_dtype": "bfloat16",

        # Core dims
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
        "max_seq_len": ma["max_seq_len"],

        # Norm / attention flags
        "norm_eps": ma["norm_eps"],
        "rope_theta": ma["rope_theta"],
        "qk_norm": ma.get("qk_norm", False),
        "qk_rms_clip": ma.get("qk_rms_clip", False),
        "qk_rms_limit": ma.get("qk_rms_limit", 3.0),
        "attention_bias": ma.get("attention_bias", False),
        "weight_tying": ma.get("weight_tying", False),
        "gated_attention": ma.get("gated_attention", False),
        "diff_attention": ma.get("diff_attention", False),

        # YOCO
        "yoco_cross_layers": ma.get("yoco_cross_layers", 0),
        "yoco_window_size": ma.get("yoco_window_size", 512),
        "universal_loop": ma.get("universal_loop", 1),

        # MoE
        "moe": ma.get("moe", False),
        "moe_expert_num": ma.get("moe_expert_num", 0),
        "moe_top_k": ma.get("moe_top_k", 0),
        "moe_ffn_dim": ma.get("moe_ffn_dim", 0),
        "d_shared_expert": ma.get("d_shared_expert", 0),
        "dense_layers": ma.get("dense_layers", 0),
        "swiglu_limit": ma.get("swiglu_limit", 10.0),

        # HF aliases
        "hidden_size": ma["d_model"],
        "intermediate_size": ma["d_ffn"],
        "num_attention_heads": ma["head"],
        "num_key_value_heads": ma["kv_head"],
        "num_hidden_layers": ma["n_layers"],
        "max_position_embeddings": ma["max_seq_len"],
        "rms_norm_eps": ma["norm_eps"],
        "tie_word_embeddings": ma.get("weight_tying", False),

        # Token IDs
        "bos_token_id": BOS_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "unk_token_id": UNK_TOKEN_ID,

        "transformers_version": "4.36.0",
        "use_cache": True,
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"   Wrote config.json -> {config_path}")
    return config


def create_generation_config(output_dir: str) -> None:
    gen_cfg = {
        "bos_token_id": BOS_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "do_sample": True,
        "transformers_version": "4.36.0",
    }
    path = os.path.join(output_dir, "generation_config.json")
    with open(path, "w") as f:
        json.dump(gen_cfg, f, indent=2)
    print(f"   Wrote generation_config.json -> {path}")


def copy_tokenizer_files(output_dir: str) -> None:
    """Copy GLM-5.1 (agens_tokenizer) files into the HF model directory.

    The source is ``yoco_vllm/agens_tokenizer/`` (downloaded from
    https://msranlp.blob.core.windows.net/unilm/yutao/hf_cache/agens_tokenizer/).
    """
    src = os.path.join(SCRIPT_DIR, TOKENIZER_DIR_NAME)
    if not os.path.isdir(src):
        raise FileNotFoundError(
            f"Tokenizer dir not found: {src}\n"
            f"Download with:\n"
            f"  python {SCRIPT_DIR}/blob_skill/blob_skill.py download \\\n"
            f"      --url https://msranlp.blob.core.windows.net/unilm/yutao/hf_cache/agens_tokenizer/ "
            f"-o {src}/"
        )

    # tokenizer.json
    tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
    shutil.copy2(os.path.join(src, "tokenizer.json"), tokenizer_json_path)
    add_bos_post_processor(tokenizer_json_path)
    print("   Copied tokenizer.json")

    # chat_template.jinja (kept as a separate file too for convenience)
    chat_tpl_src = os.path.join(src, "chat_template.jinja")
    chat_template = None
    if os.path.exists(chat_tpl_src):
        with open(chat_tpl_src, "r", encoding="utf-8") as f:
            chat_template = ensure_chat_template_has_bos(f.read())
        with open(os.path.join(output_dir, "chat_template.jinja"), "w", encoding="utf-8") as f:
            f.write(chat_template)
        print("   Wrote chat_template.jinja with <sop> prefix")

    # Build a clean tokenizer_config.json:
    #   - tokenizer_class = PreTrainedTokenizerFast so HF AutoTokenizer loads it.
    #   - Embed the chat template so vLLM picks it up via tokenizer.chat_template.
    #   - Drop the upstream ``extra_special_tokens`` list (incompatible with
    #     modern transformers, which expects a dict).
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
    }
    if chat_template is not None:
        tokenizer_config["chat_template"] = chat_template
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    print("   Wrote tokenizer_config.json")

    # special_tokens_map.json (mirror the tokens above so transformers is happy)
    special_tokens_map = {
        "bos_token": "<sop>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|reserved_154856|>",
        "unk_token": "<|reserved_154857|>",
    }
    with open(os.path.join(output_dir, "special_tokens_map.json"), "w") as f:
        json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)
    print("   Wrote special_tokens_map.json")


# ---------------------------------------------------------------------------
# Sharded safetensors save
# ---------------------------------------------------------------------------
def save_sharded(state_dict: Dict[str, torch.Tensor], output_dir: str,
                 max_shard_size: int = 5 * 1024 * 1024 * 1024) -> None:
    sorted_keys = sorted(state_dict.keys())

    # First pass: decide shard assignments
    shards = []  # list of dict
    current = {}
    current_size = 0
    for key in sorted_keys:
        tensor = state_dict[key]
        if tensor.dtype == torch.float32:
            tensor = tensor.to(torch.bfloat16)
        size = tensor.numel() * tensor.element_size()
        if current and current_size + size > max_shard_size:
            shards.append(current)
            current = {}
            current_size = 0
        current[key] = tensor
        current_size += size
    if current:
        shards.append(current)

    num_shards = len(shards)
    print(f"   Splitting into {num_shards} shard(s)")

    weight_map = {}
    total_size = 0
    for i, shard in enumerate(shards):
        name = f"model-{i+1:05d}-of-{num_shards:05d}.safetensors"
        path = os.path.join(output_dir, name)
        print(f"   Saving shard {i+1}/{num_shards}: {name} "
              f"({sum(t.numel() * t.element_size() for t in shard.values()) / 1024**3:.2f} GB)")
        save_file(shard, path, metadata={"format": "pt"})
        for k, t in shard.items():
            weight_map[k] = name
            total_size += t.numel() * t.element_size()

    index = {
        "metadata": {"total_size": int(total_size)},
        "weight_map": weight_map,
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print("   Wrote model.safetensors.index.json")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def convert_checkpoint(input_dir: str, output_dir: str, verbose: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print("\n1. Loading metadata ...")
    metadata = load_metadata(input_dir)
    ma = metadata["modelargs"]
    print(f"   d_model={ma['d_model']}  n_layers={ma['n_layers']}  "
          f"head={ma['head']}  kv_head={ma['kv_head']}  "
          f"moe={ma.get('moe')}  diff_attn={ma.get('diff_attention')}  "
          f"yoco_cross_layers={ma.get('yoco_cross_layers')}  "
          f"universal_loop={ma.get('universal_loop')}  updates={metadata.get('updates')}")

    print("\n2. Writing config.json ...")
    create_hf_config(metadata, output_dir)

    print("\n3. Writing generation_config.json ...")
    create_generation_config(output_dir)

    print("\n4. Copying tokenizer files ...")
    copy_tokenizer_files(output_dir)

    print("\n5. Loading model weights ...")
    state_dict = load_model_state(input_dir)
    print(f"   Loaded {len(state_dict)} parameters")

    print("\n6. Converting parameter names / shapes ...")
    new_state = convert_state_dict(state_dict, verbose=verbose)
    print(f"   Produced {len(new_state)} parameters")

    print("\n7. Saving sharded safetensors ...")
    save_sharded(new_state, output_dir)

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"\nOutput: {output_dir}")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")
    print("\nServe with vLLM:")
    print(f"  python {SCRIPT_DIR}/serve_yoco.py --model {output_dir} --trust-remote-code ...")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert new-format YOCO checkpoint to HuggingFace/vLLM format")
    parser.add_argument("--input_dir", required=True,
                        help="Dir containing model_state_rank_0.pth + metadata.json")
    parser.add_argument("--output_dir", required=True,
                        help="HF output directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every renamed key")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: input dir not found: {args.input_dir}")
        return 1
    for required in ("model_state_rank_0.pth", "metadata.json"):
        if not os.path.exists(os.path.join(args.input_dir, required)):
            print(f"Error: {required} not found in {args.input_dir}")
            return 1

    try:
        convert_checkpoint(args.input_dir, args.output_dir, verbose=args.verbose)
        return 0
    except Exception as e:
        print(f"\nConversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
