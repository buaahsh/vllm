# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attach preserved Kimi-K3 vision weights to a YOCO language HF export."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm.transformers_utils.configs.yoco_vl import YOCOVLConfig


def expected_vision_shapes(text_width: int) -> dict[str, tuple[int, ...]]:
    shapes = {
        "vision_tower.patch_embed.proj.weight": (1024, 3, 14, 14),
        "vision_tower.patch_embed.pos_emb.weight": (64, 64, 1024),
        "vision_tower.encoder.final_layernorm.weight": (1024,),
        "vision_projector.proj.0.weight": (4096, 4096),
        "vision_projector.proj.2.weight": (text_width, 4096),
        "vision_projector.post_norm.weight": (text_width,),
    }
    for layer in range(27):
        for name, shape in {
            "norm0.weight": (1024,),
            "norm1.weight": (1024,),
            "wqkv.weight": (4608, 1024),
            "wo.weight": (1024, 1536),
            "mlp.fc0.weight": (4096, 1024),
            "mlp.fc1.weight": (1024, 4096),
        }.items():
            shapes[f"vision_tower.encoder.blocks.{layer}.{name}"] = shape
    return shapes


def prepare(source: Path, destination: Path) -> dict:
    source = source.resolve()
    destination = destination.absolute()
    staging = destination.with_name(destination.name + ".incomplete")
    if destination.exists() or staging.exists():
        raise FileExistsError("Destination or staging directory already exists")
    text_config = json.loads((source / "config.json").read_text())
    if text_config.get("model_type") != "yoco":
        raise ValueError("Source must be a YOCO language HF export")
    text_config["sliding_window_size"] = 512
    auxiliary = source / "auxiliary" / "vision_and_mtp.safetensors"
    with safe_open(auxiliary, framework="pt", device="cpu") as handle:
        vision = {
            name.removeprefix("vision_backbone."): handle.get_tensor(name)
            for name in handle.keys()  # noqa: SIM118 -- safe_open is not iterable
            if name.startswith("vision_backbone.")
        }
    expected = expected_vision_shapes(text_config["hidden_size"])
    if set(vision) != set(expected):
        raise ValueError(
            f"Kimi-K3 tensor inventory mismatch: {set(vision) ^ set(expected)}"
        )
    for name, tensor in vision.items():
        if tuple(tensor.shape) != expected[name] or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid vision tensor: {name}, {tensor.shape}")
    config = YOCOVLConfig(text_config=text_config)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    staging.mkdir(parents=True)
    for shard in set(index["weight_map"].values()):
        if Path(shard).name != shard:
            raise ValueError(f"Expected a shard filename, got {shard}")
        os.link(source / shard, staging / shard)
    for entry in source.iterdir():
        if entry.is_file() and entry.suffix not in (".safetensors", ".json"):
            if entry.name not in ("README.md", "chat_template.jinja"):
                shutil.copy2(entry, staging / entry.name)
        elif entry.is_file() and entry.name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
            "added_tokens.json",
            "training_model_args.json",
        ):
            shutil.copy2(entry, staging / entry.name)
    save_file(vision, staging / "vision.safetensors")
    with safe_open(staging / "vision.safetensors", framework="pt") as handle:
        for name, tensor in vision.items():
            if not torch.equal(handle.get_tensor(name), tensor):
                raise ValueError(f"Vision tensor write verification failed: {name}")
    index["weight_map"].update({name: "vision.safetensors" for name in vision})
    vision_bytes = sum(t.numel() * t.element_size() for t in vision.values())
    index["metadata"]["total_size"] += vision_bytes
    (staging / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )
    config.save_pretrained(staging)
    template_path = (
        Path(__file__).resolve().parents[2]
        / "vllm/transformers_utils/chat_templates/template_yoco_vl.jinja"
    )
    template = template_path.read_text()
    (staging / "chat_template.jinja").write_text(template)
    tokenizer_config_path = staging / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text())
    tokenizer_config["chat_template"] = template
    tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2) + "\n")
    manifest = {
        "source_text_export": str(source),
        "source_auxiliary": str(auxiliary),
        "architecture": "YOCOVLForConditionalGeneration",
        "vision_type": "kimi_k3",
        "vision_tensor_count": len(vision),
        "vision_bytes": vision_bytes,
        "vision_values_verified_exactly": True,
        "language_shards": "hardlinks to validated source export; values unchanged",
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "mtp": "preserved in source auxiliary file, not used for inference",
    }
    (staging / "vl_export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (staging / "README.md").write_text(
        "# YOCO-VL with Kimi-K3 vision\n\n"
        "Requires the msrallm/vllm vllm-yoco-version-0.29-vl branch.\n"
        "Language safetensors retain their original YOCO names; the model loader\n"
        "maps them to language_model. Vision weights use vision_tower and\n"
        "vision_projector prefixes. See vl_export_manifest.json for provenance.\n"
    )
    staging.rename(destination)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
