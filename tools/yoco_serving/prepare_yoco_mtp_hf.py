# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export a standalone MTP drafter from preserved YOCO training auxiliaries."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm.transformers_utils.configs.yoco import YOCOMTPConfig
from vllm.transformers_utils.configs.yoco_mtp import expected_mtp_shapes

PREFIX = "pipeline_model.backbone.model."


def convert_mtp_weights(tensors: dict, config: dict) -> dict[str, torch.Tensor]:
    weights = {}
    for name, tensor in tensors.items():
        if name.startswith(PREFIX + "mtp_block."):
            name = "mtp." + name.removeprefix(PREFIX + "mtp_block.")
        elif name == PREFIX + "mtp_norm.weight":
            name = "norm.weight"
        else:
            raise ValueError(f"Unexpected MTP source tensor: {name}")
        if name == "mtp.self_attn.q_proj.weight":
            tensor = tensor.flatten(0, 1)
        name = name.replace("self_attn.gate_proj.", "self_attn.lambda_proj.")
        weights[name] = tensor.contiguous()
    gate = weights.pop("mtp.ffn.gate_proj.weight")
    up = weights.pop("mtp.ffn.up_proj.weight")
    weights["mtp.ffn.gate_up_proj.weight"] = torch.cat((gate, up))
    expected = expected_mtp_shapes(config)
    if set(weights) != set(expected):
        raise ValueError(f"MTP inventory mismatch: {set(weights) ^ set(expected)}")
    for name, shape in expected.items():
        tensor = weights[name]
        if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
            raise ValueError(
                f"Invalid MTP tensor {name}: {tensor.shape}, {tensor.dtype}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite MTP tensor: {name}")
    return weights


def prepare(source: Path, destination: Path) -> dict:
    staging = destination.with_name(destination.name + ".incomplete")
    if destination.exists() or staging.exists():
        raise FileExistsError("MTP destination or staging already exists")
    config = json.loads((source / "config.json").read_text())
    args = json.loads((source / "training_model_args.json").read_text())
    if config["model_type"] != "yoco" or args.get("mtp_n") != 1:
        raise ValueError("Expected a YOCO text export with trained mtp_n=1")
    auxiliary = source / "auxiliary/vision_and_mtp.safetensors"
    with safe_open(auxiliary, framework="pt", device="cpu") as handle:
        tensors = {
            name: handle.get_tensor(name)
            for name in handle.keys()  # noqa: SIM118
            if name.startswith((PREFIX + "mtp_block.", PREFIX + "mtp_norm."))
        }
    if len(tensors) != 13:
        raise ValueError(f"Expected all 13 MTP training tensors, got {len(tensors)}")
    weights = convert_mtp_weights(tensors, config)
    config.update(
        model_type="yoco_mtp",
        architectures=["YOCOMTPModel"],
        n_predict=1,
        num_nextn_predict_layers=1,
        mtp_n=1,
        num_hidden_layers=1,
        n_layers=1,
        yoco_cross_layers=1,
        universal_loop=1,
        num_key_value_heads=config["cross_kv_head"],
        moe=False,
        num_experts=0,
        moe_expert_num=0,
    )
    staging.mkdir(parents=True)
    save_file(weights, staging / "model.safetensors")
    with safe_open(staging / "model.safetensors", framework="pt") as handle:
        for name, tensor in weights.items():
            if not torch.equal(handle.get_tensor(name), tensor):
                raise ValueError(f"MTP tensor write mismatch: {name}")
    YOCOMTPConfig(**config).save_pretrained(staging)
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        if (source / filename).exists():
            shutil.copy2(source / filename, staging / filename)
    manifest = {
        "source_auxiliary": str(auxiliary),
        "source_tensors": len(tensors),
        "export_tensors": len(weights),
        "all_values_verified": True,
        "tensor_bytes": sum(t.numel() * t.element_size() for t in weights.values()),
        "embedding": "Shared target output/lm_head weight; NOT target input embedding",
        "attention": "Reads target cross KV at the unshifted hidden-state position",
        "source_model_args": args,
    }
    (staging / "mtp_export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    staging.rename(destination)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    print(json.dumps(prepare(cli.source, cli.output), indent=2))
