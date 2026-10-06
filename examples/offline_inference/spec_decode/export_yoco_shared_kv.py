# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export a committed SharedKVDraft checkpoint for the V2 DSpark speculator."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def export(checkpoint: Path, target: Path, output: Path, committed_only=False):
    checkpoint, target = checkpoint.resolve(), target.resolve()
    source = checkpoint / "draft"
    config = json.loads((source / "config.json").read_text())
    complete = json.loads((checkpoint / "complete.json").read_text())
    published_path = checkpoint / "publication.json"
    if published_path.exists():
        published = json.loads(published_path.read_text())
    elif committed_only:
        published = complete
    else:
        raise ValueError("Publication receipt missing; committed-only mode is explicit")
    if complete["status"] != "PASS" or published["status"] != "PASS":
        raise ValueError("Checkpoint must be committed and independently published")
    if complete["step"] != published["step"]:
        raise ValueError("Checkpoint publication step mismatch")
    if config["architecture"] != "SharedKVDraft" or "frozen_target" not in config:
        raise ValueError("Expected balanced SharedKVDraft checkpoint")
    run = complete["config"]["run_name"]
    window = config["config"].get("draft_kv_window", 0)
    if run == "sharedkv4-swa8k-dp16-full20k":
        if (
            window != 8192
            or config.get("initialization_schema") != "sharedkv-swa8k-full20k-v1"
        ):
            raise ValueError("SWA8K checkpoint schema/window mismatch")
    elif run != "sharedkv-balanced-dp16-8k10k-full10k" or window != 0:
        raise ValueError("Checkpoint belongs to a different training run")
    # The training target directory may contain symlinked shards rather than
    # itself being a symlink. Compare the actual weight files in that case.
    recorded = Path(config["frozen_target"])
    balanced = Path(
        "/mnt/pvc/shaohanh/exp/agens/30A3B/merged/balanced-b040-c035-d025-hf"
    )
    expected = recorded if recorded.exists() else balanced
    if target != expected.resolve():
        left = json.loads((target / "model.safetensors.index.json").read_text())
        right = json.loads((expected / "model.safetensors.index.json").read_text())
        if left["weight_map"] != right["weight_map"] or any(
            (target / shard).resolve() != (expected / shard).resolve()
            for shard in set(right["weight_map"].values())
        ):
            raise ValueError("Target weights differ from the balanced frozen target")
    for name in ("config.json", "model.safetensors"):
        key = "draft/" + name
        if complete["files"][key] != published["files"][key]:
            raise ValueError("Commit and publication hashes differ")
        with (source / name).open("rb") as f:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: f.read(8 << 20), b""):
                hasher.update(chunk)
            digest = hasher.hexdigest()
        if digest != complete["files"][key]["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
    c = config["config"]
    target_config = json.loads((target / "config.json").read_text())
    if target_config["model_type"] != "yoco":
        raise ValueError("Expected YOCO target")
    layers = target_config.get("num_hidden_layers", target_config.get("n_layers"))
    if layers != 20:
        raise ValueError("This checkpoint was trained using raw h20")
    hf = {
        "model_type": "qwen3",
        "architectures": ["Qwen3DSparkModel"],
        "hidden_size": c["hidden"],
        "intermediate_size": c["ffn"],
        "num_hidden_layers": c["layers"],
        "num_attention_heads": c["heads"],
        "num_key_value_heads": c["kv_heads"],
        "head_dim": c["head_dim"],
        "vocab_size": c["vocab"],
        "draft_vocab_size": c["vocab"],
        "max_position_embeddings": target_config["max_position_embeddings"],
        "rms_norm_eps": c["eps"],
        "rope_theta": c["theta"],
        "dtype": "bfloat16",
        "sample_from_anchor": True,
        "mask_token_id": c["mask_id"],
        "markov_rank": c["rank"],
        "n_predict": c["block"],
        "use_aux_hidden_state": True,
        "eagle_aux_hidden_state_layer_ids": [layers],
        "shared_kv_config": c,
        "shared_kv_target": str(target),
        "shared_kv_weights_sha256": complete["files"]["draft/model.safetensors"][
            "sha256"
        ],
        "shared_kv_source": str(checkpoint),
        "shared_kv_step": complete["step"],
        "shared_kv_receipt_scope": "committed-only"
        if not published_path.exists()
        else "published",
    }
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source / "model.safetensors", output / "model.safetensors")
    (output / "config.json").write_text(json.dumps(hf, indent=2) + "\n")
    (output / "training-config.json").write_text(json.dumps(config, indent=2) + "\n")
    return hf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--committed-only", action="store_true")
    args = parser.parse_args()
    result = export(args.checkpoint, args.target, args.output, args.committed_only)
    print(json.dumps({"output": str(args.output), "step": result["shared_kv_step"]}))
