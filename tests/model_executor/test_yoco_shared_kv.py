# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-KV cache isolation, rejection boundaries and local causality."""

import pytest
import torch
from torch.nn import functional as F

from vllm.model_executor.models.yoco_shared_kv import (
    SharedKVConfig,
    SharedKVDraft,
    paged_prefix_attention,
)
from vllm.v1.worker.gpu.spec_decode.dspark.shared_kv import select_shared_kv_context


def test_shared_kv_rejection_selects_previous_hidden_not_bonus_or_suffix():
    positions = torch.tensor([100, 101, 102, 103, 9, 10, 11])
    hidden = torch.arange(7 * 4).reshape(7, 4)
    anchors, current = select_shared_kv_context(
        positions, torch.tensor([0, 4, 7]), torch.tensor([2, 0]), hidden
    )
    assert anchors.tolist() == [102, 12]
    torch.testing.assert_close(current, hidden[[1, 6]])


def test_shared_kv_local_attention_cannot_see_future_slots():
    torch.manual_seed(42)
    c = SharedKVConfig(
        hidden=16,
        ffn=24,
        heads=4,
        kv_heads=2,
        head_dim=4,
        layers=2,
        vocab=32,
        block=8,
        mask_id=31,
        rank=8,
    )
    model = SharedKVDraft(c).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.2, 0.2)
    ids = torch.randint(0, c.vocab, (2, 8))
    current = torch.randn(2, c.hidden)
    positions = torch.tensor([37, 81])[:, None] + torch.arange(8)
    keys, values = torch.randn(2, 2, 9, 4), torch.randn(2, 2, 9, 4)

    def attend(q):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), keys, values, enable_gqa=True
        ).transpose(1, 2)

    with torch.no_grad():
        expected = model(ids, current, positions, attend)
        changed = ids.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % c.vocab
        actual = model(changed, current, positions, attend)
        short = model(ids[:, :4], current, positions[:, :4], attend)
    torch.testing.assert_close(actual[:, :4], expected[:, :4], rtol=0, atol=0)
    torch.testing.assert_close(short, expected[:, :4], rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA paged attention")
@pytest.mark.parametrize("physical_layout", ["HND", "NHD"])
def test_shared_kv_paged_attention_matches_strict_prefix_and_never_writes(
    physical_layout,
):
    torch.manual_seed(42)
    device, dtype = "cuda", torch.bfloat16
    batch, block, heads, kvheads, dim, page = 3, 8, 4, 2, 64, 256
    q = torch.randn(batch, block, heads, dim, device=device, dtype=dtype)
    cache = torch.randn(10, kvheads, page, 2 * dim, device=device, dtype=dtype)
    if physical_layout == "NHD":
        cache = cache.transpose(1, 2).contiguous().transpose(1, 2)
    table = torch.tensor([[4, 2], [7, 1], [8, 3]], device=device, dtype=torch.int32)
    lengths = torch.tensor([257, 311, 9], device=device, dtype=torch.int32)
    before = cache.clone()
    expected = []
    for i, length in enumerate(lengths.tolist()):
        history = (
            cache[table[i].long()].permute(1, 0, 2, 3).reshape(kvheads, -1, 2 * dim)
        )
        key, value = history[:, :length].split(dim, -1)
        expected.append(
            F.scaled_dot_product_attention(
                q[i].transpose(0, 1).float(),
                key.float(),
                value.float(),
                enable_gqa=True,
            ).transpose(0, 1)
        )
    actual = paged_prefix_attention(q, cache, table, lengths, 311)
    torch.testing.assert_close(
        actual.float(), torch.stack(expected), atol=0.006, rtol=0.02
    )
    torch.testing.assert_close(cache, before, rtol=0, atol=0)
    # Poison every rejected/future position; none is part of the strict prefix.
    for i, length in enumerate(lengths.tolist()):
        for position in range(length, 2 * page):
            cache[table[i, position // page], :, position % page] = 100
    after = paged_prefix_attention(q, cache, table, lengths, 311)
    torch.testing.assert_close(after, actual, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA proposal buffers")
def test_shared_kv_propose_handles_bonus_column_and_chunked_prefill():
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.spec_decode.dspark.shared_kv import SharedKVSpeculator

    device = torch.device("cuda")
    c = SharedKVConfig(hidden=8, vocab=32, mask_id=31, block=4)
    observed: dict[str, torch.Tensor] = {}

    class Recorder:
        def __init__(self):
            self.c = c

        def __call__(self, ids, current, positions, attend):
            observed.update(
                ids=ids.clone(), current=current.clone(), positions=positions.clone()
            )
            return current[:, None].expand(-1, 4, -1)

    model = Recorder()
    spec = SharedKVSpeculator.__new__(SharedKVSpeculator)
    spec.model, spec.device, spec.dtype = model, device, torch.bfloat16
    spec.num_speculative_steps = 4
    spec._cache_owner = SimpleNamespace(kv_cache=torch.empty(0, device=device))
    spec._target_gid = 0
    spec.block_tables = SimpleNamespace(
        input_block_tables=[torch.zeros(2, 1, device=device)]
    )
    spec.input_buffers = SimpleNamespace(
        input_ids=torch.empty(8, device=device, dtype=torch.long)
    )
    for name in ("sample_indices", "sample_pos", "sample_idx_mapping"):
        setattr(spec, name, torch.empty(8, device=device, dtype=torch.long))
    spec.temperature = torch.zeros(3, device=device)
    spec.seeds = torch.zeros(3, device=device, dtype=torch.long)
    spec.draft_tokens = torch.zeros(2, 4, device=device, dtype=torch.long)
    spec._sample_sequential = lambda n, hidden: None
    batch = SimpleNamespace(
        num_reqs=2,
        positions=torch.tensor([100, 101, 102, 103, 8, 9], device=device),
        query_start_loc=torch.tensor([0, 4, 6], device=device),
        idx_mapping=torch.tensor([2, 0], device=device),
        seq_lens_cpu_upper_bound=torch.tensor([104, 10]),
    )
    hidden = torch.arange(48, device=device).reshape(6, 8).to(torch.bfloat16)
    spec.propose(
        batch,
        {},
        {},
        hidden,
        [hidden],
        torch.tensor([2, 0], device=device),
        torch.tensor([2, 0], device=device),
        torch.tensor([[11], [12], [13]], device=device),
        torch.tensor([[21, 22, 23]], device=device),
        spec.temperature,
        spec.seeds,
    )
    assert observed["ids"].tolist() == [[13, 31, 31, 31], [21, 31, 31, 31]]
    assert observed["positions"].tolist() == [[102, 103, 104, 105], [10, 11, 12, 13]]
    torch.testing.assert_close(observed["current"], hidden[[1, 5]])
    assert spec.sample_pos.tolist() == [103, 104, 105, 106, 11, 12, 13, 14]
    assert spec.sample_idx_mapping.tolist() == [2, 2, 2, 2, 0, 0, 0, 0]


def test_shared_kv_export_pins_checkpoint_and_rejects_changed_weights(tmp_path):
    import hashlib
    import json
    import runpy
    from dataclasses import asdict
    from pathlib import Path

    exporter = (
        Path(__file__).parents[2]
        / "examples/offline_inference/spec_decode/export_yoco_shared_kv.py"
    )
    export = runpy.run_path(str(exporter))["export"]
    target, checkpoint = tmp_path / "target", tmp_path / "step-003750"
    target.mkdir()
    (target / "config.json").write_text(
        json.dumps(
            {
                "model_type": "yoco",
                "num_hidden_layers": 20,
                "max_position_embeddings": 262144,
            }
        )
    )
    source = checkpoint / "draft"
    source.mkdir(parents=True)
    (source / "config.json").write_text(
        json.dumps(
            {
                "architecture": "SharedKVDraft",
                "config": asdict(SharedKVConfig()),
                "frozen_target": str(target),
            }
        )
    )
    (source / "model.safetensors").write_bytes(b"hash-verification-fixture")
    receipt = {
        "status": "PASS",
        "step": 3750,
        "config": {"run_name": "sharedkv-balanced-dp16-8k10k-full10k"},
        "files": {
            "draft/" + p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in source.iterdir()
        },
    }
    for name in ("complete.json", "publication.json"):
        (checkpoint / name).write_text(json.dumps(receipt))
    output = tmp_path / "export"
    config = export(checkpoint, target, output)
    assert config["eagle_aux_hidden_state_layer_ids"] == [20]
    assert config["shared_kv_step"] == 3750
    assert config["shared_kv_target"] == str(target.resolve())
    assert (output / "model.safetensors").read_bytes() == b"hash-verification-fixture"
    (source / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checkpoint hash mismatch"):
        export(checkpoint, target, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()
