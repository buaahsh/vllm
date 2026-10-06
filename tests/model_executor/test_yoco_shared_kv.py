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

    from vllm.config.compilation import CUDAGraphMode
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
        input_ids=torch.empty(8, device=device, dtype=torch.long),
        positions=torch.empty(8, device=device, dtype=torch.long),
    )
    for name in ("sample_indices", "sample_pos", "sample_idx_mapping"):
        setattr(spec, name, torch.empty(8, device=device, dtype=torch.long))
    spec.temperature = torch.zeros(3, device=device)
    spec.seeds = torch.zeros(3, device=device, dtype=torch.long)
    spec.draft_tokens = torch.zeros(2, 4, device=device, dtype=torch.long)
    spec._sample_sequential = lambda n, hidden: None
    spec._current_hidden = torch.zeros(2, 8, device=device, dtype=spec.dtype)
    spec._prefix_lens = torch.zeros(2, device=device, dtype=torch.int32)
    spec._prefix_block_table = torch.zeros(2, 1, device=device, dtype=torch.int32)
    spec.shared_kv_cudagraph_manager = SimpleNamespace(
        dispatch=lambda *args: SimpleNamespace(cg_mode=CUDAGraphMode.NONE)
    )
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


@pytest.mark.parametrize("window", [0, 8192])
def test_shared_kv_export_pins_checkpoint_and_rejects_changed_weights(tmp_path, window):
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
                "config": asdict(SharedKVConfig(draft_kv_window=window)),
                "frozen_target": str(target),
                "initialization_schema": "sharedkv-swa8k-full20k-v1"
                if window
                else None,
            }
        )
    )
    (source / "model.safetensors").write_bytes(b"hash-verification-fixture")
    receipt = {
        "status": "PASS",
        "step": 3750,
        "config": {
            "run_name": "sharedkv4-swa8k-dp16-full20k"
            if window
            else "sharedkv-balanced-dp16-8k10k-full10k"
        },
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
    assert config["shared_kv_config"]["draft_kv_window"] == window
    assert config["shared_kv_receipt_scope"] == "published"
    assert (output / "model.safetensors").read_bytes() == b"hash-verification-fixture"
    (checkpoint / "publication.json").unlink()
    with pytest.raises(ValueError, match="Publication receipt missing"):
        export(checkpoint, target, tmp_path / "unpublished-rejected")
    config = export(checkpoint, target, tmp_path / "committed", committed_only=True)
    assert config["shared_kv_receipt_scope"] == "committed-only"
    # A present, contradictory publication can never be bypassed by the flag.
    conflicting = dict(receipt, step=3751)
    (checkpoint / "publication.json").write_text(json.dumps(conflicting))
    with pytest.raises(ValueError, match="publication step mismatch"):
        export(checkpoint, target, tmp_path / "conflicting", committed_only=True)
    (checkpoint / "publication.json").unlink()
    (source / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checkpoint hash mismatch"):
        export(checkpoint, target, tmp_path / "must-not-exist", committed_only=True)
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay")
@pytest.mark.parametrize("probabilistic", [False, True])
@torch.inference_mode()
def test_shared_kv_graph_replay_updates_prefix_pages_and_sampling(probabilistic):
    """A captured backbone + Markov sampler must follow new inputs and inert padding."""
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.spec_decode.dspark.shared_kv import (
        SharedKVSpeculator,
        prepare_shared_kv_inputs,
    )

    torch.manual_seed(17)
    device, dtype = torch.device("cuda"), torch.bfloat16
    c = SharedKVConfig(
        hidden=64,
        ffn=96,
        heads=4,
        kv_heads=2,
        head_dim=32,
        layers=2,
        vocab=128,
        block=4,
        mask_id=127,
        rank=16,
    )
    model = SharedKVDraft(c).to(device).eval()
    for p in model.parameters():
        p.uniform_(-0.2, 0.2)
    model.embed_tokens.to(dtype)
    model.lm_head.to(dtype)
    spec = SharedKVSpeculator.__new__(SharedKVSpeculator)
    spec.model, spec.device, spec.dtype = model, device, dtype
    spec.num_speculative_steps = c.block
    spec._draft_topk = spec._d2t_scatter_index = spec.acceptance_estimator = None
    spec.draft_watermarker = None
    spec.use_confidence_head, spec.use_fp64_gumbel = True, False
    spec._step_cols = torch.arange(c.block, device=device, dtype=torch.int32)
    spec._anchor_idx = torch.arange(4, device=device) * c.block
    spec.input_buffers = SimpleNamespace(
        input_ids=torch.zeros(16, device=device, dtype=torch.long),
        positions=torch.zeros(16, device=device, dtype=torch.long),
    )
    spec._current_hidden = torch.zeros(4, c.hidden, device=device, dtype=dtype)
    spec._prefix_lens = torch.zeros(4, device=device, dtype=torch.int32)
    spec._prefix_block_table = torch.zeros(4, 2, device=device, dtype=torch.int32)
    spec.sample_indices = torch.arange(16, device=device)
    spec.sample_pos = torch.zeros(16, device=device, dtype=torch.long)
    spec.sample_idx_mapping = torch.full((16,), -1, device=device, dtype=torch.int32)
    spec.temperature = torch.zeros(4, device=device)
    spec.seeds = torch.zeros(4, device=device, dtype=torch.long)
    spec.draft_tokens = torch.zeros(4, 4, device=device, dtype=torch.long)
    spec.draft_token_confidence_probs = torch.zeros(4, 4, device=device)
    spec.draft_logits = (
        torch.full((4, 4, c.vocab), 123.0, device=device) if probabilistic else None
    )
    cache = torch.randn(10, c.kv_heads, 256, 2 * c.head_dim, device=device, dtype=dtype)
    before = cache.clone()
    table = torch.tensor(
        [[3, 5], [8, 4], [6, 2], [7, 1]], device=device, dtype=torch.int32
    )
    spec._target_gid = 0
    spec.block_tables = SimpleNamespace(input_block_tables=[table])
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            spec._run_shared_kv(4, cache, spec._prefix_block_table, 512)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph, stream=stream):
        spec._run_shared_kv(4, cache, spec._prefix_block_table, 512)

    for n, base in [(3, 300), (1, 11), (4, 401)]:
        table.copy_(table.roll(1, 0))
        positions = torch.arange(n * 4, device=device).reshape(n, 4) + base
        mapping = torch.tensor([3, 1, 0, 2], device=device, dtype=torch.int32)[:n]
        batch = SimpleNamespace(
            num_reqs=n,
            positions=positions.flatten(),
            query_start_loc=torch.arange(n + 1, device=device, dtype=torch.int32) * 4,
            idx_mapping=mapping,
        )
        hidden = torch.randn(n * 4, c.hidden, device=device, dtype=dtype)
        rejected = torch.tensor([2, 0, 3, 1], device=device, dtype=torch.int32)[:n]
        sampled = torch.tensor([1, 0, 1, 2], device=device, dtype=torch.int32)[:n]
        prepare_shared_kv_inputs(
            spec,
            batch,
            hidden,
            sampled,
            rejected,
            torch.tensor([[10], [20], [30], [40]], device=device),
            torch.tensor([[50, 60, 70, 80]], device=device),
            torch.tensor([0.0, 0.7, 1.0, 0.2], device=device),
            torch.tensor([11, 22, 33, 44], device=device),
            4,
        )
        anchors, current = select_shared_kv_context(
            batch.positions, batch.query_start_loc, rejected, hidden
        )
        torch.testing.assert_close(spec._prefix_lens[:n].long(), anchors)
        torch.testing.assert_close(spec._current_hidden[:n], current)
        assert spec._prefix_lens[n:].count_nonzero() == 0
        assert (spec.sample_idx_mapping[n * 4 :] == -1).all()
        if spec.draft_logits is not None:
            spec.draft_logits.fill_(123)
        spec._run_shared_kv(4, cache, spec._prefix_block_table, 512)
        expected = spec.draft_tokens.clone()
        confidence = spec.draft_token_confidence_probs.clone()
        logits = spec.draft_logits.clone() if probabilistic else None
        if spec.draft_logits is not None:
            spec.draft_logits.fill_(123)
        spec.draft_tokens.fill_(-1)
        graph.replay()
        torch.testing.assert_close(spec.draft_tokens[:n], expected[:n], rtol=0, atol=0)
        torch.testing.assert_close(
            spec.draft_token_confidence_probs[:n], confidence[:n], rtol=0, atol=0
        )
        if spec.draft_logits is not None:
            torch.testing.assert_close(spec.draft_logits, logits, rtol=0, atol=0)
    torch.testing.assert_close(cache, before, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA window pages")
@pytest.mark.parametrize("physical_layout", ["HND", "NHD"])
@torch.inference_mode()
def test_swa8192_reads_exact_window_and_replays_changed_pages(physical_layout):
    """Check exact windows, immutable cache, shared slots and changing graphs."""
    from vllm.model_executor.models.yoco_shared_kv_swa import paged_window_attention

    torch.manual_seed(42)
    page, heads, kvheads, dim = 256, 4, 2, 64
    cache = torch.randn(80, kvheads, page, 2 * dim, device="cuda", dtype=torch.bfloat16)
    if physical_layout == "NHD":
        cache = cache.transpose(1, 2).contiguous().transpose(1, 2)
    # Different mappings exercise physical pages, including non-page-aligned starts.
    table = torch.stack(
        [torch.randperm(80, device="cuda")[:70] for _ in range(8)]
    ).int()
    ends = torch.tensor(
        [0, 1, 8191, 8192, 8193, 8255, 8449, 17003], device="cuda", dtype=torch.int32
    )
    q = torch.randn(8, 8, heads, dim, device="cuda", dtype=torch.bfloat16)
    before = cache.clone()

    def reference(query):
        outputs = []
        for i, end in enumerate(ends.tolist()):
            if not end:
                outputs.append(torch.zeros_like(query[i]).float())
                continue
            history = (
                cache[table[i].long()].permute(1, 0, 2, 3).reshape(kvheads, -1, 2 * dim)
            )
            keys, values = history[:, max(0, end - 8192) : end].split(dim, -1)
            outputs.append(
                F.scaled_dot_product_attention(
                    query[i].transpose(0, 1).float(),
                    keys.float(),
                    values.float(),
                    enable_gqa=True,
                ).transpose(0, 1)
            )
        return torch.stack(outputs)

    for block in [1, 6, 8]:
        query = q[:, :block].contiguous()
        actual = paged_window_attention(query, cache, table, ends)
        torch.testing.assert_close(
            actual.float(), reference(query), atol=0.006, rtol=0.03
        )
    torch.testing.assert_close(cache, before, rtol=0, atol=0)
    # Every slot uses the SAME interval; identical queries must produce equal rows.
    identical = q[:, :1].expand(-1, 8, -1, -1).contiguous()
    output = paged_window_attention(identical, cache, table, ends)
    torch.testing.assert_close(output, output[:, :1].expand_as(output), rtol=0, atol=0)
    # Poison positions strictly outside one window, including rejection/future tail.
    end = 17003
    one_table = table[7:8].clone()
    one_end = ends[7:8].clone()
    one_q = q[7:8].clone()
    expected = paged_window_attention(one_q, cache, one_table, one_end)
    poisoned = cache.clone()
    for pos in list(range(end - 8192)) + list(range(end, 70 * page)):
        poisoned[one_table[0, pos // page], :, pos % page] = 100
    actual = paged_window_attention(one_q, poisoned, one_table, one_end)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            paged_window_attention(q, cache, table, ends)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = paged_window_attention(q, cache, table, ends)
    for values in [
        [0, 5, 8193, 8300, 8448, 16385, 16999, 17001],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]:
        ends.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
        table.copy_(table.roll(1, 1))
        eager = paged_window_attention(q, cache, table, ends)
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        torch.testing.assert_close(
            captured.float(), reference(q), atol=0.006, rtol=0.03
        )
