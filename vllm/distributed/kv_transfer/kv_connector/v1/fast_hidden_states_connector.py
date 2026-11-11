# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Non-blocking hidden-states connector for the OPD teacher.

Same on-disk format and filename convention as ExampleHiddenStatesConnector
({storage}/{req_id}.safetensors holding {hidden_states, token_ids}), so the
reader is unchanged.  The difference is purely performance:

  * ExampleHiddenStatesConnector.save_kv_layer does a *synchronous* GPU->CPU copy
    AND safetensors serialize+disk write, for EVERY request, on EVERY step -> it
    stalls the engine step (measured ~5k tok/s / 47% GPU util on GLM-5.2).
  * This connector (a) saves each request ONCE (dedup; with chunked prefill
    disabled the prefill step already holds every prompt-token hidden), and
    (b) offloads the serialize+disk write to a background thread pool so the
    engine step is not stalled.  The tiny GPU->CPU copy stays on the hot path
    (needed before the paged slot is reused) but the slow part overlaps compute.

Migrated (jw/opd-teacher-migrate-vllm, 2026-11-10) from msrallm verl
teacher_vllm_server/fast_hidden_states_connector.py into the vLLM fork and registered
in the connector factory as "FastHiddenStatesConnector". The only change is the
startup writer check, which no longer imports verl: the 0.29 stock aux-HS cache writer
is padding-safe, so the check is the same behavioural probe the verl side used.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import safetensors.torch
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    ExampleHiddenStatesConnector,
    extract_from_kv_cache,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata

logger = init_logger(__name__)


@dataclass
class _ChunkedReqMeta:
    """Per-request save descriptor for the chunked-prefill-safe connector.

    The base ReqMeta is emitted every step and read via the CURRENT step's
    slot_mapping (only correct when the whole prompt is prefilled in one forward).
    This descriptor is instead emitted EXACTLY ONCE -- on the step whose forward
    completes the prompt's prefill -- and carries the full per-kv-cache-group block
    lists so the worker can rebuild the WHOLE sequence's slot mapping even when the
    prompt was chunk-prefilled across many forwards."""

    req_id: str
    filename: str
    token_ids: torch.Tensor
    save_now: bool = False
    # Full-sequence block ids for EACH kv-cache group. The worker selects the
    # aux-hidden (CacheOnly) group via self._aux_group_idx. Populated only when
    # save_now=True (keeps per-step metadata tiny).
    block_ids: tuple = ()


@dataclass
class _ChunkedHSMetadata(KVConnectorMetadata):
    requests: list = field(default_factory=list)

# NOTE: the final pre-norm (end-layer sentinel) capture this connector serializes is
# native in this fork (models/deepseek_v32/nvidia/model.py); no runtime patch is needed.


def _require_padding_safe_hidden_state_writer() -> None:
    """Refuse to serve unless the aux-HS cache writer skips padding slots.

    Behavioural probe copied from msrallm vllm_aux_hidden_patch._native_writer_is_padding_safe:
    write one real slot and one -1 padding slot into a [blocks, heads, block_size, head]
    cache; the 0.29+ writer clamps padding to the null block and must not touch the
    neighbouring slots.
    """
    from vllm.model_executor.models import extract_hidden_states

    writer = getattr(extract_hidden_states, "basic_cache", None)
    ok = False
    if writer is not None:
        try:
            kv = torch.zeros(2, 1, 4, 3)
            x = torch.ones(2, 1, 3)
            writer(x, kv, torch.tensor([5, -1]))
            ok = bool((kv[1, 0, 1] == 1).all()) and bool(
                (kv[1, 0, 2] == 0).all() and (kv[1, 0, 3] == 0).all())
        except Exception:
            ok = False
    if not ok:
        raise RuntimeError(
            "FastHiddenStatesConnector: the aux-HS cache writer is not padding-safe on "
            "this vLLM build; refusing to serve teacher hidden states")


def _atomic_save(filename: str, hidden_states: torch.Tensor, token_ids: torch.Tensor) -> None:
    """Serialize + write on a worker thread; rename so readers never see a partial file.

    The temp name is per-process (pid) so that even if two writers ever target the same
    final path, they never share -- and truncate -- the same temp file; the final file
    (produced by an atomic os.replace) is therefore always complete."""
    tmp = f"{filename}.{os.getpid()}.tmp"
    safetensors.torch.save_file({"hidden_states": hidden_states, "token_ids": token_ids}, tmp)
    os.replace(tmp, filename)


def _atomic_save_async(filename, event, pinned, token_ids, gpu_ref) -> None:
    """Path A: wait for the in-flight async GPU->CPU copy to finish (event.synchronize
    is CPU-side and thread-safe), release the GPU staging tensor, then serialize the
    pinned host copy to disk (temp + rename)."""
    event.synchronize()
    del gpu_ref
    tmp = f"{filename}.{os.getpid()}.tmp"
    safetensors.torch.save_file({"hidden_states": pinned, "token_ids": token_ids}, tmp)
    os.replace(tmp, filename)


class FastHiddenStatesConnector(ExampleHiddenStatesConnector):
    def __init__(self, *args, **kwargs):
        # This runs in scheduler/TP processes independently of plugin discovery.
        # Refuse native extraction if a filtered or unloadable plugin left the
        # image's unsafe writer active; never repair it after model warmup.
        _require_padding_safe_hidden_state_writer()
        super().__init__(*args, **kwargs)
        # The 0.29 base connector no longer keeps this map (the 0.23 base did);
        # our build_connector_meta override relies on it.
        if not hasattr(self, "_active_requests"):
            self._active_requests = {}
        # Capture kv_cache_config (positional idx 2 or kwarg) so register_kv_caches can
        # resolve which kv-cache group holds the aux-hidden layer.
        self._kv_cache_config = kwargs.get("kv_cache_config")
        if self._kv_cache_config is None and len(args) >= 3:
            self._kv_cache_config = args[2]
        n_writers = int(os.getenv("HS_CONN_WRITERS", "8"))
        self._pool = ThreadPoolExecutor(max_workers=n_writers,
                                        thread_name_prefix="hs-writer")
        # Single-writer gate (lazily resolved). Only TP rank 0 writes each file.
        self._writer_rank: int | None = None
        # Path A: issue the GPU->CPU copy on a side stream so save_kv_layer does NOT
        # force a per-step device sync (that sync -- not bandwidth -- was the cap).
        # The writer thread waits on a CUDA event, then serializes.
        self._async_copy = os.getenv("HS_CONN_ASYNC_COPY", "1") == "1"
        self._copy_stream = None
        # --- chunked-prefill state (scheduler-side) ---
        # Under chunked prefill a prompt is prefilled across MANY forwards; each request
        # must be saved ONCE, after its LAST prefill chunk, reading every position from
        # the full block list (not the current step's partial slot_mapping).
        self._save_emitted: set[str] = set()       # req_id -> save descriptor already emitted
        self._req_blocks_all: dict[str, list] = {}  # req_id -> [block_ids per kv-cache group]
        # kv-cache group index of the aux-hidden CacheOnly layer (resolved worker-side in
        # register_kv_caches). The MLA KV lives in a DIFFERENT group, so group 0 is NOT
        # reliably the aux group.
        self._aux_group_idx: int = 0
        logger.info("FastHiddenStatesConnector: %d writer threads, async_copy=%s, path=%s",
                    n_writers, self._async_copy, self._storage_path)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        if layer_name not in self.cache_layers:
            return

        # Single-writer gate. The aux hidden (final pre-norm residual stream) is
        # REPLICATED across TP ranks, so every rank's save_kv_layer would write the
        # SAME {req_id}.safetensors. Multiple processes racing on one path truncate it
        # -> the reader hits "incomplete metadata, file not fully covered". This is rare
        # with one hidden-states request per step but frequent under streaming (many
        # small pre_compute_hs calls/step). Write only from TP rank 0 (its hidden is the
        # full, correct tensor). Falls back to writing if the rank can't be resolved.
        if self._writer_rank is None:
            try:
                from vllm.distributed import get_tensor_model_parallel_rank
                self._writer_rank = get_tensor_model_parallel_rank()
            except Exception:
                self._writer_rank = 0
        if self._writer_rank != 0:
            return

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, _ChunkedHSMetadata)
        # Only act on steps where some request's prefill just completed (save_now).
        if not any(r.save_now for r in connector_metadata.requests):
            return
        os.makedirs(self._storage_path, exist_ok=True)

        use_async = self._async_copy and kv_layer.is_cuda
        if use_async and self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream()

        # KV view layout differs by vLLM line: 0.23 = [blocks, block_size, heads, head],
        # 0.29 = [blocks, heads, block_size, head]. OPD captures exactly ONE aux layer
        # (heads == 1), so the singleton axis identifies which dimension is block_size
        # and the two layouts are byte-identical in memory. The bundled
        # extract_from_kv_cache is version-matched; only OUR slot math needs this.
        if kv_layer.shape[2] == 1 and kv_layer.shape[1] > 1:
            block_size = kv_layer.shape[1]          # 0.23: [B, N, 1, C]
        elif kv_layer.shape[1] == 1 and kv_layer.shape[2] > 1:
            block_size = kv_layer.shape[2]          # 0.29: [B, 1, N, C]
        else:
            raise RuntimeError(
                f"cannot infer aux-HS block_size from kv layout {tuple(kv_layer.shape)}; "
                f"this connector supports exactly one captured aux layer")
        gi = self._aux_group_idx
        for request in connector_metadata.requests:
            if not request.save_now:
                continue
            num_tokens = int(request.token_ids.shape[0])
            # Rebuild the FULL sequence slot mapping from the aux-hidden group's block
            # table: slot = block_id * block_size + offset_within_block. This is exactly
            # the mapping vLLM builds in attn_metadata.slot_mapping for a single-forward
            # prefill (so results are byte-for-byte identical when NOT chunked), and it
            # also covers positions written in earlier chunked-prefill forwards (whose
            # aux hidden persists in the paged buffer until the request is freed).
            groups = request.block_ids
            if not 0 <= gi < len(groups):
                raise RuntimeError(
                    f"aux-HS group {gi} is missing from {len(groups)} block groups for {request.req_id}")
            blocks = groups[gi]
            n_blocks = (num_tokens + block_size - 1) // block_size
            if len(blocks) < n_blocks:
                raise RuntimeError(
                    f"incomplete aux-HS block table for {request.req_id}: "
                    f"{len(blocks)} blocks, need {n_blocks} for {num_tokens} tokens")
            if any(b < 0 or b >= kv_layer.shape[0] for b in blocks[:n_blocks]):
                raise RuntimeError(f"aux-HS block ID out of range for {request.req_id}")
            blk = torch.as_tensor(list(blocks[:n_blocks]), dtype=torch.long,
                                  device=kv_layer.device)
            pos = torch.arange(num_tokens, device=kv_layer.device)
            slot_mapping = blk[pos // block_size] * block_size + (pos % block_size)

            # extract_from_kv_cache returns a FRESH tensor (advanced indexing), so it
            # is safe from paged-buffer reuse while an async copy is in flight.
            hidden_states = extract_from_kv_cache(kv_layer, slot_mapping, num_tokens)
            # Do NOT nan_to_num here: a finite-looking replacement is still a wrong
            # distillation target and can evade the actor's all-zero-row probe. The
            # native reader validates the full CPU tensor before accepting it. Keeping
            # that check there preserves this path's asynchronous GPU->CPU copy.
            tok_cpu = request.token_ids.detach().cpu()

            if use_async:
                # Issue GPU->CPU on a side stream; do NOT sync here. The stream first
                # waits for the gather on the current stream; the writer thread later
                # waits on the event before serializing the pinned host buffer.
                self._copy_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._copy_stream):
                    pinned = torch.empty(tuple(hidden_states.shape),
                                         dtype=hidden_states.dtype,
                                         device="cpu", pin_memory=True)
                    pinned.copy_(hidden_states, non_blocking=True)
                    event = torch.cuda.Event()
                    event.record(self._copy_stream)
                self._pool.submit(_atomic_save_async, request.filename, event,
                                  pinned, tok_cpu, hidden_states)
            else:
                hs_cpu = hidden_states.detach().to("cpu")
                self._pool.submit(_atomic_save, request.filename, hs_cpu, tok_cpu)

    def register_kv_caches(self, kv_caches):
        # Base sets self.cache_layers (the CacheOnlyAttentionLayer name(s)).
        super().register_kv_caches(kv_caches)
        # Resolve which kv-cache group holds the aux-hidden (CacheOnly) layer. The MLA
        # KV lives in a separate group with a different spec, so we CANNOT assume the
        # aux group is index 0 -- match by layer name.
        groups = getattr(self._kv_cache_config, "kv_cache_groups", None)
        if not groups:
            raise RuntimeError("cannot resolve aux-HS group: KV cache groups are missing")
        cache_set = set(self.cache_layers)
        matches = [i for i, group in enumerate(groups) if cache_set.intersection(group.layer_names)]
        if len(matches) != 1:
            raise RuntimeError(
                f"cannot resolve a unique aux-HS group for {self.cache_layers}: matches={matches}")
        self._aux_group_idx = matches[0]
        logger.info("FastHiddenStatesConnector: aux-hidden kv-cache group idx=%d "
                    "(of %d group(s)); cache_layers=%s",
                    self._aux_group_idx, len(groups), self.cache_layers)

    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
        """Scheduler-side. Emit a save descriptor for each request EXACTLY ONCE -- on
        the step whose forward completes the prompt's prefill. Completion uses the
        AUTHORITATIVE num_computed_tokens (which resets correctly on preemption+resume),
        and block lists are REPLACED (not appended) for resumed requests, so the full
        slot mapping stays correct even if the scheduler preempts under KV pressure."""
        meta = _ChunkedHSMetadata()
        num_sched = scheduler_output.num_scheduled_tokens

        for new_req in scheduler_output.scheduled_new_reqs:
            rid = new_req.req_id
            self._request_filenames[rid] = os.path.join(
                self._storage_path, f"{rid}.safetensors")
            self._active_requests[rid] = new_req
            self._req_blocks_all[rid] = [list(g) for g in new_req.block_ids]
            computed_after = new_req.num_computed_tokens + num_sched.get(rid, 0)
            self._maybe_emit(meta, rid, new_req.prompt_token_ids or [], computed_after)

        cached = scheduler_output.scheduled_cached_reqs
        resumed = cached.resumed_req_ids or set()
        for i, rid in enumerate(cached.req_ids):
            if rid not in self._active_requests:
                continue
            new_block_ids = cached.new_block_ids[i]
            if new_block_ids is not None:
                if rid in resumed:
                    # Preemption resume: new_block_ids REPLACES the block list.
                    self._req_blocks_all[rid] = [list(g) for g in new_block_ids]
                else:
                    for gidx, g in enumerate(new_block_ids):
                        if gidx < len(self._req_blocks_all[rid]):
                            self._req_blocks_all[rid][gidx].extend(g)
            computed_after = cached.num_computed_tokens[i] + num_sched.get(rid, 0)
            prompt = self._active_requests[rid].prompt_token_ids or []
            self._maybe_emit(meta, rid, prompt, computed_after)

        return meta

    def _maybe_emit(self, meta, rid, prompt, computed_after):
        """Append a save descriptor once the whole prompt has been computed (prefill
        done). computed_after = num_computed_tokens (authoritative, pre-step) + tokens
        scheduled this step."""
        if rid in self._save_emitted:
            return
        if computed_after < len(prompt):
            return  # prefill not finished yet -> more chunks to come
        self._save_emitted.add(rid)
        meta.requests.append(_ChunkedReqMeta(
            req_id=rid,
            filename=self._request_filenames[rid],
            token_ids=torch.tensor(prompt),
            save_now=True,
            block_ids=tuple(list(g) for g in self._req_blocks_all[rid]),
        ))

    def request_finished(self, request, block_ids):
        rid = request.request_id
        self._save_emitted.discard(rid)
        self._req_blocks_all.pop(rid, None)
        # The 0.23 base pops _active_requests in its request_finished; the 0.29
        # base dropped the map entirely (we re-create it in __init__ for our
        # build_connector_meta override), so WE must pop it or every request
        # leaks a NewRequestData holding the full prompt token-id list for the
        # life of the scheduler process. No-op on 0.23 (super pops it below).
        self._active_requests.pop(rid, None)
        out = super().request_finished(request, block_ids)
        # 0.29 base queues a PendingSave here for its own finish-time extraction
        # flow; our build_connector_meta override never drains it -> unbounded
        # scheduler-side growth. Worker-side get_finished is unaffected (no copy
        # event under our metadata -> reports done immediately).
        _pending = getattr(self, "_pending_saves", None)
        if isinstance(_pending, dict):
            _pending.pop(rid, None)
        return out

    def wait_for_save(self) -> None:
        # Deliberately non-blocking: writes finish on the background pool and the
        # reader waits for the files (atomic rename) after generate() returns.
        pass
