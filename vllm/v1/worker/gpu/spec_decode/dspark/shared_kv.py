# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP1 Shared-KV DSpark with full draft CUDA graphs for the V2 runner."""

import hashlib
from pathlib import Path

import torch
from safetensors.torch import load_file

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.models.yoco_shared_kv import (
    SharedKVConfig,
    SharedKVDraft,
    paged_prefix_attention,
)
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

logger = init_logger(__name__)


def select_shared_kv_context(positions, query_start_loc, num_rejected, raw_hidden):
    """The last accepted input precedes the bonus anchor by exactly one token."""
    last = query_start_loc[1:].long() - num_rejected.long() - 1
    return positions[last] + 1, raw_hidden[last]


class SharedKVSpeculator(DSparkSpeculator):
    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device)
        p = vllm_config.parallel_config
        c = vllm_config.model_config
        hf = self.draft_model_config.hf_config
        if (
            p.tensor_parallel_size != 1
            or p.pipeline_parallel_size != 1
            or p.data_parallel_size != 1
            or p.prefill_context_parallel_size != 1
            or p.decode_context_parallel_size != 1
            or self.speculative_config.draft_parallel_config.tensor_parallel_size != 1
        ):
            raise ValueError("Shared-KV currently requires TP1/PP1/DP1 without CP")
        if c.hf_text_config.model_type != "yoco" or c.dtype != torch.bfloat16:
            raise ValueError("Shared-KV requires the paired BF16 YOCO target")
        if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise ValueError("Shared-KV does not support quantized target KV")
        if (
            c.quantization is not None
            or self.draft_model_config.quantization is not None
        ):
            raise ValueError("Shared-KV requires unquantized target and draft weights")
        if (
            not self.sample_from_anchor
            or not 1 <= self.num_speculative_steps <= hf.shared_kv_config["block"]
        ):
            raise ValueError(
                "Shared-KV requires anchor sampling and K <= trained block"
            )
        if self._draft_topk is not None or self.use_local_argmax_reduction:
            raise ValueError("Shared-KV does not implement top-k/local argmax drafting")
        if vllm_config.lora_config is not None:
            raise ValueError("Shared-KV does not support target LoRA")
        self._draft_topk = None
        self.shared_kv_cudagraph_manager: CudaGraphManager | None = None
        self._current_hidden = torch.zeros(
            self.max_num_reqs, self.hidden_size, dtype=self.dtype, device=device
        )
        self._prefix_lens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.sample_indices.copy_(
            torch.arange(self.sample_indices.numel(), device=device)
        )

    def load_draft_model(self, target_model, target_attn_layer_names):
        hf = self.draft_model_config.hf_config
        expected_target = Path(hf.shared_kv_target).resolve()
        if Path(self.vllm_config.model_config.model).resolve() != expected_target:
            raise ValueError("Target differs from the balanced target pinned at export")
        c = SharedKVConfig(**hf.shared_kv_config)
        target = target_model.model
        if target.num_hidden_layers != hf.eagle_aux_hidden_state_layer_ids[0]:
            raise ValueError("Shared-KV must consume the target's final raw residual")
        self._cache_owner = target.decoder_layers[
            target.first_cross_layer_idx
        ].self_attn.attn
        if self._cache_owner.layer_name not in target_attn_layer_names:
            raise ValueError("Cannot resolve YOCO shared KV owner")
        if self._cache_owner.num_kv_heads != c.kv_heads:
            raise ValueError("Draft and target KV head counts differ")
        weights_path = Path(self.draft_model_config.model) / "model.safetensors"
        with weights_path.open("rb") as f:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: f.read(8 << 20), b""):
                hasher.update(chunk)
            digest = hasher.hexdigest()
        if digest != hf.shared_kv_weights_sha256:
            raise ValueError(
                "Shared-KV weights no longer match the committed checkpoint"
            )
        with torch.device("meta"):
            model = SharedKVDraft(c)
        state = load_file(str(weights_path), device=str(self.device))
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        if set(missing) != {"embed_tokens.weight", "lm_head.weight"} or unexpected:
            raise ValueError(f"Incomplete Shared-KV weights: {missing}, {unexpected}")
        model.embed_tokens = target.embed_tokens
        model.lm_head = target_model.lm_head
        if model.embed_tokens.weight.shape != (
            c.vocab,
            c.hidden,
        ) or model.lm_head.weight.shape != (c.vocab, c.hidden):
            raise ValueError("Balanced frozen vocabulary tensor shapes do not match")
        model.eval().requires_grad_(False)
        self.use_confidence_head = self.enable_adaptive_verification
        if self.use_confidence_head:
            self.use_acceptance_estimator = False
        return model

    def set_attn(
        self,
        model_state,
        kv_cache_config,
        block_tables,
        target_input_buffers,
        target_attn_groups,
    ):
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.block_tables = block_tables
        self.target_input_buffers = target_input_buffers
        self.target_attn_groups = target_attn_groups
        self.attn_groups = []
        self._target_gid = next(
            i
            for i, group in enumerate(kv_cache_config.kv_cache_groups)
            if self._cache_owner.layer_name in group.layer_names
        )
        # Reallocated together with the KV cache, including after memory profiling.
        self._prefix_block_table = torch.zeros_like(
            block_tables.input_block_tables[self._target_gid]
        )

    def init_cudagraph_manager(self, cudagraph_mode):
        mode = (
            CUDAGraphMode.FULL_DECODE_ONLY
            if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            else CUDAGraphMode.NONE
        )
        self.shared_kv_cudagraph_manager = CudaGraphManager(
            self.vllm_config,
            self.device,
            mode,
            decode_query_len=self.num_speculative_steps,
        )
        logger.info("Shared-KV draft CUDA graph mode: %s", mode)

    @torch.inference_mode()
    def capture(self):
        manager = self.shared_kv_cudagraph_manager
        assert manager is not None

        def create_forward_fn(desc, warmup):
            # Empty prefixes never read the target's dummy/uninitialized pages.
            # Inert sampling rows cannot scatter into any live request slot.
            self.input_buffers.input_ids.zero_()
            self.input_buffers.positions.zero_()
            self._current_hidden.zero_()
            self._prefix_lens.zero_()
            self._prefix_block_table.zero_()
            self.sample_pos.zero_()
            self.sample_idx_mapping.fill_(-1)
            return lambda mode: self._run_shared_kv(
                desc.num_reqs,
                self._cache_owner.kv_cache,
                self._prefix_block_table[: desc.num_reqs],
                self.max_model_len,
            )

        manager.capture(create_forward_fn, "Capturing Shared-KV draft CUDA graphs")

    def _run_shared_kv(self, n, cache, table, upper):
        k = self.num_speculative_steps

        def attend_prefix(q):
            return paged_prefix_attention(
                q.to(self.dtype), cache, table, self._prefix_lens[:n], upper
            )

        # Preserve FP32 trained parameters/residuals and BF16 matmuls.
        with torch.autocast("cuda", dtype=self.dtype):
            hidden = self.model(
                self.input_buffers.input_ids[: n * k].view(n, k),
                self._current_hidden[:n],
                self.input_buffers.positions[: n * k].view(n, k),
                attend_prefix,
            )
            self._sample_sequential(n, hidden.flatten(0, 1))

    @torch.inference_mode()
    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        dp_sync=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
    ):
        n = input_batch.num_reqs
        k = self.num_speculative_steps
        device = self.device
        c = self.model.c
        if dummy_run:
            anchors = torch.ones(n, device=device, dtype=torch.long)
            current = torch.zeros(n, c.hidden, device=device, dtype=self.dtype)
            cache = torch.zeros(
                1, c.kv_heads, 256, 2 * c.head_dim, device=device, dtype=self.dtype
            )
            table = torch.zeros(n, 1, device=device, dtype=torch.int32)
            upper = 1
            bonus = torch.zeros(n, device=device, dtype=torch.long)
            self.input_buffers.input_ids[: n * k].fill_(c.mask_id)
            self.input_buffers.input_ids[: n * k : k].copy_(bonus)
            self.input_buffers.positions[: n * k].copy_(
                (anchors[:, None] + torch.arange(k, device=device)).flatten()
            )
            self._current_hidden[:n].copy_(current)
            self._prefix_lens[:n].copy_(anchors)
            self.sample_pos[: n * k].zero_()
            self.sample_idx_mapping[: n * k].fill_(-1)
            self._run_shared_kv(n, cache, table, upper)
            return self.draft_tokens[:n]

        if not aux_hidden_states or len(aux_hidden_states) != 1:
            raise ValueError("Shared-KV expects exactly the raw h20 auxiliary stream")
        manager = self.shared_kv_cudagraph_manager
        assert manager is not None
        desc = manager.dispatch(n, n * k, k, 0)
        use_graph = desc.cg_mode == CUDAGraphMode.FULL and not is_profile
        padded = desc.num_reqs if use_graph else n
        prepare_shared_kv_inputs(
            self,
            input_batch,
            aux_hidden_states[0],
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            padded,
        )
        if use_graph:
            manager.run_fullgraph(desc)
        else:
            upper = int(input_batch.seq_lens_cpu_upper_bound[:n].max())
            self._run_shared_kv(
                n, self._cache_owner.kv_cache, self._prefix_block_table[:n], upper
            )
        return self.draft_tokens[:n]


@triton.jit
def _prepare_shared_kv_inputs(
    ids,
    positions,
    current,
    prefix_lens,
    table,
    sample_pos,
    sample_idx,
    out_temperature,
    out_seeds,
    target_positions,
    query_start,
    raw_hidden,
    idx_mapping,
    num_sampled,
    num_rejected,
    last_sampled,
    next_prefill,
    temperature,
    seeds,
    target_table,
    hidden_stride,
    target_table_stride,
    table_stride,
    num_reqs,
    hidden_size: tl.constexpr,
    block: tl.constexpr,
    mask_id: tl.constexpr,
    num_blocks: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    chunk = tl.program_id(1)
    col = chunk * BLOCK + tl.arange(0, BLOCK)
    live = req < num_reqs
    end = tl.load(query_start + req + 1, mask=live, other=1)
    rejected = tl.load(num_rejected + req, mask=live, other=0)
    last = end - rejected - 1
    anchor = tl.load(target_positions + last, mask=live, other=-1) + 1
    state_idx = tl.load(idx_mapping + req, mask=live, other=-1)
    sampled = tl.load(num_sampled + req, mask=live, other=0)
    bonus = tl.load(last_sampled + state_idx, mask=live & (sampled > 0), other=0)
    next_token = tl.load(next_prefill + state_idx, mask=live & (sampled == 0), other=0)
    bonus = tl.where(sampled > 0, bonus, next_token)
    tl.store(ids + req * block + col, tl.where(col == 0, bonus, mask_id), col < block)
    tl.store(positions + req * block + col, anchor + col, col < block)
    tl.store(sample_pos + req * block + col, anchor + col + 1, col < block)
    tl.store(sample_idx + req * block + col, state_idx, col < block)
    h = tl.load(
        raw_hidden + last * hidden_stride + col,
        mask=live & (col < hidden_size),
        other=0,
    )
    tl.store(current + req * hidden_size + col, h, col < hidden_size)
    page = tl.load(
        target_table + req * target_table_stride + col,
        mask=live & (col < num_blocks),
        other=0,
    )
    tl.store(table + req * table_stride + col, page, col < num_blocks)
    if chunk == 0:
        tl.store(prefix_lens + req, anchor)
        temp = tl.load(temperature + state_idx, mask=live, other=0)
        seed = tl.load(seeds + state_idx, mask=live, other=0)
        tl.store(out_temperature + state_idx, temp, mask=live)
        tl.store(out_seeds + state_idx, seed, mask=live)


def prepare_shared_kv_inputs(
    spec,
    batch,
    hidden,
    num_sampled,
    num_rejected,
    last_sampled,
    next_prefill,
    temperature,
    seeds,
    padded,
):
    table = spec.block_tables.input_block_tables[spec._target_gid]
    c = spec.model.c
    width = max(c.hidden, table.shape[1], spec.num_speculative_steps)
    _prepare_shared_kv_inputs[(padded, triton.cdiv(width, 256))](
        spec.input_buffers.input_ids,
        spec.input_buffers.positions,
        spec._current_hidden,
        spec._prefix_lens,
        spec._prefix_block_table,
        spec.sample_pos,
        spec.sample_idx_mapping,
        spec.temperature,
        spec.seeds,
        batch.positions,
        batch.query_start_loc,
        hidden,
        batch.idx_mapping,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill,
        temperature,
        seeds,
        table,
        hidden.stride(0),
        table.stride(0),
        spec._prefix_block_table.stride(0),
        batch.num_reqs,
        c.hidden,
        spec.num_speculative_steps,
        c.mask_id,
        table.shape[1],
        BLOCK=256,
    )
