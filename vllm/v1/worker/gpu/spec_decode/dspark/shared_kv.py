# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager TP1 Shared-KV DSpark integration for the V2 runner."""

import hashlib
from pathlib import Path

import torch
from safetensors.torch import load_file

from vllm.logger import init_logger
from vllm.model_executor.models.yoco_shared_kv import (
    SharedKVConfig,
    SharedKVDraft,
    paged_prefix_attention,
)
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

    def init_cudagraph_manager(self, cudagraph_mode):
        logger.info("Shared-KV draft runs eagerly; target CUDA graphs remain available")

    def capture(self):
        pass

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
        else:
            if not aux_hidden_states or len(aux_hidden_states) != 1:
                raise ValueError(
                    "Shared-KV expects exactly the raw h20 auxiliary stream"
                )
            anchors, current = select_shared_kv_context(
                input_batch.positions,
                input_batch.query_start_loc[: n + 1],
                num_rejected[:n],
                aux_hidden_states[0],
            )
            cache = self._cache_owner.kv_cache
            table = self.block_tables.input_block_tables[self._target_gid][:n]
            upper = int(input_batch.seq_lens_cpu_upper_bound[:n].max())
            idx = input_batch.idx_mapping[:n].long()
            bonus = torch.where(
                num_sampled[:n] > 0,
                last_sampled.reshape(-1)[idx],
                next_prefill_tokens[0, idx],
            )
        ids = torch.full((n, k), c.mask_id, device=device, dtype=torch.long)
        ids[:, 0] = bonus
        positions = anchors[:, None] + torch.arange(k, device=device)[None]
        self.input_buffers.input_ids[: n * k].copy_(ids.flatten())
        self.sample_indices[: n * k].copy_(torch.arange(n * k, device=device))
        self.sample_pos[: n * k].copy_((positions + 1).flatten())
        if dummy_run:
            self.sample_idx_mapping[: n * k].fill_(-1)
        else:
            self.sample_idx_mapping[: n * k].copy_(idx[:, None].expand(n, k).flatten())
            self.temperature.copy_(temperature)
            self.seeds.copy_(seeds)

        def attend_prefix(q):
            return paged_prefix_attention(
                q.to(self.dtype), cache, table, anchors, upper
            )

        # Preserve FP32 trained parameters/residuals and BF16 matmuls, as in validation.
        with torch.autocast("cuda", dtype=self.dtype):
            hidden = self.model(ids, current, positions, attend_prefix)
            self._sample_sequential(n, hidden.flatten(0, 1))
        return self.draft_tokens[:n]
