# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-token YOCO MTP with read-only cross-model KV sharing."""

from copy import copy

import torch
from torch import nn

from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig, replace
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)

logger = init_logger(__name__)


class YOCOMTPSpeculator(AutoRegressiveSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        spec = vllm_config.speculative_config
        assert spec is not None
        if spec.num_speculative_tokens != 1:
            raise ValueError("YOCO MTP=1 requires num_speculative_tokens=1")
        parallel = vllm_config.parallel_config
        if (
            parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
        ):
            raise ValueError("YOCO MTP currently requires PP=PCP=DCP=1")
        if not vllm_config.model_config.enforce_eager:
            raise ValueError("YOCO MTP currently requires enforce_eager=True")
        if (
            spec.draft_parallel_config.tensor_parallel_size
            != parallel.tensor_parallel_size
        ):
            raise ValueError("YOCO MTP must use the same TP size as the target")
        super().__init__(vllm_config, device)
        # Keep the pre-final-norm target residual until MTP's own hidden_norm.
        self.hidden_states = torch.zeros_like(self.hidden_states, dtype=torch.float32)

    @property
    def advance_draft_positions(self) -> bool:
        return False

    @property
    def attn_vllm_config(self) -> VllmConfig:
        # Metadata builders need all draft queries, including during prefill.
        # A shallow copy preserves the engine-resolved, init=False KV layout;
        # dataclass replacement would silently discard it.
        config = copy(self.vllm_config)
        config.cache_config = copy(self.vllm_config.cache_config)
        config.cache_config.kv_sharing_fast_prefill = False
        return config

    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        target = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        if not hasattr(target, "get_mtp_target_hidden_states"):
            raise ValueError("YOCO MTP requires a YOCO target exposing pre-norm states")
        owner = target.model.decoder_layers[
            target.model.first_cross_layer_idx
        ].self_attn.attn
        if owner.layer_name not in target_attn_layer_names:
            raise ValueError("YOCO shared KV owner is missing from target attention")
        draft_config = replace(
            # Attention implementations retain this cache object while the
            # engine resolves its layout after model loading. Share the target
            # object here; only the metadata builders use the copy above.
            self.vllm_config,
            model_config=self.speculative_config.draft_model_config,
            parallel_config=self.speculative_config.draft_parallel_config,
        )
        with set_model_tag("eagle_head"):
            draft = get_model(
                vllm_config=draft_config,
                model_config=self.speculative_config.draft_model_config,
                load_config=self.speculative_config.draft_load_config,
            )
        attention = draft.model.self_attn.attn
        if (
            attention.num_kv_heads != owner.num_kv_heads
            or attention.head_size != owner.head_size
            or attention.get_attn_backend() != owner.get_attn_backend()
        ):
            raise ValueError("YOCO MTP and target KV layouts/backends must match")
        attention.kv_sharing_target_layer_name = owner.layer_name
        draft.lm_head = target.lm_head
        logger.info(
            "YOCO MTP=1: sharing target output embedding and KV cache %s",
            owner.layer_name,
        )
        return draft
