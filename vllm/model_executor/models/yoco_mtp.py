# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO MTP=1: Q-only attention over the target's shared cross-attention KV."""

from collections.abc import Iterable
from typing import Any, cast

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.layers.yoco_attention import YOCOCrossAttention
from vllm.model_executor.layers.yoco_moe import YOCOSharedExperts
from vllm.model_executor.layers.yoco_ops.norm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.models.yoco_config import _cfg_int


class YOCOMTPBlock(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        config = vllm_config.model_config.hf_config
        width = _cfg_int(config, "hidden_size", "d_model")
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        for name in ("embedding_norm", "hidden_norm", "input_norm", "post_attn_norm"):
            setattr(self, name, RMSNorm(width, eps=eps))
        self.proj = ColumnParallelLinear(
            width, width, bias=False, gather_output=True, prefix=f"{prefix}.proj"
        )
        # An owner during construction; the speculator links it to the target
        # before KV cache discovery. No draft call ever writes K/V.
        self.self_attn = YOCOCrossAttention(
            config,
            0,
            0,
            vllm_config.cache_config,
            None,
            prefix=f"{prefix}.self_attn",
        )
        self.ffn = YOCOSharedExperts(
            width,
            _cfg_int(config, "d_ffn", "intermediate_size"),
            None,
            reduce_results=True,
            prefix=f"{prefix}.ffn",
            swiglu_limit=float(getattr(config, "swiglu_limit", 10.0)),
        )

    def forward(self, hidden_states: torch.Tensor, token_emb: torch.Tensor):
        h = self.embedding_norm(token_emb) + self.hidden_norm(hidden_states)
        h, _ = self.proj(h)
        attention = self.self_attn(self.input_norm(h), None, None)
        h = h.float() + attention.float()
        return h + self.ffn(self.post_attn_norm(h)).float()


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "hidden_states": 0,
        "inputs_embeds": 0,
    }
)
class YOCOMTPModel(nn.Module):
    """The speculator supplies the target output head for lookup and logits."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        if getattr(config, "n_predict", None) != 1:
            raise ValueError("YOCO MTP currently supports exactly one trained depth")
        if vllm_config.quant_config is not None:
            raise ValueError("YOCO MTP requires unquantized BF16 draft weights")
        self.config = config
        self.model = YOCOMTPBlock(vllm_config, prefix=maybe_prefix(prefix, "mtp.0"))
        self.norm = RMSNorm(
            _cfg_int(config, "hidden_size", "d_model"),
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        # Do not allocate or reload a second 1.5 GiB output embedding.
        self.lm_head = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.lm_head is not None
        # The trainer embeds shifted tokens with output.weight, not the
        # independent input embedding. ParallelLMHead overrides forward().
        return VocabParallelEmbedding.forward(self.lm_head, input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_emb = self.embed_input_ids(input_ids)
        return self.norm(self.model(hidden_states, token_emb))

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()
        source_names = set()
        for name, weight in weights:
            if name in source_names:
                raise ValueError(f"Duplicate YOCO MTP tensor: {name}")
            source_names.add(name)
            shard = None
            target = (
                name.replace("mtp.", "model.", 1) if name.startswith("mtp.") else name
            )
            if (
                target.endswith(".self_attn.q_proj.weight")
                and self.model.self_attn.q_lambda_proj is not None
            ):
                target = target.replace("q_proj", "q_lambda_proj")
                shard = 0
            elif (
                target.endswith(".self_attn.lambda_proj.weight")
                and self.model.self_attn.q_lambda_proj is not None
            ):
                target = target.replace("lambda_proj", "q_lambda_proj")
                shard = 1
            elif target.endswith(".ffn.gate_up_proj.weight"):
                param = params[target]
                gate, up = weight.chunk(2, dim=0)
                param.weight_loader(param, gate, 0)
                param.weight_loader(param, up, 1)
                loaded.add(target)
                continue
            if target not in params:
                raise ValueError(f"Unexpected YOCO MTP tensor: {name}")
            param = params[target]
            loader = cast(Any, getattr(param, "weight_loader", default_weight_loader))
            if shard is None:
                loader(param, weight)
            else:
                loader(param, weight, shard)
            loaded.add(target)
        # Check source projections separately: a partially populated packed
        # q/gate parameter must not count as a complete checkpoint load.
        from vllm.transformers_utils.configs.yoco_mtp import expected_mtp_shapes

        expected = set(expected_mtp_shapes(self.config.to_dict()))
        if source_names != expected or loaded != set(params):
            raise ValueError(
                f"Incomplete YOCO MTP weights: source={expected - source_names}, "
                f"parameters={set(params) - loaded}"
            )
        return loaded
