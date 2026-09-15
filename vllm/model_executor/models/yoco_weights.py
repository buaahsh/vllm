# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from vllm.model_executor.models.yoco import YOCOForCausalLM

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.yoco_moe import YOCOMoE as YOCOMoE
from vllm.model_executor.layers.yoco_moe import YOCOSharedExperts as YOCOSharedExperts
from vllm.model_executor.layers.yoco_ops.rotary import (
    YOCORotaryEmbedding as YOCORotaryEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.model_executor.models.yoco_config import _cfg_int as _cfg_int


def _initialize_rotary_caches(self: YOCOForCausalLM) -> None:
    if self._rotary_caches_initialized:
        return
    device = self.model.embed_tokens.weight.device
    if device.type != "cuda":
        return

    rotary_caches: dict[tuple[int, int, float], torch.Tensor] = {}
    for module in self.modules():
        if not isinstance(module, YOCORotaryEmbedding):
            continue
        cache_key = (
            module.head_size,
            module.max_position_embeddings,
            module.base,
        )
        cache = rotary_caches.get(cache_key)
        if cache is None:
            cache = module._get_cos_sin_cache(device)
            rotary_caches[cache_key] = cache
        else:
            module.cos_sin_cache = cache
    self._rotary_caches_initialized = True


def _initialize_router_weight_caches(self: YOCOForCausalLM) -> None:
    if self._router_weight_caches_initialized:
        return
    for module in self.modules():
        if isinstance(module, YOCOMoE):
            module.initialize_router_weight_cache()
    self._router_weight_caches_initialized = True


def _initialize_shared_expert_weight_caches(self: YOCOForCausalLM) -> None:
    if getattr(self, "_shared_expert_weight_caches_initialized", False):
        return
    for module in self.modules():
        if isinstance(module, YOCOSharedExperts):
            module.initialize_fast_weight_cache()
    self._shared_expert_weight_caches_initialized = True


@dataclass(frozen=True)
class YocoWeightLoadReport:
    """Diagnostics for the most recent complete checkpoint load."""

    loaded: frozenset[str]
    ignored: tuple[str, ...]
    pipeline_missing: tuple[str, ...]
    default_loader_fallbacks: tuple[str, ...]


_SHARED_KV_ALIASES = {
    "model.k_proj.weight": "model.yoco_k_proj.weight",
    "model.v_proj.weight": "model.yoco_v_proj.weight",
}
_SHARED_KV_SHARDS = {"model.yoco_k_proj.weight": 0, "model.yoco_v_proj.weight": 1}
_SELF_PROJECTION_SHARDS = {"q_proj": 0, "k_proj": 1, "v_proj": 2, "lambda_proj": 3}
_CROSS_PROJECTION_SHARDS = {"q_proj": 0, "lambda_proj": 1}
_QKV_SHARDS = {"q_proj": "q", "k_proj": "k", "v_proj": "v"}


def _checkpoint_layer_index(name: str) -> int | None:
    try:
        return int(name.split(".layers.")[-1].split(".")[0])
    except ValueError:
        return None


class _YocoWeightLoader:
    """Per-load mapping state; never registered as a model submodule."""

    def __init__(self, model: YOCOForCausalLM) -> None:
        self.model = model
        self.params = dict(model.named_parameters(remove_duplicate=False))
        self.aliases = {
            name.replace(".experts.routed_experts.", ".experts."): name
            for name in self.params
            if ".experts.routed_experts." in name
        }
        for alias, target in self.aliases.items():
            if alias in self.params and self.params[alias] is not self.params[target]:
                raise ValueError(f"Ambiguous YOCO checkpoint parameter alias: {alias}")
            self.params[alias] = self.params[target]
        config = model.config
        self.first_cross = _cfg_int(config, "num_hidden_layers", "n_layers") - _cfg_int(
            config, "yoco_cross_layers", default=0
        )
        self.intermediate_size = _cfg_int(
            config, "moe_intermediate_size", "moe_ffn_dim"
        )
        self.num_experts = _cfg_int(config, "num_experts", "moe_expert_num")
        self.loaded: set[str] = set()
        self.ignored: list[str] = []
        self.pipeline_missing: list[str] = []
        self.default_loader_fallbacks: list[str] = []

    def projection_target(self, name: str) -> tuple[str, int | str] | None:
        """Match a fused projection only when that target exists at this precision."""
        if name in _SHARED_KV_SHARDS and "model.yoco_kv_proj.weight" in self.params:
            return "model.yoco_kv_proj.weight", _SHARED_KV_SHARDS[name]
        layer_index = _checkpoint_layer_index(name)
        if layer_index is None:
            return None
        if 0 <= layer_index < self.first_cross:
            shards, fused = _SELF_PROJECTION_SHARDS, "qkv_lambda_proj"
        elif layer_index >= self.first_cross:
            shards, fused = _CROSS_PROJECTION_SHARDS, "q_lambda_proj"
        else:
            shards, fused = {}, ""
        for source, shard in shards.items():
            if name.endswith(f".self_attn.{source}.weight"):
                target = name.replace(f"self_attn.{source}", f"self_attn.{fused}")
                if target in self.params:
                    return target, shard
        for source, qkv_shard in _QKV_SHARDS.items():
            source = f"self_attn.{source}"
            if source not in name:
                continue
            if layer_index >= self.first_cross and qkv_shard == "q":
                break
            target = name.replace(source, "self_attn.qkv_proj")
            if target in self.params:
                return target, qkv_shard
        return None

    def parameter(self, name: str) -> torch.nn.Parameter | None:
        if name not in self.params:
            self.ignored.append(name)
            return None
        if is_pp_missing_parameter(name, self.model):
            self.pipeline_missing.append(name)
            return None
        return self.params[name]

    def load_experts(
        self, name: str, weight: torch.Tensor, param: torch.Tensor
    ) -> None:
        """Preserve the HF expert order and the existing per-expert shard loaders."""
        loader = cast(Any, param).weight_loader
        if name.endswith(".w13_weight"):
            shards = weight.view(self.num_experts, 2 * self.intermediate_size, -1)
            for expert_id, expert_weight in enumerate(shards):
                loader(
                    param,
                    expert_weight[: self.intermediate_size],
                    name,
                    "w1",
                    expert_id,
                )
                loader(
                    param,
                    expert_weight[self.intermediate_size :],
                    name,
                    "w3",
                    expert_id,
                )
        else:
            shards = weight.view(self.num_experts, -1, self.intermediate_size)
            for expert_id, expert_weight in enumerate(shards):
                loader(param, expert_weight, name, "w2", expert_id)

    def load_one(self, source_name: str, weight: torch.Tensor) -> None:
        name = _SHARED_KV_ALIASES.get(source_name, source_name)
        projection = self.projection_target(name)
        if projection is not None:
            target, shard = projection
            param = self.parameter(target)
            if param is not None:
                cast(Any, param).weight_loader(param, weight, shard)
                self.loaded.add(target)
            return
        param = self.parameter(name)
        if param is None:
            return
        if name.endswith((".mlp.experts.w13_weight", ".mlp.experts.w2_weight")):
            self.load_experts(name, weight, param)
        elif name.endswith(".mlp.shared_experts.gate_up_proj.weight"):
            half = weight.shape[0] // 2
            loader = cast(Any, param).weight_loader
            loader(param, weight[:half, :], 0)
            loader(param, weight[half:, :], 1)
        else:
            if name.endswith(".mlp.gate.weight") and param.dtype != weight.dtype:
                weight = weight.to(param.dtype)
            loader = getattr(param, "weight_loader", default_weight_loader)
            try:
                loader(param, weight)
            except TypeError:
                # Preserve the existing loader fallback; report it for inspection.
                self.default_loader_fallbacks.append(name)
                default_weight_loader(param, weight)
        self.loaded.add(self.aliases.get(name, name))

    def report(self) -> YocoWeightLoadReport:
        return YocoWeightLoadReport(
            loaded=frozenset(self.loaded),
            ignored=tuple(self.ignored),
            pipeline_missing=tuple(self.pipeline_missing),
            default_loader_fallbacks=tuple(self.default_loader_fallbacks),
        )


def refresh_derived_weight_caches(model: YOCOForCausalLM) -> None:
    """Refresh derived weights in place before any profile or graph capture."""
    model._initialize_rotary_caches()
    model._router_weight_caches_initialized = False
    model._initialize_router_weight_caches()
    model._shared_expert_weight_caches_initialized = False
    model._initialize_shared_expert_weight_caches()


def load_weights(
    model: YOCOForCausalLM, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    loader = _YocoWeightLoader(model)
    for name, weight in weights:
        loader.load_one(name, weight)
    refresh_derived_weight_caches(model)
    model._yoco_weight_load_report = loader.report()
    return set(model._yoco_weight_load_report.loaded)
