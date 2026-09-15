# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.yoco_ops.routing import (
    _yoco_align_topk_routing as _yoco_align_topk_routing,
)
from vllm.model_executor.layers.yoco_ops.routing import (
    _yoco_topk_routing as _yoco_topk_routing,
)

_YOCO_LOGICAL_ROUTE_DUMP_ROOT = os.getenv("VLLM_YOCO_LOGICAL_ROUTE_DUMP")


_YOCO_LOGICAL_ROUTE_DUMP_BATCHES = frozenset(
    int(value)
    for value in os.getenv("VLLM_YOCO_LOGICAL_ROUTE_DUMP_BATCHES", "").split(",")
    if value
)


_YOCO_LOGICAL_ROUTE_DUMP_INDEX = 0


def _yoco_logical_moe_layer_id(
    layer_idx: int,
    loop_idx: int,
    first_cross_layer_idx: int,
    universal_loop: int,
) -> int:
    if layer_idx < first_cross_layer_idx:
        if not 0 <= loop_idx < universal_loop:
            raise ValueError(f"invalid YOCO universal loop index {loop_idx}")
        return loop_idx * first_cross_layer_idx + layer_idx
    return first_cross_layer_idx * universal_loop + layer_idx - first_cross_layer_idx


@dataclass(frozen=True)
class YocoRouteDumper:
    root: str
    batches: frozenset[int]

    def __call__(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        logical_route_info: tuple[int, int, int] | None,
        loop_idx: int,
        execution_mode: str = "fast",
    ) -> None:
        if logical_route_info is None or hidden_states.shape[0] not in self.batches:
            return
        _write_yoco_logical_routes(
            self.root,
            hidden_states,
            router_logits,
            top_k,
            logical_route_info,
            loop_idx,
            execution_mode,
        )


def create_yoco_route_dumper() -> YocoRouteDumper | None:
    """Select the eager-only diagnostic before model execution begins."""
    root = _YOCO_LOGICAL_ROUTE_DUMP_ROOT
    if root is None or not os.path.exists(os.path.join(root, "ENABLED")):
        return None
    return YocoRouteDumper(root, _YOCO_LOGICAL_ROUTE_DUMP_BATCHES)


def _maybe_dump_yoco_logical_routes(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    logical_route_info: tuple[int, int, int] | None,
    loop_idx: int,
    execution_mode: str = "fast",
) -> None:
    """Compatibility helper for standalone diagnostic scripts."""
    dumper = create_yoco_route_dumper()
    if dumper is not None:
        dumper(
            hidden_states,
            router_logits,
            top_k,
            logical_route_info,
            loop_idx,
            execution_mode,
        )


def _write_yoco_logical_routes(
    root: str,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    logical_route_info: tuple[int, int, int],
    loop_idx: int,
    execution_mode: str,
) -> None:
    num_tokens = hidden_states.shape[0]
    if torch.compiler.is_compiling():
        raise RuntimeError("YOCO logical routing dump requires eager execution")
    layer_idx, first_cross_layer_idx, universal_loop = logical_route_info
    logical_layer_id = _yoco_logical_moe_layer_id(
        layer_idx,
        loop_idx,
        first_cross_layer_idx,
        universal_loop,
    )
    routing = (
        _yoco_align_topk_routing if execution_mode == "align" else _yoco_topk_routing
    )
    _, topk_ids = routing(
        hidden_states,
        router_logits,
        top_k,
        True,
    )
    global _YOCO_LOGICAL_ROUTE_DUMP_INDEX
    index = _YOCO_LOGICAL_ROUTE_DUMP_INDEX
    _YOCO_LOGICAL_ROUTE_DUMP_INDEX += 1
    os.makedirs(root, exist_ok=True)
    torch.save(
        {
            "index": index,
            "num_tokens": num_tokens,
            "logical_layer_id": logical_layer_id,
            "execution_mode": execution_mode,
            "topk_ids": topk_ids.to(torch.int16).cpu(),
        },
        os.path.join(
            root,
            f"{index:08d}-m{num_tokens}-l{logical_layer_id:02d}.pt",
        ),
    )
