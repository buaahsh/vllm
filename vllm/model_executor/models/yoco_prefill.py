# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from vllm.model_executor.layers.yoco_attention import YOCOCrossAttention
    from vllm.model_executor.models.yoco import YOCODecoderLayer, YOCOForCausalLM

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import DPMetadata, get_forward_context
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.utils import KVSharingFastPrefillMetadata


def _fast_prefill_forward(
    self: YOCOForCausalLM,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    kv_only_prefill: bool = False,
) -> torch.Tensor:
    """Forward with fast prefill: cross-attention layers that share KV
    only process decode tokens during prefill.

    The self portion and the cross portion run as two separately-compiled
    ``@support_torch_compile`` units (``self_block`` / ``cross_block``).
    The first eager profile run compiles the ordinary full model before
    CUDA graph capture, then piecewise profiling compiles both split
    blocks. Uniform FULL decode reuses the ordinary model path; prefill
    uses the split blocks."""
    model = self.model

    # No dedicated fast-prefill blocks (e.g. a single cross layer): fall
    # back to the standard dense forward.
    if model.self_block is None or model.cross_block is None:
        return model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    # Decode-token indices. Piecewise profile/warmup runs without fast
    # metadata fall back to all tokens so both split blocks are compiled.
    (
        logits_indices_padded,
        num_logits_indices,
        fast_prefill_num_tokens_across_dp_cpu,
    ) = self._get_fast_prefill_indices()
    fwd_ctx = get_forward_context()
    global_fast_prefill = kv_only_prefill or logits_indices_padded is not None
    if (
        not global_fast_prefill
        and fwd_ctx.dp_metadata is not None
        and fast_prefill_num_tokens_across_dp_cpu is not None
    ):
        global_fast_prefill = not torch.equal(
            fast_prefill_num_tokens_across_dp_cpu,
            fwd_ctx.dp_metadata.num_tokens_across_dp_cpu,
        )

    if (
        not global_fast_prefill
        and fwd_ctx.cudagraph_runtime_mode == CUDAGraphMode.NONE
        and not model.full_model_warmed
    ):
        model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )
        model.full_model_warmed = True

    if not global_fast_prefill and fwd_ctx.cudagraph_runtime_mode == CUDAGraphMode.FULL:
        assert model.full_model_warmed, (
            "YOCO full model must be compiled during eager profiling "
            "before CUDA graph capture."
        )
        return model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )

    # Self portion on ALL tokens (separate piecewise CUDA graph). It also
    # produces and writes the full shared KV cache without executing any
    # cross layer.
    hidden_states, yoco_key, yoco_value, kv_cache_dummy_dep = model.self_block(
        input_ids,
        positions,
        inputs_embeds,
    )

    # A disaggregated P request only needs the shared K/V produced above.
    # Its sampled token is ignored by the proxy, so use the self-decoder
    # state as a disposable logits input and avoid all ten cross layers.
    if kv_only_prefill:
        return model.norm(hidden_states)

    if logits_indices_padded is None:
        logits_indices_padded = torch.arange(
            positions.size(0),
            dtype=torch.int64,
            device=positions.device,
        )

    # Clone the self-decoder output before it is potentially freed by the
    # piecewise cudagraph machinery when multiple compile units are used.
    out_hidden = hidden_states.clone()

    # Feed the cross block through static buffers — vLLM runs with
    # cudagraph_copy_inputs=False, so inputs need stable addresses.
    n = logits_indices_padded.size(0)
    model.fp_positions[:n].copy_(positions[logits_indices_padded])
    model.fp_hidden_states[:n].copy_(hidden_states[logits_indices_padded])
    if model.fuse_fp8_shared_kv:
        model.fp_yoco_key[:n].view(torch.uint8).copy_(
            yoco_key.view(torch.uint8)[logits_indices_padded]
        )
        model.fp_yoco_value[:n].view(torch.uint8).copy_(
            yoco_value.view(torch.uint8)[logits_indices_padded]
        )
    else:
        model.fp_yoco_key[:n].copy_(yoco_key[logits_indices_padded])
        model.fp_yoco_value[:n].copy_(yoco_value[logits_indices_padded])

    original_dp_metadata = fwd_ctx.dp_metadata
    if (
        original_dp_metadata is not None
        and fast_prefill_num_tokens_across_dp_cpu is not None
    ):
        fwd_ctx.dp_metadata = DPMetadata(fast_prefill_num_tokens_across_dp_cpu)
    try:
        decode_hidden = model.cross_block(
            model.fp_positions[:n],
            model.fp_hidden_states[:n],
            model.fp_yoco_key[:n],
            model.fp_yoco_value[:n],
            kv_cache_dummy_dep,
        )
    finally:
        fwd_ctx.dp_metadata = original_dp_metadata

    # Merge cross-decoder outputs back into the full hidden states.
    if num_logits_indices is not None:
        assert num_logits_indices > 0
        real_indices = logits_indices_padded[:num_logits_indices]
        out_hidden[real_indices] = decode_hidden[:num_logits_indices]
    else:
        out_hidden[logits_indices_padded] = decode_hidden

    return model.norm(out_hidden)


def _get_fast_prefill_indices(
    self: YOCOForCausalLM,
) -> tuple[torch.Tensor | None, int | None, torch.Tensor | None]:
    """Retrieve logits_indices from forward context attention metadata."""
    fwd_ctx = get_forward_context()
    attn_metadata = fwd_ctx.attn_metadata
    fast_prefill_num_tokens_across_dp_cpu = (
        fwd_ctx.fast_prefill_num_tokens_across_dp_cpu
    )
    if attn_metadata is None:
        return None, None, fast_prefill_num_tokens_across_dp_cpu
    if not isinstance(attn_metadata, dict):
        return None, None, fast_prefill_num_tokens_across_dp_cpu
    # Find a KV-sharing layer's metadata to get logits_indices.
    # Use the last layer's attention (which is a fast prefill layer).
    last_layer = cast("YOCODecoderLayer", self.model.layers[-1])
    cross_attention = cast("YOCOCrossAttention", last_layer.self_attn)
    layer_name = cross_attention.attn.layer_name
    layer_meta = attn_metadata.get(layer_name)
    if layer_meta is None:
        return None, None, fast_prefill_num_tokens_across_dp_cpu
    if isinstance(layer_meta, KVSharingFastPrefillMetadata):
        return (
            layer_meta.logits_indices_padded,
            layer_meta.num_logits_indices,
            fast_prefill_num_tokens_across_dp_cpu,
        )
    return None, None, fast_prefill_num_tokens_across_dp_cpu
