# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared producer/consumer boundaries for YOCO's final prompt block."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.request import Request


def yoco_remote_prefill_token_count(num_prompt_tokens: int, block_size: int) -> int:
    return max(0, (num_prompt_tokens - 1) // block_size * block_size)


def truncate_yoco_remote_prefill(request: "Request", block_size: int) -> None:
    """Retain P's prefix once, before prefix lookup or SWA history eviction."""
    params = request.kv_transfer_params
    if params is None:
        return
    if params.get("_p_side_truncated"):
        request.skip_reading_prefix_cache = True
        return
    prefix = yoco_remote_prefill_token_count(request.num_prompt_tokens, block_size)
    if prefix == 0 or prefix == request.num_prompt_tokens:
        return
    remove = request.num_prompt_tokens - prefix
    if request.prompt_token_ids is not None:
        del request.prompt_token_ids[-remove:]
    elif request.prompt_embeds is not None:
        request.prompt_embeds = request.prompt_embeds[:-remove]
    else:
        return
    del request._all_token_ids[-remove:]
    request.num_prompt_tokens = prefix
    request.max_tokens = 1
    params["_p_side_truncated"] = True
    request.skip_reading_prefix_cache = True
