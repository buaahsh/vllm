# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any


def should_skip_indexer_topk(config: Any, layer_id: int) -> bool:
    if not getattr(config, "use_index_cache", False):
        return False

    index_topk_freq = getattr(config, "index_topk_freq", 1)
    index_topk_pattern = getattr(config, "index_topk_pattern", None)
    index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)

    if index_topk_pattern is None:
        return max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq != 0
    if 0 <= layer_id < len(index_topk_pattern):
        return index_topk_pattern[layer_id] == "S"
    return False
