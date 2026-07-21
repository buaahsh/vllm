# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch

from vllm.config.attention import AttentionConfig
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder


def _build_metadata(*, force_num_splits_one: bool):
    builder = object.__new__(FlashAttentionMetadataBuilder)
    builder.aot_schedule = False
    builder.aot_sliding_window = (-1, -1)
    builder.use_full_cuda_graph = False
    builder.max_cudagraph_size = None
    builder.max_num_splits = 0
    builder.attention_config = AttentionConfig(
        flash_attn_force_num_splits_one=force_num_splits_one
    )
    builder.cache_config = SimpleNamespace(cache_dtype="auto")
    builder.kv_cache_dtype = torch.bfloat16
    builder.dcp_world_size = 1
    builder.device = torch.device("cpu")

    common_metadata = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=5,
        max_query_len=3,
        max_seq_len=3,
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 5], dtype=torch.int32),
        seq_lens=torch.tensor([2, 3], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([2, 3], dtype=torch.int32),
        block_table_tensor=torch.empty((2, 0), dtype=torch.int32),
        slot_mapping=torch.arange(5, dtype=torch.int64),
        causal=True,
    )
    return builder.build(common_prefix_len=0, common_attn_metadata=common_metadata)


def test_force_flash_attn_num_splits_one() -> None:
    assert _build_metadata(force_num_splits_one=False).max_num_splits == 0
    assert _build_metadata(force_num_splits_one=True).max_num_splits == 1
