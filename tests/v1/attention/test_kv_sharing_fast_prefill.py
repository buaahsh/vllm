# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import pytest
import torch

from vllm.v1.attention.backend import AttentionMetadata, CommonAttentionMetadata
from vllm.v1.attention.backends.utils import create_fast_prefill_custom_backend


@dataclass
class _TestMetadata(AttentionMetadata):
    max_query_len: int
    num_actual_tokens: int


class _TestBuilder:
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> _TestMetadata:
        return _TestMetadata(
            max_query_len=common_attn_metadata.max_query_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
        )


class _TestBackend:
    @staticmethod
    def get_builder_cls():
        return _TestBuilder


def _make_common_metadata(query_len: int) -> CommonAttentionMetadata:
    query_start_loc = torch.tensor([0, query_len], dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=torch.tensor([query_len], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=query_len,
        max_query_len=query_len,
        max_seq_len=query_len,
        block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.arange(query_len, dtype=torch.int64),
        logits_indices_padded=torch.tensor([query_len - 1], dtype=torch.int64),
        num_logits_indices=1,
    )


@pytest.mark.parametrize(
    ("query_len", "min_query_len", "expected_query_len", "uses_fast_prefill"),
    [
        (8, 8, 8, False),
        (9, 8, 1, True),
        (8, 0, 1, True),
    ],
)
def test_fast_prefill_min_query_len(
    query_len: int,
    min_query_len: int,
    expected_query_len: int,
    uses_fast_prefill: bool,
) -> None:
    backend = create_fast_prefill_custom_backend(
        "Test", _TestBackend, min_query_len=min_query_len
    )
    metadata = backend.get_builder_cls()().build(
        common_prefix_len=0,
        common_attn_metadata=_make_common_metadata(query_len),
    )

    assert metadata.max_query_len == expected_query_len
    assert metadata.num_actual_tokens == expected_query_len
    if uses_fast_prefill:
        assert metadata.logits_indices_padded is not None
        assert metadata.num_logits_indices == 1
    else:
        assert metadata.logits_indices_padded is None
        assert metadata.num_logits_indices is None
