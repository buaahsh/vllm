# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import PretrainedConfig

from vllm.model_executor.models.deepseek_v2_utils import should_skip_indexer_topk


def test_index_cache_disabled_never_skips_topk():
    config = PretrainedConfig(
        use_index_cache=False,
        index_topk_freq=3,
        index_skip_topk_offset=2,
    )

    assert not should_skip_indexer_topk(config, 3)


def test_index_cache_uses_frequency_and_offset():
    config = PretrainedConfig(
        use_index_cache=True,
        index_topk_freq=3,
        index_skip_topk_offset=2,
    )

    assert not should_skip_indexer_topk(config, 1)
    assert should_skip_indexer_topk(config, 2)
    assert should_skip_indexer_topk(config, 3)
    assert not should_skip_indexer_topk(config, 4)


def test_index_cache_uses_explicit_pattern():
    config = PretrainedConfig(
        use_index_cache=True,
        index_topk_pattern=["N", "S", "N"],
    )

    assert should_skip_indexer_topk(config, 1)
    assert not should_skip_indexer_topk(config, 2)
    assert not should_skip_indexer_topk(config, 3)
