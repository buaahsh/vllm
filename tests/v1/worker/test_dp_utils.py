# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.worker.dp_utils import (
    _post_process_fast_prefill,
    _post_process_yoco_kv_only_prefill,
)


def test_fast_prefill_preserves_active_rank_sizes() -> None:
    coordination = torch.zeros(4, 2, dtype=torch.int32)
    coordination[2] = torch.tensor([(2 << 3) | 2, (3 << 3) | 2])

    result = _post_process_fast_prefill(
        coordination,
        torch.tensor([128, 256], dtype=torch.int32),
    )

    torch.testing.assert_close(result, torch.tensor([2, 3], dtype=torch.int32))


def test_fast_prefill_uses_main_size_for_inactive_rank() -> None:
    coordination = torch.zeros(4, 2, dtype=torch.int32)
    coordination[2] = torch.tensor([(1 << 3) | 2, 0])

    result = _post_process_fast_prefill(
        coordination,
        torch.tensor([202, 1], dtype=torch.int32),
    )

    torch.testing.assert_close(result, torch.tensor([1, 1], dtype=torch.int32))


def test_fast_prefill_returns_none_when_all_ranks_inactive() -> None:
    coordination = torch.zeros(4, 2, dtype=torch.int32)

    result = _post_process_fast_prefill(
        coordination,
        torch.tensor([2, 3], dtype=torch.int32),
    )

    assert result is None


def test_yoco_kv_only_prefill_requires_all_dp_ranks() -> None:
    coordination = torch.zeros(4, 2, dtype=torch.int32)
    coordination[2] = torch.tensor([4, 4])
    assert _post_process_yoco_kv_only_prefill(coordination)

    coordination[2, 1] = 0
    assert not _post_process_yoco_kv_only_prefill(coordination)
