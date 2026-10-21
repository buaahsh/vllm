# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.dp_utils import _post_process_fast_prefill


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_v2_fast_prefill_dp_sizes_include_inactive_and_dummy_ranks(
    monkeypatch, active, rank
):
    import vllm.v1.worker.gpu.dp_utils as dp

    sizes = [2, 128, 1] if active else [64, 128, 1]
    combined = torch.tensor([sizes, [int(active), 0, 0]], dtype=torch.int32)
    monkeypatch.setattr(dp, "should_skip_dp_coordination", lambda: False)
    monkeypatch.setattr(dp, "get_dp_group", lambda: SimpleNamespace(cpu_group=None))

    def reduce(tensor, group):
        assert tensor[0, rank] == sizes[rank]
        assert tensor[1, rank] == (active and rank == 0)
        tensor.copy_(combined)

    monkeypatch.setattr(dp.dist, "all_reduce", reduce)
    result = dp.sync_fast_prefill_tokens(
        [64, 128, 1][rank], 2 if active and rank == 0 else None, 3, rank
    )
    if active:
        torch.testing.assert_close(result, torch.tensor(sizes, dtype=torch.int32))
    else:
        assert result is None


def test_fast_prefill_preserves_active_rank_sizes() -> None:
    coordination = torch.zeros(4, 2, dtype=torch.int32)
    coordination[2] = torch.tensor([(2 << 2) | 2, (3 << 2) | 2])

    result = _post_process_fast_prefill(
        coordination,
        torch.tensor([128, 256], dtype=torch.int32),
    )

    torch.testing.assert_close(result, torch.tensor([2, 3], dtype=torch.int32))


def test_fast_prefill_uses_main_size_for_inactive_rank() -> None:
    coordination = torch.zeros(4, 2, dtype=torch.int32)
    coordination[2] = torch.tensor([(1 << 2) | 2, 0])

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
