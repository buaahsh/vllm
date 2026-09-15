# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO diagnostics regression tests."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.yoco_diagnostics import (
    _maybe_dump_yoco_logical_routes,
    _yoco_logical_moe_layer_id,
)


def test_route_observer_does_not_poll_enable_file_during_forward(monkeypatch, tmp_path):
    from vllm.model_executor.models import yoco_diagnostics as diagnostics

    monkeypatch.setattr(diagnostics, "_YOCO_LOGICAL_ROUTE_DUMP_ROOT", str(tmp_path))
    monkeypatch.setattr(diagnostics, "_YOCO_LOGICAL_ROUTE_DUMP_BATCHES", frozenset({8}))
    (tmp_path / "ENABLED").touch()
    observer = diagnostics.create_yoco_route_dumper()
    assert observer is not None
    written = []
    monkeypatch.setattr(
        diagnostics, "_write_yoco_logical_routes", lambda *args: written.append(args)
    )

    def unexpected_probe(*args):
        raise AssertionError("forward must not probe the filesystem")

    monkeypatch.setattr(
        diagnostics,
        "os",
        SimpleNamespace(path=SimpleNamespace(exists=unexpected_probe)),
    )
    observer(torch.zeros(8, 3), torch.zeros(8, 128), 8, (0, 10, 3), 0)
    assert len(written) == 1
    assert written[0][0] == str(tmp_path)


def test_yoco_logical_moe_layer_ids_cover_all_universal_calls() -> None:
    logical_ids = [
        _yoco_logical_moe_layer_id(layer, loop, 10, 3)
        for loop in range(3)
        for layer in range(10)
    ]
    logical_ids.extend(
        _yoco_logical_moe_layer_id(layer, 0, 10, 3) for layer in range(10, 20)
    )
    assert logical_ids == list(range(40))
    with pytest.raises(ValueError, match="universal loop index"):
        _yoco_logical_moe_layer_id(0, 3, 10, 3)


@pytest.mark.parametrize("execution_mode", ["fast", "align"])
def test_yoco_logical_route_dump_records_eager_topk(
    monkeypatch, tmp_path, execution_mode
) -> None:
    import vllm.model_executor.models.yoco_diagnostics as yoco_module

    monkeypatch.setattr(yoco_module, "_YOCO_LOGICAL_ROUTE_DUMP_ROOT", str(tmp_path))
    monkeypatch.setattr(yoco_module, "_YOCO_LOGICAL_ROUTE_DUMP_BATCHES", frozenset({8}))
    monkeypatch.setattr(yoco_module, "_YOCO_LOGICAL_ROUTE_DUMP_INDEX", 0)
    (tmp_path / "ENABLED").touch()
    topk_ids = torch.arange(64, dtype=torch.int64).view(8, 8)
    monkeypatch.setattr(
        yoco_module,
        "_yoco_align_topk_routing"
        if execution_mode == "align"
        else "_yoco_topk_routing",
        lambda *args: (torch.ones(8, 8), topk_ids),
    )

    _maybe_dump_yoco_logical_routes(
        torch.zeros(8, 3),
        torch.zeros(8, 128),
        8,
        (4, 10, 3),
        2,
        execution_mode=execution_mode,
    )

    paths = list(tmp_path.glob("*.pt"))
    assert len(paths) == 1
    record = torch.load(paths[0], weights_only=True)
    assert record["logical_layer_id"] == 24
    assert record["execution_mode"] == execution_mode
    assert record["num_tokens"] == 8
    assert torch.equal(record["topk_ids"], topk_ids.to(torch.int16))
