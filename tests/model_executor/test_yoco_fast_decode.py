# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.fused_moe.experts import yoco_flashinfer_decode
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
)


@pytest.mark.parametrize("rows", [1, 32, 63, 64, 65, 128, 129, 256, 257, 1024])
def test_yoco_decode_cutlass_rows_and_gates(rows, monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts import flashinfer_cutlass_moe

    descriptor = SimpleNamespace(uniform=True, num_tokens=rows, num_reqs=rows)
    context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.FULL, batch_descriptor=descriptor
    )
    monkeypatch.setattr(flashinfer_cutlass_moe, "get_forward_context", lambda: context)
    experts = SimpleNamespace(yoco_fast_decode_cutlass=True, tp_size=1, ep_size=1)
    x = torch.empty(rows, 1024, device="meta", dtype=torch.bfloat16)
    w1 = torch.empty(128, 7680, 1024, device="meta", dtype=torch.bfloat16)
    w2 = torch.empty(128, 1024, 3840, device="meta", dtype=torch.bfloat16)
    check = FlashInferExperts._use_yoco_decode_cutlass
    assert check(experts, x, w1, w2) == (rows in (64, 128, 256))
    assert not check(experts, x.half(), w1, w2)
    assert not check(experts, x, w1[:64], w2[:64])
    for mode in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE):
        context.cudagraph_runtime_mode = mode
        assert not check(experts, x, w1, w2)
    context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    descriptor.uniform = False
    assert not check(experts, x, w1, w2)
    descriptor.uniform = True
    descriptor.num_tokens = 2 * rows
    assert not check(experts, x, w1, w2)
    descriptor.num_tokens = rows
    context.batch_descriptor = None
    assert not check(experts, x, w1, w2)
    context.batch_descriptor = descriptor
    experts.tp_size = 2
    assert not check(experts, x, w1, w2)
    experts.tp_size = 1
    experts.yoco_fast_decode_cutlass = False
    assert not check(experts, x, w1, w2)


@pytest.mark.parametrize("accepted", [False, True])
def test_yoco_decode_cache_requires_environment_acceptance(monkeypatch, accepted):
    paths = []

    def load(path):
        paths.append(path)
        return accepted

    tuner = SimpleNamespace(load_configs=load)
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.autotuner",
        SimpleNamespace(AutoTuner=SimpleNamespace(get=lambda: tuner)),
    )
    yoco_flashinfer_decode.load_yoco_decode_cutlass_cache.cache_clear()
    try:
        assert yoco_flashinfer_decode.load_yoco_decode_cutlass_cache() is accepted
        assert len(paths) == 1
        data = json.loads(Path(paths[0]).read_text())
        shapes = [ast.literal_eval(k)[2][0][0] for k in data if k != "_metadata"]
        assert sorted(shapes) == [64, 64, 128, 128, 256, 256]
        assert data["_metadata"]["gpu"] == "NVIDIA B200"
    finally:
        yoco_flashinfer_decode.load_yoco_decode_cutlass_cache.cache_clear()
