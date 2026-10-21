# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mega scale layout must preserve the original FP8 weight quantization."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.yoco import YocoMoEDispatch
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.yoco_ops.mega_moe import (
    expand_block128_scales,
    use_yoco_mega_moe,
)


@pytest.mark.parametrize(
    "runtime,capture,rows,expected",
    [
        (CUDAGraphMode.NONE, CUDAGraphMode.FULL, 9, False),
        (CUDAGraphMode.NONE, CUDAGraphMode.FULL, 17, True),
        (CUDAGraphMode.NONE, CUDAGraphMode.FULL, 36, True),
        (CUDAGraphMode.NONE, CUDAGraphMode.NONE, 36, False),
        (CUDAGraphMode.PIECEWISE, CUDAGraphMode.PIECEWISE, 36, False),
        (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, 36, False),
        (CUDAGraphMode.FULL, CUDAGraphMode.NONE, 36, True),
    ],
)
def test_dispatch_uses_intended_full_graph_during_v2_capture(
    runtime, capture, rows, expected
):
    layer = SimpleNamespace(
        _yoco_moe_dispatch=YocoMoEDispatch("test", 16, 17, False),
        _yoco_moe_decode_only_graphs=True,
        _yoco_moe_capture_backends={},
    )
    with set_forward_context(
        {},
        VllmConfig(),
        num_tokens=rows,
        cudagraph_runtime_mode=runtime,
        cudagraph_capture_mode=capture,
    ):
        assert get_forward_context().cudagraph_capture_mode == capture
        assert use_yoco_mega_moe(layer, rows) is expected
    if capture != CUDAGraphMode.NONE:
        assert layer._yoco_moe_capture_backends == {rows: expected}


@pytest.mark.parametrize("rows", [0, 1, 9, 16, 17, 18, 36, 72])
def test_hybrid_keeps_small_batches_and_prefill_on_original_path(rows):
    dispatch = YocoMoEDispatch("test", 16, 17, False)
    assert dispatch.use_mega(rows, decode_graph=True) == (rows >= 17)
    assert not dispatch.use_mega(rows, decode_graph=False)


def test_hybrid_prefill_requires_explicit_opt_in():
    dispatch = YocoMoEDispatch("test", 16, 17, True)
    assert dispatch.use_mega(18, decode_graph=False)
    assert not dispatch.use_mega(16, decode_graph=False)


@pytest.mark.parametrize("columns", [256, 1024, 1280, 3840])
@pytest.mark.parametrize("packed_input", [False, True])
def test_mega_scale_expansion_preserves_every_dequantized_weight(columns, packed_input):
    rows, experts = 256, 2
    exponents = torch.arange(experts * 2 * (columns // 128)).reshape(experts, 2, -1)
    exponents = (exponents % 20 + 117).to(torch.int32)
    scales = (exponents << 23).view(torch.float32)
    source = scales
    if packed_input:
        expanded = exponents.repeat_interleave(128, 1).to(torch.int64)
        padded = torch.full((experts, rows, ((columns // 128 + 3) // 4) * 4), 255)
        padded[..., : columns // 128] = expanded
        source = sum(
            padded[..., i::4].to(torch.int64) << (8 * i) for i in range(4)
        ).int()
        source = source.transpose(-1, -2).contiguous().transpose(-1, -2)
    before = source.clone()
    actual = expand_block128_scales(source, rows, columns)
    assert torch.equal(source, before)
    assert actual.stride(-2) == 1
    shifts = torch.arange(4) * 8
    decoded = ((actual.to(torch.int64).unsqueeze(-1) >> shifts) & 255).flatten(-2)
    decoded = (decoded.int() << 23).view(torch.float32).repeat_interleave(32, -1)
    expected = scales.repeat_interleave(128, 1).repeat_interleave(128, 2)
    assert torch.equal(decoded, expected)


@pytest.mark.parametrize("value", [0.0, -1.0, 1.5, float("nan"), float("inf")])
def test_mega_refuses_weight_scales_that_require_requantization(value):
    with pytest.raises(ValueError, match="powers of two"):
        expand_block128_scales(torch.full((1, 2, 2), value), 256, 256)


@pytest.mark.parametrize("exponent", [0, 255])
def test_mega_refuses_invalid_packed_scales(exponent):
    packed = torch.full((1, 256, 1), exponent * 0x01010101, dtype=torch.int64).int()
    with pytest.raises(ValueError, match="finite normal"):
        expand_block128_scales(packed, 256, 256)
