# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mega scale layout must preserve the original FP8 weight quantization."""

import pytest
import torch

from vllm.model_executor.layers.yoco_ops.mega_moe import expand_block128_scales


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
