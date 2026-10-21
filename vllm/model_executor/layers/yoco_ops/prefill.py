# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy selected self-decoder rows directly into stable cross-block inputs."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gather_rows(
    source,
    output,
    row,
    offsets,
    width: tl.constexpr,
    source_strides: tl.constexpr,
    output_strides: tl.constexpr,
):
    values = tl.load(
        source + row * source_strides[0] + offsets * source_strides[1],
        offsets < width,
        other=0,
    )
    tl.store(
        output + tl.program_id(0) * output_strides[0] + offsets * output_strides[1],
        values,
        offsets < width,
    )


@triton.jit
def _gather_prefill_inputs(
    indices,
    positions,
    hidden,
    key,
    value,
    out_positions,
    out_hidden,
    out_key,
    out_value,
    num_source_rows,
    index_stride: tl.constexpr,
    position_stride: tl.constexpr,
    out_position_stride: tl.constexpr,
    hidden_width: tl.constexpr,
    key_width: tl.constexpr,
    value_width: tl.constexpr,
    hidden_strides: tl.constexpr,
    key_strides: tl.constexpr,
    value_strides: tl.constexpr,
    out_hidden_strides: tl.constexpr,
    out_key_strides: tl.constexpr,
    out_value_strides: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.load(indices + tl.program_id(0) * index_stride)
    row = tl.where(row < 0, row + num_source_rows, row)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    if tl.program_id(1) == 0:
        position = tl.load(positions + row * position_stride)
        tl.store(out_positions + tl.program_id(0) * out_position_stride, position)
    _gather_rows(
        hidden,
        out_hidden,
        row,
        offsets,
        hidden_width,
        hidden_strides,
        out_hidden_strides,
    )
    _gather_rows(key, out_key, row, offsets, key_width, key_strides, out_key_strides)
    _gather_rows(
        value, out_value, row, offsets, value_width, value_strides, out_value_strides
    )


def gather_prefill_inputs(
    indices: torch.Tensor,
    positions: torch.Tensor,
    hidden: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out_positions: torch.Tensor,
    out_hidden: torch.Tensor,
    out_key: torch.Tensor,
    out_value: torch.Tensor,
) -> None:
    """Gather valid rows into the first ``indices.numel()`` destination rows.

    Sources and outputs must not overlap. The caller supplies valid indices
    (including repeated padding rows), paired dtypes, and enough output space.
    Integer views preserve all bits, including FP8 NaNs, without conversion.
    """
    n = indices.numel()
    if n == 0:
        return
    assert indices.ndim == positions.ndim == out_positions.ndim == 1
    assert out_positions.shape[0] >= n
    for source, output in ((hidden, out_hidden), (key, out_key), (value, out_value)):
        assert source.ndim == output.ndim == 2
        assert source.dtype == output.dtype
        assert source.shape[0] == positions.shape[0]
        assert source.shape[1] == output.shape[1] and output.shape[0] >= n
    bit_dtypes = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    hidden, key, value, out_hidden, out_key, out_value = (
        tensor.view(bit_dtypes[tensor.element_size()])
        for tensor in (hidden, key, value, out_hidden, out_key, out_value)
    )
    block = 256
    _gather_prefill_inputs[
        (n, triton.cdiv(max(hidden.shape[1], key.shape[1], value.shape[1]), block))
    ](
        indices,
        positions,
        hidden,
        key,
        value,
        out_positions,
        out_hidden,
        out_key,
        out_value,
        positions.shape[0],
        indices.stride(0),
        positions.stride(0),
        out_positions.stride(0),
        hidden.shape[1],
        key.shape[1],
        value.shape[1],
        hidden.stride(),
        key.stride(),
        value.stride(),
        out_hidden.stride(),
        out_key.stride(),
        out_value.stride(),
        BLOCK=block,
    )
