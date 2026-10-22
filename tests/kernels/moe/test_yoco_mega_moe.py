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
    normalize_yoco_mega_input,
    prepare_yoco_latent_megamoe_inputs,
    use_yoco_mega_moe,
)
from vllm.model_executor.layers.yoco_ops.norm import RMSNorm
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs
from vllm.platforms import current_platform


def _latent_staging_case(rows, magnitude=1.0):
    torch.manual_seed(20261021)
    x = (torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16) * magnitude)[
        :, :1024
    ]
    norm = RMSNorm(1024, dtype=torch.bfloat16).cuda()
    norm.weight.data.copy_(torch.randn_like(norm.weight) * 0.2 + 1)
    ids = torch.randint(0, 128, (rows, 16), device="cuda", dtype=torch.int32)[:, :8]
    weights = torch.rand(rows, 16, device="cuda")[:, :8]
    buffer = SimpleNamespace(
        x=torch.full((rows + 3, 1056), -7.0, device="cuda", dtype=torch.float8_e4m3fn)[
            :, :1024
        ],
        x_sf=torch.full((rows + 3, 16), -17, device="cuda", dtype=torch.int32)[:, :8],
        topk_idx=torch.full((rows + 3, 16), -17, device="cuda", dtype=torch.int64)[
            :, :8
        ],
        topk_weights=torch.full((rows + 3, 16), -17.0, device="cuda")[:, :8],
    )
    return x, norm, ids, weights, buffer


def _check_latent_staging(x, norm, ids, weights, actual):
    expected = SimpleNamespace(
        **{name: value.clone() for name, value in vars(actual).items()}
    )
    n = x.shape[0]
    prepare_megamoe_inputs(
        norm(x),
        weights,
        ids,
        expected.x[:n],
        expected.x_sf[:n],
        expected.topk_idx[:n],
        expected.topk_weights[:n],
    )
    prepare_yoco_latent_megamoe_inputs(x, norm, weights, ids, actual)
    for name, value in vars(actual).items():
        reference = getattr(expected, name)
        if name == "x":
            value, reference = value.view(torch.uint8), reference.view(torch.uint8)
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.has_device_capability(89), reason="requires native FP8"
)
@pytest.mark.parametrize("rows", [0, 1, 9, 17, 18, 36, 72, 513])
@torch.inference_mode()
def test_latent_norm_stages_exact_bytes_without_overwriting_workspace_tail(rows):
    _check_latent_staging(*_latent_staging_case(rows))


@pytest.mark.skipif(
    not current_platform.has_device_capability(89), reason="requires native FP8"
)
@pytest.mark.parametrize("magnitude", [0.0, 1e-8, 1e-4, 100.0])
@torch.inference_mode()
def test_latent_staging_preserves_norm_and_quantization_extremes(magnitude):
    _check_latent_staging(*_latent_staging_case(36, magnitude))


@pytest.mark.skipif(
    not current_platform.has_device_capability(89), reason="requires native FP8"
)
@pytest.mark.parametrize("gamma", [0.109375, 0.875, 1.75, 14.0])
@torch.inference_mode()
def test_latent_staging_power_of_two_scales_and_int64_routes(gamma):
    x, norm, ids, weights, buffer = _latent_staging_case(18)
    x.fill_(1)
    norm.weight.fill_(gamma)
    _check_latent_staging(x, norm, ids.long(), weights, buffer)


@pytest.mark.skipif(
    not current_platform.has_device_capability(89), reason="requires native FP8"
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@torch.inference_mode()
def test_latent_staging_nonfinite_graph_rows_match_existing_path(value):
    x, norm, ids, weights, buffer = _latent_staging_case(18)
    x[3, 17] = value
    _check_latent_staging(x, norm, ids, weights, buffer)


@pytest.mark.skipif(
    not current_platform.has_device_capability(89), reason="requires native FP8"
)
@pytest.mark.parametrize("rows", [0, 9, 2048])
@torch.inference_mode()
def test_deferred_norm_fallback_keeps_bf16_output(rows):
    x, norm, _, _, _ = _latent_staging_case(rows)
    x[:, ::2] = 1
    x[:, 1::2] = 2
    torch.testing.assert_close(
        normalize_yoco_mega_input(x, norm), norm(x), rtol=0, atol=0
    )


@pytest.mark.skipif(
    not current_platform.has_device_capability(89), reason="requires native FP8"
)
@torch.inference_mode()
def test_latent_staging_reduction_rounding_has_bounded_dequantized_error():
    # Integer-derived values exercise BF16 rounding boundaries without
    # depending on device RNG or private checkpoint activations.
    rows = 512
    index = torch.arange(rows * 1024, device="cuda", dtype=torch.int64)
    bits = ((index + 32) * 1103515245 + 12345) & 0x7FFFFFFF
    bits = ((bits ^ (bits >> 16)) * 1664525) & 0x7FFFFFFF
    x = ((bits & 65535).float() / 32768 - 1).bfloat16().view(rows, 1024)
    norm = RMSNorm(1024, dtype=torch.bfloat16).cuda()
    norm.weight.copy_(0.25 + ((bits[:1024] >> 8) & 1023).float() / 2048)
    ids = torch.arange(8, device="cuda", dtype=torch.int32).repeat(rows, 1)
    weights = torch.full((rows, 8), 0.125, device="cuda")
    expected = SimpleNamespace(
        x=torch.empty_like(x, dtype=torch.float8_e4m3fn),
        x_sf=torch.empty(rows, 8, device="cuda", dtype=torch.int32),
        topk_idx=torch.empty_like(ids, dtype=torch.int64),
        topk_weights=torch.empty_like(weights),
    )
    actual = SimpleNamespace(
        **{name: torch.empty_like(tensor) for name, tensor in vars(expected).items()}
    )
    prepare_megamoe_inputs(
        normalize_yoco_mega_input(x, norm),
        weights,
        ids,
        expected.x,
        expected.x_sf,
        expected.topk_idx,
        expected.topk_weights,
    )
    prepare_yoco_latent_megamoe_inputs(x, norm, weights, ids, actual)

    def dequantized(buffer):
        shifts = torch.arange(4, device="cuda") * 8
        exponents = (buffer.x_sf.long().unsqueeze(-1) >> shifts) & 255
        scales = torch.exp2(exponents.flatten(-2).float() - 127)
        return buffer.x.float().reshape(rows, 32, 32) * scales.unsqueeze(-1)

    reference = dequantized(expected)
    value = dequantized(actual)
    relative_rms = (value - reference).square().mean().sqrt() / (
        reference.square().mean().sqrt()
    )
    # FP32 reduction order may cross a BF16 rounding tie and one FP8 bin.
    assert relative_rms < 1e-3
    assert (value != reference).float().mean() < 1e-3
    torch.testing.assert_close(value, reference, rtol=0.13, atol=1e-7)
    torch.testing.assert_close(actual.topk_idx, expected.topk_idx, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.topk_weights, expected.topk_weights, rtol=0, atol=0
    )


@pytest.mark.skipif(
    not current_platform.has_device_capability(89), reason="requires native FP8"
)
@torch.inference_mode()
def test_latent_staging_graph_reads_live_input_norm_and_routes():
    x, norm, ids, weights, buffer = _latent_staging_case(36)
    for _ in range(3):
        prepare_yoco_latent_megamoe_inputs(x, norm, weights, ids, buffer)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepare_yoco_latent_megamoe_inputs(x, norm, weights, ids, buffer)
    for magnitude in [1e-4, 1.0, 16.0]:
        x.copy_(torch.randn_like(x) * magnitude)
        norm.weight.copy_(torch.randn_like(norm.weight))
        ids.copy_(torch.randint_like(ids, 0, 128))
        weights.copy_(torch.rand_like(weights))
        graph.replay()
        # Save graph output before the eager correctness helper writes it again.
        saved = {name: value.clone() for name, value in vars(buffer).items()}
        _check_latent_staging(x, norm, ids, weights, buffer)
        for name, before in saved.items():
            after = getattr(buffer, name)
            if name == "x":
                before, after = before.view(torch.uint8), after.view(torch.uint8)
            torch.testing.assert_close(before, after, rtol=0, atol=0)


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
