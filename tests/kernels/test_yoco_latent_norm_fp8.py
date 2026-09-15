# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Latent norm + quantization against the existing compiled BF16 boundary."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.quantization import resolve_quantization_config
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8_packed_for_deepgemm,
)
from vllm.model_executor.layers.yoco_moe import YOCOLatentOutputTransform
from vllm.model_executor.layers.yoco_ops.norm import RMSNorm
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used
from vllm.utils.torch_utils import set_default_torch_dtype

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability(100) or not is_deep_gemm_e8m0_used(),
    reason="requires B200 DeepGEMM UE8M0",
)


@pytest.fixture(autouse=True)
def enable_experimental_fusion(monkeypatch):
    monkeypatch.setenv("VLLM_YOCO_FP8_LATENT_NORM_FUSION", "1")


def inputs(rows, dtype, magnitude=1.0):
    torch.manual_seed(920)
    x = (torch.randn(rows, 2048, device="cuda", dtype=dtype) * magnitude)[:, :1024]
    norm = RMSNorm(1024, eps=1e-6, dtype=torch.bfloat16).to("cuda")
    norm.weight.data.copy_(torch.randn_like(norm.weight) * 0.2 + 1.0)
    return x, norm


def reference(x, norm):
    return per_token_group_quant_fp8_packed_for_deepgemm(
        norm(x), 128, eps=1e-4, use_ue8m0=True
    )


def equal(actual, expected):
    qa, sa = actual
    qb, sb = expected
    torch.testing.assert_close(
        qa.view(torch.uint8), qb.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(sa, sb, rtol=0, atol=0)
    assert sa.stride() == sb.stride()


@pytest.mark.parametrize("rows", [0, 1, 2, 3, 8, 33, 128, 513, 1025])
@torch.inference_mode()
def test_norm_fp8_matches_existing_quantizer(rows):
    x, norm = inputs(rows, torch.bfloat16)
    actual = torch.ops.vllm.yoco_latent_rms_norm_fp8(x, norm.weight, norm.eps)
    if rows:
        equal(actual, reference(x, norm))
    else:
        assert actual[0].shape == (0, 1024) and actual[1].shape == (0, 2)


@pytest.mark.parametrize("magnitude", [0.0, 1e-8, 1e-4, 100.0])
@torch.inference_mode()
def test_norm_quantization_extremes(magnitude):
    x, norm = inputs(17, torch.bfloat16, magnitude)
    equal(
        torch.ops.vllm.yoco_latent_rms_norm_fp8(x, norm.weight, norm.eps),
        reference(x, norm),
    )


@torch.inference_mode()
def test_norm_fp8_graph_reads_updated_input_and_affine_weight():
    x, norm = inputs(8, torch.bfloat16)
    for _ in range(3):
        torch.ops.vllm.yoco_latent_rms_norm_fp8(x, norm.weight, norm.eps)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = torch.ops.vllm.yoco_latent_rms_norm_fp8(x, norm.weight, norm.eps)
    for magnitude in [1e-4, 1.0, 16.0]:
        x.copy_(torch.randn_like(x) * magnitude)
        norm.weight.copy_(torch.randn_like(norm.weight))
        graph.replay()
        equal(actual, reference(x, norm))


@pytest.fixture
def projection(dist_init):
    runtime = VllmConfig()
    runtime.model_config = SimpleNamespace(
        dtype=torch.bfloat16, hf_text_config=SimpleNamespace(model_type="yoco")
    )
    runtime.compilation_config.custom_ops = ["all"]
    args = deepcopy(resolve_quantization_config("fp8_per_block", None))
    with (
        torch.inference_mode(),
        set_current_vllm_config(runtime),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        layer = ReplicatedLinear(
            1024,
            3072,
            bias=False,
            quant_config=OnlineQuantizationConfig(args),
            prefix="model.layers.0.mlp.fc2_latent_proj",
            return_bias=False,
        )
        layer.weight.weight_loader(layer.weight, torch.randn(3072, 1024) / 32.0)
        yield layer


@pytest.mark.parametrize("rows", [1, 8, 129])
@torch.inference_mode()
def test_fused_projection_bypasses_quantizer(projection, monkeypatch, rows):
    x, norm = inputs(rows, torch.bfloat16)
    transform = YOCOLatentOutputTransform(norm, projection)
    assert transform.fuse_fp8_norm
    expected = projection(norm(x))

    def unexpected(*args, **kwargs):
        raise AssertionError("standalone quantizer called after fused norm")

    monkeypatch.setattr(projection.quant_method.fp8_linear, "quant_fp8", unexpected)
    actual = transform(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_float32_input_keeps_existing_projection_path(projection, monkeypatch):
    x, norm = inputs(8, torch.float32)
    transform = YOCOLatentOutputTransform(norm, projection)
    expected = projection(norm(x))

    def unexpected(*args, **kwargs):
        raise AssertionError("FP32 input entered the BF16-only fused producer")

    monkeypatch.setattr(torch.ops.vllm, "yoco_latent_rms_norm_fp8", unexpected)
    torch.testing.assert_close(transform(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@torch.inference_mode()
def test_nonfinite_rows_match_cuda_quantizer(value):
    x, norm = inputs(5, torch.bfloat16)
    x[3, 17] = value
    equal(
        torch.ops.vllm.yoco_latent_rms_norm_fp8(x, norm.weight, norm.eps),
        reference(x, norm),
    )


@pytest.mark.parametrize(
    "gamma", [0.109375, 0.21875, 0.4375, 0.875, 1.75, 3.5, 7.0, 14.0]
)
@torch.inference_mode()
def test_power_of_two_scale_boundary(gamma):
    x, norm = inputs(3, torch.bfloat16)
    x.fill_(1.0)
    norm.weight.fill_(gamma)
    actual = torch.ops.vllm.yoco_latent_rms_norm_fp8(x, norm.weight, norm.eps)
    expected = reference(x, norm)
    equal(actual, expected)
    # Check scale storage holes, beyond the logical tensor's valid rows.
    span = actual[1].stride(1) + 3
    a = actual[1].as_strided((span,), (1,))
    b = expected[1].as_strided((span,), (1,))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
