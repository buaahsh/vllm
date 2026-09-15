# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup must discover the online block-FP8 kernels used in serving."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
    DeepGemmFp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
)
from vllm.model_executor.warmup import deep_gemm_warmup as warmup


def make_layer(kind="online", shape=(256, 128), backend=True):
    layer = LinearBase(128, 256, disable_tp=True)
    if kind == "online":
        method = object.__new__(Fp8PerBlockOnlineLinearMethod)
        method.weight_block_size = [128, 128]
        method.fp8_linear = (
            object.__new__(DeepGemmFp8BlockScaledMMKernel) if backend else object()
        )
    else:
        method = object.__new__(Fp8LinearMethod)
        method.block_quant = True
        method.use_marlin = False
        method.quant_config = SimpleNamespace(weight_block_size=[128, 128])
    layer.quant_method = method
    layer.weight = torch.empty(shape)
    layer.weight_scale_inv = torch.ones(2, 1)
    return layer


@pytest.fixture(autouse=True)
def deep_gemm_alignment(monkeypatch):
    monkeypatch.setattr(
        warmup, "get_mk_alignment_for_contiguous_layout", lambda: [128, 128]
    )
    monkeypatch.setattr(warmup, "FP8_GEMM_NT_WARMUP_CACHE", set())


@pytest.mark.parametrize("kind", ["online", "legacy"])
def test_discovers_block_fp8_and_extracts_loaded_weights(kind):
    layer = make_layer(kind)
    assert warmup._fp8_linear_may_use_deep_gemm(layer)
    weight, scales, block = warmup._extract_data_from_linear_base_module(layer)
    assert weight is layer.weight
    assert scales is layer.weight_scale_inv
    assert block == [128, 128]


@pytest.mark.parametrize("shape", [(255, 128), (256, 127), (2, 256, 128)])
def test_unsupported_shapes_are_not_warmed(shape):
    assert not warmup._fp8_linear_may_use_deep_gemm(make_layer(shape=shape))


def test_online_other_backend_is_not_warmed():
    assert not warmup._fp8_linear_may_use_deep_gemm(make_layer(backend=False))


def test_legacy_marlin_is_not_warmed():
    layer = make_layer("legacy")
    layer.quant_method.use_marlin = True
    assert not warmup._fp8_linear_may_use_deep_gemm(layer)


def test_unquantized_linear_is_not_warmed():
    assert not warmup._fp8_linear_may_use_deep_gemm(
        LinearBase(128, 256, disable_tp=True)
    )


def test_online_layer_reaches_startup_warmup(monkeypatch):
    layer = make_layer()
    model = torch.nn.Sequential(layer)
    calls = []
    monkeypatch.setattr(warmup, "_get_fp8_gemm_nt_m_values", lambda w, m: [1, 2, 4])
    monkeypatch.setattr(
        warmup,
        "_deepgemm_fp8_gemm_nt_warmup",
        lambda **kwargs: calls.append(kwargs),
    )
    assert warmup._count_warmup_iterations(model, 4) == 3
    warmup.deepgemm_fp8_gemm_nt_warmup(model, 4)
    assert len(calls) == 1
    assert calls[0]["w"] is layer.weight
    assert calls[0]["ws"] is layer.weight_scale_inv
    assert calls[0]["max_tokens"] == 4
