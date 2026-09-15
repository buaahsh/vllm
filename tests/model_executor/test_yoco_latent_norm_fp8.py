# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
)
from vllm.model_executor.layers.yoco_moe import YOCOLatentOutputTransform
from vllm.model_executor.layers.yoco_ops.norm import RMSNorm


@pytest.mark.parametrize(
    "mode,quantized,width,enabled,expected",
    [
        ("fast", True, 1024, True, True),
        ("align", True, 1024, True, False),
        ("fast", False, 1024, True, False),
        ("fast", True, 3072, True, False),
        ("fast", True, 1024, False, False),
    ],
)
def test_norm_fusion_respects_precision_and_mode(
    monkeypatch, mode, quantized, width, enabled, expected
):
    monkeypatch.setenv("VLLM_YOCO_FP8_LATENT_NORM_FUSION", str(int(enabled)))
    method: Any = (
        object.__new__(Fp8PerBlockOnlineLinearMethod) if quantized else object()
    )
    if quantized:
        monkeypatch.setattr(method, "supports_silu_mul_fusion", lambda: True)
        method.fp8_linear = SimpleNamespace(
            quant_fp8=SimpleNamespace(_enforce_enable=True, enabled=lambda: True)
        )
    proj = object.__new__(ReplicatedLinear)
    nn.Module.__init__(proj)
    proj.bias = None
    proj.quant_method = method
    norm = RMSNorm(width, dtype=torch.bfloat16, execution_mode=mode)
    with set_current_vllm_config(VllmConfig()):
        assert YOCOLatentOutputTransform(norm, proj).fuse_fp8_norm == expected


def test_identity_norm_and_bias_remain_unfused(monkeypatch):
    monkeypatch.setenv("VLLM_YOCO_FP8_LATENT_NORM_FUSION", "1")
    proj = object.__new__(ReplicatedLinear)
    nn.Module.__init__(proj)
    proj.bias = nn.Parameter(torch.zeros(3072))
    proj.quant_method = object.__new__(Fp8PerBlockOnlineLinearMethod)
    assert not YOCOLatentOutputTransform(nn.Identity(), proj).fuse_fp8_norm
    assert not YOCOLatentOutputTransform(
        RMSNorm(1024, dtype=torch.bfloat16), proj
    ).fuse_fp8_norm


def test_latent_norm_fusion_is_opt_in(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_YOCO_FP8_LATENT_NORM_FUSION", raising=False)
    assert not envs.VLLM_YOCO_FP8_LATENT_NORM_FUSION
