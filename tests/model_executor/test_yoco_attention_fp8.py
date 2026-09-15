# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention import attention as attention_module
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
)
from vllm.model_executor.models import yoco


def layer():
    return SimpleNamespace(
        calculate_kv_scales=False,
        kv_cache_dtype="fp8",
        impl=SimpleNamespace(
            supports_quant_query_input=True, vllm_flash_attn_version=4
        ),
        query_quant=lambda *a: (_ for _ in ()).throw(AssertionError("double quant")),
        num_heads=8,
        head_size=128,
        head_size_v=128,
        num_kv_heads=2,
        use_direct_call=True,
        kv_sharing_target_layer_name="owner",
        attn_backend=SimpleNamespace(forward_includes_kv_cache_update=False),
        layer_name="reader",
        _q_scale=torch.ones(1),
        _k_scale=torch.ones(1),
        _v_scale=torch.ones(1),
    )


def test_prequantized_query_bypasses_quantizer_with_bf16_output(monkeypatch):
    calls = []

    def forward(q, k, v, out, name, **kwargs):
        calls.append((q.dtype, out.dtype))
        out.fill_(1.0)

    monkeypatch.setattr(attention_module, "unified_attention_with_output", forward)
    q = torch.zeros(3, 8 * 128, dtype=torch.float8_e4m3fn)
    kv = torch.zeros(3, 2 * 128)
    out = attention_module.Attention.forward(
        layer(), q, kv, kv, output_dtype=torch.bfloat16
    )
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert calls == [(torch.float8_e4m3fn, torch.bfloat16)]
    assert torch.all(out == 1.0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("calculate_kv_scales", True),
        ("kv_cache_dtype", "auto"),
        ("query_quant", None),
        ("supports_quant_query_input", False),
        ("output_dtype", None),
    ],
)
def test_prequantized_query_rejects_wrong_contract(field, value):
    attn = layer()
    output_dtype = torch.bfloat16
    if field == "supports_quant_query_input":
        attn.impl.supports_quant_query_input = value
    elif field == "output_dtype":
        output_dtype = value
    else:
        setattr(attn, field, value)
    with pytest.raises(ValueError, match="Prequantized E4M3"):
        attention_module.Attention.forward(
            attn,
            torch.empty(1, 1024, dtype=torch.float8_e4m3fn),
            None,
            None,
            output_dtype=output_dtype,
        )


@pytest.mark.parametrize(
    "mode,tp,quantized,enabled,expected",
    [
        ("fast", 1, True, True, True),
        ("align", 1, True, True, False),
        ("fast", 2, True, True, False),
        ("fast", 1, False, True, False),
        ("fast", 1, True, False, False),
    ],
)
def test_output_fusion_respects_mode_tp_ignores_and_toggle(
    monkeypatch, mode, tp, quantized, enabled, expected
):
    monkeypatch.setenv("VLLM_YOCO_FP8_ATTENTION_FUSION", str(int(enabled)))
    method = object.__new__(Fp8PerBlockOnlineLinearMethod) if quantized else object()
    if quantized:
        monkeypatch.setattr(method, "supports_silu_mul_fusion", lambda: True)
    projection = SimpleNamespace(tp_size=tp, bias=None, quant_method=method)
    assert yoco._yoco_can_fuse_fp8_output(projection, mode) == expected


@pytest.mark.parametrize(
    "mode,cache,calibrate,version,expected",
    [
        ("fast", "fp8", False, 4, True),
        ("align", "fp8", False, 4, False),
        ("fast", "auto", False, 4, False),
        ("fast", "fp8", True, 4, False),
        ("fast", "fp8", False, 3, False),
    ],
)
def test_input_fusion_rejects_incompatible_attention(
    monkeypatch, mode, cache, calibrate, version, expected
):
    monkeypatch.setenv("VLLM_YOCO_FP8_ATTENTION_FUSION", "1")
    monkeypatch.setattr(yoco.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(yoco.current_platform, "is_device_capability", lambda _: True)
    monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.bfloat16)
    attn = layer()
    attn.kv_cache_dtype = cache
    attn.calculate_kv_scales = calibrate
    attn.impl.vllm_flash_attn_version = version
    assert yoco._yoco_can_fuse_fp8_attention(attn, mode) == expected
