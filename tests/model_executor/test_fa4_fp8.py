# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

try:
    from vllm.vllm_flash_attn import fa4_compat, flash_attn_interface
except ImportError:
    pytest.skip("requires CUDA FlashAttention extensions", allow_module_level=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_fa4_forwards_descales_only_for_fp8(monkeypatch, dtype):
    calls = []

    def forward(q, k, v, **kwargs):
        calls.append(kwargs)
        return torch.zeros_like(q, dtype=torch.bfloat16), None

    monkeypatch.setattr(fa4_compat, "get_fa4_fwd", lambda: forward)
    monkeypatch.setattr(fa4_compat, "fa4_supports_fp8", lambda: True)
    q = torch.zeros(1, 8, 128, dtype=dtype)
    kv = torch.zeros(3, 2, 128, dtype=dtype)
    scales = [torch.tensor([[0.25, 0.5]]) * v for v in (1, 2, 3)]
    flash_attn_interface.flash_attn_varlen_func(
        q,
        kv,
        kv,
        1,
        torch.tensor([0, 1], dtype=torch.int32),
        3,
        cu_seqlens_k=torch.tensor([0, 3], dtype=torch.int32),
        fa_version=4,
        q_descale=scales[0],
        k_descale=scales[1],
        v_descale=scales[2],
    )
    for name, scale in zip(("q_descale", "k_descale", "v_descale"), scales):
        if dtype == torch.float8_e4m3fn:
            assert calls[0][name] is scale
        else:
            assert name not in calls[0]


@pytest.mark.parametrize(
    "version,family,installed,expected",
    [
        (3, 90, False, True),
        (3, 100, True, False),
        (4, 90, True, False),
        (4, 100, True, True),
        (4, 100, False, False),
        (2, 100, True, False),
    ],
)
def test_fp8_capability_tracks_selected_version(
    monkeypatch, version, family, installed, expected
):
    from vllm.v1.attention.backends import fa_utils

    monkeypatch.setattr(fa_utils.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(
        fa_utils.current_platform, "is_device_capability_family", lambda v: v == family
    )
    monkeypatch.setattr(fa4_compat, "fa4_supports_fp8", lambda: installed)
    assert fa_utils.flash_attn_supports_fp8(version) == expected


@pytest.mark.parametrize(
    "dtype,cache_dtype,head_size,reason",
    [
        (torch.bfloat16, "fp8", 128, None),
        (torch.float16, "auto", 128, None),
        (torch.float16, "fp8", 128, "bfloat16 attention output"),
        (torch.bfloat16, "fp8", 40, "multiple of 16"),
    ],
)
def test_fa4_fp8_backend_checks_output_dtype_and_alignment(
    monkeypatch, dtype, cache_dtype, head_size, reason
):
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends import flash_attn

    monkeypatch.setattr(flash_attn, "get_flash_attn_version", lambda **_: 4)
    actual = flash_attn.FlashAttentionBackend.supports_combination(
        head_size,
        dtype,
        cache_dtype,
        16,
        False,
        False,
        False,
        DeviceCapability(10, 0),
    )
    if reason is None:
        assert actual is None
    else:
        assert reason in actual


def test_shared_fp8_cache_uses_writer_scales(monkeypatch):
    from vllm.v1.attention.backends import flash_attn

    owner = SimpleNamespace(
        kv_sharing_target_layer_name=None,
        kv_cache_torch_dtype=torch.uint8,
        _k_scale=torch.tensor(0.125),
        _v_scale=torch.tensor(0.25),
    )
    intermediate = SimpleNamespace(kv_sharing_target_layer_name="writer")
    reader = SimpleNamespace(
        kv_cache_torch_dtype=torch.uint8,
        _k_scale=torch.tensor(4.0),
        _v_scale=torch.tensor(8.0),
    )
    impl = flash_attn.FlashAttentionImpl.__new__(flash_attn.FlashAttentionImpl)
    impl.kv_sharing_target_layer_name = "intermediate"
    monkeypatch.setattr(
        flash_attn,
        "get_forward_context",
        lambda: SimpleNamespace(
            no_compile_layers={"writer": owner, "intermediate": intermediate}
        ),
    )
    assert impl._get_kv_scale_source(reader) is owner
    owner._k_scale.fill_(0.0625)
    assert impl._get_kv_scale_source(reader)._k_scale.item() == 0.0625
    owner.kv_cache_torch_dtype = torch.bfloat16
    with pytest.raises(ValueError, match="same dtype"):
        impl._get_kv_scale_source(reader)


@pytest.mark.parametrize("cache_dtype", ["auto", "fp8"])
def test_yoco_fp8_decode_keeps_flash_attention(monkeypatch, cache_dtype):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.attention.backends import flash_attn

    runtime = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="yoco")),
        additional_config={"yoco_execution_mode": "fast"},
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, dcp_comm_backend="a2a"
        ),
    )
    monkeypatch.setattr(flash_attn, "get_current_vllm_config_or_none", lambda: runtime)
    monkeypatch.setattr(flash_attn, "get_flash_attn_version", lambda **_: 4)
    monkeypatch.setattr(flash_attn, "flash_attn_supports_fp8", lambda *_: True)
    monkeypatch.setattr(
        flash_attn.current_platform,
        "get_device_capability",
        lambda: SimpleNamespace(major=10),
    )
    impl = flash_attn.FlashAttentionImpl(
        num_heads=64,
        head_size=128,
        scale=128**-0.5,
        num_kv_heads=8,
        alibi_slopes=None,
        sliding_window=513,
        kv_cache_dtype=cache_dtype,
    )
    assert impl.use_triton_yoco_decode == (cache_dtype == "auto")
    runtime.additional_config["yoco_execution_mode"] = "align"
    if cache_dtype == "fp8":
        with pytest.raises(ValueError, match="requires Fast"):
            flash_attn.FlashAttentionImpl(
                num_heads=64,
                head_size=128,
                scale=128**-0.5,
                num_kv_heads=8,
                alibi_slopes=None,
                sliding_window=513,
                kv_cache_dtype=cache_dtype,
            )


def test_yoco_rejects_recalibrating_after_shared_cache_write():
    from tests.model_executor.test_yoco_config import _make_vllm_config
    from vllm.config.compilation import CUDAGraphMode
    from vllm.model_executor.models.config import YOCOForCausalLMConfig

    runtime = _make_vllm_config(cudagraph_mode=CUDAGraphMode.FULL)
    runtime.cache_config.cache_dtype = "fp8"
    runtime.cache_config.calculate_kv_scales = True
    with pytest.raises(ValueError, match="requires fixed KV scales"):
        YOCOForCausalLMConfig.verify_and_update_config(runtime)
    runtime.cache_config.calculate_kv_scales = False
    YOCOForCausalLMConfig.verify_and_update_config(runtime)
