# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import vllm.platforms
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models.config import YOCOForCausalLMConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _make_vllm_config(
    *,
    cudagraph_mode: CUDAGraphMode | None,
    cudagraph_capture_sizes: list[int] | None = None,
    backend: AttentionBackendEnum | None = None,
    flash_attn_version: int | None = 4,
    kv_sharing_fast_prefill: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=cudagraph_mode,
            cudagraph_capture_sizes=cudagraph_capture_sizes,
            fast_moe_cold_start=True,
        ),
        attention_config=SimpleNamespace(
            backend=backend,
            flash_attn_version=flash_attn_version,
        ),
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=kv_sharing_fast_prefill),
    )


def test_yoco_fa4_full_cudagraph_forces_triton() -> None:
    vllm_config = _make_vllm_config(cudagraph_mode=CUDAGraphMode.FULL)

    YOCOForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.attention_config.backend == AttentionBackendEnum.TRITON_ATTN
    assert vllm_config.compilation_config.cudagraph_capture_sizes == [1, 2, 4]
    assert not vllm_config.cache_config.kv_sharing_fast_prefill
    assert not vllm_config.compilation_config.fast_moe_cold_start


def test_yoco_fa4_full_decode_only_forces_triton() -> None:
    vllm_config = _make_vllm_config(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)

    YOCOForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.attention_config.backend == AttentionBackendEnum.TRITON_ATTN
    assert not vllm_config.cache_config.kv_sharing_fast_prefill


def test_yoco_auto_fa4_full_cudagraph_forces_triton(monkeypatch) -> None:
    vllm_config = _make_vllm_config(
        cudagraph_mode=CUDAGraphMode.FULL,
        flash_attn_version=None,
    )
    fake_platform = SimpleNamespace(
        get_device_capability=lambda: SimpleNamespace(major=10)
    )
    monkeypatch.setattr(vllm.platforms, "current_platform", fake_platform)

    YOCOForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.attention_config.backend == AttentionBackendEnum.TRITON_ATTN
    assert not vllm_config.cache_config.kv_sharing_fast_prefill


def test_yoco_fa4_eager_keeps_flash_attention() -> None:
    vllm_config = _make_vllm_config(cudagraph_mode=CUDAGraphMode.NONE)

    YOCOForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.attention_config.backend is None
    assert vllm_config.cache_config.kv_sharing_fast_prefill
    assert not vllm_config.compilation_config.fast_moe_cold_start


def test_yoco_enforce_eager_keeps_flash_attention() -> None:
    vllm_config = _make_vllm_config(cudagraph_mode=None)

    YOCOForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.attention_config.backend is None
    assert vllm_config.cache_config.kv_sharing_fast_prefill
    assert not vllm_config.compilation_config.fast_moe_cold_start


def test_yoco_fa3_full_cudagraph_keeps_flash_attention() -> None:
    vllm_config = _make_vllm_config(
        cudagraph_mode=CUDAGraphMode.FULL,
        flash_attn_version=3,
    )

    YOCOForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.attention_config.backend is None
    assert vllm_config.cache_config.kv_sharing_fast_prefill


def test_yoco_preserves_explicit_cudagraph_capture_sizes() -> None:
    vllm_config = _make_vllm_config(
        cudagraph_mode=CUDAGraphMode.FULL,
        cudagraph_capture_sizes=[1, 2],
    )

    YOCOForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.compilation_config.cudagraph_capture_sizes == [1, 2]
