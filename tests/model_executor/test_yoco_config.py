# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

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
    moe_backend: str = "auto",
    data_parallel_size: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=cudagraph_mode,
            cudagraph_capture_sizes=cudagraph_capture_sizes,
            fast_moe_cold_start=True,
        ),
        kernel_config=SimpleNamespace(moe_backend=moe_backend),
        parallel_config=SimpleNamespace(data_parallel_size=data_parallel_size),
        attention_config=SimpleNamespace(
            backend=backend,
            flash_attn_version=flash_attn_version,
        ),
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=kv_sharing_fast_prefill),
    )


def test_yoco_defaults_to_triton_moe() -> None:
    vllm_config = _make_vllm_config(cudagraph_mode=CUDAGraphMode.FULL)

    YOCOForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.kernel_config.moe_backend == "triton"


def test_yoco_preserves_explicit_moe_backend() -> None:
    vllm_config = _make_vllm_config(
        cudagraph_mode=CUDAGraphMode.FULL,
        moe_backend="flashinfer_cutlass",
    )

    YOCOForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.kernel_config.moe_backend == "flashinfer_cutlass"


def test_yoco_dp_preserves_kv_sharing_fast_prefill() -> None:
    vllm_config = _make_vllm_config(
        cudagraph_mode=CUDAGraphMode.FULL,
        data_parallel_size=4,
    )

    YOCOForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.cache_config.kv_sharing_fast_prefill


def test_yoco_fa4_full_cudagraph_keeps_flash_attention() -> None:
    vllm_config = _make_vllm_config(cudagraph_mode=CUDAGraphMode.FULL)

    YOCOForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.attention_config.backend is None
    assert vllm_config.compilation_config.cudagraph_capture_sizes is None
    assert vllm_config.cache_config.kv_sharing_fast_prefill
    assert not vllm_config.compilation_config.fast_moe_cold_start


def test_yoco_fa4_full_decode_only_keeps_flash_attention() -> None:
    vllm_config = _make_vllm_config(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)

    YOCOForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.attention_config.backend is None
    assert vllm_config.cache_config.kv_sharing_fast_prefill


def test_yoco_auto_fa4_full_cudagraph_keeps_flash_attention(monkeypatch) -> None:
    vllm_config = _make_vllm_config(
        cudagraph_mode=CUDAGraphMode.FULL,
        flash_attn_version=None,
    )
    fake_platform = SimpleNamespace(
        get_device_capability=lambda: SimpleNamespace(major=10)
    )
    monkeypatch.setattr(vllm.platforms, "current_platform", fake_platform)

    YOCOForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.attention_config.backend is None
    assert vllm_config.compilation_config.cudagraph_capture_sizes is None
    assert vllm_config.cache_config.kv_sharing_fast_prefill


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


def test_yoco_triton_full_cudagraph_disables_kv_sharing() -> None:
    vllm_config = _make_vllm_config(
        cudagraph_mode=CUDAGraphMode.FULL,
        backend=AttentionBackendEnum.TRITON_ATTN,
    )

    YOCOForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.attention_config.backend == AttentionBackendEnum.TRITON_ATTN
    assert vllm_config.compilation_config.cudagraph_capture_sizes == [
        1,
        2,
        4,
        8,
        16,
        32,
    ]
    assert not vllm_config.cache_config.kv_sharing_fast_prefill


def test_yoco_preserves_explicit_cudagraph_capture_sizes() -> None:
    vllm_config = _make_vllm_config(
        cudagraph_mode=CUDAGraphMode.FULL,
        cudagraph_capture_sizes=[1, 2],
    )

    YOCOForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.compilation_config.cudagraph_capture_sizes == [1, 2]


@pytest.mark.parametrize(
    "execution_mode,fa_version,capability_major,use_triton_decode,min_batch_size",
    [
        ("align", 2, 8, False, 0),
        ("fast", 2, 8, True, 0),
        ("fast", 4, 9, False, 0),
        ("align", 4, 10, False, 224),
        ("fast", 4, 10, True, 224),
    ],
)
def test_yoco_execution_mode_controls_triton_decode(
    monkeypatch,
    execution_mode: str,
    fa_version: int,
    capability_major: int,
    use_triton_decode: bool,
    min_batch_size: int,
) -> None:
    from vllm.v1.attention.backends import flash_attn

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="yoco"),
        ),
        additional_config={"yoco_execution_mode": execution_mode},
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            dcp_comm_backend="a2a",
        ),
    )
    monkeypatch.setattr(
        flash_attn,
        "get_current_vllm_config_or_none",
        lambda: vllm_config,
    )
    monkeypatch.setattr(
        flash_attn.current_platform,
        "get_device_capability",
        lambda: SimpleNamespace(major=capability_major),
    )
    monkeypatch.setattr(
        flash_attn,
        "get_flash_attn_version",
        lambda **_: fa_version,
    )
    monkeypatch.setattr(
        flash_attn,
        "flash_attn_supports_quant_query_input",
        lambda: False,
    )

    impl = flash_attn.FlashAttentionImpl(
        num_heads=16,
        head_size=128,
        scale=128**-0.5,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=513,
        kv_cache_dtype="auto",
    )

    assert impl.sliding_window == (512, 0)
    assert impl.use_triton_yoco_decode is use_triton_decode
    assert impl.yoco_triton_decode_min_batch_size == min_batch_size


@pytest.mark.parametrize(
    "batch_size,expected_kernel",
    [(223, "fa4"), (224, "triton")],
)
def test_yoco_sm100_fast_decode_dispatch(
    monkeypatch,
    batch_size: int,
    expected_kernel: str,
) -> None:
    from vllm.v1.attention.backends import flash_attn

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="yoco"),
        ),
        additional_config={"yoco_execution_mode": "fast"},
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            dcp_comm_backend="a2a",
        ),
    )
    monkeypatch.setattr(
        flash_attn,
        "get_current_vllm_config_or_none",
        lambda: vllm_config,
    )
    monkeypatch.setattr(
        flash_attn.current_platform,
        "get_device_capability",
        lambda: SimpleNamespace(major=10),
    )
    monkeypatch.setattr(flash_attn, "get_flash_attn_version", lambda **_: 4)
    monkeypatch.setattr(
        flash_attn,
        "flash_attn_supports_quant_query_input",
        lambda: False,
    )

    calls = []
    monkeypatch.setattr(
        flash_attn,
        "unified_attention",
        lambda **_: calls.append("triton"),
    )
    monkeypatch.setattr(
        flash_attn,
        "flash_attn_varlen_func",
        lambda **_: calls.append("fa4"),
    )

    num_heads = 16
    num_kv_heads = 2
    head_size = 128
    impl = flash_attn.FlashAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=513,
        kv_cache_dtype="auto",
    )
    metadata = flash_attn.FlashAttentionMetadata(
        num_actual_tokens=batch_size,
        max_query_len=1,
        query_start_loc=torch.arange(batch_size + 1, dtype=torch.int32),
        max_seq_len=513,
        seq_lens=torch.full((batch_size,), 513, dtype=torch.int32),
        block_table=torch.zeros((batch_size, 33), dtype=torch.int32),
        slot_mapping=torch.zeros(batch_size, dtype=torch.int64),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )
    query = torch.empty(batch_size, num_heads, head_size)
    key = torch.empty(batch_size, num_kv_heads, head_size)
    value = torch.empty_like(key)
    kv_cache = torch.empty(2, 33, 16, num_kv_heads, head_size)
    output = torch.empty_like(query)
    layer = SimpleNamespace(
        _q_scale=torch.tensor(1.0),
        _k_scale=torch.tensor(1.0),
        _v_scale=torch.tensor(1.0),
    )

    impl.forward(layer, query, key, value, kv_cache, metadata, output)

    assert calls == [expected_kernel]
