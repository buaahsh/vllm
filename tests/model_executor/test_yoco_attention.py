# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO attention regression tests."""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import vllm.model_executor.models.yoco as yoco_module
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.yoco_attention import (
    YOCOCrossAttention,
    YOCOSelfAttention,
)
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_align_qkv_linear,
)
from vllm.model_executor.models.yoco import (
    YOCOForCausalLM,
    YOCOModel,
)
from vllm.model_executor.models.yoco_config import (
    _select_yoco_fast_moe_backend,
    _yoco_runtime_sliding_window,
)


def test_yoco_fast_selects_flashinfer_only_for_pure_prefill(
    monkeypatch, tmp_path
) -> None:
    import vllm.utils.flashinfer as flashinfer_utils
    from vllm.model_executor.layers.fused_moe.experts import yoco_trtllm_bf16

    monkeypatch.setattr(
        yoco_module.current_platform,
        "get_device_capability",
        lambda: SimpleNamespace(major=10),
    )
    monkeypatch.setattr(
        flashinfer_utils, "has_flashinfer_cutlass_fused_moe", lambda: True
    )
    monkeypatch.setattr(yoco_trtllm_bf16, "has_yoco_trtllm_bf16_clamp", lambda: False)
    config = SimpleNamespace(
        hidden_size=3072,
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=3840,
        moe_latent_dim=1024,
        swiglu_limit=10.0,
    )
    vllm_config = SimpleNamespace(
        additional_config={},
        kv_transfer_config=SimpleNamespace(
            kv_connector="MooncakeConnector",
            kv_role="kv_producer",
        ),
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=True),
        kernel_config=SimpleNamespace(
            moe_backend="triton",
            enable_flashinfer_autotune=False,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=256,
            max_num_batched_tokens=32768,
        ),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=32),
    )

    assert (
        _select_yoco_fast_moe_backend(
            execution_mode="fast",
            quant_config=None,
            tp_size=1,
            config=config,
            vllm_config=vllm_config,
        )
        == "flashinfer_cutlass"
    )

    vllm_config.kv_transfer_config.kv_role = "kv_consumer"
    assert (
        _select_yoco_fast_moe_backend(
            execution_mode="fast",
            quant_config=None,
            tp_size=1,
            config=config,
            vllm_config=vllm_config,
        )
        == "flashinfer_cutlass"
    )
    assert vllm_config.kernel_config.enable_flashinfer_autotune is True

    monkeypatch.setattr(yoco_trtllm_bf16, "has_yoco_trtllm_bf16_clamp", lambda: True)
    assert (
        _select_yoco_fast_moe_backend(
            execution_mode="fast",
            quant_config=None,
            tp_size=1,
            config=config,
            vllm_config=vllm_config,
        )
        == "yoco_flashinfer_trtllm"
    )
    vllm_config.additional_config["yoco_fast_decode_trtllm_moe"] = False
    assert (
        _select_yoco_fast_moe_backend(
            execution_mode="fast",
            quant_config=None,
            tp_size=1,
            config=config,
            vllm_config=vllm_config,
        )
        == "flashinfer_cutlass"
    )
    del vllm_config.additional_config["yoco_fast_decode_trtllm_moe"]

    vllm_config.compilation_config.max_cudagraph_capture_size = 64
    assert (
        _select_yoco_fast_moe_backend(
            execution_mode="fast",
            quant_config=None,
            tp_size=1,
            config=config,
            vllm_config=vllm_config,
        )
        == "flashinfer_cutlass"
    )
    monkeypatch.setenv("VLLM_YOCO_FLASHINFER_AUTOTUNE_CACHE", "/tmp/cache.json")
    assert (
        _select_yoco_fast_moe_backend(
            execution_mode="fast",
            quant_config=None,
            tp_size=1,
            config=config,
            vllm_config=vllm_config,
        )
        == "yoco_flashinfer_trtllm"
    )
    verified_cache = tmp_path / "verified-graph128.json"

    def cache_key(num_tokens: int) -> str:
        return (
            "('flashinfer::trtllm_bf16_moe', 'MoERunner', "
            f"(({num_tokens}, 1024), (0,), ({num_tokens}, 8), "
            f"({num_tokens}, 8), ({num_tokens}, 1024), (0,), (0,), (0,)), ())"
        )

    configs = {
        cache_key(32): ["MoERunner", [32, 24]],
        cache_key(64): ["MoERunner", [32, 17]],
        cache_key(128): ["MoERunner", [32, 17]],
    }
    verified_cache.write_text(json.dumps(configs))
    monkeypatch.setenv("VLLM_YOCO_FLASHINFER_AUTOTUNE_CACHE", str(verified_cache))
    vllm_config.compilation_config.max_cudagraph_capture_size = 128
    assert (
        _select_yoco_fast_moe_backend(
            execution_mode="fast",
            quant_config=None,
            tp_size=1,
            config=config,
            vllm_config=vllm_config,
        )
        == "yoco_flashinfer_trtllm"
    )
    configs[cache_key(256)] = ["MoERunner", [64, 0]]
    verified_cache.write_text(json.dumps(configs))
    vllm_config.compilation_config.max_cudagraph_capture_size = 256
    assert (
        _select_yoco_fast_moe_backend(
            execution_mode="fast",
            quant_config=None,
            tp_size=1,
            config=config,
            vllm_config=vllm_config,
        )
        == "yoco_flashinfer_trtllm"
    )
    configs[cache_key(256)] = ["MoERunner", [32, 17]]
    verified_cache.write_text(json.dumps(configs))
    assert (
        _select_yoco_fast_moe_backend(
            execution_mode="fast",
            quant_config=None,
            tp_size=1,
            config=config,
            vllm_config=vllm_config,
        )
        == "flashinfer_cutlass"
    )
    vllm_config.compilation_config.max_cudagraph_capture_size = 128
    configs[cache_key(128)] = ["MoERunner", [16, 65]]
    verified_cache.write_text(json.dumps(configs))
    assert (
        _select_yoco_fast_moe_backend(
            execution_mode="fast",
            quant_config=None,
            tp_size=1,
            config=config,
            vllm_config=vllm_config,
        )
        == "flashinfer_cutlass"
    )
    vllm_config.compilation_config.max_cudagraph_capture_size = 64
    monkeypatch.delenv("VLLM_YOCO_FLASHINFER_AUTOTUNE_CACHE")
    vllm_config.additional_config["yoco_fast_decode_trtllm_max_capture"] = 64
    assert (
        _select_yoco_fast_moe_backend(
            execution_mode="fast",
            quant_config=None,
            tp_size=1,
            config=config,
            vllm_config=vllm_config,
        )
        == "yoco_flashinfer_trtllm"
    )
    del vllm_config.additional_config["yoco_fast_decode_trtllm_max_capture"]
    vllm_config.compilation_config.max_cudagraph_capture_size = 32

    vllm_config.scheduler_config.max_num_seqs = 8
    vllm_config.kernel_config.enable_flashinfer_autotune = False
    assert (
        _select_yoco_fast_moe_backend(
            execution_mode="fast",
            quant_config=None,
            tp_size=1,
            config=config,
            vllm_config=vllm_config,
        )
        is None
    )
    vllm_config.scheduler_config.max_num_seqs = 256
    vllm_config.additional_config["yoco_fast_decode_flashinfer_moe"] = False
    assert (
        _select_yoco_fast_moe_backend(
            execution_mode="fast",
            quant_config=None,
            tp_size=1,
            config=config,
            vllm_config=vllm_config,
        )
        is None
    )
    del vllm_config.additional_config["yoco_fast_decode_flashinfer_moe"]
    vllm_config.kv_transfer_config.kv_role = "kv_producer"
    vllm_config.kernel_config.enable_flashinfer_autotune = True
    assert (
        _select_yoco_fast_moe_backend(
            execution_mode="fast",
            quant_config=None,
            tp_size=1,
            config=config,
            vllm_config=vllm_config,
        )
        is None
    )
    vllm_config.kernel_config.enable_flashinfer_autotune = False
    vllm_config.additional_config["yoco_fast_prefill_flashinfer_moe"] = False
    assert (
        _select_yoco_fast_moe_backend(
            execution_mode="fast",
            quant_config=None,
            tp_size=1,
            config=config,
            vllm_config=vllm_config,
        )
        is None
    )


def test_yoco_self_attention_merges_qkv_lambda_only_in_fast_bf16_tp1(
    monkeypatch,
) -> None:
    import vllm.model_executor.layers.yoco_attention as yoco_module

    class FakeLinear(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            del args
            self.output_size = kwargs.get("output_size")
            self.output_sizes = kwargs.get("output_sizes")
            self.quant_config = kwargs.get("quant_config")
            self.quant_method = (
                UnquantizedLinearMethod() if self.quant_config is None else object()
            )
            self.tp_size = 1
            self.bias = None

    class FakeAttention(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.kv_cache_dtype = "auto"
            del args, kwargs

    monkeypatch.setattr(yoco_module, "QKVParallelLinear", FakeLinear)
    monkeypatch.setattr(yoco_module, "ColumnParallelLinear", FakeLinear)
    monkeypatch.setattr(yoco_module, "MergedColumnParallelLinear", FakeLinear)
    monkeypatch.setattr(yoco_module, "RowParallelLinear", FakeLinear)
    monkeypatch.setattr(yoco_module, "Attention", FakeAttention)
    monkeypatch.setattr(yoco_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        yoco_module,
        "_supports_yoco_sm100_diff_v3_kernel",
        lambda execution_mode: False,
    )
    monkeypatch.setattr(yoco_module, "_build_qk_norm", lambda *args, **kwargs: None)

    config = SimpleNamespace(
        hidden_size=3072,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        diff_v3=True,
        sliding_window_size=512,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
    )

    def build(
        mode: str,
        quant_config=None,
        default_dtype: torch.dtype = torch.bfloat16,
    ) -> YOCOSelfAttention:
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(default_dtype)
        try:
            return YOCOSelfAttention(
                config=config,
                layer_idx=0,
                universal_loop=1,
                num_hidden_layers=2,
                cache_config=None,
                quant_config=quant_config,
                prefix="model.layers.0.self_attn",
                execution_mode=mode,
            )
        finally:
            torch.set_default_dtype(previous_dtype)

    fast = build("fast")
    assert isinstance(fast.qkv_lambda_proj, FakeLinear)
    assert fast.qkv_lambda_proj.output_sizes == [8192, 1024, 1024, 64]
    assert fast.qkv_proj is None
    assert fast.lambda_proj is None

    align = build("align")
    assert align.qkv_lambda_proj is None
    assert isinstance(align.qkv_proj, FakeLinear)
    assert isinstance(align.lambda_proj, FakeLinear)
    assert align.lambda_proj.output_size == 64

    quant_config = object()
    quantized_fast = build("fast", quant_config)
    assert quantized_fast.qkv_lambda_proj is None
    assert isinstance(quantized_fast.qkv_proj, FakeLinear)
    assert quantized_fast.qkv_proj.quant_config is quant_config
    assert isinstance(quantized_fast.lambda_proj, FakeLinear)
    assert quantized_fast.lambda_proj.quant_config is None

    fp16_fast = build("fast", default_dtype=torch.float16)
    assert fp16_fast.qkv_lambda_proj is None
    assert isinstance(fp16_fast.qkv_proj, FakeLinear)
    assert isinstance(fp16_fast.lambda_proj, FakeLinear)


def test_yoco_fast_self_qkv_lambda_loads_all_checkpoint_weights() -> None:
    class FakeMergedLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(8, 4))

            def weight_loader(param, loaded_weight, shard_id) -> None:
                shard_sizes = (4, 1, 1, 2)
                offset = sum(shard_sizes[:shard_id])
                param.data.narrow(0, offset, shard_sizes[shard_id]).copy_(loaded_weight)

            self.weight.weight_loader = weight_loader

    merged = FakeMergedLinear()
    self_attn = torch.nn.Module()
    self_attn.qkv_lambda_proj = merged
    self_layer = torch.nn.Module()
    self_layer.self_attn = self_attn

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([self_layer, torch.nn.Module()])
    model.embed_tokens = torch.nn.Embedding(1, 4)

    causal_lm = YOCOForCausalLM.__new__(YOCOForCausalLM)
    torch.nn.Module.__init__(causal_lm)
    causal_lm.model = model
    causal_lm.config = SimpleNamespace(
        num_hidden_layers=2,
        yoco_cross_layers=1,
        moe_intermediate_size=2,
        num_experts=1,
    )
    causal_lm._rotary_caches_initialized = False
    causal_lm._router_weight_caches_initialized = False

    weights = [
        torch.arange(16, dtype=merged.weight.dtype).reshape(4, 4),
        torch.arange(4, dtype=merged.weight.dtype).reshape(1, 4) + 100,
        torch.arange(4, dtype=merged.weight.dtype).reshape(1, 4) + 200,
        torch.arange(8, dtype=merged.weight.dtype).reshape(2, 4) + 300,
    ]
    loaded_names = causal_lm.load_weights(
        [
            ("model.layers.0.self_attn.q_proj.weight", weights[0]),
            ("model.layers.0.self_attn.k_proj.weight", weights[1]),
            ("model.layers.0.self_attn.v_proj.weight", weights[2]),
            ("model.layers.0.self_attn.lambda_proj.weight", weights[3]),
        ]
    )

    target_name = "model.layers.0.self_attn.qkv_lambda_proj.weight"
    assert loaded_names == {target_name}
    assert torch.equal(merged.weight, torch.cat(weights))


def test_yoco_fast_self_qkv_lambda_dispatches_at_b200_crossover() -> None:
    class FakeMergedLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(8, 3))
            self.calls = 0

        def forward(self, hidden_states):
            self.calls += 1
            return F.linear(hidden_states, self.weight), None

    attention = YOCOSelfAttention.__new__(YOCOSelfAttention)
    torch.nn.Module.__init__(attention)
    attention.execution_mode = "fast"
    attention.q_size = 4
    attention.kv_size = 1
    attention.num_gate_heads = 2
    attention.qkv_lambda_proj = FakeMergedLinear()
    attention.qkv_proj = None
    attention.lambda_proj = None

    small_input = torch.randn(4096, 3)
    small_q, small_k, small_v, small_lambda = attention._project_qkv(small_input)
    expected_small = F.linear(small_input, attention.qkv_lambda_proj.weight)
    assert attention.qkv_lambda_proj.calls == 1
    assert small_lambda is not None
    assert torch.equal(
        torch.cat((small_q, small_k, small_v, small_lambda), dim=-1),
        expected_small,
    )

    large_input = torch.randn(4097, 3)
    large_q, large_k, large_v, large_lambda = attention._project_qkv(large_input)
    assert attention.qkv_lambda_proj.calls == 1
    assert large_lambda is None
    assert torch.equal(
        torch.cat((large_q, large_k, large_v), dim=-1),
        F.linear(large_input, attention.qkv_lambda_proj.weight[:6]),
    )
    assert torch.equal(
        attention._project_lambda(large_input),
        F.linear(large_input, attention.qkv_lambda_proj.weight[6:]),
    )


@pytest.mark.parametrize(
    "quantization,ignore,expected_merged",
    [
        (None, (), True),
        ("fp8_per_block", (), True),
        ("fp8_per_tensor", (), False),
        ("fp8_per_block", ("model.yoco_k_proj",), False),
        ("fp8_per_block", ("re:model.yoco_[kv]_proj$",), True),
    ],
)
def test_yoco_model_merges_shared_kv_when_precision_compatible(
    monkeypatch, quantization, ignore, expected_merged
) -> None:
    from tests.model_executor.test_yoco_fast_precision import quant_config

    class FakeEmbedding(torch.nn.Module):
        def __init__(self, num_embeddings, embedding_dim, **kwargs) -> None:
            super().__init__()
            del num_embeddings, kwargs
            self.weight = torch.nn.Parameter(torch.empty(1, embedding_dim))

    class FakeMergedLinear(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            del args
            self.output_sizes = kwargs["output_sizes"]

    class FakeColumnLinear(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            del args, kwargs

    class FakeDecoderLayer(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            del args, kwargs

    monkeypatch.setattr(yoco_module, "VocabParallelEmbedding", FakeEmbedding)
    monkeypatch.setattr(yoco_module, "MergedColumnParallelLinear", FakeMergedLinear)
    monkeypatch.setattr(yoco_module, "ColumnParallelLinear", FakeColumnLinear)
    monkeypatch.setattr(yoco_module, "YOCODecoderLayer", FakeDecoderLayer)
    monkeypatch.setattr(
        yoco_module, "RMSNorm", lambda *args, **kwargs: torch.nn.Identity()
    )
    monkeypatch.setattr(yoco_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        yoco_module,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=1),
    )

    config = SimpleNamespace(
        hidden_size=3072,
        vocab_size=128,
        num_hidden_layers=2,
        universal_loop=1,
        yoco_cross_layers=1,
        cross_kv_head=4,
        head_dim=128,
        rms_norm_eps=1e-6,
    )
    cache_config = SimpleNamespace(kv_sharing_fast_prefill=False)

    def build(mode: str) -> YOCOModel:
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=config),
            cache_config=cache_config,
            scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
            quant_config=quant_config(quantization, ignore),
            kv_transfer_config=None,
            kernel_config=SimpleNamespace(
                moe_backend="triton", enable_flashinfer_autotune=False
            ),
            additional_config={"yoco_execution_mode": mode},
            compilation_config=SimpleNamespace(mode=0),
        )
        return YOCOModel(vllm_config=vllm_config, prefix="model")

    fast_model = build("fast")
    if expected_merged:
        assert isinstance(fast_model.yoco_kv_proj, FakeMergedLinear)
        assert fast_model.yoco_kv_proj.output_sizes == [512, 512]
        assert fast_model.yoco_k_proj is None
        assert fast_model.yoco_v_proj is None
    else:
        assert fast_model.yoco_kv_proj is None
        assert isinstance(fast_model.yoco_k_proj, FakeColumnLinear)
        assert isinstance(fast_model.yoco_v_proj, FakeColumnLinear)

    align_model = build("align")
    assert align_model.yoco_kv_proj is None
    assert isinstance(align_model.yoco_k_proj, FakeColumnLinear)
    assert isinstance(align_model.yoco_v_proj, FakeColumnLinear)


def test_yoco_fast_shared_kv_loads_both_checkpoint_names() -> None:
    class FakeMergedLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(4, 4))

            def weight_loader(param, loaded_weight, shard_id) -> None:
                param.data.narrow(0, shard_id * 2, 2).copy_(loaded_weight)

            self.weight.weight_loader = weight_loader

    merged = FakeMergedLinear()
    model = torch.nn.Module()
    model.yoco_kv_proj = merged
    model.embed_tokens = torch.nn.Embedding(1, 4)

    causal_lm = YOCOForCausalLM.__new__(YOCOForCausalLM)
    torch.nn.Module.__init__(causal_lm)
    causal_lm.model = model
    causal_lm.config = SimpleNamespace(
        num_hidden_layers=2,
        yoco_cross_layers=1,
        moe_intermediate_size=2,
        num_experts=1,
    )
    causal_lm._rotary_caches_initialized = False
    causal_lm._router_weight_caches_initialized = False

    key_weight = torch.arange(8, dtype=merged.weight.dtype).reshape(2, 4)
    value_weight = key_weight + 100
    loaded_names = causal_lm.load_weights(
        [
            ("model.k_proj.weight", key_weight),
            ("model.yoco_v_proj.weight", value_weight),
        ]
    )

    assert loaded_names == {"model.yoco_kv_proj.weight"}
    assert torch.equal(merged.weight[:2], key_weight)
    assert torch.equal(merged.weight[2:], value_weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_fast_shared_kv_merged_gemm_matches_separate_gemms() -> None:
    class MergedKVLinear(torch.nn.Module):
        def __init__(self, key_weight, value_weight) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.cat((key_weight, value_weight)))

        def forward(self, hidden_states):
            return F.linear(hidden_states, self.weight), None

    generator = torch.Generator(device="cuda").manual_seed(9300)
    local_kv_dim = 512
    key_weight = (
        torch.randn(
            local_kv_dim,
            3072,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.02
    )
    value_weight = (
        torch.randn(
            local_kv_dim,
            3072,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.02
    )
    merged = MergedKVLinear(key_weight, value_weight)

    model = YOCOModel.__new__(YOCOModel)
    torch.nn.Module.__init__(model)
    model.yoco_kv_proj = merged
    model.yoco_k_proj = None
    model.yoco_v_proj = None
    model.yoco_local_kv_dim = local_kv_dim
    hidden_states = torch.randn(
        33,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    actual_key, actual_value = model.project_yoco_kv(hidden_states)
    expected_key = F.linear(hidden_states, key_weight)
    expected_value = F.linear(hidden_states, value_weight)

    torch.testing.assert_close(actual_key, expected_key, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_value, expected_value, rtol=2e-2, atol=2e-2)


def test_yoco_cross_attention_merges_q_lambda_only_in_fast_bf16_tp1(
    monkeypatch,
) -> None:
    import vllm.model_executor.layers.yoco_attention as yoco_module

    class FakeLinear(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            del args
            self.output_size = kwargs.get("output_size")
            self.output_sizes = kwargs.get("output_sizes")
            self.tp_size = 1
            self.bias = None
            self.quant_config = kwargs.get("quant_config")
            self.quant_method = (
                UnquantizedLinearMethod() if self.quant_config is None else object()
            )

    class FakeAttention(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.kv_cache_dtype = "auto"
            del args, kwargs

    monkeypatch.setattr(yoco_module, "ColumnParallelLinear", FakeLinear)
    monkeypatch.setattr(yoco_module, "MergedColumnParallelLinear", FakeLinear)
    monkeypatch.setattr(yoco_module, "RowParallelLinear", FakeLinear)
    monkeypatch.setattr(yoco_module, "Attention", FakeAttention)
    monkeypatch.setattr(yoco_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        yoco_module,
        "_supports_yoco_sm100_diff_v3_kernel",
        lambda execution_mode: False,
    )
    monkeypatch.setattr(yoco_module, "_build_qk_norm", lambda *args, **kwargs: None)

    config = SimpleNamespace(
        hidden_size=3072,
        cross_head=32,
        cross_kv_head=8,
        head_dim=128,
        diff_v3=True,
        rms_norm_eps=1e-6,
    )

    def build(mode: str) -> YOCOCrossAttention:
        return YOCOCrossAttention(
            config=config,
            layer_idx=10,
            first_cross_layer_idx=10,
            cache_config=None,
            quant_config=None,
            prefix="model.layers.10.self_attn",
            execution_mode=mode,
        )

    fast = build("fast")
    assert isinstance(fast.q_lambda_proj, FakeLinear)
    assert fast.q_lambda_proj.output_sizes == [8192, 64]
    assert fast.q_proj is None
    assert fast.lambda_proj is None

    align = build("align")
    assert align.q_lambda_proj is None
    assert isinstance(align.q_proj, FakeLinear)
    assert align.q_proj.output_size == 8192
    assert isinstance(align.lambda_proj, FakeLinear)
    assert align.lambda_proj.output_size == 64


def test_yoco_fast_cross_q_lambda_loads_both_checkpoint_weights() -> None:
    class FakeMergedLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(6, 4))

            def weight_loader(param, loaded_weight, shard_id) -> None:
                shard_sizes = (4, 2)
                offset = sum(shard_sizes[:shard_id])
                param.data.narrow(0, offset, shard_sizes[shard_id]).copy_(loaded_weight)

            self.weight.weight_loader = weight_loader

    merged = FakeMergedLinear()
    cross_attn = torch.nn.Module()
    cross_attn.q_lambda_proj = merged
    cross_layer = torch.nn.Module()
    cross_layer.self_attn = cross_attn

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module(), cross_layer])
    model.embed_tokens = torch.nn.Embedding(1, 4)

    causal_lm = YOCOForCausalLM.__new__(YOCOForCausalLM)
    torch.nn.Module.__init__(causal_lm)
    causal_lm.model = model
    causal_lm.config = SimpleNamespace(
        num_hidden_layers=2,
        yoco_cross_layers=1,
        moe_intermediate_size=2,
        num_experts=1,
    )
    causal_lm._rotary_caches_initialized = False
    causal_lm._router_weight_caches_initialized = False

    q_weight = torch.arange(16, dtype=merged.weight.dtype).reshape(4, 4)
    lambda_weight = torch.arange(8, dtype=merged.weight.dtype).reshape(2, 4) + 100
    loaded_names = causal_lm.load_weights(
        [
            ("model.layers.1.self_attn.q_proj.weight", q_weight),
            ("model.layers.1.self_attn.lambda_proj.weight", lambda_weight),
        ]
    )

    target_name = "model.layers.1.self_attn.q_lambda_proj.weight"
    assert loaded_names == {target_name}
    assert torch.equal(merged.weight[:4], q_weight)
    assert torch.equal(merged.weight[4:], lambda_weight)


def test_yoco_fast_cross_q_lambda_dispatches_at_b200_crossover() -> None:
    class FakeMergedLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(6, 3))
            self.calls = 0

        def forward(self, hidden_states):
            self.calls += 1
            return F.linear(hidden_states, self.weight), None

    attention = YOCOCrossAttention.__new__(YOCOCrossAttention)
    torch.nn.Module.__init__(attention)
    attention.q_size = 4
    attention.num_gate_heads = 2
    attention.q_lambda_proj = FakeMergedLinear()
    attention.q_proj = None
    attention.lambda_proj = None

    small_input = torch.randn(2048, 3)
    small_q, small_lambda = attention._project_query(small_input)
    expected_small = F.linear(small_input, attention.q_lambda_proj.weight)
    assert attention.q_lambda_proj.calls == 1
    assert small_lambda is not None
    assert torch.equal(small_q, expected_small[:, :4])
    assert torch.equal(small_lambda, expected_small[:, 4:])

    large_input = torch.randn(2049, 3)
    large_q, large_lambda = attention._project_query(large_input)
    assert attention.q_lambda_proj.calls == 1
    assert large_lambda is None
    assert torch.equal(
        large_q,
        F.linear(large_input, attention.q_lambda_proj.weight[:4]),
    )
    assert torch.equal(
        attention._project_lambda(large_input),
        F.linear(large_input, attention.q_lambda_proj.weight[4:]),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 33, 256])
def test_yoco_fast_cross_q_lambda_merged_gemm_matches_separate_gemms(
    num_tokens: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(9400 + num_tokens)
    q_size = 8192
    lambda_size = 64
    q_weight = (
        torch.randn(
            q_size,
            3072,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.02
    )
    lambda_weight = (
        torch.randn(
            lambda_size,
            3072,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.02
    )
    hidden_states = torch.randn(
        num_tokens,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    merged = F.linear(hidden_states, torch.cat((q_weight, lambda_weight)))
    actual_q, actual_lambda = merged.split((q_size, lambda_size), dim=-1)
    expected_q = F.linear(hidden_states, q_weight)
    expected_lambda = F.linear(hidden_states, lambda_weight)

    torch.testing.assert_close(actual_q, expected_q, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_lambda, expected_lambda, rtol=2e-2, atol=2e-2)


def test_yoco_runtime_sliding_window_matches_flash_attention_semantics() -> None:
    # llm-train's (512, 0) means 512 previous tokens plus the current token.
    assert _yoco_runtime_sliding_window(512) == 513


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [17, 513])
def test_yoco_prefill_attention_matches_training_flash_attention(
    num_tokens: int,
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    generator = torch.Generator(device="cuda").manual_seed(31000 + num_tokens)
    query = torch.randn(
        num_tokens,
        64,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    key = torch.randn(
        num_tokens,
        8,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    value = torch.randn(
        num_tokens,
        8,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    cu_seqlens = torch.tensor([0, num_tokens], device="cuda", dtype=torch.int32)
    scale = 128**-0.5

    expected = flash_attn.flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens,
        cu_seqlens,
        num_tokens,
        num_tokens,
        softmax_scale=scale,
        causal=True,
        window_size=(512, 0),
    )
    actual = torch.empty_like(query)
    runtime_window = _yoco_runtime_sliding_window(512)
    flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        out=actual,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=num_tokens,
        softmax_scale=scale,
        causal=True,
        window_size=[runtime_window - 1, 0],
        fa_version=2,
    )

    assert torch.equal(actual, expected)


def test_yoco_align_qkv_uses_three_independent_bf16_linears() -> None:
    generator = torch.Generator().manual_seed(2026)
    hidden_states = torch.randn(7, 16, dtype=torch.bfloat16, generator=generator)
    packed_weight = torch.randn(20, 16, dtype=torch.bfloat16, generator=generator)

    actual = _yoco_align_qkv_linear(hidden_states, packed_weight, 12, 4)
    expected = tuple(
        torch.nn.functional.linear(hidden_states, weight)
        for weight in packed_weight.split((12, 4, 4), dim=0)
    )

    assert all(torch.equal(x, y) for x, y in zip(actual, expected))
