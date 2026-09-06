# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import vllm.model_executor.models.yoco as yoco_module
from convert_to_hf import convert_state_dict, create_hf_config
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.fused_moe.experts.yoco_triton import (
    yoco_weighted_swiglu,
)
from vllm.model_executor.models.yoco import (
    RMSClip,
    RMSNorm,
    YOCOCombinedOutputTransform,
    YOCOCrossAttention,
    YOCOCrossBlock,
    YOCODecoderLayer,
    YOCOForCausalLM,
    YOCOLatentInputTransform,
    YOCOLatentOutputTransform,
    YOCOModel,
    YOCOMoE,
    YOCORotaryEmbedding,
    YOCOSelfAttention,
    _maybe_dump_yoco_logical_routes,
    _select_yoco_fast_moe_backend,
    _yoco_align_qkv_linear,
    _yoco_align_rms_clip,
    _yoco_align_rms_norm,
    _yoco_align_rotary_embedding,
    _yoco_align_router_linear,
    _yoco_align_topk_routing,
    _yoco_apply_rotary_emb,
    _yoco_diff_attention_v2,
    _yoco_diff_attention_v3,
    _yoco_diff_attention_v3_dispatch,
    _yoco_logical_moe_layer_id,
    _yoco_runtime_sliding_window,
    _yoco_topk_routing,
)


def test_yoco_logical_moe_layer_ids_cover_all_universal_calls() -> None:
    logical_ids = [
        _yoco_logical_moe_layer_id(layer, loop, 10, 3)
        for loop in range(3)
        for layer in range(10)
    ]
    logical_ids.extend(
        _yoco_logical_moe_layer_id(layer, 0, 10, 3) for layer in range(10, 20)
    )
    assert logical_ids == list(range(40))
    with pytest.raises(ValueError, match="universal loop index"):
        _yoco_logical_moe_layer_id(0, 3, 10, 3)


def test_yoco_logical_route_dump_records_eager_topk(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(yoco_module, "_YOCO_LOGICAL_ROUTE_DUMP_ROOT", str(tmp_path))
    monkeypatch.setattr(yoco_module, "_YOCO_LOGICAL_ROUTE_DUMP_BATCHES", frozenset({8}))
    monkeypatch.setattr(yoco_module, "_YOCO_LOGICAL_ROUTE_DUMP_INDEX", 0)
    (tmp_path / "ENABLED").touch()
    topk_ids = torch.arange(64, dtype=torch.int64).view(8, 8)
    monkeypatch.setattr(
        yoco_module,
        "_yoco_topk_routing",
        lambda *args: (torch.ones(8, 8), topk_ids),
    )

    _maybe_dump_yoco_logical_routes(
        torch.zeros(8, 3),
        torch.zeros(8, 128),
        8,
        (4, 10, 3),
        2,
    )

    paths = list(tmp_path.glob("*.pt"))
    assert len(paths) == 1
    record = torch.load(paths[0], weights_only=True)
    assert record["logical_layer_id"] == 24
    assert record["num_tokens"] == 8
    assert torch.equal(record["topk_ids"], topk_ids.to(torch.int16))


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
    class FakeLinear(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            del args
            self.output_size = kwargs.get("output_size")
            self.output_sizes = kwargs.get("output_sizes")
            self.quant_config = kwargs.get("quant_config")

    class FakeAttention(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
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


def test_yoco_model_merges_shared_kv_only_in_fast_bf16_tp1(monkeypatch) -> None:
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
            quant_config=None,
            additional_config={"yoco_execution_mode": mode},
            compilation_config=SimpleNamespace(mode=0),
        )
        return YOCOModel(vllm_config=vllm_config)

    fast_model = build("fast")
    assert isinstance(fast_model.yoco_kv_proj, FakeMergedLinear)
    assert fast_model.yoco_kv_proj.output_sizes == [512, 512]
    assert fast_model.yoco_k_proj is None
    assert fast_model.yoco_v_proj is None

    align_model = build("align")
    assert align_model.yoco_kv_proj is None
    assert isinstance(align_model.yoco_k_proj, FakeColumnLinear)
    assert isinstance(align_model.yoco_v_proj, FakeColumnLinear)


def test_yoco_lm_head_fallback_matches_training_linear() -> None:
    hidden_states = torch.randn(4, 8, dtype=torch.bfloat16)
    weight = torch.randn(12, 8, dtype=torch.bfloat16)

    output = yoco_module._yoco_lm_head_dispatch(
        hidden_states,
        weight,
        use_sm100_kernel=False,
    )

    assert torch.equal(output, F.linear(hidden_states, weight))


def test_yoco_fast_lm_head_fallback_includes_training_fp32_cast() -> None:
    hidden_states = torch.randn(4, 8, dtype=torch.bfloat16)
    weight = torch.randn(12, 8, dtype=torch.bfloat16)

    output = yoco_module._yoco_lm_head_dispatch(
        hidden_states,
        weight,
        use_sm100_kernel=True,
    )

    assert output.dtype == torch.float32
    assert torch.equal(output, F.linear(hidden_states, weight).float())


def test_yoco_fast_compute_logits_uses_private_lm_head(monkeypatch) -> None:
    calls = 0

    def fake_dispatch(hidden_states, weight, use_sm100_kernel):
        nonlocal calls
        calls += 1
        assert use_sm100_kernel
        assert weight.shape == (3, 4)
        return hidden_states.new_full((hidden_states.shape[0], 3), 4.0)

    monkeypatch.setattr(yoco_module, "_yoco_lm_head_dispatch", fake_dispatch)
    causal_lm = YOCOForCausalLM.__new__(YOCOForCausalLM)
    torch.nn.Module.__init__(causal_lm)
    causal_lm.use_sm100_lm_head_kernel = True
    causal_lm.lm_head = SimpleNamespace(weight=torch.randn(3, 4))
    causal_lm.logits_processor = SimpleNamespace(scale=0.5)

    output = YOCOForCausalLM.compute_logits(causal_lm, torch.randn(2, 4))

    assert calls == 1
    assert torch.equal(output, torch.full((2, 3), 2.0))


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
    class FakeLinear(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            del args
            self.output_size = kwargs.get("output_size")
            self.output_sizes = kwargs.get("output_sizes")

    class FakeAttention(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
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


def test_yoco_latent_projections_stay_unquantized(monkeypatch) -> None:
    """llm-train leaves both latent projections at MixPrecisionLinear's
    BF16 default, including when the routed experts use MXFP8.
    """

    class FakeLinear(torch.nn.Module):
        def __init__(self, *args, quant_config=None, **kwargs) -> None:
            super().__init__()
            self.quant_config = quant_config

        def set_out_dtype(self, dtype: torch.dtype) -> None:
            self.out_dtype = dtype

    class FakeFusedMoE(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.quant_config = kwargs["quant_config"]
            self.use_tuned_config = kwargs["use_tuned_config"]
            self.apply_router_weight_before_w2 = kwargs["apply_router_weight_before_w2"]
            self.combined_output_transform = kwargs["combined_output_transform"]

    monkeypatch.setattr(yoco_module, "GateLinear", FakeLinear)
    monkeypatch.setattr(yoco_module, "ReplicatedLinear", FakeLinear)
    monkeypatch.setattr(yoco_module, "YOCOSharedExperts", FakeLinear)
    monkeypatch.setattr(yoco_module, "FusedMoE", FakeFusedMoE)
    monkeypatch.setattr(
        yoco_module,
        "RMSNorm",
        lambda *args, **kwargs: torch.nn.Identity(),
    )

    config = type(
        "Config",
        (),
        {
            "hidden_size": 3072,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 1024,
            "moe_latent_dim": 1024,
            "moe_latent_norm": True,
            "shared_expert_intermediate_size": 1024,
            "swiglu_limit": 10.0,
            "router_weights_normalized": True,
            "rms_norm_eps": 1e-6,
        },
    )()
    online_quant_config = object()

    module = YOCOMoE(
        config,
        quant_config=online_quant_config,
        prefix="model.layers.0.mlp",
    )

    assert module.fc1_latent_proj.quant_config is None
    assert module.fc2_latent_proj.quant_config is None
    assert module.shared_gate.quant_config is None
    assert module.shared_experts.quant_config is online_quant_config
    assert module.experts.quant_config is online_quant_config
    assert module.experts.use_tuned_config
    assert module.experts.apply_router_weight_before_w2
    assert not module.experts.yoco_align_weighted_swiglu
    assert not module.experts.yoco_align_deep_gemm_w2
    assert module.experts.yoco_separate_w2_config
    assert module.experts.yoco_fast_w13_config
    assert module.experts.yoco_triton_fallback_max_tokens == 1
    assert not module.experts.yoco_align_moe_sum
    assert module.experts.yoco_fast_moe_sum
    assert module.experts.combined_output_transform.execution_mode == "fast"

    align_module = YOCOMoE(
        config,
        quant_config=online_quant_config,
        prefix="model.layers.1.mlp",
        execution_mode="align",
    )
    assert not align_module.experts.use_tuned_config
    assert align_module.experts.apply_router_weight_before_w2
    assert align_module.experts.yoco_align_weighted_swiglu
    assert not align_module.experts.yoco_align_deep_gemm_w2
    assert not align_module.experts.yoco_separate_w2_config
    assert not align_module.experts.yoco_fast_w13_config
    assert align_module.experts.yoco_triton_fallback_max_tokens == 0
    assert align_module.experts.yoco_align_moe_sum
    assert not align_module.experts.yoco_fast_moe_sum
    assert align_module.experts.combined_output_transform.execution_mode == "align"

    fast_bf16_module = YOCOMoE(
        config,
        quant_config=None,
        prefix="model.layers.2.mlp",
        execution_mode="fast",
    )
    assert fast_bf16_module.experts.apply_router_weight_before_w2
    assert not fast_bf16_module.experts.yoco_align_weighted_swiglu
    assert not fast_bf16_module.experts.yoco_align_deep_gemm_w2
    assert fast_bf16_module.experts.yoco_separate_w2_config
    assert fast_bf16_module.experts.yoco_fast_w13_config
    assert fast_bf16_module.experts.yoco_triton_fallback_max_tokens == 1
    assert not fast_bf16_module.experts.yoco_align_moe_sum
    assert fast_bf16_module.experts.yoco_fast_moe_sum


def test_yoco_align_shared_expert_restores_separate_gemms(monkeypatch) -> None:
    projection_calls = 0
    observed: dict[str, torch.Tensor | float] = {}

    class FakeMergedLinear(torch.nn.Module):
        def __init__(self, input_size, output_sizes, **kwargs) -> None:
            super().__init__()
            del kwargs
            self.weight = torch.nn.Parameter(
                torch.empty(sum(output_sizes), input_size, dtype=torch.bfloat16)
            )

        def forward(self, x: torch.Tensor):
            nonlocal projection_calls
            projection_calls += 1
            return F.linear(x, self.weight), None

    class FakeRowLinear(torch.nn.Module):
        def __init__(self, input_size, output_size, **kwargs) -> None:
            super().__init__()
            del kwargs
            self.weight = torch.nn.Parameter(
                torch.empty(output_size, input_size, dtype=torch.bfloat16)
            )

        def forward(self, x: torch.Tensor):
            return F.linear(x, self.weight), None

    def record_swiglu(
        up: torch.Tensor,
        gate: torch.Tensor,
        limit: float,
    ) -> torch.Tensor:
        observed["up"] = up.detach().clone()
        observed["gate"] = gate.detach().clone()
        observed["limit"] = limit
        gate_fp32 = gate.float().clamp(max=limit)
        up_fp32 = up.float().clamp(min=-limit, max=limit)
        return (up_fp32 * F.silu(gate_fp32)).to(up.dtype)

    monkeypatch.setattr(yoco_module, "MergedColumnParallelLinear", FakeMergedLinear)
    monkeypatch.setattr(yoco_module, "RowParallelLinear", FakeRowLinear)
    monkeypatch.setattr(yoco_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        yoco_module,
        "SiluAndMulWithClampFP32",
        lambda *args, **kwargs: torch.nn.Identity(),
    )
    monkeypatch.setattr(
        yoco_module,
        "_yoco_align_shared_expert_swiglu",
        record_swiglu,
    )

    module = yoco_module.YOCOSharedExperts(
        hidden_size=8,
        intermediate_size=4,
        quant_config=None,
        reduce_results=False,
        prefix="model.layers.0.mlp.shared_experts",
        swiglu_limit=10.0,
        execution_mode="align",
    )
    gate_weight = torch.arange(32, dtype=torch.bfloat16).view(4, 8) / 32
    up_weight = -torch.arange(32, dtype=torch.bfloat16).view(4, 8) / 48
    module.gate_up_proj.weight.data.copy_(torch.cat((gate_weight, up_weight)))
    module.down_proj.weight.data.copy_(
        torch.arange(32, dtype=torch.bfloat16).view(8, 4) / 64
    )
    hidden = torch.arange(16, dtype=torch.bfloat16).view(2, 8) / 16

    actual = module(hidden)
    expected_up = F.linear(hidden, up_weight)
    expected_gate = F.linear(hidden, gate_weight)
    expected_activated = (
        expected_up.float().clamp(min=-10.0, max=10.0)
        * F.silu(expected_gate.float().clamp(max=10.0))
    ).to(expected_up.dtype)
    expected = F.linear(expected_activated, module.down_proj.weight)

    # Align bypasses MergedColumnParallelLinear.forward and performs the two
    # original training GEMMs over contiguous views of the packed parameter.
    assert projection_calls == 0
    assert torch.equal(observed["up"], expected_up)
    assert torch.equal(observed["gate"], expected_gate)
    assert observed["limit"] == 10.0
    assert torch.equal(actual, expected)


def test_yoco_fast_shared_expert_uses_m1_down_transpose(monkeypatch) -> None:
    gate_up_projection_calls = 0
    down_projection_calls = 0

    class FakeMergedLinear(torch.nn.Module):
        def __init__(self, input_size, output_sizes, **kwargs) -> None:
            super().__init__()
            del kwargs
            self.weight = torch.nn.Parameter(
                torch.empty(sum(output_sizes), input_size, dtype=torch.bfloat16)
            )

        def forward(self, x: torch.Tensor):
            nonlocal gate_up_projection_calls
            gate_up_projection_calls += 1
            return F.linear(x, self.weight), None

    class FakeRowLinear(torch.nn.Module):
        def __init__(self, input_size, output_size, **kwargs) -> None:
            super().__init__()
            del kwargs
            self.weight = torch.nn.Parameter(
                torch.empty(output_size, input_size, dtype=torch.bfloat16)
            )

        def forward(self, x: torch.Tensor):
            nonlocal down_projection_calls
            down_projection_calls += 1
            return F.linear(x, self.weight), None

    class FakeActivation(torch.nn.Module):
        def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
            gate, up = gate_up.chunk(2, dim=-1)
            return gate * up

    monkeypatch.setattr(yoco_module, "MergedColumnParallelLinear", FakeMergedLinear)
    monkeypatch.setattr(yoco_module, "RowParallelLinear", FakeRowLinear)
    monkeypatch.setattr(yoco_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        yoco_module,
        "SiluAndMulWithClampFP32",
        lambda *args, **kwargs: FakeActivation(),
    )

    module = yoco_module.YOCOSharedExperts(
        hidden_size=8,
        intermediate_size=4,
        quant_config=None,
        reduce_results=False,
        prefix="model.layers.0.mlp.shared_experts",
        execution_mode="fast",
    )
    module.gate_up_proj.weight.data.copy_(
        torch.arange(64, dtype=torch.bfloat16).view(8, 8) / 64
    )
    module.down_proj.weight.data.copy_(
        torch.arange(32, dtype=torch.bfloat16).view(8, 4) / 32
    )
    module._fast_down_weight_t = module.down_proj.weight.t().contiguous()

    single_input = torch.arange(8, dtype=torch.bfloat16).view(1, 8) / 8
    single_activated = module.act_fn(F.linear(single_input, module.gate_up_proj.weight))
    expected_single = torch.mm(single_activated, module._fast_down_weight_t)
    actual_single = module(single_input)
    assert gate_up_projection_calls == 1
    assert down_projection_calls == 0
    assert torch.equal(actual_single, expected_single)

    batch64_input = single_input.expand(64, -1).contiguous()
    batch64_activated = module.act_fn(
        F.linear(batch64_input, module.gate_up_proj.weight)
    )
    expected_batch64 = F.linear(batch64_activated, module.down_proj.weight)
    actual_batch64 = module(batch64_input)
    assert gate_up_projection_calls == 2
    assert down_projection_calls == 1
    assert torch.equal(actual_batch64, expected_batch64)

    batch2_input = single_input.expand(2, -1).contiguous()
    module(batch2_input)
    assert gate_up_projection_calls == 3
    assert down_projection_calls == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 7, 128])
def test_yoco_weighted_swiglu_matches_training_formula(num_tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(8100 + num_tokens)
    x = (
        5
        * torch.randn(
            num_tokens,
            2 * 3840,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
    ).contiguous()
    routing_weights = torch.rand(
        num_tokens,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).contiguous()
    actual = torch.empty(num_tokens, 3840, device="cuda", dtype=torch.bfloat16)

    gate, up = x.float().chunk(2, dim=-1)
    gate = gate.clamp(max=10.0)
    up = up.clamp(min=-10.0, max=10.0)
    expected = (torch.nn.functional.silu(gate) * up * routing_weights.unsqueeze(-1)).to(
        torch.bfloat16
    )
    yoco_weighted_swiglu(actual, x, routing_weights, 10.0)

    torch.testing.assert_close(actual, expected, rtol=4e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_weighted_swiglu_is_batch_invariant() -> None:
    generator = torch.Generator(device="cuda").manual_seed(8200)
    row = torch.randn(
        1,
        2 * 3840,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).contiguous()
    routing_weight = torch.tensor([0.125], device="cuda", dtype=torch.float32)
    expected = torch.empty(1, 3840, device="cuda", dtype=torch.bfloat16)
    yoco_weighted_swiglu(expected, row, routing_weight, 10.0)

    for num_tokens in (2, 17, 257):
        x = torch.randn(
            num_tokens,
            2 * 3840,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        x[0].copy_(row[0])
        routing_weights = torch.rand(
            num_tokens,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        routing_weights[0] = routing_weight[0]
        actual = torch.empty(num_tokens, 3840, device="cuda", dtype=torch.bfloat16)
        yoco_weighted_swiglu(actual, x, routing_weights, 10.0)
        assert torch.equal(actual[0], expected[0])


@torch.compile
def _llm_train_rms_norm_reference(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    return torch.nn.functional.rms_norm(
        x.to(torch.bfloat16),
        (x.shape[-1],),
        weight=weight.to(torch.bfloat16),
        eps=eps,
    )


@torch.compile
def _llm_train_rms_clip_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    limit: float,
) -> torch.Tensor:
    x_float = x.float()
    clip_coef = (
        limit * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + eps)
    ).clamp(max=1.0)
    return (x_float * clip_coef).to(x.dtype) * weight.to(x.dtype)


@torch.compile
def _llm_train_rotary_reference(
    cache: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_sin = cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)

    def apply(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat((y1, y2), dim=-1).to(x.dtype)

    return apply(query), apply(key)


@torch.compile
def _llm_train_diff_v3_reference(
    output: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    output = output * torch.sigmoid(gate).unsqueeze(-1)
    return output[:, 0::2] - output[:, 1::2]


@torch.compile
def _llm_train_topk_routing_reference(
    logits: torch.Tensor,
    topk: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    gate_scores = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
    scores, top_indices = torch.topk(gate_scores, k=topk, dim=-1)
    probs = scores / scores.sum(dim=-1, keepdim=True)
    routing_probs = torch.zeros_like(logits).scatter(
        1, top_indices, probs.to(logits.dtype)
    )
    routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    return probs, top_indices, routing_probs, routing_map, gate_scores


def test_yoco_embedding_conversion_preserves_lookup_exactly() -> None:
    weight = torch.arange(35, dtype=torch.bfloat16).view(7, 5)
    tokens = torch.tensor([6, 0, 3, 3, 1])

    converted = convert_state_dict({"tok_embeddings.weight": weight})

    converted_weight = converted["model.embed_tokens.weight"]
    assert torch.equal(converted_weight, weight)
    assert torch.equal(
        torch.nn.functional.embedding(tokens, converted_weight),
        torch.nn.functional.embedding(tokens, weight),
    )


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


def test_yoco_latent_transforms_preserve_reference_order() -> None:
    class Linear(torch.nn.Module):
        def __init__(self, weight: torch.Tensor) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(weight)

        def forward(self, x: torch.Tensor):
            return torch.nn.functional.linear(x, self.weight), None

    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.bfloat16)
    down = Linear(torch.arange(12, dtype=torch.bfloat16).view(3, 4) / 10)
    up = Linear(torch.arange(12, dtype=torch.bfloat16).view(4, 3) / 20)
    norm = RMSNorm(3, eps=1e-6, dtype=torch.float32)

    input_transform = YOCOLatentInputTransform(down, norm)
    latent = input_transform(x)
    expected_latent = norm(down(x)[0])
    torch.testing.assert_close(latent, expected_latent, rtol=0, atol=0)

    output_transform = YOCOLatentOutputTransform(norm, up)
    output = output_transform(latent)
    expected_output = up(norm(latent))[0]
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)


@pytest.mark.parametrize("execution_mode", ["align", "fast"])
def test_yoco_shared_output_gate_runs_after_reduction(execution_mode: str) -> None:
    gate = torch.nn.Linear(4, 1, bias=False)
    gate.weight.data.copy_(torch.tensor([[0.25, -0.5, 0.75, 1.0]]))
    transform = YOCOCombinedOutputTransform(  # type: ignore[arg-type]
        gate,
        execution_mode=execution_mode,
    )
    hidden_states = torch.tensor([[1.0, 2.0, -1.0, 0.5]])
    reduced_shared = torch.tensor([[2.0, -3.0, 4.0, -5.0]])
    routed = torch.tensor([[-1.0, 1.5, -2.0, 2.5]])

    actual = transform(reduced_shared, routed, hidden_states)
    scale = torch.sigmoid(torch.nn.functional.linear(hidden_states, gate.weight))
    expected = routed + scale * reduced_shared
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 8, 128])
def test_yoco_fast_combined_output_matches_sequential(num_tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(9100 + num_tokens)
    gate = torch.nn.Linear(3072, 1, bias=False, dtype=torch.bfloat16, device="cuda")
    gate.weight.data.normal_(generator=generator)
    transform = YOCOCombinedOutputTransform(  # type: ignore[arg-type]
        gate,
        execution_mode="fast",
    )
    hidden_states = torch.randn(
        num_tokens,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    shared = torch.randn(
        num_tokens,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    routed = torch.randn(
        num_tokens,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    expected_scale = torch.sigmoid(F.linear(hidden_states, gate.weight))
    expected = routed + expected_scale * shared
    actual = transform(shared, routed, hidden_states)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3.2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 8, 128])
def test_yoco_align_combined_output_is_bitwise_exact(num_tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(9150 + num_tokens)
    gate = torch.nn.Linear(3072, 1, bias=False, dtype=torch.bfloat16, device="cuda")
    gate.weight.data.normal_(generator=generator)
    transform = YOCOCombinedOutputTransform(  # type: ignore[arg-type]
        gate,
        execution_mode="align",
    )
    hidden_states = torch.randn(
        num_tokens,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    shared = torch.randn(
        num_tokens,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    routed = torch.randn(
        num_tokens,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    expected_scale = torch.sigmoid(F.linear(hidden_states, gate.weight))
    expected = routed + expected_scale * shared
    actual = transform(shared, routed, hidden_states)

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_fast_combined_output_is_batch_invariant() -> None:
    generator = torch.Generator(device="cuda").manual_seed(9200)
    gate = torch.nn.Linear(3072, 1, bias=False, dtype=torch.bfloat16, device="cuda")
    gate.weight.data.normal_(generator=generator)
    transform = YOCOCombinedOutputTransform(  # type: ignore[arg-type]
        gate,
        execution_mode="fast",
    )
    row_hidden = torch.randn(
        1,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    row_shared = torch.randn(
        1,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    row_routed = torch.randn(
        1,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    expected = transform(row_shared, row_routed, row_hidden)

    for num_tokens in (2, 17, 257):
        hidden_states = torch.randn(
            num_tokens,
            3072,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        shared = torch.randn(
            num_tokens,
            3072,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        routed = torch.randn(
            num_tokens,
            3072,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        hidden_states[0].copy_(row_hidden[0])
        shared[0].copy_(row_shared[0])
        routed[0].copy_(row_routed[0])
        actual = transform(shared, routed, hidden_states)
        assert torch.equal(actual[0], expected[0])


def test_yoco_separate_shared_reduction_keeps_collective_order(monkeypatch) -> None:
    import vllm.model_executor.layers.fused_moe.runner.moe_runner as runner_module
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    class MoEConfig:
        is_sequence_parallel = False
        tp_size = 2
        ep_size = 1

    class QuantMethod:
        moe_kernel = None

    runner = MoERunner.__new__(MoERunner)
    runner.reduce_shared_experts_separately = True
    runner.moe_config = MoEConfig()
    runner._quant_method = QuantMethod()

    calls: list[torch.Tensor] = []

    def fake_all_reduce(x: torch.Tensor) -> torch.Tensor:
        calls.append(x)
        return x + len(calls)

    monkeypatch.setattr(
        runner_module, "tensor_model_parallel_all_reduce", fake_all_reduce
    )
    shared = torch.tensor([10.0])
    routed = torch.tensor([20.0])
    reduced_shared, reduced_routed = runner._maybe_reduce_expert_outputs_separately(
        shared, routed
    )

    assert calls[0] is routed
    assert calls[1] is shared
    torch.testing.assert_close(reduced_routed, routed + 1)
    torch.testing.assert_close(reduced_shared, shared + 2)


def test_yoco_diff_v2_and_v3_formulas() -> None:
    attn1 = torch.tensor([[[1.0], [2.0]]])
    attn2 = torch.tensor([[[3.0], [4.0]]])
    v2_gate = torch.tensor([[0.0, 1.0]])
    v3_gate = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    torch.testing.assert_close(
        _yoco_diff_attention_v2(attn1, attn2, v2_gate),
        attn1 - torch.sigmoid(v2_gate).unsqueeze(-1) * attn2,
    )
    torch.testing.assert_close(
        _yoco_diff_attention_v3(attn1, attn2, v3_gate),
        attn1 * torch.sigmoid(v3_gate[:, 0::2]).unsqueeze(-1)
        - attn2 * torch.sigmoid(v3_gate[:, 1::2]).unsqueeze(-1),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 17, 128])
def test_yoco_diff_v3_is_training_compiled_exact(num_tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(41000 + num_tokens)
    output = torch.randn(
        num_tokens,
        64,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    gate = torch.randn(
        num_tokens,
        64,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )

    expected = _llm_train_diff_v3_reference(output, gate)
    actual = _yoco_diff_attention_v3(output[:, 0::2], output[:, 1::2], gate)

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("num_tokens", "twice_num_heads"),
    [
        (31, 16),
        (32, 16),
        (128, 16),
        (512, 16),
        (1, 64),
        (63, 64),
        (64, 64),
        (511, 64),
        (512, 64),
        (1410, 64),
    ],
)
def test_yoco_fast_diff_v3_is_training_compiled_exact(
    num_tokens: int, twice_num_heads: int
) -> None:
    if not hasattr(torch.ops.vllm, "yoco_diff_attention_v3"):
        pytest.skip("YOCO Triton custom op is unavailable")

    generator = torch.Generator(device="cuda").manual_seed(42000 + num_tokens)
    output = torch.randn(
        num_tokens,
        twice_num_heads,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    if twice_num_heads == 64:
        gate_storage = torch.randn(
            num_tokens,
            8256,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        gate = gate_storage[:, -twice_num_heads:]
        assert not gate.is_contiguous()
    else:
        gate = torch.randn(
            num_tokens,
            twice_num_heads,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )

    expected = _llm_train_diff_v3_reference(output, gate)
    compiled_dispatch = torch.compile(_yoco_diff_attention_v3_dispatch, fullgraph=True)
    actual = compiled_dispatch(output, gate, use_sm100_kernel=True)

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_fast_diff_v3_is_batch_independent_across_tuning_ranges() -> None:
    if not hasattr(torch.ops.vllm, "yoco_diff_attention_v3"):
        pytest.skip("YOCO Triton custom op is unavailable")

    generator = torch.Generator(device="cuda").manual_seed(43000)
    output = torch.randn(
        512,
        64,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    gate_storage = torch.randn(
        512,
        8256,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    gate = gate_storage[:, -64:]

    for smaller, larger in ((63, 64), (511, 512)):
        smaller_result = torch.ops.vllm.yoco_diff_attention_v3(
            output[:smaller], gate[:smaller]
        )
        larger_result = torch.ops.vllm.yoco_diff_attention_v3(
            output[:larger], gate[:larger]
        )
        assert torch.equal(smaller_result, larger_result[:smaller])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_heads", [64, 8])
@pytest.mark.parametrize("num_tokens", [17, 260])
def test_yoco_align_weighted_rms_clip_uses_fixed_reduction(
    num_heads: int,
    num_tokens: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(
        3100 + num_heads + num_tokens
    )
    x = 4 * torch.randn(
        num_tokens,
        num_heads,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    module = (
        RMSClip(
            128,
            eps=1e-6,
            limit=3.0,
            has_weight=True,
            execution_mode="align",
        )
        .cuda()
        .to(torch.bfloat16)
    )
    module.weight.data.uniform_(-2.0, 2.0, generator=generator)

    with torch.no_grad():
        expected = _llm_train_rms_clip_reference(
            x, module.weight, module.eps, module.limit
        )
        direct = _yoco_align_rms_clip(x, module.weight, module.eps, module.limit)
        actual = module(x)

    assert torch.equal(direct, expected)
    # Invariant Align deliberately fixes the reduction that the historical
    # training expression changed with M. Retain the BF16 accuracy check and
    # require exact results when splitting the same logical input.
    torch.testing.assert_close(actual, expected, rtol=1 / 128, atol=1e-6)
    with torch.no_grad():
        split = torch.cat([module(part) for part in x.split(7)])
    assert torch.equal(actual, split)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 64, 256])
def test_yoco_fast_cross_q_weighted_rms_clip_is_exact_and_strided(
    num_tokens: int,
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("YOCO fast cross-Q RMSClip is SM100-only")
    if not hasattr(torch.ops.vllm, "yoco_weighted_rms_clip"):
        pytest.skip("YOCO Triton custom op is unavailable")

    generator = torch.Generator(device="cuda").manual_seed(3200 + num_tokens)
    projected = 4 * torch.randn(
        num_tokens,
        8192 + 64,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    query = projected[:, :8192]
    if num_tokens > 1:
        assert not query.is_contiguous()
    norm = (
        RMSClip(
            128,
            eps=1e-6,
            limit=3.0,
            has_weight=True,
            execution_mode="fast",
        )
        .cuda()
        .to(torch.bfloat16)
    )
    norm.weight.data.uniform_(-2.0, 2.0, generator=generator)

    attention = YOCOCrossAttention.__new__(YOCOCrossAttention)
    torch.nn.Module.__init__(attention)
    attention.q_norm = norm
    attention.use_sm100_weighted_rms_clip_kernel = True
    attention.num_heads = 64
    attention.head_dim = 128

    expected = _llm_train_rms_clip_reference(
        query.unflatten(-1, (64, 128)),
        norm.weight,
        norm.eps,
        norm.limit,
    ).flatten(-2)
    actual = attention._normalize_query(query)

    assert actual.is_contiguous()
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_fast_cross_q_weighted_rms_clip_is_batch_invariant() -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("YOCO fast cross-Q RMSClip is SM100-only")
    if not hasattr(torch.ops.vllm, "yoco_weighted_rms_clip"):
        pytest.skip("YOCO Triton custom op is unavailable")

    generator = torch.Generator(device="cuda").manual_seed(3250)
    weight = torch.empty(128, device="cuda", dtype=torch.bfloat16)
    weight.uniform_(-2.0, 2.0, generator=generator)
    target = 4 * torch.randn(
        1,
        8192 + 64,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )

    def run(projected: torch.Tensor) -> torch.Tensor:
        query = projected[:, :8192].unflatten(-1, (64, 128))
        return torch.ops.vllm.yoco_weighted_rms_clip(query, weight, 1e-6, 3.0)

    expected = run(target)
    batch = 4 * torch.randn(
        256,
        8192 + 64,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    batch[0].copy_(target[0])
    actual = run(batch)

    assert torch.equal(actual[0], expected[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_align_rotary_is_native_compiled_exact() -> None:
    generator = torch.Generator(device="cuda").manual_seed(4100)
    rope = YOCORotaryEmbedding(
        head_size=128,
        max_position_embeddings=4096,
        base=10000.0,
        execution_mode="align",
    ).cuda()
    query = torch.randn(
        17, 64, 128, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    key = torch.randn(
        17, 8, 128, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    positions = torch.arange(17, device="cuda", dtype=torch.long) * 7
    cache = rope._get_cos_sin_cache(query.device)

    with torch.no_grad():
        expected = _llm_train_rotary_reference(cache, positions, query, key)
        direct = _yoco_align_rotary_embedding(cache, positions, query, key)
        actual = rope(positions, query.flatten(-2), key.flatten(-2))

    assert torch.equal(direct[0], expected[0])
    assert torch.equal(direct[1], expected[1])
    assert torch.equal(actual[0].view_as(expected[0]), expected[0])
    assert torch.equal(actual[1].view_as(expected[1]), expected[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 17, 128])
@pytest.mark.parametrize("query_heads,key_heads", [(48, 4), (64, 8)])
def test_yoco_rotary_cuda_matches_compiled_fallback(
    num_tokens: int, query_heads: int, key_heads: int
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(19 + num_tokens)
    head_dim = 128
    total_dim = (query_heads + 2 * key_heads) * head_dim
    qkv = torch.randn(
        num_tokens,
        total_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    query, key, _ = qkv.split(
        [query_heads * head_dim, key_heads * head_dim, key_heads * head_dim],
        dim=-1,
    )
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.long) * 3
    rope = YOCORotaryEmbedding(
        head_size=head_dim,
        max_position_embeddings=max(4096, num_tokens * 3),
        base=10000.0,
    ).cuda()

    cache = rope._get_cos_sin_cache(query.device)
    cos, sin = cache.index_select(0, positions).chunk(2, dim=-1)
    expected_query, expected_key = _yoco_apply_rotary_emb(
        query.view(num_tokens, query_heads, head_dim),
        key.view(num_tokens, key_heads, head_dim),
        cos,
        sin,
    )
    actual_query, actual_key = rope(positions, query, key)

    torch.testing.assert_close(
        actual_query.view_as(expected_query), expected_query, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_key.view_as(expected_key), expected_key, rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_rotary_custom_op_opcheck() -> None:
    qkv = torch.randn(3, 80 * 128, device="cuda", dtype=torch.bfloat16)
    query, key, _ = qkv.split([64 * 128, 8 * 128, 8 * 128], dim=-1)
    query = query.view(3, 64, 128)
    key = key.view(3, 8, 128)
    positions = torch.tensor([0, 7, 31], device="cuda", dtype=torch.long)
    rope = YOCORotaryEmbedding(128, 128, 10000.0).cuda()
    cache = rope._get_cos_sin_cache(query.device)

    torch.library.opcheck(
        torch.ops.vllm.yoco_rotary.default,
        (query, key, positions, cache),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 7, 17, 128])
@pytest.mark.parametrize("query_heads,key_heads", [(64, 8), (32, 4), (48, 4)])
def test_yoco_fused_qk_rms_clip_rotary_is_bitwise_exact(
    num_tokens: int,
    query_heads: int,
    key_heads: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(8120 + num_tokens)
    head_dim = 128
    total_dim = (query_heads + 2 * key_heads) * head_dim
    qkv = torch.randn(
        num_tokens,
        total_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    query, key, _ = qkv.split(
        [query_heads * head_dim, key_heads * head_dim, key_heads * head_dim],
        dim=-1,
    )
    query = query.view(num_tokens, query_heads, head_dim)
    key = key.view(num_tokens, key_heads, head_dim)
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.long) * 7
    rope = YOCORotaryEmbedding(
        head_size=head_dim,
        max_position_embeddings=max(4096, num_tokens * 7),
        base=10000.0,
    ).cuda()
    cache = rope._get_cos_sin_cache(query.device)
    clip = RMSClip(head_dim, eps=1e-6, limit=3.0).cuda()

    expected_query, expected_key = rope(
        positions,
        clip(query),
        clip(key),
    )
    actual_query, actual_key = torch.ops.vllm.yoco_qk_rms_clip_rotary(
        query,
        key,
        positions,
        cache,
        clip.eps,
        clip.limit,
    )

    torch.testing.assert_close(
        actual_query, expected_query.view_as(actual_query), rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_key, expected_key.view_as(actual_key), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 17, 128])
def test_yoco_fused_weighted_qk_rms_clip_rotary_matches_native_bf16(
    num_tokens: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(9200 + num_tokens)
    query = 4 * torch.randn(
        num_tokens,
        64,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    key = 4 * torch.randn(
        num_tokens,
        8,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    query_weight = torch.empty(128, device="cuda", dtype=torch.bfloat16).uniform_(
        -2.0, 2.0, generator=generator
    )
    key_weight = torch.empty(128, device="cuda", dtype=torch.bfloat16).uniform_(
        -2.0, 2.0, generator=generator
    )
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.long) * 7
    rope = YOCORotaryEmbedding(
        128,
        max(4096, num_tokens * 7),
        10000.0,
        execution_mode="align",
    ).cuda()
    cache = rope._get_cos_sin_cache(query.device)

    with torch.no_grad():
        clipped_query = _yoco_align_rms_clip(query, query_weight, 1e-6, 3.0)
        clipped_key = _yoco_align_rms_clip(key, key_weight, 1e-6, 3.0)
        expected_query, expected_key = _yoco_align_rotary_embedding(
            cache, positions, clipped_query, clipped_key
        )
        actual_query, actual_key = torch.ops.vllm.yoco_qk_rms_clip_rotary_weighted(
            query,
            key,
            query_weight,
            key_weight,
            positions,
            cache,
            1e-6,
            3.0,
        )

    # Inductor changes the 128-wide reduction tree between static and dynamic
    # shape compilations.  The fast kernel deliberately fixes one tree, so a
    # handful of values can land on the neighboring BF16 rounding point.
    for actual, expected in (
        (actual_query, expected_query),
        (actual_key, expected_key),
    ):
        error = actual.float() - expected.float()
        nrmse = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(
            expected.float()
        )
        assert error.abs().max() <= 0.0625
        assert nrmse <= 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_fused_qk_rms_clip_rotary_opcheck() -> None:
    qkv = torch.randn(3, 80 * 128, device="cuda", dtype=torch.bfloat16)
    query, key, _ = qkv.split([64 * 128, 8 * 128, 8 * 128], dim=-1)
    query = query.view(3, 64, 128)
    key = key.view(3, 8, 128)
    positions = torch.tensor([0, 7, 31], device="cuda", dtype=torch.long)
    rope = YOCORotaryEmbedding(128, 128, 10000.0).cuda()
    cache = rope._get_cos_sin_cache(query.device)

    torch.library.opcheck(
        torch.ops.vllm.yoco_qk_rms_clip_rotary.default,
        (query, key, positions, cache, 1e-6, 3.0),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_fused_weighted_qk_rms_clip_rotary_opcheck() -> None:
    query = torch.randn(3, 64, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(3, 8, 128, device="cuda", dtype=torch.bfloat16)
    query_weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    key_weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([0, 7, 31], device="cuda", dtype=torch.long)
    rope = YOCORotaryEmbedding(128, 128, 10000.0).cuda()
    cache = rope._get_cos_sin_cache(query.device)

    torch.library.opcheck(
        torch.ops.vllm.yoco_qk_rms_clip_rotary_weighted.default,
        (
            query,
            key,
            query_weight,
            key_weight,
            positions,
            cache,
            1e-6,
            3.0,
        ),
    )


def test_weighted_rms_clip_matches_training_order() -> None:
    module = RMSClip(2, eps=1e-6, limit=1.0, has_weight=True)
    module.weight.data.copy_(torch.tensor([2.0, 3.0]))
    x = torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16)

    x_float = x.float()
    coef = (torch.rsqrt(x_float.square().mean(-1, keepdim=True) + 1e-6)).clamp(max=1.0)
    expected = (x_float * coef).to(x.dtype) * module.weight.to(x.dtype)

    torch.testing.assert_close(module(x), expected)


def test_yoco_fused_add_rms_norm_cpu_fallback_matches_sequential() -> None:
    module = RMSNorm(4, eps=1e-6, dtype=torch.float32)
    module.weight.data.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.bfloat16)
    residual = torch.tensor([[0.25, 0.5, -0.75, 1.0]], dtype=torch.float32)

    expected_residual = residual + x.float()
    expected_normalized = module(expected_residual)
    actual = module(x, residual)
    assert isinstance(actual, tuple)
    actual_normalized, actual_residual = actual

    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(actual_normalized, expected_normalized, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [1024, 3072])
def test_yoco_align_rms_norm_uses_fixed_reduction(
    input_dtype: torch.dtype,
    hidden_size: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(1977)
    x = torch.randn(
        7,
        hidden_size,
        device="cuda",
        dtype=input_dtype,
        generator=generator,
    )
    weight = torch.randn(
        hidden_size,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    module = RMSNorm(
        hidden_size,
        eps=1e-6,
        dtype=torch.float32,
        execution_mode="align",
    ).cuda()
    module.weight.data.copy_(weight)

    # Native alignment inference runs under torch.no_grad(). Keep all three
    # compiled calls in that same specialization.
    with torch.no_grad():
        expected = _llm_train_rms_norm_reference(x, module.weight, 1e-6)
        direct = _yoco_align_rms_norm(x, module.weight, 1e-6)
        actual = module(x)

    assert torch.equal(direct, expected)
    deterministic = torch.ops.vllm.yoco_align_rms_norm(x, module.weight, 1e-6)
    assert torch.equal(actual, deterministic)
    split = torch.cat([module(row[None]) for row in x])
    assert torch.equal(actual, split)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 66, 128])
def test_yoco_fused_add_rms_norm_cuda_matches_sequential(num_tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(4321 + num_tokens)
    module = RMSNorm(3072, eps=1e-6, dtype=torch.bfloat16).cuda()
    module.weight.data.uniform_(-1.0, 1.0, generator=generator)
    x = torch.randn(
        num_tokens,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    residual = torch.randn(
        num_tokens,
        3072,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )

    expected_residual = residual + x.float()
    expected_normalized = module(expected_residual)
    actual = module(x, residual)
    assert isinstance(actual, tuple)
    actual_normalized, actual_residual = actual

    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(actual_normalized, expected_normalized, rtol=0, atol=0)


def test_yoco_decoder_residual_fusions_match_unfused_forward() -> None:
    class SelfAttention(torch.nn.Module):
        def forward(self, positions, hidden_states, loop_idx):
            del positions
            return (hidden_states * (loop_idx + 1) * 0.25).to(torch.bfloat16)

    class MLP(torch.nn.Module):
        def forward(self, hidden_states):
            return torch.tanh(hidden_states).to(torch.bfloat16)

    layer = YOCODecoderLayer.__new__(YOCODecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.is_self_layer = True
    layer.input_layernorm = RMSNorm(4, eps=1e-6, dtype=torch.float32)
    layer.post_attention_layernorm = RMSNorm(4, eps=1e-6, dtype=torch.float32)
    layer.self_attn = SelfAttention()
    layer.mlp = MLP()

    positions = torch.arange(2)
    initial = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0], [0.5, 1.5, -2.5, 3.5]],
        dtype=torch.float32,
    )

    legacy_hidden = initial
    for loop_idx in range(2):
        legacy_residual = legacy_hidden
        legacy_attention_input = layer.input_layernorm(legacy_hidden)
        assert isinstance(legacy_attention_input, torch.Tensor)
        legacy_attention_output = layer.self_attn(
            positions, legacy_attention_input, loop_idx
        )
        legacy_hidden = legacy_residual + legacy_attention_output.float()
        legacy_residual = legacy_hidden
        legacy_mlp_input = layer.post_attention_layernorm(legacy_hidden)
        assert isinstance(legacy_mlp_input, torch.Tensor)
        legacy_hidden = legacy_residual + layer.mlp(legacy_mlp_input).float()

    hidden_states = initial
    for loop_idx in range(2):
        hidden_states = layer(
            positions,
            hidden_states,
            loop_idx,
            None,
            None,
        )

    torch.testing.assert_close(hidden_states, legacy_hidden, rtol=0, atol=0)

    # Fast mode carries the MLP output and FP32 residual separately, then
    # folds their addition into the next layer's input RMSNorm.
    hidden_states = initial
    residual = None
    for loop_idx in range(2):
        hidden_states, residual = layer.forward_with_residual(
            positions,
            hidden_states,
            loop_idx,
            None,
            None,
            input_residual=residual,
        )
    assert residual is not None
    fused_hidden = residual + hidden_states.float()

    torch.testing.assert_close(fused_hidden, legacy_hidden, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 66, 128])
def test_yoco_align_topk_routing_uses_training_expert_order(
    num_tokens: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(9876 + num_tokens)
    logits = torch.randn(
        num_tokens,
        128,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    hidden_states = torch.empty(num_tokens, 1, device="cuda")

    expected_weights, expected_ids, _, _, _ = _llm_train_topk_routing_reference(
        logits, 8
    )
    expert_order = torch.argsort(expected_ids, dim=-1)
    expected_ids = torch.gather(expected_ids, dim=-1, index=expert_order)
    expected_weights = torch.gather(expected_weights, dim=-1, index=expert_order)
    actual_weights, actual_ids = _yoco_align_topk_routing(
        hidden_states,
        logits,
        topk=8,
        renormalize=True,
    )

    torch.testing.assert_close(actual_weights, expected_weights, rtol=2e-6, atol=0)
    assert torch.equal(actual_ids, expected_ids)
    assert torch.all(actual_ids[:, 1:] > actual_ids[:, :-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 3, 66, 110, 256])
def test_yoco_router_fused_topk_matches_reference(num_tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(1234 + num_tokens)
    logits = torch.randn(
        num_tokens,
        128,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    hidden_states = torch.empty(num_tokens, 1, device="cuda")

    actual_weights, actual_ids = _yoco_topk_routing(
        hidden_states,
        logits,
        topk=8,
        renormalize=True,
    )

    reference_scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    reference_weights, reference_ids = torch.topk(reference_scores, k=8, dim=-1)
    reference_weights /= reference_weights.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(actual_weights, reference_weights, rtol=2e-6, atol=0)
    assert actual_ids.dtype == torch.int32
    torch.testing.assert_close(
        actual_ids, reference_ids.to(torch.int32), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_router_fused_topk_handles_ties_deterministically() -> None:
    logits = torch.zeros(4, 128, dtype=torch.float32, device="cuda")
    hidden_states = torch.empty(4, 1, device="cuda")

    actual_weights, actual_ids = _yoco_topk_routing(
        hidden_states,
        logits,
        topk=8,
        renormalize=True,
    )
    # Both modes use a deterministic left-most tie break; torch.topk's
    # historical training ordering is not stable for tied values.
    expected_ids = torch.arange(8, device="cuda", dtype=torch.int32).expand(4, -1)
    expected_weights = torch.full_like(actual_weights, 1 / 8)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    align_weights, align_ids = _yoco_align_topk_routing(
        hidden_states, logits, topk=8, renormalize=True
    )
    assert torch.equal(align_weights, expected_weights)
    assert torch.equal(align_ids, expected_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_router_fused_topk_is_batch_independent() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260831)
    target = torch.randn(
        1,
        128,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    hidden_states = torch.empty(1, 1, device="cuda")
    expected_weights, expected_ids = _yoco_topk_routing(
        hidden_states,
        target,
        topk=8,
        renormalize=True,
    )

    for num_tokens, position in ((3, 1), (66, 37), (110, 109), (1024, 511)):
        logits = torch.randn(
            num_tokens,
            128,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        logits[position].copy_(target[0])
        actual_weights, actual_ids = _yoco_topk_routing(
            torch.empty(num_tokens, 1, device="cuda"),
            logits,
            topk=8,
            renormalize=True,
        )
        assert torch.equal(actual_weights[position], expected_weights[0])
        assert torch.equal(actual_ids[position], expected_ids[0])


def test_yoco_router_weight_cache_matches_runtime_normalization() -> None:
    module = YOCOMoE.__new__(YOCOMoE)
    torch.nn.Module.__init__(module)
    module.gate = torch.nn.Linear(8, 4, bias=False, dtype=torch.float32)
    module.gate.weight.data.copy_(
        torch.arange(32, dtype=torch.float32).view(4, 8) / 7 - 2
    )
    module.execution_mode = "fast"
    module.router_weights_normalized = False
    module.register_buffer("_normalized_gate_weight", None, persistent=False)

    expected = module.gate.weight / module.gate.weight.norm(
        dim=1, keepdim=True
    ).clamp_min(1e-6)
    module.initialize_router_weight_cache()

    assert module._normalized_gate_weight is not None
    torch.testing.assert_close(module._normalized_gate_weight, expected, rtol=0, atol=0)
    assert "_normalized_gate_weight" not in module.state_dict()


def test_yoco_router_weight_cache_skips_already_normalized_weights() -> None:
    module = YOCOMoE.__new__(YOCOMoE)
    torch.nn.Module.__init__(module)
    module.gate = torch.nn.Linear(8, 4, bias=False, dtype=torch.float32)
    module.execution_mode = "fast"
    module.router_weights_normalized = True
    module.register_buffer(
        "_normalized_gate_weight",
        torch.full_like(module.gate.weight, float("nan")),
        persistent=False,
    )

    module.initialize_router_weight_cache()

    assert module._normalized_gate_weight is None


def test_yoco_align_router_does_not_cache_normalized_weight() -> None:
    module = YOCOMoE.__new__(YOCOMoE)
    torch.nn.Module.__init__(module)
    module.gate = torch.nn.Linear(8, 4, bias=False, dtype=torch.float32)
    module.execution_mode = "align"
    module.router_weights_normalized = False
    module.register_buffer(
        "_normalized_gate_weight",
        torch.full_like(module.gate.weight, float("nan")),
        persistent=False,
    )

    module.initialize_router_weight_cache()

    assert module._normalized_gate_weight is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 8, 128])
def test_yoco_cached_router_linear_is_bitwise_exact(num_tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(8128 + num_tokens)
    hidden_states = torch.randn(
        num_tokens,
        3072,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    weight = torch.randn(
        128,
        3072,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    cached_weight = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-6)

    expected = torch.ops.vllm.yoco_router_linear_tf32(hidden_states, weight, True)
    actual = torch.ops.vllm.yoco_router_linear_tf32(hidden_states, cached_weight, False)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 12, 127])
def test_yoco_fast_router_linear_uses_actual_batch_shape(
    num_tokens: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260809)
    hidden_states = torch.randn(
        num_tokens,
        3072,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    weight = torch.randn(
        128,
        3072,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )

    previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_matmul_precision = torch.get_float32_matmul_precision()
    previous_cuda_precision = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    try:
        expected = torch.nn.functional.linear(hidden_states, weight)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
        torch.set_float32_matmul_precision(previous_matmul_precision)
        torch.backends.cuda.matmul.fp32_precision = previous_cuda_precision
    actual = torch.ops.vllm.yoco_router_linear_tf32(hidden_states, weight, False)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 12, 127])
def test_yoco_align_router_linear_is_ieee_and_batch_invariant(
    num_tokens: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260809)
    hidden_states = torch.randn(
        num_tokens,
        3072,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    weight = torch.randn(
        128,
        3072,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )

    expected = F.linear(hidden_states.double(), weight.double()).float()
    actual = _yoco_align_router_linear(hidden_states, weight, False)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=3e-5)
    split = torch.cat(
        [_yoco_align_router_linear(row[None], weight, False) for row in hidden_states]
    )
    assert torch.equal(actual, split)


def test_fast_prefill_runs_all_cross_layers_on_compact_tokens() -> None:
    calls = []

    class RecordingCrossLayer(torch.nn.Module):
        def forward(
            self,
            positions,
            hidden_states,
            loop_idx,
            yoco_key,
            yoco_value,
            kv_cache_dummy_dep=None,
            skip_kv_cache_update=False,
        ):
            calls.append(
                {
                    "num_tokens": hidden_states.shape[0],
                    "loop_idx": loop_idx,
                    "has_cache_dependency": kv_cache_dummy_dep is not None,
                    "skip_kv_cache_update": skip_kv_cache_update,
                }
            )
            return hidden_states + 1

    block = YOCOCrossBlock.__new__(YOCOCrossBlock)
    torch.nn.Module.__init__(block)
    block._cross_layers = [RecordingCrossLayer() for _ in range(10)]

    num_logits_tokens = 3
    hidden_states = torch.zeros(num_logits_tokens, 8)
    output = YOCOCrossBlock.forward(
        block,
        torch.arange(num_logits_tokens),
        hidden_states,
        torch.zeros(num_logits_tokens, 2),
        torch.zeros(num_logits_tokens, 2),
        torch.empty(0),
    )

    assert len(calls) == 10
    assert all(call["num_tokens"] == num_logits_tokens for call in calls)
    assert all(call["loop_idx"] == 0 for call in calls)
    assert calls[0]["has_cache_dependency"]
    assert calls[0]["skip_kv_cache_update"]
    assert not any(call["has_cache_dependency"] for call in calls[1:])
    assert not any(call["skip_kv_cache_update"] for call in calls[1:])
    torch.testing.assert_close(output, hidden_states + 10)


def test_fast_cross_block_carries_residual_between_layers() -> None:
    class ResidualCrossLayer(torch.nn.Module):
        def __init__(self, output_value: float) -> None:
            super().__init__()
            self.output_value = output_value

        def forward_with_residual(
            self,
            positions,
            hidden_states,
            loop_idx,
            yoco_key,
            yoco_value,
            kv_cache_dummy_dep=None,
            skip_kv_cache_update=False,
            input_residual=None,
        ):
            del (
                positions,
                loop_idx,
                yoco_key,
                yoco_value,
                kv_cache_dummy_dep,
                skip_kv_cache_update,
            )
            residual = (
                hidden_states
                if input_residual is None
                else input_residual + hidden_states.float()
            )
            output = torch.full_like(
                hidden_states,
                self.output_value,
                dtype=torch.bfloat16,
            )
            return output, residual

    block = YOCOCrossBlock.__new__(YOCOCrossBlock)
    torch.nn.Module.__init__(block)
    block._cross_layers = [
        ResidualCrossLayer(1.0),
        ResidualCrossLayer(2.0),
        ResidualCrossLayer(3.0),
    ]
    block.execution_mode = "fast"

    hidden_states = torch.zeros(2, 8)
    output = YOCOCrossBlock.forward(
        block,
        torch.arange(2),
        hidden_states,
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        torch.empty(0),
    )

    # Layer inputs materialize as 0, 1, and 3; the final pending output is 3.
    torch.testing.assert_close(output, torch.full_like(output, 6.0))


def test_kv_only_prefill_skips_every_cross_layer() -> None:
    class SelfBlock(torch.nn.Module):
        def forward(self, input_ids, positions, inputs_embeds=None):
            hidden_states = torch.full((positions.numel(), 8), 2.0)
            kv = torch.zeros(positions.numel(), 2)
            return hidden_states, kv, kv, torch.empty(0)

    class CrossBlock(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("KV-only prefill must not execute cross layers")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_block = SelfBlock()
            self.cross_block = CrossBlock()
            self.norm = torch.nn.Identity()
            self.full_model_warmed = True

    causal_lm = YOCOForCausalLM.__new__(YOCOForCausalLM)
    torch.nn.Module.__init__(causal_lm)
    causal_lm.model = Model()

    context = ForwardContext(
        no_compile_layers={},
        attn_metadata=None,  # type: ignore[arg-type]
        slot_mapping={},
    )
    with override_forward_context(context):
        output = YOCOForCausalLM._fast_prefill_forward(
            causal_lm,
            input_ids=torch.arange(4),
            positions=torch.arange(4),
            kv_only_prefill=True,
        )

    torch.testing.assert_close(output, torch.full((4, 8), 2.0))


def test_convert_yoco_v3_attention_and_latent_moe_weights() -> None:
    gate = torch.randn(64, 16)
    fc1 = torch.randn(8, 16)
    fc2 = torch.randn(16, 8)
    norm = torch.randn(8)
    state = {
        "layers.0.self_attn.gate_proj.weight": gate,
        "layers.0.mlp.fc1_latent_proj.weight": fc1,
        "layers.0.mlp.fc2_latent_proj.weight": fc2,
        "layers.0.mlp.fc1_latent_norm.weight": norm,
        "layers.0.mlp.fc2_latent_norm.weight": norm.clone(),
    }

    converted = convert_state_dict(state)

    assert torch.equal(converted["model.layers.0.self_attn.lambda_proj.weight"], gate)
    assert torch.equal(converted["model.layers.0.mlp.fc1_latent_proj.weight"], fc1)
    assert torch.equal(converted["model.layers.0.mlp.fc2_latent_proj.weight"], fc2)
    assert torch.equal(converted["model.layers.0.mlp.fc1_latent_norm.weight"], norm)
    assert torch.equal(converted["model.layers.0.mlp.fc2_latent_norm.weight"], norm)


def test_convert_legacy_diff_v3_interleaves_gate_rows() -> None:
    legacy_gate = torch.arange(8).reshape(4, 2)

    converted = convert_state_dict(
        {"layers.0.self_attn.lambda_proj.weight": legacy_gate},
        legacy_diff_v3=True,
    )

    expected = legacy_gate[[0, 2, 1, 3]]
    assert torch.equal(
        converted["model.layers.0.self_attn.lambda_proj.weight"],
        expected,
    )


def test_create_yoco_v3_latent_config(tmp_path) -> None:
    metadata = {
        "modelargs": {
            "d_model": 3072,
            "d_ffn": 9216,
            "head": 32,
            "cross_head": 32,
            "kv_head": 8,
            "cross_kv_head": 8,
            "head_dim": 128,
            "n_layers": 20,
            "vocab_size": 154880,
            "max_seq_len": 131072,
            "norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "qk_norm": False,
            "qk_rms_clip": True,
            "qk_rms_gamma": True,
            "diff_v2": False,
            "diff_v3": True,
            "yoco_cross_layers": 10,
            "yoco_window_size": 512,
            "universal_loop": 1,
            "moe": True,
            "moe_expert_num": 128,
            "moe_top_k": 8,
            "moe_ffn_dim": 3840,
            "moe_latent_dim": 1024,
            "moe_latent_norm": True,
            "d_shared_expert": 1280,
        }
    }

    config = create_hf_config(metadata, str(tmp_path))

    assert config["diff_v3"]
    assert not config["diff_v2"]
    assert not config["diff_attention"]
    assert config["qk_rms_gamma"]
    assert config["moe_latent_dim"] == 1024
    assert config["moe_latent_norm"]


def test_create_yoco_v2_defaults_to_weight_free_qk_clip(tmp_path) -> None:
    metadata = {
        "modelargs": {
            "d_model": 3072,
            "d_ffn": 9216,
            "head": 32,
            "cross_head": 32,
            "kv_head": 8,
            "cross_kv_head": 8,
            "head_dim": 128,
            "n_layers": 20,
            "vocab_size": 154880,
            "max_seq_len": 131072,
            "norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "qk_rms_clip": True,
            "diff_v2": True,
            "diff_v3": False,
        }
    }

    config = create_hf_config(metadata, str(tmp_path))

    assert config["diff_v2"]
    assert not config["diff_v3"]
    assert not config["qk_rms_gamma"]


def test_create_legacy_diff_both_lamb_as_v3(tmp_path) -> None:
    metadata = {
        "modelargs": {
            "d_model": 3072,
            "d_ffn": 9216,
            "head": 32,
            "cross_head": 32,
            "kv_head": 8,
            "cross_kv_head": 8,
            "head_dim": 128,
            "n_layers": 20,
            "vocab_size": 154880,
            "max_seq_len": 131072,
            "norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "diff_attention": True,
            "diff_both_lamb": True,
        }
    }

    config = create_hf_config(metadata, str(tmp_path))

    assert config["diff_v3"]
    assert not config["diff_v2"]
    assert not config["diff_attention"]
