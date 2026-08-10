# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from convert_to_hf import convert_state_dict, create_hf_config
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.models.yoco import (
    RMSClip,
    RMSNorm,
    YOCOCrossBlock,
    YOCODecoderLayer,
    YOCOForCausalLM,
    YOCOLatentInputTransform,
    YOCOLatentOutputTransform,
    YOCOMoE,
    YOCORotaryEmbedding,
    YOCOSharedOutputTransform,
    _yoco_apply_rotary_emb,
    _yoco_diff_attention_v2,
    _yoco_diff_attention_v3,
    _yoco_topk_routing,
)


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


def test_yoco_shared_output_gate_runs_after_reduction() -> None:
    gate = torch.nn.Linear(4, 1, bias=False)
    gate.weight.data.copy_(torch.tensor([[0.25, -0.5, 0.75, 1.0]]))
    transform = YOCOSharedOutputTransform(gate)  # type: ignore[arg-type]
    hidden_states = torch.tensor([[1.0, 2.0, -1.0, 0.5]])
    reduced_shared = torch.tensor([[2.0, -3.0, 4.0, -5.0]])

    actual = transform(reduced_shared, hidden_states)
    scale = torch.sigmoid(torch.nn.functional.linear(hidden_states, gate.weight))
    expected = scale * reduced_shared
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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


def test_yoco_decoder_post_attention_fusion_matches_unfused_forward() -> None:
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
    torch.testing.assert_close(actual_ids, reference_ids, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yoco_router_fused_topk_preserves_tie_order() -> None:
    logits = torch.zeros(4, 128, dtype=torch.float32, device="cuda")
    hidden_states = torch.empty(4, 1, device="cuda")

    actual_weights, actual_ids = _yoco_topk_routing(
        hidden_states,
        logits,
        topk=8,
        renormalize=True,
    )
    reference_weights, reference_ids = torch.topk(
        torch.softmax(logits, dim=-1), k=8, dim=-1
    )
    reference_weights /= reference_weights.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(actual_weights, reference_weights, rtol=0, atol=0)
    torch.testing.assert_close(actual_ids, reference_ids, rtol=0, atol=0)


def test_yoco_router_weight_cache_matches_runtime_normalization() -> None:
    module = YOCOMoE.__new__(YOCOMoE)
    torch.nn.Module.__init__(module)
    module.gate = torch.nn.Linear(8, 4, bias=False, dtype=torch.float32)
    module.gate.weight.data.copy_(
        torch.arange(32, dtype=torch.float32).view(4, 8) / 7 - 2
    )
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
    module.router_weights_normalized = True
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
def test_yoco_router_linear_pads_small_batches_consistently(
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
        expected = torch.nn.functional.linear(
            torch.nn.functional.pad(hidden_states, (0, 0, 0, 128 - num_tokens)),
            weight,
        )[:num_tokens]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
        torch.set_float32_matmul_precision(previous_matmul_precision)
        torch.backends.cuda.matmul.fp32_precision = previous_cuda_precision
    actual = torch.ops.vllm.yoco_router_linear_tf32(hidden_states, weight, False)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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
