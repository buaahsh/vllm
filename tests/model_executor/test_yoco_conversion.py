# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.model_executor.models.yoco as yoco_module
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
    _yoco_runtime_sliding_window,
    _yoco_topk_routing,
)


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

    align_module = YOCOMoE(
        config,
        quant_config=online_quant_config,
        prefix="model.layers.1.mlp",
        execution_mode="align",
    )
    assert not align_module.experts.use_tuned_config


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
@pytest.mark.parametrize("num_tokens", [31, 32, 128, 512])
def test_yoco_fast_diff_v3_is_training_compiled_exact(num_tokens: int) -> None:
    if not hasattr(torch.ops.vllm, "yoco_diff_attention_v3"):
        pytest.skip("YOCO Triton custom op is unavailable")

    generator = torch.Generator(device="cuda").manual_seed(42000 + num_tokens)
    output = torch.randn(
        num_tokens,
        16,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    gate = torch.randn(
        num_tokens,
        16,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )

    expected = _llm_train_diff_v3_reference(output, gate)
    compiled_dispatch = torch.compile(_yoco_diff_attention_v3_dispatch, fullgraph=True)
    actual = compiled_dispatch(output, gate, use_sm100_kernel=True)

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_heads", [64, 8])
def test_yoco_align_weighted_rms_clip_is_native_compiled_exact(
    num_heads: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(3100 + num_heads)
    x = 4 * torch.randn(
        17,
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
    assert torch.equal(actual, expected)


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
def test_yoco_align_rms_norm_is_native_compiled_exact(
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
    assert torch.equal(actual, expected)


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
def test_yoco_align_topk_routing_is_training_compiled_exact(
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
    actual_weights, actual_ids = _yoco_align_topk_routing(
        hidden_states,
        logits,
        topk=8,
        renormalize=True,
    )

    assert torch.equal(actual_weights, expected_weights)
    assert torch.equal(actual_ids, expected_ids)


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
def test_yoco_router_fused_topk_handles_ties_deterministically() -> None:
    logits = torch.zeros(4, 128, dtype=torch.float32, device="cuda")
    hidden_states = torch.empty(4, 1, device="cuda")

    actual_weights, actual_ids = _yoco_topk_routing(
        hidden_states,
        logits,
        topk=8,
        renormalize=True,
    )
    # torch.topk explicitly does not promise stable indices for ties.  Fast
    # mode uses a deterministic left-most tie break while align mode retains
    # torch.topk's training ordering.
    expected_ids = torch.arange(8, device="cuda").expand(4, -1)
    expected_weights = torch.full_like(actual_weights, 1 / 8)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)


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
def test_yoco_align_router_linear_uses_training_shape(
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

    expected = torch.nn.functional.linear(hidden_states, weight)
    actual = _yoco_align_router_linear(hidden_states, weight, False)

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
