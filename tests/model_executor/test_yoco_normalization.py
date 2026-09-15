# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO normalization regression tests."""

import pytest
import torch

from vllm.model_executor.layers.yoco_attention import YOCOCrossAttention
from vllm.model_executor.layers.yoco_ops.norm import (
    RMSClip,
    RMSNorm,
    _yoco_align_rms_clip,
    _yoco_align_rms_norm,
)
from vllm.model_executor.models.yoco import (
    YOCODecoderLayer,
)


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
def test_yoco_fast_cross_q_weighted_rms_clip_accuracy_and_strides(
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
    attention.fuse_fp8_attention = False
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
    # Fast explicitly rounds the clipped value to BF16 before gamma; the
    # compiled training expression may fuse that intermediate conversion.
    # The original Fast kernel has ~0.28% relative L2 on these inputs. Bound
    # that error rather than imposing Align's cross-implementation contract.
    torch.testing.assert_close(actual, expected, rtol=1 / 64, atol=1e-6)
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_l2.item() < 3e-3


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
        def forward(self, hidden_states, loop_idx=0):
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
