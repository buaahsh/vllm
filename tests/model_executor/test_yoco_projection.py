# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO projection regression tests."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import vllm.model_executor.models.yoco as yoco_module
from vllm.model_executor.layers.yoco_ops.projection import (
    _yoco_diff_attention_v2,
    _yoco_diff_attention_v3,
    _yoco_diff_attention_v3_dispatch,
)
from vllm.model_executor.models.yoco import YOCOForCausalLM


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

    def fake_dispatch(
        hidden_states, weight, use_sm100_kernel, output_dtype=torch.float32
    ):
        nonlocal calls
        calls += 1
        assert use_sm100_kernel
        assert output_dtype == torch.float32
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


@torch.compile
def _llm_train_diff_v3_reference(
    output: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    output = output * torch.sigmoid(gate).unsqueeze(-1)
    return output[:, 0::2] - output[:, 1::2]


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
        if num_tokens > 1:
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
