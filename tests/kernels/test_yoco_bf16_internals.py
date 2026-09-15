# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native BF16 reduction and logits tests, distinct from GEMM accumulation."""

from types import SimpleNamespace

import pytest
import regex as re
import torch

from vllm.model_executor.layers import yoco_bf16_math as math16
from vllm.model_executor.layers.yoco_ops.norm import RMSNorm
from vllm.platforms import current_platform
from vllm.v1.sample.yoco_bf16 import YocoBf16GreedySampler

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(100),
    reason="B200 native BF16 experiment",
)


@pytest.mark.parametrize("width", [128, 1024, 3072])
@pytest.mark.parametrize("rows", [0, 1, 8, 129])
@pytest.mark.parametrize("add", [False, True])
@torch.inference_mode()
def test_bf16_norm_error_and_graph_metadata(width, rows, add):
    torch.manual_seed(1930 + rows)
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    r = torch.randn_like(x) * 32
    w = torch.randn(width, device="cuda", dtype=torch.bfloat16)
    ref_x = (x + r) if add else x
    ref = ref_x.double()
    ref = ref * torch.rsqrt(ref.square().mean(-1, keepdim=True) + 1e-6) * w.double()
    if add:
        actual, residual = torch.ops.vllm.yoco_bf16_add_rms_reduction(x, r, w, 1e-6)
        assert torch.equal(residual, ref_x)
    else:
        actual = torch.ops.vllm.yoco_bf16_rms_reduction(x, w, 1e-6)
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    if rows:
        relative = (actual.double() - ref).norm() / ref.norm()
        assert relative < 0.015, relative.item()
    op = (
        torch.ops.vllm.yoco_bf16_add_rms_reduction.default
        if add
        else torch.ops.vllm.yoco_bf16_rms_reduction.default
    )
    args = (x, r, w, 1e-6) if add else (x, w, 1e-6)
    torch.library.opcheck(op, args, test_utils=("test_schema", "test_faketensor"))


@pytest.mark.parametrize("width", [128, 1024, 3072])
@torch.inference_mode()
def test_norm_ptx_uses_bf16_arithmetic(width):
    x = torch.ones(1, width, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(width, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(x)
    kernel = math16._norm_kernel[(1,)](
        x,
        x,
        w,
        math16._tables(x.device),
        o,
        o,
        width,
        math16.triton.next_power_of_2(width),
        math16._bits(1.0 / width),
        math16._bits(1e-6),
        False,
        num_warps=4 if width <= 1024 else 8,
    )
    ptx = kernel.asm["ptx"]
    assert "add.rn.bf16" in ptx and "mul.rn.bf16" in ptx
    assert not re.search(
        r"\b(?:add|mul|fma|div|sqrt|rsqrt|ex2|lg2)(?:\.[a-z0-9]+)*\.f32\b", ptx
    )


@pytest.mark.parametrize("vocab", [17, 128, 4097, 154880])
@pytest.mark.parametrize("rows", [1, 8])
@torch.inference_mode()
def test_logprobs_bf16_boundaries_and_error(vocab, rows):
    torch.manual_seed(1931 + vocab)
    logits = torch.randn(rows, vocab, device="cuda", dtype=torch.bfloat16) * 2
    logits[:, 0] = -float("inf")
    actual = torch.ops.vllm.yoco_bf16_logprobs(logits)
    expected = logits.double().log_softmax(-1)
    assert actual.dtype == torch.bfloat16
    assert torch.isneginf(actual[:, 0]).all()
    assert torch.isfinite(actual[:, 1:]).all()
    assert (actual[:, 1:].double() - expected[:, 1:]).abs().max() < 0.15
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))
    torch.library.opcheck(
        torch.ops.vllm.yoco_bf16_logprobs.default,
        (logits,),
        test_utils=("test_schema", "test_faketensor"),
    )


@torch.inference_mode()
def test_bf16_graph_replay_updates_inputs():
    x = torch.randn(8, 3072, device="cuda", dtype=torch.bfloat16)
    r = torch.randn_like(x)
    w = torch.ones(3072, device="cuda", dtype=torch.bfloat16)

    def run():
        y, _ = torch.ops.vllm.yoco_bf16_add_rms_reduction(x, r, w, 1e-6)
        return torch.ops.vllm.yoco_bf16_logprobs(y)

    for _ in range(3):
        run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for scale in [0.01, 1.0, 10.0]:
        x.copy_(torch.randn_like(x) * scale)
        graph.replay()
        assert torch.equal(actual, run())


def test_align_ignores_reduction_flag(monkeypatch):
    monkeypatch.setenv("VLLM_YOCO_BF16_REDUCTIONS", "1")
    norm = RMSNorm(1024, execution_mode="align")
    assert not norm.bf16_reductions


def test_greedy_sampler_rejects_unimplemented_precision_paths():
    sampler = YocoBf16GreedySampler()
    assert sampler.logits_dtype == torch.bfloat16
    with pytest.raises(ValueError, match="greedy"):
        sampler(
            torch.zeros(1, 128, device="cuda", dtype=torch.bfloat16),
            SimpleNamespace(all_greedy=False),
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_sampler_min_tokens_bias_and_returned_logprobs(dtype):
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.logits_processor.builtin import (
        LogitBiasLogitsProcessor,
        MinTokensLogitsProcessor,
    )
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.sampler import Sampler

    device = torch.device("cuda")
    metadata = SamplingMetadata(
        temperature=torch.zeros(1, device=device),
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=0,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(1, device=device),
        presence_penalties=torch.zeros(1, device=device),
        repetition_penalties=torch.ones(1, device=device),
        output_token_ids=[[]],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )
    censor = MinTokensLogitsProcessor(None, device, False)
    censor.min_toks = {0: (2, [], {0})}
    censor.logits_slice = (
        torch.tensor([0], device=device),
        torch.tensor([0], device=device),
    )
    bias = LogitBiasLogitsProcessor(None, device, False)
    bias.biases = {0: {3: 2.0}}
    bias.logits_slice = (
        torch.tensor([0], device=device),
        torch.tensor([3], device=device),
    )
    bias.bias_tensor = torch.tensor([2.0], device=device)
    metadata.logitsprocs = LogitsProcessors()
    metadata.logitsprocs.non_argmax_invariant.extend([censor, bias])
    sampler = YocoBf16GreedySampler() if dtype == torch.bfloat16 else Sampler()
    logits = torch.zeros(1, 128, device=device, dtype=dtype)
    logits[0, 0] = 4
    original = logits.clone()
    result = sampler(logits=logits, sampling_metadata=metadata)
    assert result.sampled_token_ids.item() == 3
    assert result.logprobs_tensors.logprobs.dtype == dtype
    assert censor.neg_inf_tensor.dtype == bias.bias_tensor.dtype == dtype
    expected = sampler.compute_logprobs(original)[0, 3]
    assert result.logprobs_tensors.logprobs[0, 0] == expected
    exported = result.logprobs_tensors.tolists()
    assert exported.logprobs[0, 0] == expected.item()
