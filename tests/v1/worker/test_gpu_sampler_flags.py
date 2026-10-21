# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for sampler flag tests", allow_module_level=True)

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 128


class MockReasoningConfig:
    reasoning_start_token_ids = [90]
    reasoning_end_token_ids = [91]
    natural_reasoning_end_token_ids = [91]


def _make_sampler() -> Sampler:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    return Sampler(
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        reasoning_config=MockReasoningConfig(),
    )


@pytest.mark.parametrize("offset", [0.0, 80.0])
def test_yoco_v2_bf16_greedy_preserves_legacy_scores_and_request_counts(offset):
    from vllm.platforms import current_platform
    from vllm.v1.sample.yoco_bf16 import YocoBf16GreedySampler as LegacyBf16
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.sample.yoco_bf16 import YocoBf16GreedySampler

    if current_platform.get_device_capability().major < 9:
        pytest.skip("native BF16 reductions require SM90+")
    base = _make_sampler()
    base.req_states.add_request("request", 1, [1], 0, 8)
    base.req_states.apply_staged_writes()
    sampler = YocoBf16GreedySampler(base)
    sampler.add_request(0, 1, SamplingParams(temperature=0, logprobs=5))
    sampler.apply_staged_writes()
    batch = InputBatch.make_dummy(1, 1, InputBuffers(4, 16, DEVICE))
    logits = torch.linspace(-3, 3, VOCAB_SIZE, device=DEVICE)[None] + offset
    output = sampler(logits, batch)
    sampled = logits.bfloat16().argmax(-1)
    reference = LegacyBf16.gather_logprobs(
        LegacyBf16.compute_logprobs(logits), 5, sampled
    )
    assert output.sampled_token_ids.item() == sampled.item()
    assert output.num_sampled.item() == 1 and output.num_rejected.item() == 0
    for name in ["logprob_token_ids", "logprobs", "selected_token_ranks"]:
        assert torch.equal(
            getattr(output.logprobs_tensors, name), getattr(reference, name)
        )


def test_yoco_v2_bf16_rejects_stochastic_and_custom_token_requests():
    from vllm.v1.worker.gpu.sample.yoco_bf16 import YocoBf16GreedySampler

    sampler = YocoBf16GreedySampler(_make_sampler())
    with pytest.raises(ValueError, match="greedy"):
        sampler.add_request(0, 1, SamplingParams(temperature=1))
    with pytest.raises(ValueError, match="logprob_token_ids"):
        sampler.add_request(0, 1, SamplingParams(temperature=0, logprob_token_ids=[1]))


def test_yoco_v2_bf16_prompt_scores_keep_the_sampler_precision_across_chunks():
    from vllm.platforms import current_platform
    from vllm.v1.sample.yoco_bf16 import YocoBf16GreedySampler as LegacyBf16
    from vllm.v1.worker.gpu.sample.prompt_logprob import (
        compute_prompt_logprobs_with_chunking,
    )
    from vllm.v1.worker.gpu.sample.yoco_bf16 import YocoBf16GreedySampler

    if current_platform.get_device_capability().major < 9:
        pytest.skip("native BF16 reductions require SM90+")
    logits = torch.linspace(70, 76, VOCAB_SIZE, device=DEVICE)[None].repeat(1030, 1)
    sampled = torch.full((1030,), VOCAB_SIZE - 1, device=DEVICE, dtype=torch.int64)
    actual = compute_prompt_logprobs_with_chunking(
        sampled,
        logits,
        lambda x: x,
        2,
        score_fn=YocoBf16GreedySampler.compute_topk_scores,
    )
    expected = LegacyBf16.gather_logprobs(
        LegacyBf16.compute_logprobs(logits), 2, sampled
    )
    for got, want in zip(
        actual,
        (expected.logprob_token_ids, expected.logprobs, expected.selected_token_ranks),
    ):
        assert torch.equal(got, want)


def test_yoco_v2_bf16_rejection_preserves_scores_and_rejected_rows(
    monkeypatch,
):
    from vllm.platforms import current_platform
    from vllm.v1.sample.yoco_bf16 import YocoBf16GreedySampler as LegacyBf16
    from vllm.v1.worker.gpu.model_states.yoco import YOCOModelState

    if current_platform.get_device_capability().major < 9:
        pytest.skip("native BF16 reductions require SM90+")
    monkeypatch.setenv("VLLM_YOCO_BF16_SAMPLING", "1")
    state = YOCOModelState.__new__(YOCOModelState)
    state.device = DEVICE
    state.vllm_config = SimpleNamespace(
        additional_config={"yoco_execution_mode": "fast"},
        speculative_config=SimpleNamespace(
            num_speculative_tokens=2,
            enable_adaptive_verification=False,
            rejection_sample_method="standard",
        ),
    )
    sampler, rejection = state.custom_sampler(_make_sampler())
    logits = torch.linspace(70, 76, VOCAB_SIZE, device=DEVICE)[None].repeat(3, 1)
    sampled = logits.bfloat16().argmax(-1)
    kept = torch.tensor([1], device=DEVICE, dtype=torch.int32)
    actual = rejection._get_logprobs_tensors(
        sampled[None],
        kept,
        logits,
        torch.tensor([0, 3], device=DEVICE, dtype=torch.int32),
        np.array([0, 3], dtype=np.int32),
        5,
    )
    flat = torch.tensor([sampled[0].item(), 0, 0], device=DEVICE, dtype=torch.int64)
    expected = LegacyBf16.gather_logprobs(LegacyBf16.compute_logprobs(logits), 5, flat)
    assert actual.cu_num_generated_tokens == [0, 3]
    for name in ["logprob_token_ids", "logprobs", "selected_token_ranks"]:
        assert torch.equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize(
    ("sampling_params", "expected"),
    [
        pytest.param(SamplingParams(), False, id="defaults"),
        pytest.param(SamplingParams(temperature=0.0), False, id="greedy"),
        pytest.param(
            SamplingParams(thinking_token_budget=3), True, id="thinking-budget"
        ),
        pytest.param(SamplingParams(logit_bias={1: 1.0}), True, id="logit-bias"),
        pytest.param(SamplingParams(frequency_penalty=0.1), True, id="penalty"),
        pytest.param(SamplingParams(_bad_words_token_ids=[[1]]), True, id="bad-words"),
        pytest.param(SamplingParams(temperature=0.7), True, id="temperature"),
        pytest.param(SamplingParams(min_p=0.1), True, id="min-p"),
        pytest.param(SamplingParams(top_k=10), True, id="top-k"),
        pytest.param(SamplingParams(top_p=0.9), True, id="top-p"),
        pytest.param(
            SamplingParams.for_sampler_warmup(), True, id="all-logits-processors"
        ),
    ],
)
def test_logits_processing_cache_matches_request_features(
    sampling_params: SamplingParams, expected: bool
):
    sampler = _make_sampler()
    sampler.add_request(3, prompt_len=1, sampling_params=sampling_params)

    assert sampler.needs_logits_processing[3] == expected


def test_logits_processing_cache_is_overwritten_when_slot_is_reused():
    sampler = _make_sampler()
    sampler.add_request(3, 1, SamplingParams.for_sampler_warmup())
    sampler.add_request(3, 1, SamplingParams())

    assert not sampler.needs_logits_processing[3]


def test_logits_processing_cache_only_checks_active_requests():
    sampler = _make_sampler()
    sampler.add_request(0, 1, SamplingParams(temperature=0.0))
    sampler.add_request(2, 1, SamplingParams.for_sampler_warmup())

    sampling_only = np.array([0], dtype=np.int32)
    with_processing = np.array([0, 2], dtype=np.int32)

    assert not np.any(sampler.needs_logits_processing[sampling_only])
    assert np.any(sampler.needs_logits_processing[with_processing])
