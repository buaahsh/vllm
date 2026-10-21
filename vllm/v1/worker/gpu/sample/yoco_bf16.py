# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded BF16 greedy sampling on the V2 request/sampling state."""

from typing import cast

import torch

from vllm.model_executor.layers import yoco_bf16_math  # noqa: F401
from vllm.v1.sample.sampler import Sampler as LegacySampler
from vllm.v1.worker.gpu.sample.sampler import Sampler


class YocoBf16GreedySampler:
    greedy_only = True

    def __init__(self, base: Sampler):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def add_request(self, req_idx, prompt_len, sampling_params):
        if sampling_params.temperature != 0:
            raise ValueError("YOCO BF16 sampling currently requires greedy requests")
        if sampling_params.logprob_token_ids:
            raise ValueError("YOCO BF16 sampling does not support logprob_token_ids")
        self.base.add_request(req_idx, prompt_len, sampling_params)

    def __call__(self, logits, input_batch):
        return Sampler.__call__(
            cast(Sampler, self), logits.to(torch.bfloat16), input_batch
        )

    def apply_sampling_params(self, *args, **kwargs):
        return self.base.apply_sampling_params(*args, **kwargs).to(torch.bfloat16)

    def sample(
        self,
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        pos,
        input_ids,
        expanded_local_pos,
        return_logprobs=False,
    ):
        processed = self.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            skip_top_k_top_p=True,
        )
        return processed.argmax(dim=-1), processed

    @staticmethod
    def compute_topk_scores(
        logits,
        num_logprobs,
        sampled_token_ids,
        cu_num_logits=None,
        *,
        logits_mode=False,
        **kwargs,
    ):
        values = logits.to(torch.bfloat16)
        if not logits_mode:
            values = torch.ops.vllm.yoco_bf16_logprobs(values)
        output = LegacySampler.gather_logprobs(
            values, num_logprobs, sampled_token_ids.long()
        )
        is_tensor = isinstance(cu_num_logits, torch.Tensor)
        return output._replace(
            cu_num_generated_tokens=None if is_tensor else cu_num_logits,
            cu_num_generated_tokens_tensor=cu_num_logits if is_tensor else None,
        )
