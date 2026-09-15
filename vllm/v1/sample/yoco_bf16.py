# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicitly bounded BF16 greedy sampler for YOCO precision experiments."""

import torch

from vllm.config.model import LogprobsMode
from vllm.model_executor.layers import yoco_bf16_math  # noqa: F401
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler


class YocoBf16GreedySampler(Sampler):
    greedy_only = True

    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs"):
        super().__init__(logprobs_mode)
        self.logits_dtype = torch.bfloat16

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> SamplerOutput:
        if not sampling_metadata.all_greedy:
            raise ValueError("YOCO BF16 sampling currently requires greedy requests")
        if sampling_metadata.logprob_token_ids:
            raise ValueError("YOCO BF16 sampling does not support logprob_token_ids")
        return super().forward(
            logits, sampling_metadata, predict_bonus_token, logprobs_mode_override
        )

    @staticmethod
    def compute_logprobs(logits: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.yoco_bf16_logprobs(logits.to(torch.bfloat16))
