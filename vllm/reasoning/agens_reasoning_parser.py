# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Agens reasoning parser, adapted from vLLM's GLM-4.5 reasoning parser."""

from vllm.reasoning.deepseek_v3_reasoning_parser import (
    DeepSeekV3ReasoningWithThinkingParser,
)


class AgensReasoningParser(DeepSeekV3ReasoningWithThinkingParser):
    reasoning_output_field = "reasoning_content"
