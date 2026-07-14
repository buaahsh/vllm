"""Agens reasoning parser, adapted from vLLM's GLM-4.5 reasoning parser."""

from vllm.reasoning.deepseek_v3_reasoning_parser import (
    DeepSeekV3ReasoningWithThinkingParser,
)


class AgensReasoningParser(DeepSeekV3ReasoningWithThinkingParser):
    pass