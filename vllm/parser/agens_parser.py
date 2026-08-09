# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Agens unified parser, adapted from vLLM's DelegatingParser."""
# Reason: preserve trailing and explicitly empty reasoning at the tool boundary.

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.abstract_parser import DelegatingParser
from vllm.reasoning.agens_reasoning_parser import AgensReasoningParser
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.agens_tool_parser import AgensToolParser
from vllm.tool_parsers.utils import Tool


class AgensParser(DelegatingParser):
    reasoning_parser_cls = AgensReasoningParser
    tool_parser_cls = AgensToolParser

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(tokenizer, *args, **kwargs)
        self._reasoning_parser = AgensReasoningParser(tokenizer, *args, **kwargs)
        self._tool_parser = AgensToolParser(tokenizer, tools)
        self._pending_reasoning_end_text: str | None = None

    def _strip_delayed_reasoning_end_text(self, delta_text: str) -> str:
        pending_text = getattr(self, "_pending_reasoning_end_text", None)
        if pending_text is None:
            return delta_text

        matching_chars = 0
        for actual, expected in zip(delta_text, pending_text):
            if actual != expected:
                break
            matching_chars += 1

        if matching_chars == len(pending_text):
            self._pending_reasoning_end_text = None
            return delta_text[matching_chars:]
        if matching_chars == len(delta_text):
            self._pending_reasoning_end_text = pending_text[matching_chars:]
            return ""

        self._pending_reasoning_end_text = None
        return delta_text

    def _split_delayed_reasoning_end(
        self, delta_text: str
    ) -> tuple[str, str] | None:
        end_text = self._reasoning_parser.reasoning_end_str
        if end_text is None or end_text in delta_text:
            return None

        max_prefix_length = min(len(delta_text), len(end_text) - 1)
        for prefix_length in range(max_prefix_length, 0, -1):
            if delta_text.endswith(end_text[:prefix_length]):
                self._pending_reasoning_end_text = end_text[prefix_length:]
                return delta_text[:-prefix_length], end_text
        return None

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
    ) -> DeltaMessage | None:
        delta_text = self._strip_delayed_reasoning_end_text(delta_text)
        state = self._stream_state
        if not state.prompt_reasoning_checked and prompt_token_ids is not None:
            state.prompt_reasoning_checked = True
            if self.is_reasoning_end(prompt_token_ids):
                state.reasoning_ended = True

        reasoning_delta = None
        if self._in_reasoning_phase(state) and self.is_reasoning_end(delta_token_ids):
            delayed_end = self._split_delayed_reasoning_end(delta_text)
            if delayed_end is not None:
                reasoning_delta, delta_text = delayed_end
            else:
                message = self.extract_reasoning_streaming(
                    previous_text=state.previous_text,
                    current_text=state.previous_text + delta_text,
                    delta_text=delta_text,
                    previous_token_ids=state.previous_token_ids,
                    current_token_ids=state.previous_token_ids + delta_token_ids,
                    delta_token_ids=delta_token_ids,
                )
                reasoning_delta = message.reasoning if message else ""
                if reasoning_delta is None:
                    reasoning_delta = ""

        message = super().parse_delta(
            delta_text,
            delta_token_ids,
            request,
            prompt_token_ids=prompt_token_ids,
        )
        if isinstance(request, ResponsesRequest):
            if reasoning_delta is None:
                return message
            if message is None:
                return DeltaMessage(reasoning=reasoning_delta)
            return message.model_copy(update={"reasoning": reasoning_delta})

        if not request.include_reasoning:
            if message is None:
                return None
            message = message.model_copy(update={"reasoning": None})
            if message.role is None and not message.content and not message.tool_calls:
                return None
            return message

        if reasoning_delta is None:
            if message is None or message.reasoning is None:
                return message
            return message.model_copy(
                update={
                    "reasoning_content": message.reasoning,
                }
            )
        if message is None:
            return DeltaMessage(
                reasoning=reasoning_delta,
                reasoning_content=reasoning_delta,
            )
        return message.model_copy(
            update={
                "reasoning": reasoning_delta,
                "reasoning_content": reasoning_delta,
            }
        )
