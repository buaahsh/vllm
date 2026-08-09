# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatMessage,
)
from vllm.entrypoints.openai.chat_completion.serving import (
    _convert_reasoning_output_field,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.streaming_events import (
    SimpleStreamingEventProcessor,
    _StateType,
)
from vllm.outputs import CompletionOutput
from vllm.parser.abstract_parser import DelegatingParser, StreamState
from vllm.parser.agens_parser import AgensParser
from vllm.reasoning.agens_reasoning_parser import AgensReasoningParser
from vllm.tool_parsers.agens_tool_parser import AgensToolParser
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser


def _make_agens_parser() -> AgensParser:
    parser = object.__new__(AgensParser)
    parser._stream_state = StreamState()
    parser._reasoning_parser = None
    parser._tool_parser = None
    return parser


def test_chat_completion_preserves_reasoning_fields(monkeypatch):
    monkeypatch.setattr(
        DelegatingParser,
        "parse_delta",
        lambda *args, **kwargs: DeltaMessage(reasoning="thinking"),
    )
    request = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])

    result = _make_agens_parser().parse_delta("", [], request)

    assert result is not None
    assert result.reasoning == "thinking"
    assert result.reasoning_content == "thinking"


def test_responses_api_keeps_reasoning(monkeypatch):
    monkeypatch.setattr(
        DelegatingParser,
        "parse_delta",
        lambda *args, **kwargs: DeltaMessage(reasoning="thinking"),
    )

    result = _make_agens_parser().parse_delta("", [], ResponsesRequest(input="hi"))

    assert result is not None
    assert result.reasoning == "thinking"
    assert not hasattr(result, "reasoning_content")


def test_chat_completion_can_exclude_reasoning(monkeypatch):
    monkeypatch.setattr(
        DelegatingParser,
        "parse_delta",
        lambda *args, **kwargs: DeltaMessage(reasoning="thinking", content="answer"),
    )
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        include_reasoning=False,
    )

    result = _make_agens_parser().parse_delta("", [], request)

    assert result is not None
    assert result.reasoning is None
    assert result.content == "answer"
    assert not hasattr(result, "reasoning_content")


def test_non_streaming_chat_uses_reasoning_content():
    parser = object.__new__(AgensReasoningParser)

    result = _convert_reasoning_output_field(
        ChatMessage(role="assistant", reasoning="thinking"), parser
    )

    assert result.reasoning is None
    assert result.reasoning_content == "thinking"


def test_agens_reasoning_parser_declares_text_end_marker():
    assert AgensReasoningParser.additional_stop_strings == ("<|end|>",)


def test_streaming_strips_delayed_reasoning_end_text():
    class FakeReasoningParser:
        reasoning_end_str = "</think>"

    parser = _make_agens_parser()
    parser._reasoning_parser = FakeReasoningParser()
    parser._pending_reasoning_end_text = None

    assert parser._split_delayed_reasoning_end('answer"</') == (
        'answer"',
        "</think>",
    )
    assert parser._strip_delayed_reasoning_end_text("t") == ""
    assert parser._strip_delayed_reasoning_end_text("hink>") == ""
    assert parser._strip_delayed_reasoning_end_text("\nHello") == "\nHello"


def test_responses_emits_compound_reasoning_and_tool_delta():
    processor = SimpleStreamingEventProcessor()
    delta = DeltaMessage(
        reasoning="thinking",
        content="answer",
        tool_calls=[
            DeltaToolCall(
                index=0,
                id="call-0",
                type="function",
                function=DeltaFunctionCall(
                    name="get_weather", arguments='{"city":"Seattle"}'
                ),
            ),
            DeltaToolCall(
                index=1,
                id="call-1",
                type="function",
                function=DeltaFunctionCall(name="get_time", arguments="{}"),
            ),
        ],
    )
    output = CompletionOutput(
        index=0,
        text="",
        token_ids=[],
        cumulative_logprob=0.0,
        logprobs=None,
        finish_reason=None,
        stop_reason=None,
    )

    target_state, tool_call = processor.resolve_target_state(delta)
    assert target_state == _StateType.REASONING
    assert tool_call is None

    events = processor.open(target_state)
    events.extend(processor.emit_delta(delta, output))

    event_types = [event.type for event in events]
    assert "response.reasoning_text.delta" in event_types
    assert "response.output_text.delta" in event_types
    assert event_types.count("response.output_item.added") == 4
    assert event_types.count("response.function_call_arguments.delta") == 2
    assert event_types.index("response.reasoning_text.delta") < event_types.index(
        "response.output_text.delta"
    )
    assert event_types.index("response.output_text.delta") < event_types.index(
        "response.function_call_arguments.delta"
    )
    assert processor.state.current_state == _StateType.TOOL_CALL
    assert processor.state.tool_call_index == 1


def test_tool_parser_merges_same_index_deltas(monkeypatch):
    message = DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=0,
                id="call-0",
                type="function",
                function=DeltaFunctionCall(name="get_"),
            ),
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(name="weather", arguments='{"city":'),
            ),
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(arguments='"Seattle"}'),
            ),
            DeltaToolCall(
                index=1,
                id="call-1",
                type="function",
                function=DeltaFunctionCall(name="get_time", arguments="{}"),
            ),
        ]
    )
    monkeypatch.setattr(
        Glm47MoeModelToolParser,
        "extract_tool_calls_streaming",
        lambda *args, **kwargs: message,
    )

    result = object.__new__(AgensToolParser).extract_tool_calls_streaming(
        previous_text="",
        current_text="",
        delta_text="",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}]),
    )

    assert result is not None
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].function is not None
    assert result.tool_calls[0].function.name == "get_weather"
    assert result.tool_calls[0].function.arguments == '{"city":"Seattle"}'
    assert result.tool_calls[1].function is not None
    assert result.tool_calls[1].function.name == "get_time"
