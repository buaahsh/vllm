"""Agens tool parser, adapted from vLLM's GLM-4.7 tool parser."""
# Reason: merge same-index name and arguments emitted in one streaming delta.

from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser


class AgensToolParser(Glm47MoeModelToolParser):
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        message = super().extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )
        if message is None or len(message.tool_calls) < 2:
            return message

        merged = {}
        for tool_call in message.tool_calls:
            entry = merged.setdefault(
                tool_call.index,
                {"id": None, "type": None, "names": [], "arguments": []},
            )
            entry["id"] = entry["id"] or tool_call.id
            entry["type"] = entry["type"] or tool_call.type
            if tool_call.function is not None:
                if tool_call.function.name is not None:
                    entry["names"].append(tool_call.function.name)
                if tool_call.function.arguments is not None:
                    entry["arguments"].append(tool_call.function.arguments)

        tool_calls = []
        for index, entry in merged.items():
            function = None
            if entry["names"] or entry["arguments"]:
                function = DeltaFunctionCall(
                    name="".join(entry["names"]) if entry["names"] else None,
                    arguments=(
                        "".join(entry["arguments"])
                        if entry["arguments"]
                        else None
                    ),
                )
            tool_calls.append(
                DeltaToolCall(
                    index=index,
                    id=entry["id"],
                    type=entry["type"],
                    function=function,
                )
            )
        return message.model_copy(update={"tool_calls": tool_calls})