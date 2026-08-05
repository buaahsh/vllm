# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionStreamResponse,
)
from vllm.entrypoints.openai.completion.protocol import CompletionStreamResponse


@pytest.mark.parametrize(
    "response_cls",
    [ChatCompletionStreamResponse, CompletionStreamResponse],
)
def test_streaming_response_serializes_kv_transfer_params(response_cls):
    kv_transfer_params = {
        "remote_engine_id": "decode-engine",
        "remote_request_id": "decode-request",
        "remote_block_ids": [[1, 2]],
    }
    response = response_cls(
        model="test-model",
        choices=[],
        kv_transfer_params=kv_transfer_params,
    )

    assert response.model_dump(exclude_none=True)["kv_transfer_params"] == (
        kv_transfer_params
    )
