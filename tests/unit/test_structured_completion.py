"""Regression tests for the stable structured completion contract."""

import json
from unittest.mock import MagicMock

import pytest

from flexllm import OpenAIClient


@pytest.mark.asyncio
async def test_cache_hit_checkpoint_preserves_structured_result(tmp_path):
    messages = [[{"role": "user", "content": "hello"}]]
    raw_response = {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}
    cached = {
        "content": "hello",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "reasoning_content": "reasoning",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ],
        "finish_reason": "stop",
        "raw_response": raw_response,
    }
    output = tmp_path / "results.jsonl"
    client = OpenAIClient(base_url="http://unused/v1", api_key="k", model="m")
    cache = MagicMock()
    cache.get_batch.return_value = ([cached], [])
    client._response_cache = cache

    first = await client.complete_batch(messages, output_jsonl=str(output), show_progress=False)
    record = json.loads(output.read_text().strip())
    assert record["result"]["raw_response"] == raw_response
    assert first[0].finish_reason == "stop"

    client._response_cache = None
    restored = await client.complete_batch(messages, output_jsonl=str(output), show_progress=False)
    assert restored[0].content == "hello"
    assert restored[0].reasoning_content == "reasoning"
    assert restored[0].finish_reason == "stop"
    assert restored[0].raw_response == raw_response
    assert restored[0].tool_calls[0].function["name"] == "lookup"
    await client.aclose()
