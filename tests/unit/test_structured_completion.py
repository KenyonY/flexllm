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
    assert first[0].finish_reason == "stop"
    # checkpoint 默认瘦身：content/usage 只存顶层，raw_response 不落盘
    assert "raw_response" not in record["result"]
    assert "content" not in record["result"]
    assert "usage" not in record["result"]
    assert record["output"] == "hello"
    assert record["usage"]["total_tokens"] == 2

    client._response_cache = None
    restored = await client.complete_batch(messages, output_jsonl=str(output), show_progress=False)
    assert restored[0].content == "hello"
    assert restored[0].usage["total_tokens"] == 2
    assert restored[0].reasoning_content == "reasoning"
    assert restored[0].finish_reason == "stop"
    assert restored[0].tool_calls[0].function["name"] == "lookup"
    assert restored[0].raw_response is None
    await client.aclose()


@pytest.mark.asyncio
async def test_save_raw_keeps_provider_response_in_checkpoint(tmp_path):
    messages = [[{"role": "user", "content": "hello"}]]
    raw_response = {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}
    cached = {"content": "hello", "finish_reason": "stop", "raw_response": raw_response}
    output = tmp_path / "results.jsonl"
    client = OpenAIClient(base_url="http://unused/v1", api_key="k", model="m")
    cache = MagicMock()
    cache.get_batch.return_value = ([cached], [])
    client._response_cache = cache

    await client.complete_batch(
        messages, output_jsonl=str(output), show_progress=False, save_raw=True
    )
    record = json.loads(output.read_text().strip())
    assert record["result"]["raw_response"] == raw_response

    client._response_cache = None
    restored = await client.complete_batch(
        messages, output_jsonl=str(output), show_progress=False, save_raw=True
    )
    assert restored[0].raw_response == raw_response
    await client.aclose()
