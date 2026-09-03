"""Tests for provider-native thinking request parameters."""

import pytest

from flexllm.async_api.interface import RequestResult
from flexllm.clients import OpenAIClient


def test_structured_thinking_is_passed_through_without_boolean_compat_fields():
    client = OpenAIClient(base_url="https://api.deepseek.com/v1", api_key="test-key")
    messages = [{"role": "user", "content": "Hello"}]

    thinking = {"type": "enabled"}
    body = client._build_request_body(messages, "deepseek-v4", thinking=thinking)

    assert body["thinking"] == thinking
    assert "think" not in body
    assert "chat_template_kwargs" not in body


def test_boolean_thinking_keeps_openai_compatible_server_fields():
    client = OpenAIClient(base_url="http://localhost:8000/v1", api_key="test-key")
    messages = [{"role": "user", "content": "Hello"}]

    body = client._build_request_body(messages, "qwen3", thinking=True)

    assert body["think"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert "thinking" not in body


@pytest.mark.asyncio
async def test_deepseek_tool_round_preserves_reasoning_for_next_request(monkeypatch):
    client = OpenAIClient(
        base_url="https://api.deepseek.com/v1", api_key="test-key", model="deepseek-v4"
    )
    response_message = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "I need the current weather.",
        "tool_calls": [
            {
                "id": "call_weather",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Beijing"}'},
            }
        ],
    }
    payload = {
        "choices": [{"message": response_message, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8},
    }

    async def fake_process_requests(**kwargs):
        return [RequestResult(0, payload, "success", 0.0)], None

    monkeypatch.setattr(client._client, "process_requests", fake_process_requests)
    result = await client.chat_completions(
        [{"role": "user", "content": "Weather?"}], return_usage=True
    )

    assert result.reasoning_content == "I need the current weather."
    assert result.assistant_message == response_message
    next_body = client._build_request_body(
        [
            {"role": "user", "content": "Weather?"},
            result.assistant_message,
            {"role": "tool", "tool_call_id": "call_weather", "content": "Sunny"},
        ],
        "deepseek-v4",
        thinking={"type": "enabled"},
    )
    assert next_body["messages"][1]["reasoning_content"] == "I need the current weather."


def test_stream_continuation_uses_deepseek_reasoning_content_field():
    """The generic OpenAI stream aggregator emits DeepSeek's replay field."""
    # The end-to-end SSE aggregation is covered in test_finish_reason; this locks
    # down the wire field used by the continuation contract.
    client = OpenAIClient(base_url="https://api.deepseek.com/v1", api_key="test-key")
    body = client._build_request_body(
        [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "reasoning",
                "tool_calls": [],
            }
        ],
        "deepseek-v4",
    )
    assert body["messages"][0]["reasoning_content"] == "reasoning"
