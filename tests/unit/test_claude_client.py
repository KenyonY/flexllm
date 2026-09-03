"""Test ClaudeClient"""

import json

import pytest

from flexllm import ClaudeClient, LLMClient
from flexllm.clients import claude as claude_mod


class _FakeStreamContent:
    def __init__(self, events):
        self._lines = [f"data: {json.dumps(event)}\n".encode() for event in events]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


class _FakeStreamResponse:
    status = 200

    def __init__(self, events):
        self.content = _FakeStreamContent(events)


class _FakeAsyncContext:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


class _FakeStreamSession:
    def __init__(self, events):
        self.events = events

    def post(self, *args, **kwargs):
        return _FakeAsyncContext(_FakeStreamResponse(self.events))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class TestClaudeClientInit:
    """Test ClaudeClient initialization"""

    def test_init_basic(self):
        """Test basic initialization"""
        client = ClaudeClient(api_key="test-key", model="claude-3-5-sonnet-20241022")
        assert client._api_key == "test-key"
        assert client._model == "claude-3-5-sonnet-20241022"

    def test_init_default_base_url(self):
        """Test default base URL"""
        client = ClaudeClient(api_key="test-key")
        assert client._base_url == "https://api.anthropic.com/v1"

    def test_init_custom_base_url(self):
        """Test custom base URL"""
        client = ClaudeClient(api_key="test-key", base_url="https://custom.anthropic.com/v1")
        assert client._base_url == "https://custom.anthropic.com/v1"

    def test_init_default_api_version(self):
        """Test default API version"""
        client = ClaudeClient(api_key="test-key")
        assert client._api_version == "2023-06-01"


class TestClaudeClientHeaders:
    """Test ClaudeClient header generation"""

    def test_get_headers(self):
        """Test header generation"""
        client = ClaudeClient(api_key="test-key")
        headers = client._get_headers()

        assert headers["Content-Type"] == "application/json"
        assert headers["x-api-key"] == "test-key"
        assert headers["anthropic-version"] == "2023-06-01"


class TestClaudeClientUrl:
    """Test ClaudeClient URL generation"""

    def test_get_url(self):
        """Test URL generation"""
        client = ClaudeClient(api_key="test-key")
        url = client._get_url("claude-3-5-sonnet-20241022")
        assert url == "https://api.anthropic.com/v1/messages"

    def test_get_url_stream(self):
        """Test stream URL is same as non-stream"""
        client = ClaudeClient(api_key="test-key")
        url = client._get_url("claude-3-5-sonnet-20241022", stream=True)
        assert url == "https://api.anthropic.com/v1/messages"


class TestClaudeClientRequestBody:
    """Test ClaudeClient request body building"""

    def test_build_request_body_basic(self):
        """Test basic request body"""
        client = ClaudeClient(api_key="test-key", model="claude-3-5-sonnet-20241022")
        messages = [{"role": "user", "content": "Hello"}]
        body = client._build_request_body(messages, "claude-3-5-sonnet-20241022")

        assert body["model"] == "claude-3-5-sonnet-20241022"
        assert body["max_tokens"] == 4096  # default
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"] == "Hello"

    def test_build_request_body_with_system(self):
        """Test request body with system message"""
        client = ClaudeClient(api_key="test-key")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        body = client._build_request_body(messages, "claude-3-5-sonnet-20241022")

        assert body["system"] == "You are helpful."
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"

    def test_build_request_body_multiple_system_messages(self):
        """Test multiple system messages are merged"""
        client = ClaudeClient(api_key="test-key")
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ]
        body = client._build_request_body(messages, "claude-3-5-sonnet-20241022")

        assert "Be concise." in body["system"]
        assert "Be helpful." in body["system"]

    def test_build_request_body_with_thinking(self):
        """Test thinking parameter"""
        client = ClaudeClient(api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        # thinking=True
        body = client._build_request_body(messages, "claude-3-7-sonnet-20250219", thinking=True)
        assert body["thinking"]["type"] == "enabled"
        assert body["thinking"]["budget_tokens"] == 10000

        # thinking=False
        body = client._build_request_body(messages, "claude-3-7-sonnet-20250219", thinking=False)
        assert body["thinking"]["type"] == "disabled"

        # thinking as int
        body = client._build_request_body(messages, "claude-3-7-sonnet-20250219", thinking=5000)
        assert body["thinking"]["budget_tokens"] == 5000

    def test_claude_35_rejects_thinking_controls_but_allows_omission(self):
        client = ClaudeClient(api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        body = client._build_request_body(messages, "claude-3-5-sonnet-20241022", thinking=False)
        assert "thinking" not in body
        with pytest.raises(ValueError, match="does not support extended thinking"):
            client._build_request_body(
                messages, "claude-3-5-sonnet-20241022", reasoning_effort="low"
            )

    @pytest.mark.parametrize("model", ["claude-3-7-sonnet-20250219", "claude-sonnet-4-5"])
    def test_manual_thinking_versions_use_token_budgets(self, model):
        client = ClaudeClient(api_key="test-key")
        body = client._build_request_body(
            [{"role": "user", "content": "Hello"}], model, reasoning_effort="low"
        )

        assert body["thinking"] == {"type": "enabled", "budget_tokens": 4000}

    @pytest.mark.parametrize("model", ["claude-sonnet-4-6", "claude-opus-4-7", "claude-sonnet-5"])
    def test_adaptive_thinking_versions_use_output_effort(self, model):
        client = ClaudeClient(api_key="test-key")
        body = client._build_request_body(
            [{"role": "user", "content": "Hello"}], model, reasoning_effort="low"
        )

        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"] == {"effort": "low"}

    def test_claude_46_allows_manual_budget_but_47_and_5_reject_it(self):
        client = ClaudeClient(api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        body = client._build_request_body(messages, "claude-sonnet-4-6", thinking=5000)
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 5000}
        for model in ("claude-opus-4-7", "claude-sonnet-5"):
            with pytest.raises(ValueError, match="requires adaptive thinking"):
                client._build_request_body(messages, model, thinking=5000)

    def test_newer_adaptive_models_preserve_xhigh_but_46_maps_it_to_max(self):
        client = ClaudeClient(api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        body_46 = client._build_request_body(
            messages, "claude-sonnet-4-6", reasoning_effort="xhigh"
        )
        body_47 = client._build_request_body(messages, "claude-opus-4-7", reasoning_effort="xhigh")

        assert body_46["output_config"]["effort"] == "max"
        assert body_47["output_config"]["effort"] == "xhigh"

    def test_always_on_adaptive_models_reject_disable(self):
        client = ClaudeClient(api_key="test-key")
        with pytest.raises(ValueError, match="always-on adaptive thinking"):
            client._build_request_body(
                [{"role": "user", "content": "Hello"}],
                "claude-fable-5",
                thinking=False,
            )

    def test_current_claude_maps_effort_to_adaptive_thinking(self):
        client = ClaudeClient(api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        body = client._build_request_body(messages, "claude-sonnet-4-6", reasoning_effort="low")

        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"] == {"effort": "low"}
        assert "reasoning_effort" not in body

    def test_current_claude_maps_thinking_strength_to_adaptive_effort(self):
        client = ClaudeClient(api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        body = client._build_request_body(messages, "claude-opus-4-7", thinking="xhigh")

        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"] == {"effort": "xhigh"}

    def test_legacy_claude_maps_effort_to_token_budget(self):
        client = ClaudeClient(api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        body = client._build_request_body(
            messages,
            "claude-sonnet-4-5-20250929",
            max_tokens=2000,
            reasoning_effort="low",
        )

        assert body["thinking"] == {"type": "enabled", "budget_tokens": 4000}
        assert body["max_tokens"] == 8096
        assert "reasoning_effort" not in body

    def test_native_thinking_config_is_preserved(self):
        client = ClaudeClient(api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        body = client._build_request_body(
            messages,
            "claude-sonnet-4-6",
            thinking={"type": "adaptive"},
            reasoning_effort="medium",
            output_config={"effort": "high", "format": {"type": "json_schema"}},
        )

        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"] == {
            "effort": "high",
            "format": {"type": "json_schema"},
        }

    def test_explicit_disabled_thinking_wins_over_reasoning_effort(self):
        client = ClaudeClient(api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        body = client._build_request_body(
            messages, "claude-sonnet-4-6", thinking=False, reasoning_effort="high"
        )

        assert body["thinking"] == {"type": "disabled"}
        assert "output_config" not in body

    def test_non_claude_anthropic_endpoint_does_not_apply_claude_budget_mapping(self):
        client = ClaudeClient(api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        body = client._build_request_body(messages, "vendor-model", reasoning_effort="low")

        assert body["reasoning_effort"] == "low"
        assert "thinking" not in body

    def test_build_request_body_stream(self):
        """Test stream parameter"""
        client = ClaudeClient(api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]
        body = client._build_request_body(messages, "claude-3-5-sonnet-20241022", stream=True)

        assert body["stream"] is True


class TestClaudeClientExtractContent:
    """Test ClaudeClient content extraction"""

    def test_extract_content_single_text(self):
        """Test extracting single text block"""
        client = ClaudeClient(api_key="test-key")
        response_data = {"content": [{"type": "text", "text": "Hello, world!"}]}
        content = client._extract_content(response_data)
        assert content == "Hello, world!"

    def test_extract_content_multiple_text(self):
        """Test extracting multiple text blocks"""
        client = ClaudeClient(api_key="test-key")
        response_data = {
            "content": [
                {"type": "text", "text": "Hello, "},
                {"type": "text", "text": "world!"},
            ]
        }
        content = client._extract_content(response_data)
        assert content == "Hello, world!"

    def test_extract_content_with_tool_use(self):
        """Test extracting content when tool_use is present"""
        client = ClaudeClient(api_key="test-key")
        response_data = {
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_123", "name": "search", "input": {}},
            ]
        }
        content = client._extract_content(response_data)
        assert content == "Let me check."

    def test_extract_content_empty(self):
        """Test extracting from empty content"""
        client = ClaudeClient(api_key="test-key")
        assert client._extract_content({"content": []}) is None
        assert client._extract_content({}) is None

    @pytest.mark.asyncio
    async def test_tool_round_preserves_signed_thinking_for_next_request(self, monkeypatch):
        from flexllm.async_api.interface import RequestResult

        client = ClaudeClient(api_key="test-key", model="claude-sonnet-4-6")
        content = [
            {"type": "thinking", "thinking": "I should inspect it.", "signature": "sig_abc"},
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "inspect",
                "input": {"path": "README.md"},
            },
        ]
        payload = {
            "content": content,
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        async def fake_process_requests(**kwargs):
            return [RequestResult(0, payload, "success", 0.0)], None

        monkeypatch.setattr(client._client, "process_requests", fake_process_requests)
        result = await client.chat_completions(
            [{"role": "user", "content": "Inspect the README"}], return_usage=True
        )

        assert result.reasoning_content == "I should inspect it."
        assert result.assistant_message["content"] == content
        assert result.assistant_message["tool_calls"][0]["id"] == "toolu_123"

        next_body = client._build_request_body(
            [
                {"role": "user", "content": "Inspect the README"},
                result.assistant_message,
                {"role": "tool", "tool_call_id": "toolu_123", "content": "contents"},
            ],
            "claude-sonnet-4-6",
            reasoning_effort="low",
        )
        assert next_body["messages"][1] == {"role": "assistant", "content": content}

    @pytest.mark.asyncio
    async def test_stream_preserves_signed_thinking_for_next_request(self, monkeypatch):
        events = [
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 10, "output_tokens": 0}},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Inspect first."},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig_stream"},
            },
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_stream",
                    "name": "inspect",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"path":"README.md"}'},
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 8},
            },
            {"type": "message_stop"},
        ]
        session = _FakeStreamSession(events)
        monkeypatch.setattr(claude_mod, "create_proxied_session", lambda proxy: (session, {}))
        client = ClaudeClient(api_key="test-key", model="claude-sonnet-4-6")

        chunks = [
            chunk
            async for chunk in client.chat_completions_stream(
                [{"role": "user", "content": "Inspect"}],
                return_usage=True,
                reasoning_effort="low",
            )
        ]
        continuation = next(
            chunk["message"] for chunk in chunks if chunk["type"] == "assistant_message"
        )

        assert continuation["content"] == [
            {
                "type": "thinking",
                "thinking": "Inspect first.",
                "signature": "sig_stream",
            },
            {
                "type": "tool_use",
                "id": "toolu_stream",
                "name": "inspect",
                "input": {"path": "README.md"},
            },
        ]
        next_body = client._build_request_body(
            [
                continuation,
                {"role": "tool", "tool_call_id": "toolu_stream", "content": "contents"},
            ],
            "claude-sonnet-4-6",
            reasoning_effort="low",
        )
        assert next_body["messages"][0]["content"] == continuation["content"]


class TestClaudeClientExtractUsage:
    """Test ClaudeClient usage extraction"""

    def test_extract_usage(self):
        """Test extracting usage info"""
        client = ClaudeClient(api_key="test-key")
        response_data = {"usage": {"input_tokens": 100, "output_tokens": 50}}
        usage = client._extract_usage(response_data)

        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150

    def test_extract_usage_none(self):
        """Test returns None when no usage"""
        client = ClaudeClient(api_key="test-key")
        assert client._extract_usage({}) is None
        assert client._extract_usage(None) is None


class TestClaudeClientParseThoughts:
    """Test ClaudeClient parse_thoughts"""

    def test_parse_thoughts_with_thinking(self):
        """Test parsing response with thinking blocks"""
        response_data = {
            "content": [
                {"type": "thinking", "thinking": "Let me think about this..."},
                {"type": "text", "text": "The answer is 42."},
            ]
        }
        parsed = ClaudeClient.parse_thoughts(response_data)

        assert parsed["thought"] == "Let me think about this..."
        assert parsed["answer"] == "The answer is 42."

    def test_parse_thoughts_without_thinking(self):
        """Test parsing response without thinking"""
        response_data = {"content": [{"type": "text", "text": "Hello!"}]}
        parsed = ClaudeClient.parse_thoughts(response_data)

        assert parsed["thought"] == ""
        assert parsed["answer"] == "Hello!"


class TestLLMClientClaudeProvider:
    """Test LLMClient with Claude provider"""

    def test_infer_provider_anthropic(self):
        """Test provider inference for anthropic.com"""
        provider = LLMClient._infer_provider("https://api.anthropic.com/v1", False)
        assert provider == "claude"

    def test_llm_client_claude_init(self):
        """Test LLMClient initialization with Claude provider"""
        client = LLMClient(
            provider="claude",
            api_key="test-key",
            model="claude-3-5-sonnet-20241022",
        )
        assert client._provider == "claude"
        assert isinstance(client._client, ClaudeClient)

    def test_llm_client_claude_requires_api_key(self):
        """Test Claude provider requires api_key"""
        with pytest.raises(ValueError, match="api_key"):
            LLMClient(provider="claude", model="claude-3-5-sonnet-20241022")


class TestClaudeClientModelList:
    """Test ClaudeClient model list"""

    def test_model_list(self):
        """Test model list returns valid models"""
        client = ClaudeClient(api_key="test-key")
        models = client.model_list()

        assert isinstance(models, list)
        assert len(models) > 0
        assert "claude-3-5-sonnet-20241022" in models
