"""Test ClaudeClient"""

import pytest
from flexllm import ClaudeClient, LLMClient


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
        client = ClaudeClient(
            api_key="test-key", base_url="https://custom.anthropic.com/v1"
        )
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
        body = client._build_request_body(
            messages, "claude-3-5-sonnet-20241022", thinking=True
        )
        assert body["thinking"]["type"] == "enabled"
        assert body["thinking"]["budget_tokens"] == 10000

        # thinking=False
        body = client._build_request_body(
            messages, "claude-3-5-sonnet-20241022", thinking=False
        )
        assert body["thinking"]["type"] == "disabled"

        # thinking as int
        body = client._build_request_body(
            messages, "claude-3-5-sonnet-20241022", thinking=5000
        )
        assert body["thinking"]["budget_tokens"] == 5000

    def test_build_request_body_stream(self):
        """Test stream parameter"""
        client = ClaudeClient(api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]
        body = client._build_request_body(
            messages, "claude-3-5-sonnet-20241022", stream=True
        )

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
