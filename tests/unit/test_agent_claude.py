"""Test ClaudeClient: _convert_message, _build_json_instruction, response_format."""

import json

import pytest

from flexllm import ClaudeClient


class TestClaudeConvertToolResult:
    """Test _convert_message handles tool result messages correctly."""

    def test_tool_result_message(self):
        """Test tool result message is converted to Claude tool_result format."""
        client = ClaudeClient(api_key="test-key")
        msg = {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "The weather is sunny.",
        }
        result = client._convert_message(msg)

        assert result["role"] == "user"
        assert len(result["content"]) == 1
        block = result["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "call_123"
        assert block["content"] == "The weather is sunny."

    def test_tool_result_message_empty_content(self):
        """Test tool result with empty content."""
        client = ClaudeClient(api_key="test-key")
        msg = {
            "role": "tool",
            "tool_call_id": "call_456",
            "content": "",
        }
        result = client._convert_message(msg)

        assert result["role"] == "user"
        assert result["content"][0]["type"] == "tool_result"
        assert result["content"][0]["content"] == ""

    def test_tool_result_message_missing_tool_call_id(self):
        """Test tool result without tool_call_id defaults to empty string."""
        client = ClaudeClient(api_key="test-key")
        msg = {"role": "tool", "content": "result"}
        result = client._convert_message(msg)

        assert result["content"][0]["tool_use_id"] == ""


class TestClaudeConvertAssistantToolCalls:
    """Test _convert_message handles assistant messages with tool_calls."""

    def test_assistant_with_tool_calls(self):
        """Test assistant message with tool_calls is converted to Claude tool_use."""
        client = ClaudeClient(api_key="test-key")
        msg = {
            "role": "assistant",
            "content": "Let me check the weather.",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "Beijing"}),
                    },
                }
            ],
        }
        result = client._convert_message(msg)

        assert result["role"] == "assistant"
        assert len(result["content"]) == 2

        # First block: text
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Let me check the weather."

        # Second block: tool_use
        tool_use = result["content"][1]
        assert tool_use["type"] == "tool_use"
        assert tool_use["id"] == "call_abc"
        assert tool_use["name"] == "get_weather"
        assert tool_use["input"] == {"city": "Beijing"}

    def test_assistant_with_tool_calls_no_text(self):
        """Test assistant message with tool_calls but no text content."""
        client = ClaudeClient(api_key="test-key")
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_xyz",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": json.dumps({"query": "test"}),
                    },
                }
            ],
        }
        result = client._convert_message(msg)

        assert result["role"] == "assistant"
        # Empty content should not generate text block
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "tool_use"

    def test_assistant_with_multiple_tool_calls(self):
        """Test assistant message with multiple tool_calls."""
        client = ClaudeClient(api_key="test-key")
        msg = {
            "role": "assistant",
            "content": "I'll check both.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": '{"timezone": "UTC"}'},
                },
            ],
        }
        result = client._convert_message(msg)

        assert len(result["content"]) == 3  # 1 text + 2 tool_use
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "tool_use"
        assert result["content"][1]["name"] == "get_weather"
        assert result["content"][2]["type"] == "tool_use"
        assert result["content"][2]["name"] == "get_time"

    def test_assistant_with_dict_arguments(self):
        """Test assistant tool_calls where arguments is already a dict."""
        client = ClaudeClient(api_key="test-key")
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_d",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": {"query": "test"},
                    },
                }
            ],
        }
        result = client._convert_message(msg)

        tool_use = result["content"][0]
        assert tool_use["input"] == {"query": "test"}


class TestClaudeConvertRegularMessages:
    """Ensure regular messages still work correctly after tool support."""

    def test_user_message(self):
        """Test regular user message is unchanged."""
        client = ClaudeClient(api_key="test-key")
        msg = {"role": "user", "content": "Hello"}
        result = client._convert_message(msg)

        assert result["role"] == "user"
        assert result["content"] == "Hello"

    def test_assistant_message_without_tool_calls(self):
        """Test regular assistant message without tool_calls."""
        client = ClaudeClient(api_key="test-key")
        msg = {"role": "assistant", "content": "Hi there!"}
        result = client._convert_message(msg)

        assert result["role"] == "assistant"
        assert result["content"] == "Hi there!"


class TestClaudeBuildRequestBodyWithTools:
    """Test _build_request_body handles tool messages in conversation."""

    def test_full_tool_conversation(self):
        """Test building request body with a full tool-use conversation."""
        client = ClaudeClient(api_key="test-key")
        messages = [
            {"role": "user", "content": "What's the weather in Beijing?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Beijing"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "Sunny, 25°C",
            },
            {"role": "user", "content": "Thanks!"},
        ]

        body = client._build_request_body(messages, "claude-3-5-sonnet-20241022")

        assert len(body["messages"]) == 4
        # First: user
        assert body["messages"][0]["role"] == "user"
        # Second: assistant with tool_use
        assert body["messages"][1]["role"] == "assistant"
        assert body["messages"][1]["content"][0]["type"] == "text"
        assert body["messages"][1]["content"][1]["type"] == "tool_use"
        # Third: tool_result
        assert body["messages"][2]["role"] == "user"
        assert body["messages"][2]["content"][0]["type"] == "tool_result"
        # Fourth: user
        assert body["messages"][3]["role"] == "user"


class TestClaudeBuildJsonInstruction:
    """测试 _build_json_instruction"""

    def test_json_object(self):
        """type=json_object 返回 JSON 指令"""
        result = ClaudeClient._build_json_instruction({"type": "json_object"})
        assert result is not None
        assert "JSON" in result

    def test_json_schema_with_schema(self):
        """type=json_schema 有 schema 时返回包含 schema 的指令"""
        fmt = {
            "type": "json_schema",
            "json_schema": {
                "name": "Person",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                },
            },
        }
        result = ClaudeClient._build_json_instruction(fmt)
        assert result is not None
        assert "name" in result
        assert "age" in result

    def test_json_schema_without_schema(self):
        """type=json_schema 无 schema 时返回 None"""
        fmt = {"type": "json_schema", "json_schema": {}}
        result = ClaudeClient._build_json_instruction(fmt)
        assert result is None

    def test_unknown_type(self):
        """未知 type 返回 None"""
        result = ClaudeClient._build_json_instruction({"type": "text"})
        assert result is None

    def test_empty_dict(self):
        """空 dict 返回 None"""
        result = ClaudeClient._build_json_instruction({})
        assert result is None


class TestClaudeResponseFormatInjection:
    """测试 response_format 通过 system prompt 注入"""

    def test_creates_system_when_none(self):
        """无 system message 时 response_format 创建 system"""
        client = ClaudeClient(api_key="test-key")
        messages = [{"role": "user", "content": "输出 JSON"}]
        body = client._build_request_body(
            messages, "claude-3-5-sonnet-20241022", response_format={"type": "json_object"}
        )
        assert "system" in body
        assert "JSON" in body["system"]

    def test_merges_with_existing_system(self):
        """有 system message 时 response_format 与之合并"""
        client = ClaudeClient(api_key="test-key")
        messages = [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "输出 JSON"},
        ]
        body = client._build_request_body(
            messages, "claude-3-5-sonnet-20241022", response_format={"type": "json_object"}
        )
        assert "你是助手" in body["system"]
        assert "JSON" in body["system"]

    def test_not_leaked_to_body(self):
        """response_format 不应出现在请求 body 中"""
        client = ClaudeClient(api_key="test-key")
        messages = [{"role": "user", "content": "test"}]
        body = client._build_request_body(
            messages, "claude-3-5-sonnet-20241022", response_format={"type": "json_object"}
        )
        assert "response_format" not in body
