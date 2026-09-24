"""Test Function Calling / Tool Use support"""

import json

from flexllm import ClaudeClient, GeminiClient, OpenAIClient, ToolCall

from .test_extra_passthrough import ScriptedServer, _sse


class TestToolCallDataClass:
    """Test ToolCall dataclass"""

    def test_tool_call_creation(self):
        """Test creating a ToolCall instance"""
        tc = ToolCall(
            id="call_123",
            type="function",
            function={"name": "get_weather", "arguments": '{"location": "Tokyo"}'},
        )
        assert tc.id == "call_123"
        assert tc.type == "function"
        assert tc.function["name"] == "get_weather"

    def test_tool_call_from_dict(self):
        """Test ToolCall can hold any dict in function field"""
        tc = ToolCall(
            id="call_456",
            type="function",
            function={"name": "search", "arguments": json.dumps({"query": "test"})},
        )
        args = json.loads(tc.function["arguments"])
        assert args["query"] == "test"


class TestOpenAIToolCallExtraction:
    """Test OpenAI format tool_calls extraction"""

    def test_extract_tool_calls_single(self):
        """Test extracting single tool call"""
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Tokyo"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        client = OpenAIClient(base_url="http://localhost", model="test")
        tool_calls = client._extract_tool_calls(response_data)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_abc123"
        assert tool_calls[0].type == "function"
        assert tool_calls[0].function["name"] == "get_weather"

    def test_extract_tool_calls_multiple(self):
        """Test extracting multiple tool calls"""
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "func1", "arguments": "{}"},
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {"name": "func2", "arguments": "{}"},
                            },
                        ],
                    }
                }
            ]
        }
        client = OpenAIClient(base_url="http://localhost", model="test")
        tool_calls = client._extract_tool_calls(response_data)

        assert tool_calls is not None
        assert len(tool_calls) == 2
        assert tool_calls[0].function["name"] == "func1"
        assert tool_calls[1].function["name"] == "func2"

    def test_extract_tool_calls_none_when_missing(self):
        """Test returns None when no tool_calls"""
        response_data = {"choices": [{"message": {"content": "Hello", "tool_calls": None}}]}
        client = OpenAIClient(base_url="http://localhost", model="test")
        tool_calls = client._extract_tool_calls(response_data)
        assert tool_calls is None

    def test_extract_tool_calls_none_when_empty(self):
        """Test returns None when tool_calls is empty list"""
        response_data = {"choices": [{"message": {"content": "Hello", "tool_calls": []}}]}
        client = OpenAIClient(base_url="http://localhost", model="test")
        tool_calls = client._extract_tool_calls(response_data)
        assert tool_calls is None

    def test_extract_tool_calls_invalid_response(self):
        """Test handles invalid response gracefully"""
        client = OpenAIClient(base_url="http://localhost", model="test")
        assert client._extract_tool_calls({}) is None
        assert client._extract_tool_calls({"choices": []}) is None
        assert client._extract_tool_calls(None) is None


class TestGeminiToolCallExtraction:
    """Test Gemini format function call extraction"""

    def test_extract_function_call(self):
        """Test extracting Gemini functionCall"""
        response_data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "Tokyo"},
                                }
                            }
                        ]
                    }
                }
            ]
        }
        client = GeminiClient(api_key="test", model="test")
        tool_calls = client._extract_tool_calls(response_data)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].type == "function"
        assert tool_calls[0].function["name"] == "get_weather"
        # Arguments should be JSON string
        args = json.loads(tool_calls[0].function["arguments"])
        assert args["location"] == "Tokyo"

    def test_extract_function_call_none_when_missing(self):
        """Test returns None when no functionCall"""
        response_data = {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}
        client = GeminiClient(api_key="test", model="test")
        tool_calls = client._extract_tool_calls(response_data)
        assert tool_calls is None


class TestGeminiStreamToolCalls:
    """Gemini 流式曾经完全丢弃 functionCall part"""

    async def test_stream_yields_function_calls_with_distinct_indexes(self):
        def chunk(parts, finish=None):
            candidate = {"content": {"role": "model", "parts": parts}}
            if finish:
                candidate["finishReason"] = finish
            return _sse({"candidates": [candidate]})

        lines = [
            chunk([{"text": "checking"}]),
            chunk([{"functionCall": {"name": "get_weather", "args": {"city": "Tokyo"}}}]),
            chunk(
                [
                    {"functionCall": {"name": "get_time", "args": {}}},
                    {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}},
                ],
                finish="STOP",
            ),
        ]
        async with ScriptedServer(
            stream_lines=lines, path="/v1/models/m:streamGenerateContent"
        ) as server:
            client = GeminiClient(base_url=server.base_url, api_key="k", model="m")
            events = [
                e
                async for e in client.chat_completions_stream(
                    [{"role": "user", "content": "hi"}], return_usage=True
                )
            ]
            await client.aclose()

        calls = [tc for e in events if e["type"] == "tool_call_delta" for tc in e["tool_calls"]]
        assert [(tc["index"], tc["id"], tc["function"]["name"]) for tc in calls] == [
            (0, "call_0", "get_weather"),
            (1, "call_1", "get_time"),
            (2, "call_2", "get_weather"),
        ]
        assert json.loads(calls[2]["function"]["arguments"]) == {"city": "Paris"}
        assert [e["content"] for e in events if e["type"] == "content"] == ["checking"]


class TestClaudeToolCallExtraction:
    """Test Claude format tool_use extraction"""

    def test_extract_tool_use(self):
        """Test extracting Claude tool_use"""
        response_data = {
            "content": [
                {"type": "text", "text": "Let me check the weather."},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"location": "Tokyo"},
                },
            ]
        }
        client = ClaudeClient(api_key="test", model="test")
        tool_calls = client._extract_tool_calls(response_data)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "toolu_123"
        assert tool_calls[0].type == "function"
        assert tool_calls[0].function["name"] == "get_weather"
        args = json.loads(tool_calls[0].function["arguments"])
        assert args["location"] == "Tokyo"

    def test_extract_tool_use_none_when_missing(self):
        """Test returns None when no tool_use"""
        response_data = {"content": [{"type": "text", "text": "Hello"}]}
        client = ClaudeClient(api_key="test", model="test")
        tool_calls = client._extract_tool_calls(response_data)
        assert tool_calls is None
