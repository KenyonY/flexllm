"""Gemini 工具调用全链路：请求转换、响应提取、流式拼回

协议细节均以真实 API（gemini-3-flash-preview）实测为准：
- functionCall 自带 id，第一个 functionCall part 带 thoughtSignature，缺了下一轮 400
- functionResponse.response 必须是对象，纯字符串 400
- tools 的 parameters 不收 additionalProperties，parametersJsonSchema 收完整 JSON Schema
- 工具调用时 finishReason 仍是 STOP；usage 的 candidatesTokenCount 不含思考 token
"""

import json

import pytest

from flexllm import GeminiClient
from flexllm.clients.gemini import _SKIP_SIGNATURE

from .test_extra_passthrough import ScriptedServer, _sse

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "additionalProperties": False,
        },
    },
}


def _client():
    return GeminiClient(api_key="k", model="m")


def _response(parts, finish="STOP", usage=None):
    data = {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}]}
    if usage:
        data["usageMetadata"] = usage
    return data


CALL_PARTS = [
    {"text": "想一想", "thought": True},
    {
        "functionCall": {"name": "get_weather", "args": {"city": "东京"}, "id": "g1"},
        "thoughtSignature": "SIG",
    },
    {"functionCall": {"name": "get_weather", "args": {"city": "巴黎"}, "id": "g2"}},
]


class TestRequestConversion:
    def test_openai_tools_become_function_declarations(self):
        native = {"googleSearch": {}}
        body = _client()._build_request_body(
            [{"role": "user", "content": "hi"}], "m", tools=[WEATHER, native]
        )
        assert body["tools"] == [
            {
                "functionDeclarations": [
                    {
                        "name": "get_weather",
                        "description": "查天气",
                        "parametersJsonSchema": WEATHER["function"]["parameters"],
                    }
                ]
            },
            native,
        ]

    @pytest.mark.parametrize(
        "choice,expected",
        [
            ("auto", {"mode": "AUTO"}),
            ("none", {"mode": "NONE"}),
            ("required", {"mode": "ANY"}),
            (
                {"type": "function", "function": {"name": "get_weather"}},
                {"mode": "ANY", "allowedFunctionNames": ["get_weather"]},
            ),
        ],
    )
    def test_tool_choice(self, choice, expected):
        body = _client()._build_request_body(
            [{"role": "user", "content": "hi"}], "m", tool_choice=choice
        )
        assert body["toolConfig"] == {"functionCallingConfig": expected}

    def test_unknown_tool_choice_raises(self):
        with pytest.raises(ValueError):
            _client()._build_request_body(
                [{"role": "user", "content": "hi"}], "m", tool_choice="any"
            )

    def test_openai_history_is_rebuilt_with_placeholder_signature(self):
        msgs = [
            {"role": "system", "content": "a"},
            {"role": "system", "content": "b"},
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"x": 1}'},
                    },
                    {"id": "c2", "type": "function", "function": {"name": "g", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "tool", "tool_call_id": "c2", "content": [{"type": "text", "text": "r2"}]},
        ]
        contents, system = _client()._convert_messages_to_contents(msgs)

        assert system == {"parts": [{"text": "a\n\nb"}]}
        assert contents[1] == {
            "role": "model",
            "parts": [
                {
                    "functionCall": {"name": "f", "args": {"x": 1}, "id": "c1"},
                    "thoughtSignature": _SKIP_SIGNATURE,
                },
                {"functionCall": {"name": "g", "args": {}, "id": "c2"}},
            ],
        }
        # 同一步的两条结果并进一条 user 消息，response 必须是对象
        assert contents[2] == {
            "role": "user",
            "parts": [
                {"functionResponse": {"name": "f", "id": "c1", "response": {"result": "r1"}}},
                {"functionResponse": {"name": "g", "id": "c2", "response": {"result": "r2"}}},
            ],
        }

    def test_native_assistant_message_round_trips_verbatim(self):
        message = _client()._extract_assistant_message(_response(CALL_PARTS))
        msgs = [
            {"role": "user", "content": "q"},
            message,
            {"role": "tool", "tool_call_id": "g1", "content": "晴"},
            {"role": "tool", "tool_call_id": "g2", "content": "雨"},
        ]
        contents, _ = _client()._convert_messages_to_contents(msgs)

        assert contents[1]["parts"] == CALL_PARTS  # 签名逐字节保留，不插占位签名
        assert [p["functionResponse"]["name"] for p in contents[2]["parts"]] == ["get_weather"] * 2

    def test_repeated_synthetic_ids_match_the_nearest_call(self):
        """老模型没有 id 时每轮都是 call_0，tool 消息必须配到最近那次调用"""

        def turn(name):
            call = {
                "id": "call_0",
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
            return [
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "call_0", "content": "ok"},
            ]

        msgs = [{"role": "user", "content": "q"}, *turn("first"), *turn("second")]
        contents, _ = _client()._convert_messages_to_contents(msgs)
        assert contents[-1]["parts"][0]["functionResponse"]["name"] == "second"

    def test_orphan_tool_result_raises(self):
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "tool", "tool_call_id": "x", "content": "r"},
        ]
        with pytest.raises(ValueError, match="tool_call_id"):
            _client()._convert_messages_to_contents(msgs)

    def test_remote_media_forces_preprocessing(self):
        remote = [{"type": "image_url", "image_url": {"url": "https://x/a.png"}}]
        inline = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]
        assert GeminiClient._has_remote_media([{"role": "user", "content": remote}])
        assert not GeminiClient._has_remote_media([{"role": "user", "content": inline}])


class TestResponseExtraction:
    def test_tool_call_response(self):
        client = _client()
        data = _response(
            CALL_PARTS,
            usage={
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "thoughtsTokenCount": 20,
                "totalTokenCount": 35,
            },
        )
        calls = client._extract_tool_calls(data)
        assert [c.id for c in calls] == ["g1", "g2"]  # 真实 id 优先
        assert client._extract_finish_reason(data) == "tool_calls"
        assert client._extract_reasoning_content(data) == "想一想"
        assert client._extract_usage(data) == {
            "prompt_tokens": 10,
            "completion_tokens": 25,  # 思考按输出计
            "total_tokens": 35,
            "completion_tokens_details": {"reasoning_tokens": 20},
        }
        message = client._extract_assistant_message(data)
        assert message["content"] == CALL_PARTS
        assert [c["id"] for c in message["tool_calls"]] == ["g1", "g2"]

    def test_missing_id_falls_back_to_part_index(self):
        data = _response([{"text": "x"}, {"functionCall": {"name": "f", "args": {}}}])
        assert _client()._extract_tool_calls(data)[0].id == "call_1"

    def test_plain_text_needs_no_continuation(self):
        data = _response([{"text": "hi"}])
        assert _client()._extract_assistant_message(data) is None
        assert _client()._extract_finish_reason(data) == "stop"


class TestStream:
    async def _events(self, chunks):
        lines = [_sse(c) for c in chunks]
        async with ScriptedServer(
            stream_lines=lines, path="/v1/models/m:streamGenerateContent"
        ) as server:
            client = GeminiClient(base_url=server.base_url, api_key="k", model="m")
            events = [e async for e in client.complete_stream([{"role": "user", "content": "q"}])]
            await client.aclose()
        return events

    async def test_tool_calls_and_signature_survive_streaming(self):
        events = await self._events(
            [
                _response([CALL_PARTS[1]], finish=None),
                _response([CALL_PARTS[2]], finish=None),
                _response(
                    [{"text": ""}],
                    usage={"promptTokenCount": 1, "candidatesTokenCount": 2, "totalTokenCount": 3},
                ),
            ]
        )
        result = events[-1]["result"]
        assert [c.id for c in result.tool_calls] == ["g1", "g2"]
        assert json.loads(result.tool_calls[0].function["arguments"]) == {"city": "东京"}
        assert result.finish_reason == "tool_calls"
        # 拼回的 parts 与非流式等价：签名在第一个 functionCall 上
        assert result.assistant_message["content"][:2] == CALL_PARTS[1:]

    async def test_text_fragments_merge_and_keep_trailing_signature(self):
        events = await self._events(
            [
                _response([{"text": "a", "thought": True}, {"text": "1"}], finish=None),
                _response([{"text": "2"}], finish=None),
                _response([{"text": "", "thoughtSignature": "S"}]),
            ]
        )
        assert [e["type"] for e in events] == ["thinking", "content", "content", "result"]
        result = events[-1]["result"]
        assert result.content == "12"
        assert result.reasoning_content == "a"
        assert result.finish_reason == "stop"
        assert result.assistant_message["content"] == [
            {"text": "a", "thought": True},
            {"text": "12", "thoughtSignature": "S"},
        ]


class TestAgainstMockServer:
    """对着按真实协议还原的 mock 跑完整工具循环：签名丢失、tools 格式、
    functionResponse 形态任何一处出错，mock 都会像真实 API 一样 400。"""

    TOOLS = [WEATHER]

    @staticmethod
    def _port():
        import socket

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    async def _loop(self, mode):
        from flexllm.mock import MockLLMServer, MockServerConfig

        cfg = MockServerConfig(port=self._port(), delay_min=0, delay_max=0, thinking=True)
        with MockLLMServer(cfg) as server:
            client = GeminiClient(base_url=server.gemini_url, api_key="k", model="mock-model")
            msgs = [{"role": "user", "content": "东京天气？"}]
            results = []
            for _ in range(2):
                if mode == "stream":
                    async for event in client.complete_stream(
                        msgs, tools=self.TOOLS, thinking=True
                    ):
                        pass
                    result = event["result"]
                else:
                    result = await client.complete(msgs, tools=self.TOOLS, thinking=True)
                results.append(result)
                if not result.tool_calls:
                    break
                if mode == "openai_history":
                    msgs.append(
                        {
                            "role": "assistant",
                            "content": result.content,
                            "tool_calls": [
                                {"id": c.id, "type": c.type, "function": c.function}
                                for c in result.tool_calls
                            ],
                        }
                    )
                else:
                    msgs.append(result.assistant_message)
                for call in result.tool_calls:
                    msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": [
                                {"type": "text", "text": "晴"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                                },
                            ],
                        }
                    )
            await client.aclose()
        return results

    @pytest.mark.parametrize("mode", ["complete", "stream", "openai_history"])
    async def test_tool_loop_completes(self, mode):
        first, second = await self._loop(mode)
        assert first.finish_reason == "tool_calls"
        assert first.tool_calls[0].function["name"] == "get_weather"
        assert first.usage["completion_tokens_details"]["reasoning_tokens"] > 0
        assert second.finish_reason == "stop"
        assert second.content
