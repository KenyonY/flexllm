"""finish_reason 透传：非流式结果字段、流式 finish 事件、各 provider 映射、流式 idle 超时。

回归背景：此前 finish_reason 全链路丢弃，模型输出被 max_tokens 截断（"length"）时
调用方拿到的和正常结束一模一样，只能静默收尾。
"""

import json

import pytest

from flexllm import ClaudeClient, GeminiClient, OpenAIClient
from flexllm.clients import base as base_mod

# ============== 非流式提取 ==============


class TestOpenAIExtraction:
    def test_finish_reason_from_choices(self):
        client = OpenAIClient(base_url="http://x", model="m", api_key="k")
        data = {"choices": [{"message": {"content": "hi"}, "finish_reason": "length"}]}
        assert client._extract_finish_reason(data) == "length"

    def test_missing_is_none(self):
        client = OpenAIClient(base_url="http://x", model="m", api_key="k")
        assert client._extract_finish_reason({"choices": [{"message": {}}]}) is None
        assert client._extract_finish_reason({}) is None
        assert client._extract_finish_reason(None) is None

    def test_stream_chunk_same_shape(self):
        client = OpenAIClient(base_url="http://x", model="m", api_key="k")
        chunk = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        assert client._extract_finish_reason(chunk) == "stop"

    def test_reasoning_not_promoted_when_truncated(self):
        """content 空 + reasoning 非空的启发式（parser 误判）不适用于 length：
        那是思考把 max_tokens 吃光，reasoning 是半截思维链，不能冒充回答。"""
        client = OpenAIClient(base_url="http://x", model="m", api_key="k")
        truncated = {
            "choices": [
                {
                    "message": {"content": None, "reasoning_content": "Let me think..."},
                    "finish_reason": "length",
                }
            ]
        }
        assert client._extract_content(truncated) in (None, "")

        # 对照：正常结束时启发式照旧生效
        misparsed = {
            "choices": [
                {
                    "message": {"content": None, "reasoning_content": "The answer is 42"},
                    "finish_reason": "stop",
                }
            ]
        }
        assert client._extract_content(misparsed) == "The answer is 42"

    def test_reasoning_not_promoted_during_tool_call(self):
        client = OpenAIClient(base_url="http://x", model="deepseek-v4", api_key="k")
        tool_round = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning_content": "I should inspect the file.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "inspect", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        assert client._extract_content(tool_round) in (None, "")


class TestClaudeMapping:
    @pytest.mark.parametrize(
        "stop_reason,expected",
        [
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("max_tokens", "length"),
            ("tool_use", "tool_calls"),
            ("refusal", "content_filter"),
            ("model_context_window_exceeded", "length"),
            ("some_future_reason", "some_future_reason"),
            (None, None),
        ],
    )
    def test_map(self, stop_reason, expected):
        client = ClaudeClient(base_url="http://x", model="m", api_key="k")
        assert client._extract_finish_reason({"stop_reason": stop_reason}) == expected


class TestGeminiMapping:
    @pytest.mark.parametrize(
        "reason,expected",
        [("STOP", "stop"), ("MAX_TOKENS", "length"), ("SAFETY", "content_filter"), (None, None)],
    )
    def test_map(self, reason, expected):
        client = GeminiClient(base_url="http://x", model="m", api_key="k")
        data = {"candidates": [{"content": {"parts": []}, "finishReason": reason}]}
        assert client._extract_finish_reason(data) == expected

    def test_no_candidates(self):
        client = GeminiClient(base_url="http://x", model="m", api_key="k")
        assert client._extract_finish_reason({}) is None


# ============== 流式：finish 事件 + 超时语义 ==============


class _FakeContent:
    def __init__(self, lines):
        self._lines = lines

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


class _FakeResponse:
    status = 200

    def __init__(self, lines):
        self.content = _FakeContent(lines)


class _FakeCtx:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, lines):
        self.lines = lines
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(kwargs)
        return _FakeCtx(_FakeResponse(list(self.lines)))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _sse(obj) -> bytes:
    return f"data: {json.dumps(obj)}\n".encode()


def _openai_stream_lines(finish_reason):
    return [
        _sse({"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]}),
        # 真实 OpenAI：finish chunk 与最终 usage chunk 分开
        _sse({"choices": [{"delta": {}, "finish_reason": finish_reason}]}),
        _sse({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}),
        b"data: [DONE]\n",
    ]


@pytest.fixture
def fake_stream(monkeypatch):
    holder = {}

    def factory(lines):
        session = _FakeSession(lines)
        holder["session"] = session
        monkeypatch.setattr(base_mod, "create_proxied_session", lambda proxy: (session, {}))
        return session

    return factory


class TestOpenAIStream:
    @pytest.mark.parametrize("reason", ["stop", "length", "tool_calls"])
    async def test_finish_event_before_usage(self, fake_stream, reason):
        fake_stream(_openai_stream_lines(reason))
        client = OpenAIClient(base_url="http://x", model="m", api_key="k")
        events = [
            e
            async for e in client.chat_completions_stream(
                [{"role": "user", "content": "hi"}], return_usage=True
            )
        ]
        types = [e["type"] for e in events]
        assert types == ["content", "content", "finish", "usage"]
        assert events[2]["reason"] == reason
        assert events[3]["usage"]["completion_tokens"] == 2

    async def test_finish_event_none_when_provider_omits(self, fake_stream):
        """有些网关不给 finish_reason：事件仍发，reason=None，调用方不用特判有无"""
        fake_stream(
            [
                _sse({"choices": [{"delta": {"content": "x"}}]}),
                b"data: [DONE]\n",
            ]
        )
        client = OpenAIClient(base_url="http://x", model="m", api_key="k")
        events = [
            e
            async for e in client.chat_completions_stream(
                [{"role": "user", "content": "hi"}], return_usage=True
            )
        ]
        assert events[-1] == {"type": "finish", "reason": None}

    async def test_reasoning_tool_stream_emits_replayable_assistant_message(self, fake_stream):
        fake_stream(
            [
                _sse(
                    {
                        "choices": [
                            {
                                "delta": {"reasoning_content": "Need weather."},
                                "finish_reason": None,
                            }
                        ]
                    }
                ),
                _sse(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_weather",
                                            "type": "function",
                                            "function": {
                                                "name": "get_weather",
                                                "arguments": '{"city":',
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                ),
                _sse(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {"index": 0, "function": {"arguments": '"Beijing"}'}}
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                ),
                _sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
                b"data: [DONE]\n",
            ]
        )
        client = OpenAIClient(base_url="http://x", model="deepseek-v4", api_key="k")

        events = [
            event
            async for event in client.chat_completions_stream(
                [{"role": "user", "content": "Weather?"}], return_usage=True
            )
        ]
        message = next(event["message"] for event in events if event["type"] == "assistant_message")

        assert message == {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Need weather.",
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Beijing"}',
                    },
                }
            ],
        }

    async def test_plain_mode_unaffected(self, fake_stream):
        """return_usage=False 仍是纯 str 流"""
        fake_stream(_openai_stream_lines("length"))
        client = OpenAIClient(base_url="http://x", model="m", api_key="k")
        chunks = [
            c async for c in client.chat_completions_stream([{"role": "user", "content": "hi"}])
        ]
        assert chunks == ["Hel", "lo"]

    async def test_stream_timeout_is_idle_not_total(self, fake_stream):
        """流式超时 = 相邻 chunk 间隔上限，不是整条流的总时长。

        回归：此前 ClientTimeout(total=120)，长思考模型一轮超过 120s 就在中途抛
        TimeoutError，即使 token 一直在流入。
        """
        session = fake_stream(_openai_stream_lines("stop"))
        client = OpenAIClient(base_url="http://x", model="m", api_key="k", timeout=77)
        async for _ in client.chat_completions_stream([{"role": "user", "content": "hi"}]):
            pass
        t = session.calls[0]["timeout"]
        assert t.total is None
        assert t.sock_read == 77
        assert t.sock_connect == 30

    async def test_per_call_timeout_override(self, fake_stream):
        session = fake_stream(_openai_stream_lines("stop"))
        client = OpenAIClient(base_url="http://x", model="m", api_key="k", timeout=77)
        async for _ in client.chat_completions_stream(
            [{"role": "user", "content": "hi"}], timeout=5
        ):
            pass
        t = session.calls[0]["timeout"]
        assert t.sock_read == 5 and t.sock_connect == 5


# ============== 非流式：ChatCompletionResult.finish_reason ==============


class TestNonStreamResult:
    async def test_result_carries_finish_reason(self, monkeypatch):
        from flexllm.async_api.interface import RequestResult

        client = OpenAIClient(base_url="http://x", model="m", api_key="k")
        payload = {
            "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 9},
        }

        async def fake_process_requests(**kwargs):
            return [RequestResult(request_id=0, data=payload, status="success", latency=0.0)], None

        monkeypatch.setattr(client._client, "process_requests", fake_process_requests)
        result = await client.chat_completions(
            [{"role": "user", "content": "hi"}], return_usage=True
        )
        assert result.content == "partial"
        assert result.finish_reason == "length"
