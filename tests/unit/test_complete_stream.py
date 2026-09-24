"""complete_stream：事件恒为 dict，结尾给出与 complete() 同构的 ChatCompletionResult"""

import json

import pytest

from flexllm import LLMClientPool
from flexllm.clients.base import ChatCompletionResult, ToolCall
from flexllm.clients.claude import ClaudeClient
from flexllm.clients.openai import OpenAIClient

from .test_extra_passthrough import ScriptedServer, _sse

MESSAGES = [{"role": "user", "content": "hi"}]


def _chunk(delta: dict, finish_reason=None, **top) -> dict:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        **top,
    }


USAGE = {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}


async def _collect_openai(lines, client_cls=OpenAIClient):
    async with ScriptedServer(stream_lines=lines) as server:
        client = client_cls(base_url=server.base_url, api_key="k", model="m")
        events = [e async for e in client.complete_stream(MESSAGES)]
        await client.aclose()
    return events


class TestCompleteStream:
    async def test_plain_text_ends_with_single_result(self):
        lines = [
            _sse(_chunk({"content": "hel"})),
            _sse(_chunk({"content": "lo"}, finish_reason="stop")),
            _sse({**_chunk({}), "choices": [], "usage": USAGE}),
            "data: [DONE]\n\n",
        ]
        events = await _collect_openai(lines)

        assert [e["type"] for e in events] == ["content", "content", "result"]
        result = events[-1]["result"]
        assert isinstance(result, ChatCompletionResult)
        assert result.content == "hello"
        assert result.finish_reason == "stop"
        assert result.usage == USAGE
        assert result.reasoning_content is None
        assert result.tool_calls is None
        assert result.assistant_message is None
        assert result.latency is not None

    async def test_thinking_and_tool_calls_are_aggregated(self):
        lines = [
            _sse(_chunk({"reasoning_content": "let me "})),
            _sse(_chunk({"reasoning_content": "check"})),
            _sse(
                _chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"ci'},
                            }
                        ]
                    }
                )
            ),
            _sse(_chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ty": "X"}'}}]})),
            _sse(_chunk({}, finish_reason="tool_calls")),
            "data: [DONE]\n\n",
        ]
        events = await _collect_openai(lines)

        # 思考内容走独立事件，从不以 <think> 标签混进正文
        assert [e["type"] for e in events] == [
            "thinking",
            "thinking",
            "tool_call_delta",
            "tool_call_delta",
            "result",
        ]
        result = events[-1]["result"]
        assert result.content is None
        assert result.reasoning_content == "let me check"
        assert result.finish_reason == "tool_calls"
        assert result.tool_calls == [
            ToolCall(
                id="call_a",
                type="function",
                function={"name": "get_weather", "arguments": '{"city": "X"}'},
            )
        ]
        assert result.assistant_message["reasoning_content"] == "let me check"
        assert result.assistant_message["tool_calls"][0]["id"] == "call_a"

    async def test_extra_is_streamed_and_collected(self):
        lines = [_sse(_chunk({"content": "hi"}, x_gateway={"blocked": True})), "data: [DONE]\n\n"]
        events = await _collect_openai(lines)

        assert [e["type"] for e in events] == ["extra", "content", "result"]
        assert events[-1]["result"].extra == {"x_gateway": {"blocked": True}}

    async def test_pool_exposes_complete_stream(self):
        lines = [_sse(_chunk({"content": "ok"}, finish_reason="stop")), "data: [DONE]\n\n"]
        async with ScriptedServer(stream_lines=lines) as server:
            pool = LLMClientPool(
                endpoints=[{"base_url": server.base_url, "api_key": "k", "model": "m"}]
            )
            events = [e async for e in pool.complete_stream("hi")]
            await pool.aclose()

        assert events[-1]["result"].content == "ok"

    async def test_claude_keeps_signed_thinking_for_next_turn(self):
        """Claude 下一轮必须回传带签名的 thinking block，result.assistant_message 要带上"""
        lines = [
            _sse({"type": "message_start", "message": {"usage": {"input_tokens": 2}}}),
            _sse(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                }
            ),
            _sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "hmm"},
                }
            ),
            _sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "sig"},
                }
            ),
            _sse(
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "tool_use", "id": "tu_1", "name": "f", "input": {}},
                }
            ),
            _sse(
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '{"a": 1}'},
                }
            ),
            _sse(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 5},
                }
            ),
            _sse({"type": "message_stop"}),
        ]
        async with ScriptedServer(stream_lines=lines, path="/v1/messages") as server:
            client = ClaudeClient(base_url=server.base_url, api_key="k", model="m")
            events = [e async for e in client.complete_stream(MESSAGES)]
            await client.aclose()

        result = events[-1]["result"]
        assert result.reasoning_content == "hmm"
        assert result.tool_calls[0].id == "tu_1"
        assert json.loads(result.tool_calls[0].function["arguments"]) == {"a": 1}
        thinking_block = result.assistant_message["content"][0]
        assert thinking_block["signature"] == "sig"

    async def test_rejects_return_shape_options(self):
        client = OpenAIClient(base_url="http://x/v1", api_key="k", model="m")
        with pytest.raises(ValueError, match="return_usage"):
            async for _ in client.complete_stream(MESSAGES, return_usage=False):
                pass


class TestCompleteStreamEdges:
    async def test_empty_stream_still_yields_result(self):
        events = await _collect_openai(["data: [DONE]\n\n"])

        assert [e["type"] for e in events] == ["result"]
        assert events[0]["result"].content is None

    async def test_http_error_raises_typed_error(self):
        from aiohttp import web

        from flexllm import LLMHTTPError

        async def handler(request):
            return web.json_response({"error": "boom"}, status=500)

        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        client = OpenAIClient(base_url=f"http://127.0.0.1:{port}/v1", api_key="k", model="m")
        try:
            with pytest.raises(LLMHTTPError) as exc:
                async for _ in client.complete_stream(MESSAGES):
                    pass
            assert exc.value.status_code == 500
        finally:
            await client.aclose()
            await runner.cleanup()
