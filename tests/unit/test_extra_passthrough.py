"""带外字段透传：上游标准信封之外的顶层字段不再被静默丢弃

动机是网关场景 —— 代理会在响应上挂"这次工具调用被策略拦了"这类信息，而解析器只认
自己关心的几个路径，其余字段随 data 出作用域消失，调用方连"有东西被丢了"都不知道。

两条必须同时成立的性质：
- 有带外字段时能拿到（否则功能不存在）
- 没有带外字段时一条 extra 都不发（否则每个 chunk 都冒噪音，调用方没法用）
"""

import json

from aiohttp import web

from flexllm.clients.claude import ClaudeClient
from flexllm.clients.openai import OpenAIClient

MESSAGES = [{"role": "user", "content": "hi"}]
SIGNAL = {"events": [{"type": "policy_violation", "tool": "bash", "reason": "denied"}]}


class ScriptedServer:
    """按预设内容回复的假上游。stream_lines 非空时走 SSE，否则返回 json_body。"""

    def __init__(self, json_body=None, stream_lines=None, path="/v1/chat/completions"):
        self.json_body = json_body
        self.stream_lines = stream_lines
        self.path = path
        self._runner = None
        self.base_url = None

    async def _handler(self, request):
        if self.stream_lines is not None:
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            for line in self.stream_lines:
                await resp.write(line.encode())
            await resp.write_eof()
            return resp
        return web.json_response(self.json_body)

    async def __aenter__(self):
        app = web.Application()
        app.router.add_post(self.path, self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}/v1"
        return self

    async def __aexit__(self, *args):
        await self._runner.cleanup()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _openai_chunk(**overrides) -> dict:
    base = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "system_fingerprint": "fp",
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
    }
    base.update(overrides)
    return base


async def _collect(client, **kwargs):
    return [c async for c in client.chat_completions_stream(MESSAGES, return_usage=True, **kwargs)]


class TestOpenAIStream:
    async def test_extra_field_surfaces(self):
        lines = [_sse(_openai_chunk(x_gateway=SIGNAL)), "data: [DONE]\n\n"]
        async with ScriptedServer(stream_lines=lines) as server:
            client = OpenAIClient(base_url=server.base_url, api_key="k", model="m")
            chunks = await _collect(client)
            await client.aclose()

        extras = [c for c in chunks if c["type"] == "extra"]
        assert len(extras) == 1
        assert extras[0]["extra"] == {"x_gateway": SIGNAL}

    async def test_content_on_the_same_chunk_still_arrives(self):
        """信号往往就挂在带 content 的那条 chunk 上，提取顺序错了会把正文吃掉。"""
        lines = [_sse(_openai_chunk(x_gateway=SIGNAL)), "data: [DONE]\n\n"]
        async with ScriptedServer(stream_lines=lines) as server:
            client = OpenAIClient(base_url=server.base_url, api_key="k", model="m")
            chunks = await _collect(client)
            await client.aclose()

        assert [c["content"] for c in chunks if c["type"] == "content"] == ["hi"]

    async def test_tool_call_chunk_still_arrives(self):
        """tool_call 分支带 continue，带外提取必须排在它之前。"""
        tool_delta = {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{}"},
        }
        chunk = _openai_chunk(
            x_gateway=SIGNAL,
            choices=[{"index": 0, "delta": {"tool_calls": [tool_delta]}, "finish_reason": None}],
        )
        lines = [_sse(chunk), "data: [DONE]\n\n"]
        async with ScriptedServer(stream_lines=lines) as server:
            client = OpenAIClient(base_url=server.base_url, api_key="k", model="m")
            chunks = await _collect(client)
            await client.aclose()

        assert any(c["type"] == "extra" for c in chunks)
        assert any(c["type"] == "tool_call_delta" for c in chunks)

    async def test_no_noise_on_ordinary_stream(self):
        lines = [_sse(_openai_chunk()), _sse(_openai_chunk()), "data: [DONE]\n\n"]
        async with ScriptedServer(stream_lines=lines) as server:
            client = OpenAIClient(base_url=server.base_url, api_key="k", model="m")
            chunks = await _collect(client)
            await client.aclose()

        assert [c for c in chunks if c["type"] == "extra"] == []


class TestOpenAINonStream:
    async def test_extra_field_surfaces(self):
        body = {
            "id": "c1",
            "object": "chat.completion",
            "model": "m",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "x_gateway": SIGNAL,
        }
        async with ScriptedServer(json_body=body) as server:
            client = OpenAIClient(base_url=server.base_url, api_key="k", model="m")
            result = await client.chat_completions(MESSAGES, return_usage=True)
            await client.aclose()

        assert result.extra == {"x_gateway": SIGNAL}
        assert result.content == "hi"

    async def test_none_when_envelope_only(self):
        body = {
            "id": "c1",
            "object": "chat.completion",
            "model": "m",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        async with ScriptedServer(json_body=body) as server:
            client = OpenAIClient(base_url=server.base_url, api_key="k", model="m")
            result = await client.chat_completions(MESSAGES, return_usage=True)
            await client.aclose()

        assert result.extra is None


class TestClaudeStream:
    """Claude 的流式实现是独立副本，按 event type 分发，要单独钉。"""

    async def test_unknown_event_type_surfaces(self):
        lines = [
            _sse({"type": "message_start", "message": {"usage": {"input_tokens": 1}}}),
            _sse({"type": "gateway_notice", **SIGNAL}),
            _sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hi"},
                }
            ),
            _sse({"type": "message_stop"}),
        ]
        async with ScriptedServer(stream_lines=lines, path="/v1/messages") as server:
            client = ClaudeClient(base_url=server.base_url, api_key="k", model="m")
            chunks = await _collect(client)
            await client.aclose()

        extras = [c for c in chunks if c["type"] == "extra"]
        assert len(extras) == 1
        assert extras[0]["extra"]["type"] == "gateway_notice"
        assert extras[0]["extra"]["events"] == SIGNAL["events"]
        # 正文没被吃掉
        assert [c["content"] for c in chunks if c["type"] == "content"] == ["hi"]

    async def test_standard_events_produce_no_noise(self):
        """ping / *_stop 属于规范内事件，不能被当成带外信息。"""
        lines = [
            _sse({"type": "ping"}),
            _sse({"type": "message_start", "message": {"usage": {"input_tokens": 1}}}),
            _sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hi"},
                }
            ),
            _sse({"type": "content_block_stop", "index": 0}),
            _sse({"type": "message_stop"}),
        ]
        async with ScriptedServer(stream_lines=lines, path="/v1/messages") as server:
            client = ClaudeClient(base_url=server.base_url, api_key="k", model="m")
            chunks = await _collect(client)
            await client.aclose()

        assert [c for c in chunks if c["type"] == "extra"] == []

    async def test_non_stream_uses_anthropic_envelope(self):
        body = {
            "id": "m1",
            "type": "message",
            "role": "assistant",
            "model": "m",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "x_gateway": SIGNAL,
        }
        async with ScriptedServer(json_body=body, path="/v1/messages") as server:
            client = ClaudeClient(base_url=server.base_url, api_key="k", model="m")
            result = await client.chat_completions(MESSAGES, return_usage=True)
            await client.aclose()

        assert result.extra == {"x_gateway": SIGNAL}
