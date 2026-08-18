"""per-call extra_headers：向网关声明调用方身份

约束（这三条一起构成 extra_headers 的正确性，任一条破了都是静默故障）：
- header 必须真的到达上游，且覆盖同名的客户端固定 header
- 绝不能出现在请求体里（上游对未知 body 字段可能 400，也会污染语义）
- 绝不能进缓存键（否则每个 session 一份缓存，缓存等于失效）

第三条是最隐蔽的：如果哪天有人把具名参数改回 kwargs.pop 并放在缓存查询之后，
表现不是报错而是缓存命中率悄悄归零。
"""

import json

from aiohttp import web

from flexllm.cache.response_cache import ResponseCacheConfig
from flexllm.clients.openai import OpenAIClient


class RecordingServer:
    """记录每次请求的 header 与 body 的假上游，支持流式与非流式。"""

    def __init__(self):
        self.requests: list[tuple[dict, dict]] = []  # (headers, body)
        self._runner = None
        self.base_url = None

    @property
    def count(self) -> int:
        return len(self.requests)

    async def _handler(self, request):
        body = await request.json()
        self.requests.append((dict(request.headers), body))

        if body.get("stream"):
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            chunk = {"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]}
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
            done = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
            await resp.write(f"data: {json.dumps(done)}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp

        return web.json_response(
            {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    async def __aenter__(self):
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}/v1"
        return self

    async def __aexit__(self, *args):
        await self._runner.cleanup()


MESSAGES = [{"role": "user", "content": "hi"}]
IDENTITY = {"x-agent-id": "demo/main", "x-session-id": "sess-1"}


class TestNonStream:
    async def test_headers_reach_upstream(self):
        async with RecordingServer() as server:
            client = OpenAIClient(base_url=server.base_url, api_key="k", model="m")
            await client.chat_completions(MESSAGES, extra_headers=IDENTITY)
            await client.aclose()

        headers, _ = server.requests[0]
        assert headers["x-agent-id"] == "demo/main"
        assert headers["x-session-id"] == "sess-1"
        # 固定 header 不受影响
        assert headers["Authorization"] == "Bearer k"

    async def test_not_leaked_into_body(self):
        async with RecordingServer() as server:
            client = OpenAIClient(base_url=server.base_url, api_key="k", model="m")
            await client.chat_completions(MESSAGES, extra_headers=IDENTITY)
            await client.aclose()

        _, body = server.requests[0]
        assert "extra_headers" not in body
        assert "x-agent-id" not in body

    async def test_client_headers_not_mutated_across_calls(self):
        """_get_headers() 常返回构造期缓存的 dict，原地 update 会毒化后续请求。"""
        async with RecordingServer() as server:
            client = OpenAIClient(base_url=server.base_url, api_key="k", model="m")
            await client.chat_completions(MESSAGES, extra_headers=IDENTITY)
            await client.chat_completions(MESSAGES)
            await client.aclose()

        second_headers, _ = server.requests[1]
        assert "x-agent-id" not in second_headers

    async def test_or_raise_forwards_extra_headers(self):
        """mens 主循环走的是 or_raise，它靠 **kwargs 转发到具名参数上。"""
        async with RecordingServer() as server:
            client = OpenAIClient(base_url=server.base_url, api_key="k", model="m")
            await client.chat_completions_or_raise(
                MESSAGES, return_usage=True, extra_headers=IDENTITY
            )
            await client.aclose()

        headers, body = server.requests[0]
        assert headers["x-agent-id"] == "demo/main"
        assert "extra_headers" not in body


class TestStream:
    async def test_headers_reach_upstream(self):
        async with RecordingServer() as server:
            client = OpenAIClient(base_url=server.base_url, api_key="k", model="m")
            async for _ in client.chat_completions_stream(MESSAGES, extra_headers=IDENTITY):
                pass
            await client.aclose()

        headers, body = server.requests[0]
        assert headers["x-agent-id"] == "demo/main"
        assert headers["x-session-id"] == "sess-1"
        assert "extra_headers" not in body


class TestCacheKey:
    async def test_extra_headers_do_not_pollute_cache_key(self, tmp_path):
        """同 messages、不同身份 header —— 第二次必须命中缓存，不再打上游。

        header 是传输层的事，与"这次问了什么"无关。若它进了缓存键，
        每个 session / 每个 subagent 都会各自持有一份缓存副本。

        cache_dir 必须指到 tmp_path：默认目录是 ~/.flexllm/cache，跨进程持久，
        第二次跑测试时首个请求也会命中缓存，count 变 0。
        """
        cache = ResponseCacheConfig(enabled=True, cache_dir=str(tmp_path))
        async with RecordingServer() as server:
            client = OpenAIClient(base_url=server.base_url, api_key="k", model="m", cache=cache)
            await client.chat_completions(MESSAGES, extra_headers={"x-session-id": "a"})
            await client.chat_completions(MESSAGES, extra_headers={"x-session-id": "b"})
            await client.aclose()

        assert server.count == 1


class AnthropicRecordingServer:
    """记录 header 与 body 的假 Anthropic 上游（/v1/messages，流式 SSE）。"""

    def __init__(self):
        self.requests: list[tuple[dict, dict]] = []
        self._runner = None
        self.base_url = None

    async def _handler(self, request):
        body = await request.json()
        self.requests.append((dict(request.headers), body))
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        events = [
            ("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 1}}}),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "ok"},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 1},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        for name, data in events:
            await resp.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
        await resp.write_eof()
        return resp

    async def __aenter__(self):
        app = web.Application()
        app.router.add_post("/v1/messages", self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}/v1"
        return self

    async def __aexit__(self, *args):
        await self._runner.cleanup()


class TestClaudeStream:
    """Claude 的流式是独立实现，不走基类：具名参数漏掉时 extra_headers 会随 **kwargs
    进请求体，Anthropic 对未知字段直接 400（"extra_headers: Extra inputs are not
    permitted"）——整个 provider 在带身份的调用方手里不可用。"""

    async def test_headers_reach_upstream_not_body(self):
        from flexllm.clients.claude import ClaudeClient

        async with AnthropicRecordingServer() as server:
            client = ClaudeClient(base_url=server.base_url, api_key="k", model="m")
            async for _ in client.chat_completions_stream(MESSAGES, extra_headers=IDENTITY):
                pass
            await client.aclose()

        headers, body = server.requests[0]
        assert headers["x-agent-id"] == "demo/main"
        assert headers["x-session-id"] == "sess-1"
        assert "extra_headers" not in body
