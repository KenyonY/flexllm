"""Structured HTTP errors stay machine-readable across chat call shapes."""

import pytest
from aiohttp import web

from flexllm import ClaudeClient, LLMRequestError, OpenAIClient

MESSAGES = [{"role": "user", "content": "hi"}]
BLOCK = {"error": "blocked", "x_gateway": {"gateway": "flowlens", "actions": ["block"]}}


class ErrorServer:
    def __init__(self, *, json_body: bool = True):
        self._json_body = json_body
        self._runner = None
        self.base_url = None

    async def _handler(self, request):
        if self._json_body:
            return web.json_response(BLOCK, status=403)
        return web.Response(text="plain failure", status=400)

    async def __aenter__(self):
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._handler)
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


def _client(kind: str, base_url: str):
    cls = OpenAIClient if kind == "openai" else ClaudeClient
    return cls(base_url=base_url, api_key="k", model="m")


@pytest.mark.parametrize("kind", ["openai", "claude"])
async def test_non_stream_error_preserves_decoded_body(kind):
    async with ErrorServer() as server:
        client = _client(kind, server.base_url)
        with pytest.raises(LLMRequestError) as raised:
            await client.chat_completions_or_raise(MESSAGES)
        await client.aclose()

    assert raised.value.status_code == 403
    assert raised.value.response_data == BLOCK
    assert "LLM 请求失败" in str(raised.value)


@pytest.mark.parametrize("kind", ["openai", "claude"])
async def test_stream_error_preserves_decoded_body(kind):
    async with ErrorServer() as server:
        client = _client(kind, server.base_url)
        with pytest.raises(LLMRequestError) as raised:
            async for _ in client.chat_completions_stream(MESSAGES):
                pass
        await client.aclose()

    assert raised.value.status_code == 403
    assert raised.value.response_data == BLOCK
    assert str(raised.value).startswith("HTTP 403:")


async def test_stream_non_json_error_retains_raw_text():
    async with ErrorServer(json_body=False) as server:
        client = OpenAIClient(base_url=server.base_url, api_key="k", model="m")
        with pytest.raises(LLMRequestError) as raised:
            async for _ in client.chat_completions_stream(MESSAGES):
                pass
        await client.aclose()

    assert raised.value.status_code == 400
    assert raised.value.response_data == {"raw": "plain failure"}
