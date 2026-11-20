"""Real TCP tests: streaming responses borrow the client's connection pool."""

import asyncio
import gc
import json
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from aiohttp import web

from flexllm.clients.claude import ClaudeClient
from flexllm.clients.gemini import GeminiClient
from flexllm.clients.openai import OpenAIClient


@pytest.fixture(params=[OpenAIClient, ClaudeClient, GeminiClient])
def provider(request):
    return request.param


def frame(provider):
    if provider is ClaudeClient:
        data = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello"},
        }
    elif provider is GeminiClient:
        data = {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}
    else:
        data = {"choices": [{"delta": {"content": "hello"}}]}
    return f"data: {json.dumps(data)}\n\n".encode()


async def collect(client):
    return [c async for c in client.chat_completions_stream([{"role": "user", "content": "hi"}])]


async def test_sequential_streams_reuse_tcp_and_close_pool(provider, aiohttp_server):
    transports = []

    async def handler(request):
        await request.read()
        transports.append(request.transport)
        return web.Response(
            body=frame(provider) + b"data: [DONE]\n\n", content_type="text/event-stream"
        )

    app = web.Application()
    app.router.add_post("/{path:.*}", handler)
    server = await aiohttp_server(app)
    client = provider(base_url=str(server.make_url("/")), api_key="test", model="test")
    try:
        assert await collect(client) == ["hello"]
        assert await collect(client) == ["hello"]
        assert transports[0] is transports[1]
        session = client._client._session
        assert not session.closed
    finally:
        await client.aclose()
    assert session.closed


async def test_cancel_one_stream_does_not_close_concurrent_or_next_stream(provider, aiohttp_server):
    started = asyncio.Event()
    second_started = asyncio.Event()
    release = asyncio.Event()
    count = 0

    async def handler(request):
        nonlocal count
        await request.read()
        count += 1
        if count == 1:
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(frame(provider))
            started.set()
            await release.wait()
            return response
        if count == 2:
            second_started.set()
            await release.wait()
        return web.Response(
            body=frame(provider) + b"data: [DONE]\n\n", content_type="text/event-stream"
        )

    app = web.Application()
    app.router.add_post("/{path:.*}", handler)
    server = await aiohttp_server(app)
    client = provider(base_url=str(server.make_url("/")), api_key="test", model="test")
    first = asyncio.create_task(collect(client))
    second = None
    try:
        await asyncio.wait_for(started.wait(), 3)
        second = asyncio.create_task(collect(client))
        await asyncio.wait_for(second_started.wait(), 3)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not client._client._session.closed
        release.set()
        assert await second == ["hello"]
        assert await collect(client) == ["hello"]
    finally:
        release.set()
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        if second:
            second.cancel()
            await asyncio.gather(second, return_exceptions=True)
        await client.aclose()


async def test_shared_pool_never_carries_gateway_cookies(provider, aiohttp_server):
    cookies = []

    async def handler(request):
        await request.read()
        cookies.append(dict(request.cookies))
        response = web.Response(
            body=frame(provider) + b"data: [DONE]\n\n", content_type="text/event-stream"
        )
        response.set_cookie("route", "previous-call")
        return response

    app = web.Application()
    app.router.add_post("/{path:.*}", handler)
    server = await aiohttp_server(app)
    # CookieJar rejects IP literals by default: use a hostname to detect leakage.
    url = str(server.make_url("/")).replace("127.0.0.1", "localhost")
    client = provider(base_url=url, api_key="test", model="test")
    try:
        await collect(client)
        await collect(client)
    finally:
        await client.aclose()
    assert cookies == [{}, {}]


def test_reuse_across_loops_closes_in_each_owning_loop(provider):
    body = frame(provider) + b"data: [DONE]\n\n"

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = provider(
        base_url=f"http://127.0.0.1:{server.server_port}", api_key="test", model="test"
    )

    async def call():
        try:
            assert await collect(client) == ["hello"]
        finally:
            await client.aclose()
            await asyncio.sleep(0)

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            asyncio.run(call())
            asyncio.run(call())
            gc.collect()
            assert not [w for w in caught if issubclass(w.category, ResourceWarning)]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
