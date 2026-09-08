"""Exercise YAML-backed pools through real HTTP, including the CLI and SSE."""

import asyncio
import json

import pytest
import typer
import yaml
from aiohttp import web
from typer.testing import CliRunner

from flexllm import LLMClient
from flexllm.cli import config as config_module
from flexllm.cli.commands import register_commands
from flexllm.cli.config import FlexLLMConfig


@pytest.fixture
async def replicas(aiohttp_server, tmp_path, monkeypatch):
    received = []
    fail = set()

    async def start(name):
        async def complete(request):
            body = await request.json()
            received.append((name, body, request.headers.get("Authorization")))
            if name in fail:
                return web.json_response({"error": {"message": "unavailable"}}, status=503)
            await asyncio.sleep(0.01)
            if body.get("stream"):
                chunk = {"choices": [{"index": 0, "delta": {"content": name}}]}
                return web.Response(
                    text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
                    content_type="text/event-stream",
                )
            return web.json_response(
                {
                    "model": body["model"],
                    "choices": [{"message": {"role": "assistant", "content": name}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            )

        application = web.Application()
        application.router.add_post("/v1/chat/completions", complete)
        server = await aiohttp_server(application)
        return str(server.make_url("/v1"))

    urls = [await start(name) for name in ("replica-a", "replica-b")]
    entry = {
        "name": "replicas",
        "id": "qwen",
        "api_key": "test-key",
        "fallback": True,
        "system": "Be concise.",
        "temperature": 0.25,
        "endpoints": [{"base_url": url, "concurrency_limit": 1} for url in urls],
    }
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "default": "replicas",
                "models": [entry],
                "batch": {"cache": False, "retry_times": 0, "track_cost": False},
            }
        )
    )
    monkeypatch.setattr(config_module, "_config", FlexLLMConfig(path))
    return path, received, fail


@pytest.mark.asyncio
async def test_yaml_pool_distributes_requests_and_streams(replicas):
    path, received, _ = replicas
    async with LLMClient.from_config(str(path), model="replicas", retry_times=0) as client:
        results = await asyncio.gather(*(client.chat_completions("hi") for _ in range(4)))
        assert set(results) == {"replica-a", "replica-b"}
        chunks = [chunk async for chunk in client.chat_completions_stream("hi")]
        assert "".join(chunks) in {"replica-a", "replica-b"}
    for _, body, auth in received:
        assert body["model"] == "qwen"
        assert body["temperature"] == 0.25
        assert body["messages"][0] == {"role": "system", "content": "Be concise."}
        assert auth == "Bearer test-key"
        assert "endpoints" not in body and "fallback" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_yaml_pool_fails_over(replicas, stream):
    path, received, fail = replicas
    fail.add("replica-a")
    async with LLMClient.from_config(str(path), model="replicas", retry_times=0) as client:
        if stream:
            result = "".join([chunk async for chunk in client.chat_completions_stream("hi")])
        else:
            result = await client.chat_completions("hi")
        assert result == "replica-b"
    assert [name for name, _, _ in received] == ["replica-a", "replica-b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["ask", "chat", "batch"])
async def test_cli_uses_named_pool_over_http(replicas, tmp_path, command):
    _, received, _ = replicas
    app = typer.Typer()
    register_commands(app)
    if command == "batch":
        source = tmp_path / "input.jsonl"
        source.write_text("".join(json.dumps({"prompt": f"hi {i}"}) + "\n" for i in range(4)))
        args = [command, str(source), "-m", "replicas", "--format", "json"]
    else:
        args = [command, "hi", "-m", "replicas"]
    result = await asyncio.to_thread(CliRunner().invoke, app, args)
    assert result.exit_code == 0, (result.output, result.exception)
    assert received
    if command == "batch":
        assert {name for name, _, _ in received} == {"replica-a", "replica-b"}
    if command == "chat":
        assert received[0][1]["stream"] is True
    for _, body, auth in received:
        assert body["model"] == "qwen"
        assert body["temperature"] == 0.25
        assert auth == "Bearer test-key"
        assert "endpoints" not in body and "fallback" not in body
