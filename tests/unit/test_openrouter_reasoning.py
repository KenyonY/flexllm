"""OpenRouter reasoning stays distinct from the OpenAI wire protocol."""

import json
from unittest.mock import MagicMock

import pytest
from aiohttp import web

from flexllm import LLMClient, OpenAIClient, Reasoning
from flexllm.reasoning import ReasoningCapabilities, apply_reasoning, resolve_adapter

MESSAGES = [{"role": "user", "content": "OK"}]


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://openrouter.ai/api/v1", "openrouter"),
        ("https://openrouter.ai/api/v1/", "openrouter"),
        ("https://openrouter.ai.example/api/v1", "openai"),
    ],
)
def test_openrouter_detection_uses_exact_host(url, expected):
    assert resolve_adapter(None, "openai", url) == expected
    assert resolve_adapter("openai", "openai", url) == "openai"
    with pytest.raises(ValueError, match="does not match provider"):
        resolve_adapter("openrouter", "gemini", url)


@pytest.fixture
async def router_server():
    bodies = []

    async def handler(request):
        body = await request.json()
        bodies.append(body)
        message = {"content": "OK", "reasoning": "short reasoning"}
        if body.get("stream"):
            data = {"choices": [{"delta": message, "finish_reason": "stop"}]}
            return web.Response(
                text=f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n",
                content_type="text/event-stream",
            )
        return web.json_response({"choices": [{"message": message, "finish_reason": "stop"}]})

    app = web.Application()
    app.router.add_post("/api/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/api/v1", bodies
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("mode", ["chat", "stream", "batch"])
@pytest.mark.parametrize(
    "policy,expected",
    [
        (Reasoning(effort="high"), {"effort": "high"}),
        (Reasoning(enabled=True), {"enabled": True}),
        (Reasoning(enabled=False), {"enabled": False}),
        (Reasoning(budget_tokens=128), {"max_tokens": 128}),
        (Reasoning(enabled=True, effort="minimal"), {"enabled": True, "effort": "minimal"}),
        (None, None),
    ],
)
async def test_public_openrouter_wire_and_results(router_server, mode, policy, expected):
    url, bodies = router_server
    async with LLMClient(
        base_url=url,
        model="google/test-model",
        reasoning_adapter="openrouter",
        reasoning=Reasoning(effort="low"),
    ) as client:
        if mode == "chat":
            result = await client.chat(MESSAGES, reasoning=policy, max_tokens=1024)
        elif mode == "stream":
            events = [
                event
                async for event in client.chat_stream(MESSAGES, reasoning=policy, max_tokens=1024)
            ]
            result = events[-1]["result"]
        else:
            results = await client.chat_batch(
                [MESSAGES],
                params_list=[{"reasoning": policy}],
                max_tokens=1024,
                show_progress=False,
            )
            result = results[0]
        assert result.content == "OK"
        assert result.reasoning_content == "short reasoning"
    assert len(bodies) == 1
    body = bodies[0]
    assert body["max_tokens"] == 1024
    assert "reasoning_effort" not in body
    assert "thinking" not in body
    if expected is None:
        assert "reasoning" not in body
    else:
        assert body["reasoning"] == expected


async def test_openrouter_defaults_replacement_and_cache():
    async with OpenAIClient(
        base_url="https://openrouter.ai/api/v1",
        model="m",
        reasoning=Reasoning(effort="high"),
    ) as client:
        cache = MagicMock()
        cache.get.return_value = {"content": "cached"}
        client._response_cache = cache
        for params, wire in [
            ({}, {"reasoning": {"effort": "high"}}),
            ({"reasoning": Reasoning(budget_tokens=128)}, {"reasoning": {"max_tokens": 128}}),
            ({"reasoning": None}, {}),
        ]:
            result = await client.chat(MESSAGES, **params)
            assert result.content == "cached"
            assert cache.get.call_args.kwargs == {"model": "m", **wire}


def test_openrouter_rejects_combination_even_if_capabilities_allow_it():
    with pytest.raises(ValueError, match="cannot combine"):
        apply_reasoning(
            {"reasoning": Reasoning(effort="high", budget_tokens=128)},
            default=None,
            capabilities=ReasoningCapabilities(supports_effort_and_budget=True),
            adapter="openrouter",
        )
