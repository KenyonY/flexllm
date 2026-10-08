"""Public reasoning contract: policy replacement, routing, wire fields and cache."""

import copy
import json
from unittest.mock import MagicMock

import pytest
import typer
import yaml
from aiohttp import web
from typer.testing import CliRunner

from flexllm import ClaudeClient, GeminiClient, LLMClient, OpenAIClient, Reasoning
from flexllm.async_api.interface import RequestResult
from flexllm.cli import config as config_module
from flexllm.cli.commands import register_commands
from flexllm.cli.config import FlexLLMConfig
from flexllm.reasoning import ReasoningCapabilities, TokenBudget

MESSAGES = [{"role": "user", "content": "OK"}]


@pytest.mark.parametrize(
    "values",
    [
        {"enabled": "false"},
        {"enabled": False, "effort": "low"},
        {"enabled": False, "budget_tokens": 128},
        {"budget_tokens": True},
        {"budget_tokens": 0},
        {"effort": "none"},
        {"effort": ""},
    ],
)
def test_invalid_policy_is_rejected(values):
    with pytest.raises(ValueError):
        Reasoning(**values)


@pytest.mark.parametrize("values", [{1: "high"}, {"unknown": "high"}])
def test_malformed_policy_fields_raise_usage_error(values):
    with pytest.raises(ValueError, match="Invalid reasoning fields"):
        Reasoning.parse(values)


def test_capabilities_distinguish_unknown_from_unsupported_and_validate_ranges():
    ReasoningCapabilities().validate(Reasoning(effort="new-provider-level"))
    with pytest.raises(ValueError, match="supported levels"):
        ReasoningCapabilities(effort_levels=()).validate(Reasoning(effort="high"))
    caps = ReasoningCapabilities(budget_tokens=TokenBudget(128, 32768), can_disable=False)
    for policy in [Reasoning(budget_tokens=127), Reasoning(enabled=False)]:
        with pytest.raises(ValueError):
            caps.validate(policy)
    with pytest.raises(ValueError, match="combined"):
        caps.validate(Reasoning(effort="high", budget_tokens=1024))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cls,model,options,policy,expected,response",
    [
        (
            OpenAIClient,
            "gpt",
            {},
            Reasoning(effort="max"),
            {"reasoning_effort": "max"},
            {"choices": [{"message": {"content": "OK", "reasoning_content": "thought"}}]},
        ),
        (
            OpenAIClient,
            "deepseek",
            {"reasoning_adapter": "deepseek"},
            Reasoning(enabled=False),
            {"thinking": {"type": "disabled"}},
            {"choices": [{"message": {"content": "OK"}}]},
        ),
        (
            OpenAIClient,
            "sf",
            {"reasoning_adapter": "siliconflow"},
            Reasoning(budget_tokens=256),
            {"enable_thinking": True, "thinking_budget": 256},
            {"choices": [{"message": {"content": "OK"}}]},
        ),
        (
            ClaudeClient,
            "claude-opus-5-5",
            {},
            Reasoning(effort="xhigh"),
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "xhigh"}},
            {"content": [{"type": "text", "text": "OK"}]},
        ),
        (
            GeminiClient,
            "gemini-3-flash",
            {},
            Reasoning(effort="low"),
            {
                "generationConfig": {
                    "thinkingConfig": {"includeThoughts": True, "thinkingLevel": "low"}
                }
            },
            {"candidates": [{"content": {"parts": [{"text": "OK"}]}}]},
        ),
    ],
)
async def test_public_chat_serializes_exact_native_fields(
    monkeypatch, cls, model, options, policy, expected, response
):
    async with cls(
        base_url="https://gateway.example/v1", api_key="test", model=model, **options
    ) as client:

        async def request(**kwargs):
            body = kwargs["request_params"][0]["json"]
            assert "reasoning" not in body
            for key, value in expected.items():
                assert body[key] == value
            return [RequestResult(0, response, "success", 0.0)], None

        monkeypatch.setattr(client._client, "process_requests", request)
        result = await client.chat(MESSAGES, reasoning=policy)
        assert result.content == "OK"
        assert "<think>" not in result.content


@pytest.mark.asyncio
async def test_replacement_and_server_default_happen_before_cache_lookup():
    async with OpenAIClient(
        base_url="https://api.siliconflow.cn/v1",
        model="a",
        reasoning=Reasoning(enabled=False),
        reasoning_capabilities={"effort_levels": ["high", "max"]},
    ) as client:
        cache = MagicMock()
        cache.get.return_value = {"content": "cached"}
        client._response_cache = cache
        await client.chat(MESSAGES, reasoning=Reasoning(effort="high"))
        assert cache.get.call_args.kwargs == {
            "model": "a",
            "enable_thinking": True,
            "reasoning_effort": "high",
        }
        await client.chat(MESSAGES, reasoning=None)
        assert cache.get.call_args.kwargs == {"model": "a"}
        await client.chat(MESSAGES)
        assert cache.get.call_args.kwargs == {"model": "a", "enable_thinking": False}
        with pytest.raises(ValueError, match="supported levels: high, max"):
            await client.chat(MESSAGES, reasoning=Reasoning(effort="medium"))
        assert client.get_capabilities("different-model").reasoning.effort_levels is None
        await client.chat(MESSAGES, model="different-model", reasoning=Reasoning(effort="medium"))


@pytest.mark.asyncio
async def test_batch_policies_replace_defaults_and_use_effective_cache_keys():
    async with OpenAIClient(
        base_url="https://api.siliconflow.cn/v1", model="a", reasoning={"enabled": False}
    ) as client:
        cache = MagicMock()
        cache.get_batch.return_value = ([{"content": "one"}, {"content": "two"}], [])
        client._response_cache = cache
        result = await client.chat_batch(
            [MESSAGES, MESSAGES],
            params_list=[{"reasoning": {"effort": "max"}}, {"reasoning": None}],
            show_progress=False,
        )
        assert [r.content for r in result] == ["one", "two"]
        assert cache.get_batch.call_args.kwargs["params_list"] == [
            {"enable_thinking": True, "reasoning_effort": "max"},
            {},
        ]


@pytest.mark.asyncio
async def test_stream_and_batch_use_the_same_policy():
    bodies = []

    async def handler(request):
        body = await request.json()
        bodies.append(body)
        if body.get("stream"):
            data = {"choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}]}
            return web.Response(
                text=f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n",
                content_type="text/event-stream",
            )
        return web.json_response(
            {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        async with LLMClient(
            base_url=f"http://127.0.0.1:{port}/v1", model="m", reasoning={"effort": "low"}
        ) as client:
            events = [
                event
                async for event in client.chat_stream(MESSAGES, reasoning=Reasoning(effort="high"))
            ]
            assert events[-1]["result"].content == "OK"
            results = await client.chat_batch(
                [MESSAGES, MESSAGES], params_list=[None, {"reasoning": None}], show_progress=False
            )
            assert all(result.ok for result in results)
        assert bodies[0]["reasoning_effort"] == "high"
        # Concurrent batch requests may reach the server in either order.
        assert sorted(body.get("reasoning_effort", "") for body in bodies[1:]) == ["", "low"]
    finally:
        await runner.cleanup()


def test_per_endpoint_capabilities_do_not_collapse_to_first_endpoint():
    with LLMClient(
        endpoints=[
            {
                "base_url": "https://one.example/v1",
                "model": "m",
                "reasoning_capabilities": {"effort_levels": ["high"]},
            },
            {
                "base_url": "https://two.example/v1",
                "model": "m",
                "reasoning_capabilities": {"effort_levels": ["low"]},
            },
        ]
    ) as client:
        assert client.endpoint_capabilities[0].reasoning.effort_levels == ("high",)
        assert client.endpoint_capabilities[1].reasoning.effort_levels == ("low",)
        with pytest.raises(ValueError, match="differ"):
            _ = client.capabilities


@pytest.fixture
def configured_cli(tmp_path, monkeypatch):
    entry = {
        "id": "model-b",
        "name": "b",
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key": "do-not-print-this",
        "reasoning": {"enabled": False},
        "reasoning_capabilities": {"effort_levels": ["high", "max"], "budget_tokens": False},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"default": "b", "models": [entry]}))
    cfg = FlexLLMConfig(path)
    monkeypatch.setattr(config_module, "_config", cfg)
    app = typer.Typer()
    register_commands(app)
    return app, path, cfg


def test_config_and_cli_share_capabilities_and_validate_before_dry_run(configured_cli):
    app, path, cfg = configured_cli
    original = copy.deepcopy(cfg.config)
    with LLMClient.from_config(str(path), model="b") as client:
        assert client.capabilities.reasoning.effort_levels == ("high", "max")
        assert client._config_params == {}
    runner = CliRunner()
    result = runner.invoke(app, ["capabilities", "-m", "b", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["endpoints"][0]["reasoning"]["effort_levels"] == [
        "high",
        "max",
    ]
    assert "do-not-print-this" not in result.output
    for command in ("ask", "chat"):
        result = runner.invoke(
            app, [command, "hello", "-m", "b", "--reasoning-effort", "medium", "--dry-run"]
        )
        assert result.exit_code == 2, result.output
        assert "high, max" in result.output
        result = runner.invoke(
            app, [command, "hello", "-m", "b", "--reasoning-effort", "high", "--dry-run"]
        )
        assert result.exit_code == 10, result.output
        assert json.loads(result.stdout)["params"]["reasoning"] == {"effort": "high"}
    assert cfg.config == original


def test_native_conflicts_and_unsupported_wire_controls():
    with OpenAIClient(base_url="https://gateway.example/v1", model="m") as client:
        for params in [
            {"reasoning": {"effort": "high"}, "reasoning_effort": "low"},
            {"reasoning": {"effort": "high"}, "output_config": {"effort": "low"}},
            {"reasoning": {"effort": "high"}, "chat_template_kwargs": {"enable_thinking": True}},
            {"reasoning": {"effort": "high"}, "thinking_config": {"thinkingBudget": 128}},
            {"reasoning": {"effort": "high"}, "thinkingConfig": {"thinkingBudget": 128}},
            {"reasoning": {"enabled": True}},
            {"reasoning": {"budget_tokens": 1024}},
        ]:
            with pytest.raises(ValueError):
                client._prepare_reasoning_kwargs("m", params)


def test_changing_target_drops_bound_declarations(configured_cli):
    _, path, _ = configured_cli
    with LLMClient.from_config(str(path), model="b", base_url="https://new.example/v1") as client:
        assert client.capabilities.reasoning.effort_levels is None
        assert client._single_client._reasoning_adapter == "openai"


def test_capability_query_without_model_id_is_offline(configured_cli, monkeypatch):
    app, _, cfg = configured_cli
    cfg.config["models"][0]["id"] = None

    def unexpected(*args, **kwargs):
        raise AssertionError("capability lookup must not fetch models")

    monkeypatch.setattr("flexllm.cli.utils._fetch_model_id", unexpected)
    result = CliRunner().invoke(app, ["capabilities", "-m", "b"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["endpoints"][0]["model"] is None


@pytest.mark.parametrize("command", ["serve", "chat-web"])
def test_server_cli_rejects_invalid_reasoning(configured_cli, command):
    app, _, _ = configured_cli
    result = CliRunner().invoke(
        app, [command, "-m", "b", "--reasoning-effort", "medium", "--dry-run"]
    )
    assert result.exit_code == 2, result.output
    assert "high, max" in result.output


@pytest.mark.parametrize("command", ["serve", "chat-web"])
@pytest.mark.parametrize("thinking", ["true", "false", "high"])
def test_server_cli_rejects_legacy_thinking_with_reasoning_default(
    configured_cli, command, thinking
):
    app, _, cfg = configured_cli
    original = copy.deepcopy(cfg.config)
    result = CliRunner().invoke(app, [command, "-m", "b", "--thinking", thinking, "--dry-run"])
    assert result.exit_code == 2, result.output
    assert "Do not combine reasoning with native controls: thinking" in result.output
    assert cfg.config == original


def test_batch_cli_validates_row_reasoning(configured_cli, tmp_path):
    app, _, _ = configured_cli
    source = tmp_path / "input.jsonl"
    source.write_text(
        json.dumps({"prompt": "hello", "params": {"reasoning": {"effort": "medium"}}}) + "\n"
    )
    result = CliRunner().invoke(app, ["batch", str(source), "-m", "b", "--dry-run"])
    assert result.exit_code == 2, result.output
    assert "high, max" in result.output


@pytest.mark.parametrize("native", [{"thinking": False}, {"reasoning_effort": "low"}])
def test_batch_cli_checks_native_row_conflicts_with_default(configured_cli, tmp_path, native):
    app, _, _ = configured_cli
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"prompt": "hello", "params": native}) + "\n")
    result = CliRunner().invoke(app, ["batch", str(source), "-m", "b", "--dry-run"])
    assert result.exit_code == 2, result.output
    assert "Do not combine reasoning with native controls" in result.output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,model", [("openai", "m"), ("claude", "claude-opus-5-5"), ("gemini", "gemini-3-flash")]
)
@pytest.mark.parametrize("fail", [False, True])
async def test_native_streams_preserve_effort_and_surface_in_band_errors(provider, model, fail):
    error = {"error": {"type": "upstream_error", "message": "overloaded"}}

    async def handler(request):
        body = await request.json()
        if provider == "claude":
            assert body["output_config"]["effort"] == "high"
            data = [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "OK"},
                },
            ]
        elif provider == "gemini":
            assert body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "high"
            data = [
                {"candidates": [{"content": {"parts": [{"text": "OK"}]}, "finishReason": "STOP"}]}
            ]
        else:
            assert body["reasoning_effort"] == "high"
            data = [{"choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}]}]
        if fail:
            data.append(error)
        return web.Response(
            text="".join(f"data: {json.dumps(d)}\n\n" for d in data) + "data: [DONE]\n\n",
            content_type="text/event-stream",
        )

    app = web.Application()
    app.router.add_post("/{path:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        async with LLMClient(
            provider=provider, base_url=f"http://127.0.0.1:{port}/v1", api_key="k", model=model
        ) as client:
            events = []

            async def collect():
                async for event in client.chat_stream(MESSAGES, reasoning=Reasoning(effort="high")):
                    events.append(event)

            if fail:
                from flexllm import LLMResponseError

                with pytest.raises(LLMResponseError, match="overloaded") as info:
                    await collect()
                assert info.value.response_data == error
                assert not any(e["type"] == "result" for e in events)
            else:
                await collect()
                assert events[-1]["result"].content == "OK"
    finally:
        await runner.cleanup()


def test_gemini_never_drops_effort_when_budget_is_also_supplied():
    with GeminiClient(
        api_key="test",
        model="gemini-3-flash",
        reasoning_capabilities={"supports_effort_and_budget": True},
    ) as client:
        with pytest.raises(ValueError, match="cannot combine"):
            client._prepare_reasoning_kwargs(
                "gemini-3-flash", {"reasoning": Reasoning(effort="high", budget_tokens=4096)}
            )


def test_cli_preserves_capabilities_when_provider_supplies_default_url(configured_cli):
    app, path, cfg = configured_cli
    entry = cfg.config["models"][0]
    entry.pop("base_url")
    entry.update(provider="claude", id="claude-opus-5-5")
    path.write_text(yaml.safe_dump(cfg.config))
    with LLMClient.from_config(str(path), model="b") as client:
        assert client.capabilities.reasoning.effort_levels == ("high", "max")
    for command in ("ask", "chat", "serve", "chat-web"):
        args = [command, "hello"] if command in ("ask", "chat") else [command]
        result = CliRunner().invoke(
            app, [*args, "-m", "b", "--reasoning-effort", "medium", "--dry-run"]
        )
        assert result.exit_code == 2, result.output
        assert "high, max" in result.output


@pytest.mark.asyncio
async def test_typed_per_row_policy_is_serializable_in_checkpoint(tmp_path):
    policy = Reasoning(effort="high")
    params = [{"reasoning": policy}]
    output = tmp_path / "results.jsonl"
    async with OpenAIClient(base_url="https://example.com/v1", model="m") as client:
        cache = MagicMock()
        cache.get_batch.return_value = ([{"content": "OK"}], [])
        client._response_cache = cache
        result = await client.chat_batch(
            [MESSAGES], params_list=params, output_jsonl=str(output), show_progress=False
        )
        assert result[0].content == "OK"
    record = json.loads(output.read_text())
    assert record["params"] == {"reasoning": {"effort": "high"}}
    assert params[0]["reasoning"] is policy


def test_equivalent_endpoint_url_preserves_adapter_and_capabilities(configured_cli):
    _, path, cfg = configured_cli
    entry = cfg.config["models"][0]
    entry.update(base_url="https://gateway.example/v1", reasoning_adapter="siliconflow")
    path.write_text(yaml.safe_dump(cfg.config))
    with LLMClient.from_config(
        str(path), model="b", base_url="https://gateway.example/v1/"
    ) as client:
        assert client.capabilities.reasoning.effort_levels == ("high", "max")
        assert client._single_client._reasoning_adapter == "siliconflow"


def test_explicit_on_with_effort_needs_no_separate_openai_toggle():
    with OpenAIClient(
        base_url="https://gateway.example/v1",
        model="m",
        reasoning_capabilities={"can_enable": False, "effort_levels": ["high"]},
    ) as client:
        assert client._prepare_reasoning_kwargs(
            "m", {"reasoning": Reasoning(enabled=True, effort="high")}
        ) == {"reasoning_effort": "high"}


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [Reasoning(effort="high"), Reasoning(budget_tokens=4096)])
async def test_custom_claude_alias_uses_declared_capabilities(monkeypatch, policy):
    async with ClaudeClient(
        base_url="https://gateway.example/v1",
        api_key="test",
        model="claude-corporate-alias",
        reasoning_capabilities={
            "effort_levels": ["high"],
            "budget_tokens": {"min": 1024, "max": 8192},
        },
    ) as client:

        async def request(**kwargs):
            body = kwargs["request_params"][0]["json"]
            assert "_reasoning_validated" not in body
            if policy.effort is not None:
                assert body["thinking"] == {"type": "adaptive"}
                assert body["output_config"]["effort"] == "high"
            else:
                assert body["thinking"] == {"type": "enabled", "budget_tokens": 4096}
                assert body["max_tokens"] > 4096
            payload = {"content": [{"type": "text", "text": "OK"}]}
            return [RequestResult(0, payload, "success", 0.0)], None

        monkeypatch.setattr(client._client, "process_requests", request)
        assert (await client.chat(MESSAGES, reasoning=policy)).content == "OK"
        with pytest.raises(ValueError, match="does not support extended thinking"):
            await client.chat(MESSAGES, thinking="high")


@pytest.mark.parametrize("cls", [OpenAIClient, ClaudeClient, GeminiClient])
@pytest.mark.parametrize("entrypoint", ["chat", "stream", "batch", "batch_row"])
@pytest.mark.parametrize("use_default", [False, True])
async def test_request_url_override_cannot_reuse_reasoning_binding(cls, entrypoint, use_default):
    policy = Reasoning(effort="high")
    async with cls(
        api_key="test",
        base_url="https://original.example/v1",
        model="m",
        reasoning=policy if use_default else None,
    ) as client:
        kwargs = {"url": "https://different.example/v1/chat/completions"}
        if not use_default:
            if entrypoint == "batch_row":
                kwargs["params_list"] = [{"reasoning": policy}]
            else:
                kwargs["reasoning"] = policy
        with pytest.raises(ValueError, match="different request url"):
            if entrypoint == "chat":
                await client.chat(MESSAGES, **kwargs)
            elif entrypoint == "stream":
                async for _ in client.chat_stream(MESSAGES, **kwargs):
                    pytest.fail("URL must be checked before streaming")
            else:
                await client.chat_batch([MESSAGES], **kwargs)


@pytest.mark.parametrize("cls", [OpenAIClient, ClaudeClient, GeminiClient])
def test_equivalent_request_url_and_empty_policy_remain_allowed(cls):
    with cls(
        api_key="test",
        base_url="https://original.example/v1",
        model="m",
        reasoning=Reasoning(effort="high"),
    ) as client:
        for stream in (False, True):
            url = client._get_stream_url("m") if stream else client._get_url("m")
            assert client._prepare_reasoning_kwargs("m", {}, url=url, stream=stream)
        assert (
            client._prepare_reasoning_kwargs(
                "m", {"reasoning": None}, url="https://different.example/v1"
            )
            == {}
        )
