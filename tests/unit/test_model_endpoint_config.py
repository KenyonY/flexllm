"""Model-level endpoints must reach the pool without becoming generation parameters."""

import copy
import json

import pytest
import typer
import yaml
from typer.testing import CliRunner

from flexllm import LLMClient
from flexllm.cli import config as config_module
from flexllm.cli.commands import register_commands
from flexllm.cli.config import FlexLLMConfig, model_client_kwargs


@pytest.fixture
def app():
    application = typer.Typer()
    register_commands(application)
    return application


@pytest.fixture
def pool_config(tmp_path, monkeypatch):
    entry = {
        "name": "replicas",
        "id": "qwen",
        "api_key": "shared-key",
        "provider": "openai",
        "fallback": False,
        "system": "Be concise.",
        "user_template": "Question: {content}",
        "temperature": 0.2,
        "endpoints": [
            {"base_url": "http://node-a.example/v1", "concurrency_limit": 2},
            {"base_url": "http://node-b.example/v1", "model": "qwen-b", "api_key": "b-key"},
        ],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"default": "replicas", "models": [entry]}))
    for key in config_module.os.environ.copy():
        if key.startswith(("FLEXLLM_", "OPENAI_")):
            monkeypatch.delenv(key)
    cfg = FlexLLMConfig(path)
    monkeypatch.setattr(config_module, "_config", cfg)
    return path, cfg, entry


@pytest.mark.asyncio
async def test_from_yaml_preserves_pool_defaults_and_endpoint_overrides(pool_config):
    path, cfg, entry = pool_config
    before = copy.deepcopy(cfg.config)
    async with LLMClient.from_config(str(path), model="replicas") as client:
        assert client._mode == "multi"
        assert client._fallback is False
        assert [ep.model for ep in client._endpoints] == ["qwen", "qwen-b"]
        assert [ep.api_key for ep in client._endpoints] == ["shared-key", "b-key"]
        assert client._endpoints[0].concurrency_limit == 2
        assert all(ep.provider == "openai" for ep in client._endpoints)
        assert client._config_system == entry["system"]
        assert client._config_user_template == entry["user_template"]
        assert client._config_params == {"temperature": 0.2}
    assert cfg.config == before


@pytest.mark.asyncio
async def test_from_config_overrides_routing_and_fallback(pool_config):
    path, _, _ = pool_config
    async with LLMClient.from_config(str(path), fallback=True, api_key="override-key") as pool:
        assert pool._fallback is True
        assert all(ep.api_key == "override-key" for ep in pool._endpoints)
    async with LLMClient.from_config(
        str(path), base_url="http://gateway.example/v1", api_key="gateway-key"
    ) as client:
        assert client._mode == "single"
        assert client._single_client._base_url == "http://gateway.example/v1"


def test_explicit_endpoints_replace_single_target():
    entry = {"id": "qwen", "base_url": "http://old.example/v1"}
    options = model_client_kwargs(entry, endpoints=[{"base_url": "http://new.example/v1"}])
    assert "base_url" not in options
    assert options["endpoints"][0]["model"] == "qwen"
    assert entry == {"id": "qwen", "base_url": "http://old.example/v1"}


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"endpoints": []}, "非空列表"),
        ({"endpoints": "http://node.example/v1"}, "非空列表"),
        ({"endpoints": [None]}, "包含 base_url"),
        ({"endpoints": [{}]}, "包含 base_url"),
        ({"base_url": "http://node.example/v1"}, "不能同时配置"),
        ({"fallback": "false"}, "布尔值"),
    ],
)
def test_invalid_model_pool_is_rejected(pool_config, extra, message):
    _, _, entry = pool_config
    with pytest.raises(ValueError, match=message):
        model_client_kwargs({**entry, **extra})


@pytest.mark.parametrize("command", ["ask", "chat"])
def test_cli_pool_preview_and_explicit_gateway(pool_config, app, command):
    result = CliRunner().invoke(app, [command, "hello", "-m", "replicas", "--dry-run"])
    assert result.exit_code == 10, result.output
    assert json.loads(result.stdout)["endpoint_count"] == 2
    assert "shared-key" not in result.output
    result = CliRunner().invoke(
        app,
        [command, "hello", "-m", "replicas", "--base-url", "http://gateway/v1", "--dry-run"],
    )
    assert result.exit_code == 10, result.output
    assert json.loads(result.stdout)["base_url"] == "http://gateway/v1"
    assert json.loads(result.stdout)["endpoint_count"] == 1


def test_cli_invalid_pool_reports_usage_error(pool_config, app):
    _, cfg, _ = pool_config
    cfg.config["models"][0]["endpoints"] = []
    result = CliRunner().invoke(app, ["ask", "hello", "-m", "replicas", "--dry-run"])
    assert result.exit_code == 2, result.output
    assert "非空列表" in result.output


@pytest.mark.parametrize("selection", [[], ["-m", "replicas"]])
def test_batch_named_pool_preview(pool_config, tmp_path, app, selection):
    data = tmp_path / "input.jsonl"
    data.write_text('{"prompt":"hello"}\n')
    result = CliRunner().invoke(app, ["batch", str(data), *selection, "--dry-run"])
    assert result.exit_code == 10, result.output
    assert json.loads(result.stdout)["endpoint_count"] == 2


@pytest.mark.parametrize("provider", ["claude", "gemini"])
def test_batch_pool_gateway_override_preserves_protocol_and_proxy(
    pool_config, tmp_path, app, monkeypatch, provider
):
    _, cfg, _ = pool_config
    cfg.config["models"][0].update(provider=provider, proxy="http://proxy.example:8080")
    options = {}

    class Client:
        def __init__(self, **kwargs):
            options.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def _run_batch(self, **kwargs):
            from flexllm.clients.base import _BatchRun

            assert "proxy" not in kwargs
            return _BatchRun(responses=[], errors={}, summary={}, cost=None, elapsed=0.0)

    monkeypatch.setattr("flexllm.LLMClient", Client)
    source = tmp_path / "input.jsonl"
    source.write_text('{"prompt":"hello"}\n')
    result = CliRunner().invoke(
        app, ["batch", str(source), "-m", "replicas", "--base-url", "http://gateway.example/v1"]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert options["base_url"] == "http://gateway.example/v1"
    assert options["provider"] == provider
    assert options["proxy"] == "http://proxy.example:8080"
    assert "endpoints" not in options
