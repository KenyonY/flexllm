"""CLI 配置 get_model_params / get_model_config 测试"""

import pytest

from flexllm.cli.config import FlexLLMConfig

ENV_VARS = [
    "FLEXLLM_BASE_URL",
    "FLEXLLM_API_KEY",
    "FLEXLLM_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
]


def _clear_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _make_config(models, default=None, **extra):
    cfg = FlexLLMConfig.__new__(FlexLLMConfig)
    cfg.config = {"models": models, **extra}
    if default:
        cfg.config["default"] = default
    cfg._config_path = None
    cfg._loaded_path = None
    return cfg


class TestGetModelParams:
    """测试 get_model_params 元字段排除"""

    def _make_config(self, models):
        """构造带 models 的 config 对象"""
        cfg = FlexLLMConfig.__new__(FlexLLMConfig)
        cfg.config = {"models": models}
        return cfg

    def test_meta_fields_excluded(self):
        """元字段（id/name/provider/base_url/api_key/system/user_template）不出现在结果中"""
        cfg = self._make_config(
            [
                {
                    "id": "gpt-4",
                    "name": "gpt-4",
                    "provider": "openai",
                    "base_url": "http://api.openai.com/v1",
                    "api_key": "sk-xxx",
                    "system": "你是助手",
                    "user_template": "{content}",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                }
            ]
        )
        params = cfg.get_model_params("gpt-4")
        for field in FlexLLMConfig.META_FIELDS:
            assert field not in params
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 4096

    def test_model_not_found_returns_empty(self):
        """模型不存在时返回空 dict"""
        cfg = self._make_config(
            [
                {"id": "gpt-4", "name": "gpt-4", "temperature": 0.5},
            ]
        )
        params = cfg.get_model_params("nonexistent-model")
        assert params == {}

    def test_match_by_id(self):
        """通过 id 匹配模型"""
        cfg = self._make_config(
            [
                {"id": "my-model-id", "name": "My Model", "top_p": 0.9},
            ]
        )
        params = cfg.get_model_params("my-model-id")
        assert params["top_p"] == 0.9

    def test_match_by_name(self):
        """通过 name 匹配模型"""
        cfg = self._make_config(
            [
                {"id": "some-id", "name": "My Model", "top_k": 50},
            ]
        )
        params = cfg.get_model_params("My Model")
        assert params["top_k"] == 50

    def test_endpoints_and_fallback_excluded(self):
        """回归：pool 型模型的 endpoints/fallback 是元信息，不得泄进请求参数"""
        cfg = self._make_config(
            [
                {
                    "id": "pool-model",
                    "name": "pool-model",
                    "endpoints": [{"base_url": "http://a/v1"}, {"base_url": "http://b/v1"}],
                    "fallback": True,
                    "temperature": 0.3,
                }
            ]
        )
        params = cfg.get_model_params("pool-model")
        assert "endpoints" not in params
        assert "fallback" not in params
        assert params == {"temperature": 0.3}


class TestGetModelConfigPriority:
    """回归：环境变量不得覆盖显式选中的配置文件具名模型

    优先级链: CLI 参数 > 配置文件具名模型 > 环境变量 > 配置文件 default
    """

    def _make(self, models, default=None):
        return _make_config(models, default=default)

    def test_explicit_model_not_overridden_by_env(self, monkeypatch):
        """-m 显式选中的模型不被 OPENAI_*/FLEXLLM_* 环境变量覆盖"""
        _clear_env(monkeypatch)
        monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.com/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
        monkeypatch.setenv("OPENAI_MODEL", "env-model")
        cfg = self._make(
            [
                {
                    "id": "deepseek-chat",
                    "name": "ds",
                    "base_url": "https://api.deepseek.com/v1",
                    "api_key": "sk-deepseek",
                }
            ]
        )
        mc = cfg.get_model_config("ds")
        assert mc["base_url"] == "https://api.deepseek.com/v1"
        assert mc["api_key"] == "sk-deepseek"
        assert mc["id"] == "deepseek-chat"

    def test_env_as_default_when_no_model_selected(self, monkeypatch):
        """未显式选择模型时，环境变量优先于配置文件 default"""
        _clear_env(monkeypatch)
        monkeypatch.setenv("FLEXLLM_BASE_URL", "http://env-host/v1")
        monkeypatch.setenv("FLEXLLM_API_KEY", "sk-env")
        cfg = self._make(
            [{"id": "m1", "name": "m1", "base_url": "http://cfg-host/v1"}], default="m1"
        )
        mc = cfg.get_model_config(None)
        assert mc["base_url"] == "http://env-host/v1"
        assert mc["api_key"] == "sk-env"

    def test_env_model_matches_config_entry(self, monkeypatch):
        """FLEXLLM_MODEL 命中配置文件具名模型时等同 -m 选中"""
        _clear_env(monkeypatch)
        monkeypatch.setenv("FLEXLLM_MODEL", "m2")
        cfg = self._make(
            [
                {"id": "m1", "name": "m1", "base_url": "http://one/v1", "api_key": "k1"},
                {"id": "m2", "name": "m2", "base_url": "http://two/v1", "api_key": "k2"},
            ],
            default="m1",
        )
        mc = cfg.get_model_config(None)
        assert mc["id"] == "m2"
        assert mc["base_url"] == "http://two/v1"

    def test_config_default_when_no_env(self, monkeypatch):
        """无环境变量时用配置文件 default"""
        _clear_env(monkeypatch)
        cfg = self._make(
            [
                {"id": "m1", "name": "m1", "base_url": "http://one/v1"},
                {"id": "m2", "name": "m2", "base_url": "http://two/v1"},
            ],
            default="m2",
        )
        mc = cfg.get_model_config(None)
        assert mc["id"] == "m2"

    def test_named_model_env_fallback_when_not_in_config(self, monkeypatch):
        """-m 未命中配置文件时，环境变量端点作为 fallback（模型名透传）"""
        _clear_env(monkeypatch)
        monkeypatch.setenv("FLEXLLM_BASE_URL", "http://env-host/v1")
        cfg = self._make([{"id": "m1", "name": "m1", "base_url": "http://one/v1"}])
        mc = cfg.get_model_config("unregistered")
        assert mc["id"] == "unregistered"
        assert mc["base_url"] == "http://env-host/v1"


class TestConfigPathFailFast:
    """回归：显式指定的配置路径不存在时 fail fast，而不是静默回落环境变量"""

    def test_explicit_config_path_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FlexLLMConfig(tmp_path / "nonexistent.yaml")

    def test_get_config_path_reflects_loaded_source(self, tmp_path):
        p = tmp_path / "custom.yaml"
        p.write_text("default: m1\nmodels:\n  - id: m1\n    base_url: http://x/v1\n")
        cfg = FlexLLMConfig(p)
        assert cfg.get_config_path() == p
        assert cfg.config["default"] == "m1"
