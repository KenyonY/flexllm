"""CLI 配置 get_model_params 测试"""

from unittest.mock import patch

import pytest

from flexllm.cli.config import FlexLLMConfig


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
