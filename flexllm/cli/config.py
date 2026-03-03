"""CLI 配置管理"""

from __future__ import annotations

import os
from pathlib import Path


class FlexLLMConfig:
    """配置管理"""

    # 元信息字段，不作为模型调用参数传递
    META_FIELDS = {"id", "name", "provider", "base_url", "api_key", "system", "user_template"}

    CONFIG_PATHS = [
        Path("flexllm_config.yaml"),
        Path("~/.flexllm/config.yaml").expanduser(),
    ]

    def __init__(self):
        from dotenv import load_dotenv

        load_dotenv()
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """加载配置文件"""
        # 尝试从配置文件加载
        for path in self.CONFIG_PATHS:
            if path.exists():
                try:
                    import yaml

                    with open(path, encoding="utf-8") as f:
                        config = yaml.safe_load(f) or {}
                    return config
                except ImportError:
                    pass

        # 从环境变量构建配置
        return self._config_from_env()

    def _config_from_env(self) -> dict:
        """从环境变量构建配置"""
        base_url = os.environ.get("FLEXLLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get("FLEXLLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        model = os.environ.get("FLEXLLM_MODEL") or os.environ.get("OPENAI_MODEL")

        if base_url or api_key or model:
            return {
                "models": [
                    {
                        "id": model or "default",
                        "name": model or "default",
                        "base_url": base_url,
                        "api_key": api_key or "EMPTY",
                    }
                ],
                "default": model or "default",
            }
        return {}

    def get_model_config(self, model_name_or_id: str = None) -> dict | None:
        """获取模型配置"""
        models = self.config.get("models", [])

        # 先检查环境变量
        env_base_url = os.environ.get("FLEXLLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        env_api_key = os.environ.get("FLEXLLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        env_model = os.environ.get("FLEXLLM_MODEL") or os.environ.get("OPENAI_MODEL")

        if model_name_or_id:
            # 精确匹配 name 或 id
            for m in models:
                if m.get("name") == model_name_or_id or m.get("id") == model_name_or_id:
                    result = dict(m)
                    # 环境变量覆盖
                    if env_base_url:
                        result["base_url"] = env_base_url
                    if env_api_key:
                        result["api_key"] = env_api_key
                    if env_model:
                        result["id"] = env_model
                    return result
            # 如果有环境变量，作为 fallback 返回
            if env_base_url:
                return {
                    "id": model_name_or_id,
                    "name": model_name_or_id,
                    "base_url": env_base_url,
                    "api_key": env_api_key or "EMPTY",
                }
            return None
        else:
            # 使用默认模型
            default = self.config.get("default")
            if default:
                for m in models:
                    if m.get("name") == default or m.get("id") == default:
                        result = dict(m)
                        if env_base_url:
                            result["base_url"] = env_base_url
                        if env_api_key:
                            result["api_key"] = env_api_key
                        if env_model:
                            result["id"] = env_model
                        return result
            # 返回第一个模型
            if models:
                result = dict(models[0])
                if env_base_url:
                    result["base_url"] = env_base_url
                if env_api_key:
                    result["api_key"] = env_api_key
                if env_model:
                    result["id"] = env_model
                return result
            # 环境变量兜底
            if env_base_url:
                return {
                    "id": env_model or "default",
                    "name": env_model or "default",
                    "base_url": env_base_url,
                    "api_key": env_api_key or "EMPTY",
                }
            return None

    def get_config_path(self) -> Path | None:
        """获取当前使用的配置文件路径"""
        for path in self.CONFIG_PATHS:
            if path.exists():
                return path
        return None

    def _resolve_model_name(self, model_name_or_id: str = None) -> str | None:
        """解析模型名称，未指定时使用默认模型"""
        return model_name_or_id or self.config.get("default")

    def get_system(self, model_name_or_id: str = None) -> str | None:
        """获取系统提示词

        优先级:
        1. 模型级别 system
        2. 全局 system
        3. None
        """
        model_name_or_id = self._resolve_model_name(model_name_or_id)
        # 模型级别
        if model_name_or_id:
            for m in self.config.get("models", []):
                if m.get("name") == model_name_or_id or m.get("id") == model_name_or_id:
                    if "system" in m:
                        return m["system"]

        # 全局级别
        return self.config.get("system")

    def get_user_template(self, model_name_or_id: str = None) -> str | None:
        """获取 user content 模板

        优先级:
        1. 模型级别 user_template
        2. 全局 user_template
        3. None
        """
        model_name_or_id = self._resolve_model_name(model_name_or_id)
        # 模型级别
        if model_name_or_id:
            for m in self.config.get("models", []):
                if m.get("name") == model_name_or_id or m.get("id") == model_name_or_id:
                    if "user_template" in m:
                        return m["user_template"]

        # 全局级别
        return self.config.get("user_template")

    def get_model_params(self, model_name_or_id: str = None) -> dict:
        """获取模型调用参数

        返回模型配置中除元信息字段外的所有字段，作为 chat_completions 的 kwargs
        """
        model_name_or_id = self._resolve_model_name(model_name_or_id)
        if model_name_or_id:
            for m in self.config.get("models", []):
                if m.get("name") == model_name_or_id or m.get("id") == model_name_or_id:
                    return {k: v for k, v in m.items() if k not in self.META_FIELDS}
        return {}

    def get_batch_config(self) -> dict:
        """获取 batch 命令的配置（配置文件中的 batch 节 + 默认值）"""
        batch_config = self.config.get("batch", {}) or {}

        defaults = {
            # 缓存
            "cache": False,
            "cache_ttl": 86400,
            # 网络
            "concurrency": 10,
            "max_qps": None,
            "timeout": 120,
            "retry_times": 3,
            "retry_delay": 1.0,
            # 采样
            "temperature": None,
            "max_tokens": None,
            "top_p": None,
            "top_k": None,
            # 思考
            "thinking": None,
            # 处理
            "preprocess_msg": False,
            "flush_interval": 1.0,
            # 输出
            "return_usage": True,
            "track_cost": True,
            # 多 endpoint
            "endpoints": None,
            "fallback": True,
        }

        result = {}
        for key, default_value in defaults.items():
            result[key] = batch_config.get(key, default_value)

        return result


# 全局配置实例
_config: FlexLLMConfig | None = None


def get_config() -> FlexLLMConfig:
    global _config
    if _config is None:
        _config = FlexLLMConfig()
    return _config
