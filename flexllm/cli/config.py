"""CLI 配置管理"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


class FlexLLMConfig:
    """配置管理"""

    # 元信息字段，不作为模型调用参数传递
    META_FIELDS = {"id", "name", "provider", "base_url", "api_key", "system", "user_template"}

    CONFIG_PATHS = [
        Path("flexllm_config.yaml"),
        Path("~/.flexllm/config.yaml").expanduser(),
    ]

    def __init__(self, config_path: str | Path = None):
        from dotenv import load_dotenv

        load_dotenv()
        self._config_path = Path(config_path) if config_path else None
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """加载配置文件"""
        # 优先使用指定的配置文件路径
        paths = [self._config_path] if self._config_path else self.CONFIG_PATHS
        for path in paths:
            if path.exists():
                try:
                    import yaml
                except ImportError:
                    raise ImportError(
                        f"找到配置文件 {path}，但缺少 PyYAML。" "请安装: pip install pyyaml"
                    )
                with open(path, encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                return config

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

    def get_agent_config(self) -> dict:
        """获取 agent 配置（合并全局 + 项目级）

        全局配置 (~/.flexllm/config.yaml):
            agent:
              mcp_servers:
                github:
                  command: npx
                  args: ["-y", "@mcp/server-github"]
                  env:
                    GITHUB_TOKEN: "xxx"
                local:
                  type: http
                  url: "http://localhost:8080/sse"
                  headers:
                    Authorization: "Bearer xxx"

        也支持简写（command 为完整命令字符串）:
            agent:
              mcp_servers:
                github:
                  command: "npx -y @mcp/server-github"

        项目配置 (.flexllm/settings.yaml，从 cwd 向上搜索):
            mcp_servers:
              db:
                command: npx
                args: ["@mcp/server-postgres"]

        合并规则：项目级覆盖全局同名 server。

        Returns:
            {"mcp_servers": [...]}  # 转换为 list[dict] 格式供 _connect_mcp_servers 使用
        """
        # 全局 agent 配置
        agent_config = self.config.get("agent", {}) or {}
        global_mcp = agent_config.get("mcp_servers", {}) or {}

        # 项目级配置
        project_settings = self._load_project_settings()
        project_mcp = project_settings.get("mcp_servers", {}) or {}

        # 合并：项目级覆盖全局
        merged_mcp = {}
        if isinstance(global_mcp, dict):
            merged_mcp.update(global_mcp)
        elif isinstance(global_mcp, list):
            # 兼容旧的 list 格式
            for i, item in enumerate(global_mcp):
                name = item.get("name", f"server-{i}")
                merged_mcp[name] = item
        if isinstance(project_mcp, dict):
            merged_mcp.update(project_mcp)
        elif isinstance(project_mcp, list):
            for i, item in enumerate(project_mcp):
                name = item.get("name", f"project-server-{i}")
                merged_mcp[name] = item

        # Claude Code 兼容：读取 Claude Code MCP 配置作为 fallback
        claude_mcp = self._load_claude_mcp_servers()
        for name, spec in claude_mcp.items():
            if name not in merged_mcp:
                merged_mcp[name] = spec

        # 转换为 list[dict] 格式，name 字段来自 key
        mcp_list = []
        for name, spec in merged_mcp.items():
            if isinstance(spec, dict):
                entry = dict(spec)
                entry.setdefault("name", name)
                # 展开 env 中的 ${VAR} 语法
                if "env" in entry and isinstance(entry["env"], dict):
                    entry["env"] = {
                        k: self._expand_env_vars(v) if isinstance(v, str) else v
                        for k, v in entry["env"].items()
                    }
                mcp_list.append(entry)

        return {"mcp_servers": mcp_list}

    @staticmethod
    def _load_project_settings() -> dict:
        """从 cwd 向上搜索 .flexllm/settings.yaml 项目配置"""
        current = Path.cwd()
        for parent in [current, *current.parents]:
            candidate = parent / ".flexllm" / "settings.yaml"
            if candidate.is_file():
                try:
                    import yaml

                    with open(candidate, encoding="utf-8") as f:
                        return yaml.safe_load(f) or {}
                except Exception:
                    return {}
        return {}

    @staticmethod
    def _load_claude_mcp_servers() -> dict:
        """从 Claude Code 配置读取 MCP servers

        读取顺序（低 → 高优先级）：
        1. ~/.claude.json → mcpServers（全局）
        2. ~/.claude.json → projects[cwd].mcpServers（项目级）
        3. .mcp.json（项目根目录，向上搜索）
        """
        merged = {}

        # 1 & 2: ~/.claude.json
        claude_json = Path("~/.claude.json").expanduser()
        if claude_json.is_file():
            try:
                data = json.loads(claude_json.read_text(encoding="utf-8"))
                # 全局 mcpServers
                merged.update(data.get("mcpServers", {}))
                # 项目级 mcpServers
                cwd = str(Path.cwd())
                project = data.get("projects", {}).get(cwd, {})
                merged.update(project.get("mcpServers", {}))
            except (json.JSONDecodeError, OSError):
                pass

        # 3: .mcp.json（从 cwd 向上搜索）
        current = Path.cwd()
        for parent in [current, *current.parents]:
            mcp_json = parent / ".mcp.json"
            if mcp_json.is_file():
                try:
                    data = json.loads(mcp_json.read_text(encoding="utf-8"))
                    merged.update(data.get("mcpServers", {}))
                except (json.JSONDecodeError, OSError):
                    pass
                break

        return merged

    @staticmethod
    def _expand_env_vars(value: str) -> str:
        """展开 ${VAR} 和 ${VAR:-default} 语法"""

        def _replace(match):
            var_name = match.group(1)
            default = match.group(3)  # group(3) 是 :- 后的默认值
            return os.environ.get(var_name, default if default is not None else match.group(0))

        return re.sub(r"\$\{([^}:]+)(?:(:-)(.*?))?\}", _replace, value)

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
