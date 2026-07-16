"""Claude Code 配置兼容性测试"""

import json
import os
from pathlib import Path
from unittest.mock import patch

from flexllm.cli.config import FlexLLMConfig


class TestExpandEnvVars:
    """测试 _expand_env_vars"""

    def test_simple_var(self):
        with patch.dict(os.environ, {"MY_TOKEN": "abc123"}):
            assert FlexLLMConfig._expand_env_vars("${MY_TOKEN}") == "abc123"

    def test_var_with_default(self):
        # 变量不存在，使用默认值
        env = dict(os.environ)
        env.pop("MISSING_VAR", None)
        with patch.dict(os.environ, env, clear=True):
            assert FlexLLMConfig._expand_env_vars("${MISSING_VAR:-fallback}") == "fallback"

    def test_var_exists_ignores_default(self):
        with patch.dict(os.environ, {"EXISTS": "real"}):
            assert FlexLLMConfig._expand_env_vars("${EXISTS:-fallback}") == "real"

    def test_no_vars(self):
        assert FlexLLMConfig._expand_env_vars("plain text") == "plain text"

    def test_multiple_vars(self):
        with patch.dict(os.environ, {"A": "1", "B": "2"}):
            assert FlexLLMConfig._expand_env_vars("${A}-${B}") == "1-2"

    def test_empty_default(self):
        env = dict(os.environ)
        env.pop("X", None)
        with patch.dict(os.environ, env, clear=True):
            assert FlexLLMConfig._expand_env_vars("${X:-}") == ""

    def test_unknown_var_no_default_kept(self):
        """未设置且无默认值的变量保持原样"""
        env = dict(os.environ)
        env.pop("UNKNOWN", None)
        with patch.dict(os.environ, env, clear=True):
            assert FlexLLMConfig._expand_env_vars("${UNKNOWN}") == "${UNKNOWN}"


class TestLoadClaudeMcpServers:
    """测试 _load_claude_mcp_servers"""

    def test_global_mcp(self, tmp_path, monkeypatch):
        """~/.claude.json 全局 mcpServers"""
        claude_json = tmp_path / ".claude.json"
        claude_json.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "github": {
                            "command": "npx",
                            "args": ["-y", "@mcp/server-github"],
                        }
                    }
                }
            )
        )
        monkeypatch.setattr(Path, "expanduser", lambda self: tmp_path / self.name)
        monkeypatch.chdir(tmp_path)

        result = FlexLLMConfig._load_claude_mcp_servers()
        assert "github" in result
        assert result["github"]["command"] == "npx"

    def test_project_mcp_overrides_global(self, tmp_path, monkeypatch):
        """~/.claude.json 项目级 mcpServers 覆盖全局"""
        monkeypatch.chdir(tmp_path)
        claude_json = tmp_path / ".claude.json"
        claude_json.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "db": {"command": "global-db"},
                    },
                    "projects": {
                        str(tmp_path): {
                            "mcpServers": {
                                "db": {"command": "project-db"},
                            }
                        }
                    },
                }
            )
        )
        monkeypatch.setattr(Path, "expanduser", lambda self: tmp_path / self.name)

        result = FlexLLMConfig._load_claude_mcp_servers()
        assert result["db"]["command"] == "project-db"

    def test_mcp_json_overrides_claude_json(self, tmp_path, monkeypatch):
        """.mcp.json 优先级高于 ~/.claude.json"""
        monkeypatch.chdir(tmp_path)

        # ~/.claude.json
        claude_json = tmp_path / ".claude.json"
        claude_json.write_text(json.dumps({"mcpServers": {"srv": {"command": "from-claude"}}}))
        monkeypatch.setattr(Path, "expanduser", lambda self: tmp_path / self.name)

        # .mcp.json
        mcp_json = tmp_path / ".mcp.json"
        mcp_json.write_text(json.dumps({"mcpServers": {"srv": {"command": "from-mcp-json"}}}))

        result = FlexLLMConfig._load_claude_mcp_servers()
        assert result["srv"]["command"] == "from-mcp-json"

    def test_no_claude_files(self, tmp_path, monkeypatch):
        """无 Claude 配置文件时返回空"""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(Path, "expanduser", lambda self: tmp_path / self.name)
        result = FlexLLMConfig._load_claude_mcp_servers()
        assert result == {}


class TestGetAgentConfigMerge:
    """测试 MCP 合并优先级：flexllm > .mcp.json > claude.json"""

    def test_flexllm_overrides_claude(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        # Claude Code 配置
        claude_json = tmp_path / ".claude.json"
        claude_json.write_text(json.dumps({"mcpServers": {"srv": {"command": "claude-cmd"}}}))
        monkeypatch.setattr(Path, "expanduser", lambda self: tmp_path / self.name)

        # flexllm 配置 - 同名 server
        config = FlexLLMConfig.__new__(FlexLLMConfig)
        config._config_path = None
        config.config = {"agent": {"mcp_servers": {"srv": {"command": "flexllm-cmd"}}}}

        result = config.get_agent_config()
        mcp_list = result["mcp_servers"]
        srv = next(s for s in mcp_list if s["name"] == "srv")
        assert srv["command"] == "flexllm-cmd"

    def test_claude_only_added_when_not_in_flexllm(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        claude_json = tmp_path / ".claude.json"
        claude_json.write_text(json.dumps({"mcpServers": {"extra": {"command": "claude-extra"}}}))
        monkeypatch.setattr(Path, "expanduser", lambda self: tmp_path / self.name)

        config = FlexLLMConfig.__new__(FlexLLMConfig)
        config._config_path = None
        config.config = {"agent": {"mcp_servers": {"existing": {"command": "flexllm-existing"}}}}

        result = config.get_agent_config()
        mcp_list = result["mcp_servers"]
        names = [s["name"] for s in mcp_list]
        assert "existing" in names
        assert "extra" in names
