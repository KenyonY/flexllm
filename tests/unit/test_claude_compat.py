"""Claude Code 配置兼容性测试"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

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


class TestLoadProjectInstructions:
    """测试项目指令 fallback"""

    def test_flexllm_md_priority(self, tmp_path, monkeypatch):
        """.flexllm.md 优先于 CLAUDE.md"""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".flexllm.md").write_text("flexllm instructions")
        (tmp_path / "CLAUDE.md").write_text("claude instructions")

        from flexllm.cli.chat_helpers import load_project_instructions

        result = load_project_instructions()
        assert result == "flexllm instructions"

    def test_claude_md_fallback(self, tmp_path, monkeypatch):
        """无 .flexllm.md 时 fallback 到 CLAUDE.md"""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "CLAUDE.md").write_text("claude instructions")

        from flexllm.cli.chat_helpers import load_project_instructions

        result = load_project_instructions()
        assert result == "claude instructions"

    def test_no_instructions(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from flexllm.cli.chat_helpers import load_project_instructions

        result = load_project_instructions()
        assert result is None


class TestLoadSkillMultiPath:
    """测试 skills 多路径搜索"""

    def test_flexllm_global_priority(self, tmp_path, monkeypatch):
        """~/.flexllm/skills 优先"""
        import flexllm.cli.chat_helpers as ch

        # 设置多个 skills 目录
        flexllm_skills = tmp_path / "flexllm_skills"
        claude_skills = tmp_path / "claude_skills"
        flexllm_skills.mkdir()
        claude_skills.mkdir()

        # 两个目录都有同名 skill
        (flexllm_skills / "test-skill.md").write_text(
            "---\nname: test-skill\ndescription: from flexllm\n---\nFlexLLM content"
        )
        (claude_skills / "test-skill.md").write_text(
            "---\nname: test-skill\ndescription: from claude\n---\nClaude content"
        )

        original_dirs = ch.SKILLS_DIRS
        monkeypatch.setattr(ch, "SKILLS_DIRS", [flexllm_skills, claude_skills])

        result = ch.load_skill("test-skill")
        assert result is not None
        assert result["content"] == "FlexLLM content"

        monkeypatch.setattr(ch, "SKILLS_DIRS", original_dirs)

    def test_fallback_to_claude_skills(self, tmp_path, monkeypatch):
        """flexllm 没有时 fallback 到 claude skills"""
        import flexllm.cli.chat_helpers as ch

        flexllm_skills = tmp_path / "flexllm_skills"
        claude_skills = tmp_path / "claude_skills"
        flexllm_skills.mkdir()
        claude_skills.mkdir()

        # 只在 claude 目录有
        (claude_skills / "only-claude.md").write_text(
            "---\nname: only-claude\n---\nClaude only content"
        )

        monkeypatch.setattr(ch, "SKILLS_DIRS", [flexllm_skills, claude_skills])

        result = ch.load_skill("only-claude")
        assert result is not None
        assert result["content"] == "Claude only content"

    def test_allowed_tools_parsing(self, tmp_path, monkeypatch):
        """测试 allowed-tools 解析"""
        import flexllm.cli.chat_helpers as ch

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "restricted.md").write_text(
            "---\nname: restricted\nallowed-tools: Read, Grep, Bash(git *)\n---\nContent"
        )

        monkeypatch.setattr(ch, "SKILLS_DIRS", [skills_dir])

        result = ch.load_skill("restricted")
        assert result["allowed_tools"] == ["read", "grep", "bash"]

    def test_model_field(self, tmp_path, monkeypatch):
        """测试 model 字段"""
        import flexllm.cli.chat_helpers as ch

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "smart.md").write_text("---\nname: smart\nmodel: gpt-4o\n---\nContent")

        monkeypatch.setattr(ch, "SKILLS_DIRS", [skills_dir])

        result = ch.load_skill("smart")
        assert result["model"] == "gpt-4o"

    def test_list_skills_across_dirs(self, tmp_path, monkeypatch):
        """list_skills 跨目录收集"""
        import flexllm.cli.chat_helpers as ch

        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        (dir1 / "skill-a.md").write_text("---\nname: a\n---\nA")
        (dir2 / "skill-b.md").write_text("---\nname: b\n---\nB")

        monkeypatch.setattr(ch, "SKILLS_DIRS", [dir1, dir2])

        result = ch.list_skills()
        assert "skill-a" in result
        assert "skill-b" in result


class TestBuildAgentSystem:
    """测试 build_agent_system 返回 dict"""

    def test_returns_dict(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # 避免找到真实的项目指令
        from flexllm.cli.chat_helpers import build_agent_system

        result = build_agent_system("You are helpful.")
        assert isinstance(result, dict)
        assert "system" in result
        assert "You are helpful." in result["system"]
        assert result["allowed_tools"] is None
        assert result["model"] is None

    def test_with_skill_allowed_tools(self, tmp_path, monkeypatch):
        import flexllm.cli.chat_helpers as ch

        monkeypatch.chdir(tmp_path)
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "limited.md").write_text(
            "---\nname: limited\nallowed-tools: Read, Grep\nmodel: gpt-4\n---\nDo stuff"
        )
        monkeypatch.setattr(ch, "SKILLS_DIRS", [skills_dir])

        result = ch.build_agent_system(None, skill="limited")
        assert result["allowed_tools"] == ["read", "grep"]
        assert result["model"] == "gpt-4"
        assert "Do stuff" in result["system"]
