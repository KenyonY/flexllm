"""测试 agent 新功能：项目指令、skills、配置文件 MCP"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestProjectInstructions:
    """测试 .flexllm.md 项目指令加载"""

    def test_load_from_cwd(self, tmp_path):
        """当前目录有 .flexllm.md 时应加载"""
        md_file = tmp_path / ".flexllm.md"
        md_file.write_text("你是代码助手，请用中文回答。")

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            from flexllm.cli.chat_helpers import load_project_instructions

            result = load_project_instructions()
            assert result == "你是代码助手，请用中文回答。"

    def test_load_from_parent(self, tmp_path):
        """子目录中应能找到父目录的 .flexllm.md"""
        md_file = tmp_path / ".flexllm.md"
        md_file.write_text("项目根指令")
        sub_dir = tmp_path / "src" / "module"
        sub_dir.mkdir(parents=True)

        with patch("pathlib.Path.cwd", return_value=sub_dir):
            from flexllm.cli.chat_helpers import load_project_instructions

            result = load_project_instructions()
            assert result == "项目根指令"

    def test_no_file_returns_none(self, tmp_path):
        """没有 .flexllm.md 时返回 None"""
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            from flexllm.cli.chat_helpers import load_project_instructions

            result = load_project_instructions()
            assert result is None

    def test_build_agent_system_with_project_instructions(self, tmp_path):
        """build_agent_system 应自动包含项目指令"""
        md_file = tmp_path / ".flexllm.md"
        md_file.write_text("这是项目指令")

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            from flexllm.cli.chat_helpers import build_agent_system

            result = build_agent_system("你是助手")
            assert "你是助手" in result
            assert "这是项目指令" in result
            assert "# Project Instructions" in result

    def test_build_agent_system_without_project_instructions(self, tmp_path):
        """没有 .flexllm.md 时不影响 system prompt"""
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            from flexllm.cli.chat_helpers import build_agent_system

            result = build_agent_system("你是助手")
            assert result == "你是助手"
            assert "Project Instructions" not in result


class TestSkills:
    """测试 skills 系统"""

    def test_load_skill_flat_mode(self, tmp_path):
        """扁平模式：skills/{name}.md"""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "code-review.md").write_text("你是代码审查专家，关注安全性和性能。")

        with patch("flexllm.cli.chat_helpers.SKILLS_DIR", skills_dir):
            from flexllm.cli.chat_helpers import load_skill

            result = load_skill("code-review")
            assert result is not None
            assert result["content"] == "你是代码审查专家，关注安全性和性能。"
            assert result["name"] == "code-review"

    def test_load_skill_dir_mode(self, tmp_path):
        """目录模式：skills/{name}/SKILL.md（Claude Code 风格）"""
        skills_dir = tmp_path / "skills"
        skill_sub = skills_dir / "code-review"
        skill_sub.mkdir(parents=True)
        (skill_sub / "SKILL.md").write_text(
            "---\nname: code-review\ndescription: 代码审核专家\n---\n\n你是代码审查专家。"
        )

        with patch("flexllm.cli.chat_helpers.SKILLS_DIR", skills_dir):
            from flexllm.cli.chat_helpers import load_skill

            result = load_skill("code-review")
            assert result is not None
            assert result["name"] == "code-review"
            assert result["description"] == "代码审核专家"
            assert result["content"] == "你是代码审查专家。"

    def test_load_skill_dir_mode_priority(self, tmp_path):
        """目录模式优先于扁平模式"""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "review.md").write_text("扁平版本")
        sub = skills_dir / "review"
        sub.mkdir()
        (sub / "SKILL.md").write_text("目录版本")

        with patch("flexllm.cli.chat_helpers.SKILLS_DIR", skills_dir):
            from flexllm.cli.chat_helpers import load_skill

            result = load_skill("review")
            assert result["content"] == "目录版本"

    def test_load_skill_not_found(self, tmp_path):
        """skill 不存在时返回 None"""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        with patch("flexllm.cli.chat_helpers.SKILLS_DIR", skills_dir):
            from flexllm.cli.chat_helpers import load_skill

            result = load_skill("nonexistent")
            assert result is None

    def test_list_skills_mixed(self, tmp_path):
        """列出目录模式和扁平模式的 skills"""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        # 扁平模式
        (skills_dir / "translate.md").write_text("翻译")
        # 目录模式
        sub = skills_dir / "code-review"
        sub.mkdir()
        (sub / "SKILL.md").write_text("审查")
        # 非 skill 文件
        (skills_dir / "not-a-skill.txt").write_text("忽略")

        with patch("flexllm.cli.chat_helpers.SKILLS_DIR", skills_dir):
            from flexllm.cli.chat_helpers import list_skills

            result = list_skills()
            assert result == ["code-review", "translate"]

    def test_list_skills_empty(self, tmp_path):
        """skills 目录不存在时返回空列表"""
        with patch("flexllm.cli.chat_helpers.SKILLS_DIR", tmp_path / "nonexistent"):
            from flexllm.cli.chat_helpers import list_skills

            result = list_skills()
            assert result == []

    def test_parse_frontmatter(self):
        """解析 SKILL.md frontmatter"""
        from flexllm.cli.chat_helpers import _parse_skill_frontmatter

        content = (
            "---\n"
            "name: test-skill\n"
            "description: 这是一个测试 skill\n"
            "allowed-tools: Bash(git:*)\n"
            "---\n\n"
            "正文内容"
        )
        meta, body = _parse_skill_frontmatter(content)
        assert meta["name"] == "test-skill"
        assert meta["description"] == "这是一个测试 skill"
        assert meta["allowed-tools"] == "Bash(git:*)"
        assert body == "正文内容"

    def test_parse_frontmatter_none(self):
        """无 frontmatter 时返回空 metadata"""
        from flexllm.cli.chat_helpers import _parse_skill_frontmatter

        meta, body = _parse_skill_frontmatter("纯正文内容")
        assert meta == {}
        assert body == "纯正文内容"

    def test_build_agent_system_with_skill(self, tmp_path):
        """build_agent_system 应包含 skill 内容"""
        skills_dir = tmp_path / "skills"
        sub = skills_dir / "review"
        sub.mkdir(parents=True)
        (sub / "SKILL.md").write_text("---\nname: review\n---\n\n审查代码时关注以下要点...")

        with (
            patch("flexllm.cli.chat_helpers.SKILLS_DIR", skills_dir),
            patch("pathlib.Path.cwd", return_value=tmp_path),
        ):
            from flexllm.cli.chat_helpers import build_agent_system

            result = build_agent_system("你是助手", skill="review")
            assert "你是助手" in result
            assert "审查代码时关注以下要点" in result
            assert "# Skill: review" in result

    def test_build_agent_system_unknown_skill(self, tmp_path):
        """未知 skill 应抛出 ValueError"""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        with (
            patch("flexllm.cli.chat_helpers.SKILLS_DIR", skills_dir),
            patch("pathlib.Path.cwd", return_value=tmp_path),
        ):
            from flexllm.cli.chat_helpers import build_agent_system

            with pytest.raises(ValueError, match="未知的 skill"):
                build_agent_system("你是助手", skill="nonexistent")

    def test_build_agent_system_all_combined(self, tmp_path):
        """system + 项目指令 + skill 三者叠加"""
        md_file = tmp_path / ".flexllm.md"
        md_file.write_text("项目：flexllm")

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "debug.md").write_text("调试指导")

        with (
            patch("flexllm.cli.chat_helpers.SKILLS_DIR", skills_dir),
            patch("pathlib.Path.cwd", return_value=tmp_path),
        ):
            from flexllm.cli.chat_helpers import build_agent_system

            result = build_agent_system("基础系统提示", skill="debug")
            assert "基础系统提示" in result
            assert "项目：flexllm" in result
            assert "调试指导" in result


class TestMCPConfigMerge:
    """测试 MCP 配置合并"""

    def test_merge_cli_only(self):
        from flexllm.cli.chat_helpers import _merge_mcp_servers

        result = _merge_mcp_servers(["npx @mcp/github"], None)
        assert result == ["npx @mcp/github"]

    def test_merge_config_only(self):
        from flexllm.cli.chat_helpers import _merge_mcp_servers

        config_mcp = [{"command": "npx @mcp/github", "name": "github"}]
        result = _merge_mcp_servers(None, config_mcp)
        assert result == config_mcp

    def test_merge_both(self):
        from flexllm.cli.chat_helpers import _merge_mcp_servers

        config_mcp = [{"command": "npx @mcp/github", "name": "github"}]
        cli_mcp = ["http://localhost:8080/sse"]
        result = _merge_mcp_servers(cli_mcp, config_mcp)
        # 配置文件在前，CLI 在后
        assert len(result) == 2
        assert result[0] == config_mcp[0]
        assert result[1] == "http://localhost:8080/sse"

    def test_merge_empty(self):
        from flexllm.cli.chat_helpers import _merge_mcp_servers

        result = _merge_mcp_servers(None, None)
        assert result == []

    def test_agent_config_defaults(self):
        """agent 配置默认值"""
        from flexllm.cli.config import FlexLLMConfig

        with (
            patch.object(FlexLLMConfig, "_load_config", return_value={}),
            patch.object(FlexLLMConfig, "_load_project_settings", return_value={}),
        ):
            config = FlexLLMConfig()
            agent_config = config.get_agent_config()
            assert agent_config["mcp_servers"] == []

    def test_agent_config_named_dict_format(self):
        """Claude Code 风格命名字典格式"""
        from flexllm.cli.config import FlexLLMConfig

        mock_config = {
            "agent": {
                "mcp_servers": {
                    "github": {"command": "npx @mcp/server-github"},
                    "local": {"url": "http://localhost:8080/sse"},
                },
            }
        }
        with (
            patch.object(FlexLLMConfig, "_load_config", return_value=mock_config),
            patch.object(FlexLLMConfig, "_load_project_settings", return_value={}),
        ):
            config = FlexLLMConfig()
            agent_config = config.get_agent_config()
            servers = agent_config["mcp_servers"]
            assert len(servers) == 2
            names = {s["name"] for s in servers}
            assert names == {"github", "local"}

    def test_agent_config_project_settings(self, tmp_path):
        """项目级 .flexllm/settings.yaml 配置"""
        from flexllm.cli.config import FlexLLMConfig

        settings_dir = tmp_path / ".flexllm"
        settings_dir.mkdir()
        settings_file = settings_dir / "settings.yaml"
        settings_file.write_text(
            "mcp_servers:\n  project-tool:\n    command: npx @mcp/project-tool\n"
        )

        with (
            patch.object(FlexLLMConfig, "_load_config", return_value={}),
            patch("pathlib.Path.cwd", return_value=tmp_path),
        ):
            config = FlexLLMConfig()
            agent_config = config.get_agent_config()
            servers = agent_config["mcp_servers"]
            assert len(servers) == 1
            assert servers[0]["name"] == "project-tool"
            assert servers[0]["command"] == "npx @mcp/project-tool"

    def test_agent_config_merge_global_and_project(self, tmp_path):
        """项目级覆盖全局同名 server"""
        from flexllm.cli.config import FlexLLMConfig

        mock_config = {
            "agent": {
                "mcp_servers": {
                    "github": {"command": "npx @mcp/server-github"},
                    "global-only": {"url": "http://global:8080/sse"},
                },
            }
        }

        settings_dir = tmp_path / ".flexllm"
        settings_dir.mkdir()
        settings_file = settings_dir / "settings.yaml"
        settings_file.write_text(
            "mcp_servers:\n"
            "  github:\n"
            "    command: npx @mcp/server-github --token project\n"
            "  project-only:\n"
            "    url: http://project:9090/sse\n"
        )

        with (
            patch.object(FlexLLMConfig, "_load_config", return_value=mock_config),
            patch("pathlib.Path.cwd", return_value=tmp_path),
        ):
            config = FlexLLMConfig()
            agent_config = config.get_agent_config()
            servers = agent_config["mcp_servers"]
            assert len(servers) == 3
            by_name = {s["name"]: s for s in servers}
            # github 被项目级覆盖
            assert by_name["github"]["command"] == "npx @mcp/server-github --token project"
            # 全局独有
            assert "global-only" in by_name
            # 项目独有
            assert "project-only" in by_name

    def test_agent_config_legacy_list_format(self):
        """兼容旧的 list 格式"""
        from flexllm.cli.config import FlexLLMConfig

        mock_config = {
            "agent": {
                "mcp_servers": [
                    {"command": "npx @mcp/github", "name": "github"},
                    {"url": "http://localhost:8080/sse"},
                ],
            }
        }
        with (
            patch.object(FlexLLMConfig, "_load_config", return_value=mock_config),
            patch.object(FlexLLMConfig, "_load_project_settings", return_value={}),
        ):
            config = FlexLLMConfig()
            agent_config = config.get_agent_config()
            assert len(agent_config["mcp_servers"]) == 2

    def test_agent_config_command_args_format(self):
        """Claude Code 风格 command + args 分开格式"""
        from flexllm.cli.config import FlexLLMConfig

        mock_config = {
            "agent": {
                "mcp_servers": {
                    "context7": {
                        "command": "npx",
                        "args": ["-y", "@upstash/context7-mcp"],
                    },
                    "github": {
                        "command": "npx",
                        "args": ["-y", "@mcp/server-github"],
                        "env": {"GITHUB_TOKEN": "xxx"},
                    },
                },
            }
        }
        with (
            patch.object(FlexLLMConfig, "_load_config", return_value=mock_config),
            patch.object(FlexLLMConfig, "_load_project_settings", return_value={}),
        ):
            config = FlexLLMConfig()
            agent_config = config.get_agent_config()
            servers = agent_config["mcp_servers"]
            assert len(servers) == 2
            by_name = {s["name"]: s for s in servers}
            assert by_name["context7"]["command"] == "npx"
            assert by_name["context7"]["args"] == ["-y", "@upstash/context7-mcp"]
            assert by_name["github"]["env"] == {"GITHUB_TOKEN": "xxx"}
