from pathlib import Path

import pytest

from flexllm.cli.commands import get_skill_target_dir


def test_skill_target_directories():
    claude_dir, claude_name = get_skill_target_dir("claude")
    codex_dir, codex_name = get_skill_target_dir("codex")

    assert claude_dir == Path.home() / ".claude/skills/flexllm"
    assert claude_name == "Claude Code"
    assert codex_dir == Path.home() / ".agents/skills/flexllm"
    assert codex_name == "Codex"


def test_unknown_skill_target_is_rejected():
    with pytest.raises(ValueError, match="未知 skill 安装目标"):
        get_skill_target_dir("other")
