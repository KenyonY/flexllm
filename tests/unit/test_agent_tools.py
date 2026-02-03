"""Agent 工具单元测试"""

import json
import tempfile
from pathlib import Path

import pytest

from flexllm.agent.tools import (
    TOOL_REGISTRY,
    edit_file,
    get_tool_defs,
    glob_files,
    grep_content,
    make_tool_executor,
    read_file,
    write_file,
)
from flexllm.agent.tools.shell_tool import bash_exec


class TestToolRegistry:
    """工具注册器测试"""

    def test_tools_registered(self):
        """验证所有工具已注册"""
        expected = {"read", "write", "edit", "glob", "grep", "bash"}
        assert expected.issubset(set(TOOL_REGISTRY.keys()))

    def test_get_tool_defs(self):
        """测试获取工具定义"""
        defs = get_tool_defs(["read", "edit"])
        assert len(defs) == 2
        assert all(d["type"] == "function" for d in defs)
        assert defs[0]["function"]["name"] == "read"
        assert defs[1]["function"]["name"] == "edit"

    def test_get_tool_defs_unknown(self):
        """测试获取未知工具"""
        with pytest.raises(ValueError, match="未知工具"):
            get_tool_defs(["unknown_tool"])

    def test_make_tool_executor(self):
        """测试创建工具执行器"""
        executor = make_tool_executor(["bash"])
        result = executor("bash", json.dumps({"command": "echo hello"}))
        assert "hello" in result


class TestReadFile:
    """Read 工具测试"""

    def test_read_basic(self, tmp_path):
        """基本读取"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\nline3\n")

        result = read_file(str(test_file))
        assert "1|" in result
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result

    def test_read_with_offset_limit(self, tmp_path):
        """带偏移和限制的读取"""
        test_file = tmp_path / "test.txt"
        lines = [f"line{i}" for i in range(100)]
        test_file.write_text("\n".join(lines))

        # 从第 10 行开始，读 5 行
        result = read_file(str(test_file), offset=10, limit=5)
        assert "11|" in result  # 行号从 1 开始显示
        assert "line10" in result
        assert "line14" in result
        assert "line15" not in result or "use offset=15" in result

    def test_read_not_found(self):
        """读取不存在的文件"""
        result = read_file("/nonexistent/path/file.txt")
        assert "[error:" in result
        assert "not found" in result

    def test_read_directory(self, tmp_path):
        """读取目录报错"""
        result = read_file(str(tmp_path))
        assert "[error:" in result
        assert "directory" in result

    def test_read_long_lines_truncated(self, tmp_path):
        """长行被截断"""
        test_file = tmp_path / "test.txt"
        long_line = "x" * 3000
        test_file.write_text(long_line)

        result = read_file(str(test_file))
        assert "..." in result
        assert len(result) < 3000


class TestWriteFile:
    """Write 工具测试"""

    def test_write_basic(self, tmp_path):
        """基本写入"""
        test_file = tmp_path / "new.txt"
        result = write_file(str(test_file), "hello world")

        assert "[written:" in result
        assert test_file.exists()
        assert test_file.read_text() == "hello world"

    def test_write_creates_parent_dirs(self, tmp_path):
        """自动创建父目录"""
        test_file = tmp_path / "subdir" / "deep" / "file.txt"
        result = write_file(str(test_file), "content")

        assert "[written:" in result
        assert test_file.exists()

    def test_write_overwrite(self, tmp_path):
        """覆盖现有文件"""
        test_file = tmp_path / "existing.txt"
        test_file.write_text("old content")

        write_file(str(test_file), "new content")
        assert test_file.read_text() == "new content"


class TestEditFile:
    """Edit 工具测试"""

    def test_edit_unique_match(self, tmp_path):
        """唯一匹配替换"""
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    print('hi')\n")

        result = edit_file(str(test_file), "hello", "world")
        assert "[edited:" in result
        assert "replaced 1" in result
        assert "def world():" in test_file.read_text()

    def test_edit_multiple_matches_error(self, tmp_path):
        """多次匹配时报错"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello hello hello")

        result = edit_file(str(test_file), "hello", "world")
        assert "[error:" in result
        assert "3 times" in result

    def test_edit_replace_all(self, tmp_path):
        """replace_all 模式"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello hello hello")

        result = edit_file(str(test_file), "hello", "world", replace_all=True)
        assert "[edited:" in result
        assert "replaced 3" in result
        assert test_file.read_text() == "world world world"

    def test_edit_not_found(self, tmp_path):
        """匹配不到"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        result = edit_file(str(test_file), "xyz", "abc")
        assert "[error:" in result
        assert "not found" in result

    def test_edit_file_not_exists(self):
        """文件不存在"""
        result = edit_file("/nonexistent/file.txt", "old", "new")
        assert "[error:" in result
        assert "not found" in result


class TestGlobFiles:
    """Glob 工具测试"""

    def test_glob_pattern(self, tmp_path):
        """模式匹配"""
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        (tmp_path / "c.txt").write_text("c")

        result = glob_files("*.py", str(tmp_path))
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    def test_glob_recursive(self, tmp_path):
        """递归模式"""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "a.py").write_text("a")
        (subdir / "b.py").write_text("b")

        result = glob_files("**/*.py", str(tmp_path))
        assert "a.py" in result
        assert "b.py" in result

    def test_glob_no_matches(self, tmp_path):
        """无匹配"""
        result = glob_files("*.xyz", str(tmp_path))
        assert "no matches" in result

    def test_glob_invalid_path(self):
        """无效路径"""
        result = glob_files("*.py", "/nonexistent/path")
        assert "[error:" in result


class TestGrepContent:
    """Grep 工具测试"""

    def test_grep_files_with_matches(self, tmp_path):
        """files_with_matches 模式"""
        (tmp_path / "a.py").write_text("def hello():\n    pass\n")
        (tmp_path / "b.py").write_text("class World:\n    pass\n")

        result = grep_content("hello", str(tmp_path), output_mode="files_with_matches")
        assert "a.py" in result
        assert "b.py" not in result

    def test_grep_content_mode(self, tmp_path):
        """content 模式"""
        (tmp_path / "test.py").write_text("line1\nfind_me\nline3\n")

        result = grep_content("find_me", str(tmp_path), output_mode="content")
        assert "find_me" in result
        assert "2" in result  # 行号

    def test_grep_with_glob_filter(self, tmp_path):
        """带文件过滤"""
        (tmp_path / "a.py").write_text("hello")
        (tmp_path / "b.txt").write_text("hello")

        result = grep_content("hello", str(tmp_path), glob="*.py", output_mode="files_with_matches")
        assert "a.py" in result
        assert "b.txt" not in result

    def test_grep_case_insensitive(self, tmp_path):
        """大小写不敏感"""
        (tmp_path / "test.txt").write_text("Hello World")

        result = grep_content("hello", str(tmp_path), case_insensitive=True)
        assert "test.txt" in result

    def test_grep_no_matches(self, tmp_path):
        """无匹配"""
        (tmp_path / "test.txt").write_text("hello")

        result = grep_content("xyz", str(tmp_path))
        assert "no matches" in result


class TestBashExec:
    """Bash 工具测试"""

    def test_bash_echo(self):
        """基本命令"""
        result = bash_exec("echo hello")
        assert "hello" in result

    def test_bash_exit_code(self):
        """非零退出码"""
        result = bash_exec("exit 1")
        assert "[exit code: 1]" in result

    def test_bash_empty_command(self):
        """空命令"""
        result = bash_exec("")
        assert "[error:" in result

    def test_bash_stderr(self):
        """stderr 输出"""
        result = bash_exec("echo error >&2")
        assert "error" in result

    def test_bash_timeout(self):
        """超时"""
        result = bash_exec("sleep 10", timeout=1)
        assert "timed out" in result


class TestParallelExecution:
    """并行执行测试"""

    @pytest.mark.asyncio
    async def test_parallel_tool_calls(self):
        """测试并行工具调用"""
        from flexllm.agent import AgentClient
        from flexllm.agent.tools import make_tool_executor
        from flexllm.clients.base import ChatCompletionResult, ToolCall

        call_count = 0

        # Mock LLM client
        class MockClient:
            async def chat_completions_or_raise(self, messages, **kwargs):
                nonlocal call_count
                call_count += 1
                # 第一次调用：返回多个工具调用
                if call_count == 1:
                    return ChatCompletionResult(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id="call_1",
                                type="function",
                                function={"name": "bash", "arguments": '{"command": "echo 1"}'},
                            ),
                            ToolCall(
                                id="call_2",
                                type="function",
                                function={"name": "bash", "arguments": '{"command": "echo 2"}'},
                            ),
                        ],
                    )
                else:  # 第二次调用（有工具结果后）
                    return ChatCompletionResult(content="done")

        executor = make_tool_executor(["bash"])
        agent = AgentClient(
            client=MockClient(),
            tools=[{"type": "function", "function": {"name": "bash", "parameters": {}}}],
            tool_executor=executor,
        )

        result = await agent.run("test")
        assert result.content == "done"
        assert len(result.tool_calls) == 2
        assert "1" in result.tool_calls[0].result
        assert "2" in result.tool_calls[1].result
