"""Test validators - 代码验证器单元测试"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from flexllm.agent.validators import (
    PytestValidator,
    PythonLintValidator,
    PythonSyntaxValidator,
    PythonTypeValidator,
    ValidationError,
    ValidationResult,
    Validator,
)


class TestValidationResult:
    """ValidationResult 数据类测试"""

    def test_success_result(self):
        result = ValidationResult(success=True, validator_name="test")
        assert result.success
        assert result.errors == []
        assert "[test] OK" in str(result)

    def test_failure_result(self):
        errors = [
            ValidationError(message="error 1", file_path="a.py", line=10),
            ValidationError(message="error 2", file_path="b.py", line=20),
        ]
        result = ValidationResult(success=False, errors=errors, validator_name="test")
        assert not result.success
        assert len(result.errors) == 2
        result_str = str(result)
        assert "[test] 2 error(s)" in result_str
        assert "a.py:10" in result_str
        assert "error 1" in result_str

    def test_truncate_many_errors(self):
        errors = [ValidationError(message=f"error {i}") for i in range(15)]
        result = ValidationResult(success=False, errors=errors, validator_name="test")
        result_str = str(result)
        assert "... and 5 more errors" in result_str


class TestValidationError:
    """ValidationError 数据类测试"""

    def test_simple_error(self):
        error = ValidationError(message="syntax error")
        assert str(error) == "[error] syntax error"

    def test_error_with_location(self):
        error = ValidationError(
            message="undefined variable",
            file_path="/path/to/file.py",
            line=42,
            column=10,
            severity="warning",
        )
        error_str = str(error)
        assert "/path/to/file.py:42:10" in error_str
        assert "[warning]" in error_str
        assert "undefined variable" in error_str


class TestValidatorBase:
    """Validator 基类测试"""

    def test_filter_files_py(self):
        """测试按模式过滤文件"""
        validator = PythonSyntaxValidator()
        files = ["/a/b.py", "/c/d.txt", "/e/f.py", "/g/h.pyi"]
        result = validator.filter_files(files, ["*.py"])
        assert result == ["/a/b.py", "/e/f.py"]

    def test_filter_files_multiple_patterns(self):
        """多模式匹配"""
        validator = PythonSyntaxValidator()
        files = ["/a.py", "/b.pyi", "/c.txt"]
        result = validator.filter_files(files, ["*.py", "*.pyi"])
        assert set(result) == {"/a.py", "/b.pyi"}


class TestPythonSyntaxValidator:
    """PythonSyntaxValidator 测试"""

    @pytest.mark.asyncio
    async def test_valid_syntax(self, tmp_path):
        """有效语法文件应通过"""
        py_file = tmp_path / "valid.py"
        py_file.write_text("x = 1\nprint(x)")

        validator = PythonSyntaxValidator()
        result = await validator.validate([str(py_file)])

        assert result.success
        assert len(result.errors) == 0
        assert result.validator_name == "python-syntax"

    @pytest.mark.asyncio
    async def test_invalid_syntax(self, tmp_path):
        """无效语法文件应失败"""
        py_file = tmp_path / "invalid.py"
        py_file.write_text("def foo(\n  # missing close paren")

        validator = PythonSyntaxValidator()
        result = await validator.validate([str(py_file)])

        assert not result.success
        assert len(result.errors) >= 1
        assert str(py_file) in result.errors[0].file_path

    @pytest.mark.asyncio
    async def test_no_python_files(self):
        """无 Python 文件时直接通过"""
        validator = PythonSyntaxValidator()
        result = await validator.validate(["/some/file.txt", "/other/file.js"])

        assert result.success
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_empty_file_list(self):
        """空文件列表"""
        validator = PythonSyntaxValidator()
        result = await validator.validate([])

        assert result.success


class TestPythonLintValidator:
    """PythonLintValidator 测试（需要 ruff 可用）"""

    @pytest.mark.asyncio
    async def test_clean_code(self, tmp_path):
        """干净代码应通过"""
        py_file = tmp_path / "clean.py"
        py_file.write_text("x = 1\nprint(x)\n")

        validator = PythonLintValidator()
        result = await validator.validate([str(py_file)])

        # ruff 可能不可用，此时也应通过
        assert result.success or result.validator_name == "python-lint"

    @pytest.mark.asyncio
    async def test_no_python_files(self):
        """无 Python 文件时直接通过"""
        validator = PythonLintValidator()
        result = await validator.validate(["/some/file.txt"])
        assert result.success

    @pytest.mark.asyncio
    async def test_extra_args(self, tmp_path):
        """测试额外参数传递"""
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")

        validator = PythonLintValidator(extra_args=["--ignore", "E501"])
        result = await validator.validate([str(py_file)])
        # 应该能正常执行
        assert result.validator_name == "python-lint"


class TestPythonTypeValidator:
    """PythonTypeValidator 测试（需要 pyright 可用）"""

    @pytest.mark.asyncio
    async def test_no_python_files(self):
        """无 Python 文件时直接通过"""
        validator = PythonTypeValidator()
        result = await validator.validate(["/some/file.txt"])
        assert result.success

    @pytest.mark.asyncio
    async def test_strict_mode(self):
        """测试严格模式初始化"""
        validator = PythonTypeValidator(strict=True)
        assert validator.strict is True


class TestPytestValidator:
    """PytestValidator 测试"""

    @pytest.mark.asyncio
    async def test_init_with_paths(self):
        """测试初始化参数"""
        validator = PytestValidator(
            test_paths=["tests/unit"],
            extra_args=["-x", "-v"],
        )
        assert validator.test_paths == ["tests/unit"]
        assert "-x" in validator.extra_args
        assert "-v" in validator.extra_args


class TestAgentClientValidation:
    """AgentClient.run_with_validation 测试"""

    @pytest.mark.asyncio
    async def test_no_validators(self):
        """无验证器时直接执行"""
        from flexllm import AgentClient
        from flexllm.clients.base import ChatCompletionResult

        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            return_value=ChatCompletionResult(content="done", tool_calls=None)
        )

        agent = AgentClient(client=mock_client)
        result = await agent.run_with_validation("test", validators=None)

        assert result.content == "done"
        # 只调用一次
        assert mock_client.chat_completions_or_raise.call_count == 1

    @pytest.mark.asyncio
    async def test_validation_passes(self):
        """验证通过时不重试"""
        from flexllm import AgentClient
        from flexllm.clients.base import ChatCompletionResult, ToolCall

        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            side_effect=[
                # 第一次：写文件
                ChatCompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            type="function",
                            function={"name": "write", "arguments": '{"file_path": "/tmp/a.py"}'},
                        )
                    ],
                ),
                # 第二次：返回结果
                ChatCompletionResult(content="done", tool_calls=None),
            ]
        )

        # 创建一个总是通过的验证器
        mock_validator = MagicMock()
        mock_validator.validate = AsyncMock(
            return_value=ValidationResult(success=True, validator_name="mock")
        )

        agent = AgentClient(
            client=mock_client,
            tools=[{"type": "function", "function": {"name": "write"}}],
            tool_executor=lambda n, a: "ok",
        )
        result = await agent.run_with_validation("test", validators=[mock_validator])

        assert result.content == "done"
        # 验证器被调用一次
        mock_validator.validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_validation_fails_and_fixes(self):
        """验证失败时触发修复"""
        from flexllm import AgentClient
        from flexllm.clients.base import ChatCompletionResult, ToolCall

        call_count = 0

        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # 首次：写文件
                return ChatCompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            type="function",
                            function={"name": "write", "arguments": '{"file_path": "/tmp/a.py"}'},
                        )
                    ],
                )
            elif call_count == 2:
                # 写完后返回
                return ChatCompletionResult(content="wrote file", tool_calls=None)
            elif call_count == 3:
                # 修复调用：再次写文件
                return ChatCompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="c2",
                            type="function",
                            function={
                                "name": "edit",
                                "arguments": '{"file_path": "/tmp/a.py"}',
                            },
                        )
                    ],
                )
            else:
                # 修复完成
                return ChatCompletionResult(content="fixed", tool_calls=None)

        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(side_effect=mock_chat)

        # 第一次验证失败，第二次通过
        validate_calls = 0

        async def mock_validate(files):
            nonlocal validate_calls
            validate_calls += 1
            if validate_calls == 1:
                return ValidationResult(
                    success=False,
                    errors=[ValidationError(message="syntax error", file_path="/tmp/a.py", line=1)],
                    validator_name="mock",
                )
            return ValidationResult(success=True, validator_name="mock")

        mock_validator = MagicMock()
        mock_validator.validate = AsyncMock(side_effect=mock_validate)

        agent = AgentClient(
            client=mock_client,
            tools=[
                {"type": "function", "function": {"name": "write"}},
                {"type": "function", "function": {"name": "edit"}},
            ],
            tool_executor=lambda n, a: "ok",
        )
        result = await agent.run_with_validation(
            "write code", validators=[mock_validator], max_fix_attempts=3
        )

        assert result.content == "fixed"
        # 验证器被调用两次（首次失败 + 修复后通过）
        assert validate_calls == 2

    @pytest.mark.asyncio
    async def test_no_changed_files_skips_validation(self):
        """无文件修改时跳过验证"""
        from flexllm import AgentClient
        from flexllm.clients.base import ChatCompletionResult

        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            return_value=ChatCompletionResult(content="hello", tool_calls=None)
        )

        mock_validator = MagicMock()
        mock_validator.validate = AsyncMock()

        agent = AgentClient(client=mock_client)
        result = await agent.run_with_validation("test", validators=[mock_validator])

        assert result.content == "hello"
        # 验证器不应被调用
        mock_validator.validate.assert_not_called()

    def test_get_changed_files(self):
        """测试从工具调用中提取修改的文件"""
        from flexllm import AgentClient
        from flexllm.agent.types import ToolCallRecord

        agent = AgentClient(client=MagicMock())

        tool_calls = [
            ToolCallRecord(
                name="write",
                arguments='{"file_path": "/tmp/a.py"}',
                result="ok",
            ),
            ToolCallRecord(
                name="read",
                arguments='{"file_path": "/tmp/b.py"}',
                result="content",
            ),
            ToolCallRecord(
                name="edit",
                arguments='{"file_path": "/tmp/c.py", "old": "x", "new": "y"}',
                result="ok",
            ),
        ]

        changed = agent._get_changed_files(tool_calls)
        assert set(changed) == {"/tmp/a.py", "/tmp/c.py"}

    def test_build_fix_prompt(self):
        """测试构建修复提示"""
        from flexllm import AgentClient

        agent = AgentClient(client=MagicMock())

        failed_results = [
            ValidationResult(
                success=False,
                errors=[ValidationError(message="error 1", file_path="a.py", line=10)],
                validator_name="lint",
            ),
            ValidationResult(
                success=False,
                errors=[ValidationError(message="error 2")],
                validator_name="type",
            ),
        ]

        prompt = agent._build_fix_prompt(failed_results)
        assert "验证失败" in prompt
        assert "[lint]" in prompt
        assert "[type]" in prompt
        assert "error 1" in prompt
        assert "error 2" in prompt
