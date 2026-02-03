"""Python 代码验证器

包含：
- PythonSyntaxValidator: 语法检查（py_compile）
- PythonLintValidator: Lint 检查（ruff）
- PythonTypeValidator: 类型检查（pyright）
"""

from __future__ import annotations

import json
import shutil
import sys

from .base import ValidationError, ValidationResult, Validator, run_command


class PythonSyntaxValidator(Validator):
    """Python 语法验证器，使用 py_compile"""

    name = "python-syntax"

    async def validate(self, changed_files: list[str]) -> ValidationResult:
        py_files = self.filter_files(changed_files, ["*.py"])
        if not py_files:
            return ValidationResult(success=True, validator_name=self.name)

        errors = []
        for file_path in py_files:
            # 使用 python -m py_compile 检查语法
            returncode, stdout, stderr = await run_command(
                [sys.executable, "-m", "py_compile", file_path]
            )
            if returncode != 0:
                # 解析错误信息
                error_msg = stderr.strip() or stdout.strip() or "Syntax error"
                # 尝试提取行号
                line = None
                if "line" in error_msg.lower():
                    import re

                    match = re.search(r"line (\d+)", error_msg, re.IGNORECASE)
                    if match:
                        line = int(match.group(1))

                errors.append(
                    ValidationError(
                        message=error_msg,
                        file_path=file_path,
                        line=line,
                        severity="error",
                    )
                )

        return ValidationResult(
            success=len(errors) == 0,
            errors=errors,
            validator_name=self.name,
        )


class PythonLintValidator(Validator):
    """Python Lint 验证器，使用 ruff"""

    name = "python-lint"

    def __init__(self, extra_args: list[str] | None = None):
        """
        Args:
            extra_args: 额外的 ruff 参数，如 ["--ignore", "E501"]
        """
        self.extra_args = extra_args or []

    async def validate(self, changed_files: list[str]) -> ValidationResult:
        py_files = self.filter_files(changed_files, ["*.py"])
        if not py_files:
            return ValidationResult(success=True, validator_name=self.name)

        # 检查 ruff 是否可用
        if not shutil.which("ruff"):
            return ValidationResult(
                success=True,
                errors=[],
                validator_name=self.name,
            )

        cmd = ["ruff", "check", "--output-format=json", *self.extra_args, *py_files]
        returncode, stdout, stderr = await run_command(cmd)

        errors = []
        if stdout.strip():
            try:
                issues = json.loads(stdout)
                for issue in issues:
                    errors.append(
                        ValidationError(
                            message=f"[{issue.get('code', '?')}] {issue.get('message', 'Unknown')}",
                            file_path=issue.get("filename"),
                            line=issue.get("location", {}).get("row"),
                            column=issue.get("location", {}).get("column"),
                            severity="warning"
                            if (issue.get("code") or "").startswith("W")
                            else "error",
                        )
                    )
            except json.JSONDecodeError:
                # 非 JSON 输出，作为单个错误
                if returncode != 0:
                    errors.append(ValidationError(message=stdout.strip() or stderr.strip()))

        return ValidationResult(
            success=len(errors) == 0,
            errors=errors,
            validator_name=self.name,
        )


class PythonTypeValidator(Validator):
    """Python 类型验证器，使用 pyright"""

    name = "python-type"

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: 是否使用严格模式
        """
        self.strict = strict

    async def validate(self, changed_files: list[str]) -> ValidationResult:
        py_files = self.filter_files(changed_files, ["*.py", "*.pyi"])
        if not py_files:
            return ValidationResult(success=True, validator_name=self.name)

        # 检查 pyright 是否可用
        if not shutil.which("pyright"):
            return ValidationResult(
                success=True,
                errors=[],
                validator_name=self.name,
            )

        cmd = ["pyright", "--outputjson"]
        if self.strict:
            cmd.append("--strict")
        cmd.extend(py_files)

        returncode, stdout, stderr = await run_command(cmd)

        errors = []
        if stdout.strip():
            try:
                result = json.loads(stdout)
                for diag in result.get("generalDiagnostics", []):
                    severity = diag.get("severity", "error")
                    if severity == "information":
                        continue  # 跳过信息级别
                    errors.append(
                        ValidationError(
                            message=diag.get("message", "Unknown type error"),
                            file_path=diag.get("file"),
                            line=diag.get("range", {}).get("start", {}).get("line"),
                            column=diag.get("range", {}).get("start", {}).get("character"),
                            severity=severity,
                        )
                    )
            except json.JSONDecodeError:
                # 非 JSON 输出
                if returncode != 0 and (stdout.strip() or stderr.strip()):
                    errors.append(ValidationError(message=stdout.strip() or stderr.strip()))

        return ValidationResult(
            success=len(errors) == 0,
            errors=errors,
            validator_name=self.name,
        )


class PytestValidator(Validator):
    """Python 测试验证器，运行 pytest"""

    name = "pytest"

    def __init__(
        self,
        test_paths: list[str] | None = None,
        extra_args: list[str] | None = None,
    ):
        """
        Args:
            test_paths: 测试文件/目录路径，默认自动检测
            extra_args: 额外的 pytest 参数，如 ["-x", "--tb=short"]
        """
        self.test_paths = test_paths or []
        self.extra_args = extra_args or ["-x", "--tb=short", "-q"]

    async def validate(self, changed_files: list[str]) -> ValidationResult:
        # 检查 pytest 是否可用
        if not shutil.which("pytest"):
            return ValidationResult(
                success=True,
                errors=[],
                validator_name=self.name,
            )

        cmd = ["pytest", *self.extra_args]
        if self.test_paths:
            cmd.extend(self.test_paths)

        returncode, stdout, stderr = await run_command(cmd)

        if returncode == 0:
            return ValidationResult(success=True, errors=[], validator_name=self.name)

        # 解析失败输出
        output = stdout + "\n" + stderr
        errors = [
            ValidationError(
                message=output.strip()[:2000],  # 限制长度
                severity="error",
            )
        ]

        return ValidationResult(
            success=False,
            errors=errors,
            validator_name=self.name,
        )
