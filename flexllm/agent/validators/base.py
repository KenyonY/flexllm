"""验证器抽象基类和数据类型

验证器用于在 Agent 修改代码后自动检查代码质量，
支持语法检查、类型检查、Lint 等。
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationError:
    """单个验证错误"""

    message: str
    file_path: str | None = None
    line: int | None = None
    column: int | None = None
    severity: str = "error"  # error, warning, info

    def __str__(self) -> str:
        parts = []
        if self.file_path:
            loc = self.file_path
            if self.line:
                loc += f":{self.line}"
                if self.column:
                    loc += f":{self.column}"
            parts.append(loc)
        parts.append(f"[{self.severity}] {self.message}")
        return " ".join(parts)


@dataclass
class ValidationResult:
    """验证结果"""

    success: bool
    errors: list[ValidationError] = field(default_factory=list)
    validator_name: str = ""

    def __str__(self) -> str:
        if self.success:
            return f"[{self.validator_name}] OK"
        error_lines = [str(e) for e in self.errors[:10]]
        if len(self.errors) > 10:
            error_lines.append(f"... and {len(self.errors) - 10} more errors")
        return f"[{self.validator_name}] {len(self.errors)} error(s):\n" + "\n".join(error_lines)


class Validator(ABC):
    """验证器抽象基类"""

    name: str = "base"

    @abstractmethod
    async def validate(self, changed_files: list[str]) -> ValidationResult:
        """验证修改后的文件

        Args:
            changed_files: 被修改的文件路径列表

        Returns:
            ValidationResult 验证结果
        """

    def filter_files(self, files: list[str], patterns: list[str]) -> list[str]:
        """按模式过滤文件

        Args:
            files: 文件路径列表
            patterns: glob 模式列表，如 ["*.py", "*.pyi"]

        Returns:
            匹配的文件列表
        """
        result = []
        for f in files:
            path = Path(f)
            for pattern in patterns:
                if path.match(pattern):
                    result.append(f)
                    break
        return result


async def run_command(cmd: list[str], cwd: str | None = None) -> tuple[int, str, str]:
    """异步执行命令

    Args:
        cmd: 命令和参数列表
        cwd: 工作目录

    Returns:
        (returncode, stdout, stderr)
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout, stderr = await proc.communicate()
    return (
        proc.returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )
