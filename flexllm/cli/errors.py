"""统一错误处理模块

提供语义化退出码、结构化错误输出和 dry-run 支持。
- TTY 模式：中文文本 "错误: xxx" 到 stderr
- 非 TTY 模式：JSON 到 stderr（适合 Agent/脚本解析）

退出码:
  0  成功
  1  一般错误
  2  参数/用法错误
  3  资源未找到
  4  认证失败
  5  冲突/已存在
  6  网络错误
  7  依赖缺失
  8  文件 IO 错误
  10 dry-run 通过
"""

from __future__ import annotations

import enum
import json
import sys


class ExitCode(enum.IntEnum):
    SUCCESS = 0
    ERROR = 1
    USAGE = 2
    NOT_FOUND = 3
    AUTH = 4
    CONFLICT = 5
    NETWORK = 6
    DEPENDENCY = 7
    IO_ERROR = 8
    DRY_RUN = 10


class ErrorType(str, enum.Enum):
    INVALID_ARGS = "invalid_args"
    NOT_FOUND = "not_found"
    AUTH_FAILED = "auth_failed"
    CONFLICT = "conflict"
    NETWORK_ERROR = "network_error"
    DEPENDENCY_MISSING = "dependency_missing"
    IO_ERROR = "io_error"
    GENERAL = "general_error"


_ERROR_EXIT_MAP: dict[ErrorType, ExitCode] = {
    ErrorType.INVALID_ARGS: ExitCode.USAGE,
    ErrorType.NOT_FOUND: ExitCode.NOT_FOUND,
    ErrorType.AUTH_FAILED: ExitCode.AUTH,
    ErrorType.CONFLICT: ExitCode.CONFLICT,
    ErrorType.NETWORK_ERROR: ExitCode.NETWORK,
    ErrorType.DEPENDENCY_MISSING: ExitCode.DEPENDENCY,
    ErrorType.IO_ERROR: ExitCode.IO_ERROR,
    ErrorType.GENERAL: ExitCode.ERROR,
}


def cli_error(
    error_type: ErrorType,
    message: str,
    suggestion: str | None = None,
    retryable: bool = False,
) -> None:
    """统一错误输出并退出。

    TTY:   "错误: {message}" + 可选 "提示: {suggestion}" 到 stderr
    非 TTY: JSON 对象到 stderr
    """
    import typer

    exit_code = _ERROR_EXIT_MAP.get(error_type, ExitCode.ERROR)

    if sys.stderr.isatty():
        print(f"错误: {message}", file=sys.stderr)
        if suggestion:
            print(f"提示: {suggestion}", file=sys.stderr)
    else:
        err = {"error": error_type.value, "message": message, "retryable": retryable}
        if suggestion:
            err["suggestion"] = suggestion
        print(json.dumps(err, ensure_ascii=False), file=sys.stderr)

    raise typer.Exit(exit_code)


def dry_run_output(data: dict) -> None:
    """输出 dry-run 预览 JSON 并以 DRY_RUN 退出码退出。"""
    import typer

    print(json.dumps(data, indent=2, ensure_ascii=False))
    raise typer.Exit(ExitCode.DRY_RUN)
