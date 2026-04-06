"""统一错误处理模块

为 AI Agent 友好的 CLI 错误输出提供语义化退出码、结构化 JSON、
失败输入回显（context）与帮助引用（doc）。

## 设计原则（参考 agent-cli-guide Principle 9）

Agent 读不懂散文式错误消息，但擅长解析结构化字段。好的错误输出应满足：

1. **机器可读的错误类型码**（error 字段）
2. **人类可读的描述**（message 字段）
3. **具体的恢复建议**（suggestion 字段，尽可能是可执行命令）
4. **瞬时 vs 永久**（retryable 字段）
5. **回显失败输入**（context 字段，让 Agent 无需正则就能读出实际值）
6. **帮助引用**（doc 字段，通常是 "flexllm <cmd> --help"）

## JSON 输出格式（非 TTY / 管道场景）

```json
{
  "error": "invalid_args",
  "message": "--save-input 参数值无效",
  "context": {
    "arg": "--save-input",
    "received": "foo",
    "expected": ["true", "last", "false"]
  },
  "suggestion": "使用 --save-input true 保存完整输入",
  "doc": "flexllm batch --help",
  "retryable": false
}
```

## TTY 输出格式（终端人工使用）

```
错误: --save-input 参数值无效
  arg: --save-input
  received: foo
  expected: true, last, false
提示: 使用 --save-input true 保存完整输入
详见: flexllm batch --help
```

## 退出码（稳定 API，跨版本语义不变）

| 码 | 含义 | 典型来源 |
|----|------|---------|
| 0  | 成功 | 正常完成 |
| 1  | 一般错误 | 未分类异常 |
| 2  | 参数/用法错误 | INVALID_ARGS |
| 3  | 资源未找到 | NOT_FOUND（模型、配置、文件） |
| 4  | 认证失败 | AUTH_FAILED（API Key 无效） |
| 5  | 冲突/已存在 | CONFLICT |
| 6  | 网络错误 | NETWORK_ERROR（retryable 常为 True） |
| 7  | 依赖缺失 | DEPENDENCY_MISSING（缺 pip 包） |
| 8  | 文件 IO 错误 | IO_ERROR |
| 10 | dry-run 通过 | dry_run_output() |
"""

from __future__ import annotations

import enum
import json
import sys
from typing import Any


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


def _format_context_value(value: Any) -> str:
    """TTY 模式下将 context 值格式化为单行字符串。"""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def cli_error(
    error_type: ErrorType,
    message: str,
    suggestion: str | None = None,
    retryable: bool = False,
    context: dict[str, Any] | None = None,
    doc: str | None = None,
) -> None:
    """统一错误输出并退出。

    Args:
        error_type: 错误类型枚举（决定退出码）
        message: 人类可读的错误描述（保持简短，详细信息放 context）
        suggestion: 恢复建议，**强烈推荐提供可执行命令**
                    如 "flexllm list" 而非 "查看可用模型列表"
        retryable: 瞬时错误（网络超时、限流、服务器 5xx）设为 True
                   Agent 据此决定是否退避重试
        context: 失败输入/状态回显字典，让 Agent 无需正则就能读出值：
                 {"arg": "--format", "received": "xml", "expected": ["json", "table"]}
                 {"file": "data.jsonl", "line": 42, "field": "messages"}
        doc: 更多帮助的引用，通常是对应命令的 "flexllm <cmd> --help"

    输出:
        TTY:   中文文本 + 缩进的 context 键值对到 stderr
        非 TTY: 紧凑 JSON 到 stderr（一行，便于逐行解析）

    退出:
        按 error_type 映射到语义化退出码
    """
    import typer

    exit_code = _ERROR_EXIT_MAP.get(error_type, ExitCode.ERROR)

    if sys.stderr.isatty():
        print(f"错误: {message}", file=sys.stderr)
        if context:
            for k, v in context.items():
                print(f"  {k}: {_format_context_value(v)}", file=sys.stderr)
        if suggestion:
            print(f"提示: {suggestion}", file=sys.stderr)
        if doc:
            print(f"详见: {doc}", file=sys.stderr)
    else:
        err: dict[str, Any] = {
            "error": error_type.value,
            "message": message,
            "retryable": retryable,
        }
        if context:
            err["context"] = context
        if suggestion:
            err["suggestion"] = suggestion
        if doc:
            err["doc"] = doc
        print(json.dumps(err, ensure_ascii=False), file=sys.stderr)

    raise typer.Exit(exit_code)


def dry_run_output(data: dict) -> None:
    """输出 dry-run 预览 JSON 并以 DRY_RUN 退出码退出。"""
    import typer

    print(json.dumps(data, indent=2, ensure_ascii=False))
    raise typer.Exit(ExitCode.DRY_RUN)
