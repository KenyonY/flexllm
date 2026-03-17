"""Shell 命令执行工具"""

import subprocess

from .base import register_tool


@register_tool(
    "bash",
    "执行 shell 命令并返回输出。可运行任意命令，包括 curl/wget 访问网络、pip 安装包等。",
    readonly=False,
)
def bash_exec(command: str, timeout: int = 60) -> str:
    """执行 shell 命令。

    Args:
        command: 要执行的命令
        timeout: 超时时间（秒），默认 60

    Returns:
        命令输出或错误信息
    """
    if not command or not command.strip():
        return "[error: empty command]"

    # 限制超时时间
    timeout = min(max(timeout, 1), 300)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"[error: command timed out after {timeout}s]"
    except Exception as e:
        return f"[error: {e}]"
