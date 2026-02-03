"""搜索工具：Glob/Grep"""

import re
import subprocess
from pathlib import Path

from .base import register_tool


@register_tool("glob", "按模式匹配文件路径", readonly=True)
def glob_files(pattern: str, path: str = ".") -> str:
    """按 glob 模式搜索文件。

    Args:
        pattern: glob 模式，如 "**/*.py", "src/**/*.ts"
        path: 搜索根目录（默认当前目录）

    Returns:
        匹配的文件路径列表，按修改时间排序（最新在前）
    """
    root = Path(path)
    if not root.exists():
        return f"[error: path not found: {path}]"

    try:
        matches = list(root.glob(pattern))
    except Exception as e:
        return f"[error: invalid pattern: {e}]"

    if not matches:
        return "(no matches)"

    # 按修改时间排序（最新的在前）
    try:
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        # 某些文件可能无法访问，忽略排序错误
        pass

    # 限制结果数量
    total = len(matches)
    if total > 100:
        matches = matches[:100]
        suffix = f"\n\n[showing 100 of {total} matches]"
    else:
        suffix = f"\n\n[{total} matches]"

    return "\n".join(str(m) for m in matches) + suffix


@register_tool("grep", "在文件中搜索内容", readonly=True)
def grep_content(
    pattern: str,
    path: str = ".",
    glob: str = None,
    output_mode: str = "files_with_matches",
    context: int = 0,
    case_insensitive: bool = False,
) -> str:
    """使用正则搜索文件内容。

    Args:
        pattern: 正则表达式
        path: 搜索路径
        glob: 文件过滤模式，如 "*.py"
        output_mode: "files_with_matches" | "content" | "count"
        context: 显示匹配行前后的行数（仅 content 模式）
        case_insensitive: 是否忽略大小写

    Returns:
        搜索结果
    """
    # 优先使用 ripgrep
    if _has_ripgrep():
        return _ripgrep(pattern, path, glob, output_mode, context, case_insensitive)
    else:
        return _python_grep(pattern, path, glob, output_mode, context, case_insensitive)


def _has_ripgrep() -> bool:
    """检查 ripgrep 是否可用"""
    try:
        subprocess.run(["rg", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _ripgrep(
    pattern: str,
    path: str,
    glob: str | None,
    output_mode: str,
    context: int,
    case_insensitive: bool,
) -> str:
    """使用 ripgrep 搜索"""
    cmd = ["rg", pattern, path]

    if glob:
        cmd.extend(["--glob", glob])
    if case_insensitive:
        cmd.append("-i")

    if output_mode == "files_with_matches":
        cmd.append("-l")
    elif output_mode == "count":
        cmd.append("-c")
    else:  # content
        cmd.append("-n")  # 显示行号
        if context > 0:
            cmd.extend(["-C", str(context)])

    # 限制输出
    cmd.extend(["--max-count", "100"])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip()
        if not output:
            return "(no matches)"
        return output
    except subprocess.TimeoutExpired:
        return "[error: search timed out after 30s]"
    except Exception as e:
        return f"[error: ripgrep failed: {e}]"


def _python_grep(
    pattern: str,
    path: str,
    glob_pattern: str | None,
    output_mode: str,
    context: int,
    case_insensitive: bool,
) -> str:
    """纯 Python 实现的 grep（ripgrep 不可用时的后备）"""
    root = Path(path)
    if not root.exists():
        return f"[error: path not found: {path}]"

    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"[error: invalid regex: {e}]"

    # 收集要搜索的文件
    if root.is_file():
        files = [root]
    else:
        file_pattern = glob_pattern or "**/*"
        files = [f for f in root.glob(file_pattern) if f.is_file()]

    results = []
    match_count = 0
    max_results = 100

    for file_path in files:
        if match_count >= max_results:
            break

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
        except Exception:
            continue

        file_matches = []
        for i, line in enumerate(lines):
            if regex.search(line):
                file_matches.append((i, line))
                match_count += 1
                if match_count >= max_results:
                    break

        if file_matches:
            if output_mode == "files_with_matches":
                results.append(str(file_path))
            elif output_mode == "count":
                results.append(f"{file_path}:{len(file_matches)}")
            else:  # content
                for line_num, line_content in file_matches:
                    if context > 0:
                        start = max(0, line_num - context)
                        end = min(len(lines), line_num + context + 1)
                        for j in range(start, end):
                            prefix = ">" if j == line_num else " "
                            results.append(f"{file_path}:{j + 1}:{prefix}{lines[j]}")
                        results.append("--")
                    else:
                        results.append(f"{file_path}:{line_num + 1}:{line_content}")

    if not results:
        return "(no matches)"

    output = "\n".join(results)
    if match_count >= max_results:
        output += f"\n\n[truncated at {max_results} matches]"
    return output
