"""文件操作工具：Read/Write/Edit"""

from pathlib import Path

from .base import register_tool


@register_tool("read", "读取文件内容，返回带行号的文本", readonly=True)
def read_file(file_path: str, offset: int = 0, limit: int = 2000) -> str:
    """读取文件，支持分页。

    Args:
        file_path: 文件绝对路径
        offset: 起始行号（从 0 开始）
        limit: 读取行数（默认 2000）

    Returns:
        带行号的文件内容
    """
    path = Path(file_path)
    if not path.exists():
        return f"[error: file not found: {file_path}]"
    if path.is_dir():
        return f"[error: path is a directory: {file_path}]"

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = path.read_text(encoding="latin-1")
        except Exception as e:
            return f"[error: cannot read file: {e}]"
    except Exception as e:
        return f"[error: {e}]"

    lines = content.splitlines()
    total_lines = len(lines)

    if offset >= total_lines:
        return f"[info: file has {total_lines} lines, offset {offset} is beyond end]"

    selected = lines[offset : offset + limit]

    result = []
    for i, line in enumerate(selected, start=offset + 1):
        # 截断过长的行
        if len(line) > 2000:
            line = line[:2000] + "..."
        result.append(f"{i:>6}| {line}")

    output = "\n".join(result)

    # 添加分页提示
    end_line = offset + len(selected)
    if end_line < total_lines:
        output += f"\n\n[showing lines {offset + 1}-{end_line} of {total_lines}, use offset={end_line} to continue]"

    return output


@register_tool("write", "创建或覆盖文件", readonly=False)
def write_file(file_path: str, content: str) -> str:
    """写入文件（覆盖模式）。

    Args:
        file_path: 文件绝对路径
        content: 文件内容

    Returns:
        写入结果
    """
    path = Path(file_path)

    try:
        # 自动创建父目录
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        lines = content.count("\n") + 1
        return f"[written: {file_path} ({lines} lines, {len(content)} chars)]"
    except Exception as e:
        return f"[error: {e}]"


@register_tool("edit", "精确替换文件中的字符串", readonly=False)
def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """精确字符串替换。

    Args:
        file_path: 文件绝对路径
        old_string: 要替换的原字符串
        new_string: 替换后的新字符串
        replace_all: 是否替换所有匹配（默认只替换第一个）

    Returns:
        替换结果或错误信息
    """
    path = Path(file_path)
    if not path.exists():
        return f"[error: file not found: {file_path}]"

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = path.read_text(encoding="latin-1")
        except Exception as e:
            return f"[error: cannot read file: {e}]"
    except Exception as e:
        return f"[error: {e}]"

    count = content.count(old_string)

    if count == 0:
        # 提供更有帮助的错误信息
        preview = old_string[:100] + "..." if len(old_string) > 100 else old_string
        return f"[error: old_string not found in file]\n  searched for: {repr(preview)}"

    if count > 1 and not replace_all:
        return f"[error: old_string found {count} times, set replace_all=true or provide more context to make it unique]"

    if replace_all:
        new_content = content.replace(old_string, new_string)
        replaced = count
    else:
        new_content = content.replace(old_string, new_string, 1)
        replaced = 1

    try:
        path.write_text(new_content, encoding="utf-8")
        return f"[edited: {file_path}, replaced {replaced} occurrence(s)]"
    except Exception as e:
        return f"[error: write failed: {e}]"
