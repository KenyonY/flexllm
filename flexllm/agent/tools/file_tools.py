"""文件操作工具：Read/Write/Edit"""

from pathlib import Path

from .base import register_tool


def _fuzzy_find(content: str, old_string: str) -> str | None:
    """尝试用空白容错匹配 old_string（忽略行尾空白差异）。

    Returns:
        匹配到的实际文本（原始内容中的），或 None
    """
    content_lines = content.split("\n")
    search_lines = old_string.split("\n")
    if not search_lines:
        return None

    norm_search = [line.rstrip() for line in search_lines]

    for i in range(len(content_lines) - len(search_lines) + 1):
        match = True
        for j, ns in enumerate(norm_search):
            if content_lines[i + j].rstrip() != ns:
                match = False
                break
        if match:
            matched = content_lines[i : i + len(search_lines)]
            return "\n".join(matched)
    return None


def _find_closest_hint(content: str, old_string: str) -> str:
    """查找最接近的匹配位置，给出行号提示。"""
    import re

    first_line = old_string.split("\n")[0].strip()
    if not first_line:
        return "  hint: old_string starts with an empty line, check your input"

    content_lines = content.split("\n")
    matches = []

    # 精确子串匹配
    for i, line in enumerate(content_lines):
        if first_line in line:
            matches.append((i + 1, line.rstrip()))
            if len(matches) >= 3:
                break

    # 如果没找到，尝试标识符关键词匹配（提取字母数字标识符，忽略短词）
    if not matches:
        identifiers = [w for w in re.findall(r"[a-zA-Z_]\w{2,}", first_line)]
        if identifiers:
            keyword = max(identifiers, key=len)
            for i, line in enumerate(content_lines):
                if keyword in line:
                    matches.append((i + 1, line.rstrip()))
                    if len(matches) >= 3:
                        break

    if matches:
        hint_lines = ["  hint: similar content found at:"]
        for line_num, line_content in matches:
            preview = line_content[:120] + "..." if len(line_content) > 120 else line_content
            hint_lines.append(f"    line {line_num}: {preview}")
        return "\n".join(hint_lines)

    return "  hint: no similar content found in file"


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
        # 尝试空白容错匹配（忽略行尾空白差异）
        actual_match = _fuzzy_find(content, old_string)
        if actual_match:
            match_count = content.count(actual_match)
            if match_count > 1 and not replace_all:
                return (
                    f"[error: fuzzy match found {match_count} times, "
                    "set replace_all=true or provide more context to make it unique]"
                )
            replaced = match_count if replace_all else 1
            new_content = content.replace(actual_match, new_string, 0 if replace_all else 1)
            try:
                path.write_text(new_content, encoding="utf-8")
                return (
                    f"[edited: {file_path}, replaced {replaced} occurrence(s) "
                    "(fuzzy match: trailing whitespace differences ignored)]"
                )
            except Exception as e:
                return f"[error: write failed: {e}]"

        # 完全找不到，给出定位提示
        hint = _find_closest_hint(content, old_string)
        preview = old_string[:100] + "..." if len(old_string) > 100 else old_string
        return f"[error: old_string not found in file]\n  searched for: {repr(preview)}\n{hint}"

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
