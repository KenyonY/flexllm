import ast
import re
import json5
from typing import Optional


def extract_code_snippets(text, strict=True):
    """Extract code snippets"""
    # 首先处理带有 ``` 标志的代码块
    pattern = r"```(\w+)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, text)

    code_snippets = []
    for lang, code in matches:
        code_snippets.append({
            "language": lang.strip() if lang else "unknown",
            "code": code.strip(),
        })

    if not strict:
        # 查找并排除已经被处理过的 ``` ... ``` 内的代码块
        text = re.sub(pattern, "", text)

        # 处理剩下的 { ... } 格式的代码块
        pattern = r"\{[\s\S]*?\}"
        matches = re.findall(pattern, text)

        for code in matches:
            code_snippets.append({
                "language": "unknown",
                "code": code.strip(),
            })

    return code_snippets


def parse_to_obj(text: str, strict=False):
    """Parse to obj"""
    code_snippets = extract_code_snippets(text, strict=strict)
    code_snippets = [code_snippet["code"] for code_snippet in code_snippets]
    code_snippets = [code_snippet.strip() for code_snippet in code_snippets if code_snippet.strip()]
    if not code_snippets:
        return None
    code_str = code_snippets[-1]
    try:
        return ast.literal_eval(code_str)
    except:
        return json5.loads(code_str)


def parse_to_code(text: str, strict=False) -> Optional[str]:
    """Parse to code"""
    code_snippets = extract_code_snippets(text, strict=strict)
    code_snippets = [code_snippet["code"] for code_snippet in code_snippets]
    code_snippets = [code_snippet.strip() for code_snippet in code_snippets if code_snippet.strip()]
    if not code_snippets:
        return None
    code_str = code_snippets[-1]
    return code_str

