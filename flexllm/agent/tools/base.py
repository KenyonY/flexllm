"""工具基类和注册器

支持装饰器自动生成 JSON Schema，简化工具开发。
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Callable, get_type_hints

TOOL_REGISTRY: dict[str, "ToolDef"] = {}


@dataclass
class ToolDef:
    """工具定义"""

    name: str
    description: str
    parameters: dict  # JSON Schema
    executor: Callable  # (kwargs) -> str
    readonly: bool = True  # 是否只读（影响确认逻辑）


def _python_type_to_json_schema(py_type: type) -> dict:
    """Python 类型转 JSON Schema 类型"""
    if py_type is str:
        return {"type": "string"}
    elif py_type is int:
        return {"type": "integer"}
    elif py_type is float:
        return {"type": "number"}
    elif py_type is bool:
        return {"type": "boolean"}
    elif py_type is list:
        return {"type": "array"}
    elif py_type is dict:
        return {"type": "object"}
    else:
        # 默认字符串
        return {"type": "string"}


def _extract_params_schema(func: Callable) -> dict:
    """从函数签名和 docstring 提取 JSON Schema"""
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

    properties = {}
    required = []

    # 解析 docstring 中的 Args 部分获取参数描述
    param_docs = {}
    if func.__doc__:
        in_args = False
        current_param = None
        for line in func.__doc__.split("\n"):
            stripped = line.strip()
            if stripped.startswith("Args:"):
                in_args = True
                continue
            if in_args:
                if stripped.startswith("Returns:") or stripped.startswith("Raises:"):
                    break
                if ":" in stripped and not stripped.startswith(" "):
                    # 新参数行: "param_name: description"
                    parts = stripped.split(":", 1)
                    current_param = parts[0].strip()
                    param_docs[current_param] = parts[1].strip() if len(parts) > 1 else ""
                elif current_param and stripped:
                    # 续行描述
                    param_docs[current_param] += " " + stripped

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        py_type = hints.get(param_name, str)
        schema = _python_type_to_json_schema(py_type)

        # 添加描述
        if param_name in param_docs:
            schema["description"] = param_docs[param_name]

        properties[param_name] = schema

        # 无默认值的参数为 required
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def register_tool(name: str, description: str, readonly: bool = True):
    """装饰器：注册工具，自动从函数签名生成 JSON Schema

    Example:
        @register_tool("read", "读取文件内容", readonly=True)
        def read_file(file_path: str, offset: int = 0, limit: int = 2000) -> str:
            ...
    """

    def decorator(func: Callable) -> Callable:
        params = _extract_params_schema(func)
        TOOL_REGISTRY[name] = ToolDef(
            name=name,
            description=description,
            parameters=params,
            executor=func,
            readonly=readonly,
        )
        return func

    return decorator


def get_tool_defs(names: list[str]) -> list[dict]:
    """获取工具的 OpenAI 格式定义

    Args:
        names: 工具名称列表

    Returns:
        OpenAI tools 格式的列表
    """
    tool_defs = []
    for name in names:
        if name not in TOOL_REGISTRY:
            available = ", ".join(TOOL_REGISTRY.keys())
            raise ValueError(f"未知工具: {name}，可用: {available}")

        tool = TOOL_REGISTRY[name]
        tool_defs.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
        )
    return tool_defs


def make_tool_executor(names: list[str]) -> Callable[[str, str], str]:
    """创建工具执行器

    Args:
        names: 启用的工具名称列表

    Returns:
        执行器函数 (name, arguments_json) -> result_str
    """
    enabled = {name: TOOL_REGISTRY[name] for name in names if name in TOOL_REGISTRY}

    def executor(name: str, arguments: str) -> str:
        if name not in enabled:
            return f"[error: tool '{name}' not enabled]"

        tool = enabled[name]
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as e:
            return f"[error: invalid JSON arguments: {e}]"

        try:
            result = tool.executor(**args)
            return str(result)
        except Exception as e:
            return f"[error: {type(e).__name__}: {e}]"

    return executor


def get_all_tool_names() -> list[str]:
    """获取所有已注册工具的名称"""
    return list(TOOL_REGISTRY.keys())
