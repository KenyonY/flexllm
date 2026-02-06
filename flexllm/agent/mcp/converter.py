"""MCP 工具 → ToolDef 转换"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..tools.base import ToolDef, ToolRegistry
    from .client import MCPConnection

logger = logging.getLogger(__name__)


def mcp_tool_to_tool_def(mcp_tool: Any, connection: "MCPConnection") -> "ToolDef":
    """将单个 MCP 工具转换为 flexllm ToolDef

    Args:
        mcp_tool: mcp.types.Tool 对象
        connection: 所属的 MCPConnection（用于绑定 executor）
    """
    from ..tools.base import ToolDef

    tool_name = mcp_tool.name
    description = mcp_tool.description or tool_name
    parameters = mcp_tool.inputSchema or {"type": "object", "properties": {}}

    # 判断是否只读（基于 MCP annotations）
    readonly = True
    if mcp_tool.annotations:
        # destructiveHint / readOnlyHint 等
        if getattr(mcp_tool.annotations, "destructiveHint", None):
            readonly = False
        if getattr(mcp_tool.annotations, "readOnlyHint", None) is False:
            readonly = False

    async def executor(**kwargs):
        return await connection.call_tool(tool_name, kwargs)

    return ToolDef(
        name=tool_name,
        description=description,
        parameters=parameters,
        executor=executor,
        readonly=readonly,
    )


async def mcp_tools_to_registry(
    connections: list["MCPConnection"],
) -> "ToolRegistry":
    """将多个 MCP server 的工具合并为一个 ToolRegistry

    每个工具的 executor 自动绑定到对应的 MCPConnection.call_tool。
    工具名称冲突时以 "{connection.name}.{tool_name}" 处理。

    Args:
        connections: 已连接的 MCPConnection 列表
    """
    from ..tools.base import ToolDef, ToolRegistry

    registry = ToolRegistry()
    seen_names: dict[str, str] = {}  # tool_name -> connection.name

    for conn in connections:
        mcp_tools = await conn.list_tools()
        for mcp_tool in mcp_tools:
            tool_def = mcp_tool_to_tool_def(mcp_tool, conn)

            if tool_def.name in seen_names:
                # 名称冲突，加前缀
                prefixed_name = f"{conn.name}.{tool_def.name}"
                logger.warning(
                    f"工具名称冲突: '{tool_def.name}' "
                    f"(来自 {seen_names[tool_def.name]} 和 {conn.name})，"
                    f"使用 '{prefixed_name}'"
                )
                tool_def = ToolDef(
                    name=prefixed_name,
                    description=tool_def.description,
                    parameters=tool_def.parameters,
                    executor=tool_def.executor,
                    readonly=tool_def.readonly,
                )
            else:
                seen_names[tool_def.name] = conn.name

            registry.register(tool_def)

    logger.info(f"从 {len(connections)} 个 MCP server 加载了 {len(registry)} 个工具")
    return registry
