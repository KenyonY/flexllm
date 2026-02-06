"""flexllm.agent.mcp - MCP 客户端集成

连接外部 MCP Server，将其工具注册到 ToolRegistry。

Example:
    from flexllm.agent.mcp import MCPConnection, mcp_tools_to_registry

    async with MCPConnection(command="npx @mcp/server-github") as conn:
        registry = await mcp_tools_to_registry([conn])
"""

from .client import MCPConnection
from .converter import mcp_tools_to_registry

__all__ = ["MCPConnection", "mcp_tools_to_registry"]
