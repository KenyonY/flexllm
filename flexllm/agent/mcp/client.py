"""MCP 客户端连接

支持 stdio 和 SSE 两种模式连接 MCP Server。

Example:
    # stdio 模式
    async with MCPConnection(command="npx @mcp/server-github") as conn:
        tools = await conn.list_tools()

    # SSE 模式
    async with MCPConnection(url="http://localhost:8080/sse") as conn:
        result = await conn.call_tool("search", {"query": "test"})
"""

from __future__ import annotations

import json
import logging
import shlex
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp import ClientSession

logger = logging.getLogger(__name__)


class MCPConnection:
    """MCP Server 连接

    Args:
        command: stdio 模式，启动子进程的命令 (如 "npx @mcp/server-github")
        url: SSE 模式，远程 server URL
        env: 环境变量（stdio 模式）
        name: 连接名称（用于日志和工具前缀）
    """

    def __init__(
        self,
        command: str | list[str] | None = None,
        url: str | None = None,
        env: dict[str, str] | None = None,
        name: str | None = None,
        headers: dict[str, str] | None = None,
        cwd: str | None = None,
    ):
        if not command and not url:
            raise ValueError("必须提供 command (stdio) 或 url (SSE)")
        if command and url:
            raise ValueError("command 和 url 不能同时提供")

        self.command = command
        self.url = url
        self.env = env
        self.name = name or self._infer_name()
        self.headers = headers
        self.cwd = cwd

        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None

    def _infer_name(self) -> str:
        """从 command 或 url 推断名称"""
        if self.command:
            cmd = self.command if isinstance(self.command, str) else " ".join(self.command)
            # 取最后一个路径组件
            parts = cmd.split()
            return parts[-1].split("/")[-1] if parts else "mcp"
        if self.url:
            from urllib.parse import urlparse

            return urlparse(self.url).hostname or "mcp"
        return "mcp"

    async def connect(self):
        """建立连接"""
        try:
            from mcp import ClientSession
        except ImportError:
            raise ImportError("MCP 客户端需要 mcp SDK，请安装: pip install flexllm[agent]")

        self._exit_stack = AsyncExitStack()

        if self.command:
            await self._connect_stdio()
        else:
            await self._connect_sse()

        await self._session.initialize()
        logger.info(f"MCP 连接已建立: {self.name}")

    async def _connect_stdio(self):
        """stdio 模式连接"""
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        if isinstance(self.command, str):
            parts = shlex.split(self.command)
        else:
            parts = list(self.command)

        params_kwargs = {
            "command": parts[0],
            "args": parts[1:],
            "env": self.env,
        }
        if self.cwd:
            params_kwargs["cwd"] = self.cwd
        server_params = StdioServerParameters(**params_kwargs)

        transport = await self._exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

    async def _connect_sse(self):
        """SSE 模式连接"""
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        sse_kwargs = {"url": self.url}
        if self.headers:
            sse_kwargs["headers"] = self.headers
        transport = await self._exit_stack.enter_async_context(sse_client(**sse_kwargs))
        read_stream, write_stream = transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

    async def list_tools(self) -> list[Any]:
        """获取 server 提供的工具列表（返回 mcp.types.Tool 列表）"""
        if not self._session:
            raise RuntimeError("未连接，请先调用 connect() 或使用 async with")
        result = await self._session.list_tools()
        return result.tools

    async def call_tool(self, name: str, arguments: dict | None = None) -> str:
        """调用工具，返回结果文本"""
        if not self._session:
            raise RuntimeError("未连接，请先调用 connect() 或使用 async with")
        result = await self._session.call_tool(name, arguments or {})

        if result.isError:
            parts = []
            for content in result.content:
                parts.append(getattr(content, "text", str(content)))
            return f"[error: {' '.join(parts)}]"

        # 拼接所有文本内容
        parts = []
        for content in result.content:
            if hasattr(content, "text"):
                parts.append(content.text)
            else:
                parts.append(json.dumps(content.model_dump(), ensure_ascii=False))
        return "\n".join(parts)

    async def close(self):
        """关闭连接"""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._session = None
        logger.info(f"MCP 连接已关闭: {self.name}")

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()
