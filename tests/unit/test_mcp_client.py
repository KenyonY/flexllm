"""MCP 客户端单元测试"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMCPConnectionInit:
    def test_requires_command_or_url(self):
        from flexllm.agent.mcp.client import MCPConnection

        with pytest.raises(ValueError, match="必须提供"):
            MCPConnection()

    def test_cannot_have_both(self):
        from flexllm.agent.mcp.client import MCPConnection

        with pytest.raises(ValueError, match="不能同时提供"):
            MCPConnection(command="echo", url="http://localhost")

    def test_stdio_mode(self):
        from flexllm.agent.mcp.client import MCPConnection

        conn = MCPConnection(command="npx @mcp/server-github")
        assert conn.command == "npx @mcp/server-github"
        assert conn.url is None
        assert conn.name == "server-github"

    def test_sse_mode(self):
        from flexllm.agent.mcp.client import MCPConnection

        conn = MCPConnection(url="http://localhost:8080/sse")
        assert conn.url == "http://localhost:8080/sse"
        assert conn.command is None
        assert conn.name == "localhost"

    def test_custom_name(self):
        from flexllm.agent.mcp.client import MCPConnection

        conn = MCPConnection(command="echo test", name="my-server")
        assert conn.name == "my-server"

    def test_command_as_list(self):
        from flexllm.agent.mcp.client import MCPConnection

        conn = MCPConnection(command=["npx", "@mcp/server-github"])
        assert conn.command == ["npx", "@mcp/server-github"]

    def test_infer_name_from_command(self):
        from flexllm.agent.mcp.client import MCPConnection

        conn = MCPConnection(command="python -m my_server")
        assert conn.name == "my_server"

    def test_env_param(self):
        from flexllm.agent.mcp.client import MCPConnection

        conn = MCPConnection(command="echo", env={"TOKEN": "abc"})
        assert conn.env == {"TOKEN": "abc"}


class TestMCPConnectionLifecycle:
    @pytest.mark.asyncio
    async def test_not_connected_raises(self):
        from flexllm.agent.mcp.client import MCPConnection

        conn = MCPConnection(command="echo test")
        with pytest.raises(RuntimeError, match="未连接"):
            await conn.list_tools()
        with pytest.raises(RuntimeError, match="未连接"):
            await conn.call_tool("test", {})

    @pytest.mark.asyncio
    async def test_connect_stdio(self):
        from flexllm.agent.mcp.client import MCPConnection

        conn = MCPConnection(command="echo test")
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()

        # mock _connect_stdio，直接设置 session
        async def fake_connect_stdio():
            conn._session = mock_session

        conn._connect_stdio = fake_connect_stdio

        from contextlib import AsyncExitStack

        conn._exit_stack = AsyncExitStack()
        await conn.connect()
        mock_session.initialize.assert_called_once()
        assert conn._session is mock_session

        await conn.close()
        assert conn._session is None

    @pytest.mark.asyncio
    async def test_connect_sse(self):
        from flexllm.agent.mcp.client import MCPConnection

        conn = MCPConnection(url="http://localhost:8080/sse")
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()

        async def fake_connect_sse():
            conn._session = mock_session

        conn._connect_sse = fake_connect_sse

        from contextlib import AsyncExitStack

        conn._exit_stack = AsyncExitStack()
        await conn.connect()
        mock_session.initialize.assert_called_once()

        await conn.close()

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        from flexllm.agent.mcp.client import MCPConnection

        conn = MCPConnection(command="echo test")
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()

        async def fake_connect_stdio():
            conn._session = mock_session

        conn._connect_stdio = fake_connect_stdio

        from contextlib import AsyncExitStack

        # mock connect 和 close
        original_connect = conn.connect

        async def patched_connect():
            conn._exit_stack = AsyncExitStack()
            await fake_connect_stdio()
            await conn._session.initialize()

        conn.connect = patched_connect

        async with conn:
            assert conn._session is mock_session


class TestMCPConnectionOperations:
    def _make_connected_conn(self):
        """创建一个已 mock 连接的 MCPConnection"""
        from flexllm.agent.mcp.client import MCPConnection

        conn = MCPConnection(command="echo test")
        conn._session = AsyncMock()
        return conn

    @pytest.mark.asyncio
    async def test_list_tools(self):
        conn = self._make_connected_conn()

        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.description = "Search something"
        mock_tool.inputSchema = {"type": "object", "properties": {"query": {"type": "string"}}}

        mock_result = MagicMock()
        mock_result.tools = [mock_tool]
        conn._session.list_tools = AsyncMock(return_value=mock_result)

        tools = await conn.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "search"

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        conn = self._make_connected_conn()

        mock_content = MagicMock()
        mock_content.text = "result text"
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.content = [mock_content]
        conn._session.call_tool = AsyncMock(return_value=mock_result)

        result = await conn.call_tool("search", {"query": "test"})
        assert result == "result text"

    @pytest.mark.asyncio
    async def test_call_tool_error(self):
        conn = self._make_connected_conn()

        mock_content = MagicMock()
        mock_content.text = "not found"
        mock_result = MagicMock()
        mock_result.isError = True
        mock_result.content = [mock_content]
        conn._session.call_tool = AsyncMock(return_value=mock_result)

        result = await conn.call_tool("search", {"query": "test"})
        assert "[error:" in result
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_call_tool_no_args(self):
        conn = self._make_connected_conn()

        mock_content = MagicMock()
        mock_content.text = "ok"
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.content = [mock_content]
        conn._session.call_tool = AsyncMock(return_value=mock_result)

        await conn.call_tool("ping")
        conn._session.call_tool.assert_called_once_with("ping", {})


class TestConverter:
    def _make_mock_tool(self, name="search", description="Search", schema=None, annotations=None):
        tool = MagicMock()
        tool.name = name
        tool.description = description
        tool.inputSchema = schema or {"type": "object", "properties": {"q": {"type": "string"}}}
        tool.annotations = annotations
        return tool

    def test_mcp_tool_to_tool_def(self):
        from flexllm.agent.mcp.client import MCPConnection
        from flexllm.agent.mcp.converter import mcp_tool_to_tool_def

        conn = MCPConnection(command="echo test")
        mcp_tool = self._make_mock_tool()

        tool_def = mcp_tool_to_tool_def(mcp_tool, conn)
        assert tool_def.name == "search"
        assert tool_def.description == "Search"
        assert tool_def.readonly is True
        assert "properties" in tool_def.parameters

    def test_mcp_tool_destructive_annotation(self):
        from flexllm.agent.mcp.client import MCPConnection
        from flexllm.agent.mcp.converter import mcp_tool_to_tool_def

        conn = MCPConnection(command="echo test")
        annotations = MagicMock()
        annotations.destructiveHint = True
        annotations.readOnlyHint = None
        mcp_tool = self._make_mock_tool(annotations=annotations)

        tool_def = mcp_tool_to_tool_def(mcp_tool, conn)
        assert tool_def.readonly is False

    @pytest.mark.asyncio
    async def test_mcp_tools_to_registry(self):
        from flexllm.agent.mcp.client import MCPConnection
        from flexllm.agent.mcp.converter import mcp_tools_to_registry

        conn = MCPConnection(command="echo test")
        conn._session = AsyncMock()

        mock_tool1 = self._make_mock_tool(name="read")
        mock_tool2 = self._make_mock_tool(name="write")
        mock_result = MagicMock()
        mock_result.tools = [mock_tool1, mock_tool2]
        conn._session.list_tools = AsyncMock(return_value=mock_result)

        registry = await mcp_tools_to_registry([conn])
        assert len(registry) == 2
        assert "read" in registry
        assert "write" in registry

    @pytest.mark.asyncio
    async def test_name_conflict_adds_prefix(self):
        from flexllm.agent.mcp.client import MCPConnection
        from flexllm.agent.mcp.converter import mcp_tools_to_registry

        conn1 = MCPConnection(command="echo test", name="server1")
        conn1._session = AsyncMock()
        conn2 = MCPConnection(command="echo test", name="server2")
        conn2._session = AsyncMock()

        mock_tool = self._make_mock_tool(name="search")
        result1 = MagicMock()
        result1.tools = [mock_tool]
        result2 = MagicMock()
        result2.tools = [self._make_mock_tool(name="search")]
        conn1._session.list_tools = AsyncMock(return_value=result1)
        conn2._session.list_tools = AsyncMock(return_value=result2)

        registry = await mcp_tools_to_registry([conn1, conn2])
        assert len(registry) == 2
        assert "search" in registry
        assert "server2.search" in registry

    @pytest.mark.asyncio
    async def test_executor_calls_connection(self):
        from flexllm.agent.mcp.client import MCPConnection
        from flexllm.agent.mcp.converter import mcp_tool_to_tool_def

        conn = MCPConnection(command="echo test")
        conn._session = AsyncMock()

        mock_content = MagicMock()
        mock_content.text = "result"
        mock_call_result = MagicMock()
        mock_call_result.isError = False
        mock_call_result.content = [mock_content]
        conn._session.call_tool = AsyncMock(return_value=mock_call_result)

        mcp_tool = self._make_mock_tool()
        tool_def = mcp_tool_to_tool_def(mcp_tool, conn)

        result = await tool_def.executor(q="test")
        assert result == "result"
        conn._session.call_tool.assert_called_once_with("search", {"q": "test"})

    @pytest.mark.asyncio
    async def test_empty_connections(self):
        from flexllm.agent.mcp.converter import mcp_tools_to_registry

        registry = await mcp_tools_to_registry([])
        assert len(registry) == 0
