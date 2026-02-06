"""CLI MCP 集成测试"""

import pytest


class TestBuildRegistry:
    def test_build_code_registry(self):
        from flexllm.cli.chat_helpers import _build_registry

        registry = _build_registry("code")
        assert "read" in registry
        assert "edit" in registry
        assert "glob" in registry
        assert "grep" in registry
        assert "bash" in registry
        assert len(registry) == 5

    def test_build_all_registry(self):
        from flexllm.cli.chat_helpers import _build_registry

        registry = _build_registry("all")
        assert "read" in registry
        assert "write" in registry
        assert len(registry) >= 6

    def test_build_shell_registry(self):
        from flexllm.cli.chat_helpers import _build_registry

        registry = _build_registry("shell")
        assert "shell" in registry
        assert len(registry) == 1

    def test_build_mixed_registry(self):
        from flexllm.cli.chat_helpers import _build_registry

        registry = _build_registry("read,bash,shell")
        assert "read" in registry
        assert "bash" in registry
        assert "shell" in registry
        assert len(registry) == 3

    def test_build_unknown_tool_raises(self):
        from flexllm.cli.chat_helpers import _build_registry

        with pytest.raises(ValueError, match="未知的工具"):
            _build_registry("nonexistent_tool_xyz")

    def test_shell_executor_works(self):
        from flexllm.cli.chat_helpers import _build_registry

        registry = _build_registry("shell")
        result = registry.execute("shell", '{"command": "echo hello_test_123"}')
        assert "hello_test_123" in result

    def test_legacy_tool_is_not_readonly(self):
        from flexllm.cli.chat_helpers import _build_registry

        registry = _build_registry("shell")
        assert registry.is_readonly("shell") is False


class TestConnectMCPServers:
    @pytest.mark.asyncio
    async def test_detect_url_vs_command(self):
        """测试 URL 和命令的区分逻辑"""
        from flexllm.agent.mcp import MCPConnection

        # URL 模式
        conn = MCPConnection(url="http://localhost:8080/sse")
        assert conn.url == "http://localhost:8080/sse"
        assert conn.command is None

        # 命令模式
        conn = MCPConnection(command="npx @mcp/server-github")
        assert conn.command == "npx @mcp/server-github"
        assert conn.url is None


class TestTopLevelExports:
    def test_toolregistry_exported(self):
        from flexllm import ToolDef, ToolRegistry

        assert ToolRegistry is not None
        assert ToolDef is not None

    def test_agent_exports(self):
        from flexllm import AgentClient, AgentResult, ToolCallRecord

        assert AgentClient is not None
        assert AgentResult is not None
        assert ToolCallRecord is not None

    def test_approval_exports(self):
        from flexllm.agent import auto_approve, console_approval

        assert callable(auto_approve)
        assert callable(console_approval)

    def test_lazy_imports(self):
        """测试延迟导入模块"""
        from flexllm import agent

        # 这些应该通过 __getattr__ 延迟导入
        mcp = agent.mcp
        assert hasattr(mcp, "MCPConnection")
        assert hasattr(mcp, "mcp_tools_to_registry")

        tracing = agent.tracing
        assert hasattr(tracing, "Span")
        assert hasattr(tracing, "Trace")
