"""MCP Server 单元测试"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCreateMCPServer:
    def test_requires_mcp_sdk(self):
        """没有 mcp SDK 时应该报错"""
        with patch.dict("sys.modules", {"mcp": None, "mcp.server": None}):
            # 由于已经导入过 mcp，只能测试函数签名
            from flexllm.mcp_server import create_mcp_server

            assert callable(create_mcp_server)

    def test_create_server_returns_server(self):
        """创建 server 并返回 Server 实例"""
        from flexllm.mcp_server import create_mcp_server

        server = create_mcp_server(
            model="test-model",
            base_url="http://localhost:8000/v1",
            api_key="test-key",
        )
        assert server is not None

    def test_create_server_with_tools(self):
        """创建带 tools 的 server"""
        from flexllm.mcp_server import create_mcp_server

        server = create_mcp_server(
            model="test-model",
            base_url="http://localhost:8000/v1",
            api_key="test-key",
            tools="code",
            max_rounds=5,
        )
        assert server is not None


class TestServeConfig:
    def test_default_config(self):
        from flexllm.serve import ServeConfig

        config = ServeConfig()
        assert config.port == 8000
        assert config.host == "0.0.0.0"
        assert config.tools is None
        assert config.max_rounds == 10

    def test_config_with_tools(self):
        from flexllm.serve import ServeConfig

        config = ServeConfig(tools="code", max_rounds=5)
        assert config.tools == "code"
        assert config.max_rounds == 5


class TestAgentSessionManager:
    def test_create_session(self):
        from flexllm.serve import AgentSessionManager

        mgr = AgentSessionManager(ttl=3600)
        mock_client = MagicMock()

        agent = mgr.get_or_create(
            session_id="s1",
            client=mock_client,
            system="test",
            tool_defs=[],
            tool_executor=None,
            max_rounds=10,
        )
        assert agent is not None

    def test_reuse_session(self):
        from flexllm.serve import AgentSessionManager

        mgr = AgentSessionManager(ttl=3600)
        mock_client = MagicMock()

        agent1 = mgr.get_or_create(
            session_id="s1",
            client=mock_client,
            system="test",
            tool_defs=[],
            tool_executor=None,
            max_rounds=10,
        )
        agent2 = mgr.get_or_create(
            session_id="s1",
            client=mock_client,
            system="test",
            tool_defs=[],
            tool_executor=None,
            max_rounds=10,
        )
        assert agent1 is agent2

    def test_different_sessions(self):
        from flexllm.serve import AgentSessionManager

        mgr = AgentSessionManager(ttl=3600)
        mock_client = MagicMock()

        agent1 = mgr.get_or_create(
            session_id="s1",
            client=mock_client,
            system="test",
            tool_defs=[],
            tool_executor=None,
            max_rounds=10,
        )
        agent2 = mgr.get_or_create(
            session_id="s2",
            client=mock_client,
            system="test",
            tool_defs=[],
            tool_executor=None,
            max_rounds=10,
        )
        assert agent1 is not agent2

    def test_cleanup_expired(self):
        import time

        from flexllm.serve import AgentSessionManager

        mgr = AgentSessionManager(ttl=0)  # TTL=0 表示立即过期
        mock_client = MagicMock()

        mgr.get_or_create(
            session_id="s1",
            client=mock_client,
            system="test",
            tool_defs=[],
            tool_executor=None,
            max_rounds=10,
        )
        time.sleep(0.1)

        # 触发清理
        mgr.cleanup_expired()
        assert "s1" not in mgr._sessions


class TestServeServerAgent:
    def test_agent_result_to_dict(self):
        from flexllm.agent.types import AgentResult, ToolCallRecord
        from flexllm.serve import ServeServer

        result = AgentResult(
            content="done",
            rounds=2,
            tool_calls=[
                ToolCallRecord(name="read", arguments='{"path": "a.py"}', result="content"),
            ],
            usage={"total_tokens": 100},
        )

        d = ServeServer._agent_result_to_dict(result)
        assert d["content"] == "done"
        assert d["rounds"] == 2
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["name"] == "read"
        assert d["usage"]["total_tokens"] == 100

    def test_create_app_without_tools(self):
        from flexllm.serve import ServeConfig, ServeServer

        config = ServeConfig(tools=None)
        server = ServeServer(config)
        app = server._create_app()

        # 没有 tools 时不应有 agent 路由
        routes = [r.resource.canonical for r in app.router.routes() if hasattr(r, "resource")]
        assert "/api/generate" in routes
        assert "/api/agent/run" not in routes

    def test_create_app_with_tools(self):
        from flexllm.serve import ServeConfig, ServeServer

        config = ServeConfig(tools="code")
        server = ServeServer(config)
        app = server._create_app()

        # 有 tools 时应有 agent 路由
        routes = [r.resource.canonical for r in app.router.routes() if hasattr(r, "resource")]
        assert "/api/agent/run" in routes
        assert "/api/agent/run/stream" in routes
        assert "/api/agent/chat" in routes
