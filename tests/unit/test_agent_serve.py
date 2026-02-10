"""Agent HTTP 服务端点单元测试"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, TestClient, TestServer

from flexllm.agent.types import AgentResult, ToolCallRecord
from flexllm.serve import AgentSessionManager, ServeConfig, ServeServer


class TestAgentSessionManager:
    def test_get_or_create_new(self):
        mgr = AgentSessionManager(ttl=60)
        agent = mgr.get_or_create(
            session_id="s1",
            client=MagicMock(),
            system="test",
            tool_registry=None,
            max_rounds=5,
        )
        assert agent is not None
        assert "s1" in mgr._sessions

    def test_get_or_create_existing(self):
        mgr = AgentSessionManager(ttl=60)
        client = MagicMock()
        agent1 = mgr.get_or_create("s1", client, "test", None, 5)
        agent2 = mgr.get_or_create("s1", client, "test", None, 5)
        assert agent1 is agent2

    def test_cleanup_expired(self):
        mgr = AgentSessionManager(ttl=0)  # TTL=0，立即过期
        mgr.get_or_create("s1", MagicMock(), "test", None, 5)
        # 让 timestamp 过期
        mgr._timestamps["s1"] = time.time() - 1
        mgr.cleanup_expired()
        assert "s1" not in mgr._sessions

    def test_cleanup_not_expired(self):
        mgr = AgentSessionManager(ttl=3600)
        mgr.get_or_create("s1", MagicMock(), "test", None, 5)
        mgr.cleanup_expired()
        assert "s1" in mgr._sessions


class TestServeConfigAgent:
    def test_default_no_tools(self):
        config = ServeConfig()
        assert config.tools is None
        assert config.max_rounds == 10

    def test_with_tools(self):
        config = ServeConfig(tools="code", max_rounds=5)
        assert config.tools == "code"
        assert config.max_rounds == 5


# Helper for aiohttp test client
def _make_mock_agent_result(content="test response", rounds=1):
    return AgentResult(
        content=content,
        rounds=rounds,
        tool_calls=[ToolCallRecord(name="bash", arguments='{"cmd": "ls"}', result="file.py")],
        usage={"prompt_tokens": 10, "completion_tokens": 20},
    )


@pytest.fixture
def serve_config():
    return ServeConfig(
        model="test-model",
        base_url="http://localhost:8080",
        api_key="test-key",
        tools="code",
        max_rounds=5,
    )


@pytest.fixture
def serve_server(serve_config):
    server = ServeServer(serve_config)
    # Mock internal state
    server._client = MagicMock()
    server._tool_registry = None
    server._session_mgr = AgentSessionManager()
    return server


@pytest.fixture
async def client(serve_server, aiohttp_client):
    app = serve_server._create_app()
    return await aiohttp_client(app)


class TestAgentRunEndpoint:
    @pytest.mark.asyncio
    async def test_agent_run_success(self, client, serve_server):
        mock_result = _make_mock_agent_result()
        with patch.object(
            serve_server,
            "_make_agent",
            return_value=MagicMock(run=AsyncMock(return_value=mock_result)),
        ):
            resp = await client.post(
                "/api/agent/run",
                json={"content": "列出文件"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["content"] == "test response"
            assert data["rounds"] == 1
            assert len(data["tool_calls"]) == 1
            assert data["tool_calls"][0]["name"] == "bash"

    @pytest.mark.asyncio
    async def test_agent_run_missing_content(self, client):
        resp = await client.post("/api/agent/run", json={})
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_agent_run_invalid_json(self, client):
        resp = await client.post("/api/agent/run", data=b"not json")
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_agent_run_with_custom_system(self, client, serve_server):
        mock_result = _make_mock_agent_result()
        mock_agent = MagicMock(run=AsyncMock(return_value=mock_result))
        with patch.object(serve_server, "_make_agent", return_value=mock_agent) as make:
            await client.post(
                "/api/agent/run",
                json={"content": "test", "system": "自定义 system"},
            )
            make.assert_called_once_with(system="自定义 system", max_rounds=None)


class TestAgentChatEndpoint:
    @pytest.mark.asyncio
    async def test_agent_chat_success(self, client, serve_server):
        mock_result = _make_mock_agent_result()
        mock_agent = MagicMock(chat=AsyncMock(return_value=mock_result))
        with patch.object(serve_server._session_mgr, "get_or_create", return_value=mock_agent):
            resp = await client.post(
                "/api/agent/chat",
                json={"content": "你好", "session_id": "s1"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["content"] == "test response"

    @pytest.mark.asyncio
    async def test_agent_chat_missing_session_id(self, client):
        resp = await client.post("/api/agent/chat", json={"content": "test"})
        assert resp.status == 400
        data = await resp.json()
        assert "session_id" in data["error"]

    @pytest.mark.asyncio
    async def test_agent_chat_missing_content(self, client):
        resp = await client.post("/api/agent/chat", json={"session_id": "s1"})
        assert resp.status == 400


class TestAgentRunStreamEndpoint:
    @pytest.mark.asyncio
    async def test_agent_run_stream_success(self, client, serve_server):
        mock_result = _make_mock_agent_result()
        mock_agent = MagicMock(run=AsyncMock(return_value=mock_result))
        # No callbacks set in mock; just verify stream returns
        with patch.object(serve_server, "_make_agent", return_value=mock_agent):
            resp = await client.post(
                "/api/agent/run/stream",
                json={"content": "列出文件"},
            )
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "text/event-stream"
            body = await resp.read()
            text = body.decode("utf-8")
            # 应该包含 done 事件
            assert '"type": "done"' in text

    @pytest.mark.asyncio
    async def test_agent_run_stream_missing_content(self, client):
        resp = await client.post("/api/agent/run/stream", json={})
        assert resp.status == 400


class TestNoToolsConfig:
    @pytest.mark.asyncio
    async def test_agent_endpoints_not_available_without_tools(self, aiohttp_client):
        config = ServeConfig(model="test", base_url="http://localhost:8080")
        server = ServeServer(config)
        server._client = MagicMock()
        app = server._create_app()
        client = await aiohttp_client(app)

        resp = await client.post("/api/agent/run", json={"content": "test"})
        assert resp.status == 404

        resp = await client.post("/api/agent/chat", json={"content": "test", "session_id": "s1"})
        assert resp.status == 404
