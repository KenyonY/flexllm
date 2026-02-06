"""Test ToolRegistry - 动态工具注册"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from flexllm.agent.tools.base import TOOL_REGISTRY, ToolDef, ToolRegistry
from flexllm.clients.base import ChatCompletionResult, ToolCall


class TestToolRegistryBasic:
    """基础功能测试"""

    def test_init_empty(self):
        registry = ToolRegistry()
        assert len(registry) == 0
        assert registry.names == []

    def test_register_and_access(self):
        registry = ToolRegistry()
        tool = ToolDef(
            name="test",
            description="test tool",
            parameters={"type": "object", "properties": {}, "required": []},
            executor=lambda: "ok",
        )
        registry.register(tool)
        assert "test" in registry
        assert len(registry) == 1
        assert registry.names == ["test"]

    def test_unregister(self):
        registry = ToolRegistry()
        tool = ToolDef(name="test", description="", parameters={}, executor=lambda: "ok")
        registry.register(tool)
        assert "test" in registry
        registry.unregister("test")
        assert "test" not in registry
        # 移除不存在的工具不报错
        registry.unregister("nonexistent")

    def test_from_global(self):
        """从全局 TOOL_REGISTRY 加载"""
        # 全局 registry 中应该有 read, write, edit, glob, grep, bash
        registry = ToolRegistry.from_global(["read", "bash"])
        assert "read" in registry
        assert "bash" in registry
        assert len(registry) == 2

    def test_from_global_unknown(self):
        """加载不存在的工具抛异常"""
        with pytest.raises(ValueError, match="未知工具"):
            ToolRegistry.from_global(["nonexistent_tool_xyz"])

    def test_merge(self):
        """合并两个 registry"""
        r1 = ToolRegistry()
        r1.register(ToolDef(name="a", description="", parameters={}, executor=lambda: "a"))
        r2 = ToolRegistry()
        r2.register(ToolDef(name="b", description="", parameters={}, executor=lambda: "b"))
        r1.merge(r2)
        assert "a" in r1
        assert "b" in r1
        assert len(r1) == 2

    def test_readonly_tools(self):
        registry = ToolRegistry()
        registry.register(
            ToolDef(name="read", description="", parameters={}, executor=lambda: "", readonly=True)
        )
        registry.register(
            ToolDef(
                name="write", description="", parameters={}, executor=lambda: "", readonly=False
            )
        )
        assert registry.readonly_tools == ["read"]
        assert registry.is_readonly("read") is True
        assert registry.is_readonly("write") is False
        assert registry.is_readonly("nonexistent") is False

    def test_repr(self):
        registry = ToolRegistry()
        registry.register(ToolDef(name="a", description="", parameters={}, executor=lambda: ""))
        assert "a" in repr(registry)


class TestToolRegistryGetToolDefs:
    """get_tool_defs 测试"""

    def test_openai_format(self):
        registry = ToolRegistry()
        registry.register(
            ToolDef(
                name="my_tool",
                description="A test tool",
                parameters={
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
                executor=lambda x: x,
            )
        )
        defs = registry.get_tool_defs()
        assert len(defs) == 1
        assert defs[0]["type"] == "function"
        assert defs[0]["function"]["name"] == "my_tool"
        assert defs[0]["function"]["description"] == "A test tool"
        assert defs[0]["function"]["parameters"]["properties"]["x"]["type"] == "string"


class TestToolRegistryExecute:
    """execute 和 execute_async 测试"""

    def test_execute_sync(self):
        registry = ToolRegistry()
        registry.register(
            ToolDef(
                name="add",
                description="",
                parameters={},
                executor=lambda a, b: str(int(a) + int(b)),
            )
        )
        result = registry.execute("add", '{"a": "1", "b": "2"}')
        assert result == "3"

    def test_execute_not_found(self):
        registry = ToolRegistry()
        result = registry.execute("nonexistent", "{}")
        assert "[error:" in result
        assert "not found" in result

    def test_execute_invalid_json(self):
        registry = ToolRegistry()
        registry.register(ToolDef(name="t", description="", parameters={}, executor=lambda: "ok"))
        result = registry.execute("t", "not json")
        assert "[error:" in result
        assert "JSON" in result

    def test_execute_exception(self):
        def bad_fn():
            raise ValueError("boom")

        registry = ToolRegistry()
        registry.register(ToolDef(name="t", description="", parameters={}, executor=bad_fn))
        result = registry.execute("t", "{}")
        assert "[error:" in result
        assert "boom" in result

    @pytest.mark.asyncio
    async def test_execute_async_sync_fn(self):
        registry = ToolRegistry()
        registry.register(
            ToolDef(name="t", description="", parameters={}, executor=lambda: "sync_result")
        )
        result = await registry.execute_async("t", "{}")
        assert result == "sync_result"

    @pytest.mark.asyncio
    async def test_execute_async_async_fn(self):
        async def async_fn(x):
            return f"async_{x}"

        registry = ToolRegistry()
        registry.register(ToolDef(name="t", description="", parameters={}, executor=async_fn))
        result = await registry.execute_async("t", '{"x": "hello"}')
        assert result == "async_hello"

    @pytest.mark.asyncio
    async def test_execute_async_not_found(self):
        registry = ToolRegistry()
        result = await registry.execute_async("nonexistent", "{}")
        assert "[error:" in result


class TestAgentClientWithRegistry:
    """AgentClient + ToolRegistry 集成测试"""

    @pytest.mark.asyncio
    async def test_agent_with_registry(self):
        """使用 ToolRegistry 代替 tools + tool_executor"""
        from flexllm import AgentClient

        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            side_effect=[
                ChatCompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            type="function",
                            function={"name": "echo", "arguments": '{"msg": "hello"}'},
                        ),
                    ],
                ),
                ChatCompletionResult(content="done", tool_calls=None),
            ]
        )

        registry = ToolRegistry()
        registry.register(
            ToolDef(
                name="echo",
                description="Echo message",
                parameters={
                    "type": "object",
                    "properties": {"msg": {"type": "string"}},
                    "required": ["msg"],
                },
                executor=lambda msg: f"echo: {msg}",
            )
        )

        agent = AgentClient(client=mock_client, tool_registry=registry)

        # 验证 tools 被自动设置
        assert agent.tools is not None
        assert len(agent.tools) == 1
        assert agent.tool_executor is None
        assert agent.tool_registry is registry

        result = await agent.run("test")
        assert result.content == "done"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].result == "echo: hello"

    @pytest.mark.asyncio
    async def test_agent_backward_compatible(self):
        """旧接口 tools + tool_executor 仍然工作"""
        from flexllm import AgentClient

        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            side_effect=[
                ChatCompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            type="function",
                            function={"name": "fn", "arguments": "{}"},
                        ),
                    ],
                ),
                ChatCompletionResult(content="ok", tool_calls=None),
            ]
        )

        agent = AgentClient(
            client=mock_client,
            tools=[{"type": "function", "function": {"name": "fn"}}],
            tool_executor=lambda n, a: "old_style_result",
        )

        assert agent.tool_registry is None

        result = await agent.run("test")
        assert result.tool_calls[0].result == "old_style_result"

    @pytest.mark.asyncio
    async def test_registry_with_approval(self):
        """ToolRegistry + approval_handler"""
        from flexllm import AgentClient

        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            side_effect=[
                ChatCompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            type="function",
                            function={"name": "safe", "arguments": "{}"},
                        ),
                        ToolCall(
                            id="c2",
                            type="function",
                            function={"name": "danger", "arguments": "{}"},
                        ),
                    ],
                ),
                ChatCompletionResult(content="done", tool_calls=None),
            ]
        )

        registry = ToolRegistry()
        registry.register(
            ToolDef(
                name="safe",
                description="",
                parameters={},
                executor=lambda: "safe_ok",
                readonly=True,
            )
        )
        registry.register(
            ToolDef(
                name="danger",
                description="",
                parameters={},
                executor=lambda: "danger_ok",
                readonly=False,
            )
        )

        def only_readonly(name, args, readonly):
            return readonly

        agent = AgentClient(
            client=mock_client, tool_registry=registry, approval_handler=only_readonly
        )

        result = await agent.run("test")
        # safe (readonly=True) 应该执行
        assert result.tool_calls[0].result == "safe_ok"
        # danger (readonly=False) 应该被拒绝
        assert "denied" in result.tool_calls[1].result.lower()

    def test_from_global_tools(self):
        """from_global 加载的 registry 能正常工作"""
        registry = ToolRegistry.from_global(["read", "bash"])
        defs = registry.get_tool_defs()
        assert len(defs) == 2
        names = [d["function"]["name"] for d in defs]
        assert "read" in names
        assert "bash" in names

        # 执行 bash
        result = registry.execute("bash", json.dumps({"command": "echo test123"}))
        assert "test123" in result
