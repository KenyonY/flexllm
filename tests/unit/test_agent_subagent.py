"""Subagent 机制单元测试"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flexllm.agent.agent_types import AGENT_TYPES, get_agent_type, register_agent_type
from flexllm.agent.tools.base import TOOL_REGISTRY, ToolDef, ToolRegistry
from flexllm.agent.tools.task_tool import TASK_TOOL_SCHEMA

# ========== Agent Type Registry ==========


def test_agent_type_registry():
    """AGENT_TYPES 包含 explore/code/plan"""
    assert "explore" in AGENT_TYPES
    assert "code" in AGENT_TYPES
    assert "plan" in AGENT_TYPES

    for name, config in AGENT_TYPES.items():
        assert "description" in config
        assert "tools" in config
        assert "prompt" in config


def test_get_agent_type_invalid():
    """未知类型抛 ValueError"""
    with pytest.raises(ValueError, match="未知 agent 类型"):
        get_agent_type("nonexistent")


def test_register_custom_agent_type():
    """register_agent_type 正常工作"""
    register_agent_type(
        name="test_custom",
        description="测试自定义代理",
        tools=["read", "bash"],
        prompt="你是测试代理。",
    )
    assert "test_custom" in AGENT_TYPES
    config = get_agent_type("test_custom")
    assert config["tools"] == ["read", "bash"]
    assert config["prompt"] == "你是测试代理。"

    # 清理
    del AGENT_TYPES["test_custom"]


# ========== Task Tool Schema ==========


def test_task_tool_schema():
    """TASK_TOOL_SCHEMA 格式正确"""
    assert TASK_TOOL_SCHEMA["name"] == "task"
    assert "description" in TASK_TOOL_SCHEMA
    params = TASK_TOOL_SCHEMA["parameters"]
    assert params["type"] == "object"
    assert "description" in params["properties"]
    assert "prompt" in params["properties"]
    assert "agent_type" in params["properties"]
    assert params["properties"]["agent_type"]["enum"] == ["explore", "code", "plan"]
    assert set(params["required"]) == {"description", "prompt", "agent_type"}


# ========== Subagent Registry 构建 ==========


def test_subagent_registry_explore():
    """explore 类型无 write/edit"""
    from flexllm.agent.client import AgentClient

    mock_client = MagicMock()
    registry = ToolRegistry.from_global(list(TOOL_REGISTRY.keys()))
    agent = AgentClient(client=mock_client, tool_registry=registry)

    config = get_agent_type("explore")
    sub_registry = agent._build_subagent_registry(config)

    assert "read" in sub_registry
    assert "glob" in sub_registry
    assert "grep" in sub_registry
    assert "bash" in sub_registry
    # explore 不含 write/edit
    if "write" in TOOL_REGISTRY:
        assert "write" not in sub_registry
    if "edit" in TOOL_REGISTRY:
        assert "edit" not in sub_registry


def test_subagent_registry_code():
    """code 类型包含所有内置工具"""
    from flexllm.agent.client import AgentClient

    mock_client = MagicMock()
    registry = ToolRegistry.from_global(list(TOOL_REGISTRY.keys()))
    agent = AgentClient(client=mock_client, tool_registry=registry)

    config = get_agent_type("code")
    sub_registry = agent._build_subagent_registry(config)

    # code 类型 tools="*"，应包含所有内置工具
    for name in TOOL_REGISTRY:
        assert name in sub_registry


def test_subagent_no_task_tool():
    """子代理 registry 不含 task（防递归）"""
    from flexllm.agent.client import AgentClient

    mock_client = MagicMock()
    registry = ToolRegistry.from_global(list(TOOL_REGISTRY.keys()))
    agent = AgentClient(client=mock_client, tool_registry=registry, enable_subagent=True)

    # 父 agent 应该有 task 工具
    assert "task" in agent._tool_registry

    # 子代理不应有 task
    config = get_agent_type("code")
    sub_registry = agent._build_subagent_registry(config)
    assert "task" not in sub_registry


def test_subagent_mcp_inheritance():
    """MCP 工具被继承到子代理"""
    from flexllm.agent.client import AgentClient

    mock_client = MagicMock()
    registry = ToolRegistry.from_global(["read", "bash"])

    # 模拟 MCP 工具
    mcp_tool = ToolDef(
        name="mcp_search",
        description="MCP 搜索工具",
        parameters={"type": "object", "properties": {}, "required": []},
        executor=lambda: "result",
        readonly=True,
    )
    registry.register(mcp_tool)

    agent = AgentClient(client=mock_client, tool_registry=registry)

    config = get_agent_type("explore")
    sub_registry = agent._build_subagent_registry(config)

    # MCP 工具应被继承
    assert "mcp_search" in sub_registry
    # 内置工具根据 agent_type 过滤
    assert "read" in sub_registry
    assert "bash" in sub_registry


# ========== enable_subagent 参数 ==========


def test_enable_subagent_injects_task():
    """enable_subagent=True 时注入 task 工具"""
    from flexllm.agent.client import AgentClient

    mock_client = MagicMock()
    registry = ToolRegistry.from_global(["read", "bash"])
    agent = AgentClient(client=mock_client, tool_registry=registry, enable_subagent=True)

    assert "task" in agent._tool_registry
    # tools 列表中也应包含 task
    tool_names = [t["function"]["name"] for t in agent.tools]
    assert "task" in tool_names


def test_disable_subagent_no_task():
    """enable_subagent=False（默认）不注入 task 工具"""
    from flexllm.agent.client import AgentClient

    mock_client = MagicMock()
    registry = ToolRegistry.from_global(["read", "bash"])
    agent = AgentClient(client=mock_client, tool_registry=registry)

    assert "task" not in agent._tool_registry
    tool_names = [t["function"]["name"] for t in agent.tools]
    assert "task" not in tool_names


# ========== 边界条件 ==========


def test_enable_subagent_without_registry():
    """enable_subagent=True 但无 tool_registry 时不崩溃"""
    from flexllm.agent.client import AgentClient

    mock_client = MagicMock()
    # tool_registry=None，enable_subagent=True 应安全跳过
    agent = AgentClient(client=mock_client, enable_subagent=True)
    assert agent._tool_registry is None


def test_subagent_system_prompt_with_parent_system():
    """父 agent 有 system 时，子代理继承部分上下文"""
    from flexllm.agent.client import AgentClient

    mock_client = MagicMock()
    registry = ToolRegistry.from_global(["read", "bash"])
    parent_system = "你是一个代码助手，工作目录是 /home/user/project"
    agent = AgentClient(
        client=mock_client, system=parent_system, tool_registry=registry, enable_subagent=True
    )

    config = get_agent_type("explore")
    # 通过 task_executor 中的逻辑验证：直接检查闭包会构建的 system
    # 子代理 system 应包含 agent_type prompt + 父 system 片段
    sub_system = config["prompt"]
    if agent.system:
        sub_system = f"{config['prompt']}\n\n工作目录上下文：\n{agent.system[:500]}"
    assert config["prompt"] in sub_system
    assert "代码助手" in sub_system


def test_subagent_system_prompt_without_parent_system():
    """父 agent 无 system 时，子代理只用 agent_type prompt"""
    from flexllm.agent.client import AgentClient

    mock_client = MagicMock()
    registry = ToolRegistry.from_global(["read", "bash"])
    agent = AgentClient(client=mock_client, tool_registry=registry, enable_subagent=True)

    config = get_agent_type("explore")
    sub_system = config["prompt"]
    if agent.system:
        sub_system = f"{config['prompt']}\n\n工作目录上下文：\n{agent.system[:500]}"
    # 无父 system，子 system 应等于 agent_type prompt
    assert sub_system == config["prompt"]


# ========== task executor 集成 ==========


async def test_task_executor_runs_subagent():
    """task executor 创建子 AgentClient 并返回结果"""
    from flexllm.agent.client import AgentClient
    from flexllm.agent.types import AgentResult

    mock_client = MagicMock()
    registry = ToolRegistry.from_global(["read", "bash"])
    agent = AgentClient(client=mock_client, tool_registry=registry, enable_subagent=True)

    # Mock 子 agent 的 run 方法
    mock_result = AgentResult(content="探索完成：共 5 个文件", rounds=2, tool_calls=[])
    with patch.object(AgentClient, "run", new_callable=AsyncMock, return_value=mock_result):
        task_tool = agent._tool_registry._tools["task"]
        result = await task_tool.executor(
            description="探索项目结构",
            prompt="列出所有 Python 文件",
            agent_type="explore",
        )

    assert "探索完成" in result


async def test_task_executor_handles_empty_content():
    """子代理返回空内容时使用兜底文案"""
    from flexllm.agent.client import AgentClient
    from flexllm.agent.types import AgentResult

    mock_client = MagicMock()
    registry = ToolRegistry.from_global(["read", "bash"])
    agent = AgentClient(client=mock_client, tool_registry=registry, enable_subagent=True)

    mock_result = AgentResult(content=None, rounds=1, tool_calls=[])
    with patch.object(AgentClient, "run", new_callable=AsyncMock, return_value=mock_result):
        task_tool = agent._tool_registry._tools["task"]
        result = await task_tool.executor(
            description="空任务",
            prompt="什么也不做",
            agent_type="explore",
        )

    assert result == "(subagent 未返回内容)"
