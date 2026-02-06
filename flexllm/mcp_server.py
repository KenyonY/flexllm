"""flexllm MCP Server

将 flexllm 的 LLM 调用和 Agent 能力暴露为 MCP server，
让 Claude Desktop、Cursor 等 MCP client 直接调用。

用法:
    flexllm mcp-server                    # 启动 MCP server (stdio)
    flexllm mcp-server --tools code       # 同时暴露内置工具
"""

from __future__ import annotations

import asyncio
import json
import logging

logger = logging.getLogger(__name__)


def create_mcp_server(
    model: str = "",
    base_url: str = "",
    api_key: str = "EMPTY",
    system_prompt: str | None = None,
    tools: str | None = None,
    max_rounds: int = 10,
):
    """创建 MCP Server 实例

    Args:
        model: LLM 模型名称
        base_url: LLM API 地址
        api_key: API 密钥
        system_prompt: 系统提示词
        tools: Agent 工具集（如 "code"、"all"）
        max_rounds: Agent 最大 tool-use 轮数
    """
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool
    except ImportError:
        raise ImportError("MCP Server 需要 mcp SDK，请安装: pip install flexllm[agent]")

    server = Server("flexllm")
    _config = {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "system_prompt": system_prompt,
        "tools": tools,
        "max_rounds": max_rounds,
    }

    @server.list_tools()
    async def list_tools():
        tool_list = [
            Tool(
                name="llm_chat",
                description="调用 LLM 生成回复。支持多轮对话。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "description": "消息列表，格式: [{role, content}, ...]",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                                "required": ["role", "content"],
                            },
                        },
                        "temperature": {
                            "type": "number",
                            "description": "采样温度（0-2）",
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "最大生成 token 数",
                        },
                    },
                    "required": ["messages"],
                },
            ),
            Tool(
                name="llm_ask",
                description="快速向 LLM 提问，返回回复文本。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "问题内容",
                        },
                        "system": {
                            "type": "string",
                            "description": "系统提示词（可选）",
                        },
                    },
                    "required": ["question"],
                },
            ),
        ]

        # 如果配置了工具，暴露 agent_run
        if _config["tools"]:
            tool_list.append(
                Tool(
                    name="agent_run",
                    description=(
                        "执行 agent 任务（tool-use 循环）。"
                        f"可用工具: {_config['tools']}。"
                        "Agent 会自动调用工具完成任务。"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "任务描述",
                            },
                            "max_rounds": {
                                "type": "integer",
                                "description": f"最大 tool-use 轮数（默认 {_config['max_rounds']}）",
                            },
                        },
                        "required": ["task"],
                    },
                )
            )

        return tool_list

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        from flexllm import LLMClient

        async with LLMClient(
            model=_config["model"],
            base_url=_config["base_url"],
            api_key=_config["api_key"],
        ) as client:
            if name == "llm_chat":
                messages = arguments["messages"]
                kwargs = {}
                if "temperature" in arguments:
                    kwargs["temperature"] = arguments["temperature"]
                if "max_tokens" in arguments:
                    kwargs["max_tokens"] = arguments["max_tokens"]
                result = await client.chat_completions(messages, **kwargs)
                return [TextContent(type="text", text=str(result))]

            elif name == "llm_ask":
                question = arguments["question"]
                system = arguments.get("system") or _config["system_prompt"]
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": question})
                result = await client.chat_completions(messages)
                return [TextContent(type="text", text=str(result))]

            elif name == "agent_run":
                from flexllm import AgentClient

                from .cli.chat_helpers import _build_registry

                task = arguments["task"]
                max_rounds = arguments.get("max_rounds", _config["max_rounds"])
                registry = _build_registry(_config["tools"])

                agent = AgentClient(
                    client=client,
                    system=_config["system_prompt"],
                    tool_registry=registry,
                    max_rounds=max_rounds,
                )
                result = await agent.run(task)

                output = result.content or ""
                if result.tool_calls:
                    output += f"\n\n[{len(result.tool_calls)} tool calls, {result.rounds} rounds]"
                return [TextContent(type="text", text=output)]

            else:
                return [TextContent(type="text", text=f"[error: unknown tool '{name}']")]

    return server


def run_mcp_server(
    model: str = "",
    base_url: str = "",
    api_key: str = "EMPTY",
    system_prompt: str | None = None,
    tools: str | None = None,
    max_rounds: int = 10,
):
    """启动 MCP server (stdio 模式)"""
    from mcp.server.stdio import stdio_server

    server = create_mcp_server(
        model=model,
        base_url=base_url,
        api_key=api_key,
        system_prompt=system_prompt,
        tools=tools,
        max_rounds=max_rounds,
    )

    async def _run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(_run())
