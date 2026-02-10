"""Chat 和 Agent 相关的辅助函数"""

from __future__ import annotations

import asyncio
import os
import sys

from .utils import apply_user_template

AGENT_DEFAULT_SYSTEM = f"""You are an autonomous agent at {os.getcwd()}.

Loop: think briefly → use tools → observe results → continue until done.

Rules:
- Prefer tools over prose. Act, don't just explain.
- If a tool call fails, analyze the error and try alternative approaches.
- Never give up after a single failure — try different methods, URLs, or commands.
- After finishing, summarize what you did and the results."""


def _parse_validators(validate_str: str) -> list:
    """解析验证器字符串

    Args:
        validate_str: 验证器名称，如 "python" 或 "python,pytest" 或 "syntax,lint,type"

    Returns:
        验证器实例列表
    """
    from flexllm.agent.validators import (
        PytestValidator,
        PythonLintValidator,
        PythonSyntaxValidator,
        PythonTypeValidator,
    )

    # 预设组合
    VALIDATOR_PRESETS = {
        "python": [
            PythonSyntaxValidator(),
            PythonLintValidator(),
            PythonTypeValidator(),
        ],
    }

    # 单个验证器映射
    VALIDATOR_MAP = {
        "syntax": PythonSyntaxValidator,
        "lint": PythonLintValidator,
        "type": PythonTypeValidator,
        "pytest": PytestValidator,
        # 别名
        "ruff": PythonLintValidator,
        "pyright": PythonTypeValidator,
    }

    # 检查是否是预设
    if validate_str in VALIDATOR_PRESETS:
        return VALIDATOR_PRESETS[validate_str]

    # 解析逗号分隔的验证器列表
    validators = []
    for name in validate_str.split(","):
        name = name.strip()
        if name in VALIDATOR_MAP:
            validators.append(VALIDATOR_MAP[name]())
        elif name in VALIDATOR_PRESETS:
            validators.extend(VALIDATOR_PRESETS[name])
        else:
            available = list(VALIDATOR_MAP.keys()) + list(VALIDATOR_PRESETS.keys())
            raise ValueError(f"未知验证器: {name}，可用: {', '.join(available)}")

    return validators


# ========== Chat 辅助函数 ==========


def single_chat(
    message,
    model,
    base_url,
    api_key,
    system_prompt,
    temperature,
    max_tokens,
    stream,
    user_template=None,
):
    """单次对话"""

    async def _run():
        from flexllm import LLMClient

        async with LLMClient(model=model, base_url=base_url, api_key=api_key) as client:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            user_content = apply_user_template(message, user_template)
            messages.append({"role": "user", "content": user_content})

            if stream:
                print("Assistant: ", end="", flush=True)
                async for chunk in client.chat_completions_stream(
                    messages, temperature=temperature, max_tokens=max_tokens
                ):
                    print(chunk, end="", flush=True)
                print()
            else:
                result = await client.chat_completions(
                    messages, temperature=temperature, max_tokens=max_tokens
                )
                print(f"Assistant: {result}")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\n[中断]")
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)


def interactive_chat(
    model, base_url, api_key, system_prompt, temperature, max_tokens, stream, user_template=None
):
    """多轮交互对话"""

    async def _run():
        from flexllm import LLMClient

        async with LLMClient(model=model, base_url=base_url, api_key=api_key) as client:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            print("\n多轮对话模式")
            print(f"模型: {model}")
            print(f"服务器: {base_url}")
            print("输入 'quit' 或 Ctrl+C 退出")
            print("-" * 50)

            while True:
                try:
                    user_input = input("\nYou: ").strip()

                    if user_input.lower() in ["quit", "exit", "q"]:
                        print("再见！")
                        break

                    if not user_input:
                        continue

                    user_content = apply_user_template(user_input, user_template)
                    messages.append({"role": "user", "content": user_content})

                    if stream:
                        print("Assistant: ", end="", flush=True)
                        full_response = ""
                        async for chunk in client.chat_completions_stream(
                            messages, temperature=temperature, max_tokens=max_tokens
                        ):
                            print(chunk, end="", flush=True)
                            full_response += chunk
                        print()
                        messages.append({"role": "assistant", "content": full_response})
                    else:
                        result = await client.chat_completions(
                            messages, temperature=temperature, max_tokens=max_tokens
                        )
                        print(f"Assistant: {result}")
                        messages.append({"role": "assistant", "content": result})

                except EOFError:
                    print("\n再见！")
                    break

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\n再见！")


# ========== Agent 辅助函数 ==========


def agent_chat(
    model,
    base_url,
    api_key,
    system_prompt,
    model_params,
    tools_name,
    message=None,
    verbose=False,
    max_rounds=10,
    approve="auto",
    mcp_servers=None,
):
    """Agent 模式的交互式对话"""

    async def _run():
        from flexllm import AgentClient, LLMClient

        from .agent_console import AgentConsole

        effective_system = system_prompt or AGENT_DEFAULT_SYSTEM
        registry = _build_registry(tools_name, mcp_servers)
        ui = AgentConsole(verbose=verbose)

        mcp_conns = []
        try:
            if mcp_servers:
                mcp_conns = await _connect_mcp_servers(mcp_servers)
                from flexllm.agent.mcp.converter import mcp_tools_to_registry

                mcp_registry = await mcp_tools_to_registry(mcp_conns)
                registry.merge(mcp_registry)

            async with LLMClient(model=model, base_url=base_url, api_key=api_key) as client:
                enable_subagent = tools_name in ("all", "code") or "task" in tools_name
                agent = AgentClient(
                    client=client,
                    system=effective_system,
                    tool_registry=registry,
                    max_rounds=max_rounds,
                    enable_subagent=enable_subagent,
                )

                if approve == "manual":
                    from flexllm.agent.client import console_approval

                    agent.approval_handler = console_approval

                agent.on_tool_call = ui.on_tool_call
                agent.on_tool_result = ui.on_tool_result
                agent.on_llm_response = ui.on_llm_response

                if message:
                    ui.begin()
                    result = await agent.run(message, **model_params)
                    ui.end()
                    ui.print_summary(result)
                    ui.print_result(result)
                    return

                ui.print_chat_header(model, tools_name, mcp_servers, verbose)

                while True:
                    try:
                        user_input = input("\nYou: ").strip()
                        if user_input.lower() in ["quit", "exit", "q"]:
                            print("再见！")
                            break
                        if not user_input:
                            continue

                        ui.begin()
                        result = await agent.chat(user_input, **model_params)
                        ui.end()
                        ui.print_summary(result)
                        ui.print_result(result)

                    except EOFError:
                        print("\n再见！")
                        break
        finally:
            ui.end()
            for conn in mcp_conns:
                await conn.close()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\n再见！")


def _build_registry(tools_name, mcp_servers=None):
    """构建 ToolRegistry"""
    from flexllm.agent.tools.base import TOOL_REGISTRY, ToolRegistry

    names = [t.strip() for t in tools_name.split(",") if t.strip()]

    # 处理特殊值
    expanded_names = []
    for name in names:
        if name == "all":
            expanded_names.extend(TOOL_REGISTRY.keys())
        elif name == "code":
            expanded_names.extend(["read", "edit", "glob", "grep", "bash"])
        else:
            expanded_names.append(name)
    names = list(dict.fromkeys(expanded_names))

    # 检查未知工具
    unknown = [n for n in names if n not in TOOL_REGISTRY]
    if unknown:
        available = ", ".join(sorted(TOOL_REGISTRY.keys()))
        raise ValueError(f"未知的工具: {', '.join(unknown)}，可用: {available}")

    return ToolRegistry.from_global(names) if names else ToolRegistry()

    return registry


async def _connect_mcp_servers(mcp_servers):
    """连接 MCP servers，返回连接列表"""
    from flexllm.agent.mcp import MCPConnection

    conns = []
    for server_spec in mcp_servers:
        if server_spec.startswith("http://") or server_spec.startswith("https://"):
            conn = MCPConnection(url=server_spec)
        else:
            conn = MCPConnection(command=server_spec)
        await conn.connect()
        conns.append(conn)
    return conns


def agent_run(
    message,
    model,
    base_url,
    api_key,
    system_prompt,
    model_params,
    tools_name,
    max_rounds,
    verbose=False,
    validate=None,
    max_fix_attempts=3,
    approve="auto",
    mcp_servers=None,
):
    """Agent 非交互式执行

    Args:
        validate: 验证器名称，如 "python" 或 "python,pytest"，None 表示不验证
        max_fix_attempts: 验证失败时最大修复尝试次数
        mcp_servers: MCP server 命令或 URL 列表
    """

    async def _run():
        from flexllm import AgentClient, LLMClient

        from .agent_console import AgentConsole

        effective_system = system_prompt or AGENT_DEFAULT_SYSTEM
        registry = _build_registry(tools_name)
        validators = _parse_validators(validate) if validate else None
        ui = AgentConsole(verbose=verbose)

        mcp_conns = []
        try:
            if mcp_servers:
                mcp_conns = await _connect_mcp_servers(mcp_servers)
                from flexllm.agent.mcp.converter import mcp_tools_to_registry

                mcp_registry = await mcp_tools_to_registry(mcp_conns)
                registry.merge(mcp_registry)

            async with LLMClient(model=model, base_url=base_url, api_key=api_key) as client:
                enable_subagent = tools_name in ("all", "code") or "task" in tools_name
                agent = AgentClient(
                    client=client,
                    system=effective_system,
                    tool_registry=registry,
                    max_rounds=max_rounds,
                    enable_subagent=enable_subagent,
                )

                if approve == "manual":
                    from flexllm.agent.client import console_approval

                    agent.approval_handler = console_approval

                agent.on_tool_call = ui.on_tool_call
                agent.on_tool_result = ui.on_tool_result
                agent.on_llm_response = ui.on_llm_response
                if validators:
                    agent.on_validation = ui.on_validation

                ui.print_header(model, tools_name, mcp_servers, validators, message)
                ui.begin()

                if validators:
                    result = await agent.run_with_validation(
                        message,
                        validators=validators,
                        max_fix_attempts=max_fix_attempts,
                        **model_params,
                    )
                else:
                    result = await agent.run(message, **model_params)

                ui.end()
                ui.print_summary(result)
                ui.print_result(result)
        finally:
            ui.end()
            for conn in mcp_conns:
                await conn.close()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\n[中断]")
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
