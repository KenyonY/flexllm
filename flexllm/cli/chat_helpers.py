"""Chat 和 Agent 相关的辅助函数"""

from __future__ import annotations

import asyncio
import json
import sys

from .utils import apply_user_template

# ========== Agent 工具定义 ==========

BUILTIN_TOOLS = {
    "shell": {
        "description": "Execute a shell command and return its output.",
        "prefix": "",
    },
    "dtflow": {
        "description": "数据文件处理(JSONL/CSV/Parquet): stats/sample/head/tail/clean/dedupe/transform/validate/concat/diff",
        "prefix": "dt",
    },
    "maque": {
        "description": "ML工具: 文本嵌入(embed)/向量检索/聚类分析/模型部署(serve)/多模态批处理(mllm)",
        "prefix": "maque",
    },
    "flexllm": {
        "description": "LLM API: ask/batch/chat/pricing/credits/models/mock/serve",
        "prefix": "flexllm",
    },
}


def _cli_tool_executor(name: str, arguments: str) -> str:
    """通用 CLI 工具执行器，根据注册表的 prefix 拼命令执行"""
    import subprocess

    tool = BUILTIN_TOOLS.get(name)
    if not tool:
        return f"[error: unknown tool '{name}']"

    args = json.loads(arguments)
    command = args.get("command", "")
    prefix = tool["prefix"]
    full_command = f"{prefix} {command}".strip() if prefix else command
    print(f"  [tool] {name}: {full_command}", flush=True)
    try:
        result = subprocess.run(
            full_command, shell=True, capture_output=True, text=True, timeout=60
        )
        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "[error: command timed out after 60s]"
    except Exception as e:
        return f"[error: {e}]"


def get_builtin_tools(tools_str: str):
    """获取内置工具定义和执行器，支持逗号分隔多工具（如 'shell,dtflow'）"""
    names = [t.strip() for t in tools_str.split(",") if t.strip()]
    available = ", ".join(BUILTIN_TOOLS)
    for name in names:
        if name not in BUILTIN_TOOLS:
            raise ValueError(f"未知的内置工具: {name}，可用: {available}")

    tool_defs = []
    for name in names:
        tool = BUILTIN_TOOLS[name]
        desc = tool["description"]
        if tool["prefix"]:
            desc += f" (命令会自动添加 '{tool['prefix']}' 前缀)"
        tool_defs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to execute",
                            },
                        },
                        "required": ["command"],
                    },
                },
            }
        )
    return tool_defs, _cli_tool_executor


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


def agent_chat(model, base_url, api_key, system_prompt, model_params, tools_name, message=None):
    """Agent 模式的交互式对话"""

    async def _run():
        from flexllm import AgentClient, LLMClient

        tool_defs, tool_executor = get_builtin_tools(tools_name)

        async with LLMClient(model=model, base_url=base_url, api_key=api_key) as client:
            agent = AgentClient(
                client=client,
                system=system_prompt,
                tools=tool_defs,
                tool_executor=tool_executor,
            )

            if message:
                print("Agent 运行中...", flush=True)
                result = await agent.run(message, **model_params)
                print(f"Assistant: {result.content}")
                if result.tool_calls:
                    print(f"  (调用了 {len(result.tool_calls)} 次工具，{result.rounds} 轮)")
                return

            print("\nAgent 对话模式")
            print(f"模型: {model}")
            print(f"工具: {tools_name}")
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

                    result = await agent.chat(user_input, **model_params)
                    print(f"Assistant: {result.content}")
                    if result.tool_calls:
                        print(f"  (调用了 {len(result.tool_calls)} 次工具，{result.rounds} 轮)")

                except EOFError:
                    print("\n再见！")
                    break

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\n再见！")


def agent_run(
    message, model, base_url, api_key, system_prompt, model_params, tools_name, max_rounds
):
    """Agent 非交互式执行"""

    async def _run():
        from flexllm import AgentClient, LLMClient

        tool_defs, tool_executor = get_builtin_tools(tools_name)

        async with LLMClient(model=model, base_url=base_url, api_key=api_key) as client:
            agent = AgentClient(
                client=client,
                system=system_prompt,
                tools=tool_defs,
                tool_executor=tool_executor,
                max_rounds=max_rounds,
            )

            result = await agent.run(message, **model_params)
            print(result.content)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\n[中断]")
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
