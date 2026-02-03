"""Chat 和 Agent 相关的辅助函数"""

from __future__ import annotations

import asyncio
import json
import sys

from .utils import apply_user_template

# ========== Agent 工具定义 ==========

# 旧版 CLI 工具（保留兼容性）
LEGACY_CLI_TOOLS = {
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


def _legacy_cli_executor(name: str, arguments: str) -> str:
    """旧版 CLI 工具执行器（兼容性保留）"""
    import subprocess

    tool = LEGACY_CLI_TOOLS.get(name)
    if not tool:
        return f"[error: unknown tool '{name}']"

    args = json.loads(arguments)
    command = args.get("command", "")
    prefix = tool["prefix"]
    full_command = f"{prefix} {command}".strip() if prefix else command
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
    """获取内置工具定义和执行器

    支持两类工具：
    - 新版细粒度工具：read, write, edit, glob, grep, bash
    - 旧版 CLI 工具：shell, dtflow, maque, flexllm

    特殊值：
    - "all": 所有新版工具（read,write,edit,glob,grep,bash）
    - "code": 代码操作工具（read,edit,glob,grep,bash）
    """
    from flexllm.agent.tools import TOOL_REGISTRY, get_tool_defs, make_tool_executor

    names = [t.strip() for t in tools_str.split(",") if t.strip()]

    # 处理特殊值
    expanded_names = []
    for name in names:
        if name == "all":
            expanded_names.extend(TOOL_REGISTRY.keys())
        elif name == "code":
            expanded_names.extend(["read", "edit", "glob", "grep", "bash"])
        else:
            expanded_names.append(name)
    names = list(dict.fromkeys(expanded_names))  # 去重保序

    # 分离新版和旧版工具
    new_tools = [n for n in names if n in TOOL_REGISTRY]
    legacy_tools = [n for n in names if n in LEGACY_CLI_TOOLS and n not in TOOL_REGISTRY]

    # 检查未知工具
    all_known = set(TOOL_REGISTRY.keys()) | set(LEGACY_CLI_TOOLS.keys())
    unknown = [n for n in names if n not in all_known]
    if unknown:
        available = ", ".join(sorted(all_known))
        raise ValueError(f"未知的工具: {', '.join(unknown)}，可用: {available}")

    # 获取工具定义
    tool_defs = []
    if new_tools:
        tool_defs.extend(get_tool_defs(new_tools))
    for name in legacy_tools:
        tool = LEGACY_CLI_TOOLS[name]
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

    # 创建组合执行器
    new_executor = make_tool_executor(new_tools) if new_tools else None

    def combined_executor(name: str, arguments: str) -> str:
        if name in TOOL_REGISTRY and new_executor:
            return new_executor(name, arguments)
        elif name in LEGACY_CLI_TOOLS:
            return _legacy_cli_executor(name, arguments)
        else:
            return f"[error: unknown tool '{name}']"

    return tool_defs, combined_executor


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
):
    """Agent 模式的交互式对话"""

    async def _run():
        from flexllm import AgentClient, LLMClient

        tool_defs, tool_executor = get_builtin_tools(tools_name)

        # verbose 模式下的回调
        def on_tool_call(name, arguments):
            if verbose:
                print(f"\n  🔧 Tool: {name}")
                try:
                    args = json.loads(arguments) if arguments else {}
                    for k, v in args.items():
                        v_str = str(v)
                        if len(v_str) > 100:
                            v_str = v_str[:100] + "..."
                        print(f"     {k}: {v_str}")
                except json.JSONDecodeError:
                    print(f"     (raw): {arguments[:100]}...")
            else:
                print(f"  [tool] {name}", flush=True)

        def on_tool_result(name, result):
            if verbose:
                result_str = str(result)
                if len(result_str) > 300:
                    print(f"  📤 Result: {result_str[:300]}... ({len(result_str)} chars)")
                else:
                    lines = result_str.split("\n")[:10]
                    print(f"  📤 Result:")
                    for line in lines:
                        print(f"     {line}")

        def on_llm_response(response):
            if verbose:
                if response.tool_calls:
                    print(f"\n  📋 Tool Calls ({len(response.tool_calls)}):")
                    for i, tc in enumerate(response.tool_calls, 1):
                        fn = tc.function
                        name = fn.get("name", "?")
                        args = fn.get("arguments", "{}")
                        try:
                            parsed = json.loads(args) if args else {}
                            args_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in parsed.items())
                        except json.JSONDecodeError:
                            args_str = args[:50] + "..."
                        print(f"     [{i}] {name}({args_str})")
                if response.usage:
                    u = response.usage
                    tokens = (
                        f"in:{u.get('prompt_tokens', '?')} out:{u.get('completion_tokens', '?')}"
                    )
                    print(f"  📊 Tokens: {tokens}")

        async with LLMClient(model=model, base_url=base_url, api_key=api_key) as client:
            agent = AgentClient(
                client=client,
                system=system_prompt,
                tools=tool_defs,
                tool_executor=tool_executor,
                max_rounds=max_rounds,
            )

            # 设置回调
            agent.on_tool_call = on_tool_call
            agent.on_tool_result = on_tool_result
            agent.on_llm_response = on_llm_response

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
            if verbose:
                print("模式: verbose")
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
):
    """Agent 非交互式执行

    Args:
        validate: 验证器名称，如 "python" 或 "python,pytest"，None 表示不验证
        max_fix_attempts: 验证失败时最大修复尝试次数
    """

    async def _run():
        from flexllm import AgentClient, LLMClient

        tool_defs, tool_executor = get_builtin_tools(tools_name)

        # 解析验证器
        validators = _parse_validators(validate) if validate else None

        # verbose 模式下的回调
        round_counter = [0]  # 用列表来在闭包中修改

        def on_tool_call(name, arguments):
            if verbose:
                print(f"\n{'─' * 60}")
                print(f"🔧 Tool Call: {name}")
                print(f"{'─' * 60}")
                try:
                    args = json.loads(arguments) if arguments else {}
                    for k, v in args.items():
                        v_str = str(v)
                        if len(v_str) > 200:
                            v_str = v_str[:200] + "..."
                        print(f"  {k}: {v_str}")
                except json.JSONDecodeError:
                    print(f"  (raw): {arguments[:200]}...")
            else:
                print(f"  [tool] {name}", flush=True)

        def on_tool_result(name, result):
            if verbose:
                print(f"\n📤 Tool Result ({name}):")
                result_str = str(result)
                if len(result_str) > 500:
                    print(f"  {result_str[:500]}...")
                    print(f"  ... ({len(result_str)} chars total)")
                else:
                    for line in result_str.split("\n")[:20]:
                        print(f"  {line}")
                    if result_str.count("\n") > 20:
                        print(f"  ... ({result_str.count(chr(10))} lines total)")

        def on_llm_response(response):
            if verbose:
                round_counter[0] += 1
                print(f"\n{'═' * 60}")
                print(f"🤖 LLM Response (Round {round_counter[0]})")
                print(f"{'═' * 60}")
                if response.content:
                    content = response.content
                    if len(content) > 500:
                        print(f"{content[:500]}...")
                    else:
                        print(content)
                if response.tool_calls:
                    print(f"\n📋 Tool Calls ({len(response.tool_calls)}):")
                    for i, tc in enumerate(response.tool_calls, 1):
                        fn = tc.function
                        name = fn.get("name", "?")
                        args = fn.get("arguments", "{}")
                        print(f"  [{i}] {name}")
                        try:
                            parsed = json.loads(args) if args else {}
                            for k, v in parsed.items():
                                v_str = str(v)
                                if len(v_str) > 80:
                                    v_str = v_str[:80] + "..."
                                print(f"      {k}: {v_str}")
                        except json.JSONDecodeError:
                            print(f"      (raw): {args[:80]}...")
                if response.usage:
                    u = response.usage
                    tokens = (
                        f"in:{u.get('prompt_tokens', '?')} out:{u.get('completion_tokens', '?')}"
                    )
                    print(f"  → Tokens: {tokens}")

        async with LLMClient(model=model, base_url=base_url, api_key=api_key) as client:
            agent = AgentClient(
                client=client,
                system=system_prompt,
                tools=tool_defs,
                tool_executor=tool_executor,
                max_rounds=max_rounds,
            )

            # 验证回调
            def on_validation(files, results):
                print(f"\n{'─' * 60}")
                print(f"🔍 Validation")
                print(f"{'─' * 60}")
                print(f"Files: {files}")
                for r in results:
                    status = "✅" if r.success else "❌"
                    print(f"  {status} [{r.validator_name}]", end="")
                    if not r.success:
                        print(f" - {len(r.errors)} error(s)")
                        for e in r.errors[:3]:
                            print(f"      {e}")
                        if len(r.errors) > 3:
                            print(f"      ... and {len(r.errors) - 3} more")
                    else:
                        print()

            # 设置回调
            agent.on_tool_call = on_tool_call
            agent.on_tool_result = on_tool_result
            agent.on_llm_response = on_llm_response
            if validators:
                agent.on_validation = on_validation

            if verbose:
                print(f"{'═' * 60}")
                print(f"🚀 Agent Start")
                print(f"{'═' * 60}")
                print(f"Model: {model}")
                print(f"Tools: {tools_name}")
                if validators:
                    print(f"Validators: {[v.name for v in validators]}")
                print(f"Task: {message[:100]}{'...' if len(message) > 100 else ''}")

            if validators:
                result = await agent.run_with_validation(
                    message,
                    validators=validators,
                    max_fix_attempts=max_fix_attempts,
                    **model_params,
                )
            else:
                result = await agent.run(message, **model_params)

            if verbose:
                print(f"\n{'═' * 60}")
                print(f"✅ Agent Complete")
                print(f"{'═' * 60}")
                print(f"Rounds: {result.rounds}")
                print(f"Tool calls: {len(result.tool_calls)}")
                if result.usage:
                    total = result.usage.get("total_tokens", "?")
                    print(f"Total tokens: {total}")
                print(f"\n{'─' * 60}")
                print("Final Response:")
                print(f"{'─' * 60}")

            print(result.content)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\n[中断]")
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
