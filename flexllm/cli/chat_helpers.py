"""Chat 和 Agent 相关的辅助函数"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from .utils import apply_user_template

AGENT_DEFAULT_SYSTEM = f"""You are an autonomous agent at {os.getcwd()}.

Loop: think briefly → use tools → observe results → continue until done.

Rules:
- Prefer tools over prose. Act, don't just explain.
- If a tool call fails, analyze the error and try alternative approaches.
- Never give up after a single failure — try different methods, URLs, or commands.
- After finishing, summarize what you did and the results."""

SKILLS_DIRS = [
    Path("~/.flexllm/skills").expanduser(),  # flexllm 全局
    Path(".flexllm/skills"),  # flexllm 项目级
    Path(".claude/skills"),  # Claude Code 项目级
    Path("~/.claude/skills").expanduser(),  # Claude Code 全局
]


def load_project_instructions() -> str | None:
    """从当前目录向上搜索项目指令文件，返回内容。

    搜索顺序（每层目录）：.flexllm.md 优先，fallback 到 CLAUDE.md
    从 cwd → parent → ... → / 找到第一个即返回。
    """
    current = Path.cwd()
    for parent in [current, *current.parents]:
        for filename in (".flexllm.md", "CLAUDE.md"):
            candidate = parent / filename
            if candidate.is_file():
                try:
                    return candidate.read_text(encoding="utf-8").strip()
                except Exception:
                    return None
    return None


def _parse_skill_frontmatter(content: str) -> tuple[dict, str]:
    """解析 SKILL.md 的 frontmatter 和正文。

    格式（与 Claude Code 一致）：
        ---
        name: code-reviewer
        description: 代码审核
        allowed-tools: Bash(git add:*), Bash(git commit:*)
        ---

        正文内容...

    Returns:
        (metadata_dict, body_str)
    """
    if not content.startswith("---"):
        return {}, content

    end = content.find("---", 3)
    if end == -1:
        return {}, content

    frontmatter_str = content[3:end].strip()
    body = content[end + 3 :].strip()

    # 简单解析 YAML frontmatter（key: value 格式）
    metadata = {}
    current_key = None
    current_value_lines = []

    for line in frontmatter_str.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if ":" in stripped and not stripped.startswith(" ") and not stripped.startswith("-"):
            # 保存上一个 key
            if current_key:
                metadata[current_key] = " ".join(current_value_lines).strip()
            key, _, value = stripped.partition(":")
            current_key = key.strip()
            current_value_lines = [value.strip()]
        elif current_key:
            current_value_lines.append(stripped)

    if current_key:
        metadata[current_key] = " ".join(current_value_lines).strip()

    return metadata, body


def load_skill(skill_name: str) -> dict | None:
    """从多个 skills 目录加载 skill。

    搜索路径（按优先级）：SKILLS_DIRS 中的每个目录
    每个目录支持两种结构：
    1. {dir}/{name}/SKILL.md  (Claude Code 风格，推荐)
    2. {dir}/{name}.md        (简单模式)

    Returns:
        {"name": ..., "description": ..., "content": ..., "metadata": {...},
         "allowed_tools": list[str] | None, "model": str | None} 或 None
    """
    skill_file = None
    for skills_dir in SKILLS_DIRS:
        # 目录模式
        candidate = skills_dir / skill_name / "SKILL.md"
        if candidate.is_file():
            skill_file = candidate
            break
        # 扁平模式
        candidate = skills_dir / f"{skill_name}.md"
        if candidate.is_file():
            skill_file = candidate
            break

    if not skill_file:
        return None

    try:
        raw = skill_file.read_text(encoding="utf-8").strip()
    except Exception:
        return None

    metadata, body = _parse_skill_frontmatter(raw)

    # 解析 allowed-tools（如 "Read, Grep, Bash(git *)" → ["read", "grep", "bash"]）
    allowed_tools = None
    allowed_tools_str = metadata.get("allowed-tools") or metadata.get("allowed_tools")
    if allowed_tools_str:
        # 按逗号分割，去除工具名后的括号参数，转小写
        import re

        allowed_tools = []
        for part in allowed_tools_str.split(","):
            tool_name = re.sub(r"\(.*\)", "", part).strip().lower()
            if tool_name and tool_name not in allowed_tools:
                allowed_tools.append(tool_name)

    return {
        "name": metadata.get("name", skill_name),
        "description": metadata.get("description", ""),
        "content": body,
        "metadata": metadata,
        "allowed_tools": allowed_tools,
        "model": metadata.get("model"),
    }


def list_skills() -> list[str]:
    """列出所有可用的 skill 名称（去重，按优先级保留第一个）"""
    names = set()
    for skills_dir in SKILLS_DIRS:
        if not skills_dir.is_dir():
            continue
        # 目录模式：skills/{name}/SKILL.md
        for d in skills_dir.iterdir():
            if d.is_dir() and (d / "SKILL.md").is_file():
                names.add(d.name)
        # 扁平模式：skills/{name}.md
        for f in skills_dir.glob("*.md"):
            names.add(f.stem)
    return sorted(names)


def build_agent_system(system_prompt: str | None, skill: str | None = None) -> dict:
    """构建 agent 的最终 system prompt。

    优先级叠加：
    1. 基础 system (用户指定 or 默认)
    2. .flexllm.md / CLAUDE.md 项目指令
    3. skill 模板（如果指定）

    Returns:
        {"system": str, "allowed_tools": list[str] | None, "model": str | None}
    """
    parts = []

    # 基础 system
    base = system_prompt or AGENT_DEFAULT_SYSTEM
    parts.append(base)

    # 项目指令
    project_instructions = load_project_instructions()
    if project_instructions:
        parts.append(f"# Project Instructions\n\n{project_instructions}")

    allowed_tools = None
    skill_model = None

    # Skill
    if skill:
        skill_data = load_skill(skill)
        if skill_data:
            parts.append(f"# Skill: {skill_data['name']}\n\n{skill_data['content']}")
            allowed_tools = skill_data.get("allowed_tools")
            skill_model = skill_data.get("model")
        else:
            available = list_skills()
            hint = f"，可用: {', '.join(available)}" if available else "（~/.flexllm/skills/ 为空）"
            raise ValueError(f"未知的 skill: {skill}{hint}")

    return {
        "system": "\n\n".join(parts),
        "allowed_tools": allowed_tools,
        "model": skill_model,
    }


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
    stream=True,
    skill=None,
):
    """Agent 模式的交互式对话"""

    async def _run():
        from flexllm import AgentClient, LLMClient

        from .agent_console import AgentConsole

        agent_system = build_agent_system(system_prompt, skill=skill)
        effective_system = agent_system["system"]
        # skill 的 allowed_tools 覆盖 --tools 参数
        effective_tools = (
            ",".join(agent_system["allowed_tools"])
            if agent_system.get("allowed_tools")
            else tools_name
        )
        registry = _build_registry(effective_tools, mcp_servers)
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
                if stream:
                    agent.on_llm_token = ui.on_llm_token

                if message:
                    ui.begin()
                    result = await agent.run(message, stream=stream, **model_params)
                    ui.end()
                    ui.print_summary(result)
                    if not stream:
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
                        result = await agent.chat(user_input, stream=stream, **model_params)
                        ui.end()
                        ui.print_summary(result)
                        if not stream:
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


def _merge_mcp_servers(cli_mcp: list[str] | None, config_mcp: list[dict] | None) -> list:
    """合并 CLI 和配置文件的 MCP servers。

    CLI 参数为字符串列表（命令或 URL），配置文件为 dict 列表。
    返回统一的 spec 列表，每项为 str 或 dict。
    """
    result = []
    if config_mcp:
        result.extend(config_mcp)
    if cli_mcp:
        result.extend(cli_mcp)
    return result


async def _connect_mcp_servers(mcp_servers):
    """连接 MCP servers，返回连接列表。

    支持格式:
    - str: 命令或 URL（来自 CLI --mcp）
    - dict (stdio): {"command": "npx", "args": ["-y", "@mcp/xxx"], "env": {...}, "name": "..."}
      也支持简写: {"command": "npx -y @mcp/xxx"}（command 为完整命令字符串）
    - dict (http/sse): {"url": "http://...", "type": "http", "headers": {...}, "name": "..."}
    """
    from flexllm.agent.mcp import MCPConnection

    conns = []
    for server_spec in mcp_servers:
        if isinstance(server_spec, dict):
            command = server_spec.get("command")
            # 支持 command + args 分开格式（Claude Code 风格）
            if command and "args" in server_spec:
                args = server_spec["args"]
                if isinstance(args, list):
                    command = [command] + args
            conn = MCPConnection(
                command=command,
                url=server_spec.get("url"),
                env=server_spec.get("env"),
                name=server_spec.get("name"),
                headers=server_spec.get("headers"),
                cwd=server_spec.get("cwd"),
            )
        elif server_spec.startswith("http://") or server_spec.startswith("https://"):
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
    stream=True,
    skill=None,
    enable_tasks=False,
    enable_todo=False,
):
    """Agent 非交互式执行

    Args:
        validate: 验证器名称，如 "python" 或 "python,pytest"，None 表示不验证
        max_fix_attempts: 验证失败时最大修复尝试次数
        mcp_servers: MCP server 命令或 URL 列表
        skill: skill 名称
        enable_tasks: 启用任务系统
        enable_todo: 启用 todo 追踪
    """

    async def _run():
        from flexllm import AgentClient, LLMClient

        from .agent_console import AgentConsole

        agent_system = build_agent_system(system_prompt, skill=skill)
        effective_system = agent_system["system"]
        effective_tools = (
            ",".join(agent_system["allowed_tools"])
            if agent_system.get("allowed_tools")
            else tools_name
        )
        registry = _build_registry(effective_tools)
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
                    enable_tasks=enable_tasks,
                    enable_todo=enable_todo,
                )

                if approve == "manual":
                    from flexllm.agent.client import console_approval

                    agent.approval_handler = console_approval

                agent.on_tool_call = ui.on_tool_call
                agent.on_tool_result = ui.on_tool_result
                agent.on_llm_response = ui.on_llm_response
                if stream:
                    agent.on_llm_token = ui.on_llm_token
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
                    result = await agent.run(message, stream=stream, **model_params)

                ui.end()
                ui.print_summary(result)
                if not stream:
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
