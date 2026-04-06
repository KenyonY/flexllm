"""CLI 命令注册"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated

from .chat_helpers import interactive_chat, single_chat
from .config import get_config
from .errors import ErrorType, cli_error, dry_run_output
from .utils import (
    apply_user_template,
    convert_to_messages,
    extract_code_block,
    parse_batch_input,
    parse_schema,
    parse_thinking,
    query_credits,
    query_credits_by_key,
    read_file_contents,
    resolve_model_config,
)


def register_commands(app):
    """注册所有 CLI 命令到 typer app"""
    import typer
    from typer import Argument, Option

    @app.command()
    def ask(
        prompt: Annotated[str | None, Argument(help="用户问题")] = None,
        system: Annotated[str | None, Option("-s", "--system", help="系统提示词")] = None,
        model: Annotated[str | None, Option("-m", "--model", help="模型名称")] = None,
        base_url: Annotated[str | None, Option("--base-url", help="API 地址")] = None,
        api_key: Annotated[str | None, Option("--api-key", help="API 密钥")] = None,
        user_template: Annotated[
            str | None, Option("--user-template", help="user content 模板 (使用 {content} 占位符)")
        ] = None,
        thinking: Annotated[
            str | None,
            Option(
                "--thinking", help="思考模式 (true/false/low/medium/high 或 budget_tokens 数值)"
            ),
        ] = None,
        schema: Annotated[
            str | None,
            Option(
                "--schema",
                help="结构化输出 (json=JSON模式, @file.json=从文件读取, 或 JSON Schema 字符串)",
            ),
        ] = None,
        extract: Annotated[
            bool, Option("-x", "--extract", help="从回复中提取第一个代码块")
        ] = False,
        files: Annotated[
            list[str] | None, Option("-f", "--file", help="附加文件内容到 prompt（可多次指定）")
        ] = None,
        dry_run: Annotated[bool, Option("--dry-run", help="预览操作内容，不实际执行")] = False,
    ):
        """LLM 快速问答（支持管道输入）

        \b
        基本用法:
          flexllm ask "什么是Python"
          flexllm ask "解释代码" -s "你是代码专家"
          echo "长文本" | flexllm ask "总结一下"

        附加文件 (-f):  读取文件内容拼接到 prompt 前面
          flexllm ask -f main.py "这段代码有什么问题？"
          flexllm ask -f a.py -f b.py "对比这两个文件的实现"

        结构化输出 (--schema):  强制模型返回 JSON
          flexllm ask "列出3种编程语言及其特点" --schema json
          flexllm ask "提取姓名和年龄" --schema @schema.json

        代码提取 (-x):  只输出回复中的第一个代码块
          flexllm ask "用 Python 写个快排" -x
          flexllm ask "用 Python 写个快排" -x > sort.py

        预览:
          flexllm ask "测试" --dry-run              # 预览请求内容
        """
        stdin_content = None
        if not sys.stdin.isatty():
            stdin_content = sys.stdin.read().strip()

        file_content = read_file_contents(files) if files else None

        if not prompt and not stdin_content and not file_content:
            cli_error(
                ErrorType.INVALID_ARGS,
                "未提供问题",
                context={
                    "arg": "prompt",
                    "stdin_tty": sys.stdin.isatty(),
                    "files": files or [],
                },
                suggestion='提供位置参数、通过 -f 附加文件，或通过管道传入: flexllm ask "你的问题"',
                doc="flexllm ask --help",
            )

        parts = [p for p in [file_content, stdin_content, prompt] if p]
        full_prompt = "\n\n".join(parts)

        model_id, base_url, api_key = resolve_model_config(
            model, base_url=base_url, api_key=api_key, required=True
        )

        config = get_config()
        if not system:
            system = config.get_system(model)
        if not user_template:
            user_template = config.get_user_template(model)

        model_params = config.get_model_params(model)

        thinking_value = parse_thinking(thinking)
        if thinking_value is not None:
            model_params["thinking"] = thinking_value

        response_format = parse_schema(schema)
        if response_format is not None:
            model_params["response_format"] = response_format

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        user_content = apply_user_template(full_prompt, user_template)
        messages.append({"role": "user", "content": user_content})

        if dry_run:
            dry_run_output(
                {
                    "action": "ask",
                    "model": model_id,
                    "base_url": base_url,
                    "messages": messages,
                    "params": model_params,
                }
            )

        async def _ask():
            from flexllm import LLMClient

            async with LLMClient(model=model_id, base_url=base_url, api_key=api_key) as client:
                return await client.chat_completions(messages, **model_params)

        try:
            result = asyncio.run(_ask())
            if result is None:
                cli_error(
                    ErrorType.GENERAL,
                    "模型返回空结果",
                    context={"model": model_id, "base_url": base_url},
                    suggestion="使用 flexllm test 验证连接，或加 --dry-run 检查请求",
                    doc="flexllm ask --help",
                    retryable=True,
                )
            if hasattr(result, "status") and result.status == "error":
                error_msg = result.data.get("detail", result.data.get("error", "未知错误"))
                cli_error(
                    ErrorType.NETWORK_ERROR,
                    f"LLM 调用失败: {error_msg}",
                    context={
                        "model": model_id,
                        "base_url": base_url,
                        "response_data": result.data,
                    },
                    suggestion="检查 API Key 和 base_url，或运行 flexllm test",
                    doc="flexllm ask --help",
                    retryable=True,
                )
            output = str(result) if not isinstance(result, str) else result
            if extract:
                code = extract_code_block(output)
                if code is not None:
                    print(code)
                else:
                    print("提示: 回复中未找到代码块，输出原始内容", file=sys.stderr)
                    print(output)
            else:
                print(output)
        except typer.Exit:
            raise
        except Exception as e:
            cli_error(
                ErrorType.GENERAL,
                str(e),
                context={
                    "exception_type": type(e).__name__,
                    "model": model_id,
                    "base_url": base_url,
                },
                doc="flexllm ask --help",
            )

    @app.command()
    def chat(
        message: Annotated[str | None, Argument(help="单条消息（不提供则进入多轮对话）")] = None,
        model: Annotated[str | None, Option("-m", "--model", help="模型名称")] = None,
        base_url: Annotated[str | None, Option("--base-url", help="API 地址")] = None,
        api_key: Annotated[str | None, Option("--api-key", help="API 密钥")] = None,
        system_prompt: Annotated[str | None, Option("-s", "--system", help="系统提示词")] = None,
        temperature: Annotated[float | None, Option("-t", "--temperature", help="采样温度")] = None,
        max_tokens: Annotated[int | None, Option("--max-tokens", help="最大生成 token 数")] = None,
        no_stream: Annotated[bool, Option("--no-stream", help="禁用流式输出")] = False,
        user_template: Annotated[
            str | None, Option("--user-template", help="user content 模板 (使用 {content} 占位符)")
        ] = None,
        thinking: Annotated[
            str | None,
            Option(
                "--thinking", help="思考模式 (true/false/low/medium/high 或 budget_tokens 数值)"
            ),
        ] = None,
        schema: Annotated[
            str | None,
            Option(
                "--schema",
                help="结构化输出 (json=JSON模式, @file.json=从文件读取, 或 JSON Schema 字符串)",
            ),
        ] = None,
        extract: Annotated[
            bool, Option("-x", "--extract", help="从回复中提取第一个代码块")
        ] = False,
        files: Annotated[
            list[str] | None, Option("-f", "--file", help="附加文件内容到 prompt（可多次指定）")
        ] = None,
        dry_run: Annotated[bool, Option("--dry-run", help="预览操作内容，不实际执行")] = False,
    ):
        """交互式对话

        \b
        基本用法:
          flexllm chat                        # 多轮对话
          flexllm chat "你好"                 # 单条对话
          flexllm chat -m gpt-4 "你好"        # 指定模型

        附加文件 (-f):  读取文件内容作为对话上下文
          flexllm chat -f code.py "这段代码有什么问题？"
          flexllm chat -f a.py -f b.py "对比这两个文件"

        代码提取 (-x):  只输出回复中的代码块（仅单条模式）
          flexllm chat "写个 hello world" -x

        预览:
          flexllm chat "测试" --dry-run             # 预览请求配置
        """
        model, base_url, api_key = resolve_model_config(model, base_url, api_key)
        config = get_config()

        if not base_url:
            cli_error(
                ErrorType.NOT_FOUND,
                "未配置 base_url",
                context={"model": model},
                suggestion="设置环境变量 FLEXLLM_BASE_URL，或运行 flexllm init 创建配置文件",
                doc="flexllm chat --help",
            )

        if not system_prompt:
            system_prompt = config.get_system(model)
        if not user_template:
            user_template = config.get_user_template(model)

        model_params = config.get_model_params(model)
        if temperature is not None:
            model_params["temperature"] = temperature
        if max_tokens is not None:
            model_params["max_tokens"] = max_tokens
        model_params.setdefault("temperature", 0.7)
        model_params.setdefault("max_tokens", 2048)

        thinking_value = parse_thinking(thinking)
        if thinking_value is not None:
            model_params["thinking"] = thinking_value

        response_format = parse_schema(schema)
        if response_format is not None:
            model_params["response_format"] = response_format

        resolved_thinking = model_params.pop("thinking", None)

        stream = not no_stream

        # 如果提供了文件，拼接到 message 前面
        if files and message:
            file_content = read_file_contents(files)
            message = f"{file_content}\n\n{message}"
        elif files and not message:
            message = read_file_contents(files)

        if dry_run:
            mode = "single" if message else "interactive"
            data = {
                "action": "chat",
                "mode": mode,
                "model": model,
                "base_url": base_url,
                "system": system_prompt,
                "params": model_params,
            }
            if message:
                data["message"] = message
            dry_run_output(data)

        if message:
            single_chat(
                message,
                model,
                base_url,
                api_key,
                system_prompt,
                model_params["temperature"],
                model_params["max_tokens"],
                stream,
                user_template,
                thinking=resolved_thinking,
                extract=extract,
            )
        elif not sys.stdin.isatty():
            cli_error(
                ErrorType.INVALID_ARGS,
                "非 TTY 模式下必须提供 message 参数",
                context={"stdin_tty": False, "message": None},
                suggestion='交互模式仅支持 TTY。非 TTY 下请提供 message: flexllm chat "你的问题"',
                doc="flexllm chat --help",
            )
        else:
            interactive_chat(
                model,
                base_url,
                api_key,
                system_prompt,
                model_params["temperature"],
                model_params["max_tokens"],
                stream,
                user_template,
                thinking=resolved_thinking,
            )

    @app.command(name="chat-web")
    def chat_web(
        model: Annotated[str | None, Option("-m", "--model", help="模型名称")] = None,
        base_url: Annotated[str | None, Option("--base-url", help="API 地址")] = None,
        api_key: Annotated[str | None, Option("--api-key", help="API 密钥")] = None,
        system_prompt: Annotated[str | None, Option("-s", "--system", help="系统提示词")] = None,
        temperature: Annotated[float | None, Option("-t", "--temperature", help="采样温度")] = None,
        max_tokens: Annotated[int | None, Option("--max-tokens", help="最大生成 token 数")] = None,
        user_template: Annotated[
            str | None, Option("--user-template", help="user content 模板 (使用 {content} 占位符)")
        ] = None,
        port: Annotated[int, Option("-p", "--port", help="Web 服务端口")] = 8080,
        host: Annotated[str, Option("--host", help="监听地址")] = "localhost",
        thinking: Annotated[
            str | None,
            Option(
                "--thinking", help="启用思考模式 (true/false/low/medium/high 或 budget_tokens 数值)"
            ),
        ] = None,
        title: Annotated[str, Option("--title", help="页面 Logo 文本")] = "flexllm",
        multi_turn: Annotated[
            bool, Option("--multi-turn", help="多轮对话模式（携带上下文）")
        ] = False,
        dry_run: Annotated[bool, Option("--dry-run", help="预览操作内容，不实际执行")] = False,
    ):
        """启动 Web 聊天界面

        \b
        Examples:
        flexllm chat-web                      # 使用默认模型（单轮对话）
        flexllm chat-web --multi-turn         # 多轮对话模式
        flexllm chat-web -m gpt-4             # 指定模型
        flexllm chat-web -p 9090              # 指定端口
        flexllm chat-web --host 0.0.0.0       # 允许外部访问
        flexllm chat-web --thinking true      # 启用思考模式
        flexllm chat-web --dry-run            # 预览启动配置
        """
        model, base_url, api_key = resolve_model_config(model, base_url, api_key)
        config = get_config()

        if not base_url:
            cli_error(
                ErrorType.NOT_FOUND,
                "未配置 base_url",
                context={"model": model},
                suggestion="设置环境变量 FLEXLLM_BASE_URL，或运行 flexllm init 创建配置文件",
                doc="flexllm chat-web --help",
            )

        if not system_prompt:
            system_prompt = config.get_system(model)
        if not user_template:
            user_template = config.get_user_template(model)

        model_params = config.get_model_params(model)
        if temperature is not None:
            model_params["temperature"] = temperature
        if max_tokens is not None:
            model_params["max_tokens"] = max_tokens
        model_params.setdefault("temperature", 0.7)
        model_params.setdefault("max_tokens", 2048)

        try:
            from ..chat_web import ChatWebConfig, ChatWebServer
        except ImportError:
            cli_error(
                ErrorType.DEPENDENCY_MISSING,
                "缺少依赖: aiohttp",
                context={"missing_package": "aiohttp", "feature": "chat-web"},
                suggestion="pip install aiohttp 或 pip install 'flexllm[all]'",
                doc="flexllm chat-web --help",
            )

        thinking_value = parse_thinking(thinking)
        if thinking_value is None:
            thinking_value = model_params.get("thinking")

        if dry_run:
            dry_run_output(
                {
                    "action": "chat_web",
                    "host": host,
                    "port": port,
                    "model": model,
                    "base_url": base_url,
                    "temperature": model_params.get("temperature"),
                    "max_tokens": model_params.get("max_tokens"),
                    "thinking": thinking_value,
                    "multi_turn": multi_turn,
                }
            )

        web_config = ChatWebConfig(
            port=port,
            host=host,
            model=model,
            base_url=base_url,
            api_key=api_key,
            system_prompt=system_prompt,
            temperature=model_params["temperature"],
            max_tokens=model_params["max_tokens"],
            user_template=user_template,
            thinking=thinking_value,
            multi_turn=multi_turn,
            title=title,
        )

        print(f"flexllm Chat Web starting on http://{host}:{port}")
        print(f"  Model: {model}")
        print(f"  Server: {base_url}")
        print(f"  Temperature: {model_params['temperature']}")
        if thinking_value is not None:
            print(f"  Thinking: {thinking_value}")
        print("\nPress Ctrl+C to stop")

        try:
            server = ChatWebServer(web_config)
            server.run()
        except KeyboardInterrupt:
            print("\nServer stopped")
        except Exception as e:
            cli_error(
                ErrorType.GENERAL,
                str(e),
                context={
                    "exception_type": type(e).__name__,
                    "host": host,
                    "port": port,
                    "model": model,
                },
                doc="flexllm chat-web --help",
            )

    @app.command()
    def serve(
        model: Annotated[str | None, Option("-m", "--model", help="模型名称")] = None,
        base_url: Annotated[str | None, Option("--base-url", help="API 地址")] = None,
        api_key: Annotated[str | None, Option("--api-key", help="API 密钥")] = None,
        system_prompt: Annotated[str | None, Option("-s", "--system", help="系统提示词")] = None,
        user_template: Annotated[
            str | None, Option("--user-template", help="user content 模板 (使用 {content} 占位符)")
        ] = None,
        temperature: Annotated[float | None, Option("-t", "--temperature", help="采样温度")] = None,
        max_tokens: Annotated[int | None, Option("--max-tokens", help="最大生成 token 数")] = None,
        thinking: Annotated[
            str | None,
            Option(
                "--thinking", help="启用思考模式 (true/false/low/medium/high 或 budget_tokens 数值)"
            ),
        ] = None,
        concurrency: Annotated[
            int, Option("-c", "--concurrency", help="上游 LLM 最大并发数")
        ] = 1000,
        max_qps: Annotated[float | None, Option("--max-qps", help="每秒最大请求数")] = None,
        timeout: Annotated[int, Option("--timeout", help="请求超时（秒）")] = 120,
        port: Annotated[int, Option("-p", "--port", help="监听端口")] = 8000,
        host: Annotated[str, Option("--host", help="监听地址")] = "0.0.0.0",
        verbose: Annotated[bool, Option("--verbose", "-v", help="打印请求日志")] = False,
        dry_run: Annotated[bool, Option("--dry-run", help="预览操作内容，不实际执行")] = False,
    ):
        """启动 HTTP API 服务，将 LLM 包装为 REST API

        适用于微调模型部署：固定 system prompt 和 user template，
        调用方只需发送 content 文本，返回解析后的 thinking 和 content。

        \b
        API 端点:
        POST /api/generate             非流式生成
        POST /api/generate/stream      流式生成 (SSE)
        POST /api/generate/batch       批量生成
        GET  /health                   健康检查
        GET  /api/config               查看当前配置

        \b
        Examples:
        flexllm serve -m qwen-finetuned -s "你是助手"
        flexllm serve --thinking true -c 20 -p 8000
        flexllm serve --dry-run                     # 预览启动配置
        """
        model, base_url, api_key = resolve_model_config(model, base_url, api_key)
        config = get_config()

        if not base_url:
            cli_error(
                ErrorType.NOT_FOUND,
                "未配置 base_url",
                context={"model": model},
                suggestion="设置环境变量 FLEXLLM_BASE_URL，或运行 flexllm init 创建配置文件",
                doc="flexllm serve --help",
            )

        if not system_prompt:
            system_prompt = config.get_system(model)
        if not user_template:
            user_template = config.get_user_template(model)

        model_params = config.get_model_params(model)
        if temperature is not None:
            model_params["temperature"] = temperature
        if max_tokens is not None:
            model_params["max_tokens"] = max_tokens

        try:
            from ..serve import ServeConfig, ServeServer
        except ImportError:
            cli_error(
                ErrorType.DEPENDENCY_MISSING,
                "缺少依赖: aiohttp",
                context={"missing_package": "aiohttp", "feature": "serve"},
                suggestion="pip install aiohttp 或 pip install 'flexllm[all]'",
                doc="flexllm serve --help",
            )

        thinking_value = parse_thinking(thinking)
        if thinking_value is None:
            thinking_value = model_params.get("thinking")

        if dry_run:
            dry_run_output(
                {
                    "action": "serve",
                    "host": host,
                    "port": port,
                    "model": model,
                    "base_url": base_url,
                    "system": system_prompt,
                    "concurrency": concurrency,
                    "timeout": timeout,
                    "endpoints": [
                        "POST /api/generate",
                        "POST /api/generate/stream",
                        "POST /api/generate/batch",
                        "GET /health",
                        "GET /api/config",
                    ],
                }
            )

        serve_config = ServeConfig(
            port=port,
            host=host,
            model=model,
            base_url=base_url,
            api_key=api_key,
            system_prompt=system_prompt,
            user_template=user_template,
            temperature=model_params.get("temperature"),
            max_tokens=model_params.get("max_tokens"),
            thinking=thinking_value,
            concurrency=concurrency,
            max_qps=max_qps,
            timeout=timeout,
            verbose=verbose,
        )

        effective_temperature = model_params.get("temperature")
        effective_max_tokens = model_params.get("max_tokens")

        print(f"flexllm Serve starting on http://{host}:{port}")
        print(f"  Model: {model}")
        print(f"  Server: {base_url}")
        if effective_temperature is not None:
            print(f"  Temperature: {effective_temperature}")
        if effective_max_tokens is not None:
            print(f"  Max tokens: {effective_max_tokens}")
        if thinking_value is not None:
            print(f"  Thinking: {thinking_value}")
        if system_prompt:
            display = system_prompt[:50] + "..." if len(system_prompt) > 50 else system_prompt
            print(f"  System: {display}")
        if user_template:
            print(f"  User template: {user_template}")
        print(f"  Concurrency: {concurrency}")
        if max_qps is not None:
            print(f"  Max QPS: {max_qps}")
        print(f"\n  POST /api/generate             非流式生成")
        print(f"  POST /api/generate/stream      流式生成")
        print(f"  POST /api/generate/batch       批量生成")
        print(f"  GET  /health                   健康检查")
        print(f"  GET  /api/config               查看配置")
        if verbose:
            print(f"  Verbose: on (请求日志已开启)")
        print("\nPress Ctrl+C to stop")

        try:
            server = ServeServer(serve_config)
            server.run()
        except KeyboardInterrupt:
            print("\nServer stopped")
        except Exception as e:
            cli_error(
                ErrorType.GENERAL,
                str(e),
                context={
                    "exception_type": type(e).__name__,
                    "host": host,
                    "port": port,
                    "model": model,
                },
                doc="flexllm serve --help",
            )

    @app.command()
    def batch(
        input: Annotated[str | None, Argument(help="输入文件路径（省略则从 stdin 读取）")] = None,
        output: Annotated[
            str | None, Option("-o", "--output", help="输出文件路径（可选，默认自动生成）")
        ] = None,
        model: Annotated[str | None, Option("-m", "--model", help="模型名称")] = None,
        concurrency: Annotated[int | None, Option("-c", "--concurrency", help="并发数")] = None,
        max_qps: Annotated[float | None, Option("--max-qps", help="每秒最大请求数")] = None,
        system: Annotated[str | None, Option("-s", "--system", help="全局 system prompt")] = None,
        temperature: Annotated[float | None, Option("-t", "--temperature", help="采样温度")] = None,
        max_tokens: Annotated[int | None, Option("--max-tokens", help="最大生成 token 数")] = None,
        cache: Annotated[
            bool | None, Option("--cache/--no-cache", help="启用/禁用响应缓存")
        ] = None,
        return_usage: Annotated[bool, Option("--return-usage", help="输出 token 统计")] = False,
        preprocess_msg: Annotated[bool, Option("--preprocess-msg", help="预处理图片消息")] = False,
        track_cost: Annotated[bool, Option("--track-cost", help="在进度条中显示实时成本")] = False,
        save_input: Annotated[
            str | None,
            Option(
                "--save-input",
                help="输出文件中 input 字段的保存策略: true(默认,完整保存), last(仅最后user内容), false(不保存)",
            ),
        ] = None,
        limit: Annotated[
            int | None,
            Option("-n", "--limit", help="只处理前 N 条记录（用于快速试跑）"),
        ] = None,
        user_field: Annotated[
            str | None,
            Option("--user-field", "-uf", help="指定 user content 的字段名（跳过自动格式检测）"),
        ] = None,
        system_field: Annotated[
            str | None,
            Option("--system-field", "-sf", help="指定 system prompt 的字段名（跳过自动格式检测）"),
        ] = None,
        user_template: Annotated[
            str | None,
            Option("--user-template", help="user content 模板 (使用 {content} 占位符)"),
        ] = None,
        schema: Annotated[
            str | None,
            Option(
                "--schema",
                help="结构化输出 (json=JSON模式, @file.json=从文件读取, 或 JSON Schema 字符串)",
            ),
        ] = None,
        dry_run: Annotated[bool, Option("--dry-run", help="预览操作内容，不实际执行")] = False,
    ):
        """批量处理 JSONL 文件（支持断点续传）

        自动检测输入格式：openai_chat, alpaca, simple (q/question/prompt/input/user)
        也可用 --user-field 和 --system-field 指定任意字段名。
        高级配置可在 ~/.flexllm/config.yaml 的 batch 节中设置，CLI 参数优先级更高。

        \b
        基本用法:
          flexllm batch input.jsonl                    # 自动生成 input.output.jsonl
          flexllm batch input.jsonl -o output.jsonl    # 指定输出文件
          flexllm batch input.jsonl -c 20 -m gpt-4     # 并发数 + 模型
          cat input.jsonl | flexllm batch -o out.jsonl  # stdin 输入（需指定 -o）

        结构化输出 (--schema):  所有记录统一使用结构化输出
          flexllm batch input.jsonl -o out.jsonl --schema json
          flexllm batch input.jsonl -o out.jsonl --schema @schema.json

        其他常用参数:
          flexllm batch input.jsonl --cache --return-usage --track-cost
          flexllm batch data.jsonl -o out.jsonl -uf text -sf sys_prompt
          flexllm batch input.jsonl -n 5               # 只处理前5条（试跑）

        预览:
          flexllm batch input.jsonl --dry-run       # 预览处理计划
        """
        has_stdin = not sys.stdin.isatty()
        if not input and not has_stdin:
            cli_error(
                ErrorType.INVALID_ARGS,
                "未提供输入",
                context={"input": None, "stdin_tty": True},
                suggestion="提供位置参数: flexllm batch data.jsonl，或通过管道: cat data.jsonl | flexllm batch -o out.jsonl",
                doc="flexllm batch --help",
            )

        auto_generated_output = False
        if not output:
            if not input:
                cli_error(
                    ErrorType.INVALID_ARGS,
                    "从 stdin 读取时必须指定输出文件",
                    context={"input": None, "output": None, "stdin_tty": False},
                    suggestion="使用 -o 指定输出: cat data.jsonl | flexllm batch -o out.jsonl",
                    doc="flexllm batch --help",
                )

            input_path = Path(input)
            stem = input_path.stem
            output = str(input_path.parent / f"{stem}.output.jsonl")
            auto_generated_output = True

        if not output.endswith(".jsonl"):
            cli_error(
                ErrorType.INVALID_ARGS,
                "输出文件扩展名必须是 .jsonl",
                context={
                    "arg": "-o/--output",
                    "received": output,
                    "expected_suffix": ".jsonl",
                },
                suggestion=f"改为: -o {Path(output).stem}.jsonl",
                doc="flexllm batch --help",
            )

        config = get_config()
        batch_config = config.get_batch_config()

        model_config = None
        endpoints_config = None
        use_pool = False

        if model:
            model_config = config.get_model_config(model)
            if not model_config:
                available = [
                    m.get("name", m.get("id", "?")) for m in config.config.get("models", [])
                ]
                cli_error(
                    ErrorType.NOT_FOUND,
                    "模型未找到",
                    context={
                        "arg": "-m/--model",
                        "received": model,
                        "available_models": available,
                    },
                    suggestion="使用 flexllm list 查看已配置模型，或 flexllm list --json 获取 JSON 列表",
                    doc="flexllm batch --help",
                )
        elif batch_config.get("endpoints"):
            endpoints_config = batch_config["endpoints"]
            use_pool = len(endpoints_config) > 0
        else:
            model_config = config.get_model_config(None)
            if not model_config:
                available = [
                    m.get("name", m.get("id", "?")) for m in config.config.get("models", [])
                ]
                cli_error(
                    ErrorType.NOT_FOUND,
                    "未找到模型配置",
                    context={
                        "default_model": config.config.get("default"),
                        "available_models": available,
                        "has_endpoints": False,
                    },
                    suggestion="使用 -m 指定模型、运行 flexllm list，或在 ~/.flexllm/config.yaml 的 batch 节配置 endpoints",
                    doc="flexllm batch --help",
                )

        model_id = model_config.get("id", model) if model_config else None
        base_url = model_config.get("base_url") if model_config else None
        api_key = model_config.get("api_key", "EMPTY") if model_config else None

        effective_cache = cache if cache is not None else batch_config["cache"]
        effective_return_usage = return_usage or batch_config["return_usage"]
        effective_preprocess_msg = preprocess_msg or batch_config["preprocess_msg"]
        effective_track_cost = track_cost or batch_config["track_cost"]
        effective_concurrency = (
            concurrency if concurrency is not None else batch_config["concurrency"]
        )
        effective_max_qps = max_qps if max_qps is not None else batch_config["max_qps"]

        effective_system = system if system is not None else config.get_system(model)
        effective_user_template = (
            user_template if user_template is not None else config.get_user_template(model)
        )

        effective_save_input: bool | str = True
        if save_input is not None:
            low = save_input.lower()
            if low == "false":
                effective_save_input = False
            elif low == "last":
                effective_save_input = "last"
            elif low == "true":
                effective_save_input = True
            else:
                cli_error(
                    ErrorType.INVALID_ARGS,
                    "--save-input 参数值无效",
                    context={
                        "arg": "--save-input",
                        "received": save_input,
                        "expected": ["true", "last", "false"],
                    },
                    suggestion="true=完整保存, last=仅最后user内容, false=不保存",
                    doc="flexllm batch --help",
                )

        try:
            if user_field:
                records, _, _ = parse_batch_input(input, skip_format_detection=True)
                format_type = "custom"
                message_fields = [user_field, system_field]
                if user_field not in records[0]:
                    available = list(records[0].keys())
                    cli_error(
                        ErrorType.INVALID_ARGS,
                        "指定的字段在输入文件中不存在",
                        context={
                            "arg": "--user-field/-uf",
                            "received": user_field,
                            "available_fields": available,
                            "input_file": input,
                        },
                        suggestion="使用上方 available_fields 中的字段名，或 dt head "
                        f"{input} -n 1 查看完整结构",
                        doc="flexllm batch --help",
                    )
            else:
                records, format_type, message_fields = parse_batch_input(input)
            if limit is not None:
                records = records[:limit]
            print(f"输入格式: {format_type}", file=sys.stderr)
            print(f"记录数: {len(records)}", file=sys.stderr)
            if auto_generated_output:
                print(f"输出文件: {output} (自动生成)", file=sys.stderr)
            else:
                print(f"输出文件: {output}", file=sys.stderr)

            if use_pool:
                print(
                    f"客户端: LLMClientPool ({len(endpoints_config)} endpoints)",
                    file=sys.stderr,
                )
            else:
                print(f"客户端: LLMClient ({model_config.get('name', model_id)})", file=sys.stderr)

            messages_list = []
            metadata_list = []

            for record in records:
                messages, metadata = convert_to_messages(
                    record, format_type, message_fields, effective_system, effective_user_template
                )
                messages_list.append(messages)
                metadata_list.append(metadata if metadata else None)

            has_metadata = any(m for m in metadata_list)
            if not has_metadata:
                metadata_list = None

            if dry_run:
                dry_run_output(
                    {
                        "action": "batch",
                        "input_file": input,
                        "output_file": output,
                        "format": format_type,
                        "record_count": len(records),
                        "model": model_id,
                        "base_url": base_url,
                        "concurrency": effective_concurrency,
                        "max_qps": effective_max_qps,
                        "cache": effective_cache,
                        "sample_messages": messages_list[0] if messages_list else None,
                    }
                )

            async def _run_batch():
                from flexllm import LLMClient, LLMClientPool

                from ..cache import ResponseCacheConfig

                cache_config = None
                if effective_cache:
                    cache_config = ResponseCacheConfig.with_ttl(ttl=batch_config["cache_ttl"])

                kwargs = config.get_model_params(model)
                for key in ("temperature", "max_tokens", "top_p", "top_k", "thinking"):
                    if batch_config.get(key) is not None:
                        kwargs[key] = batch_config[key]
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens

                response_format = parse_schema(schema)
                if response_format is not None:
                    kwargs["response_format"] = response_format

                if use_pool:
                    pool_kwargs = {
                        "endpoints": endpoints_config,
                        "fallback": batch_config.get("fallback", True),
                        "concurrency_limit": effective_concurrency,
                        "timeout": batch_config["timeout"],
                        "retry_times": batch_config["retry_times"],
                        "cache": cache_config,
                    }
                    if effective_max_qps is not None:
                        pool_kwargs["max_qps"] = effective_max_qps

                    async with LLMClientPool(**pool_kwargs) as pool:
                        results, summary = await pool.chat_completions_batch(
                            messages_list=messages_list,
                            output_jsonl=output,
                            show_progress=True,
                            return_summary=True,
                            return_usage=effective_return_usage,
                            track_cost=effective_track_cost,
                            flush_interval=batch_config["flush_interval"],
                            metadata_list=metadata_list,
                            save_input=effective_save_input,
                            **kwargs,
                        )
                else:
                    client_kwargs = {
                        "model": model_id,
                        "base_url": base_url,
                        "api_key": api_key,
                        "concurrency_limit": effective_concurrency,
                        "timeout": batch_config["timeout"],
                        "retry_times": batch_config["retry_times"],
                        "retry_delay": batch_config["retry_delay"],
                        "cache": cache_config,
                    }
                    if effective_max_qps is not None:
                        client_kwargs["max_qps"] = effective_max_qps

                    async with LLMClient(**client_kwargs) as client:
                        results, summary = await client.chat_completions_batch(
                            messages_list=messages_list,
                            output_jsonl=output,
                            show_progress=True,
                            return_summary=True,
                            return_usage=effective_return_usage,
                            track_cost=effective_track_cost,
                            preprocess_msg=effective_preprocess_msg,
                            flush_interval=batch_config["flush_interval"],
                            metadata_list=metadata_list,
                            save_input=effective_save_input,
                            **kwargs,
                        )
                return results, summary

            results, summary = asyncio.run(_run_batch())

            if summary:
                print(f"\n完成: {summary}", file=sys.stderr)
            print(f"输出文件: {output}", file=sys.stderr)

        except json.JSONDecodeError as e:
            cli_error(
                ErrorType.INVALID_ARGS,
                "JSON 解析失败",
                context={
                    "input_file": input,
                    "line": getattr(e, "lineno", None),
                    "column": getattr(e, "colno", None),
                    "parser_message": e.msg if hasattr(e, "msg") else str(e),
                },
                suggestion=f"检查文件是否为合法 JSONL（每行一个 JSON）。使用 dt check {input} 验证",
                doc="flexllm batch --help",
            )
        except ValueError as e:
            cli_error(
                ErrorType.INVALID_ARGS,
                str(e),
                context={"input_file": input, "exception_type": "ValueError"},
                suggestion="检查输入格式。使用 --user-field 指定字段名跳过自动检测",
                doc="flexllm batch --help",
            )
        except FileNotFoundError:
            cli_error(
                ErrorType.IO_ERROR,
                "输入文件不存在",
                context={"arg": "input", "received": input},
                suggestion="检查路径是否正确，使用绝对路径或相对于当前目录的路径",
                doc="flexllm batch --help",
            )
        except typer.Exit:
            raise
        except Exception as e:
            # traceback 不打印到 stderr，避免污染 Agent 解析的 JSON；
            # 转而放入 context.traceback，Agent 可以按需解析。
            import traceback as _tb

            cli_error(
                ErrorType.GENERAL,
                str(e),
                context={
                    "exception_type": type(e).__name__,
                    "input_file": input,
                    "output_file": output,
                    "model": model_id,
                    "traceback": _tb.format_exc().strip().splitlines()[-5:],
                },
                doc="flexllm batch --help",
            )

    @app.command()
    def models(
        base_url: Annotated[str | None, Option("--base-url", help="API 地址")] = None,
        api_key: Annotated[str | None, Option("--api-key", help="API 密钥")] = None,
        name: Annotated[str | None, Option("-n", "--name", help="模型配置名称")] = None,
        json_output: Annotated[bool, Option("--json", help="输出 JSON 格式")] = False,
    ):
        """列出远程服务器上的可用模型（查询 /v1/models 接口）

        与 'list' 不同：models 查询远程服务器，list 读取本地配置文件。

        \b
        Examples:
          flexllm models                           # 查询默认服务器的模型
          flexllm models -n claude                 # 查询指定配置的服务器
          flexllm models --json                    # JSON 格式输出
        """
        import requests

        config = get_config()
        model_config = config.get_model_config(name)
        if model_config:
            base_url = base_url or model_config.get("base_url")
            api_key = api_key or model_config.get("api_key", "EMPTY")
            provider = model_config.get("provider", "openai")
        else:
            provider = "openai"

        if not base_url and provider == "claude":
            base_url = "https://api.anthropic.com/v1"

        if not base_url:
            cli_error(
                ErrorType.NOT_FOUND,
                "未配置 base_url",
                context={"arg": "-n/--name", "received": name, "provider": provider},
                suggestion="通过 --base-url 指定、设置 FLEXLLM_BASE_URL，或运行 flexllm init",
                doc="flexllm models --help",
            )

        is_gemini = provider == "gemini" or "generativelanguage.googleapis.com" in base_url

        try:
            if provider == "claude":
                headers = {
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                }
                if isinstance(api_key, str) and "sk-ant-oat" in api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                    headers["anthropic-beta"] = "oauth-2025-04-20"
                else:
                    headers["x-api-key"] = api_key
                response = requests.get(
                    f"{base_url.rstrip('/')}/models", headers=headers, timeout=10
                )
            elif is_gemini:
                url = f"{base_url.rstrip('/')}/models?key={api_key}"
                response = requests.get(url, timeout=10)
            else:
                headers = {"Authorization": f"Bearer {api_key}"}
                response = requests.get(
                    f"{base_url.rstrip('/')}/models", headers=headers, timeout=10
                )

            if response.status_code == 200:
                models_data = response.json()

                if is_gemini:
                    models_list = models_data.get("models", [])
                else:
                    if isinstance(models_data, dict) and "data" in models_data:
                        models_list = models_data["data"]
                    elif isinstance(models_data, list):
                        models_list = models_data
                    else:
                        models_list = []

                if json_output:
                    import json as json_module

                    ids = []
                    for m in models_list:
                        if isinstance(m, dict):
                            if is_gemini:
                                ids.append(m.get("name", "").replace("models/", ""))
                            else:
                                ids.append(m.get("id", m.get("name", "unknown")))
                        else:
                            ids.append(str(m))
                    print(
                        json_module.dumps(
                            {"base_url": base_url, "models": ids, "count": len(ids)},
                            indent=2,
                            ensure_ascii=False,
                        )
                    )
                    return

                print("\n可用模型列表")
                print(f"服务器: {base_url}")
                print("-" * 50)

                if models_list:
                    for i, m in enumerate(models_list, 1):
                        if isinstance(m, dict):
                            if is_gemini:
                                display_name = m.get("name", "").replace("models/", "")
                            else:
                                display_name = m.get("id", m.get("name", "unknown"))
                            print(f"  {i:2d}. {display_name}")
                        else:
                            print(f"  {i:2d}. {m}")
                    print(f"\n共 {len(models_list)} 个模型")
                else:
                    print("未找到可用模型")
            else:
                cli_error(
                    ErrorType.NETWORK_ERROR,
                    f"远程 /models 接口返回 HTTP {response.status_code}",
                    context={
                        "base_url": base_url,
                        "http_status": response.status_code,
                        "response_body": response.text[:200] if response.text else None,
                    },
                    suggestion="检查 base_url 是否正确，API Key 是否有权限。使用 flexllm test 诊断",
                    doc="flexllm models --help",
                    retryable=True,
                )

        except requests.exceptions.RequestException as e:
            cli_error(
                ErrorType.NETWORK_ERROR,
                f"连接失败: {e}",
                context={
                    "exception_type": type(e).__name__,
                    "base_url": base_url,
                },
                suggestion="检查网络、base_url，或使用 flexllm test 诊断",
                doc="flexllm models --help",
                retryable=True,
            )
        except Exception as e:
            cli_error(
                ErrorType.GENERAL,
                str(e),
                context={
                    "exception_type": type(e).__name__,
                    "base_url": base_url,
                },
                doc="flexllm models --help",
            )

    @app.command("list")
    def list_models(
        json_output: Annotated[bool, Option("--json", help="输出 JSON 格式")] = False,
    ):
        """列出本地配置文件中的模型（~/.flexllm/config.yaml）

        与 'models' 不同：list 读取本地配置文件，models 查询远程服务器。

        \b
        Examples:
          flexllm list                             # 列出所有已配置模型
          flexllm list --json                      # JSON 格式输出
        """
        config = get_config()
        models_cfg = config.config.get("models", [])
        default = config.config.get("default", "")

        if not models_cfg:
            print("未配置模型")
            print("提示: 创建 ~/.flexllm/config.yaml 或设置环境变量")
            return

        if json_output:
            import json as json_module

            output = []
            for m in models_cfg:
                name = m.get("name", m.get("id", "?"))
                model_id = m.get("id", "?")
                provider = m.get("provider", "openai")
                endpoints = m.get("endpoints")
                entry = {
                    "name": name,
                    "id": model_id,
                    "provider": provider,
                    "is_default": name == default or model_id == default,
                }
                if endpoints and len(endpoints) > 1:
                    entry["type"] = "pool"
                    entry["endpoints"] = len(endpoints)
                output.append(entry)
            print(json_module.dumps(output, indent=2, ensure_ascii=False))
            return

        print(f"已配置模型 (共 {len(models_cfg)} 个):\n")
        for m in models_cfg:
            name = m.get("name", m.get("id", "?"))
            model_id = m.get("id", "?")
            provider = m.get("provider", "openai")
            is_default = " (默认)" if name == default or model_id == default else ""
            endpoints = m.get("endpoints")

            print(f"  {name}{is_default}")
            if name != model_id:
                print(f"    id: {model_id}")

            if endpoints and len(endpoints) > 1:
                print(f"    type: pool ({len(endpoints)} endpoints)")
                print(f"    fallback: {m.get('fallback', True)}")
            else:
                print(f"    provider: {provider}")
            print()

    @app.command("set-model")
    def set_model(
        model_name: Annotated[str, Argument(help="模型名称或 ID")],
        dry_run: Annotated[bool, Option("--dry-run", help="预览操作内容，不实际执行")] = False,
    ):
        """设置默认模型

        \b
        Examples:
        flexllm set-model gpt-4
        flexllm set-model local-ollama
        flexllm set-model gpt-4 --dry-run          # 预览变更
        """
        config = get_config()
        config_path = config.get_config_path()

        if not config_path:
            cli_error(
                ErrorType.NOT_FOUND,
                "未找到配置文件",
                context={
                    "search_paths": [
                        "./flexllm_config.yaml",
                        str(Path.home() / ".flexllm" / "config.yaml"),
                    ]
                },
                suggestion="运行 flexllm init 创建默认配置文件",
                doc="flexllm set-model --help",
            )

        model_config = config.get_model_config(model_name)
        if not model_config:
            available = [m.get("name", m.get("id", "?")) for m in config.config.get("models", [])]
            cli_error(
                ErrorType.NOT_FOUND,
                "模型未找到",
                context={
                    "arg": "model_name",
                    "received": model_name,
                    "available_models": available,
                    "config_path": str(config_path),
                },
                suggestion="使用 flexllm list --json 获取 JSON 格式的模型列表",
                doc="flexllm set-model --help",
            )

        if dry_run:
            default_value = model_config.get("name", model_config.get("id"))
            dry_run_output(
                {
                    "action": "set_model",
                    "config_path": str(config_path),
                    "old_default": config.config.get("default"),
                    "new_default": default_value,
                }
            )

        try:
            import yaml

            with open(config_path, encoding="utf-8") as f:
                file_config = yaml.safe_load(f) or {}

            default_value = model_config.get("name", model_config.get("id"))
            old_default = file_config.get("default")
            file_config["default"] = default_value

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(file_config, f, default_flow_style=False, allow_unicode=True)

            print(f"默认模型已设置为: {default_value}")
            if old_default and old_default != default_value:
                print(f"(原默认模型: {old_default})")

            config.config["default"] = default_value

        except ImportError:
            cli_error(
                ErrorType.DEPENDENCY_MISSING,
                "缺少依赖: pyyaml",
                context={"missing_package": "pyyaml"},
                suggestion="pip install pyyaml",
                doc="flexllm set-model --help",
            )
        except Exception as e:
            cli_error(
                ErrorType.GENERAL,
                str(e),
                context={
                    "exception_type": type(e).__name__,
                    "config_path": str(config_path),
                },
                doc="flexllm set-model --help",
            )

    @app.command()
    def test(
        model: Annotated[str | None, Option("-m", "--model", help="模型名称")] = None,
        base_url: Annotated[str | None, Option("--base-url", help="API 地址")] = None,
        api_key: Annotated[str | None, Option("--api-key", help="API 密钥")] = None,
        message: Annotated[
            str, Option("--message", help="测试消息")
        ] = "Hello, please respond with 'OK' if you can see this message.",
        timeout: Annotated[int, Option("--timeout", help="超时时间（秒）")] = 30,
        json_output: Annotated[bool, Option("--json", help="输出 JSON 格式")] = False,
    ):
        """测试 LLM 服务连接

        \b
        Examples:
          flexllm test                             # 测试默认模型连接
          flexllm test -m gpt-4                    # 测试指定模型
          flexllm test --base-url http://localhost:11434/v1  # 测试自定义服务器
          flexllm test --json                      # JSON 格式输出（适合脚本）
        """
        import time

        import requests

        model, base_url, api_key = resolve_model_config(model, base_url, api_key)

        if not base_url:
            cli_error(
                ErrorType.NOT_FOUND,
                "未配置 base_url",
                context={"arg": "-m/--model", "received": model},
                suggestion="通过 --base-url 指定、设置 FLEXLLM_BASE_URL，或运行 flexllm init",
                doc="flexllm test --help",
            )

        result_data = {}

        if not json_output:
            print("\nLLM 服务连接测试")
            print("-" * 50)
            print("\n1. 测试服务器连接...")
            print(f"   地址: {base_url}")

        try:
            start = time.time()
            response = requests.get(
                f"{base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=timeout,
            )
            elapsed = time.time() - start

            if response.status_code == 200:
                models_data = response.json()
                if isinstance(models_data, dict) and "data" in models_data:
                    model_count = len(models_data["data"])
                elif isinstance(models_data, list):
                    model_count = len(models_data)
                else:
                    model_count = 0

                if json_output:
                    result_data["server"] = {
                        "status": "ok",
                        "base_url": base_url,
                        "latency_s": round(elapsed, 2),
                        "model_count": model_count,
                    }
                else:
                    print(f"   ✓ 连接成功 ({elapsed:.2f}s)")
                    print(f"   可用模型数: {model_count}")
            else:
                if json_output:
                    result_data["server"] = {
                        "status": "error",
                        "base_url": base_url,
                        "http_status": response.status_code,
                    }
                else:
                    cli_error(
                        ErrorType.NETWORK_ERROR,
                        f"连接失败: HTTP {response.status_code}",
                        context={
                            "base_url": base_url,
                            "http_status": response.status_code,
                            "response_body": response.text[:200] if response.text else None,
                        },
                        suggestion="检查 base_url 是否正确，API Key 是否有权限",
                        doc="flexllm test --help",
                        retryable=True,
                    )
        except Exception as e:
            if json_output:
                result_data["server"] = {"status": "error", "error": str(e)}
            else:
                cli_error(
                    ErrorType.NETWORK_ERROR,
                    f"连接失败: {e}",
                    context={
                        "exception_type": type(e).__name__,
                        "base_url": base_url,
                        "timeout_s": timeout,
                    },
                    suggestion="检查网络连通性或增加 --timeout",
                    doc="flexllm test --help",
                    retryable=True,
                )

        if model:
            if not json_output:
                print("\n2. 测试 Chat API...")
                print(f"   模型: {model}")
            try:
                start = time.time()
                response = requests.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": message}],
                        "max_tokens": 50,
                    },
                    timeout=timeout,
                )
                elapsed = time.time() - start

                if response.status_code == 200:
                    result = response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if json_output:
                        result_data["chat"] = {
                            "status": "ok",
                            "model": model,
                            "latency_s": round(elapsed, 2),
                            "response": content[:100],
                        }
                    else:
                        print(f"   ✓ 调用成功 ({elapsed:.2f}s)")
                        print(f"   响应: {content[:100]}...")
                else:
                    if json_output:
                        result_data["chat"] = {
                            "status": "error",
                            "model": model,
                            "http_status": response.status_code,
                        }
                    else:
                        print(f"   ✗ 调用失败: HTTP {response.status_code}")
                        print(f"   {response.text[:200]}")
            except Exception as e:
                if json_output:
                    result_data["chat"] = {"status": "error", "error": str(e)}
                else:
                    print(f"   ✗ 调用失败: {e}")

        if json_output:
            import json as json_module

            print(json_module.dumps(result_data, indent=2, ensure_ascii=False))
        else:
            print("\n测试完成")

    @app.command()
    def init(
        path: Annotated[str | None, Option("-p", "--path", help="配置文件路径")] = None,
        dry_run: Annotated[bool, Option("--dry-run", help="预览操作内容，不实际执行")] = False,
    ):
        """初始化 flexllm 配置文件

        在 ~/.flexllm/ 目录创建默认配置文件模板。如已存在则不覆盖。

        \b
        Examples:
          flexllm init                             # 创建 ~/.flexllm/config.yaml
          flexllm init -p ./flexllm_config.yaml    # 指定路径
          flexllm init --dry-run                   # 预览将创建的路径
        """
        if path is None:
            config_path = Path.home() / ".flexllm" / "config.yaml"
        else:
            config_path = Path(path)

        if dry_run:
            dry_run_output(
                {
                    "action": "init",
                    "config_path": str(config_path),
                    "exists": config_path.exists(),
                }
            )

        if config_path.exists():
            print(f"配置文件已存在: {config_path}")
            return

        config_path.parent.mkdir(parents=True, exist_ok=True)

        default_config = """# flexllm 配置文件
# 配置搜索路径:
#   1. 当前目录: ./flexllm_config.yaml
#   2. 用户目录: ~/.flexllm/config.yaml

# 默认模型
default: "gpt-4"

# 全局系统提示词（应用于所有命令，除非被覆盖）
# system: "You are a helpful assistant."

# 全局 user content 模板（使用 {content} 作为占位符）
# 适用于需要特定提示词格式的微调模型
# user_template: "{content}/detail"

# 模型列表
models:
  - id: gpt-4
    name: gpt-4
    provider: openai
    base_url: https://api.openai.com/v1
    api_key: your-api-key
    # system: "You are a GPT-4 assistant."  # 模型级别 system prompt（可选）
    # user_template: "{content}"             # 模型级别 user template（可选）

  - id: local-ollama
    name: local-ollama
    provider: openai
    base_url: http://localhost:11434/v1
    api_key: EMPTY

# batch 命令配置（可选）
# 这些配置可通过 CLI 参数覆盖
# batch:
#   concurrency: 10
#   max_qps: 100
#   timeout: 120
#   retry_times: 3
#   cache: false
#   cache_ttl: 86400
"""

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(default_config)
            print(f"已创建配置文件: {config_path}")
            print("请编辑配置文件填入 API 密钥")
        except Exception as e:
            cli_error(
                ErrorType.IO_ERROR,
                f"创建配置文件失败: {e}",
                context={
                    "exception_type": type(e).__name__,
                    "config_path": str(config_path),
                },
                suggestion="检查父目录权限，或通过 -p 指定其他路径",
                doc="flexllm init --help",
            )

    @app.command()
    def pricing(
        model: Annotated[str | None, Argument(help="模型名称（支持模糊匹配）")] = None,
        update: Annotated[bool, Option("--update", help="从 OpenRouter 更新定价表")] = False,
        json_output: Annotated[bool, Option("--json", help="输出 JSON 格式")] = False,
    ):
        """查询模型定价信息

        \b
        Examples:
        flexllm pricing                  # 列出所有模型定价
        flexllm pricing gpt-4o           # 查询 gpt-4o 定价
        flexllm pricing claude           # 模糊匹配 claude 相关模型
        flexllm pricing --update         # 从 OpenRouter 更新定价表
        """
        from ..pricing import get_pricing, reload_pricing

        MODEL_PRICING = get_pricing()

        if update:
            print("正在从 OpenRouter API 获取最新定价...")
            try:
                from ..pricing.updater import collect_pricing, update_pricing_file

                pricing_map = collect_pricing()
                print(f"获取到 {len(pricing_map)} 个模型定价")

                if update_pricing_file(pricing_map):
                    reload_pricing()
                    print("✓ 定价数据已更新")
                else:
                    cli_error(
                        ErrorType.NETWORK_ERROR,
                        "定价数据写入失败",
                        context={"source": "openrouter.ai", "fetched_count": len(pricing_map)},
                        suggestion="检查 flexllm 安装目录的写权限",
                        doc="flexllm pricing --help",
                        retryable=True,
                    )
            except Exception as e:
                cli_error(
                    ErrorType.NETWORK_ERROR,
                    f"更新失败: {e}",
                    context={
                        "exception_type": type(e).__name__,
                        "source": "openrouter.ai",
                    },
                    suggestion="检查网络或稍后重试",
                    doc="flexllm pricing --help",
                    retryable=True,
                )
            return

        if model:
            matches = {
                name: price
                for name, price in MODEL_PRICING.items()
                if model.lower() in name.lower()
            }

            if not matches:
                sample = sorted(MODEL_PRICING.keys())[:10]
                cli_error(
                    ErrorType.NOT_FOUND,
                    "无模型匹配",
                    context={
                        "arg": "model",
                        "received": model,
                        "sample_models": sample,
                        "total_models": len(MODEL_PRICING),
                    },
                    suggestion="使用 flexllm pricing --json 获取完整模型列表",
                    doc="flexllm pricing --help",
                )

            if json_output:
                import json as json_module

                output_data = {
                    name: {
                        "input_per_1m": round(p["input"] * 1e6, 4),
                        "output_per_1m": round(p["output"] * 1e6, 4),
                    }
                    for name, p in sorted(matches.items())
                }
                print(json_module.dumps(output_data, indent=2, ensure_ascii=False))
            else:
                print(f"\n模型定价 (匹配 '{model}'):\n")
                print(f"{'模型':<30} {'输入 ($/1M)':<15} {'输出 ($/1M)':<15}")
                print("-" * 60)
                for name in sorted(matches.keys()):
                    p = matches[name]
                    input_price = p["input"] * 1e6
                    output_price = p["output"] * 1e6
                    print(f"{name:<30} ${input_price:<14.4f} ${output_price:<14.4f}")
                print(f"\n共 {len(matches)} 个模型")
        else:
            if json_output:
                import json as json_module

                output_data = {
                    name: {
                        "input_per_1m": round(p["input"] * 1e6, 4),
                        "output_per_1m": round(p["output"] * 1e6, 4),
                    }
                    for name, p in sorted(MODEL_PRICING.items())
                }
                print(json_module.dumps(output_data, indent=2, ensure_ascii=False))
            else:
                groups = {}
                for name, price in MODEL_PRICING.items():
                    if name.startswith(("gpt-", "o1", "o3", "o4")):
                        group = "OpenAI"
                    elif name.startswith("claude-"):
                        group = "Anthropic"
                    elif name.startswith("gemini-"):
                        group = "Google"
                    elif name.startswith("deepseek"):
                        group = "DeepSeek"
                    elif name.startswith(("qwen", "qwen2", "qwen3")):
                        group = "Alibaba"
                    elif name.startswith(("mistral", "ministral", "codestral", "devstral")):
                        group = "Mistral"
                    elif name.startswith("llama-"):
                        group = "Meta"
                    elif name.startswith("grok"):
                        group = "xAI"
                    elif name.startswith("nova"):
                        group = "Amazon"
                    else:
                        group = "Other"

                    if group not in groups:
                        groups[group] = []
                    groups[group].append((name, price))

                print(f"\n模型定价表 (共 {len(MODEL_PRICING)} 个模型):\n")
                print(f"{'模型':<30} {'输入 ($/1M)':<15} {'输出 ($/1M)':<15}")
                print("=" * 60)

                for group_name in [
                    "OpenAI",
                    "Anthropic",
                    "Google",
                    "DeepSeek",
                    "Alibaba",
                    "Mistral",
                    "Meta",
                    "xAI",
                    "Amazon",
                    "Other",
                ]:
                    if group_name not in groups:
                        continue
                    models_in_group = groups[group_name]
                    print(f"\n[{group_name}]")
                    for name, p in sorted(models_in_group):
                        input_price = p["input"] * 1e6
                        output_price = p["output"] * 1e6
                        print(f"  {name:<28} ${input_price:<14.4f} ${output_price:<14.4f}")

    @app.command()
    def credits(
        model: Annotated[str | None, Option("-m", "--model", help="模型名称")] = None,
        key: Annotated[str | None, Option("-k", "--key", help="直接指定 API Key")] = None,
        json_output: Annotated[bool, Option("--json", help="输出 JSON 格式")] = False,
    ):
        """查询 API Key 余额

        支持的 provider:
        OpenRouter, SiliconFlow, DeepSeek, AI/ML API, OpenAI

        \b
        Examples:
        flexllm credits                        # 查询默认模型的 key 余额
        flexllm credits -m grok-4              # 查询指定模型的 key 余额
        flexllm credits -k sk-or-v1-xxx...     # 直接查询指定 key 的余额
        flexllm credits --json                 # JSON 格式输出
        """
        if key:
            result = query_credits_by_key(key)

            if result is None:
                cli_error(
                    ErrorType.AUTH_FAILED,
                    "无法识别此 API Key 对应的 provider",
                    context={
                        "key_prefix": key[:15] + "..." if len(key) > 15 else key,
                        "supported_providers": [
                            "OpenRouter",
                            "SiliconFlow",
                            "DeepSeek",
                            "AI/ML API",
                            "OpenAI",
                        ],
                    },
                    suggestion="检查 key 格式，或使用 -m 通过已配置模型查询",
                    doc="flexllm credits --help",
                )

            if "error" in result:
                cli_error(
                    ErrorType.NETWORK_ERROR,
                    result["error"],
                    context={
                        "key_prefix": key[:15] + "...",
                        "provider": result.get("provider"),
                    },
                    suggestion="检查 key 是否有效、网络是否通畅",
                    doc="flexllm credits --help",
                    retryable=True,
                )

            if json_output:
                import json as json_module

                print(json_module.dumps(result, indent=2, ensure_ascii=False))
                return

            print(f"\n{result['provider']} 账户余额")
            print(f"API Key: {key[:15]}...{key[-4:]}")
            print("-" * 40)

            for k, value in result["data"].items():
                print(f"  {k}: {value}")
            return

        config = get_config()
        model_config = config.get_model_config(model)

        if not model_config:
            available = [m.get("name", m.get("id", "?")) for m in config.config.get("models", [])]
            cli_error(
                ErrorType.NOT_FOUND,
                "未找到模型配置",
                context={
                    "arg": "-m/--model",
                    "received": model,
                    "available_models": available,
                },
                suggestion="使用 flexllm list 查看可用模型",
                doc="flexllm credits --help",
            )

        base_url = model_config.get("base_url", "")
        api_key = model_config.get("api_key", "")
        model_name = model_config.get("name", model_config.get("id", "unknown"))

        if not api_key or api_key == "EMPTY":
            cli_error(
                ErrorType.AUTH_FAILED,
                f"模型 '{model_name}' 未配置 API Key",
                context={"model": model_name, "base_url": base_url},
                suggestion="在 ~/.flexllm/config.yaml 中为此模型设置 api_key",
                doc="flexllm credits --help",
            )

        result = query_credits(base_url, api_key)

        if result is None:
            cli_error(
                ErrorType.NOT_FOUND,
                "该 provider 不支持余额查询",
                context={
                    "base_url": base_url,
                    "model": model_name,
                    "supported_providers": [
                        "OpenRouter",
                        "SiliconFlow",
                        "DeepSeek",
                        "AI/ML API",
                        "OpenAI",
                    ],
                },
                suggestion="此 provider 暂未实现余额查询接口",
                doc="flexllm credits --help",
            )

        if "error" in result:
            cli_error(
                ErrorType.NETWORK_ERROR,
                result["error"],
                context={"provider": result.get("provider"), "base_url": base_url},
                suggestion="检查网络或 API Key 有效性",
                doc="flexllm credits --help",
                retryable=True,
            )

        if json_output:
            import json as json_module

            print(json_module.dumps(result, indent=2, ensure_ascii=False))
            return

        print(f"\n{result['provider']} 账户余额")
        print(f"模型配置: {model_name}")
        print(f"API Key: {api_key[:15]}...{api_key[-4:]}")
        print("-" * 40)

        for k, value in result["data"].items():
            print(f"  {k}: {value}")

    @app.command()
    def mock(
        port: Annotated[int, Option("-p", "--port", help="端口号")] = 8001,
        delay: Annotated[
            str, Option("-d", "--delay", help="延迟时间，支持 '0.5' 或 '1-5' 格式")
        ] = "0.1",
        response_len: Annotated[
            str,
            Option("-l", "--response-len", help="响应长度（字符），支持 '100' 或 '10-1000' 格式"),
        ] = "10-1000",
        model: Annotated[str, Option("-m", "--model", help="模型名称")] = "mock-model",
        rps: Annotated[float, Option("--rps", help="每秒最大请求数，0 表示不限制")] = 0,
        token_rate: Annotated[
            float, Option("--token-rate", help="流式返回时每秒 token 数，0 表示不限制")
        ] = 0,
        error_rate: Annotated[
            float, Option("--error-rate", help="请求失败率 (0-1)，0 表示不失败")
        ] = 0,
        thinking: Annotated[bool, Option("--thinking", help="响应中包含思考/推理内容")] = False,
        qa: Annotated[
            str,
            Option(
                "--qa",
                help="QA 数据集路径（JSONL），每行 {input, output}。精确匹配优先，否则子串包含匹配（取最长）",
            ),
        ] = None,
        log: Annotated[
            str, Option("--log", help="请求日志保存路径（JSONL），记录每个请求的完整输入输出")
        ] = None,
        dry_run: Annotated[bool, Option("--dry-run", help="预览操作内容，不实际执行")] = False,
    ):
        """启动 Mock LLM 服务器

        \b
        Examples:
        flexllm mock                          # 默认配置，端口 8001
        flexllm mock -p 8080                  # 指定端口
        flexllm mock -d 0.5                   # 固定延迟 0.5s
        flexllm mock --error-rate 0.5         # 50% 请求返回错误
        flexllm mock --qa qa.jsonl            # QA 数据集确定性回复
        flexllm mock --log requests.jsonl     # 额外将请求日志写入文件
        flexllm mock --dry-run                # 预览启动配置

        QA 数据集格式（每行一个 JSON）:
        {"input": "关键词或完整问题", "output": "对应的回复内容"}

        匹配规则: 精确匹配优先 → 子串包含匹配（多个命中取最长）→ 未匹配则随机生成
        例: input="天气" 可被 "今天天气怎么样" 触发
        """
        try:
            from ..mock import MockLLMServer, MockServerConfig, parse_range
        except ImportError:
            cli_error(
                ErrorType.DEPENDENCY_MISSING,
                "缺少依赖: aiohttp",
                context={"missing_package": "aiohttp", "feature": "mock"},
                suggestion="pip install aiohttp 或 pip install 'flexllm[all]'",
                doc="flexllm mock --help",
            )

        delay_min, delay_max = parse_range(delay, float)
        response_min_len, response_max_len = parse_range(response_len, int)

        config = MockServerConfig(
            port=port,
            delay_min=delay_min,
            delay_max=delay_max,
            model=model,
            response_min_len=response_min_len,
            response_max_len=response_max_len,
            rps=rps,
            token_rate=token_rate,
            error_rate=error_rate,
            thinking=thinking,
            qa_path=qa,
            log_path=log,
        )

        if dry_run:
            dry_run_output(
                {
                    "action": "mock",
                    "port": port,
                    "delay": delay,
                    "response_len": response_len,
                    "model": model,
                    "rps": rps,
                    "token_rate": token_rate,
                    "error_rate": error_rate,
                    "thinking": thinking,
                    "qa": qa,
                    "log": log,
                }
            )

        print(f"Mock LLM Server starting on port {port}")
        print(f"  Delay: {delay_min}-{delay_max}s")
        print(f"  Response length: {response_min_len}-{response_max_len} chars")
        print(f"  Model: {model}")
        if rps > 0:
            print(f"  RPS limit: {rps}")
        if token_rate > 0:
            print(f"  Token rate: {token_rate}/s (streaming)")
        if error_rate > 0:
            print(f"  Error rate: {error_rate * 100:.1f}%")
        if thinking:
            print("  Thinking: enabled")
        if qa:
            import pathlib

            qa_count = sum(1 for line in pathlib.Path(qa).read_text().splitlines() if line.strip())
            print(f"  QA dataset: {qa} ({qa_count} entries)")
        if log:
            print(f"  Log: {log}")
        print(f"  OpenAI: http://localhost:{port}/v1/chat/completions")
        print(f"  Claude: http://localhost:{port}/v1/messages")
        print(f"  Gemini: http://localhost:{port}/models/{{model}}:generateContent")
        print(f"  MCP:    http://localhost:{port}/mcp")
        print("\nPress Ctrl+C to stop")

        try:
            server = MockLLMServer(config)
            server.run()
        except KeyboardInterrupt:
            print("\nServer stopped")
        except Exception as e:
            cli_error(
                ErrorType.GENERAL,
                str(e),
                context={
                    "exception_type": type(e).__name__,
                    "port": port,
                    "model": model,
                },
                doc="flexllm mock --help",
            )

    @app.command()
    def version(
        json_output: Annotated[bool, Option("--json", help="输出 JSON 格式")] = False,
    ):
        """显示 flexllm 版本信息

        \b
        Examples:
          flexllm version                          # 显示版本号
          flexllm version --json                   # JSON 格式输出
        """
        try:
            from flexllm import __version__

            v = __version__
        except Exception:
            v = "0.1.0"
        if json_output:
            import json as json_module

            print(
                json_module.dumps({"name": "flexllm", "version": v}, indent=2, ensure_ascii=False)
            )
        else:
            print(f"flexllm {v}")

    @app.command("install-skill")
    def install_skill():
        """安装 Claude Code skill 文件到 ~/.claude/skills/

        \b
        Examples:
          flexllm install-skill                    # 安装 skill 文件
        """
        import shutil

        skill_src = Path(__file__).parent.parent / "data" / "SKILL.md"

        if not skill_src.exists():
            cli_error(
                ErrorType.IO_ERROR,
                "找不到 skill 源文件",
                context={"expected_path": str(skill_src)},
                suggestion="pip install --force-reinstall flexllm",
                doc="flexllm install-skill --help",
            )

        skill_dir = Path.home() / ".claude" / "skills" / "flexllm"
        skill_dst = skill_dir / "SKILL.md"

        try:
            skill_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(skill_src, skill_dst)
            print(f"已安装 skill 文件到: {skill_dst}")
            print("Claude Code 现在可以使用 flexllm skill 了")
        except Exception as e:
            cli_error(
                ErrorType.IO_ERROR,
                f"安装失败: {e}",
                context={
                    "exception_type": type(e).__name__,
                    "dst_path": str(skill_dst),
                },
                suggestion="检查 ~/.claude/skills/ 的写权限",
                doc="flexllm install-skill --help",
            )
