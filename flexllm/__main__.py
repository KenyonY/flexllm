"""
flexllm CLI - LLM 客户端命令行工具

提供简洁的 LLM 调用命令:
    flexllm ask "你的问题"
    flexllm chat
    flexllm batch input.jsonl -o output.jsonl
    flexllm models
    flexllm test
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Annotated

try:
    import typer
    from typer import Argument, Option, Typer

    app = Typer(
        name="flexllm",
        help="flexllm - 高性能 LLM 客户端命令行工具",
        add_completion=True,
        no_args_is_help=True,
    )
    HAS_TYPER = True
except ImportError:
    HAS_TYPER = False
    app = None


class FlexLLMConfig:
    """配置管理"""

    def __init__(self):
        self.config = self._load_config()

    def _get_config_paths(self):
        """获取配置文件搜索路径"""
        paths = []
        paths.append(Path.cwd() / "flexllm_config.yaml")
        paths.append(Path.home() / ".flexllm" / "config.yaml")
        return paths

    def _load_config(self) -> dict:
        """加载配置文件"""
        default_config = {"default": None, "models": [], "system": None}

        for config_path in self._get_config_paths():
            if config_path.exists():
                try:
                    import yaml

                    with open(config_path, encoding="utf-8") as f:
                        file_config = yaml.safe_load(f)
                    if file_config:
                        return {**default_config, **file_config}
                except ImportError:
                    pass
                except Exception:
                    pass

        env_config = self._config_from_env()
        if env_config:
            default_config["models"] = [env_config]
            default_config["default"] = env_config.get("id")

        return default_config

    def _config_from_env(self) -> dict | None:
        """从环境变量构建配置（优先级最高）"""
        base_url = os.environ.get("FLEXLLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get("FLEXLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        model = os.environ.get("FLEXLLM_MODEL") or os.environ.get("OPENAI_MODEL")

        # 只需要 base_url 和 model，api_key 可选（本地服务通常不需要）
        if base_url and model:
            return {
                "id": model,
                "name": model,
                "base_url": base_url,
                "api_key": api_key,
                "provider": "openai",
            }
        return None

    def get_model_config(self, name_or_id: str = None) -> dict | None:
        """获取模型配置，环境变量优先级最高"""
        models = self.config.get("models", [])

        # 环境变量优先级最高：当设置了 FLEXLLM_BASE_URL + FLEXLLM_MODEL 时直接使用
        if name_or_id is None:
            env_config = self._config_from_env()
            if env_config:
                return env_config

        if not models:
            return None

        if name_or_id is None:
            name_or_id = self.config.get("default")
            if not name_or_id:
                return models[0] if models else None

        for m in models:
            if m.get("name") == name_or_id:
                return m

        for m in models:
            if m.get("id") == name_or_id:
                return m

        return None

    def get_config_path(self) -> Path | None:
        """获取存在的配置文件路径"""
        for path in self._get_config_paths():
            if path.exists():
                return path
        return None

    def get_system(self, model_name_or_id: str = None) -> str | None:
        """
        获取系统提示词配置

        优先级: batch 配置 > 模型配置 > 全局配置
        """
        # 1. batch 配置
        batch_config = self.config.get("batch", {})
        if "system" in batch_config:
            return batch_config["system"]

        # 2. 模型级别配置
        if model_name_or_id:
            model_config = self.get_model_config(model_name_or_id)
            if model_config and "system" in model_config:
                return model_config["system"]

        # 3. 全局配置
        return self.config.get("system")

    def get_user_template(self, model_name_or_id: str = None) -> str | None:
        """
        获取 user content 模板配置

        优先级: batch 配置 > 模型配置 > 全局配置
        模板中使用 {content} 作为占位符，例如: "{content}/detail"
        """
        # 1. batch 配置
        batch_config = self.config.get("batch", {})
        if "user_template" in batch_config:
            return batch_config["user_template"]

        # 2. 模型级别配置
        if model_name_or_id:
            model_config = self.get_model_config(model_name_or_id)
            if model_config and "user_template" in model_config:
                return model_config["user_template"]

        # 3. 全局配置
        return self.config.get("user_template")

    def get_batch_config(self) -> dict:
        """
        获取 batch 命令的配置

        配置优先级: 用户配置文件 > 默认值
        返回合并后的完整配置字典
        """
        # 默认值
        defaults = {
            # 缓存配置
            "cache": False,
            "cache_ttl": 86400,
            # 网络配置
            "concurrency": 10,
            "max_qps": None,
            "timeout": 120,
            "retry_times": 3,
            "retry_delay": 1.0,
            # 采样参数
            "top_p": None,
            "top_k": None,
            # 思考模式
            "thinking": None,
            # 处理配置
            "preprocess_msg": False,
            "flush_interval": 1.0,
            # 输出配置
            "return_usage": True,
            "track_cost": True,
            # 多 endpoint 配置
            "endpoints": None,
            "fallback": True,
        }

        # 从配置文件读取 batch 配置节
        user_batch_config = self.config.get("batch", {})

        # 合并配置（用户配置覆盖默认值）
        result = {**defaults}
        for key in defaults:
            if key in user_batch_config:
                result[key] = user_batch_config[key]

        return result


# 全局配置实例
_config: FlexLLMConfig | None = None


def get_config() -> FlexLLMConfig:
    global _config
    if _config is None:
        _config = FlexLLMConfig()
    return _config


def apply_user_template(content: str, template: str | None) -> str:
    """应用 user content 模板

    Args:
        content: 原始 user content
        template: 模板字符串，使用 {content} 作为占位符

    Returns:
        应用模板后的内容，如果没有模板则返回原内容
    """
    if not template:
        return content
    return template.format(content=content)


# ========== 输入格式处理 ==========


def detect_input_format(record: dict) -> tuple[str, list[str]]:
    """检测输入记录的格式类型

    支持的格式及优先级:
    1. openai_chat: 包含 "messages" 字段
    2. alpaca: 包含 "instruction" 字段（可选 "input"）
    3. simple: 包含 q/question/prompt/input/user 之一（可选 "system"）
       注意: "input" 仅在没有 "instruction" 时作为 simple 格式的 user content
    """
    if "messages" in record:
        return "openai_chat", ["messages"]
    if "instruction" in record:
        return "alpaca", ["instruction", "input"]
    for field in ["q", "question", "prompt", "input", "user"]:
        if field in record:
            return "simple", [field, "system"]
    return "unknown", []


def convert_to_messages(
    record: dict,
    format_type: str,
    message_fields: list[str],
    global_system: str = None,
    user_template: str = None,
) -> tuple[list[dict], dict]:
    """将输入记录转换为 messages 格式"""
    messages = []
    used_fields = set()

    if format_type == "openai_chat":
        messages = record["messages"]
        used_fields.add("messages")
        # 对于 openai_chat 格式，应用 user_template 到所有 user role 消息
        if user_template:
            messages = [
                {**msg, "content": apply_user_template(msg["content"], user_template)}
                if msg.get("role") == "user" and isinstance(msg.get("content"), str)
                else msg
                for msg in messages
            ]

    elif format_type == "alpaca":
        instruction = record.get("instruction", "")
        input_text = record.get("input", "")
        used_fields.update(["instruction", "input", "output"])

        system = record.get("system")
        if system:
            used_fields.add("system")
            messages.append({"role": "system", "content": system})

        content = instruction
        if input_text:
            content = f"{instruction}\n\n{input_text}"
        # 应用 user_template
        content = apply_user_template(content, user_template)
        messages.append({"role": "user", "content": content})

    elif format_type == "simple":
        prompt_field = None
        for field in ["q", "question", "prompt", "input", "user"]:
            if field in record:
                prompt_field = field
                break

        if prompt_field:
            used_fields.add(prompt_field)
            system = global_system or record.get("system")
            if "system" in record:
                used_fields.add("system")

            if system:
                messages.append({"role": "system", "content": system})
            # 应用 user_template
            user_content = apply_user_template(record[prompt_field], user_template)
            messages.append({"role": "user", "content": user_content})

    elif format_type == "custom":
        # message_fields = [user_field, system_field]（由 --user-field/--system-field 指定）
        user_field, system_field = message_fields[0], message_fields[1]
        used_fields.add(user_field)

        system = None
        if system_field and system_field in record:
            system = record[system_field]
            used_fields.add(system_field)

        if system:
            messages.append({"role": "system", "content": system})
        # 应用 user_template
        user_content = apply_user_template(record[user_field], user_template)
        messages.append({"role": "user", "content": user_content})

    if global_system and format_type != "openai_chat":
        messages = [m for m in messages if m.get("role") != "system"]
        messages.insert(0, {"role": "system", "content": global_system})

    metadata = {k: v for k, v in record.items() if k not in used_fields}
    return messages, metadata


def load_jsonl_records(input_path: str = None) -> list[dict]:
    """从 JSONL 文件或 stdin 加载记录"""
    records = []

    if input_path:
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        for line in sys.stdin:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError("输入为空")

    return records


def parse_batch_input(
    input_path: str = None, skip_format_detection: bool = False
) -> tuple[list[dict], str, list[str]]:
    """解析批量输入文件或 stdin

    Args:
        input_path: 输入文件路径，None 表示从 stdin 读取
        skip_format_detection: 是否跳过格式检测（使用 --user-field 时应跳过）
    """
    records = load_jsonl_records(input_path)

    if skip_format_detection:
        # 跳过格式检测，返回空的 format_type 和 message_fields
        return records, "", []

    format_type, message_fields = detect_input_format(records[0])

    if format_type == "unknown":
        available_fields = list(records[0].keys())
        raise ValueError(
            f"无法识别输入格式，未找到以下字段之一：\n"
            f"  - messages (openai_chat 格式)\n"
            f"  - instruction (alpaca 格式)\n"
            f"  - q/question/prompt/input/user (simple 格式)\n\n"
            f"发现的字段: {available_fields}\n"
            f"提示: 使用 dtflow 转换格式: dt transform data.jsonl --preset=openai_chat"
        )

    return records, format_type, message_fields


# ========== CLI 命令 ==========

if HAS_TYPER:

    @app.command()
    def ask(
        prompt: Annotated[str | None, Argument(help="用户问题")] = None,
        system: Annotated[str | None, Option("-s", "--system", help="系统提示词")] = None,
        model: Annotated[str | None, Option("-m", "--model", help="模型名称")] = None,
        user_template: Annotated[
            str | None, Option("--user-template", help="user content 模板 (使用 {content} 占位符)")
        ] = None,
    ):
        """LLM 快速问答（支持管道输入）

        Examples:
            flexllm ask "什么是Python"
            flexllm ask "解释代码" -s "你是代码专家"
            echo "长文本" | flexllm ask "总结一下"

        临时使用未配置的服务:
            FLEXLLM_BASE_URL="http://localhost:8000/v1" FLEXLLM_MODEL="qwen" flexllm ask "你好"
        """
        stdin_content = None
        if not sys.stdin.isatty():
            stdin_content = sys.stdin.read().strip()

        if not prompt and not stdin_content:
            print("错误: 请提供问题", file=sys.stderr)
            raise typer.Exit(1)

        if stdin_content:
            full_prompt = f"{stdin_content}\n\n{prompt}" if prompt else stdin_content
        else:
            full_prompt = prompt

        config = get_config()
        model_config = config.get_model_config(model)
        if not model_config:
            print("错误: 未找到模型配置，使用 'flexllm list' 查看可用模型", file=sys.stderr)
            print(
                "提示: 设置环境变量 FLEXLLM_BASE_URL, FLEXLLM_API_KEY, FLEXLLM_MODEL 或创建 ~/.flexllm/config.yaml",
                file=sys.stderr,
            )
            raise typer.Exit(1)

        model_id = model_config.get("id")
        base_url = model_config.get("base_url")
        api_key = model_config.get("api_key", "EMPTY")

        # 获取系统提示词：命令行参数 > 配置文件
        if not system:
            system = config.get_system(model)

        # 获取 user_template：命令行参数 > 配置文件
        if not user_template:
            user_template = config.get_user_template(model)

        async def _ask():
            from flexllm import LLMClient

            async with LLMClient(model=model_id, base_url=base_url, api_key=api_key) as client:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                # 应用 user_template
                user_content = apply_user_template(full_prompt, user_template)
                messages.append({"role": "user", "content": user_content})
                return await client.chat_completions(messages)

        try:
            result = asyncio.run(_ask())
            if result is None:
                return
            if isinstance(result, str):
                print(result)
                return
            if hasattr(result, "status") and result.status == "error":
                error_msg = result.data.get("detail", result.data.get("error", "未知错误"))
                print(f"错误: {error_msg}", file=sys.stderr)
                return
            print(str(result))
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            raise typer.Exit(1)

    @app.command()
    def chat(
        message: Annotated[str | None, Argument(help="单条消息（不提供则进入多轮对话）")] = None,
        model: Annotated[str | None, Option("-m", "--model", help="模型名称")] = None,
        base_url: Annotated[str | None, Option("--base-url", help="API 地址")] = None,
        api_key: Annotated[str | None, Option("--api-key", help="API 密钥")] = None,
        system_prompt: Annotated[str | None, Option("-s", "--system", help="系统提示词")] = None,
        temperature: Annotated[float, Option("-t", "--temperature", help="采样温度")] = 0.7,
        max_tokens: Annotated[int, Option("--max-tokens", help="最大生成 token 数")] = 2048,
        no_stream: Annotated[bool, Option("--no-stream", help="禁用流式输出")] = False,
        user_template: Annotated[
            str | None, Option("--user-template", help="user content 模板 (使用 {content} 占位符)")
        ] = None,
    ):
        """交互式对话

        Examples:
            flexllm chat                      # 多轮对话
            flexllm chat "你好"               # 单条对话
            flexllm chat --model gpt-4 "你好" # 指定模型
        """
        config = get_config()
        model_config = config.get_model_config(model)
        if model_config:
            model = model or model_config.get("id")
            base_url = base_url or model_config.get("base_url")
            api_key = api_key or model_config.get("api_key", "EMPTY")

        if not base_url:
            print("错误: 未配置 base_url", file=sys.stderr)
            raise typer.Exit(1)

        # 获取系统提示词：命令行参数 > 配置文件
        if not system_prompt:
            system_prompt = config.get_system(model)

        # 获取 user_template：命令行参数 > 配置文件
        if not user_template:
            user_template = config.get_user_template(model)

        stream = not no_stream

        if message:
            _single_chat(
                message,
                model,
                base_url,
                api_key,
                system_prompt,
                temperature,
                max_tokens,
                stream,
                user_template,
            )
        else:
            _interactive_chat(
                model,
                base_url,
                api_key,
                system_prompt,
                temperature,
                max_tokens,
                stream,
                user_template,
            )

    @app.command(name="chat-web")
    def chat_web(
        model: Annotated[str | None, Option("-m", "--model", help="模型名称")] = None,
        base_url: Annotated[str | None, Option("--base-url", help="API 地址")] = None,
        api_key: Annotated[str | None, Option("--api-key", help="API 密钥")] = None,
        system_prompt: Annotated[str | None, Option("-s", "--system", help="系统提示词")] = None,
        temperature: Annotated[float, Option("-t", "--temperature", help="采样温度")] = 0.7,
        max_tokens: Annotated[int, Option("--max-tokens", help="最大生成 token 数")] = 2048,
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
    ):
        """启动 Web 聊天界面

        Examples:
            flexllm chat-web                      # 使用默认模型
            flexllm chat-web -m gpt-4             # 指定模型
            flexllm chat-web -p 9090              # 指定端口
            flexllm chat-web --host 0.0.0.0       # 允许外部访问
            flexllm chat-web --thinking true      # 启用思考模式
        """
        config = get_config()
        model_config = config.get_model_config(model)
        if model_config:
            model = model or model_config.get("id")
            base_url = base_url or model_config.get("base_url")
            api_key = api_key or model_config.get("api_key", "EMPTY")

        if not base_url:
            print("错误: 未配置 base_url", file=sys.stderr)
            raise typer.Exit(1)

        if not system_prompt:
            system_prompt = config.get_system(model)
        if not user_template:
            user_template = config.get_user_template(model)

        try:
            from .chat_web import ChatWebConfig, ChatWebServer
        except ImportError:
            print("错误: 需要安装 aiohttp: pip install aiohttp", file=sys.stderr)
            raise typer.Exit(1)

        thinking_value = _parse_thinking(thinking)

        web_config = ChatWebConfig(
            port=port,
            host=host,
            model=model,
            base_url=base_url,
            api_key=api_key,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            user_template=user_template,
            thinking=thinking_value,
        )

        print(f"flexllm Chat Web starting on http://{host}:{port}")
        print(f"  Model: {model}")
        print(f"  Server: {base_url}")
        print(f"  Temperature: {temperature}")
        if thinking_value is not None:
            print(f"  Thinking: {thinking_value}")
        print("\nPress Ctrl+C to stop")

        try:
            server = ChatWebServer(web_config)
            server.run()
        except KeyboardInterrupt:
            print("\nServer stopped")
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            raise typer.Exit(1)

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
    ):
        """启动 HTTP API 服务，将 LLM 包装为 REST API

        适用于微调模型部署：固定 system prompt 和 user template，
        调用方只需发送 content 文本，返回解析后的 thinking 和 content。

        API 端点:
            POST /api/generate         非流式生成
            POST /api/generate/stream  流式生成 (SSE)
            POST /api/generate/batch   批量生成
            GET  /health               健康检查
            GET  /api/config           查看当前配置

        Examples:
            flexllm serve -m qwen-finetuned -s "你是助手"
            flexllm serve --thinking true -c 20 -p 8000
            flexllm serve --user-template "[INST]{content}[/INST]"
        """
        config = get_config()
        model_config = config.get_model_config(model)
        if model_config:
            model = model or model_config.get("id")
            base_url = base_url or model_config.get("base_url")
            api_key = api_key or model_config.get("api_key", "EMPTY")

        if not base_url:
            print("错误: 未配置 base_url", file=sys.stderr)
            raise typer.Exit(1)

        if not system_prompt:
            system_prompt = config.get_system(model)
        if not user_template:
            user_template = config.get_user_template(model)

        try:
            from .serve import ServeConfig, ServeServer
        except ImportError:
            print("错误: 需要安装 aiohttp: pip install aiohttp", file=sys.stderr)
            raise typer.Exit(1)

        thinking_value = _parse_thinking(thinking)

        serve_config = ServeConfig(
            port=port,
            host=host,
            model=model,
            base_url=base_url,
            api_key=api_key,
            system_prompt=system_prompt,
            user_template=user_template,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking_value,
            concurrency=concurrency,
            max_qps=max_qps,
            timeout=timeout,
            verbose=verbose,
        )

        print(f"flexllm Serve starting on http://{host}:{port}")
        print(f"  Model: {model}")
        print(f"  Server: {base_url}")
        if temperature is not None:
            print(f"  Temperature: {temperature}")
        if max_tokens is not None:
            print(f"  Max tokens: {max_tokens}")
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
        print(f"\n  POST /api/generate         非流式生成")
        print(f"  POST /api/generate/stream  流式生成")
        print(f"  POST /api/generate/batch   批量生成")
        print(f"  GET  /health               健康检查")
        print(f"  GET  /api/config           查看配置")
        if verbose:
            print(f"  Verbose: on (请求日志已开启)")
        print("\nPress Ctrl+C to stop")

        try:
            server = ServeServer(serve_config)
            server.run()
        except KeyboardInterrupt:
            print("\nServer stopped")
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            raise typer.Exit(1)

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
        # 新增 CLI 快捷选项
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
    ):
        """批量处理 JSONL 文件（支持断点续传）

        自动检测输入格式：openai_chat, alpaca, simple (q/question/prompt/input/user)
        也可用 --user-field 和 --system-field 指定任意字段名。

        高级配置可在 ~/.flexllm/config.yaml 的 batch 节中设置。
        CLI 参数优先级高于配置文件。

        Examples:
            flexllm batch input.jsonl                  # 自动生成 input.output.jsonl
            flexllm batch input.jsonl -o output.jsonl  # 指定输出文件
            flexllm batch input.jsonl -c 20 -m gpt-4   # 自动输出 + 自定义参数
            flexllm batch input.jsonl --cache --return-usage
            flexllm batch data.jsonl -o out.jsonl --user-field text --system-field sys_prompt
            cat input.jsonl | flexllm batch -o output.jsonl  # stdin 需指定 -o
        """
        has_stdin = not sys.stdin.isatty()
        if not input and not has_stdin:
            print("错误: 请提供输入文件或通过管道传入数据", file=sys.stderr)
            raise typer.Exit(1)

        # 自动生成输出文件名
        auto_generated_output = False
        if not output:
            if not input:
                # 从 stdin 读取时必须指定输出文件
                print(
                    "错误: 从 stdin 读取数据时必须指定输出文件 (-o output.jsonl)", file=sys.stderr
                )
                raise typer.Exit(1)

            # 根据输入文件名自动生成输出文件名
            input_path = Path(input)
            stem = input_path.stem  # 文件名（不含扩展名）
            output = str(input_path.parent / f"{stem}.output.jsonl")
            auto_generated_output = True

        if not output.endswith(".jsonl"):
            print(f"错误: 输出文件必须使用 .jsonl 扩展名，当前: {output}", file=sys.stderr)
            raise typer.Exit(1)

        config = get_config()

        # 获取 batch 配置（配置文件 + 默认值）
        batch_config = config.get_batch_config()

        # 优先级：命令行 -m 参数 > batch.endpoints 配置 > 默认模型
        model_config = None
        endpoints_config = None
        use_pool = False

        if model:
            # 命令行指定了 -m，使用指定的模型配置
            model_config = config.get_model_config(model)
            if not model_config:
                print(f"错误: 未找到模型 '{model}'", file=sys.stderr)
                print("提示: 使用 'flexllm list' 查看可用模型", file=sys.stderr)
                raise typer.Exit(1)
        elif batch_config.get("endpoints"):
            # 没有指定 -m，使用 batch.endpoints 配置
            endpoints_config = batch_config["endpoints"]
            use_pool = len(endpoints_config) > 0
        else:
            # 都没有，使用默认模型
            model_config = config.get_model_config(None)
            if not model_config:
                print("错误: 未找到模型配置", file=sys.stderr)
                print(
                    "提示: 使用 'flexllm list' 查看可用模型，或在 batch 节配置 endpoints",
                    file=sys.stderr,
                )
                raise typer.Exit(1)

        model_id = model_config.get("id") if model_config else None
        base_url = model_config.get("base_url") if model_config else None
        api_key = model_config.get("api_key", "EMPTY") if model_config else None

        # CLI 参数覆盖配置文件
        effective_cache = cache if cache is not None else batch_config["cache"]
        effective_return_usage = return_usage or batch_config["return_usage"]
        effective_preprocess_msg = preprocess_msg or batch_config["preprocess_msg"]
        effective_track_cost = track_cost or batch_config["track_cost"]
        effective_concurrency = (
            concurrency if concurrency is not None else batch_config["concurrency"]
        )
        effective_max_qps = max_qps if max_qps is not None else batch_config["max_qps"]

        # 系统提示词：CLI 参数 > 配置文件
        effective_system = system if system is not None else config.get_system(model)

        # user_template：CLI 参数 > 配置文件
        effective_user_template = (
            user_template if user_template is not None else config.get_user_template(model)
        )

        # 解析 save_input: CLI 字符串 -> bool | str
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
                print(
                    f"错误: --save-input 仅支持 true/last/false，当前: {save_input}",
                    file=sys.stderr,
                )
                raise typer.Exit(1)

        try:
            if user_field:
                # 指定了 --user-field，跳过自动格式检测
                records, _, _ = parse_batch_input(input, skip_format_detection=True)
                format_type = "custom"
                message_fields = [user_field, system_field]
                # 校验首条记录是否包含指定字段
                if user_field not in records[0]:
                    available = list(records[0].keys())
                    print(
                        f"错误: 字段 '{user_field}' 不存在，可用字段: {available}",
                        file=sys.stderr,
                    )
                    raise typer.Exit(1)
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

            # 显示使用的客户端类型
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

            async def _run_batch():
                from flexllm import LLMClient, LLMClientPool

                from .cache import ResponseCacheConfig

                # 构建缓存配置
                cache_config = None
                if effective_cache:
                    cache_config = ResponseCacheConfig.ipc(ttl=batch_config["cache_ttl"])

                # 构建 chat_completions_batch 的 kwargs
                kwargs = {}
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                # 从配置文件读取采样参数
                if batch_config["top_p"] is not None:
                    kwargs["top_p"] = batch_config["top_p"]
                if batch_config["top_k"] is not None:
                    kwargs["top_k"] = batch_config["top_k"]
                if batch_config["thinking"] is not None:
                    kwargs["thinking"] = batch_config["thinking"]

                # 使用 batch.endpoints 配置或单 model 配置
                if use_pool:
                    # 多 endpoint 模式：使用 LLMClientPool
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
                    # 单 endpoint 模式：使用 LLMClient
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
            print(f"错误: JSON 解析失败 - {e}", file=sys.stderr)
            raise typer.Exit(1)
        except ValueError as e:
            print(f"错误: {e}", file=sys.stderr)
            raise typer.Exit(1)
        except FileNotFoundError:
            print(f"错误: 文件不存在 - {input}", file=sys.stderr)
            raise typer.Exit(1)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"错误: {e}", file=sys.stderr)
            raise typer.Exit(1)

    @app.command()
    def models(
        base_url: Annotated[str | None, Option("--base-url", help="API 地址")] = None,
        api_key: Annotated[str | None, Option("--api-key", help="API 密钥")] = None,
        name: Annotated[str | None, Option("-n", "--name", help="模型配置名称")] = None,
    ):
        """列出远程服务器上的可用模型"""
        import requests

        config = get_config()
        model_config = config.get_model_config(name)
        if model_config:
            base_url = base_url or model_config.get("base_url")
            api_key = api_key or model_config.get("api_key", "EMPTY")
            provider = model_config.get("provider", "openai")
        else:
            provider = "openai"

        if not base_url:
            print("错误: 未配置 base_url", file=sys.stderr)
            raise typer.Exit(1)

        is_gemini = provider == "gemini" or "generativelanguage.googleapis.com" in base_url

        try:
            if is_gemini:
                url = f"{base_url.rstrip('/')}/models?key={api_key}"
                response = requests.get(url, timeout=10)
            else:
                headers = {"Authorization": f"Bearer {api_key}"}
                response = requests.get(
                    f"{base_url.rstrip('/')}/models", headers=headers, timeout=10
                )

            if response.status_code == 200:
                models_data = response.json()

                print("\n可用模型列表")
                print(f"服务器: {base_url}")
                print("-" * 50)

                if is_gemini:
                    models_list = models_data.get("models", [])
                    if models_list:
                        for i, m in enumerate(models_list, 1):
                            model_name = m.get("name", "").replace("models/", "")
                            print(f"  {i:2d}. {model_name}")
                        print(f"\n共 {len(models_list)} 个模型")
                    else:
                        print("未找到可用模型")
                else:
                    if isinstance(models_data, dict) and "data" in models_data:
                        models_list = models_data["data"]
                    elif isinstance(models_data, list):
                        models_list = models_data
                    else:
                        models_list = []

                    if models_list:
                        for i, m in enumerate(models_list, 1):
                            if isinstance(m, dict):
                                model_id = m.get("id", m.get("name", "unknown"))
                                print(f"  {i:2d}. {model_id}")
                            else:
                                print(f"  {i:2d}. {m}")
                        print(f"\n共 {len(models_list)} 个模型")
                    else:
                        print("未找到可用模型")
            else:
                print(f"错误: HTTP {response.status_code}", file=sys.stderr)
                raise typer.Exit(1)

        except requests.exceptions.RequestException as e:
            print(f"连接失败: {e}", file=sys.stderr)
            raise typer.Exit(1)
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            raise typer.Exit(1)

    @app.command("list")
    def list_models():
        """列出本地配置的模型"""
        config = get_config()
        models = config.config.get("models", [])
        default = config.config.get("default", "")

        if not models:
            print("未配置模型")
            print("提示: 创建 ~/.flexllm/config.yaml 或设置环境变量")
            return

        print(f"已配置模型 (共 {len(models)} 个):\n")
        for m in models:
            name = m.get("name", m.get("id", "?"))
            model_id = m.get("id", "?")
            provider = m.get("provider", "openai")
            is_default = " (默认)" if name == default or model_id == default else ""
            endpoints = m.get("endpoints")

            print(f"  {name}{is_default}")
            if name != model_id:
                print(f"    id: {model_id}")

            if endpoints and len(endpoints) > 1:
                # 多 endpoint 池
                print(f"    type: pool ({len(endpoints)} endpoints)")
                print(f"    fallback: {m.get('fallback', True)}")
            else:
                print(f"    provider: {provider}")
            print()

    @app.command("set-model")
    def set_model(
        model_name: Annotated[str, Argument(help="模型名称或 ID")],
    ):
        """设置默认模型

        Examples:
            flexllm set-model gpt-4
            flexllm set-model local-ollama
        """
        config = get_config()
        config_path = config.get_config_path()

        if not config_path:
            print("错误: 未找到配置文件", file=sys.stderr)
            print("提示: 先运行 'flexllm init' 初始化配置文件", file=sys.stderr)
            raise typer.Exit(1)

        model_config = config.get_model_config(model_name)
        if not model_config:
            print(f"错误: 未找到模型 '{model_name}'", file=sys.stderr)
            print("提示: 使用 'flexllm list' 查看已配置的模型", file=sys.stderr)
            raise typer.Exit(1)

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
            print("错误: 需要安装 pyyaml: pip install pyyaml", file=sys.stderr)
            raise typer.Exit(1)
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            raise typer.Exit(1)

    @app.command()
    def test(
        model: Annotated[str | None, Option("-m", "--model", help="模型名称")] = None,
        base_url: Annotated[str | None, Option("--base-url", help="API 地址")] = None,
        api_key: Annotated[str | None, Option("--api-key", help="API 密钥")] = None,
        message: Annotated[
            str, Option("--message", help="测试消息")
        ] = "Hello, please respond with 'OK' if you can see this message.",
        timeout: Annotated[int, Option("--timeout", help="超时时间（秒）")] = 30,
    ):
        """测试 LLM 服务连接"""
        import time

        import requests

        config = get_config()
        model_config = config.get_model_config(model)
        if model_config:
            model = model or model_config.get("id")
            base_url = base_url or model_config.get("base_url")
            api_key = api_key or model_config.get("api_key", "EMPTY")

        if not base_url:
            print("错误: 未配置 base_url", file=sys.stderr)
            raise typer.Exit(1)

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
                print(f"   ✓ 连接成功 ({elapsed:.2f}s)")
                models_data = response.json()
                if isinstance(models_data, dict) and "data" in models_data:
                    model_count = len(models_data["data"])
                elif isinstance(models_data, list):
                    model_count = len(models_data)
                else:
                    model_count = 0
                print(f"   可用模型数: {model_count}")
            else:
                print(f"   ✗ 连接失败: HTTP {response.status_code}")
                raise typer.Exit(1)
        except Exception as e:
            print(f"   ✗ 连接失败: {e}")
            raise typer.Exit(1)

        if model:
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
                    print(f"   ✓ 调用成功 ({elapsed:.2f}s)")
                    print(f"   响应: {content[:100]}...")
                else:
                    print(f"   ✗ 调用失败: HTTP {response.status_code}")
                    print(f"   {response.text[:200]}")
            except Exception as e:
                print(f"   ✗ 调用失败: {e}")

        print("\n测试完成")

    @app.command()
    def init(
        path: Annotated[str | None, Option("-p", "--path", help="配置文件路径")] = None,
    ):
        """初始化配置文件"""
        if path is None:
            config_path = Path.home() / ".flexllm" / "config.yaml"
        else:
            config_path = Path(path)

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
#   # 网络配置（全局默认值，可被 endpoint 级别配置覆盖）
#   concurrency: 10           # 并发数
#   max_qps: 100              # 每秒最大请求数
#   timeout: 120              # 请求超时时间（秒）
#   retry_times: 3            # 重试次数
#   retry_delay: 1.0          # 重试延迟（秒）
#
#   # 缓存配置
#   cache: false              # 是否启用响应缓存
#   cache_ttl: 86400          # 缓存过期时间（秒），默认 24 小时
#
#   # 采样参数（覆盖模型默认值）
#   # top_p: 0.9
#   # top_k: 50
#
#   # 思考模式（适用于支持的模型如 DeepSeek-R1）
#   # thinking: true          # 或 "minimal"/"low"/"medium"/"high"
#
#   # 提示词配置
#   # system: "You are a batch processing assistant."  # batch 级别 system prompt
#   # user_template: "[INST]{content}[/INST]"          # batch 级别 user template
#
#   # 处理配置
#   preprocess_msg: false     # 是否预处理图片消息（URL 转 base64）
#   flush_interval: 1.0       # 文件刷新间隔（秒）
#
#   # 输出配置
#   return_usage: false       # 是否输出 token 统计
#
#   # 多 endpoint 配置（配置后 batch 命令自动使用 LLMClientPool）
#   # 动态负载均衡：共享队列，快的 endpoint 处理更多任务
#   # endpoints:
#   #   - base_url: http://fast-api.com/v1
#   #     api_key: key1
#   #     model: qwen
#   #     concurrency_limit: 50   # endpoint 级别并发（可选）
#   #     max_qps: 500            # endpoint 级别 QPS（可选）
#   #   - base_url: http://slow-api.com/v1
#   #     api_key: key2
#   #     model: qwen
#   #     concurrency_limit: 5    # 较慢的服务使用更低的并发
#   #     max_qps: 50
#   # fallback: true            # 失败时自动切换到其他 endpoint
"""

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(default_config)
            print(f"已创建配置文件: {config_path}")
            print("请编辑配置文件填入 API 密钥")
        except Exception as e:
            print(f"创建失败: {e}", file=sys.stderr)
            raise typer.Exit(1)

    @app.command()
    def pricing(
        model: Annotated[str | None, Argument(help="模型名称（支持模糊匹配）")] = None,
        update: Annotated[bool, Option("--update", help="从 OpenRouter 更新定价表")] = False,
        json_output: Annotated[bool, Option("--json", help="输出 JSON 格式")] = False,
    ):
        """查询模型定价信息

        Examples:
            flexllm pricing                  # 列出所有模型定价
            flexllm pricing gpt-4o           # 查询 gpt-4o 定价
            flexllm pricing claude           # 模糊匹配 claude 相关模型
            flexllm pricing --update         # 从 OpenRouter 更新定价表
        """
        from .pricing import get_pricing, reload_pricing

        MODEL_PRICING = get_pricing()

        if update:
            # 调用更新脚本
            print("正在从 OpenRouter API 获取最新定价...")
            try:
                from .pricing.updater import collect_pricing, update_pricing_file

                pricing_map = collect_pricing()
                print(f"获取到 {len(pricing_map)} 个模型定价")

                if update_pricing_file(pricing_map):
                    reload_pricing()  # 重新加载定价数据
                    print("✓ data.json 已更新")
                else:
                    print("✗ 更新失败", file=sys.stderr)
                    raise typer.Exit(1)
            except Exception as e:
                print(f"更新失败: {e}", file=sys.stderr)
                raise typer.Exit(1)
            return

        # 查询定价
        if model:
            # 模糊匹配
            matches = {
                name: price
                for name, price in MODEL_PRICING.items()
                if model.lower() in name.lower()
            }

            if not matches:
                print(f"未找到匹配 '{model}' 的模型", file=sys.stderr)
                print(
                    f"\n可用模型: {', '.join(sorted(MODEL_PRICING.keys())[:10])}...",
                    file=sys.stderr,
                )
                raise typer.Exit(1)

            if json_output:
                import json as json_module

                output = {
                    name: {
                        "input_per_1m": round(p["input"] * 1e6, 4),
                        "output_per_1m": round(p["output"] * 1e6, 4),
                    }
                    for name, p in sorted(matches.items())
                }
                print(json_module.dumps(output, indent=2, ensure_ascii=False))
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
            # 列出所有模型
            if json_output:
                import json as json_module

                output = {
                    name: {
                        "input_per_1m": round(p["input"] * 1e6, 4),
                        "output_per_1m": round(p["output"] * 1e6, 4),
                    }
                    for name, p in sorted(MODEL_PRICING.items())
                }
                print(json_module.dumps(output, indent=2, ensure_ascii=False))
            else:
                # 按厂商分组显示
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
                    models = groups[group_name]
                    print(f"\n[{group_name}]")
                    for name, p in sorted(models):
                        input_price = p["input"] * 1e6
                        output_price = p["output"] * 1e6
                        print(f"  {name:<28} ${input_price:<14.4f} ${output_price:<14.4f}")

    @app.command()
    def credits(
        model: Annotated[str | None, Option("-m", "--model", help="模型名称")] = None,
    ):
        """查询 API Key 余额

        自动根据 base_url 识别 provider 并查询余额。

        支持的 provider:
          - OpenRouter (openrouter.ai)
          - SiliconFlow (siliconflow.cn)
          - DeepSeek (deepseek.com)
          - AI/ML API (aimlapi.com)
          - OpenAI (api.openai.com) - 非官方 API，可能不稳定

        不支持的 provider:
          - Anthropic: 需要 Admin API key (sk-ant-admin...)
          - xAI: 需要单独的 Management API key
          - Together AI/Groq/Mistral: 无公开余额查询 API

        Examples:
            flexllm credits                # 查询默认模型的 key 余额
            flexllm credits -m grok-4      # 查询指定模型的 key 余额
        """
        config = get_config()
        model_config = config.get_model_config(model)

        if not model_config:
            print("错误: 未找到模型配置", file=sys.stderr)
            print("提示: 使用 'flexllm list' 查看已配置的模型", file=sys.stderr)
            raise typer.Exit(1)

        base_url = model_config.get("base_url", "")
        api_key = model_config.get("api_key", "")
        model_name = model_config.get("name", model_config.get("id", "unknown"))

        if not api_key or api_key == "EMPTY":
            print(f"错误: 模型 '{model_name}' 未配置 API Key", file=sys.stderr)
            raise typer.Exit(1)

        result = _query_credits(base_url, api_key)

        if result is None:
            print("错误: 不支持查询此 provider 的余额", file=sys.stderr)
            print(f"  base_url: {base_url}", file=sys.stderr)
            print("\n支持的 provider:", file=sys.stderr)
            print("  - OpenRouter (openrouter.ai)", file=sys.stderr)
            print("  - SiliconFlow (siliconflow.cn)", file=sys.stderr)
            print("  - DeepSeek (deepseek.com)", file=sys.stderr)
            print("  - AI/ML API (aimlapi.com)", file=sys.stderr)
            print("  - OpenAI (api.openai.com)", file=sys.stderr)
            print("\n不支持的 provider:", file=sys.stderr)
            print("  - Anthropic: 需要 Admin API key", file=sys.stderr)
            print("  - xAI: 需要单独的 Management API key", file=sys.stderr)
            print("  - Together AI/Groq/Mistral: 无公开余额查询 API", file=sys.stderr)
            raise typer.Exit(1)

        if "error" in result:
            print(f"错误: {result['error']}", file=sys.stderr)
            raise typer.Exit(1)

        # 格式化输出
        print(f"\n{result['provider']} 账户余额")
        print(f"模型配置: {model_name}")
        print(f"API Key: {api_key[:15]}...{api_key[-4:]}")
        print("-" * 40)

        for key, value in result["data"].items():
            print(f"  {key}: {value}")

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
    ):
        """启动 Mock LLM 服务器

        提供一个轻量级的 Mock 服务器，用于测试和开发。
        支持 OpenAI / Claude / Gemini 三种 API 格式，支持流式和非流式。

        Examples:
            flexllm mock                          # 默认配置，端口 8001
            flexllm mock -p 8080                  # 指定端口
            flexllm mock -d 0.5                   # 固定延迟 0.5s
            flexllm mock -d 1-5                   # 随机延迟 1-5s
            flexllm mock -l 100-500               # 响应长度 100-500 字符
            flexllm mock --rps 10                 # 每秒最多 10 个请求
            flexllm mock --token-rate 50          # 流式返回每秒 50 个 token
            flexllm mock --error-rate 0.5         # 50% 请求返回错误
            flexllm mock --thinking               # 响应包含思考内容

        API 端点:
            OpenAI: POST /v1/chat/completions
            Claude: POST /v1/messages
            Gemini: POST /models/{model}:generateContent

        测试:
            # OpenAI
            curl http://localhost:8001/v1/chat/completions \\
              -d '{"model": "mock", "messages": [{"role": "user", "content": "hello"}]}'

            # Claude
            curl http://localhost:8001/v1/messages \\
              -d '{"model": "mock", "max_tokens": 1024, "messages": [{"role": "user", "content": "hello"}]}'

            # Gemini
            curl http://localhost:8001/models/mock:generateContent \\
              -d '{"contents": [{"parts": [{"text": "hello"}]}]}'
        """
        try:
            from .mock import MockLLMServer, MockServerConfig, parse_range
        except ImportError:
            print("错误: 需要安装 aiohttp: pip install aiohttp", file=sys.stderr)
            raise typer.Exit(1)

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
        print(f"  OpenAI: http://localhost:{port}/v1/chat/completions")
        print(f"  Claude: http://localhost:{port}/v1/messages")
        print(f"  Gemini: http://localhost:{port}/models/{{model}}:generateContent")
        print("\nPress Ctrl+C to stop")

        try:
            server = MockLLMServer(config)
            server.run()
        except KeyboardInterrupt:
            print("\nServer stopped")
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            raise typer.Exit(1)

    @app.command()
    def version():
        """显示版本信息"""
        try:
            from flexllm import __version__

            v = __version__
        except Exception:
            v = "0.1.0"
        print(f"flexllm {v}")

    @app.command("install-skill")
    def install_skill():
        """安装 Claude Code skill 文件

        将 flexllm 的 skill 文件安装到 ~/.claude/skills/flexllm/，
        使 Claude Code 能够获取 flexllm 的使用文档。
        """
        import shutil
        from pathlib import Path

        # 查找 skill 文件（在包的 data 目录下）
        skill_src = Path(__file__).parent / "data" / "SKILL.md"

        if not skill_src.exists():
            print("错误: 找不到 skill 文件", file=sys.stderr)
            print("请尝试重新安装 flexllm: pip install --force-reinstall flexllm", file=sys.stderr)
            raise typer.Exit(1)

        # 目标路径
        skill_dir = Path.home() / ".claude" / "skills" / "flexllm"
        skill_dst = skill_dir / "SKILL.md"

        try:
            skill_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(skill_src, skill_dst)
            print(f"已安装 skill 文件到: {skill_dst}")
            print("Claude Code 现在可以使用 flexllm skill 了")
        except Exception as e:
            print(f"安装失败: {e}", file=sys.stderr)
            raise typer.Exit(1)


# ========== 辅助函数 ==========


def _parse_thinking(value: str | None) -> bool | str | int | None:
    """解析 --thinking 参数值

    支持: true/false/low/medium/high/minimal 或整数(budget_tokens)
    """
    if value is None:
        return None
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("low", "medium", "high", "minimal"):
        return low
    try:
        return int(value)
    except ValueError:
        print(f"错误: --thinking 参数无效: {value}", file=sys.stderr)
        raise SystemExit(1)


def _query_credits(base_url: str, api_key: str) -> dict | None:
    """查询 API Key 余额

    Args:
        base_url: API 基础 URL
        api_key: API 密钥

    Returns:
        dict: {"provider": str, "data": dict} 或 {"error": str}
        None: 不支持的 provider
    """
    import requests

    headers = {"Authorization": f"Bearer {api_key}"}
    timeout = 15

    try:
        # OpenRouter
        if "openrouter.ai" in base_url:
            resp = requests.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers=headers,
                timeout=timeout,
            )
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

            data = resp.json().get("data", {})
            return {
                "provider": "OpenRouter",
                "data": {
                    "剩余额度": f"${data.get('limit_remaining', 0):.2f}",
                    "总额度上限": f"${data.get('limit', 0):.2f}",
                    "已使用": f"${data.get('usage', 0):.2f}",
                    "今日消费": f"${data.get('usage_daily', 0):.4f}",
                    "本月消费": f"${data.get('usage_monthly', 0):.2f}",
                },
            }

        # SiliconFlow
        if "siliconflow.cn" in base_url:
            resp = requests.get(
                "https://api.siliconflow.cn/v1/user/info",
                headers=headers,
                timeout=timeout,
            )
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

            data = resp.json().get("data", {})
            return {
                "provider": "SiliconFlow",
                "data": {
                    "总余额": f"¥{data.get('totalBalance', '0')}",
                    "充值余额": f"¥{data.get('chargeBalance', '0')}",
                    "赠送余额": f"¥{data.get('balance', '0')}",
                    "用户名": data.get("name", "N/A"),
                    "账户状态": data.get("status", "N/A"),
                },
            }

        # DeepSeek
        if "deepseek.com" in base_url:
            resp = requests.get(
                "https://api.deepseek.com/user/balance",
                headers=headers,
                timeout=timeout,
            )
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

            data = resp.json()
            balance_infos = data.get("balance_infos", [])
            if balance_infos:
                info = balance_infos[0]
                return {
                    "provider": "DeepSeek",
                    "data": {
                        "总余额": f"{info.get('currency', 'CNY')} {info.get('total_balance', '0')}",
                        "赠送余额": f"{info.get('currency', 'CNY')} {info.get('granted_balance', '0')}",
                        "充值余额": f"{info.get('currency', 'CNY')} {info.get('topped_up_balance', '0')}",
                        "余额充足": "是" if data.get("is_available") else "否",
                    },
                }
            return {
                "provider": "DeepSeek",
                "data": {"余额充足": "是" if data.get("is_available") else "否"},
            }

        # AI/ML API
        if "aimlapi.com" in base_url:
            resp = requests.get(
                "https://billing.aimlapi.com/v1/billing/balance",
                headers=headers,
                timeout=timeout,
            )
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

            data = resp.json()
            # credits 转换: 2,000,000 credits = $1
            credits = data.get("balance", 0)
            usd_value = credits / 2_000_000
            return {
                "provider": "AI/ML API",
                "data": {
                    "Credits": f"{credits:,.0f}",
                    "折合美元": f"${usd_value:.4f}",
                },
            }

        # OpenAI (非官方稳定 API，可能会变化)
        if "api.openai.com" in base_url:
            resp = requests.get(
                "https://api.openai.com/v1/dashboard/billing/credit_grants",
                headers=headers,
                timeout=timeout,
            )
            if resp.status_code == 404:
                return {"error": "OpenAI 余额查询 API 不可用（可能已弃用）"}
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

            data = resp.json()
            grants = data.get("grants", {}).get("data", [])
            total_granted = sum(g.get("grant_amount", 0) for g in grants) / 100
            total_used = sum(g.get("used_amount", 0) for g in grants) / 100
            total_available = data.get("total_available", 0) / 100
            total_used_all = data.get("total_used", 0) / 100

            return {
                "provider": "OpenAI",
                "data": {
                    "可用余额": f"${total_available:.2f}",
                    "已使用": f"${total_used_all:.2f}",
                    "总授予额度": f"${total_granted:.2f}",
                },
            }

        # 不支持的 provider
        return None

    except requests.exceptions.RequestException as e:
        return {"error": f"请求失败: {e}"}
    except Exception as e:
        return {"error": f"解析失败: {e}"}


def _single_chat(
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
            # 应用 user_template
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


def _interactive_chat(
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

                    # 应用 user_template
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


# ========== Fallback CLI ==========


def _fallback_cli():
    """没有 typer 时的简单 CLI"""
    args = sys.argv[1:]

    if not args or args[0] in ["-h", "--help", "help"]:
        print("flexllm CLI")
        print("\n命令:")
        print("  ask <prompt>      快速问答")
        print("  chat              交互对话")
        print("  batch             批量处理 JSONL 文件")
        print("  mock              启动 Mock LLM 服务器")
        print("  models            列出远程模型")
        print("  list              列出配置模型")
        print("  set-model <name>  设置默认模型")
        print("  test              测试连接")
        print("  init              初始化配置")
        print("  version           显示版本")
        print("\n安装 typer 获得更好的 CLI 体验: pip install typer")
        return

    print("错误: 需要安装 typer: pip install typer", file=sys.stderr)
    print("或者: pip install flexllm[cli]", file=sys.stderr)


def main():
    """CLI 入口点"""
    if HAS_TYPER:
        app()
    else:
        _fallback_cli()


if __name__ == "__main__":
    main()
