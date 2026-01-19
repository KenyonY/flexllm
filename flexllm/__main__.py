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
        default_config = {"default": None, "models": []}

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
        """从环境变量构建配置"""
        base_url = os.environ.get("FLEXLLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get("FLEXLLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        model = os.environ.get("FLEXLLM_MODEL") or os.environ.get("OPENAI_MODEL")

        if base_url and api_key and model:
            return {
                "id": model,
                "name": model,
                "base_url": base_url,
                "api_key": api_key,
                "provider": "openai",
            }
        return None

    def get_model_config(self, name_or_id: str = None) -> dict | None:
        """获取模型配置"""
        models = self.config.get("models", [])

        if not models:
            env_config = self._config_from_env()
            if env_config:
                return env_config
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
            "return_usage": False,
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


# ========== 输入格式处理 ==========


def detect_input_format(record: dict) -> tuple[str, list[str]]:
    """检测输入记录的格式类型"""
    if "messages" in record:
        return "openai_chat", ["messages"]
    if "instruction" in record:
        return "alpaca", ["instruction", "input"]
    for field in ["q", "question", "prompt"]:
        if field in record:
            return "simple", [field, "system"]
    return "unknown", []


def convert_to_messages(
    record: dict, format_type: str, message_fields: list[str], global_system: str = None
) -> tuple[list[dict], dict]:
    """将输入记录转换为 messages 格式"""
    messages = []
    used_fields = set()

    if format_type == "openai_chat":
        messages = record["messages"]
        used_fields.add("messages")

    elif format_type == "alpaca":
        instruction = record.get("instruction", "")
        input_text = record.get("input", "")
        used_fields.update(["instruction", "input", "output"])

        content = instruction
        if input_text:
            content = f"{instruction}\n\n{input_text}"
        messages = [{"role": "user", "content": content}]

    elif format_type == "simple":
        prompt_field = None
        for field in ["q", "question", "prompt"]:
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
            messages.append({"role": "user", "content": record[prompt_field]})

    if global_system and format_type != "openai_chat":
        messages = [m for m in messages if m.get("role") != "system"]
        messages.insert(0, {"role": "system", "content": global_system})

    metadata = {k: v for k, v in record.items() if k not in used_fields}
    return messages, metadata


def parse_batch_input(input_path: str = None) -> tuple[list[dict], str, list[str]]:
    """解析批量输入文件或 stdin"""
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

    format_type, message_fields = detect_input_format(records[0])

    if format_type == "unknown":
        available_fields = list(records[0].keys())
        raise ValueError(
            f"无法识别输入格式，未找到以下字段之一：\n"
            f"  - messages (openai_chat 格式)\n"
            f"  - instruction (alpaca 格式)\n"
            f"  - q/question/prompt (simple 格式)\n\n"
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
    ):
        """LLM 快速问答（支持管道输入）

        Examples:
            flexllm ask "什么是Python"
            flexllm ask "解释代码" -s "你是代码专家"
            echo "长文本" | flexllm ask "总结一下"
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

        async def _ask():
            from flexllm import LLMClient

            client = LLMClient(model=model_id, base_url=base_url, api_key=api_key)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": full_prompt})
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
        max_tokens: Annotated[int, Option("--max-tokens", help="最大生成 token 数")] = 4096,
        no_stream: Annotated[bool, Option("--no-stream", help="禁用流式输出")] = False,
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

        stream = not no_stream

        if message:
            _single_chat(
                message, model, base_url, api_key, system_prompt, temperature, max_tokens, stream
            )
        else:
            _interactive_chat(
                model, base_url, api_key, system_prompt, temperature, max_tokens, stream
            )

    @app.command()
    def batch(
        input: Annotated[str | None, Argument(help="输入文件路径（省略则从 stdin 读取）")] = None,
        output: Annotated[str | None, Option("-o", "--output", help="输出文件路径（必需）")] = None,
        model: Annotated[str | None, Option("-m", "--model", help="模型名称")] = None,
        concurrency: Annotated[int, Option("-c", "--concurrency", help="并发数")] = 10,
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
    ):
        """批量处理 JSONL 文件（支持断点续传）

        自动检测输入格式：openai_chat, alpaca, simple (q/question/prompt)

        高级配置可在 ~/.flexllm/config.yaml 的 batch 节中设置。
        CLI 参数优先级高于配置文件。

        Examples:
            flexllm batch input.jsonl -o output.jsonl
            flexllm batch input.jsonl -o output.jsonl -c 20 -m gpt-4
            flexllm batch input.jsonl -o output.jsonl --cache --return-usage
            cat input.jsonl | flexllm batch -o output.jsonl
        """
        if not output:
            print("错误: 必须指定输出文件 (-o output.jsonl)", file=sys.stderr)
            raise typer.Exit(1)

        if not output.endswith(".jsonl"):
            print(f"错误: 输出文件必须使用 .jsonl 扩展名，当前: {output}", file=sys.stderr)
            raise typer.Exit(1)

        has_stdin = not sys.stdin.isatty()
        if not input and not has_stdin:
            print("错误: 请提供输入文件或通过管道传入数据", file=sys.stderr)
            raise typer.Exit(1)

        config = get_config()
        model_config = config.get_model_config(model)
        if not model_config:
            print("错误: 未找到模型配置", file=sys.stderr)
            print("提示: 使用 'flexllm list' 查看可用模型", file=sys.stderr)
            raise typer.Exit(1)

        model_id = model_config.get("id")
        base_url = model_config.get("base_url")
        api_key = model_config.get("api_key", "EMPTY")

        # 获取 batch 配置（配置文件 + 默认值）
        batch_config = config.get_batch_config()

        # CLI 参数覆盖配置文件
        effective_cache = cache if cache is not None else batch_config["cache"]
        effective_return_usage = return_usage or batch_config["return_usage"]
        effective_preprocess_msg = preprocess_msg or batch_config["preprocess_msg"]

        try:
            records, format_type, message_fields = parse_batch_input(input)
            print(f"输入格式: {format_type}", file=sys.stderr)
            print(f"记录数: {len(records)}", file=sys.stderr)

            messages_list = []
            metadata_list = []

            for record in records:
                messages, metadata = convert_to_messages(
                    record, format_type, message_fields, system
                )
                messages_list.append(messages)
                metadata_list.append(metadata if metadata else None)

            has_metadata = any(m for m in metadata_list)
            if not has_metadata:
                metadata_list = None

            async def _run_batch():
                from flexllm import LLMClient

                from .cache import ResponseCacheConfig

                # 构建缓存配置
                cache_config = None
                if effective_cache:
                    cache_config = ResponseCacheConfig.ipc(ttl=batch_config["cache_ttl"])

                client_kwargs = {
                    "model": model_id,
                    "base_url": base_url,
                    "api_key": api_key,
                    "concurrency_limit": concurrency,
                    "timeout": batch_config["timeout"],
                    "retry_times": batch_config["retry_times"],
                    "retry_delay": batch_config["retry_delay"],
                    "cache": cache_config,
                }
                if max_qps is not None:
                    client_kwargs["max_qps"] = max_qps

                client = LLMClient(**client_kwargs)

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

                results, summary = await client.chat_completions_batch(
                    messages_list=messages_list,
                    output_jsonl=output,
                    show_progress=True,
                    return_summary=True,
                    return_usage=effective_return_usage,
                    track_cost=track_cost,
                    preprocess_msg=effective_preprocess_msg,
                    flush_interval=batch_config["flush_interval"],
                    metadata_list=metadata_list,
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

            print(f"  {name}{is_default}")
            if name != model_id:
                print(f"    id: {model_id}")
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

# 模型列表
models:
  - id: gpt-4
    name: gpt-4
    provider: openai
    base_url: https://api.openai.com/v1
    api_key: your-api-key

  - id: local-ollama
    name: local-ollama
    provider: openai
    base_url: http://localhost:11434/v1
    api_key: EMPTY

# batch 命令配置（可选）
# 这些配置可通过 CLI 参数覆盖
# batch:
#   # 缓存配置
#   cache: false              # 是否启用响应缓存
#   cache_ttl: 86400          # 缓存过期时间（秒），默认 24 小时
#
#   # 网络配置
#   timeout: 120              # 请求超时时间（秒）
#   retry_times: 3            # 重试次数
#   retry_delay: 1.0          # 重试延迟（秒）
#
#   # 采样参数（覆盖模型默认值）
#   # top_p: 0.9
#   # top_k: 50
#
#   # 思考模式（适用于支持的模型如 DeepSeek-R1）
#   # thinking: true          # 或 "minimal"/"low"/"medium"/"high"
#
#   # 处理配置
#   preprocess_msg: false     # 是否预处理图片消息（URL 转 base64）
#   flush_interval: 1.0       # 文件刷新间隔（秒）
#
#   # 输出配置
#   return_usage: false       # 是否输出 token 统计
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
    def version():
        """显示版本信息"""
        try:
            from importlib.metadata import version as get_version

            v = get_version("flexllm")
        except Exception:
            v = "0.1.0"
        print(f"flexllm {v}")


# ========== 辅助函数 ==========


def _single_chat(message, model, base_url, api_key, system_prompt, temperature, max_tokens, stream):
    """单次对话"""

    async def _run():
        from flexllm import LLMClient

        client = LLMClient(model=model, base_url=base_url, api_key=api_key)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

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


def _interactive_chat(model, base_url, api_key, system_prompt, temperature, max_tokens, stream):
    """多轮交互对话"""

    async def _run():
        from flexllm import LLMClient

        client = LLMClient(model=model, base_url=base_url, api_key=api_key)

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

                messages.append({"role": "user", "content": user_input})

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
