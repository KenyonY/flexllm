"""Mock LLM Server

提供一个轻量级的 Mock LLM 服务器，用于测试和开发。

功能:
- 可配置的响应延迟（固定或随机范围）
- 可配置的响应长度（随机范围）
- RPS 限制（每秒请求数）
- Token 速率控制（流式返回时每秒 token 数）
- 支持 OpenAI / Claude / Gemini 三种 API 格式
- 支持 Embeddings 端点（/v1/embeddings），确定性伪向量
- 支持 MCP (Model Context Protocol) JSON-RPC 端点（/mcp），支持 Streamable HTTP (SSE)
- 支持流式和非流式响应
- 可选的思考/推理内容返回
- Tool Call 支持：请求含 tools 时自动返回 tool_call 响应（OpenAI tool_calls / Claude tool_use / Gemini functionCall）

用法:
    # CLI
    flexllm mock                          # 默认配置
    flexllm mock -p 8001                  # 指定端口
    flexllm mock -d 0.5                   # 固定延迟 0.5s
    flexllm mock -d 1-5                   # 随机延迟 1-5s
    flexllm mock -l 100-500               # 响应长度 100-500 字符
    flexllm mock --rps 10                 # 每秒最多 10 个请求
    flexllm mock --token-rate 50          # 流式返回每秒 50 个 token
    flexllm mock --thinking               # 响应包含思考内容
    flexllm mock --qa qa.jsonl            # 使用 QA 数据集确定性回复

    # QA 数据集格式（JSONL，每行一个 JSON）:
    # {"input": "你好", "output": "你好！有什么可以帮你的？"}
    # {"input": "1+1等于几", "output": "2"}

    # Python
    from flexllm.mock import MockLLMServer, MockServerConfig
    server = MockLLMServer(MockServerConfig(port=8001, rps=10, thinking=True))
    with server:
        # OpenAI: server.url -> "http://localhost:8001/v1"
        # Claude: server.url -> "http://localhost:8001/v1"  (共享前缀)
        # Gemini: server.gemini_url -> "http://localhost:8001"
        ...
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import random
import time
import uuid
from dataclasses import dataclass, field

try:
    from aiohttp import web

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    web = None

# 预定义句子片段，用于生成随机响应
SENTENCES = [
    "这是一个测试响应。",
    "Mock 服务正在正常工作。",
    "人工智能正在改变我们的生活方式。",
    "Python 是一门优雅的编程语言。",
    "深度学习模型需要大量的训练数据。",
    "云计算为企业提供了灵活的资源管理方案。",
    "自然语言处理是人工智能的重要分支。",
    "数据科学家需要掌握统计学和编程技能。",
    "机器学习算法可以从数据中学习规律。",
    "分布式系统需要考虑一致性和可用性的平衡。",
    "大语言模型的参数量已经达到了千亿级别。",
    "Transformer 架构是现代 NLP 的基础。",
    "向量数据库在语义搜索中发挥重要作用。",
    "微服务架构提高了系统的可维护性。",
    "容器化技术简化了应用部署流程。",
    "API 设计需要考虑向后兼容性。",
    "测试驱动开发能提高代码质量。",
    "异步编程可以提高 I/O 密集型应用的性能。",
    "缓存策略对系统性能至关重要。",
    "代码审查是保证代码质量的重要环节。",
]

# 预定义思考过程句子
THINKING_SENTENCES = [
    "让我仔细想想这个问题。",
    "首先，我需要分析问题的核心。",
    "这个问题可以从多个角度来考虑。",
    "根据已知信息，我可以推断出以下几点。",
    "让我逐步分析这个问题的各个方面。",
    "需要考虑的关键因素包括以下几个方面。",
    "从逻辑上看，这个推理过程是合理的。",
    "我需要验证一下这个结论是否正确。",
    "综合以上分析，我得出了以下结论。",
    "让我重新审视一下这个问题的前提条件。",
]


# MCP Mock 预定义工具
MCP_MOCK_TOOLS = [
    {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "search",
        "description": "搜索信息",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "limit": {"type": "integer", "description": "结果数量限制"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_file",
        "description": "读取文件内容",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
            },
            "required": ["path"],
        },
    },
]

MCP_MOCK_RESOURCES = [
    {"uri": "file:///workspace/README.md", "name": "README.md", "mimeType": "text/markdown"},
    {"uri": "file:///workspace/config.yaml", "name": "config.yaml", "mimeType": "text/yaml"},
]

MCP_MOCK_PROMPTS = [
    {
        "name": "summarize",
        "description": "总结文本内容",
        "arguments": [{"name": "text", "required": True}],
    },
    {
        "name": "translate",
        "description": "翻译文本",
        "arguments": [{"name": "text", "required": True}, {"name": "language", "required": True}],
    },
]


class RPSLimiter:
    """RPS 限制器（令牌桶算法）"""

    def __init__(self, rps: float):
        """
        Args:
            rps: 每秒允许的请求数，0 或 None 表示不限制
        """
        self.rps = rps
        self.interval = 1.0 / rps if rps and rps > 0 else 0
        self.last_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """等待直到可以处理下一个请求"""
        if self.interval <= 0:
            return

        async with self._lock:
            now = time.perf_counter()
            wait_time = self.interval - (now - self.last_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_time = time.perf_counter()


@dataclass
class MockServerConfig:
    """Mock 服务配置"""

    port: int = 8001
    delay_min: float = 0.1  # 最小延迟（秒）
    delay_max: float = 0.1  # 最大延迟（秒），等于 delay_min 时为固定延迟
    model: str = "mock-model"
    response_min_len: int = 10  # 响应最小长度（字符）
    response_max_len: int = 1000  # 响应最大长度（字符）
    rps: float = 0  # 每秒请求数限制，0 表示不限制
    token_rate: float = 0  # 流式返回时每秒 token 数，0 表示不限制
    error_rate: float = 0  # 请求失败率 (0-1)，0 表示不失败
    thinking: bool = False  # 是否在响应中包含思考内容
    qa_path: str | None = None  # QA 数据集路径（JSONL），用于确定性回复
    log_path: str | None = None  # 请求日志保存路径（JSONL）


class MockLLMServer:
    """Mock LLM 服务器，支持 OpenAI / Claude / Gemini 三种 API 格式"""

    def __init__(self, config: MockServerConfig = None):
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp is required for MockLLMServer: pip install aiohttp")
        self.config = config or MockServerConfig()
        self.request_count = 0
        self._app = None
        self._runner = None
        self._process = None
        self._rps_limiter = RPSLimiter(self.config.rps)
        self._qa_map: dict[str, str] = self._load_qa_data(self.config.qa_path)

    @staticmethod
    def _load_qa_data(qa_path: str | None) -> dict[str, str]:
        """从 JSONL 文件加载 QA 映射，每行格式: {"input": "...", "output": "..."}"""
        if not qa_path:
            return {}
        import pathlib

        path = pathlib.Path(qa_path)
        if not path.exists():
            raise FileNotFoundError(f"QA 数据文件不存在: {qa_path}")
        qa_map = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                qa_map[item["input"]] = item["output"]
        return qa_map

    def _extract_last_user_text(self, messages: list[dict], api_format: str = "openai") -> str:
        """从消息列表中提取最后一条用户消息的文本"""
        if api_format == "gemini":
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    parts = msg.get("parts", [])
                    for part in reversed(parts):
                        text = part.get("text")
                        if text:
                            return text
            return ""

        # OpenAI / Claude 格式
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    for item in reversed(content):
                        if isinstance(item, dict) and item.get("type") == "text":
                            return item.get("text", "")
                return ""
        return ""

    @staticmethod
    def _sanitize_request(data: dict) -> dict:
        """清理请求体，将 base64 媒体数据替换为占位符"""
        import copy
        import re

        # 需要清理 base64 的字段：(父级 key, media_type 来源 key)
        # - OpenAI image_url: {"url": "data:image/png;base64,..."}
        # - OpenAI input_audio: {"data": "base64...", "format": "wav"}
        # - Gemini inline_data: {"mime_type": "video/mp4", "data": "base64..."}
        # - Claude source: {"type": "base64", "media_type": "image/png", "data": "base64..."}
        _BASE64_FIELDS = {"inline_data", "source", "input_audio"}

        def _b64_size(s: str) -> str:
            size_kb = len(s) * 3 / 4 / 1024
            if size_kb >= 1024:
                return f"{size_kb / 1024:.1f}MB"
            return f"{size_kb:.1f}KB"

        def _clean(obj, parent_key=None):
            if isinstance(obj, str):
                # data:image/png;base64,xxxxx... → <image:32.1KB>
                m = re.match(r"data:([\w/+.-]+);base64,", obj)
                if m:
                    raw = obj[len(m.group()) :]
                    return f"<{m.group(1)}:{_b64_size(raw)}>"
                return obj
            if isinstance(obj, dict):
                # inline_data/source/input_audio 中的 data 字段
                if parent_key in _BASE64_FIELDS and "data" in obj:
                    obj = dict(obj)
                    media_type = (
                        obj.get("mime_type") or obj.get("media_type") or obj.get("format", "binary")
                    )
                    obj["data"] = f"<{media_type}:{_b64_size(obj['data'])}>"
                    return {k: _clean(v, k) for k, v in obj.items()}
                return {k: _clean(v, k) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v, parent_key) for v in obj]
            return obj

        return _clean(copy.deepcopy(data))

    def _log_request(self, api_format: str, request_data: dict, output: str, tokens: dict):
        """记录请求日志：终端打印摘要，可选写入 JSONL 文件"""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        fmt = "gemini" if api_format == "gemini" else "openai"
        messages = request_data.get("contents" if fmt == "gemini" else "messages", [])
        user_text = self._extract_last_user_text(messages, fmt)
        input_summary = user_text[:50] + ("..." if len(user_text) > 50 else "")
        output_summary = output[:50] + ("..." if len(output) > 50 else "")
        pt, ct = tokens.get("prompt_tokens", 0), tokens.get("completion_tokens", 0)
        print(
            f'[{now}] {api_format} | input: "{input_summary}" | output: "{output_summary}" | tokens: {pt}→{ct}'
        )

        if self.config.log_path:
            record = {
                "timestamp": now,
                "api_format": api_format,
                "request": self._sanitize_request(request_data),
                "output": output,
                **tokens,
            }
            with open(self.config.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @property
    def url(self) -> str:
        """OpenAI / Claude API 的 base URL"""
        return f"http://localhost:{self.config.port}/v1"

    @property
    def base_url(self) -> str:
        return self.url

    @property
    def gemini_url(self) -> str:
        """Gemini API 的 base URL（不带 /v1 前缀）"""
        return f"http://localhost:{self.config.port}"

    # ── 通用辅助方法 ──

    def _get_delay(self) -> float:
        if self.config.delay_min == self.config.delay_max:
            return self.config.delay_min
        return random.uniform(self.config.delay_min, self.config.delay_max)

    def _generate_response_text(
        self, user_message: str = "", response_format: dict | None = None
    ) -> str:
        """生成响应文本。如果有 QA 映射且匹配到输入，返回确定性回复；否则随机生成。"""
        if user_message and self._qa_map:
            # 精确匹配优先
            matched = self._qa_map.get(user_message)
            if matched is not None:
                return matched
            # 子串匹配：用户输入包含 QA key 时命中，多个匹配取最长（最具体）
            best_key = ""
            for key in self._qa_map:
                if key in user_message and len(key) > len(best_key):
                    best_key = key
            if best_key:
                return self._qa_map[best_key]

        # response_format: json_object 或 json_schema → 返回合法 JSON
        if response_format and response_format.get("type") in ("json_object", "json_schema"):
            return self._generate_json_response(response_format)

        target_len = random.randint(self.config.response_min_len, self.config.response_max_len)
        result = []
        current_len = 0

        while current_len < target_len:
            sentence = random.choice(SENTENCES)
            result.append(sentence)
            current_len += len(sentence)

        text = "".join(result)
        if len(text) > target_len + 50:
            cut_pos = text.rfind("。", 0, target_len + 50)
            if cut_pos > 0:
                text = text[: cut_pos + 1]

        return text

    def _generate_json_response(self, response_format: dict) -> str:
        """根据 response_format 生成合法 JSON 响应"""
        fmt_type = response_format.get("type")

        if fmt_type == "json_schema":
            schema = response_format.get("json_schema", {}).get("schema", {})
            if schema:
                return json.dumps(self._generate_from_schema(schema), ensure_ascii=False)

        # json_object 或无 schema 的 fallback：返回通用 JSON
        return json.dumps(
            {"result": random.choice(SENTENCES), "status": "ok"},
            ensure_ascii=False,
        )

    def _generate_from_schema(self, schema: dict) -> any:
        """根据 JSON Schema 递归生成 mock 数据"""
        schema_type = schema.get("type", "object")

        if schema_type == "object":
            obj = {}
            for prop_name, prop_schema in schema.get("properties", {}).items():
                obj[prop_name] = self._generate_from_schema(prop_schema)
            return obj
        elif schema_type == "array":
            items_schema = schema.get("items", {"type": "string"})
            return [self._generate_from_schema(items_schema) for _ in range(2)]
        elif schema_type == "string":
            if "enum" in schema:
                return random.choice(schema["enum"])
            return random.choice(SENTENCES)
        elif schema_type == "integer":
            return random.randint(0, 100)
        elif schema_type == "number":
            return round(random.uniform(0, 100), 2)
        elif schema_type == "boolean":
            return random.choice([True, False])
        else:
            return None

    def _generate_thinking_text(self) -> str:
        """生成随机的思考过程文本"""
        target_len = random.randint(50, 500)
        result = []
        current_len = 0
        while current_len < target_len:
            sentence = random.choice(THINKING_SENTENCES)
            result.append(sentence)
            current_len += len(sentence)
        return "".join(result)

    def _should_include_thinking(self, data: dict) -> bool:
        """判断是否应在响应中包含思考内容（全局配置或请求参数）"""
        if self.config.thinking:
            return True
        # OpenAI: "think": true
        if data.get("think") is True:
            return True
        # Claude: "thinking": {"type": "enabled", ...}
        thinking = data.get("thinking", {})
        if isinstance(thinking, dict) and thinking.get("type") == "enabled":
            return True
        # Gemini: generationConfig.thinkingConfig.includeThoughts
        gen_config = data.get("generationConfig", {})
        thinking_config = gen_config.get("thinkingConfig", {})
        if thinking_config.get("includeThoughts") is True:
            return True
        return False

    def _estimate_tokens(self, text: str) -> int:
        """估算 token 数（简单按字符数估算，中文约 1.5 字符/token）"""
        if not text:
            return 0
        return max(1, len(text) // 2)

    def _count_prompt_tokens(self, messages: list[dict]) -> int:
        """计算 prompt 的 token 数"""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self._estimate_tokens(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total += self._estimate_tokens(item.get("text", ""))
            total += 4
        return total

    def _count_gemini_prompt_tokens(self, contents: list[dict]) -> int:
        """计算 Gemini 格式 prompt 的 token 数"""
        total = 0
        for c in contents:
            for p in c.get("parts", []):
                total += self._estimate_tokens(p.get("text", ""))
            total += 4
        return total

    def _tokenize(self, text: str) -> list[str]:
        """简单分词：中文按字符，英文按空格"""
        tokens = []
        current_word = []

        for char in text:
            if "\u4e00" <= char <= "\u9fff":  # 中文字符
                if current_word:
                    tokens.append("".join(current_word))
                    current_word = []
                tokens.append(char)
            elif char.isspace():
                if current_word:
                    tokens.append("".join(current_word))
                    current_word = []
                tokens.append(char)
            else:
                current_word.append(char)

        if current_word:
            tokens.append("".join(current_word))

        return tokens

    async def _wait_token_interval(self, token_interval: float, last_time: float) -> float:
        """Token 速率控制，返回更新后的 last_time"""
        if token_interval > 0:
            now = time.perf_counter()
            wait_time = token_interval - (now - last_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        return time.perf_counter()

    # ── Tool Call 辅助方法 ──

    def _should_respond_with_tool_call(
        self, tools: list | None, messages: list[dict], api_format: str = "openai"
    ) -> bool:
        """判断是否应该以 tool_call 形式响应

        逻辑：有 tools 且最后一条消息不是 tool result 时，返回 tool call
        """
        if not tools:
            return False
        if not messages:
            return True

        last_msg = messages[-1]

        if api_format == "openai":
            return last_msg.get("role") != "tool"
        elif api_format == "claude":
            content = last_msg.get("content", [])
            if isinstance(content, list):
                return not any(
                    isinstance(item, dict) and item.get("type") == "tool_result" for item in content
                )
            return True
        elif api_format == "gemini":
            parts = messages[-1].get("parts", []) if messages else []
            return not any(isinstance(p, dict) and "functionResponse" in p for p in parts)
        return True

    def _pick_tool(self, tools: list[dict], api_format: str = "openai") -> tuple[str, dict]:
        """从工具列表中随机选择一个工具，返回 (tool_name, mock_args)"""
        if api_format == "openai":
            tool = random.choice(tools)
            func = tool.get("function", {})
            name = func.get("name", "mock_tool")
            params = func.get("parameters", {})
        elif api_format == "claude":
            tool = random.choice(tools)
            name = tool.get("name", "mock_tool")
            params = tool.get("input_schema", {})
        elif api_format == "gemini":
            all_funcs = []
            for t in tools:
                all_funcs.extend(t.get("functionDeclarations", []))
            if not all_funcs:
                return "mock_tool", {}
            func = random.choice(all_funcs)
            name = func.get("name", "mock_tool")
            params = func.get("parameters", {})
        else:
            return "mock_tool", {}

        return name, self._generate_mock_args(params)

    def _generate_mock_args(self, params_schema: dict) -> dict:
        """根据参数 schema 生成 mock 参数值"""
        properties = params_schema.get("properties", {})
        required = params_schema.get("required", list(properties.keys()))

        args = {}
        for name in required:
            prop = properties.get(name, {})
            prop_type = prop.get("type", "string")
            if prop_type == "string":
                enum = prop.get("enum")
                args[name] = random.choice(enum) if enum else f"mock_{name}"
            elif prop_type == "integer":
                args[name] = 42
            elif prop_type == "number":
                args[name] = 3.14
            elif prop_type == "boolean":
                args[name] = True
            elif prop_type == "array":
                args[name] = []
            elif prop_type == "object":
                args[name] = {}
            else:
                args[name] = f"mock_{name}"
        return args

    # ── OpenAI 格式 ──

    async def _stream_response(
        self,
        response_text: str,
        model: str,
        prompt_tokens: int,
        request_id: str,
        thinking_text: str | None = None,
    ):
        """生成 OpenAI 格式的流式响应"""
        token_interval = 1.0 / self.config.token_rate if self.config.token_rate > 0 else 0
        last_time = time.perf_counter()
        completion_tokens = 0
        first_chunk = True

        # 先发送 reasoning chunks
        if thinking_text:
            for token in self._tokenize(thinking_text):
                last_time = await self._wait_token_interval(token_interval, last_time)
                completion_tokens += 1
                delta = {"reasoning": token}
                if first_chunk:
                    delta["role"] = "assistant"
                    first_chunk = False
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": delta, "logprobs": None, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # 发送 content chunks
        for token in self._tokenize(response_text):
            last_time = await self._wait_token_interval(token_interval, last_time)
            completion_tokens += 1
            delta = {"content": token}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # 结束标记
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    async def _stream_openai_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
        tool_call_id: str,
        model: str,
        prompt_tokens: int,
        request_id: str,
    ):
        """生成 OpenAI 格式的流式 tool_call 响应"""
        token_interval = 1.0 / self.config.token_rate if self.config.token_rate > 0 else 0
        last_time = time.perf_counter()
        args_str = json.dumps(tool_args, ensure_ascii=False)
        completion_tokens = 1

        # 第一个 chunk：tool call 开始（含 name）
        first_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": tool_call_id,
                                "type": "function",
                                "function": {"name": tool_name, "arguments": ""},
                            }
                        ],
                    },
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

        # 流式发送 arguments（分 ~5 段）
        chunk_size = max(1, len(args_str) // 5)
        for i in range(0, len(args_str), chunk_size):
            last_time = await self._wait_token_interval(token_interval, last_time)
            completion_tokens += 1
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": args_str[i : i + chunk_size]},
                                }
                            ]
                        },
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # 结束标记
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "tool_calls"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    async def _handle_chat_completions(self, request: web.Request) -> web.Response:
        """处理 /v1/chat/completions 请求（OpenAI 格式）"""
        await self._rps_limiter.acquire()
        self.request_count += 1
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        try:
            data = await request.json()
        except Exception:
            data = {}

        messages = data.get("messages", [])
        model = data.get("model", self.config.model)
        stream = data.get("stream", False)
        include_thinking = self._should_include_thinking(data)

        await asyncio.sleep(self._get_delay())

        if self.config.error_rate > 0 and random.random() < self.config.error_rate:
            return web.json_response(
                {
                    "error": {
                        "message": f"Mock server simulated error (error_rate={self.config.error_rate})",
                        "type": "server_error",
                        "code": "mock_error",
                    }
                },
                status=500,
            )

        prompt_tokens = self._count_prompt_tokens(messages)

        # Tool call 检测：有 tools 且最后消息非 tool result → 返回 tool_call
        tools = data.get("tools")
        if tools and self._should_respond_with_tool_call(tools, messages, "openai"):
            tool_name, tool_args = self._pick_tool(tools, "openai")
            tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
            args_str = json.dumps(tool_args, ensure_ascii=False)
            completion_tokens = self._estimate_tokens(args_str)
            self._log_request(
                "openai",
                data,
                f"tool_call:{tool_name}({args_str})",
                {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            )

            if stream:
                response = web.StreamResponse(
                    status=200,
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )
                await response.prepare(request)
                async for chunk in self._stream_openai_tool_call(
                    tool_name, tool_args, tool_call_id, model, prompt_tokens, request_id
                ):
                    await response.write(chunk.encode("utf-8"))
                await response.write_eof()
                return response

            return web.json_response(
                {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": tool_call_id,
                                        "type": "function",
                                        "function": {
                                            "name": tool_name,
                                            "arguments": args_str,
                                        },
                                    }
                                ],
                            },
                            "logprobs": None,
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "system_fingerprint": "mock-fp-001",
                }
            )

        user_text = self._extract_last_user_text(messages, "openai")
        response_format = data.get("response_format")
        response_text = self._generate_response_text(user_text, response_format)
        thinking_text = self._generate_thinking_text() if include_thinking else None
        self._log_request(
            "openai",
            data,
            response_text,
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": self._estimate_tokens(response_text),
            },
        )

        if stream:
            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
            await response.prepare(request)
            async for chunk in self._stream_response(
                response_text, model, prompt_tokens, request_id, thinking_text
            ):
                await response.write(chunk.encode("utf-8"))
            await response.write_eof()
            return response
        else:
            completion_tokens = self._estimate_tokens(response_text)
            if thinking_text:
                completion_tokens += self._estimate_tokens(thinking_text)
            message_obj = {"role": "assistant", "content": response_text}
            if thinking_text:
                message_obj["reasoning"] = thinking_text
            return web.json_response(
                {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": message_obj,
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "system_fingerprint": "mock-fp-001",
                }
            )

    # ── Claude 格式 ──

    async def _claude_stream_response(
        self,
        request: web.Request,
        response_text: str,
        thinking_text: str | None,
        model: str,
        prompt_tokens: int,
        output_tokens: int,
        request_id: str,
    ) -> web.StreamResponse:
        """生成 Claude 格式的流式响应"""
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await resp.prepare(request)

        token_interval = 1.0 / self.config.token_rate if self.config.token_rate > 0 else 0
        last_time = time.perf_counter()

        async def send_event(event_type: str, data: dict):
            line = f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            await resp.write(line.encode("utf-8"))

        # message_start
        await send_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": request_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
                },
            },
        )

        block_index = 0

        # thinking block
        if thinking_text:
            await send_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            )
            for token in self._tokenize(thinking_text):
                last_time = await self._wait_token_interval(token_interval, last_time)
                await send_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "thinking_delta", "thinking": token},
                    },
                )
            await send_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": block_index},
            )
            block_index += 1

        # text block
        await send_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": block_index,
                "content_block": {"type": "text", "text": ""},
            },
        )
        for token in self._tokenize(response_text):
            last_time = await self._wait_token_interval(token_interval, last_time)
            await send_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "text_delta", "text": token},
                },
            )
        await send_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": block_index},
        )

        # message_delta + message_stop
        await send_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        )
        await send_event("message_stop", {"type": "message_stop"})

        await resp.write_eof()
        return resp

    async def _claude_stream_tool_use_response(
        self,
        request: web.Request,
        tool_name: str,
        tool_args: dict,
        tool_use_id: str,
        model: str,
        prompt_tokens: int,
        request_id: str,
    ) -> web.StreamResponse:
        """生成 Claude 格式的流式 tool_use 响应"""
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await resp.prepare(request)

        token_interval = 1.0 / self.config.token_rate if self.config.token_rate > 0 else 0
        last_time = time.perf_counter()

        async def send_event(event_type: str, data: dict):
            line = f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            await resp.write(line.encode("utf-8"))

        args_str = json.dumps(tool_args, ensure_ascii=False)
        output_tokens = self._estimate_tokens(args_str)

        # message_start
        await send_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": request_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
                },
            },
        )

        # tool_use content block
        await send_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": tool_name,
                    "input": {},
                },
            },
        )

        # 流式发送 input JSON（分 ~5 段）
        chunk_size = max(1, len(args_str) // 5)
        for i in range(0, len(args_str), chunk_size):
            last_time = await self._wait_token_interval(token_interval, last_time)
            await send_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": args_str[i : i + chunk_size],
                    },
                },
            )

        await send_event("content_block_stop", {"type": "content_block_stop", "index": 0})

        # message_delta + message_stop
        await send_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        )
        await send_event("message_stop", {"type": "message_stop"})

        await resp.write_eof()
        return resp

    async def _handle_claude_messages(self, request: web.Request) -> web.Response:
        """处理 /v1/messages 请求（Claude 格式）"""
        await self._rps_limiter.acquire()
        self.request_count += 1
        request_id = f"msg_{uuid.uuid4().hex[:24]}"

        try:
            data = await request.json()
        except Exception:
            data = {}

        messages = data.get("messages", [])
        model = data.get("model", self.config.model)
        stream = data.get("stream", False)
        include_thinking = self._should_include_thinking(data)

        await asyncio.sleep(self._get_delay())

        if self.config.error_rate > 0 and random.random() < self.config.error_rate:
            return web.json_response(
                {"type": "error", "error": {"type": "server_error", "message": "Mock error"}},
                status=500,
            )

        prompt_tokens = self._count_prompt_tokens(messages)

        # Tool use 检测
        tools = data.get("tools")
        if tools and self._should_respond_with_tool_call(tools, messages, "claude"):
            tool_name, tool_args = self._pick_tool(tools, "claude")
            tool_use_id = f"toolu_{uuid.uuid4().hex[:24]}"
            args_str = json.dumps(tool_args, ensure_ascii=False)
            completion_tokens = self._estimate_tokens(args_str)
            self._log_request(
                "claude",
                data,
                f"tool_call:{tool_name}({args_str})",
                {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            )

            if stream:
                return await self._claude_stream_tool_use_response(
                    request, tool_name, tool_args, tool_use_id, model, prompt_tokens, request_id
                )

            return web.json_response(
                {
                    "id": request_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": tool_name,
                            "input": tool_args,
                        }
                    ],
                    "model": model,
                    "stop_reason": "tool_use",
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                    },
                }
            )

        user_text = self._extract_last_user_text(messages, "claude")
        # Claude 没有 response_format，但如果请求中带了也兼容处理
        response_text = self._generate_response_text(user_text)
        thinking_text = self._generate_thinking_text() if include_thinking else None
        completion_tokens = self._estimate_tokens(response_text)
        if thinking_text:
            completion_tokens += self._estimate_tokens(thinking_text)
        self._log_request(
            "claude",
            data,
            response_text,
            {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        )

        if stream:
            return await self._claude_stream_response(
                request,
                response_text,
                thinking_text,
                model,
                prompt_tokens,
                completion_tokens,
                request_id,
            )

        content_blocks = []
        if thinking_text:
            content_blocks.append({"type": "thinking", "thinking": thinking_text})
        content_blocks.append({"type": "text", "text": response_text})

        return web.json_response(
            {
                "id": request_id,
                "type": "message",
                "role": "assistant",
                "content": content_blocks,
                "model": model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                },
            }
        )

    # ── Gemini 格式 ──

    async def _gemini_stream_response(
        self,
        request: web.Request,
        response_text: str,
        thinking_text: str | None,
        prompt_tokens: int,
        output_tokens: int,
    ) -> web.StreamResponse:
        """生成 Gemini 格式的流式响应"""
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
            },
        )
        await resp.prepare(request)

        token_interval = 1.0 / self.config.token_rate if self.config.token_rate > 0 else 0
        last_time = time.perf_counter()

        # thinking chunks
        if thinking_text:
            for token in self._tokenize(thinking_text):
                last_time = await self._wait_token_interval(token_interval, last_time)
                chunk = {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": token, "thought": True}],
                                "role": "model",
                            }
                        }
                    ],
                }
                await resp.write(
                    f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                )

        # content chunks
        content_tokens = self._tokenize(response_text)
        for i, token in enumerate(content_tokens):
            last_time = await self._wait_token_interval(token_interval, last_time)
            chunk = {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": token}],
                            "role": "model",
                        }
                    }
                ],
            }
            # 最后一个 chunk 附带 usage
            if i == len(content_tokens) - 1:
                chunk["usageMetadata"] = {
                    "promptTokenCount": prompt_tokens,
                    "candidatesTokenCount": output_tokens,
                    "totalTokenCount": prompt_tokens + output_tokens,
                }
            await resp.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))

        await resp.write_eof()
        return resp

    async def _handle_gemini(self, request: web.Request) -> web.Response:
        """处理 Gemini API 请求（/models/{model}:generateContent 或 :streamGenerateContent）"""
        await self._rps_limiter.acquire()
        self.request_count += 1

        model_action = request.match_info["model_action"]
        is_stream = "streamGenerateContent" in model_action

        try:
            data = await request.json()
        except Exception:
            data = {}

        contents = data.get("contents", [])
        include_thinking = self._should_include_thinking(data)

        await asyncio.sleep(self._get_delay())

        if self.config.error_rate > 0 and random.random() < self.config.error_rate:
            return web.json_response(
                {"error": {"code": 500, "message": "Mock error", "status": "INTERNAL"}},
                status=500,
            )

        prompt_tokens = self._count_gemini_prompt_tokens(contents)

        # Function call 检测
        tools = data.get("tools")
        if tools and self._should_respond_with_tool_call(tools, contents, "gemini"):
            tool_name, tool_args = self._pick_tool(tools, "gemini")
            args_str = json.dumps(tool_args, ensure_ascii=False)
            output_tokens = self._estimate_tokens(args_str)
            self._log_request(
                "gemini",
                data,
                f"tool_call:{tool_name}({args_str})",
                {"prompt_tokens": prompt_tokens, "completion_tokens": output_tokens},
            )

            func_call_part = {"functionCall": {"name": tool_name, "args": tool_args}}

            if is_stream:
                resp = web.StreamResponse(
                    status=200,
                    headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
                )
                await resp.prepare(request)
                chunk = {
                    "candidates": [
                        {
                            "content": {"parts": [func_call_part], "role": "model"},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": prompt_tokens,
                        "candidatesTokenCount": output_tokens,
                        "totalTokenCount": prompt_tokens + output_tokens,
                    },
                }
                await resp.write(
                    f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                )
                await resp.write_eof()
                return resp

            return web.json_response(
                {
                    "candidates": [
                        {
                            "content": {"parts": [func_call_part], "role": "model"},
                            "finishReason": "STOP",
                            "index": 0,
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": prompt_tokens,
                        "candidatesTokenCount": output_tokens,
                        "totalTokenCount": prompt_tokens + output_tokens,
                    },
                }
            )

        user_text = self._extract_last_user_text(contents, "gemini")
        # Gemini: responseMimeType=application/json + responseSchema → JSON 模式
        gen_config = data.get("generationConfig", {})
        gemini_rf = None
        if gen_config.get("responseMimeType") == "application/json":
            response_schema = gen_config.get("responseSchema")
            if response_schema:
                gemini_rf = {"type": "json_schema", "json_schema": {"schema": response_schema}}
            else:
                gemini_rf = {"type": "json_object"}
        response_text = self._generate_response_text(user_text, gemini_rf)
        thinking_text = self._generate_thinking_text() if include_thinking else None
        completion_tokens = self._estimate_tokens(response_text)
        if thinking_text:
            completion_tokens += self._estimate_tokens(thinking_text)
        self._log_request(
            "gemini",
            data,
            response_text,
            {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        )

        if is_stream:
            return await self._gemini_stream_response(
                request, response_text, thinking_text, prompt_tokens, completion_tokens
            )

        parts = []
        if thinking_text:
            parts.append({"text": thinking_text, "thought": True})
        parts.append({"text": response_text})

        return web.json_response(
            {
                "candidates": [
                    {
                        "content": {"parts": parts, "role": "model"},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": prompt_tokens,
                    "candidatesTokenCount": completion_tokens,
                    "totalTokenCount": prompt_tokens + completion_tokens,
                },
            }
        )

    # ── Embeddings ──

    @staticmethod
    def _generate_embedding(text: str, dimensions: int) -> list[float]:
        """基于文本内容生成确定性的伪向量（相同文本 → 相同向量）"""
        import hashlib

        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) & 0xFFFFFFFF
        rng = random.Random(seed)
        vec = [rng.gauss(0, 1) for _ in range(dimensions)]
        # L2 归一化
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    async def _handle_embeddings(self, request: web.Request) -> web.Response:
        """处理 /v1/embeddings 请求（OpenAI 格式）"""
        await self._rps_limiter.acquire()
        self.request_count += 1

        try:
            data = await request.json()
        except Exception:
            data = {}

        model = data.get("model", self.config.model)
        input_data = data.get("input", "")
        dimensions = data.get("dimensions", 128)

        await asyncio.sleep(self._get_delay())

        if self.config.error_rate > 0 and random.random() < self.config.error_rate:
            return web.json_response(
                {
                    "error": {
                        "message": f"Mock server simulated error (error_rate={self.config.error_rate})",
                        "type": "server_error",
                        "code": "mock_error",
                    }
                },
                status=500,
            )

        # input 可以是 str 或 list[str]
        if isinstance(input_data, str):
            texts = [input_data]
        else:
            texts = input_data

        embeddings = []
        total_tokens = 0
        for i, text in enumerate(texts):
            vec = self._generate_embedding(text, dimensions)
            embeddings.append({"object": "embedding", "index": i, "embedding": vec})
            total_tokens += self._estimate_tokens(text)

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        input_summary = str(texts[0])[:50] + ("..." if len(str(texts[0])) > 50 else "")
        print(
            f'[{now}] embedding | input: "{input_summary}" | count: {len(texts)} | dim: {dimensions}'
        )

        if self.config.log_path:
            record = {
                "timestamp": now,
                "api_format": "embedding",
                "input_count": len(texts),
                "dimensions": dimensions,
                "prompt_tokens": total_tokens,
            }
            with open(self.config.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return web.json_response(
            {
                "object": "list",
                "data": embeddings,
                "model": model,
                "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
            }
        )

    # ── 通用路由和服务器生命周期 ──

    async def _handle_models(self, request: web.Request) -> web.Response:
        """处理 /v1/models 请求"""
        return web.json_response(
            {
                "object": "list",
                "data": [{"id": self.config.model, "object": "model"}],
            }
        )

    # ── MCP (Model Context Protocol) JSON-RPC ──

    async def _handle_mcp(self, request: web.Request) -> web.StreamResponse:
        """处理 MCP JSON-RPC 请求，支持 JSON 和 SSE 响应"""
        self.request_count += 1
        await self._rps_limiter.acquire()
        await asyncio.sleep(self._get_delay())

        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response(
                {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None},
                status=400,
            )

        # 批量请求
        if isinstance(data, list):
            results = [self._handle_mcp_single(item) for item in data]
            return web.json_response(results)

        # 错误模拟
        if self.config.error_rate > 0 and random.random() < self.config.error_rate:
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": "Mock MCP server simulated error"},
                    "id": data.get("id"),
                },
                status=500,
            )

        # 检查 Accept 头决定是否用 SSE
        accept = request.headers.get("Accept", "")
        use_sse = "text/event-stream" in accept

        result = self._handle_mcp_single(data)

        if use_sse and data.get("id") is not None:
            # Streamable HTTP: SSE 响应
            response = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
            )
            await response.prepare(request)
            sse_data = json.dumps(result, ensure_ascii=False)
            await response.write(f"data: {sse_data}\n\n".encode())
            await response.write_eof()
            return response

        return web.json_response(result)

    def _handle_mcp_single(self, data: dict) -> dict:
        """处理单条 MCP JSON-RPC 请求"""
        method = data.get("method", "")
        params = data.get("params", {})
        req_id = data.get("id")

        # Notification（无 id）不需要响应
        if req_id is None and method.startswith("notifications/"):
            return {}

        handler = {
            "initialize": self._mcp_initialize,
            "tools/list": self._mcp_tools_list,
            "tools/call": self._mcp_tools_call,
            "resources/list": self._mcp_resources_list,
            "resources/read": self._mcp_resources_read,
            "prompts/list": self._mcp_prompts_list,
            "prompts/get": self._mcp_prompts_get,
            "ping": self._mcp_ping,
        }.get(method)

        if handler is None:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Method not found: {method}"},
                "id": req_id,
            }

        result = handler(params)
        return {"jsonrpc": "2.0", "result": result, "id": req_id}

    def _mcp_initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": params.get("protocolVersion", "2025-03-26"),
            "serverInfo": {"name": "mock-mcp-server", "version": "1.0.0"},
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
            },
        }

    def _mcp_ping(self, params: dict) -> dict:
        return {}

    def _mcp_tools_list(self, params: dict) -> dict:
        return {"tools": MCP_MOCK_TOOLS}

    def _mcp_tools_call(self, params: dict) -> dict:
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        # 根据 tool 名称返回 mock 结果
        if tool_name == "get_weather":
            city = arguments.get("city", "未知城市")
            text = f"{city}：晴，28°C，湿度 65%，东南风 3 级"
        elif tool_name == "search":
            query = arguments.get("query", "")
            text = f"搜索 '{query}' 找到 3 个结果：\n1. {query} 简介\n2. {query} 详细说明\n3. {query} 最新动态"
        elif tool_name == "read_file":
            path = arguments.get("path", "")
            text = (
                f"# Mock File Content\n\nThis is mock content for: {path}\n\nLine 1\nLine 2\nLine 3"
            )
        else:
            text = self._generate_response_text()

        return {
            "content": [{"type": "text", "text": text}],
            "isError": False,
        }

    def _mcp_resources_list(self, params: dict) -> dict:
        return {"resources": MCP_MOCK_RESOURCES}

    def _mcp_resources_read(self, params: dict) -> dict:
        uri = params.get("uri", "")
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "text/plain",
                    "text": f"Mock content for resource: {uri}",
                }
            ],
        }

    def _mcp_prompts_list(self, params: dict) -> dict:
        return {"prompts": MCP_MOCK_PROMPTS}

    def _mcp_prompts_get(self, params: dict) -> dict:
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        if name == "summarize":
            text = arguments.get("text", "")
            content = f"摘要：{text[:50]}..." if len(text) > 50 else f"摘要：{text}"
        elif name == "translate":
            text = arguments.get("text", "")
            lang = arguments.get("language", "English")
            content = f"[翻译为 {lang}] {text}"
        else:
            content = f"Mock prompt response for: {name}"

        return {
            "description": f"Mock prompt: {name}",
            "messages": [{"role": "user", "content": {"type": "text", "text": content}}],
        }

    def _create_app(self) -> web.Application:
        """创建 aiohttp 应用"""
        app = web.Application()
        # OpenAI
        app.router.add_post("/v1/chat/completions", self._handle_chat_completions)
        app.router.add_post("/v1/embeddings", self._handle_embeddings)
        app.router.add_get("/v1/models", self._handle_models)
        # Claude（共享 /v1 前缀）
        app.router.add_post("/v1/messages", self._handle_claude_messages)
        # Gemini（model 名称中含冒号，用正则匹配）
        app.router.add_route("POST", r"/models/{model_action:.+}", self._handle_gemini)
        # MCP (JSON-RPC over HTTP)
        app.router.add_post("/mcp", self._handle_mcp)
        return app

    async def start_async(self):
        """异步启动服务器"""
        self._app = self._create_app()
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self.config.port)
        await site.start()

    async def stop_async(self):
        """异步停止服务器"""
        if self._runner:
            await self._runner.cleanup()

    def _run_server(self):
        """在独立进程中运行服务器"""
        app = self._create_app()
        web.run_app(app, host="0.0.0.0", port=self.config.port, print=lambda x: None)

    def start(self):
        """启动服务器（在独立进程中）"""
        self._process = multiprocessing.Process(target=self._run_server, daemon=True)
        self._process.start()
        time.sleep(0.3)

    def stop(self):
        """停止服务器"""
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)

    def run(self):
        """前台运行服务器（阻塞）"""
        app = self._create_app()
        web.run_app(app, host="0.0.0.0", port=self.config.port)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class MockLLMServerGroup:
    """Mock 服务器组，用于测试多 endpoint 场景"""

    def __init__(self, configs: list[MockServerConfig] = None, num_servers: int = 2):
        if configs:
            self.servers = [MockLLMServer(cfg) for cfg in configs]
        else:
            self.servers = [
                MockLLMServer(MockServerConfig(port=8001 + i)) for i in range(num_servers)
            ]

    @property
    def urls(self) -> list[str]:
        return [s.url for s in self.servers]

    @property
    def endpoints(self) -> list[dict]:
        """返回可直接用于 LLMClientPool 的 endpoints 配置"""
        return [
            {"base_url": s.url, "api_key": "EMPTY", "model": s.config.model} for s in self.servers
        ]

    def start(self):
        for s in self.servers:
            s.start()

    def stop(self):
        for s in self.servers:
            s.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def parse_range(range_str: str, value_type=float) -> tuple:
    """解析范围参数，支持 '0.5' 或 '5-10' 格式"""
    if "-" in range_str:
        parts = range_str.split("-")
        return value_type(parts[0]), value_type(parts[1])
    else:
        v = value_type(range_str)
        return v, v
