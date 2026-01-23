"""Mock LLM Server

提供一个轻量级的 Mock LLM 服务器，用于测试和开发。

功能:
- 可配置的响应延迟（固定或随机范围）
- 可配置的响应长度（随机范围）
- RPS 限制（每秒请求数）
- Token 速率控制（流式返回时每秒 token 数）
- 符合 OpenAI API 规范的完整响应格式
- 支持流式和非流式响应

用法:
    # CLI
    flexllm mock                          # 默认配置
    flexllm mock -p 8001                  # 指定端口
    flexllm mock -d 0.5                   # 固定延迟 0.5s
    flexllm mock -d 1-5                   # 随机延迟 1-5s
    flexllm mock -l 100-500               # 响应长度 100-500 字符
    flexllm mock --rps 10                 # 每秒最多 10 个请求
    flexllm mock --token-rate 50          # 流式返回每秒 50 个 token

    # Python
    from flexllm.mock import MockLLMServer, MockServerConfig
    server = MockLLMServer(MockServerConfig(port=8001, rps=10, token_rate=50))
    with server:
        # server.url -> "http://localhost:8001/v1"
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


class MockLLMServer:
    """Mock LLM 服务器"""

    def __init__(self, config: MockServerConfig = None):
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp is required for MockLLMServer: pip install aiohttp")
        self.config = config or MockServerConfig()
        self.request_count = 0
        self._app = None
        self._runner = None
        self._process = None
        self._rps_limiter = RPSLimiter(self.config.rps)

    @property
    def url(self) -> str:
        return f"http://localhost:{self.config.port}/v1"

    @property
    def base_url(self) -> str:
        return self.url

    def _get_delay(self) -> float:
        """获取延迟时间"""
        if self.config.delay_min == self.config.delay_max:
            return self.config.delay_min
        return random.uniform(self.config.delay_min, self.config.delay_max)

    def _generate_response_text(self) -> str:
        """生成随机长度的响应文本"""
        target_len = random.randint(self.config.response_min_len, self.config.response_max_len)
        result = []
        current_len = 0

        while current_len < target_len:
            sentence = random.choice(SENTENCES)
            result.append(sentence)
            current_len += len(sentence)

        text = "".join(result)
        # 截断到目标长度附近（在句子边界）
        if len(text) > target_len + 50:
            cut_pos = text.rfind("。", 0, target_len + 50)
            if cut_pos > 0:
                text = text[: cut_pos + 1]

        return text

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

    async def _stream_response(
        self, response_text: str, model: str, prompt_tokens: int, request_id: str
    ):
        """生成流式响应"""
        tokens = self._tokenize(response_text)
        token_interval = 1.0 / self.config.token_rate if self.config.token_rate > 0 else 0
        last_time = time.perf_counter()
        completion_tokens = 0

        for i, token in enumerate(tokens):
            # Token 速率控制
            if token_interval > 0:
                now = time.perf_counter()
                wait_time = token_interval - (now - last_time)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                last_time = time.perf_counter()

            completion_tokens += 1
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token}
                        if i > 0
                        else {"role": "assistant", "content": token},
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # 发送结束标记
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    async def _handle_chat_completions(self, request: web.Request) -> web.Response:
        """处理 /v1/chat/completions 请求"""
        # RPS 限制
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

        # 首次响应延迟（模拟 TTFT）
        delay = self._get_delay()
        await asyncio.sleep(delay)

        # 模拟错误（根据 error_rate）
        if self.config.error_rate > 0 and random.random() < self.config.error_rate:
            error_response = {
                "error": {
                    "message": f"Mock server simulated error (error_rate={self.config.error_rate})",
                    "type": "server_error",
                    "code": "mock_error",
                }
            }
            return web.json_response(error_response, status=500)

        # 生成响应文本
        response_text = self._generate_response_text()
        prompt_tokens = self._count_prompt_tokens(messages)

        if stream:
            # 流式响应
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
                response_text, model, prompt_tokens, request_id
            ):
                await response.write(chunk.encode("utf-8"))

            await response.write_eof()
            return response
        else:
            # 非流式响应
            completion_tokens = self._estimate_tokens(response_text)
            response = {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text,
                        },
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
            return web.json_response(response)

    async def _handle_models(self, request: web.Request) -> web.Response:
        """处理 /v1/models 请求"""
        return web.json_response(
            {
                "object": "list",
                "data": [{"id": self.config.model, "object": "model"}],
            }
        )

    def _create_app(self) -> web.Application:
        """创建 aiohttp 应用"""
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._handle_chat_completions)
        app.router.add_get("/v1/models", self._handle_models)
        return app

    async def start_async(self):
        """异步启动服务器"""
        self._app = self._create_app()
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "localhost", self.config.port)
        await site.start()

    async def stop_async(self):
        """异步停止服务器"""
        if self._runner:
            await self._runner.cleanup()

    def _run_server(self):
        """在独立进程中运行服务器"""
        app = self._create_app()
        web.run_app(app, host="localhost", port=self.config.port, print=lambda x: None)

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
        web.run_app(app, host="localhost", port=self.config.port)

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
