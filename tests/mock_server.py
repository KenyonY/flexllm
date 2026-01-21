"""Mock LLM Server for testing

支持以下功能：
- 可配置的响应延迟（固定或随机范围）
- 可配置的端口
- 可作为独立脚本运行
- 可作为 pytest fixture 使用

用法:
    # 作为独立脚本运行
    python tests/mock_server.py 8001              # 固定延迟 0.1s
    python tests/mock_server.py 8001 --delay 0.5  # 固定延迟 0.5s
    python tests/mock_server.py 8001 --delay 5-10 # 随机延迟 5-10s

    # 作为 fixture 使用
    def test_xxx(mock_llm_server):
        url = mock_llm_server.url
        ...
"""

import argparse
import asyncio
import multiprocessing
import random
import time
from dataclasses import dataclass

from aiohttp import web


@dataclass
class MockServerConfig:
    """Mock 服务配置"""

    port: int = 8001
    delay_min: float = 0.1  # 最小延迟（秒）
    delay_max: float = 0.1  # 最大延迟（秒），等于 delay_min 时为固定延迟
    model: str = "mock-model"


class MockLLMServer:
    """Mock LLM 服务器"""

    def __init__(self, config: MockServerConfig = None):
        self.config = config or MockServerConfig()
        self.request_count = 0
        self._app = None
        self._runner = None
        self._process = None

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

    async def _handle_chat_completions(self, request: web.Request) -> web.Response:
        """处理 /v1/chat/completions 请求"""
        self.request_count += 1
        req_id = self.request_count

        try:
            data = await request.json()
        except Exception:
            data = {}

        delay = self._get_delay()
        await asyncio.sleep(delay)

        response = {
            "id": f"mock-{req_id}",
            "object": "chat.completion",
            "model": data.get("model", self.config.model),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Mock response (port:{self.config.port}, delay:{delay:.2f}s)",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
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
        # 等待服务器启动
        time.sleep(0.3)

    def stop(self):
        """停止服务器"""
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)

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
            # 默认创建 num_servers 个服务器
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


def parse_delay(delay_str: str) -> tuple[float, float]:
    """解析延迟参数，支持 '0.5' 或 '5-10' 格式"""
    if "-" in delay_str:
        parts = delay_str.split("-")
        return float(parts[0]), float(parts[1])
    else:
        d = float(delay_str)
        return d, d


def main():
    parser = argparse.ArgumentParser(description="Mock LLM Server")
    parser.add_argument("port", type=int, nargs="?", default=8001, help="端口号")
    parser.add_argument(
        "--delay", type=str, default="0.1", help="延迟时间，支持 '0.5' 或 '5-10' 格式"
    )
    args = parser.parse_args()

    delay_min, delay_max = parse_delay(args.delay)
    config = MockServerConfig(port=args.port, delay_min=delay_min, delay_max=delay_max)

    print(f"Mock LLM Server starting on port {args.port}")
    print(f"Delay: {delay_min}-{delay_max}s")
    print(f"URL: http://localhost:{args.port}/v1")

    server = MockLLMServer(config)
    app = server._create_app()
    web.run_app(app, host="localhost", port=args.port)


if __name__ == "__main__":
    main()
