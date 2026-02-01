"""Serve Server

将 LLM 包装为 HTTP API 服务，适用于微调模型部署：
固定 system prompt 和 user template，调用方只需发送 content 文本。

用法:
    flexllm serve -m qwen-finetuned -s "你是助手" --user-template "[INST]{content}[/INST]"
    flexllm serve --thinking true -p 8000

API 端点:
    POST /api/generate         非流式生成
    POST /api/generate/stream  流式生成 (SSE)
    POST /api/generate/batch   批量生成
    GET  /health               健康检查
    GET  /api/config           查看当前配置
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass

from aiohttp import web

logger = logging.getLogger("flexllm.serve")

LOG_MAX_LEN = 200


def _truncate(text: str | None, max_len: int = LOG_MAX_LEN) -> str:
    if not text:
        return ""
    return text[:max_len] + "..." if len(text) > max_len else text


from .chat_web import ThinkTagParser


@dataclass
class ServeConfig:
    port: int = 8000
    host: str = "0.0.0.0"
    model: str = ""
    base_url: str = ""
    api_key: str = "EMPTY"
    system_prompt: str | None = None
    user_template: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    thinking: bool | str | int | None = None
    concurrency: int = 1000
    max_qps: float | None = None
    timeout: int = 120
    verbose: bool = False


class ServeServer:
    def __init__(self, config: ServeConfig):
        self.config = config
        self._client = None

    def _build_messages(self, content: str) -> list[dict]:
        """构造 messages：注入 system prompt + 应用 user_template"""
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        user_content = content
        if self.config.user_template:
            user_content = self.config.user_template.format(content=content)
        messages.append({"role": "user", "content": user_content})
        return messages

    def _get_kwargs(self, data: dict) -> dict:
        """构造 LLM 调用参数，支持请求级覆盖"""
        kwargs = {}
        # 服务端默认值
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens
        if self.config.thinking is not None:
            kwargs["thinking"] = self.config.thinking
        # 请求级覆盖
        if "temperature" in data:
            kwargs["temperature"] = data["temperature"]
        if "max_tokens" in data:
            kwargs["max_tokens"] = data["max_tokens"]
        return kwargs

    @staticmethod
    def _parse_result(raw_data: dict) -> dict:
        """从原始响应中提取 thinking、content 和 usage"""
        from .clients.openai import OpenAIClient

        parsed = OpenAIClient.parse_thoughts(raw_data)
        usage = None
        try:
            usage = raw_data.get("usage")
        except Exception:
            pass
        return {
            "content": parsed["answer"],
            "thinking": parsed["thought"] or None,
            "usage": usage,
        }

    def _create_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/api/config", self._handle_config)
        app.router.add_post("/api/generate", self._handle_generate)
        app.router.add_post("/api/generate/stream", self._handle_generate_stream)
        app.router.add_post("/api/generate/batch", self._handle_generate_batch)
        app.on_startup.append(self._on_startup)
        app.on_cleanup.append(self._on_cleanup)
        return app

    async def _on_startup(self, app: web.Application):
        from flexllm import LLMClient

        client_kwargs = {
            "model": self.config.model,
            "base_url": self.config.base_url,
            "api_key": self.config.api_key,
            "concurrency_limit": self.config.concurrency,
            "timeout": self.config.timeout,
        }
        if self.config.max_qps is not None:
            client_kwargs["max_qps"] = self.config.max_qps
        self._client = LLMClient(**client_kwargs)

    async def _on_cleanup(self, app: web.Application):
        if self._client:
            await self._client.aclose()

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def _handle_config(self, request: web.Request) -> web.Response:
        config_data = {
            "model": self.config.model,
            "base_url": self.config.base_url,
            "system_prompt": self.config.system_prompt,
            "user_template": self.config.user_template,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "thinking": self.config.thinking,
            "concurrency": self.config.concurrency,
            "max_qps": self.config.max_qps,
        }
        return web.json_response(config_data)

    async def _handle_generate(self, request: web.Request) -> web.Response:
        """POST /api/generate — 非流式生成"""
        start = time.perf_counter()
        try:
            data = await request.json()
        except Exception:
            logger.warning("POST /api/generate 400 invalid JSON")
            return web.json_response({"error": "invalid JSON"}, status=400)

        content = data.get("content")
        if not content:
            logger.warning("POST /api/generate 400 content is required")
            return web.json_response({"error": "content is required"}, status=400)

        logger.info("POST /api/generate input=%s", _truncate(content))
        messages = self._build_messages(content)
        kwargs = self._get_kwargs(data)

        try:
            result = await self._client.chat_completions(messages, return_raw=True, **kwargs)
            if hasattr(result, "status") and result.status == "error":
                error_msg = result.data.get("detail", result.data.get("error", str(result.data)))
                elapsed = time.perf_counter() - start
                logger.error("POST /api/generate 502 %.3fs error=%s", elapsed, error_msg)
                return web.json_response({"error": error_msg}, status=502)
            raw_data = result.data
            parsed = self._parse_result(raw_data)
            elapsed = time.perf_counter() - start
            logger.info(
                "POST /api/generate 200 %.3fs output=%s",
                elapsed,
                _truncate(parsed["content"]),
            )
            return web.json_response(parsed)
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error("POST /api/generate 500 %.3fs error=%s", elapsed, e)
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_generate_stream(self, request: web.Request) -> web.StreamResponse:
        """POST /api/generate/stream — 流式生成 (SSE)"""
        start = time.perf_counter()
        try:
            data = await request.json()
        except Exception:
            logger.warning("POST /api/generate/stream 400 invalid JSON")
            return web.json_response({"error": "invalid JSON"}, status=400)

        content = data.get("content")
        if not content:
            logger.warning("POST /api/generate/stream 400 content is required")
            return web.json_response({"error": "content is required"}, status=400)

        logger.info("POST /api/generate/stream input=%s", _truncate(content))
        messages = self._build_messages(content)
        kwargs = self._get_kwargs(data)

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        try:
            parser = ThinkTagParser()

            async for event in self._client.chat_completions_stream(
                messages, return_usage=True, **kwargs
            ):
                if event["type"] == "usage":
                    # usage 事件在最后，先 flush parser
                    for parsed in parser.flush():
                        sse = json.dumps(parsed, ensure_ascii=False)
                        await response.write(f"data: {sse}\n\n".encode("utf-8"))
                    done_event = json.dumps(
                        {"type": "done", "usage": event["usage"]}, ensure_ascii=False
                    )
                    await response.write(f"data: {done_event}\n\n".encode("utf-8"))
                    break

                for parsed in parser.feed(event):
                    if parsed["type"] in ("content", "thinking"):
                        sse = json.dumps(parsed, ensure_ascii=False)
                        await response.write(f"data: {sse}\n\n".encode("utf-8"))
            else:
                # 流结束但没有 usage 事件
                for parsed in parser.flush():
                    sse = json.dumps(parsed, ensure_ascii=False)
                    await response.write(f"data: {sse}\n\n".encode("utf-8"))
                done_event = json.dumps({"type": "done", "usage": None}, ensure_ascii=False)
                await response.write(f"data: {done_event}\n\n".encode("utf-8"))

        except Exception as e:
            error_event = json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
            await response.write(f"data: {error_event}\n\n".encode("utf-8"))
            elapsed = time.perf_counter() - start
            logger.error("POST /api/generate/stream error %.3fs error=%s", elapsed, e)

        elapsed = time.perf_counter() - start
        logger.info("POST /api/generate/stream 200 %.3fs", elapsed)
        await response.write_eof()
        return response

    async def _handle_generate_batch(self, request: web.Request) -> web.Response:
        """POST /api/generate/batch — 批量生成"""
        start = time.perf_counter()
        try:
            data = await request.json()
        except Exception:
            logger.warning("POST /api/generate/batch 400 invalid JSON")
            return web.json_response({"error": "invalid JSON"}, status=400)

        contents = data.get("contents")
        if not contents or not isinstance(contents, list):
            logger.warning("POST /api/generate/batch 400 contents (list) is required")
            return web.json_response({"error": "contents (list) is required"}, status=400)

        logger.info(
            "POST /api/generate/batch count=%d first=%s",
            len(contents),
            _truncate(contents[0]),
        )
        kwargs = self._get_kwargs(data)
        messages_list = [self._build_messages(c) for c in contents]

        try:
            tasks = [
                self._client.chat_completions(msgs, return_raw=True, **kwargs)
                for msgs in messages_list
            ]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

            results = []
            for r in raw_results:
                if isinstance(r, Exception):
                    results.append(
                        {"content": None, "thinking": None, "usage": None, "error": str(r)}
                    )
                elif hasattr(r, "status") and r.status == "error":
                    error_msg = r.data.get("detail", r.data.get("error", str(r.data)))
                    results.append(
                        {"content": None, "thinking": None, "usage": None, "error": error_msg}
                    )
                else:
                    parsed = self._parse_result(r.data)
                    results.append(parsed)

            elapsed = time.perf_counter() - start
            success = sum(1 for r in results if "error" not in r)
            logger.info(
                "POST /api/generate/batch 200 %.3fs count=%d success=%d",
                elapsed,
                len(contents),
                success,
            )
            return web.json_response({"results": results})
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error("POST /api/generate/batch 500 %.3fs error=%s", elapsed, e)
            return web.json_response({"error": str(e)}, status=500)

    def run(self):
        level = logging.INFO if self.config.verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        app = self._create_app()
        web.run_app(app, host=self.config.host, port=self.config.port)
