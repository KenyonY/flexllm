"""Serve Server

将 LLM 包装为 HTTP API 服务，适用于微调模型部署：
固定 system prompt 和 user template，调用方只需发送 content 文本。

用法:
    flexllm serve -m qwen-finetuned -s "你是助手" --user-template "[INST]{content}[/INST]"
    flexllm serve --thinking true -p 8000

API 端点:
    POST /api/generate             非流式生成
    POST /api/generate/stream      流式生成 (SSE)
    POST /api/generate/batch       批量生成
    POST /api/agent/run            Agent 单次执行 (需 --tools)
    POST /api/agent/run/stream     Agent 流式执行 (需 --tools)
    POST /api/agent/chat           Agent 多轮对话 (需 --tools)
    GET  /health                   健康检查
    GET  /api/config               查看当前配置
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
    # Agent 配置
    tools: str | None = None  # "code" / "all" / "read,edit,glob,grep,bash"
    max_rounds: int = 10


class AgentSessionManager:
    """Agent 多轮对话的 session 管理"""

    def __init__(self, ttl: int = 3600):
        self._sessions: dict[str, "AgentClient"] = {}
        self._timestamps: dict[str, float] = {}
        self._ttl = ttl

    def get_or_create(
        self,
        session_id: str,
        client,
        system: str | None,
        tool_registry,
        max_rounds: int,
    ) -> "AgentClient":
        from .agent.client import AgentClient

        self.cleanup_expired()
        if session_id not in self._sessions:
            self._sessions[session_id] = AgentClient(
                client=client,
                system=system,
                tool_registry=tool_registry,
                max_rounds=max_rounds,
            )
        self._timestamps[session_id] = time.time()
        return self._sessions[session_id]

    def cleanup_expired(self):
        now = time.time()
        expired = [sid for sid, ts in self._timestamps.items() if now - ts > self._ttl]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._timestamps.pop(sid, None)
        if expired:
            logger.info("清理过期 agent session: %d 个", len(expired))


class ServeServer:
    def __init__(self, config: ServeConfig):
        self.config = config
        self._client = None
        self._tool_registry = None
        self._session_mgr: AgentSessionManager | None = None

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
        # Agent 端点（仅在配置了 tools 时启用）
        if self.config.tools:
            app.router.add_post("/api/agent/run", self._handle_agent_run)
            app.router.add_post("/api/agent/run/stream", self._handle_agent_run_stream)
            app.router.add_post("/api/agent/chat", self._handle_agent_chat)
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

        # 初始化 Agent 工具
        if self.config.tools:
            from .cli.chat_helpers import _build_registry

            self._tool_registry = _build_registry(self.config.tools)
            self._session_mgr = AgentSessionManager()

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

    # ========== Agent 端点 ==========

    def _make_agent(self, system: str | None = None, max_rounds: int | None = None):
        """创建临时 AgentClient"""
        from .agent.client import AgentClient

        return AgentClient(
            client=self._client,
            system=system or self.config.system_prompt,
            tool_registry=self._tool_registry,
            max_rounds=max_rounds or self.config.max_rounds,
        )

    @staticmethod
    def _agent_result_to_dict(result) -> dict:
        return {
            "content": result.content,
            "rounds": result.rounds,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments, "result": tc.result}
                for tc in result.tool_calls
            ],
            "usage": result.usage,
        }

    async def _handle_agent_run(self, request: web.Request) -> web.Response:
        """POST /api/agent/run — 单次 agent 执行"""
        start = time.perf_counter()
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        content = data.get("content")
        if not content:
            return web.json_response({"error": "content is required"}, status=400)

        logger.info("POST /api/agent/run input=%s", _truncate(content))
        agent = self._make_agent(
            system=data.get("system"),
            max_rounds=data.get("max_rounds"),
        )

        try:
            result = await agent.run(content)
            elapsed = time.perf_counter() - start
            logger.info(
                "POST /api/agent/run 200 %.3fs rounds=%d output=%s",
                elapsed,
                result.rounds,
                _truncate(result.content),
            )
            return web.json_response(self._agent_result_to_dict(result))
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error("POST /api/agent/run 500 %.3fs error=%s", elapsed, e)
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_agent_run_stream(self, request: web.Request) -> web.StreamResponse:
        """POST /api/agent/run/stream — 流式 agent 执行 (SSE)"""
        start = time.perf_counter()
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        content = data.get("content")
        if not content:
            return web.json_response({"error": "content is required"}, status=400)

        logger.info("POST /api/agent/run/stream input=%s", _truncate(content))
        agent = self._make_agent(
            system=data.get("system"),
            max_rounds=data.get("max_rounds"),
        )

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

        async def _send_sse(event_data: dict):
            sse = json.dumps(event_data, ensure_ascii=False)
            await response.write(f"data: {sse}\n\n".encode("utf-8"))

        # 设置回调推送 SSE 事件
        agent.on_tool_call = lambda name, args: asyncio.ensure_future(
            _send_sse({"type": "tool_call", "name": name, "arguments": args})
        )
        agent.on_tool_result = lambda name, result: asyncio.ensure_future(
            _send_sse({"type": "tool_result", "name": name, "result": _truncate(result, 1000)})
        )

        try:
            result = await agent.run(content)
            await _send_sse(
                {
                    "type": "done",
                    **self._agent_result_to_dict(result),
                }
            )
        except Exception as e:
            await _send_sse({"type": "error", "message": str(e)})
            elapsed = time.perf_counter() - start
            logger.error("POST /api/agent/run/stream error %.3fs error=%s", elapsed, e)

        elapsed = time.perf_counter() - start
        logger.info("POST /api/agent/run/stream 200 %.3fs", elapsed)
        await response.write_eof()
        return response

    async def _handle_agent_chat(self, request: web.Request) -> web.Response:
        """POST /api/agent/chat — 多轮 agent 对话"""
        start = time.perf_counter()
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        content = data.get("content")
        session_id = data.get("session_id")
        if not content:
            return web.json_response({"error": "content is required"}, status=400)
        if not session_id:
            return web.json_response({"error": "session_id is required"}, status=400)

        logger.info("POST /api/agent/chat session=%s input=%s", session_id, _truncate(content))
        agent = self._session_mgr.get_or_create(
            session_id=session_id,
            client=self._client,
            system=data.get("system") or self.config.system_prompt,
            tool_registry=self._tool_registry,
            max_rounds=self.config.max_rounds,
        )

        try:
            result = await agent.chat(content)
            elapsed = time.perf_counter() - start
            logger.info(
                "POST /api/agent/chat 200 %.3fs session=%s rounds=%d",
                elapsed,
                session_id,
                result.rounds,
            )
            return web.json_response(self._agent_result_to_dict(result))
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error("POST /api/agent/chat 500 %.3fs error=%s", elapsed, e)
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
