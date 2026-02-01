"""Chat Web Server

提供一个 Web 聊天界面，连接当前配置的 LLM 模型进行对话。

用法:
    flexllm chat-web                              # 使用默认配置
    flexllm chat-web -m gpt-4 -p 8080             # 指定模型和端口
    flexllm chat-web --base-url http://localhost:8001/v1 -m mock-model
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from aiohttp import web

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass
class ChatWebConfig:
    port: int = 8080
    host: str = "localhost"
    model: str = ""
    base_url: str = ""
    api_key: str = "EMPTY"
    system_prompt: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2048
    user_template: str | None = None
    thinking: bool | str | int | None = None


class ThinkTagParser:
    """流式解析 <think>...</think> 标签，将 content 拆分为 thinking 和 content 事件。

    同时透传已经是 thinking 类型的事件（来自 delta.reasoning 等原生字段）。
    """

    def __init__(self):
        self._in_think = False
        self._buffer = ""

    def feed(self, event: dict) -> list[dict]:
        """输入一个上游事件，输出零或多个下游事件。"""
        # 非 content 事件直接透传
        if event["type"] != "content":
            return [event]

        text = event["content"]
        self._buffer += text
        results = []

        while self._buffer:
            if not self._in_think:
                # 寻找 <think>
                idx = self._buffer.find(THINK_OPEN)
                if idx == -1:
                    # 可能 buffer 末尾是 <think> 的前缀，保留
                    safe, self._buffer = self._flush_safe(self._buffer, THINK_OPEN)
                    if safe:
                        results.append({"type": "content", "content": safe})
                    break
                else:
                    # <think> 之前的部分是 content
                    before = self._buffer[:idx]
                    if before:
                        results.append({"type": "content", "content": before})
                    self._buffer = self._buffer[idx + len(THINK_OPEN) :]
                    self._in_think = True
            else:
                # 寻找 </think>
                idx = self._buffer.find(THINK_CLOSE)
                if idx == -1:
                    safe, self._buffer = self._flush_safe(self._buffer, THINK_CLOSE)
                    if safe:
                        results.append({"type": "thinking", "content": safe})
                    break
                else:
                    before = self._buffer[:idx]
                    if before:
                        results.append({"type": "thinking", "content": before})
                    self._buffer = self._buffer[idx + len(THINK_CLOSE) :]
                    self._in_think = False

        return results

    def flush(self) -> list[dict]:
        """流结束时冲刷剩余 buffer。"""
        if not self._buffer:
            return []
        event_type = "thinking" if self._in_think else "content"
        result = [{"type": event_type, "content": self._buffer}]
        self._buffer = ""
        return result

    @staticmethod
    def _flush_safe(buf: str, tag: str) -> tuple[str, str]:
        """将 buffer 中安全的部分（不可能是 tag 前缀的部分）输出，保留可能的前缀。"""
        # tag 的最大前缀长度 = len(tag) - 1
        max_prefix = len(tag) - 1
        if len(buf) <= max_prefix:
            # 整个 buffer 都可能是 tag 前缀
            for i in range(len(buf)):
                if tag.startswith(buf[i:]):
                    return buf[:i], buf[i:]
            return buf, ""
        # 只检查末尾 max_prefix 个字符
        safe_end = len(buf) - max_prefix
        tail = buf[safe_end:]
        for i in range(len(tail)):
            if tag.startswith(tail[i:]):
                return buf[: safe_end + i], tail[i:]
        return buf, ""


class ChatWebServer:
    def __init__(self, config: ChatWebConfig):
        self.config = config
        self._app = None
        self._client = None

    def _create_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/api/config", self._handle_config)
        app.router.add_post("/api/chat", self._handle_chat)
        app.on_startup.append(self._on_startup)
        app.on_cleanup.append(self._on_cleanup)
        return app

    async def _on_startup(self, app: web.Application):
        from flexllm import LLMClient

        self._client = LLMClient(
            model=self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )

    async def _on_cleanup(self, app: web.Application):
        if self._client:
            await self._client.aclose()

    async def _handle_index(self, request: web.Request) -> web.Response:
        html_path = Path(__file__).parent / "data" / "chat_web.html"
        return web.Response(text=html_path.read_text(encoding="utf-8"), content_type="text/html")

    async def _handle_config(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "model": self.config.model,
                "temperature": self.config.temperature,
            }
        )

    async def _handle_chat(self, request: web.Request) -> web.StreamResponse:
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        messages = data.get("messages", [])
        if not messages:
            return web.json_response({"error": "messages is required"}, status=400)

        # 注入 system prompt
        if self.config.system_prompt:
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": self.config.system_prompt})

        # 应用 user_template
        if self.config.user_template:
            messages = [
                {**msg, "content": self.config.user_template.format(content=msg["content"])}
                if msg.get("role") == "user" and isinstance(msg.get("content"), str)
                else msg
                for msg in messages
            ]

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
            stream_kwargs = dict(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                return_usage=True,
            )
            if self.config.thinking is not None:
                stream_kwargs["thinking"] = self.config.thinking

            parser = ThinkTagParser()

            async for event in self._client.chat_completions_stream(messages, **stream_kwargs):
                for parsed in parser.feed(event):
                    if parsed["type"] in ("content", "thinking"):
                        sse = json.dumps(parsed, ensure_ascii=False)
                        await response.write(f"data: {sse}\n\n".encode("utf-8"))

            # 冲刷剩余 buffer
            for parsed in parser.flush():
                if parsed["type"] in ("content", "thinking"):
                    sse = json.dumps(parsed, ensure_ascii=False)
                    await response.write(f"data: {sse}\n\n".encode("utf-8"))

            done_event = json.dumps({"type": "done"})
            await response.write(f"data: {done_event}\n\n".encode("utf-8"))
        except Exception as e:
            error_event = json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
            await response.write(f"data: {error_event}\n\n".encode("utf-8"))

        await response.write_eof()
        return response

    def run(self):
        app = self._create_app()
        web.run_app(app, host=self.config.host, port=self.config.port)
