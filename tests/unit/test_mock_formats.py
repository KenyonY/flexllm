"""Mock Server 多格式支持单元测试

测试 OpenAI / Claude / Gemini 三种 API 格式的响应，以及思考内容的返回。
"""

import json

import aiohttp
import pytest

from flexllm.mock import MockLLMServer, MockServerConfig

# 使用高端口避免与其他测试冲突
BASE_PORT = 19501


def _port():
    """每次调用返回递增的端口号"""
    _port.n += 1
    return BASE_PORT + _port.n


_port.n = 0


# ── OpenAI 格式 ──


class TestOpenAIFormat:
    @pytest.mark.asyncio
    async def test_non_stream_basic(self):
        """OpenAI 非流式基础响应"""
        port = _port()
        with MockLLMServer(MockServerConfig(port=port, delay_min=0, delay_max=0)) as server:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{server.url}/chat/completions",
                    json={"model": "mock", "messages": [{"role": "user", "content": "hi"}]},
                ) as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert data["object"] == "chat.completion"
                    msg = data["choices"][0]["message"]
                    assert msg["role"] == "assistant"
                    assert len(msg["content"]) > 0
                    assert "reasoning" not in msg  # 默认不含思考
                    assert "usage" in data

    @pytest.mark.asyncio
    async def test_non_stream_with_thinking_config(self):
        """OpenAI 非流式：config.thinking=True 时包含 reasoning"""
        port = _port()
        config = MockServerConfig(port=port, delay_min=0, delay_max=0, thinking=True)
        with MockLLMServer(config) as server:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{server.url}/chat/completions",
                    json={"model": "mock", "messages": [{"role": "user", "content": "hi"}]},
                ) as resp:
                    data = await resp.json()
                    msg = data["choices"][0]["message"]
                    assert "reasoning" in msg
                    assert len(msg["reasoning"]) > 0
                    assert len(msg["content"]) > 0

    @pytest.mark.asyncio
    async def test_non_stream_with_think_param(self):
        """OpenAI 非流式：请求中 think=true 动态触发思考"""
        port = _port()
        with MockLLMServer(MockServerConfig(port=port, delay_min=0, delay_max=0)) as server:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{server.url}/chat/completions",
                    json={
                        "model": "mock",
                        "messages": [{"role": "user", "content": "hi"}],
                        "think": True,
                    },
                ) as resp:
                    data = await resp.json()
                    assert "reasoning" in data["choices"][0]["message"]

    @pytest.mark.asyncio
    async def test_stream_basic(self):
        """OpenAI 流式基础响应"""
        port = _port()
        with MockLLMServer(MockServerConfig(port=port, delay_min=0, delay_max=0)) as server:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{server.url}/chat/completions",
                    json={
                        "model": "mock",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                ) as resp:
                    assert resp.status == 200
                    content_parts = []
                    async for line in resp.content:
                        text = line.decode("utf-8").strip()
                        if text.startswith("data: ") and text != "data: [DONE]":
                            chunk = json.loads(text[6:])
                            delta = chunk["choices"][0]["delta"]
                            if "content" in delta:
                                content_parts.append(delta["content"])
                    assert len(content_parts) > 0

    @pytest.mark.asyncio
    async def test_stream_with_thinking(self):
        """OpenAI 流式：thinking 时先发 reasoning 再发 content"""
        port = _port()
        config = MockServerConfig(port=port, delay_min=0, delay_max=0, thinking=True)
        with MockLLMServer(config) as server:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{server.url}/chat/completions",
                    json={
                        "model": "mock",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                ) as resp:
                    reasoning_parts = []
                    content_parts = []
                    async for line in resp.content:
                        text = line.decode("utf-8").strip()
                        if text.startswith("data: ") and text != "data: [DONE]":
                            chunk = json.loads(text[6:])
                            delta = chunk["choices"][0]["delta"]
                            if "reasoning" in delta:
                                reasoning_parts.append(delta["reasoning"])
                            if "content" in delta:
                                content_parts.append(delta["content"])
                    assert len(reasoning_parts) > 0
                    assert len(content_parts) > 0


# ── Claude 格式 ──


class TestClaudeFormat:
    @pytest.mark.asyncio
    async def test_non_stream_basic(self):
        """Claude 非流式基础响应"""
        port = _port()
        with MockLLMServer(MockServerConfig(port=port, delay_min=0, delay_max=0)) as server:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{server.url}/messages",
                    json={
                        "model": "mock",
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                    headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
                ) as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert data["type"] == "message"
                    assert data["role"] == "assistant"
                    assert len(data["content"]) == 1
                    assert data["content"][0]["type"] == "text"
                    assert len(data["content"][0]["text"]) > 0
                    assert "usage" in data
                    assert "input_tokens" in data["usage"]

    @pytest.mark.asyncio
    async def test_non_stream_with_thinking(self):
        """Claude 非流式：thinking 请求参数触发 thinking block"""
        port = _port()
        with MockLLMServer(MockServerConfig(port=port, delay_min=0, delay_max=0)) as server:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{server.url}/messages",
                    json={
                        "model": "mock",
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": "hi"}],
                        "thinking": {"type": "enabled", "budget_tokens": 10000},
                    },
                    headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
                ) as resp:
                    data = await resp.json()
                    assert len(data["content"]) == 2
                    assert data["content"][0]["type"] == "thinking"
                    assert len(data["content"][0]["thinking"]) > 0
                    assert data["content"][1]["type"] == "text"
                    assert len(data["content"][1]["text"]) > 0

    @pytest.mark.asyncio
    async def test_stream_basic(self):
        """Claude 流式基础响应"""
        port = _port()
        with MockLLMServer(MockServerConfig(port=port, delay_min=0, delay_max=0)) as server:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{server.url}/messages",
                    json={
                        "model": "mock",
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                    headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
                ) as resp:
                    assert resp.status == 200
                    events = []
                    text_parts = []
                    async for line in resp.content:
                        text = line.decode("utf-8").strip()
                        if text.startswith("data: "):
                            data = json.loads(text[6:])
                            events.append(data.get("type", ""))
                            if data.get("type") == "content_block_delta":
                                delta = data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text_parts.append(delta["text"])
                    assert "message_start" in events
                    assert "content_block_start" in events
                    assert "content_block_delta" in events
                    assert "message_stop" in events
                    assert len(text_parts) > 0

    @pytest.mark.asyncio
    async def test_stream_with_thinking(self):
        """Claude 流式：thinking 模式包含 thinking_delta 和 text_delta"""
        port = _port()
        config = MockServerConfig(port=port, delay_min=0, delay_max=0, thinking=True)
        with MockLLMServer(config) as server:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{server.url}/messages",
                    json={
                        "model": "mock",
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                    headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
                ) as resp:
                    thinking_parts = []
                    text_parts = []
                    async for line in resp.content:
                        text = line.decode("utf-8").strip()
                        if text.startswith("data: "):
                            data = json.loads(text[6:])
                            if data.get("type") == "content_block_delta":
                                delta = data.get("delta", {})
                                if delta.get("type") == "thinking_delta":
                                    thinking_parts.append(delta["thinking"])
                                elif delta.get("type") == "text_delta":
                                    text_parts.append(delta["text"])
                    assert len(thinking_parts) > 0
                    assert len(text_parts) > 0


# ── Gemini 格式 ──


class TestGeminiFormat:
    @pytest.mark.asyncio
    async def test_non_stream_basic(self):
        """Gemini 非流式基础响应"""
        port = _port()
        with MockLLMServer(MockServerConfig(port=port, delay_min=0, delay_max=0)) as server:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{server.gemini_url}/models/mock-model:generateContent?key=test",
                    json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
                ) as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert "candidates" in data
                    parts = data["candidates"][0]["content"]["parts"]
                    assert len(parts) == 1
                    assert len(parts[0]["text"]) > 0
                    assert "thought" not in parts[0]  # 默认不含思考
                    assert "usageMetadata" in data

    @pytest.mark.asyncio
    async def test_non_stream_with_thinking(self):
        """Gemini 非流式：includeThoughts 触发 thought parts"""
        port = _port()
        with MockLLMServer(MockServerConfig(port=port, delay_min=0, delay_max=0)) as server:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{server.gemini_url}/models/mock-model:generateContent?key=test",
                    json={
                        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                        "generationConfig": {
                            "thinkingConfig": {"includeThoughts": True},
                        },
                    },
                ) as resp:
                    data = await resp.json()
                    parts = data["candidates"][0]["content"]["parts"]
                    assert len(parts) == 2
                    assert parts[0].get("thought") is True
                    assert len(parts[0]["text"]) > 0
                    assert "thought" not in parts[1] or parts[1].get("thought") is not True
                    assert len(parts[1]["text"]) > 0

    @pytest.mark.asyncio
    async def test_stream_basic(self):
        """Gemini 流式基础响应"""
        port = _port()
        with MockLLMServer(MockServerConfig(port=port, delay_min=0, delay_max=0)) as server:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{server.gemini_url}/models/mock-model:streamGenerateContent?key=test&alt=sse",
                    json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
                ) as resp:
                    assert resp.status == 200
                    text_parts = []
                    has_usage = False
                    async for line in resp.content:
                        text = line.decode("utf-8").strip()
                        if text.startswith("data: "):
                            chunk = json.loads(text[6:])
                            parts = (
                                chunk.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                            )
                            for p in parts:
                                if "text" in p and not p.get("thought"):
                                    text_parts.append(p["text"])
                            if "usageMetadata" in chunk:
                                has_usage = True
                    assert len(text_parts) > 0
                    assert has_usage

    @pytest.mark.asyncio
    async def test_stream_with_thinking(self):
        """Gemini 流式：thinking 模式先发 thought chunks 再发正常 chunks"""
        port = _port()
        config = MockServerConfig(port=port, delay_min=0, delay_max=0, thinking=True)
        with MockLLMServer(config) as server:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{server.gemini_url}/models/mock-model:streamGenerateContent?key=test&alt=sse",
                    json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
                ) as resp:
                    thought_parts = []
                    text_parts = []
                    async for line in resp.content:
                        text = line.decode("utf-8").strip()
                        if text.startswith("data: "):
                            chunk = json.loads(text[6:])
                            parts = (
                                chunk.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                            )
                            for p in parts:
                                if p.get("thought"):
                                    thought_parts.append(p["text"])
                                elif "text" in p:
                                    text_parts.append(p["text"])
                    assert len(thought_parts) > 0
                    assert len(text_parts) > 0


# ── _should_include_thinking 判断逻辑 ──


class TestShouldIncludeThinking:
    def test_config_thinking_true(self):
        config = MockServerConfig(thinking=True)
        server = MockLLMServer.__new__(MockLLMServer)
        server.config = config
        assert server._should_include_thinking({}) is True

    def test_openai_think_param(self):
        config = MockServerConfig(thinking=False)
        server = MockLLMServer.__new__(MockLLMServer)
        server.config = config
        assert server._should_include_thinking({"think": True}) is True
        assert server._should_include_thinking({"think": False}) is False
        assert server._should_include_thinking({}) is False

    def test_claude_thinking_param(self):
        config = MockServerConfig(thinking=False)
        server = MockLLMServer.__new__(MockLLMServer)
        server.config = config
        assert (
            server._should_include_thinking(
                {"thinking": {"type": "enabled", "budget_tokens": 10000}}
            )
            is True
        )
        assert server._should_include_thinking({"thinking": {"type": "disabled"}}) is False

    def test_gemini_thinking_param(self):
        config = MockServerConfig(thinking=False)
        server = MockLLMServer.__new__(MockLLMServer)
        server.config = config
        assert (
            server._should_include_thinking(
                {"generationConfig": {"thinkingConfig": {"includeThoughts": True}}}
            )
            is True
        )
        assert (
            server._should_include_thinking(
                {"generationConfig": {"thinkingConfig": {"includeThoughts": False}}}
            )
            is False
        )
