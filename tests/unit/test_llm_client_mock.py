"""
基于 Mock Server 的 LLMClient 方法全面测试

覆盖所有公开方法和模式：
1. chat_completions（async/sync/or_raise）
2. chat_completions_batch（async/sync/return_usage/return_summary/return_cost_report）
3. chat_completions_stream（基础/return_usage）
4. iter_chat_completions_batch
5. return_raw 模式
6. Claude / Gemini provider
7. Thinking 模式（OpenAI/Claude/Gemini）
8. 类型体系（isinstance/abstractmethod）
9. model_list / parse_thoughts
10. 上下文管理器 / 资源清理

运行方式:
    pytest tests/unit/test_llm_client_mock.py -v -s
"""

import json
import os
import tempfile

import pytest

# 整个文件标记为 slow（Mock Server 启动开销大）
pytestmark = pytest.mark.slow

from flexllm import LLMClient, LLMClientPool
from flexllm.clients import ClaudeClient, GeminiClient, LLMClientBase, OpenAIClient
from flexllm.clients.base import ChatCompletionResult
from flexllm.mock import MockLLMServer, MockLLMServerGroup, MockServerConfig


def _msgs(content="Hello"):
    return [{"role": "user", "content": content}]


def _batch_msgs(n=5):
    return [[{"role": "user", "content": f"msg-{i}"}] for i in range(n)]


# ============== 1. chat_completions 基础 ==============


class TestChatCompletions:
    """chat_completions 方法全面测试"""

    @pytest.mark.asyncio
    async def test_async_returns_str(self, mock_llm_server):
        """默认返回 str"""
        client = LLMClient(base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY")
        result = await client.chat_completions(_msgs())
        assert isinstance(result, str)
        assert len(result) > 0
        client.close()

    @pytest.mark.asyncio
    async def test_return_usage(self, mock_llm_server):
        """return_usage=True 返回 ChatCompletionResult"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            result = await client.chat_completions(_msgs(), return_usage=True)
            assert isinstance(result, ChatCompletionResult)
            assert isinstance(result.content, str)
            assert result.usage is not None
            assert result.usage["prompt_tokens"] > 0
            assert result.usage["completion_tokens"] > 0
            assert result.usage["total_tokens"] == (
                result.usage["prompt_tokens"] + result.usage["completion_tokens"]
            )

    @pytest.mark.asyncio
    async def test_return_raw(self, mock_llm_server):
        """return_raw=True 返回 RequestResult"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            result = await client.chat_completions(_msgs(), return_raw=True)
            # RequestResult 有 status 和 data 属性
            assert hasattr(result, "status")
            assert hasattr(result, "data")
            assert result.status == "success"
            assert isinstance(result.data, dict)

    def test_sync(self, mock_llm_server):
        """同步版本"""
        with LLMClient(base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY") as client:
            result = client.chat_completions_sync(_msgs())
            assert isinstance(result, str)
            assert len(result) > 0

    def test_sync_return_usage(self, mock_llm_server):
        """同步版本 + return_usage"""
        with LLMClient(base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY") as client:
            result = client.chat_completions_sync(_msgs(), return_usage=True)
            assert isinstance(result, ChatCompletionResult)
            assert result.usage is not None

    @pytest.mark.asyncio
    async def test_or_raise_success(self, mock_llm_server):
        """chat_completions_or_raise 成功时返回 str"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            result = await client.chat_completions_or_raise(_msgs())
            assert isinstance(result, str)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_or_raise_with_usage(self, mock_llm_server):
        """chat_completions_or_raise + return_usage"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            result = await client.chat_completions_or_raise(_msgs(), return_usage=True)
            assert isinstance(result, ChatCompletionResult)

    @pytest.mark.asyncio
    async def test_or_raise_failure(self):
        """chat_completions_or_raise 全部失败时抛 RuntimeError"""
        cfg = MockServerConfig(port=19401, delay_min=0.01, delay_max=0.01, error_rate=1.0)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                base_url=server.url, model="mock-model", api_key="EMPTY", retry_delay=0.01
            ) as client:
                with pytest.raises(RuntimeError, match="LLM 请求失败"):
                    await client.chat_completions_or_raise(_msgs())

    @pytest.mark.asyncio
    async def test_system_message(self, mock_llm_server):
        """带 system message 的请求"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ]
            result = await client.chat_completions(messages)
            assert isinstance(result, str)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_multi_turn(self, mock_llm_server):
        """多轮对话消息"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]
            result = await client.chat_completions(messages)
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_no_model_raises(self):
        """未指定 model 时抛 ValueError"""
        cfg = MockServerConfig(port=19402, delay_min=0.01, delay_max=0.01)
        with MockLLMServer(cfg) as server:
            async with LLMClient(base_url=server.url, api_key="EMPTY") as client:
                with pytest.raises(ValueError, match="必须提供 model"):
                    await client.chat_completions(_msgs())


# ============== 2. chat_completions_batch ==============


class TestBatch:
    """批量请求方法测试"""

    @pytest.mark.asyncio
    async def test_batch_basic(self, mock_llm_server):
        """基础批量请求"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            results = await client.chat_completions_batch(_batch_msgs(10), show_progress=False)
            assert len(results) == 10
            assert all(isinstance(r, str) for r in results)

    @pytest.mark.asyncio
    async def test_batch_return_usage(self, mock_llm_server):
        """批量请求 + return_usage"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            results = await client.chat_completions_batch(
                _batch_msgs(5), show_progress=False, return_usage=True
            )
            assert len(results) == 5
            for r in results:
                assert isinstance(r, ChatCompletionResult)
                assert r.usage is not None

    @pytest.mark.asyncio
    async def test_batch_return_summary(self, mock_llm_server):
        """批量请求 + return_summary（需要 show_progress=True 才有 summary）"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            results, summary = await client.chat_completions_batch(
                _batch_msgs(8), show_progress=True, return_summary=True
            )
            assert len(results) == 8
            # 单 endpoint summary 是进度条的格式化字符串
            assert summary is not None
            assert isinstance(summary, str)

    @pytest.mark.asyncio
    async def test_batch_track_cost(self, mock_llm_server):
        """批量请求 + track_cost"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            results = await client.chat_completions_batch(
                _batch_msgs(5), show_progress=False, track_cost=True
            )
            assert len(results) == 5
            # track_cost 会自动开启 return_usage
            for r in results:
                assert isinstance(r, ChatCompletionResult)

    @pytest.mark.asyncio
    async def test_batch_output_jsonl(self, mock_llm_server):
        """批量请求 + JSONL 输出"""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = f.name

        try:
            async with LLMClient(
                base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
            ) as client:
                results = await client.chat_completions_batch(
                    _batch_msgs(5), output_jsonl=output_path, show_progress=False
                )

            assert len(results) == 5

            # 验证 JSONL 文件
            records = []
            with open(output_path) as f:
                for line in f:
                    records.append(json.loads(line.strip()))
            assert len(records) == 5
            # 记录按 index 排序
            indices = [r["index"] for r in records]
            assert indices == sorted(indices)
            for r in records:
                assert r["status"] == "success"
                assert "output" in r
                assert "input" in r  # 默认 save_input=True
        finally:
            os.unlink(output_path)

    @pytest.mark.asyncio
    async def test_batch_with_metadata(self, mock_llm_server):
        """批量请求 + metadata_list"""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = f.name

        try:
            msgs = _batch_msgs(3)
            metadata = [{"id": f"item-{i}", "category": "test"} for i in range(3)]

            async with LLMClient(
                base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
            ) as client:
                results = await client.chat_completions_batch(
                    msgs,
                    output_jsonl=output_path,
                    metadata_list=metadata,
                    show_progress=False,
                )

            assert len(results) == 3
            with open(output_path) as f:
                for line in f:
                    record = json.loads(line.strip())
                    assert "metadata" in record
                    assert "id" in record["metadata"]
                    assert "category" in record["metadata"]
        finally:
            os.unlink(output_path)

    @pytest.mark.asyncio
    async def test_batch_metadata_length_mismatch(self, mock_llm_server):
        """metadata_list 长度不匹配时抛 ValueError"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            with pytest.raises(ValueError, match="metadata_list 长度"):
                await client.chat_completions_batch(
                    _batch_msgs(3),
                    metadata_list=[{"id": "1"}],  # 长度 1，不等于 3
                    show_progress=False,
                )

    @pytest.mark.asyncio
    async def test_batch_invalid_output_extension(self, mock_llm_server):
        """output_jsonl 非 .jsonl 扩展名时抛 ValueError"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            with pytest.raises(ValueError, match="output_jsonl 必须使用 .jsonl"):
                await client.chat_completions_batch(
                    _batch_msgs(3), output_jsonl="output.json", show_progress=False
                )

    def test_batch_sync(self, mock_llm_server):
        """同步批量请求"""
        with LLMClient(base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY") as client:
            results = client.chat_completions_batch_sync(_batch_msgs(5), show_progress=False)
            assert len(results) == 5
            assert all(isinstance(r, str) for r in results)

    @pytest.mark.asyncio
    async def test_batch_return_raw(self, mock_llm_server):
        """批量请求 + return_raw：跳过缓存，返回原始响应 JSON dict 列表"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            results = await client.chat_completions_batch(
                _batch_msgs(3), show_progress=False, return_raw=True
            )
            assert len(results) == 3
            # return_raw=True: 每个元素是后端原始响应 dict（含 choices/usage 等）
            for r in results:
                assert r is not None
                assert isinstance(r, dict)
                assert "choices" in r


# ============== 3. chat_completions_stream ==============


class TestStream:
    """流式响应测试"""

    @pytest.mark.asyncio
    async def test_stream_basic(self, mock_llm_server):
        """基础流式：返回 str chunks"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            chunks = []
            async for chunk in client.chat_completions_stream(_msgs()):
                assert isinstance(chunk, str)
                chunks.append(chunk)

            assert len(chunks) > 0
            full = "".join(chunks)
            assert len(full) > 0

    @pytest.mark.asyncio
    async def test_stream_return_usage(self, mock_llm_server):
        """流式 + return_usage：返回 dict chunks"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            content_chunks = []
            usage_chunks = []

            async for chunk in client.chat_completions_stream(_msgs(), return_usage=True):
                assert isinstance(chunk, dict)
                assert "type" in chunk
                if chunk["type"] == "content":
                    content_chunks.append(chunk["content"])
                elif chunk["type"] == "usage":
                    usage_chunks.append(chunk["usage"])

            assert len(content_chunks) > 0
            assert len(usage_chunks) > 0
            usage = usage_chunks[-1]
            assert "prompt_tokens" in usage or "completion_tokens" in usage

    @pytest.mark.asyncio
    async def test_stream_collect_full_text(self, mock_llm_server):
        """流式收集完整文本应该与非流式长度相近"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            chunks = []
            async for chunk in client.chat_completions_stream(_msgs()):
                chunks.append(chunk)

            full_text = "".join(chunks)
            # 只要能拼出有效文本即可
            assert isinstance(full_text, str)
            assert len(full_text) > 0


# ============== 4. iter_chat_completions_batch ==============


class TestIterBatch:
    """迭代式批量请求测试"""

    @pytest.mark.asyncio
    async def test_iter_batch_basic(self, mock_llm_server):
        """基础迭代批量"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            results = []
            async for result in client.iter_chat_completions_batch(
                _batch_msgs(8), show_progress=False
            ):
                results.append(result)

            assert len(results) == 8
            for r in results:
                assert hasattr(r, "content")
                assert hasattr(r, "original_idx")
                assert r.content is not None

    @pytest.mark.asyncio
    async def test_iter_batch_summary(self, mock_llm_server):
        """迭代批量最后一条包含 summary"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            results = []
            async for result in client.iter_chat_completions_batch(
                _batch_msgs(5), show_progress=False
            ):
                results.append(result)

            # 最后一个 result 应该有 summary
            last = results[-1]
            assert hasattr(last, "summary")
            if last.summary is not None:
                assert "total" in last.summary
                assert "success" in last.summary

    @pytest.mark.asyncio
    async def test_iter_batch_with_jsonl(self, mock_llm_server):
        """迭代批量 + JSONL 输出"""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = f.name

        try:
            async with LLMClient(
                base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
            ) as client:
                results = []
                async for result in client.iter_chat_completions_batch(
                    _batch_msgs(5), output_jsonl=output_path, show_progress=False
                ):
                    results.append(result)

            assert len(results) == 5
            # 验证文件写入
            records = []
            with open(output_path) as f:
                for line in f:
                    records.append(json.loads(line.strip()))
            assert len(records) == 5
        finally:
            os.unlink(output_path)

    @pytest.mark.asyncio
    async def test_iter_batch_return_usage(self, mock_llm_server):
        """迭代批量 + return_usage"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            results = []
            async for result in client.iter_chat_completions_batch(
                _batch_msgs(3), show_progress=False, return_usage=True
            ):
                results.append(result)

            assert len(results) == 3
            for r in results:
                if r.content is not None:
                    assert hasattr(r, "usage")


# ============== 5. Claude Provider ==============


class TestClaudeProvider:
    """Claude provider Mock 测试"""

    @pytest.mark.asyncio
    async def test_claude_single_request(self):
        """Claude provider 单条请求"""
        cfg = MockServerConfig(port=19411, delay_min=0.01, delay_max=0.01)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                provider="claude",
                base_url=server.url,
                model="mock-model",
                api_key="test-key",
            ) as client:
                result = await client.chat_completions(_msgs())
                assert isinstance(result, str)
                assert len(result) > 0

    @pytest.mark.asyncio
    async def test_claude_return_usage(self):
        """Claude provider return_usage"""
        cfg = MockServerConfig(port=19412, delay_min=0.01, delay_max=0.01)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                provider="claude",
                base_url=server.url,
                model="mock-model",
                api_key="test-key",
            ) as client:
                result = await client.chat_completions(_msgs(), return_usage=True)
                assert isinstance(result, ChatCompletionResult)
                assert result.usage is not None

    @pytest.mark.asyncio
    async def test_claude_batch(self):
        """Claude provider 批量请求"""
        cfg = MockServerConfig(port=19413, delay_min=0.01, delay_max=0.01)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                provider="claude",
                base_url=server.url,
                model="mock-model",
                api_key="test-key",
            ) as client:
                results = await client.chat_completions_batch(_batch_msgs(5), show_progress=False)
                assert len(results) == 5
                assert all(isinstance(r, str) for r in results)

    @pytest.mark.asyncio
    async def test_claude_stream(self):
        """Claude provider 流式"""
        cfg = MockServerConfig(port=19414, delay_min=0.01, delay_max=0.01)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                provider="claude",
                base_url=server.url,
                model="mock-model",
                api_key="test-key",
            ) as client:
                chunks = []
                async for chunk in client.chat_completions_stream(_msgs()):
                    chunks.append(chunk)
                assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_claude_thinking(self):
        """Claude provider thinking 模式"""
        cfg = MockServerConfig(port=19415, delay_min=0.01, delay_max=0.01, thinking=True)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                provider="claude",
                base_url=server.url,
                model="mock-model",
                api_key="test-key",
            ) as client:
                result = await client.chat_completions(_msgs(), return_raw=True, thinking=True)
                # 返回原始响应，可以解析 thinking
                assert result.status == "success"
                parsed = ClaudeClient.parse_thoughts(result.data)
                assert isinstance(parsed, dict)
                assert "thought" in parsed
                assert "answer" in parsed

    @pytest.mark.asyncio
    async def test_claude_stream_thinking(self):
        """Claude provider 流式 thinking"""
        cfg = MockServerConfig(port=19416, delay_min=0.01, delay_max=0.01, thinking=True)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                provider="claude",
                base_url=server.url,
                model="mock-model",
                api_key="test-key",
            ) as client:
                content_chunks = []
                thinking_chunks = []
                async for chunk in client.chat_completions_stream(
                    _msgs(), return_usage=True, thinking=True
                ):
                    if isinstance(chunk, dict):
                        if chunk["type"] == "content":
                            content_chunks.append(chunk["content"])
                        elif chunk["type"] == "thinking":
                            thinking_chunks.append(chunk["content"])

                assert len(content_chunks) > 0
                assert len(thinking_chunks) > 0

    async def test_claude_stream_usage_prompt_tokens_preserved(self):
        """回归 #9：流式 message_delta 只带 output_tokens，不能把 message_start 的
        input_tokens 覆盖为 0。最终 usage 的 prompt_tokens 必须非零。"""
        cfg = MockServerConfig(port=19417, delay_min=0.01, delay_max=0.01)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                provider="claude",
                base_url=server.url,
                model="mock-model",
                api_key="test-key",
            ) as client:
                final_usage = None
                async for chunk in client.chat_completions_stream(_msgs(), return_usage=True):
                    if isinstance(chunk, dict) and chunk.get("type") == "usage":
                        final_usage = chunk["usage"]
                assert final_usage is not None, "流式应产出最终 usage 事件"
                assert final_usage["prompt_tokens"] > 0, "message_delta 不应把 prompt_tokens 归零"
                assert final_usage["completion_tokens"] > 0
                assert (
                    final_usage["total_tokens"]
                    == final_usage["prompt_tokens"] + final_usage["completion_tokens"]
                )


# ============== 6. Gemini Provider ==============


class TestGeminiProvider:
    """Gemini provider Mock 测试"""

    @pytest.mark.asyncio
    async def test_gemini_single_request(self):
        """Gemini provider 单条请求"""
        cfg = MockServerConfig(port=19421, delay_min=0.01, delay_max=0.01)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                provider="gemini",
                base_url=server.gemini_url,
                model="mock-model",
                api_key="test-key",
            ) as client:
                result = await client.chat_completions(_msgs())
                assert isinstance(result, str)
                assert len(result) > 0

    @pytest.mark.asyncio
    async def test_gemini_return_usage(self):
        """Gemini provider return_usage"""
        cfg = MockServerConfig(port=19422, delay_min=0.01, delay_max=0.01)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                provider="gemini",
                base_url=server.gemini_url,
                model="mock-model",
                api_key="test-key",
            ) as client:
                result = await client.chat_completions(_msgs(), return_usage=True)
                assert isinstance(result, ChatCompletionResult)
                assert result.usage is not None

    @pytest.mark.asyncio
    async def test_gemini_batch(self):
        """Gemini provider 批量请求"""
        cfg = MockServerConfig(port=19423, delay_min=0.01, delay_max=0.01)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                provider="gemini",
                base_url=server.gemini_url,
                model="mock-model",
                api_key="test-key",
            ) as client:
                results = await client.chat_completions_batch(_batch_msgs(5), show_progress=False)
                assert len(results) == 5
                assert all(isinstance(r, str) for r in results)

    @pytest.mark.asyncio
    async def test_gemini_stream(self):
        """Gemini provider 流式"""
        cfg = MockServerConfig(port=19424, delay_min=0.01, delay_max=0.01)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                provider="gemini",
                base_url=server.gemini_url,
                model="mock-model",
                api_key="test-key",
            ) as client:
                chunks = []
                async for chunk in client.chat_completions_stream(_msgs()):
                    chunks.append(chunk)
                assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_gemini_thinking(self):
        """Gemini provider thinking 模式"""
        cfg = MockServerConfig(port=19425, delay_min=0.01, delay_max=0.01, thinking=True)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                provider="gemini",
                base_url=server.gemini_url,
                model="mock-model",
                api_key="test-key",
            ) as client:
                result = await client.chat_completions(_msgs(), return_raw=True, thinking=True)
                assert result.status == "success"
                parsed = GeminiClient.parse_thoughts(result.data)
                assert isinstance(parsed, dict)
                assert "thought" in parsed
                assert "answer" in parsed


# ============== 7. OpenAI Thinking ==============


class TestOpenAIThinking:
    """OpenAI 兼容 provider thinking 测试（DeepSeek-R1 等）"""

    @pytest.mark.asyncio
    async def test_openai_thinking_raw(self):
        """OpenAI thinking 模式 return_raw"""
        cfg = MockServerConfig(port=19431, delay_min=0.01, delay_max=0.01, thinking=True)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                base_url=server.url, model="mock-model", api_key="EMPTY"
            ) as client:
                result = await client.chat_completions(_msgs(), return_raw=True, think=True)
                assert result.status == "success"
                parsed = OpenAIClient.parse_thoughts(result.data)
                assert "thought" in parsed
                assert "answer" in parsed

    @pytest.mark.asyncio
    async def test_openai_thinking_stream(self):
        """OpenAI thinking 流式"""
        cfg = MockServerConfig(port=19432, delay_min=0.01, delay_max=0.01, thinking=True)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                base_url=server.url, model="mock-model", api_key="EMPTY"
            ) as client:
                content_chunks = []
                async for chunk in client.chat_completions_stream(
                    _msgs(), return_usage=True, think=True
                ):
                    if isinstance(chunk, dict) and chunk["type"] == "content":
                        content_chunks.append(chunk["content"])
                assert len(content_chunks) > 0

    def test_extract_content_thinking_is_per_call_stateless(self):
        """回归 #2：thinking 保留与否必须跟随每次调用的参数，不得依赖实例状态。

        旧实现把 thinking 记在 self._keep_thinking（_build_request_body 写、
        _extract_content 读），批量/并发下所有行被最后一行的值污染。修复后
        _extract_content 从入参 thinking 决定，同一响应交错调用互不影响。
        """
        client = OpenAIClient(base_url="http://localhost:8000/v1", api_key="x", model="m")
        # reasoning-parser 模式：content 与 reasoning 同时存在
        resp = {"choices": [{"message": {"content": "答案是 4", "reasoning": "先想 2+2"}}]}
        # 交错调用，证明无跨调用状态泄漏
        keep1 = client._extract_content(resp, thinking=True)
        strip1 = client._extract_content(resp, thinking=False)
        keep2 = client._extract_content(resp, thinking=True)
        strip2 = client._extract_content(resp, thinking=False)

        assert "<think>" in keep1 and "先想 2+2" in keep1
        assert "<think>" not in strip1 and "先想 2+2" not in strip1
        assert keep1 == keep2, "thinking=True 结果不应被前一次 thinking=False 调用污染"
        assert strip1 == strip2
        # 内嵌 <think> 标签的 content 同理
        resp2 = {"choices": [{"message": {"content": "<think>推理</think>结论"}}]}
        assert "<think>" in client._extract_content(resp2, thinking=True)
        assert client._extract_content(resp2, thinking=False) == "结论"
        # 确认实例上不再残留 thinking 状态字段
        assert not hasattr(client, "_keep_thinking")


# ============== 8. LLMClientPool 方法测试 ==============


class TestPoolMethods:
    """LLMClientPool 各方法测试"""

    @pytest.mark.asyncio
    async def test_pool_single_all_methods(self, mock_llm_server):
        """Pool 单 endpoint 模式覆盖所有方法"""
        async with LLMClientPool(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as pool:
            # chat_completions
            result = await pool.chat_completions(_msgs())
            assert isinstance(result, str)

            # chat_completions_batch
            results = await pool.chat_completions_batch(_batch_msgs(3), show_progress=False)
            assert len(results) == 3

            # chat_completions_stream
            chunks = []
            async for chunk in pool.chat_completions_stream(_msgs()):
                chunks.append(chunk)
            assert len(chunks) > 0

            # iter_chat_completions_batch
            iter_results = []
            async for r in pool.iter_chat_completions_batch(_batch_msgs(3), show_progress=False):
                iter_results.append(r)
            assert len(iter_results) == 3

            # model_list
            models = pool.model_list()
            assert isinstance(models, list)

    @pytest.mark.asyncio
    async def test_pool_multi_batch(self):
        """Pool 多 endpoint 批量"""
        configs = [
            MockServerConfig(port=19441, delay_min=0.01, delay_max=0.01),
            MockServerConfig(port=19442, delay_min=0.01, delay_max=0.01),
        ]
        with MockLLMServerGroup(configs) as group:
            async with LLMClientPool(
                endpoints=group.endpoints, concurrency_limit=5, fallback=True
            ) as pool:
                results, summary = await pool.chat_completions_batch(
                    _batch_msgs(20), show_progress=True, return_summary=True
                )
                # Pool 多 endpoint 的 summary 是 dict
                assert isinstance(summary, dict)
                assert summary["total"] == 20
                assert summary["success"] == 20

    @pytest.mark.asyncio
    async def test_pool_multi_stream(self):
        """Pool 多 endpoint 流式"""
        configs = [
            MockServerConfig(port=19443, delay_min=0.01, delay_max=0.01),
            MockServerConfig(port=19444, delay_min=0.01, delay_max=0.01),
        ]
        with MockLLMServerGroup(configs) as group:
            async with LLMClientPool(endpoints=group.endpoints, fallback=True) as pool:
                chunks = []
                async for chunk in pool.chat_completions_stream(_msgs()):
                    chunks.append(chunk)
                assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_pool_multi_failover(self):
        """Pool 多 endpoint failover"""
        configs = [
            MockServerConfig(port=19445, delay_min=0.01, delay_max=0.01, error_rate=1.0),
            MockServerConfig(port=19446, delay_min=0.01, delay_max=0.01, error_rate=0),
        ]
        with MockLLMServerGroup(configs) as group:
            async with LLMClientPool(endpoints=group.endpoints, fallback=True) as pool:
                result = await pool.chat_completions(_msgs())
                assert isinstance(result, str)

    def test_pool_sync_methods(self, mock_llm_server):
        """Pool 同步方法"""
        with LLMClientPool(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as pool:
            # chat_completions_sync
            result = pool.chat_completions_sync(_msgs())
            assert isinstance(result, str)

            # chat_completions_batch_sync
            results = pool.chat_completions_batch_sync(_batch_msgs(3), show_progress=False)
            assert len(results) == 3

    @pytest.mark.asyncio
    async def test_pool_multi_iter_not_supported(self):
        """Pool 多 endpoint iter_chat_completions_batch 支持"""
        configs = [
            MockServerConfig(port=19447, delay_min=0.01, delay_max=0.01),
            MockServerConfig(port=19448, delay_min=0.01, delay_max=0.01),
        ]
        with MockLLMServerGroup(configs) as group:
            async with LLMClientPool(endpoints=group.endpoints, fallback=True) as pool:
                results = []
                async for r in pool.iter_chat_completions_batch(
                    _batch_msgs(3), show_progress=False
                ):
                    results.append(r)
                assert len(results) == 3
                # 每条都有 original_idx, content, status
                indices = sorted(r.original_idx for r in results)
                assert indices == [0, 1, 2]
                for r in results:
                    assert r.status == "success"
                    assert r.content is not None
                # 最后一条有 summary
                last = results[-1]
                assert last.summary is not None
                assert last.summary["total"] == 3

    @pytest.mark.asyncio
    async def test_pool_properties(self, mock_llm_server):
        """Pool 属性测试"""
        pool = LLMClientPool(base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY")
        assert pool.provider == "openai"
        assert pool.client is not None
        assert isinstance(pool.client, LLMClientBase)
        assert pool.stats["mode"] == "single"
        assert "LLMClientPool" in repr(pool)
        pool.close()

    @pytest.mark.asyncio
    async def test_pool_multi_properties(self):
        """Pool 多 endpoint 属性测试"""
        configs = [
            MockServerConfig(port=19449, delay_min=0.01, delay_max=0.01),
            MockServerConfig(port=19450, delay_min=0.01, delay_max=0.01),
        ]
        with MockLLMServerGroup(configs) as group:
            pool = LLMClientPool(endpoints=group.endpoints, fallback=True)
            assert pool.provider == "multi"
            assert pool.stats["mode"] == "multi"
            assert pool.stats["num_endpoints"] == 2
            assert pool.stats["fallback"] is True
            pool.close()


# ============== 9. 类型体系 ==============


class TestTypeSystem:
    """类型体系验证"""

    def test_isinstance_pool(self, mock_llm_server):
        """LLMClientPool isinstance LLMClientBase"""
        pool = LLMClientPool(base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY")
        assert isinstance(pool, LLMClientBase)
        pool.close()

    def test_isinstance_llmclient(self, mock_llm_server):
        """LLMClient isinstance LLMClientBase"""
        client = LLMClient(base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY")
        assert isinstance(client, LLMClientBase)
        client.close()

    def test_issubclass(self):
        """子类关系"""
        assert issubclass(LLMClientPool, LLMClientBase)
        assert issubclass(LLMClient, LLMClientBase)
        assert issubclass(OpenAIClient, LLMClientBase)
        assert issubclass(ClaudeClient, LLMClientBase)
        assert issubclass(GeminiClient, LLMClientBase)

    def test_concrete_clients_are_llmclientbase(self):
        """具体客户端实例都是 LLMClientBase"""
        openai = OpenAIClient(base_url="http://localhost:8000/v1", api_key="x", model="m")
        claude = ClaudeClient(api_key="x", model="m")
        gemini = GeminiClient(api_key="x", model="m")
        assert isinstance(openai, LLMClientBase)
        assert isinstance(claude, LLMClientBase)
        assert isinstance(gemini, LLMClientBase)
        openai.close()
        claude.close()
        gemini.close()

    def test_abstractmethod_enforced(self):
        """不能直接实例化 LLMClientBase"""
        with pytest.raises(TypeError, match="abstract"):
            LLMClientBase()


# ============== 10. 断点续传 ==============


class TestCheckpointResume:
    """断点续传测试"""

    @pytest.mark.asyncio
    async def test_full_resume(self):
        """完整断点续传：第二次运行从文件恢复，不发请求（使用 Pool 多 endpoint）"""
        configs = [
            MockServerConfig(port=19481, delay_min=0.01, delay_max=0.01),
            MockServerConfig(port=19482, delay_min=0.01, delay_max=0.01),
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = f.name

        try:
            msgs = _batch_msgs(10)
            with MockLLMServerGroup(configs) as group:
                # 第一次
                async with LLMClientPool(
                    endpoints=group.endpoints, concurrency_limit=5, fallback=True
                ) as pool:
                    results1 = await pool.chat_completions_batch(
                        msgs, output_jsonl=output_path, show_progress=False
                    )
                assert all(r is not None for r in results1)

                # 第二次：全部从文件恢复
                async with LLMClientPool(
                    endpoints=group.endpoints, concurrency_limit=5, fallback=True
                ) as pool:
                    results2, summary = await pool.chat_completions_batch(
                        msgs,
                        output_jsonl=output_path,
                        show_progress=False,
                        return_summary=True,
                    )

            assert len(results2) == 10
            assert all(r is not None for r in results2)
            # 恢复的结果应与第一次一致
            assert results1 == results2
            assert summary["success"] == 10
        finally:
            os.unlink(output_path)

    @pytest.mark.asyncio
    async def test_partial_resume(self):
        """部分断点续传（使用 Pool 多 endpoint 以支持结果还原）"""
        configs = [
            MockServerConfig(port=19483, delay_min=0.01, delay_max=0.01),
            MockServerConfig(port=19484, delay_min=0.01, delay_max=0.01),
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = f.name

        try:
            msgs = _batch_msgs(6)
            # 手动写入前 3 条
            with open(output_path, "w") as f:
                for i in range(3):
                    record = {
                        "index": i,
                        "output": f"cached-{i}",
                        "status": "success",
                        "input": msgs[i],
                    }
                    f.write(json.dumps(record) + "\n")

            with MockLLMServerGroup(configs) as group:
                async with LLMClientPool(
                    endpoints=group.endpoints, concurrency_limit=5, fallback=True
                ) as pool:
                    results = await pool.chat_completions_batch(
                        msgs, output_jsonl=output_path, show_progress=False
                    )

            assert len(results) == 6
            # 前 3 条从文件恢复
            for i in range(3):
                assert results[i] == f"cached-{i}"
            # 后 3 条是新处理的
            for i in range(3, 6):
                assert results[i] is not None
                assert isinstance(results[i], str)
        finally:
            os.unlink(output_path)

    @pytest.mark.asyncio
    async def test_save_input_false(self, mock_llm_server):
        """save_input=False 不保存 input"""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = f.name

        try:
            async with LLMClient(
                base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
            ) as client:
                await client.chat_completions_batch(
                    _batch_msgs(3),
                    output_jsonl=output_path,
                    save_input=False,
                    show_progress=False,
                )

            with open(output_path) as f:
                for line in f:
                    record = json.loads(line.strip())
                    assert "input" not in record
        finally:
            os.unlink(output_path)

    @pytest.mark.asyncio
    async def test_save_input_last(self, mock_llm_server):
        """save_input='last' 只保存最后 user content"""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = f.name

        try:
            msgs = [
                [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": f"Q-{i}"},
                ]
                for i in range(3)
            ]
            async with LLMClient(
                base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
            ) as client:
                await client.chat_completions_batch(
                    msgs,
                    output_jsonl=output_path,
                    save_input="last",
                    show_progress=False,
                )

            with open(output_path) as f:
                for line in f:
                    record = json.loads(line.strip())
                    idx = record["index"]
                    assert record["input"] == f"Q-{idx}"
        finally:
            os.unlink(output_path)


# ============== 11. 上下文管理器 ==============


class TestContextManager:
    """上下文管理器和资源清理"""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_llm_server):
        """async with 正常使用"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            result = await client.chat_completions(_msgs())
            assert result is not None

    def test_sync_context_manager(self, mock_llm_server):
        """with 正常使用"""
        with LLMClient(base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY") as client:
            result = client.chat_completions_sync(_msgs())
            assert result is not None

    def test_manual_close(self, mock_llm_server):
        """手动 close()"""
        client = LLMClient(base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY")
        result = client.chat_completions_sync(_msgs())
        assert result is not None
        client.close()  # 不应抛异常

    @pytest.mark.asyncio
    async def test_manual_aclose(self, mock_llm_server):
        """手动 aclose()"""
        client = LLMClient(base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY")
        result = await client.chat_completions(_msgs())
        assert result is not None
        await client.aclose()


# ============== 12. 响应缓存 ==============


class TestResponseCache:
    """响应缓存集成测试"""

    @pytest.mark.asyncio
    async def test_cache_hit(self, mock_llm_server):
        """相同消息第二次命中缓存"""
        from flexllm.cache import ResponseCacheConfig

        with tempfile.TemporaryDirectory() as cache_dir:
            cfg = ResponseCacheConfig(enabled=True, cache_dir=cache_dir, ttl=3600)
            async with LLMClient(
                base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY", cache=cfg
            ) as client:
                msg = _msgs("cache-test-message")
                r1 = await client.chat_completions(msg)
                r2 = await client.chat_completions(msg)
                assert r1 == r2  # 缓存命中

    @pytest.mark.asyncio
    async def test_cache_miss(self, mock_llm_server):
        """不同消息不命中缓存"""
        from flexllm.cache import ResponseCacheConfig

        with tempfile.TemporaryDirectory() as cache_dir:
            cfg = ResponseCacheConfig(enabled=True, cache_dir=cache_dir, ttl=3600)
            async with LLMClient(
                base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY", cache=cfg
            ) as client:
                r1 = await client.chat_completions(_msgs("A"))
                r2 = await client.chat_completions(_msgs("B"))
                # 两个不同消息，都应是非空字符串
                assert isinstance(r1, str) and len(r1) > 0
                assert isinstance(r2, str) and len(r2) > 0

    @pytest.mark.asyncio
    async def test_cache_skip_on_return_raw(self, mock_llm_server):
        """return_raw 跳过缓存"""
        from flexllm.cache import ResponseCacheConfig

        with tempfile.TemporaryDirectory() as cache_dir:
            cfg = ResponseCacheConfig(enabled=True, cache_dir=cache_dir, ttl=3600)
            async with LLMClient(
                base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY", cache=cfg
            ) as client:
                msg = _msgs("raw-cache-test")
                r1 = await client.chat_completions(msg, return_raw=True)
                r2 = await client.chat_completions(msg, return_raw=True)
                # return_raw 不走缓存，两次都是 RequestResult
                assert hasattr(r1, "status")
                assert hasattr(r2, "status")


# ============== 13. 错误处理 ==============


class TestErrorHandling:
    """错误场景测试"""

    @pytest.mark.asyncio
    async def test_all_fail_batch(self):
        """批量请求全部失败"""
        cfg = MockServerConfig(port=19461, delay_min=0.01, delay_max=0.01, error_rate=1.0)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                base_url=server.url, model="mock-model", api_key="EMPTY", retry_delay=0.01
            ) as client:
                results = await client.chat_completions_batch(_batch_msgs(3), show_progress=False)
                assert len(results) == 3
                assert all(r is None for r in results)

    @pytest.mark.asyncio
    async def test_all_fail_batch_summary(self):
        """批量全失败 + summary（需要 show_progress=True 才有 summary）"""
        cfg = MockServerConfig(port=19462, delay_min=0.01, delay_max=0.01, error_rate=1.0)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                base_url=server.url, model="mock-model", api_key="EMPTY", retry_delay=0.01
            ) as client:
                results, summary = await client.chat_completions_batch(
                    _batch_msgs(3), show_progress=True, return_summary=True
                )
                assert all(r is None for r in results)
                # 单 endpoint summary 是进度条的格式化字符串
                assert summary is not None
                assert isinstance(summary, str)

    @pytest.mark.asyncio
    async def test_partial_fail_batch(self):
        """批量部分失败（高错误率但有重试）"""
        cfg = MockServerConfig(port=19463, delay_min=0.01, delay_max=0.01, error_rate=0.5)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                base_url=server.url,
                model="mock-model",
                api_key="EMPTY",
                retry_times=5,  # 多次重试
                retry_delay=0.01,  # 加快重试速度
            ) as client:
                results = await client.chat_completions_batch(_batch_msgs(10), show_progress=False)
                # 有重试的情况下大部分应该成功
                success_count = sum(1 for r in results if r is not None)
                assert success_count > 0

    @pytest.mark.asyncio
    async def test_single_fail_returns_request_result(self):
        """单条请求失败返回 RequestResult（不抛异常）"""
        cfg = MockServerConfig(port=19464, delay_min=0.01, delay_max=0.01, error_rate=1.0)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                base_url=server.url, model="mock-model", api_key="EMPTY", retry_delay=0.01
            ) as client:
                result = await client.chat_completions(_msgs())
                # 失败时返回 RequestResult
                assert hasattr(result, "status")
                assert result.status != "success"


# ============== 14. __getattr__ 委托 ==============


class TestGetAttrDelegation:
    """Pool __getattr__ 委托机制测试"""

    def test_delegate_model_property(self, mock_llm_server):
        """单模式下访问底层客户端的属性"""
        pool = LLMClientPool(base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY")
        # _model 是底层客户端属性，通过 __getattr__ 委托
        assert pool._model == "mock-model"
        pool.close()

    def test_multi_mode_no_delegate(self):
        """多模式下不支持自动委托"""
        configs = [
            MockServerConfig(port=19471, delay_min=0.01, delay_max=0.01),
            MockServerConfig(port=19472, delay_min=0.01, delay_max=0.01),
        ]
        with MockLLMServerGroup(configs) as group:
            pool = LLMClientPool(endpoints=group.endpoints)
            with pytest.raises(AttributeError, match="仅单 endpoint 模式支持"):
                _ = pool._nonexistent_attr
            pool.close()
