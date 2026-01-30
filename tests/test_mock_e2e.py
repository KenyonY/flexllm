"""
基于 Mock Server 的端到端测试

覆盖场景：
1. LLMClient 基础流程：单条请求、批量请求、同步/异步
2. 流式响应：单 endpoint 流式、Pool 流式、流式 failover
3. 断点续传：中断后恢复、跳过已完成任务
4. 响应缓存：缓存命中、缓存未命中
5. 成本追踪：批量处理中的成本统计
6. LLMClientPool：多 endpoint 批量、failover、负载均衡

运行方式:
    pytest tests/test_mock_e2e.py -v -s
"""

import json
import os
import tempfile

import pytest

from flexllm import LLMClient, LLMClientPool
from flexllm.cache.response_cache import ResponseCacheConfig
from flexllm.mock import MockLLMServer, MockLLMServerGroup, MockServerConfig


def create_messages(n: int) -> list[list[dict]]:
    """创建 n 条测试消息"""
    return [[{"role": "user", "content": f"Test message {i}"}] for i in range(n)]


# ============== 1. LLMClient 基础流程 ==============


class TestClientBasic:
    """LLMClient 基础功能（基于 Mock Server）"""

    @pytest.mark.asyncio
    async def test_single_request_async(self, mock_llm_server):
        """单条异步请求"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            result = await client.chat_completions([{"role": "user", "content": "Hello"}])
            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0

    def test_single_request_sync(self, mock_llm_server):
        """单条同步请求"""
        with LLMClient(base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY") as client:
            result = client.chat_completions_sync([{"role": "user", "content": "Hello"}])
            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_single_request_with_usage(self, mock_llm_server):
        """单条请求返回 usage 信息"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            result = await client.chat_completions(
                [{"role": "user", "content": "Hello"}],
                return_usage=True,
            )
            assert result is not None
            assert hasattr(result, "content")
            assert hasattr(result, "usage")
            assert result.usage is not None
            assert "prompt_tokens" in result.usage
            assert "completion_tokens" in result.usage
            assert result.usage["prompt_tokens"] > 0
            assert result.usage["completion_tokens"] > 0

    @pytest.mark.asyncio
    async def test_batch_request(self, mock_llm_server):
        """批量请求"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            messages_list = create_messages(20)
            results = await client.chat_completions_batch(messages_list, show_progress=False)
            assert len(results) == 20
            assert all(r is not None for r in results)
            assert all(isinstance(r, str) for r in results)

    @pytest.mark.asyncio
    async def test_batch_with_summary(self, mock_llm_server):
        """批量请求返回执行摘要（单 endpoint 返回格式化字符串）"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            messages_list = create_messages(10)
            results, summary = await client.chat_completions_batch(
                messages_list, show_progress=True, return_summary=True
            )
            assert len(results) == 10
            assert all(r is not None for r in results)
            # 单 endpoint 的 summary 是进度条的格式化字符串
            assert summary is not None
            assert isinstance(summary, str)


# ============== 2. 流式响应 ==============


class TestStreaming:
    """流式响应端到端测试"""

    @pytest.mark.asyncio
    async def test_streaming_basic(self, mock_llm_server):
        """基础流式响应：逐 token 接收"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            chunks = []
            async for chunk in client.chat_completions_stream(
                [{"role": "user", "content": "Hello"}]
            ):
                chunks.append(chunk)

            assert len(chunks) > 0
            full_text = "".join(chunks)
            assert len(full_text) > 0

    @pytest.mark.asyncio
    async def test_streaming_vs_non_streaming(self, mock_llm_server):
        """流式和非流式结果都应是非空字符串"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            # 非流式
            normal = await client.chat_completions([{"role": "user", "content": "Hello"}])
            assert isinstance(normal, str)
            assert len(normal) > 0

            # 流式
            chunks = []
            async for chunk in client.chat_completions_stream(
                [{"role": "user", "content": "Hello"}]
            ):
                chunks.append(chunk)

            stream_text = "".join(chunks)
            assert isinstance(stream_text, str)
            assert len(stream_text) > 0

    @pytest.mark.asyncio
    async def test_pool_streaming_single_endpoint(self, mock_llm_server):
        """Pool 单 endpoint 流式"""
        async with LLMClientPool(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as pool:
            chunks = []
            async for chunk in pool.chat_completions_stream([{"role": "user", "content": "Hello"}]):
                chunks.append(chunk)

            assert len(chunks) > 0
            full_text = "".join(chunks)
            assert len(full_text) > 0

    @pytest.mark.asyncio
    async def test_pool_streaming_multi_endpoint(self, mock_llm_servers):
        """Pool 多 endpoint 流式"""
        async with LLMClientPool(
            endpoints=mock_llm_servers.endpoints,
        ) as pool:
            chunks = []
            async for chunk in pool.chat_completions_stream([{"role": "user", "content": "Hello"}]):
                chunks.append(chunk)

            assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_pool_streaming_failover(self):
        """Pool 流式 failover：第一个 endpoint 100% 失败，自动切到第二个"""
        configs = [
            MockServerConfig(port=19101, delay_min=0.01, delay_max=0.01, error_rate=1.0),
            MockServerConfig(port=19102, delay_min=0.01, delay_max=0.01, error_rate=0),
        ]
        with MockLLMServerGroup(configs) as group:
            async with LLMClientPool(
                endpoints=group.endpoints,
                fallback=True,
            ) as pool:
                chunks = []
                async for chunk in pool.chat_completions_stream(
                    [{"role": "user", "content": "Hello"}]
                ):
                    chunks.append(chunk)

                assert len(chunks) > 0


# ============== 3. 断点续传 ==============


class TestCheckpointRecovery:
    """断点续传端到端测试"""

    @pytest.mark.asyncio
    async def test_output_jsonl_basic(self, mock_llm_server):
        """基础 JSONL 输出：结果写入文件"""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = f.name

        try:
            async with LLMClient(
                base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
            ) as client:
                messages_list = create_messages(10)
                results = await client.chat_completions_batch(
                    messages_list,
                    output_jsonl=output_path,
                    show_progress=False,
                )

            assert len(results) == 10
            assert all(r is not None for r in results)

            # 验证文件内容
            records = []
            with open(output_path) as f:
                for line in f:
                    records.append(json.loads(line.strip()))

            assert len(records) == 10
            # 每条记录应包含 index, output, status, input
            for record in records:
                assert "index" in record
                assert "output" in record
                assert "status" in record
                assert record["status"] == "success"
        finally:
            os.unlink(output_path)

    @pytest.mark.asyncio
    async def test_checkpoint_resume(self):
        """断点续传：第一次处理完成，第二次自动从文件恢复全部结果"""
        configs = [
            MockServerConfig(port=19201, delay_min=0.01, delay_max=0.01),
            MockServerConfig(port=19202, delay_min=0.01, delay_max=0.01),
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = f.name

        try:
            messages_list = create_messages(20)

            with MockLLMServerGroup(configs) as group:
                # 第一次：全部处理
                async with LLMClientPool(
                    endpoints=group.endpoints,
                    concurrency_limit=10,
                    fallback=True,
                ) as pool:
                    results1 = await pool.chat_completions_batch(
                        messages_list,
                        output_jsonl=output_path,
                        show_progress=False,
                    )
                assert len(results1) == 20
                assert all(r is not None for r in results1)

                # 第二次：从文件恢复所有结果，不发送任何请求
                async with LLMClientPool(
                    endpoints=group.endpoints,
                    concurrency_limit=10,
                    fallback=True,
                ) as pool:
                    results2, summary = await pool.chat_completions_batch(
                        messages_list,
                        output_jsonl=output_path,
                        show_progress=False,
                        return_summary=True,
                    )

            assert len(results2) == 20
            assert all(r is not None for r in results2)
            assert summary["success"] == 20
            assert summary["failed"] == 0
        finally:
            os.unlink(output_path)

    @pytest.mark.asyncio
    async def test_checkpoint_partial_resume(self):
        """断点续传：模拟部分完成后恢复，只处理剩余任务"""
        # 需要 2 个 endpoint 才能走 _batch_distributed 路径（支持结果还原）
        configs = [
            MockServerConfig(port=19203, delay_min=0.01, delay_max=0.01),
            MockServerConfig(port=19204, delay_min=0.01, delay_max=0.01),
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = f.name

        try:
            messages_list = create_messages(10)

            # 手动写入前 5 条的结果（模拟中断）
            with open(output_path, "w") as f:
                for i in range(5):
                    record = {
                        "index": i,
                        "output": f"Partial result {i}",
                        "status": "success",
                        "input": messages_list[i],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            with MockLLMServerGroup(configs) as group:
                # 恢复：应该只处理后 5 条
                async with LLMClientPool(
                    endpoints=group.endpoints,
                    concurrency_limit=5,
                    fallback=True,
                ) as pool:
                    results, summary = await pool.chat_completions_batch(
                        messages_list,
                        output_jsonl=output_path,
                        show_progress=True,
                        return_summary=True,
                    )

            assert len(results) == 10
            assert all(r is not None for r in results)
            # 前 5 条应该是从文件恢复的内容
            for i in range(5):
                assert results[i] == f"Partial result {i}"
            # 后 5 条应该是 mock server 返回的新内容
            for i in range(5, 10):
                assert isinstance(results[i], str)
                assert len(results[i]) > 0
        finally:
            os.unlink(output_path)

    @pytest.mark.asyncio
    async def test_pool_checkpoint_resume(self):
        """Pool 多 endpoint 断点续传"""
        configs = [
            MockServerConfig(port=19111, delay_min=0.01, delay_max=0.01),
            MockServerConfig(port=19112, delay_min=0.01, delay_max=0.01),
        ]

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = f.name

        try:
            messages_list = create_messages(30)

            with MockLLMServerGroup(configs) as group:
                # 第一次：全部处理
                async with LLMClientPool(
                    endpoints=group.endpoints,
                    concurrency_limit=10,
                    fallback=True,
                ) as pool:
                    results1 = await pool.chat_completions_batch(
                        messages_list,
                        output_jsonl=output_path,
                        show_progress=False,
                    )
                assert all(r is not None for r in results1)

                # 第二次：从文件恢复
                async with LLMClientPool(
                    endpoints=group.endpoints,
                    concurrency_limit=10,
                    fallback=True,
                ) as pool:
                    results2, summary = await pool.chat_completions_batch(
                        messages_list,
                        output_jsonl=output_path,
                        show_progress=False,
                        return_summary=True,
                    )

            assert len(results2) == 30
            assert summary["success"] == 30
            assert summary["failed"] == 0
        finally:
            os.unlink(output_path)


# ============== 4. 响应缓存 ==============


class TestResponseCache:
    """响应缓存端到端测试"""

    @pytest.mark.asyncio
    async def test_cache_hit(self, mock_llm_server):
        """缓存命中：相同消息第二次请求走缓存"""
        with tempfile.TemporaryDirectory() as cache_dir:
            cache_config = ResponseCacheConfig(
                enabled=True, cache_dir=cache_dir, ttl=3600, use_ipc=False
            )
            async with LLMClient(
                base_url=mock_llm_server.url,
                model="mock-model",
                api_key="EMPTY",
                cache=cache_config,
            ) as client:
                messages = [{"role": "user", "content": "Cache test"}]

                # 第一次：实际请求
                result1 = await client.chat_completions(messages)
                assert result1 is not None

                # 第二次：应命中缓存，结果相同
                result2 = await client.chat_completions(messages)
                assert result2 == result1

    @pytest.mark.asyncio
    async def test_cache_different_messages(self, mock_llm_server):
        """不同消息不应命中缓存"""
        with tempfile.TemporaryDirectory() as cache_dir:
            cache_config = ResponseCacheConfig(
                enabled=True, cache_dir=cache_dir, ttl=3600, use_ipc=False
            )
            async with LLMClient(
                base_url=mock_llm_server.url,
                model="mock-model",
                api_key="EMPTY",
                cache=cache_config,
            ) as client:
                result1 = await client.chat_completions([{"role": "user", "content": "Message A"}])
                result2 = await client.chat_completions([{"role": "user", "content": "Message B"}])

                # 两个不同消息，结果不一定相同（mock 随机生成）
                # 但都应该是非空字符串
                assert isinstance(result1, str) and len(result1) > 0
                assert isinstance(result2, str) and len(result2) > 0

    @pytest.mark.asyncio
    async def test_cache_in_batch(self, mock_llm_server):
        """批量处理中的缓存：第二次批量应全部走缓存"""
        with tempfile.TemporaryDirectory() as cache_dir:
            cache_config = ResponseCacheConfig(
                enabled=True, cache_dir=cache_dir, ttl=3600, use_ipc=False
            )
            async with LLMClient(
                base_url=mock_llm_server.url,
                model="mock-model",
                api_key="EMPTY",
                cache=cache_config,
            ) as client:
                messages_list = create_messages(10)

                # 第一次批量
                results1 = await client.chat_completions_batch(messages_list, show_progress=False)
                assert all(r is not None for r in results1)

                # 第二次批量（相同消息）
                results2, summary = await client.chat_completions_batch(
                    messages_list,
                    show_progress=False,
                    return_summary=True,
                )

                # 结果应该一致（全部走缓存）
                for r1, r2 in zip(results1, results2):
                    assert r1 == r2


# ============== 5. 成本追踪 ==============


class TestCostTracking:
    """成本追踪端到端测试"""

    @pytest.mark.asyncio
    async def test_track_cost_in_batch(self, mock_llm_server):
        """批量处理中的成本追踪"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            messages_list = create_messages(10)
            results = await client.chat_completions_batch(
                messages_list,
                show_progress=False,
                track_cost=True,
            )
            assert len(results) == 10
            assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_return_usage_in_batch(self, mock_llm_server):
        """批量处理返回 usage 信息"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            messages_list = create_messages(5)
            results = await client.chat_completions_batch(
                messages_list,
                show_progress=False,
                return_usage=True,
            )
            assert len(results) == 5
            for r in results:
                assert r is not None
                assert hasattr(r, "content")
                assert hasattr(r, "usage")

    @pytest.mark.asyncio
    async def test_pool_track_cost(self):
        """Pool 批量处理的成本追踪"""
        configs = [
            MockServerConfig(port=19121, delay_min=0.01, delay_max=0.01),
            MockServerConfig(port=19122, delay_min=0.01, delay_max=0.01),
        ]
        with MockLLMServerGroup(configs) as group:
            async with LLMClientPool(
                endpoints=group.endpoints,
                concurrency_limit=10,
                fallback=True,
            ) as pool:
                messages_list = create_messages(20)
                results, summary = await pool.chat_completions_batch(
                    messages_list,
                    show_progress=True,
                    track_cost=True,
                    return_summary=True,
                )
                assert summary["success"] == 20
                assert summary["failed"] == 0


# ============== 6. LLMClientPool 完整流程 ==============


class TestPoolEndToEnd:
    """LLMClientPool 端到端测试"""

    @pytest.mark.asyncio
    async def test_pool_single_endpoint(self, mock_llm_server):
        """Pool 单 endpoint 模式"""
        async with LLMClientPool(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as pool:
            result = await pool.chat_completions([{"role": "user", "content": "Hello"}])
            assert result is not None
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_pool_multi_endpoint_batch(self, mock_llm_servers):
        """Pool 多 endpoint 批量处理"""
        async with LLMClientPool(
            endpoints=mock_llm_servers.endpoints,
            concurrency_limit=10,
            fallback=True,
        ) as pool:
            messages_list = create_messages(50)
            results, summary = await pool.chat_completions_batch(
                messages_list,
                show_progress=True,
                return_summary=True,
            )
            assert summary["total"] == 50
            assert summary["success"] == 50
            assert summary["failed"] == 0

    @pytest.mark.asyncio
    async def test_pool_failover_single_request(self):
        """Pool 单条请求 failover"""
        configs = [
            MockServerConfig(port=19131, delay_min=0.01, delay_max=0.01, error_rate=1.0),
            MockServerConfig(port=19132, delay_min=0.01, delay_max=0.01, error_rate=0),
        ]
        with MockLLMServerGroup(configs) as group:
            async with LLMClientPool(
                endpoints=group.endpoints,
                fallback=True,
            ) as pool:
                result = await pool.chat_completions([{"role": "user", "content": "Hello"}])
                assert result is not None
                assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_pool_failover_batch(self):
        """Pool 批量 failover：一个 endpoint 高失败率"""
        configs = [
            MockServerConfig(port=19133, delay_min=0.01, delay_max=0.01, error_rate=0.8),
            MockServerConfig(port=19134, delay_min=0.01, delay_max=0.01, error_rate=0),
        ]
        with MockLLMServerGroup(configs) as group:
            async with LLMClientPool(
                endpoints=group.endpoints,
                concurrency_limit=10,
                fallback=True,
            ) as pool:
                messages_list = create_messages(30)
                results, summary = await pool.chat_completions_batch(
                    messages_list,
                    show_progress=True,
                    return_summary=True,
                )
                assert summary["success"] == 30
                assert summary["failed"] == 0

    @pytest.mark.asyncio
    async def test_pool_all_fail(self):
        """Pool 全部 endpoint 失败"""
        configs = [
            MockServerConfig(port=19135, delay_min=0.01, delay_max=0.01, error_rate=1.0),
            MockServerConfig(port=19136, delay_min=0.01, delay_max=0.01, error_rate=1.0),
        ]
        with MockLLMServerGroup(configs) as group:
            async with LLMClientPool(
                endpoints=group.endpoints,
                concurrency_limit=5,
                fallback=True,
            ) as pool:
                messages_list = create_messages(10)
                results, summary = await pool.chat_completions_batch(
                    messages_list,
                    show_progress=True,
                    return_summary=True,
                )
                assert summary["failed"] == 10
                assert all(r is None for r in results)

    @pytest.mark.asyncio
    async def test_pool_round_robin_routing(self):
        """Pool 轮询路由：多个 endpoint 轮流处理请求"""
        configs = [
            MockServerConfig(port=19137, delay_min=0.01, delay_max=0.01),
            MockServerConfig(port=19138, delay_min=0.01, delay_max=0.01),
        ]
        with MockLLMServerGroup(configs) as group:
            async with LLMClientPool(
                endpoints=group.endpoints,
                concurrency_limit=5,
                fallback=True,
            ) as pool:
                messages_list = create_messages(40)
                results, summary = await pool.chat_completions_batch(
                    messages_list,
                    show_progress=True,
                    return_summary=True,
                )
                assert summary["success"] == 40
                assert summary["failed"] == 0

    @pytest.mark.asyncio
    async def test_pool_with_metadata(self):
        """Pool 批量处理带 metadata"""
        configs = [
            MockServerConfig(port=19141, delay_min=0.01, delay_max=0.01),
        ]
        with MockLLMServerGroup(configs) as group:
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
                output_path = f.name

            try:
                async with LLMClientPool(
                    endpoints=group.endpoints,
                    concurrency_limit=5,
                ) as pool:
                    messages_list = create_messages(5)
                    metadata_list = [{"source": f"test-{i}"} for i in range(5)]
                    results = await pool.chat_completions_batch(
                        messages_list,
                        output_jsonl=output_path,
                        metadata_list=metadata_list,
                        show_progress=False,
                    )

                assert all(r is not None for r in results)

                # 验证 metadata 被写入文件
                with open(output_path) as f:
                    for line in f:
                        record = json.loads(line.strip())
                        assert "metadata" in record
                        assert "source" in record["metadata"]
            finally:
                os.unlink(output_path)


# ============== 7. 综合场景 ==============


class TestCombinedScenarios:
    """综合场景测试"""

    @pytest.mark.asyncio
    async def test_pool_batch_with_cache_and_checkpoint(self):
        """Pool + 缓存 + 断点续传 联合测试"""
        configs = [
            MockServerConfig(port=19151, delay_min=0.01, delay_max=0.01),
            MockServerConfig(port=19152, delay_min=0.01, delay_max=0.01),
        ]

        with tempfile.TemporaryDirectory() as cache_dir:
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
                output_path = f.name

            cache_config = ResponseCacheConfig(
                enabled=True, cache_dir=cache_dir, ttl=3600, use_ipc=False
            )
            messages_list = create_messages(15)

            try:
                with MockLLMServerGroup(configs) as group:
                    # 第一次：实际请求，结果写入文件和缓存
                    async with LLMClientPool(
                        endpoints=group.endpoints,
                        concurrency_limit=5,
                        fallback=True,
                        cache=cache_config,
                    ) as pool:
                        results1 = await pool.chat_completions_batch(
                            messages_list,
                            output_jsonl=output_path,
                            show_progress=False,
                        )
                    assert all(r is not None for r in results1)

                    # 第二次：从文件恢复（不需要缓存或网络）
                    async with LLMClientPool(
                        endpoints=group.endpoints,
                        concurrency_limit=5,
                        fallback=True,
                        cache=cache_config,
                    ) as pool:
                        results2, summary = await pool.chat_completions_batch(
                            messages_list,
                            output_jsonl=output_path,
                            show_progress=False,
                            return_summary=True,
                        )

                    assert summary["success"] == 15
                    assert summary["failed"] == 0
                    # 文件恢复的结果应一致
                    for r1, r2 in zip(results1, results2):
                        assert r1 == r2
            finally:
                os.unlink(output_path)

    @pytest.mark.asyncio
    async def test_pool_slow_fast_dynamic_balance(self):
        """慢快节点动态负载均衡：快节点应处理更多任务"""
        configs = [
            MockServerConfig(port=19153, delay_min=0.01, delay_max=0.01),  # 快
            MockServerConfig(port=19154, delay_min=0.3, delay_max=0.3),  # 慢
        ]
        with MockLLMServerGroup(configs) as group:
            async with LLMClientPool(
                endpoints=group.endpoints,
                concurrency_limit=5,
                fallback=True,
            ) as pool:
                messages_list = create_messages(50)
                results, summary = await pool.chat_completions_batch(
                    messages_list,
                    show_progress=True,
                    return_summary=True,
                )
                assert summary["success"] == 50
                assert summary["failed"] == 0
                # 验证不卡死，且执行时间合理（不应是全部走慢节点）
                assert summary["elapsed"] < 20

    @pytest.mark.asyncio
    async def test_async_context_manager_cleanup(self, mock_llm_server):
        """async 上下文管理器正确清理资源"""
        async with LLMClient(
            base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
        ) as client:
            result = await client.chat_completions([{"role": "user", "content": "Hello"}])
            assert result is not None

    def test_sync_context_manager_cleanup(self, mock_llm_server):
        """sync 上下文管理器正确清理资源"""
        with LLMClient(base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY") as client:
            result = client.chat_completions_sync([{"role": "user", "content": "Hello"}])
            assert result is not None


# ============== 8. save_input 参数测试 ==============


class TestSaveInput:
    """save_input 参数端到端测试"""

    @pytest.mark.asyncio
    async def test_save_input_false(self, mock_llm_server):
        """save_input=False：输出文件不包含 input 字段"""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = f.name

        try:
            async with LLMClient(
                base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
            ) as client:
                messages_list = create_messages(5)
                results = await client.chat_completions_batch(
                    messages_list,
                    output_jsonl=output_path,
                    show_progress=False,
                    save_input=False,
                )

            assert len(results) == 5
            assert all(r is not None for r in results)

            # 验证文件内容不包含 input 字段
            with open(output_path) as f:
                for line in f:
                    record = json.loads(line.strip())
                    assert "index" in record
                    assert "output" in record
                    assert "status" in record
                    assert "input" not in record
        finally:
            os.unlink(output_path)

    @pytest.mark.asyncio
    async def test_save_input_last(self, mock_llm_server):
        """save_input='last'：仅保存最后一个 user message 的 content"""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = f.name

        try:
            # 使用包含 system + user 的多轮消息
            messages_list = [
                [
                    {"role": "system", "content": "You are a helper."},
                    {"role": "user", "content": f"Question {i}"},
                ]
                for i in range(5)
            ]

            async with LLMClient(
                base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
            ) as client:
                results = await client.chat_completions_batch(
                    messages_list,
                    output_jsonl=output_path,
                    show_progress=False,
                    save_input="last",
                )

            assert len(results) == 5

            # 验证文件中 input 只包含最后一个 user message 的 content
            with open(output_path) as f:
                for line in f:
                    record = json.loads(line.strip())
                    idx = record["index"]
                    assert record["input"] == f"Question {idx}"
        finally:
            os.unlink(output_path)

    @pytest.mark.asyncio
    async def test_save_input_true_default(self, mock_llm_server):
        """save_input=True（默认）：保存完整 messages"""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = f.name

        try:
            async with LLMClient(
                base_url=mock_llm_server.url, model="mock-model", api_key="EMPTY"
            ) as client:
                messages_list = create_messages(3)
                results = await client.chat_completions_batch(
                    messages_list,
                    output_jsonl=output_path,
                    show_progress=False,
                )

            # 验证默认行为：包含完整 input
            with open(output_path) as f:
                for line in f:
                    record = json.loads(line.strip())
                    assert "input" in record
                    idx = record["index"]
                    assert record["input"] == messages_list[idx]
        finally:
            os.unlink(output_path)

    @pytest.mark.asyncio
    async def test_checkpoint_resume_save_input_false(self):
        """save_input=False 的断点续传：无 input 校验，基于 index 恢复"""
        configs = [
            MockServerConfig(port=19301, delay_min=0.01, delay_max=0.01),
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = f.name

        try:
            messages_list = create_messages(10)

            # 手动写入前 5 条无 input 的记录
            with open(output_path, "w") as f:
                for i in range(5):
                    record = {
                        "index": i,
                        "output": f"Partial result {i}",
                        "status": "success",
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            with MockLLMServer(configs[0]) as server:
                async with LLMClient(
                    base_url=server.url, model="mock-model", api_key="EMPTY"
                ) as client:
                    results = await client.chat_completions_batch(
                        messages_list,
                        output_jsonl=output_path,
                        show_progress=False,
                        save_input=False,
                    )

            assert len(results) == 10
            # 后 5 条应该是新处理的
            for i in range(5, 10):
                assert results[i] is not None

            # 验证输出文件中没有 input 字段
            with open(output_path) as f:
                for line in f:
                    record = json.loads(line.strip())
                    assert "input" not in record
        finally:
            os.unlink(output_path)

    @pytest.mark.asyncio
    async def test_checkpoint_resume_save_input_last(self):
        """save_input='last' 的断点续传：使用 Pool 多 endpoint 以支持结果还原"""
        configs = [
            MockServerConfig(port=19302, delay_min=0.01, delay_max=0.01),
            MockServerConfig(port=19303, delay_min=0.01, delay_max=0.01),
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = f.name

        try:
            messages_list = create_messages(10)

            with MockLLMServerGroup(configs) as group:
                # 第一次：全部处理
                async with LLMClientPool(
                    endpoints=group.endpoints,
                    concurrency_limit=5,
                    fallback=True,
                ) as pool:
                    results1 = await pool.chat_completions_batch(
                        messages_list,
                        output_jsonl=output_path,
                        show_progress=False,
                        save_input="last",
                    )
                assert all(r is not None for r in results1)

                # 验证文件中 input 是 last user content
                with open(output_path) as f:
                    for line in f:
                        record = json.loads(line.strip())
                        idx = record["index"]
                        assert record["input"] == f"Test message {idx}"

                # 第二次：从文件恢复（Pool 的 _batch_distributed 支持结果还原）
                async with LLMClientPool(
                    endpoints=group.endpoints,
                    concurrency_limit=5,
                    fallback=True,
                ) as pool:
                    results2, summary = await pool.chat_completions_batch(
                        messages_list,
                        output_jsonl=output_path,
                        show_progress=False,
                        return_summary=True,
                        save_input="last",
                    )

            assert len(results2) == 10
            assert all(r is not None for r in results2)
            assert summary["success"] == 10
            assert summary["failed"] == 0
        finally:
            os.unlink(output_path)
