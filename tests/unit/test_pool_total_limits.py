"""pool 级全局并发/QPS 硬上限（total_concurrency_limit / total_max_qps）

策略：patch 掉 ConcurrentRequester.make_requests（真实 HTTP 调用点），
保留 _send_single_request 里真实的 semaphore / 漏桶闸门，用 in-flight
计数验证并发硬上限确实生效。
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from flexllm import LLMClientPool

ENDPOINTS = [
    {"base_url": "http://ep1.test/v1", "api_key": "k1", "model": "m"},
    {"base_url": "http://ep2.test/v1", "api_key": "k2", "model": "m"},
]


def _openai_response(content="ok"):
    data = {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    response = MagicMock()
    response.status = 200
    return response, data


class InflightRecorder:
    """替身 make_requests：记录并发峰值"""

    def __init__(self, delay: float = 0.02):
        self.delay = delay
        self.inflight = 0
        self.peak = 0
        self.calls = 0

    async def __call__(self, session, method, url, **kwargs):
        self.inflight += 1
        self.calls += 1
        self.peak = max(self.peak, self.inflight)
        try:
            await asyncio.sleep(self.delay)
            return _openai_response()
        finally:
            self.inflight -= 1


def _patch_pool(pool: LLMClientPool, recorder: InflightRecorder):
    clients = pool._clients if pool._clients else [pool._single_client]
    for client in clients:
        client._client.make_requests = recorder


class TestInjectionWiring:
    def test_default_no_pool_limits(self):
        pool = LLMClientPool(endpoints=ENDPOINTS)
        for client in pool._clients:
            assert client._client._pool_semaphore is None
            assert client._client._pool_rate_limiter is None

    def test_multi_mode_shares_same_limiter_instances(self):
        pool = LLMClientPool(endpoints=ENDPOINTS, total_concurrency_limit=5, total_max_qps=10)
        sems = {id(c._client._pool_semaphore) for c in pool._clients}
        buckets = {id(c._client._pool_rate_limiter) for c in pool._clients}
        assert sems == {id(pool._pool_limiter)}
        assert buckets == {id(pool._pool_rate_limiter)}
        assert pool._pool_limiter.limit == 5
        assert pool._pool_rate_limiter.max_qps == 10

    def test_single_mode_injection(self):
        pool = LLMClientPool(base_url="http://ep1.test/v1", model="m", total_concurrency_limit=3)
        assert pool._single_client._client._pool_semaphore is pool._pool_limiter

    def test_invalid_values_raise(self):
        with pytest.raises(ValueError, match="total_concurrency_limit"):
            LLMClientPool(endpoints=ENDPOINTS, total_concurrency_limit=0)
        with pytest.raises(ValueError, match="total_max_qps"):
            LLMClientPool(endpoints=ENDPOINTS, total_max_qps=-1)


class TestTotalConcurrencyCap:
    async def test_batch_peak_inflight_capped(self):
        pool = LLMClientPool(
            endpoints=ENDPOINTS,
            concurrency_limit=10,  # 每 endpoint 10，叠加为 20
            total_concurrency_limit=5,
        )
        recorder = InflightRecorder()
        _patch_pool(pool, recorder)

        results = await pool.chat_completions_batch(
            [f"q{i}" for i in range(30)], show_progress=False
        )
        assert len(results) == 30
        assert all(r == "ok" for r in results)
        assert recorder.peak <= 5

    async def test_batch_without_total_cap_exceeds(self):
        """对照组：不设全局上限时并发确实会叠加，证明上面的测试测到了东西"""
        pool = LLMClientPool(endpoints=ENDPOINTS, concurrency_limit=10)
        recorder = InflightRecorder()
        _patch_pool(pool, recorder)

        await pool.chat_completions_batch([f"q{i}" for i in range(30)], show_progress=False)
        assert recorder.peak > 5

    async def test_gathered_single_calls_capped(self):
        pool = LLMClientPool(
            endpoints=ENDPOINTS,
            concurrency_limit=10,
            total_concurrency_limit=3,
        )
        recorder = InflightRecorder()
        _patch_pool(pool, recorder)

        results = await asyncio.gather(*[pool.chat_completions(f"q{i}") for i in range(15)])
        assert all(r == "ok" for r in results)
        assert recorder.peak <= 3

    async def test_single_endpoint_mode_capped(self):
        pool = LLMClientPool(
            base_url="http://ep1.test/v1",
            model="m",
            concurrency_limit=10,
            total_concurrency_limit=2,
        )
        recorder = InflightRecorder()
        _patch_pool(pool, recorder)

        await asyncio.gather(*[pool.chat_completions(f"q{i}") for i in range(10)])
        assert recorder.peak <= 2

    async def test_iter_batch_capped(self):
        pool = LLMClientPool(
            endpoints=ENDPOINTS,
            concurrency_limit=10,
            total_concurrency_limit=4,
        )
        recorder = InflightRecorder()
        _patch_pool(pool, recorder)

        contents = []
        async for result in pool.iter_chat_completions_batch(
            [[{"role": "user", "content": f"q{i}"}] for i in range(20)], show_progress=False
        ):
            assert result.status == "success"
            contents.append(result.content)
        assert len(contents) == 20
        assert recorder.peak <= 4


class TestTotalQps:
    async def test_total_qps_slows_down_burst(self):
        """漏桶容量为 max_qps，超出部分必须等待补充：15 个请求、QPS=10
        → 前 10 个突发放行，后 5 个至少等 ~0.5s"""
        pool = LLMClientPool(
            endpoints=ENDPOINTS,
            concurrency_limit=20,
            max_qps=1000,
            total_max_qps=10,
        )
        recorder = InflightRecorder(delay=0.001)
        _patch_pool(pool, recorder)

        start = time.perf_counter()
        await pool.chat_completions_batch([f"q{i}" for i in range(15)], show_progress=False)
        elapsed = time.perf_counter() - start
        assert elapsed >= 0.4
        assert recorder.calls == 15


class TestReturnShapesUnchanged:
    """worker 内部强制 return_usage=True 后，对外返回形状必须与之前一致"""

    async def test_batch_returns_str_by_default(self):
        pool = LLMClientPool(endpoints=ENDPOINTS, total_concurrency_limit=5)
        _patch_pool(pool, InflightRecorder())
        results = await pool.chat_completions_batch(["q1", "q2"], show_progress=False)
        assert results == ["ok", "ok"]

    async def test_batch_returns_result_with_usage(self):
        pool = LLMClientPool(endpoints=ENDPOINTS, total_concurrency_limit=5)
        _patch_pool(pool, InflightRecorder())
        results = await pool.chat_completions_batch(
            ["q1", "q2"], return_usage=True, show_progress=False
        )
        for r in results:
            assert r.content == "ok"
            assert r.usage["total_tokens"] == 2

    async def test_iter_batch_data_is_str_by_default(self):
        pool = LLMClientPool(endpoints=ENDPOINTS, total_concurrency_limit=5)
        _patch_pool(pool, InflightRecorder())
        async for result in pool.iter_chat_completions_batch(
            [[{"role": "user", "content": "q"}]], show_progress=False
        ):
            assert result.content == "ok"
            assert result.data == "ok"
            assert result.usage is None


class TestQueueTimeTransport:
    async def test_single_call_queue_time_present(self):
        pool = LLMClientPool(endpoints=ENDPOINTS, total_concurrency_limit=5)
        _patch_pool(pool, InflightRecorder())
        result = await pool.chat_completions("q", return_usage=True)
        assert result.queue_time is not None
        assert result.queue_time >= 0

    async def test_contended_queue_time_measures_wait(self):
        """全局并发=1 时，后续请求的 queue_time 应包含等待前序请求的时间"""
        pool = LLMClientPool(endpoints=ENDPOINTS, total_concurrency_limit=1)
        _patch_pool(pool, InflightRecorder(delay=0.05))
        results = await asyncio.gather(
            *[pool.chat_completions("q", return_usage=True) for _ in range(3)]
        )
        max_queue_time = max(r.queue_time for r in results)
        assert max_queue_time >= 0.04

    async def test_iter_batch_yields_queue_time(self):
        pool = LLMClientPool(endpoints=ENDPOINTS, total_concurrency_limit=5)
        _patch_pool(pool, InflightRecorder())
        async for result in pool.iter_chat_completions_batch(
            [[{"role": "user", "content": "q"}]], show_progress=False
        ):
            assert result.queue_time is not None
