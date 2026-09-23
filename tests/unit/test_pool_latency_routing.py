"""pool 单条调用的延迟感知选路：样本采集与择优

策略：patch 掉 ConcurrentRequester.make_requests（真实 HTTP 调用点），
让不同 endpoint 有不同的响应耗时，验证 pool 把服务耗时喂给了 router，
并且后续的逐条调用（并发度为 1）确实偏向快 endpoint。
"""

import asyncio
import warnings
from unittest.mock import MagicMock

import pytest

from flexllm import LLMClientPool, ResponseCacheConfig
from flexllm.async_api.interface import RequestResult
from flexllm.clients.base import ChatCompletionResult

ENDPOINTS = [
    # 慢的那个故意排在第一位：纯容量感知下它会吃掉全部流量
    {"base_url": "http://slow.test/v1", "api_key": "k1", "model": "m"},
    {"base_url": "http://fast.test/v1", "api_key": "k2", "model": "m"},
]


def _openai_response():
    data = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    response = MagicMock()
    response.status = 200
    return response, data


class SpeedByHost:
    """替身 make_requests：按 URL 决定响应耗时，并记录各 endpoint 的命中次数"""

    def __init__(self, delays: dict[str, float]):
        self.delays = delays
        self.hits: dict[str, int] = {host: 0 for host in delays}

    async def __call__(self, session, method, url, **kwargs):
        host = next(h for h in self.delays if h in url)
        self.hits[host] += 1
        await asyncio.sleep(self.delays[host])
        return _openai_response()


def _patch(pool: LLMClientPool, recorder: SpeedByHost) -> None:
    for client in pool._clients:
        client._client.make_requests = recorder


@pytest.fixture(autouse=True)
def _silence_legacy_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


class TestLatencySampling:
    async def test_single_call_feeds_service_time_to_router(self):
        pool = LLMClientPool(endpoints=ENDPOINTS)
        _patch(pool, SpeedByHost({"slow.test": 0.05, "fast.test": 0.05}))

        assert all(p["ewma_latency"] is None for p in pool.stats["router_stats"]["providers"])

        await pool.chat_completions("q")

        sampled = [p for p in pool.stats["router_stats"]["providers"] if p["ewma_latency"]]
        assert len(sampled) == 1
        assert sampled[0]["ewma_latency"] >= 0.04

    async def test_failed_call_does_not_feed_sample(self):
        """失败往往返回得更快，计入会让坏 endpoint 显得更优"""

        async def boom(session, method, url, **kwargs):
            raise RuntimeError("endpoint down")

        pool = LLMClientPool(endpoints=ENDPOINTS, fallback=False)
        for client in pool._clients:
            client._client.make_requests = boom

        with pytest.raises(Exception):
            await pool.chat_completions("q", raise_on_error=True)

        assert all(p["ewma_latency"] is None for p in pool.stats["router_stats"]["providers"])


class TestSerialCallsPreferFastEndpoint:
    async def test_serial_calls_drift_to_fast_endpoint(self):
        """并发度为 1 的逐条调用也能避开慢 endpoint（in-flight 在这里恒为 0）"""
        recorder = SpeedByHost({"slow.test": 0.12, "fast.test": 0.01})
        pool = LLMClientPool(endpoints=ENDPOINTS, latency_decay_calls=10.0)
        _patch(pool, recorder)

        for _ in range(12):
            await pool.chat_completions("q")

        assert recorder.hits["fast.test"] > recorder.hits["slow.test"]
        # 慢的那个只承担探测流量
        assert recorder.hits["slow.test"] <= 3

    async def test_equal_speed_endpoints_share_traffic(self):
        """同速 endpoint 下流量自然分散，而不是像纯容量感知那样全压在第一个

        并发度为 1 时两者 in-flight 恒为 0，旧策略稳定选列表首个；有了延迟样本后，
        测量噪声足以让选择在两者间摆动——同速时摆到哪个都不吃亏。
        """
        recorder = SpeedByHost({"slow.test": 0.01, "fast.test": 0.01})
        pool = LLMClientPool(endpoints=ENDPOINTS)
        _patch(pool, recorder)

        for _ in range(8):
            await pool.chat_completions("q")

        assert sum(recorder.hits.values()) == 8
        assert min(recorder.hits.values()) >= 1


class TestTauPlumbing:
    def test_latency_decay_calls_reaches_router(self):
        assert (
            LLMClientPool(endpoints=ENDPOINTS, latency_decay_calls=50.0)._router.latency_decay_calls
            == 50.0
        )

    def test_latency_decay_calls_defaults_to_10(self):
        assert LLMClientPool(endpoints=ENDPOINTS)._router.latency_decay_calls == 10.0


class TestCacheHitsAreNotLatencySamples:
    """缓存命中是微秒级本地读取，且缓存跨 endpoint 共享，不能当作 endpoint 的服务耗时"""

    def _cached_pool(self, tmp_path, **kwargs) -> LLMClientPool:
        return LLMClientPool(
            endpoints=ENDPOINTS,
            cache=ResponseCacheConfig(enabled=True, cache_dir=str(tmp_path), ttl=600),
            **kwargs,
        )

    async def test_cache_hit_does_not_pollute_latency(self, tmp_path):
        recorder = SpeedByHost({"slow.test": 0.05, "fast.test": 0.05})
        pool = self._cached_pool(tmp_path)
        _patch(pool, recorder)

        await pool.chat_completions("same question")
        sampled = [p["ewma_latency"] for p in pool.stats["router_stats"]["providers"]]
        before = [x for x in sampled if x is not None]
        assert len(before) == 1 and before[0] >= 0.04

        # 同一个问题再问 9 次，全部命中缓存（真实请求数不再增长）
        calls_before = sum(recorder.hits.values())
        for _ in range(9):
            await pool.chat_completions("same question")
        assert sum(recorder.hits.values()) == calls_before, "应当全部命中缓存"

        after = [
            p["ewma_latency"]
            for p in pool.stats["router_stats"]["providers"]
            if p["ewma_latency"] is not None
        ]
        assert after == before, "缓存命中不应改变任何 endpoint 的延迟估计"

    async def test_cache_hit_does_not_make_endpoint_look_fastest(self, tmp_path):
        """回归：命中过缓存的慢 endpoint 不应因此吸走后续流量"""
        recorder = SpeedByHost({"slow.test": 0.12, "fast.test": 0.01})
        pool = self._cached_pool(tmp_path)
        _patch(pool, recorder)

        await pool.chat_completions("q0")  # 落到列表首个的 slow，并写入缓存
        assert recorder.hits["slow.test"] == 1
        for _ in range(3):
            await pool.chat_completions("q0")  # 命中缓存，耗时微秒级

        for i in range(10):
            await pool.chat_completions(f"fresh-{i}")  # 全是新问题，必然真实请求

        assert recorder.hits["fast.test"] > recorder.hits["slow.test"]
        assert recorder.hits["slow.test"] <= 3


class TestSingleCallReturnShapesUnchanged:
    """单条路径内部强制 return_usage=True 后，对外返回形状必须与之前一致"""

    async def test_default_returns_str(self):
        pool = LLMClientPool(endpoints=ENDPOINTS)
        _patch(pool, SpeedByHost({"slow.test": 0.001, "fast.test": 0.001}))
        assert await pool.chat_completions("q") == "ok"

    async def test_return_usage_gives_result(self):
        pool = LLMClientPool(endpoints=ENDPOINTS)
        _patch(pool, SpeedByHost({"slow.test": 0.001, "fast.test": 0.001}))
        result = await pool.chat_completions("q", return_usage=True)
        assert isinstance(result, ChatCompletionResult)
        assert result.content == "ok"
        assert result.usage["total_tokens"] == 2
        assert result.queue_time is not None

    async def test_return_raw_gives_request_result(self):
        pool = LLMClientPool(endpoints=ENDPOINTS)
        _patch(pool, SpeedByHost({"slow.test": 0.001, "fast.test": 0.001}))
        result = await pool.chat_completions("q", return_raw=True)
        assert isinstance(result, RequestResult)
        assert result.status == "success"

    async def test_cached_str_shape_is_str(self, tmp_path):
        """缓存命中在默认参数下仍然返回纯字符串，不泄漏 ChatCompletionResult"""
        pool = LLMClientPool(
            endpoints=ENDPOINTS,
            cache=ResponseCacheConfig(enabled=True, cache_dir=str(tmp_path), ttl=600),
        )
        _patch(pool, SpeedByHost({"slow.test": 0.001, "fast.test": 0.001}))
        assert await pool.chat_completions("q") == "ok"
        assert await pool.chat_completions("q") == "ok"


class TestLatencyExcludesLocalWork:
    """延迟样本只能是 endpoint 的耗时，不含调用方这边的开销"""

    def _pool_with_preprocess_delay(self, delay: float) -> tuple[LLMClientPool, SpeedByHost]:
        recorder = SpeedByHost({"slow.test": 0.02, "fast.test": 0.02})
        pool = LLMClientPool(endpoints=ENDPOINTS)
        for client in pool._clients:
            client._client.make_requests = recorder
            original = client._preprocess_messages

            async def delayed(messages, flag, _orig=original, _d=delay):
                await asyncio.sleep(_d)  # 模拟图片下载转 base64
                return await _orig(messages, flag)

            client._preprocess_messages = delayed
        return pool, recorder

    def _sampled(self, pool: LLMClientPool) -> float:
        values = [
            p["ewma_latency"]
            for p in pool.stats["router_stats"]["providers"]
            if p["ewma_latency"] is not None
        ]
        assert len(values) == 1
        return values[0]

    async def test_message_preprocessing_is_not_counted(self):
        """预处理发生在请求之前，拿秒表包住整个调用会把它算成 endpoint 变慢"""
        fast_pool, _ = self._pool_with_preprocess_delay(0.0)
        await fast_pool.chat_completions("q", preprocess_msg=True)
        baseline = self._sampled(fast_pool)

        slow_pool, _ = self._pool_with_preprocess_delay(0.3)
        await slow_pool.chat_completions("q", preprocess_msg=True)
        with_preprocess = self._sampled(slow_pool)

        assert with_preprocess < baseline + 0.1, (
            f"预处理耗时被计入了服务耗时：{with_preprocess:.3f}s vs 基线 {baseline:.3f}s"
        )

    async def test_latency_is_carried_on_result(self):
        pool = LLMClientPool(endpoints=ENDPOINTS)
        _patch(pool, SpeedByHost({"slow.test": 0.02, "fast.test": 0.02}))
        result = await pool.chat_completions("q", return_usage=True)
        assert result.latency is not None
        assert result.latency >= 0.02
        assert result.queue_time is not None

    async def test_cache_hit_carries_no_latency(self, tmp_path):
        pool = LLMClientPool(
            endpoints=ENDPOINTS,
            cache=ResponseCacheConfig(enabled=True, cache_dir=str(tmp_path), ttl=600),
        )
        _patch(pool, SpeedByHost({"slow.test": 0.02, "fast.test": 0.02}))
        first = await pool.chat_completions("q", return_usage=True)
        assert first.latency is not None

        second = await pool.chat_completions("q", return_usage=True)
        assert second.cached is True
        assert second.latency is None, "缓存命中没有真实请求，latency 必须为 None"
        assert second.queue_time is None


class TestResultPayloadBackCompat:
    def test_payload_without_latency_still_loads(self):
        """旧缓存/checkpoint 记录里没有 latency 字段，读取时按缺省 None 处理"""
        legacy = {"content": "ok", "usage": {"total_tokens": 2}, "queue_time": 0.1}
        result = ChatCompletionResult._from_payload(legacy)
        assert result.content == "ok"
        assert result.latency is None

    def test_latency_is_appended_at_the_end(self):
        """新字段必须追加在末尾，不能挤动既有 dataclass 位置参数"""
        from dataclasses import fields

        names = [f.name for f in fields(ChatCompletionResult)]
        assert names[-1] == "latency"
