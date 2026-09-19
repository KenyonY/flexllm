"""pool 单条调用的延迟感知选路：样本采集与择优

策略：patch 掉 ConcurrentRequester.make_requests（真实 HTTP 调用点），
让不同 endpoint 有不同的响应耗时，验证 pool 把服务耗时喂给了 router，
并且后续的逐条调用（并发度为 1）确实偏向快 endpoint。
"""

import asyncio
import warnings
from unittest.mock import MagicMock

import pytest

from flexllm import LLMClientPool

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
        pool = LLMClientPool(endpoints=ENDPOINTS, latency_tau=30.0)
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
    def test_latency_tau_reaches_router(self):
        assert LLMClientPool(endpoints=ENDPOINTS, latency_tau=120.0)._router.latency_tau == 120.0

    def test_latency_tau_defaults_to_30(self):
        assert LLMClientPool(endpoints=ENDPOINTS)._router.latency_tau == 30.0
