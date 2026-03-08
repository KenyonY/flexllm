"""Test ResponseCache batch operations (get_batch / set_batch)"""

import shutil
import tempfile

import pytest

from flexllm.cache.response_cache import ResponseCache, ResponseCacheConfig


class TestCacheBatch:
    @pytest.fixture
    def cache(self):
        tmp_dir = tempfile.mkdtemp()
        config = ResponseCacheConfig.with_ttl(ttl=3600, cache_dir=tmp_dir)
        cache = ResponseCache(config)
        yield cache
        cache.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    @pytest.fixture
    def disabled_cache(self):
        return ResponseCache(ResponseCacheConfig.disabled())

    def _msg(self, text: str) -> list[dict]:
        return [{"role": "user", "content": text}]

    def test_get_batch_all_miss(self, cache):
        """全部未命中"""
        msgs = [self._msg(f"q{i}") for i in range(5)]
        cached, uncached = cache.get_batch(msgs, model="test-model")
        assert all(r is None for r in cached)
        assert uncached == list(range(5))

    def test_set_batch_then_get_batch(self, cache):
        """set_batch 写入后 get_batch 全部命中"""
        msgs = [self._msg(f"q{i}") for i in range(10)]
        responses = [{"content": f"a{i}", "usage": {"total_tokens": i}} for i in range(10)]

        cache.set_batch(msgs, responses, model="m1")
        cached, uncached = cache.get_batch(msgs, model="m1")

        assert uncached == []
        for i, r in enumerate(cached):
            assert r["content"] == f"a{i}"

    def test_get_batch_partial_hit(self, cache):
        """部分命中"""
        msgs = [self._msg(f"q{i}") for i in range(5)]
        # 只写入偶数索引
        even_msgs = [msgs[i] for i in [0, 2, 4]]
        even_responses = [{"content": f"a{i}"} for i in [0, 2, 4]]
        cache.set_batch(even_msgs, even_responses, model="m1")

        cached, uncached = cache.get_batch(msgs, model="m1")
        assert uncached == [1, 3]
        assert cached[0]["content"] == "a0"
        assert cached[1] is None
        assert cached[2]["content"] == "a2"
        assert cached[3] is None
        assert cached[4]["content"] == "a4"

    def test_set_batch_skips_none_responses(self, cache):
        """set_batch 跳过 None 响应"""
        msgs = [self._msg("q0"), self._msg("q1")]
        responses = [{"content": "a0"}, None]
        cache.set_batch(msgs, responses, model="m1")

        cached, uncached = cache.get_batch(msgs, model="m1")
        assert cached[0]["content"] == "a0"
        assert cached[1] is None
        assert uncached == [1]

    def test_get_batch_stats(self, cache):
        """验证 stats 统计"""
        msgs = [self._msg("q0"), self._msg("q1")]
        cache.set_batch(msgs[:1], [{"content": "a0"}], model="m1")

        cache.get_batch(msgs, model="m1")
        assert cache._stats["hits"] == 1
        assert cache._stats["misses"] == 1

    def test_disabled_cache_batch(self, disabled_cache):
        """禁用缓存时 batch 操作正常返回"""
        msgs = [self._msg("q0")]
        cached, uncached = disabled_cache.get_batch(msgs, model="m1")
        assert cached == [None]
        assert uncached == [0]
        # set_batch 不报错
        disabled_cache.set_batch(msgs, [{"content": "a0"}], model="m1")

    def test_get_batch_empty(self, cache):
        """空列表"""
        cached, uncached = cache.get_batch([], model="m1")
        assert cached == []
        assert uncached == []
