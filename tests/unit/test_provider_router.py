"""ProviderRouter 单元测试"""

import time
from collections import Counter

import pytest

from flexllm import ProviderConfig, ProviderRouter, create_router_from_urls


class TestProviderConfig:
    """测试 ProviderConfig 数据类"""

    def test_default_values(self):
        config = ProviderConfig(base_url="http://api.example.com/v1")
        assert config.base_url == "http://api.example.com/v1"
        assert config.api_key == "EMPTY"
        assert config.model is None
        assert config.enabled is True

    def test_custom_values(self):
        config = ProviderConfig(
            base_url="http://api.example.com/v1",
            api_key="sk-xxx",
            model="gpt-4",
            enabled=False,
        )
        assert config.api_key == "sk-xxx"
        assert config.model == "gpt-4"
        assert config.enabled is False


class TestProviderRouterCreation:
    """测试路由器创建"""

    def test_create_with_single_provider(self):
        providers = [ProviderConfig(base_url="http://api1.com/v1")]
        router = ProviderRouter(providers)
        assert router.stats["total"] == 1
        assert router.stats["healthy"] == 1

    def test_create_with_multiple_providers(self):
        providers = [
            ProviderConfig(base_url="http://api1.com/v1"),
            ProviderConfig(base_url="http://api2.com/v1"),
            ProviderConfig(base_url="http://api3.com/v1"),
        ]
        router = ProviderRouter(providers)
        assert router.stats["total"] == 3
        assert router.stats["healthy"] == 3

    def test_create_with_empty_providers_raises(self):
        with pytest.raises(ValueError, match="至少需要一个 provider"):
            ProviderRouter([])

    def test_disabled_providers_are_filtered(self):
        providers = [
            ProviderConfig(base_url="http://api1.com/v1", enabled=True),
            ProviderConfig(base_url="http://api2.com/v1", enabled=False),
        ]
        router = ProviderRouter(providers)
        assert router.stats["total"] == 1

    def test_all_disabled_raises(self):
        providers = [
            ProviderConfig(base_url="http://api1.com/v1", enabled=False),
        ]
        with pytest.raises(ValueError, match="没有可用的 provider"):
            ProviderRouter(providers)


class TestRoundRobinStrategy:
    """测试轮询策略"""

    def test_round_robin_cycles(self):
        providers = [
            ProviderConfig(base_url="http://api1.com/v1"),
            ProviderConfig(base_url="http://api2.com/v1"),
            ProviderConfig(base_url="http://api3.com/v1"),
        ]
        router = ProviderRouter(providers)

        # 获取 6 次，应该循环 2 轮
        urls = [router.get_next().base_url for _ in range(6)]

        # 验证循环
        assert urls[0] == urls[3]
        assert urls[1] == urls[4]
        assert urls[2] == urls[5]

    def test_round_robin_distribution(self):
        providers = [
            ProviderConfig(base_url="http://api1.com/v1"),
            ProviderConfig(base_url="http://api2.com/v1"),
        ]
        router = ProviderRouter(providers)

        urls = [router.get_next().base_url for _ in range(100)]
        counter = Counter(urls)

        # 应该均匀分布
        assert counter["http://api1.com/v1"] == 50
        assert counter["http://api2.com/v1"] == 50


class TestHealthCheck:
    """测试健康检查机制"""

    def test_mark_failed_increments_failures(self):
        providers = [ProviderConfig(base_url="http://api.com/v1")]
        router = ProviderRouter(providers, failure_threshold=3)

        provider = router.get_next()
        router.mark_failed(provider)

        stats = router.stats
        assert stats["providers"][0]["failures"] == 1
        assert stats["providers"][0]["healthy"] is True

    def test_mark_failed_threshold(self):
        providers = [ProviderConfig(base_url="http://api.com/v1")]
        router = ProviderRouter(providers, failure_threshold=3)

        provider = router.get_next()
        for _ in range(3):
            router.mark_failed(provider)

        stats = router.stats
        assert stats["providers"][0]["failures"] == 3
        assert stats["providers"][0]["healthy"] is False

    def test_mark_success_resets_failures(self):
        providers = [ProviderConfig(base_url="http://api.com/v1")]
        router = ProviderRouter(providers, failure_threshold=3)

        provider = router.get_next()
        router.mark_failed(provider)
        router.mark_failed(provider)
        router.mark_success(provider)

        stats = router.stats
        assert stats["providers"][0]["failures"] == 0
        assert stats["providers"][0]["healthy"] is True

    def test_recovery_after_time(self):
        providers = [
            ProviderConfig(base_url="http://api1.com/v1"),
            ProviderConfig(base_url="http://api2.com/v1"),
        ]
        # 设置很短的恢复时间
        router = ProviderRouter(providers, failure_threshold=1, recovery_time=0.1)

        # 标记第一个失败
        provider1 = ProviderConfig(base_url="http://api1.com/v1")
        router.mark_failed(provider1)

        # 只有 1 个健康
        healthy = router.get_all_healthy()
        assert len(healthy) == 1

        # 等待恢复
        time.sleep(0.15)

        # 调用 get_all_healthy 触发恢复检查
        healthy = router.get_all_healthy()
        assert len(healthy) == 2

    def test_all_unhealthy_returns_all(self):
        """所有 provider 都不健康时，返回所有（降级）"""
        providers = [
            ProviderConfig(base_url="http://api1.com/v1"),
            ProviderConfig(base_url="http://api2.com/v1"),
        ]
        router = ProviderRouter(providers, failure_threshold=1)

        # 标记所有失败
        for p in providers:
            router.mark_failed(p)

        # get_all_healthy 应该返回所有（降级行为）
        healthy = router.get_all_healthy()
        assert len(healthy) == 2


class TestGetAllHealthy:
    """测试 get_all_healthy"""

    def test_returns_only_healthy(self):
        providers = [
            ProviderConfig(base_url="http://api1.com/v1"),
            ProviderConfig(base_url="http://api2.com/v1"),
        ]
        router = ProviderRouter(providers, failure_threshold=1)

        router.mark_failed(ProviderConfig(base_url="http://api1.com/v1"))

        healthy = router.get_all_healthy()
        assert len(healthy) == 1
        assert healthy[0].base_url == "http://api2.com/v1"


class TestCapacityAwareRouting:
    """测试容量感知选路（acquire/release）"""

    def _make_router(self, limits: list[int | None], **kwargs) -> ProviderRouter:
        providers = [
            ProviderConfig(base_url=f"http://api{i}.com/v1", concurrency_limit=limit)
            for i, limit in enumerate(limits, 1)
        ]
        return ProviderRouter(providers, **kwargs)

    def test_acquire_increments_release_decrements(self):
        router = self._make_router([10])

        provider = router.acquire()
        assert router.stats["providers"][0]["in_flight"] == 1

        router.release(provider)
        assert router.stats["providers"][0]["in_flight"] == 0

    def test_release_floors_at_zero(self):
        router = self._make_router([10])
        provider = router._providers[0].config

        router.release(provider)
        assert router.stats["providers"][0]["in_flight"] == 0

    def test_picks_lowest_load_ratio_with_heterogeneous_limits(self):
        """异构限额下按比值选：limit=50 用 10 个 (20%) 应胜过 limit=5 用 3 个 (60%)"""
        router = self._make_router([50, 5])
        router._providers[0].in_flight = 10
        router._providers[1].in_flight = 3

        provider = router.acquire()
        assert provider.base_url == "http://api1.com/v1"

    def test_cold_start_spreads_across_endpoints(self):
        """同构 endpoint 连续 acquire（不 release）应交替分布"""
        router = self._make_router([10, 10])

        urls = [router.acquire().base_url for _ in range(4)]
        assert Counter(urls) == {"http://api1.com/v1": 2, "http://api2.com/v1": 2}

    def test_no_limit_balances_by_absolute_count(self):
        """无限额时按绝对 in-flight 数均衡"""
        router = self._make_router([None, None])

        urls = [router.acquire().base_url for _ in range(4)]
        assert Counter(urls) == {"http://api1.com/v1": 2, "http://api2.com/v1": 2}

    def test_all_saturated_falls_back_to_round_robin(self):
        """全饱和时退回轮询，不返回 None（排队是正确行为）"""
        router = self._make_router([1, 1])
        router.acquire()
        router.acquire()  # 两个都到达 limit

        urls = {router.acquire().base_url for _ in range(2)}
        assert urls == {"http://api1.com/v1", "http://api2.com/v1"}
        assert all(p["in_flight"] == 2 for p in router.stats["providers"])

    def test_exclude_filters_candidates(self):
        router = self._make_router([10, 10])
        p1 = router._providers[0].config

        provider = router.acquire(exclude=[p1])
        assert provider.base_url == "http://api2.com/v1"

    def test_all_excluded_returns_none(self):
        router = self._make_router([10, 10])
        p1 = router._providers[0].config
        p2 = router._providers[1].config

        provider = router.acquire(exclude=[p1, p2])
        assert provider is None

    def test_unhealthy_not_selected(self):
        """不健康的 endpoint 即使空闲也不被选中"""
        router = self._make_router([10, 10], failure_threshold=1)
        router._providers[1].in_flight = 5
        router.mark_failed(router._providers[0].config)

        provider = router.acquire()
        assert provider.base_url == "http://api2.com/v1"


class TestStats:
    """测试统计信息"""

    def test_stats_structure(self):
        providers = [ProviderConfig(base_url="http://api.com/v1")]
        router = ProviderRouter(providers)

        stats = router.stats
        assert "total" in stats
        assert "healthy" in stats
        assert "providers" in stats
        assert "in_flight" in stats["providers"][0]


class TestCreateRouterFromUrls:
    """测试便捷函数"""

    def test_create_from_urls(self):
        urls = ["http://api1.com/v1", "http://api2.com/v1"]
        router = create_router_from_urls(urls, api_key="sk-xxx")

        assert router.stats["total"] == 2
        provider = router.get_next()
        assert provider.api_key == "sk-xxx"


class TestLatencyAwareRouting:
    """测试延迟感知选路（observe + cost）"""

    def _make_router(self, limits: list[int | None], **kwargs) -> ProviderRouter:
        providers = [
            ProviderConfig(base_url=f"http://api{i}.com/v1", concurrency_limit=limit)
            for i, limit in enumerate(limits, 1)
        ]
        return ProviderRouter(providers, **kwargs)

    def _serial(self, router: ProviderRouter, latencies: dict[str, float], n: int) -> Counter:
        """模拟严格串行调用：acquire -> observe -> release"""
        picked = Counter()
        for _ in range(n):
            p = router.acquire()
            picked[p.base_url] += 1
            router.observe(p, latencies[p.base_url])
            router.release(p)
        return picked

    def test_serial_calls_prefer_fast_endpoint(self):
        """并发度为 1 时也能避开慢 endpoint —— in-flight 在这里恒为 0，只能靠延迟"""
        router = self._make_router([8, 8, 8], latency_tau=30.0)
        latencies = {
            "http://api1.com/v1": 60.0,  # 慢 30 倍，且排在第一位
            "http://api2.com/v1": 2.0,
            "http://api3.com/v1": 2.0,
        }
        picked = self._serial(router, latencies, 30)

        # 慢 endpoint 只承担探测流量，绝大多数请求落到两个快的上
        assert picked["http://api1.com/v1"] <= 5
        assert picked["http://api2.com/v1"] + picked["http://api3.com/v1"] >= 25

    def test_serial_without_latency_signal_is_unchanged(self):
        """没有 observe 时行为与纯容量感知一致（现状：稳定选第一个）"""
        router = self._make_router([8, 8, 8])
        urls = []
        for _ in range(5):
            p = router.acquire()
            urls.append(p.base_url)
            router.release(p)
        assert set(urls) == {"http://api1.com/v1"}

    def test_cost_uses_parallel_capacity_not_raw_inflight(self):
        """未饱和时不计排队：延迟更低者胜出，即使它 in-flight 更多"""
        router = self._make_router([8, 8])
        p1, p2 = (s.config for s in router._providers)
        router.observe(p1, 1.0)
        router.observe(p2, 10.0)
        router._providers[0].in_flight = 5  # 仍未饱和 -> 不排队
        router._providers[1].in_flight = 0

        assert router.acquire().base_url == "http://api1.com/v1"

    def test_saturation_adds_queueing_penalty(self):
        """饱和后每多一批排队就乘一次延迟，慢但空闲的 endpoint 重新胜出"""
        router = self._make_router([2, 2])
        p1, p2 = (s.config for s in router._providers)
        router.observe(p1, 1.0)
        router.observe(p2, 2.5)
        router._providers[0].in_flight = 4  # 2 批排队 -> cost = 1.0 * 3
        router._providers[1].in_flight = 0  # cost = 2.5 * 1

        assert router.acquire().base_url == "http://api2.com/v1"

    def test_idle_decay_reprobes_slow_endpoint(self):
        """慢 endpoint 闲置期间估计值衰减，在持续繁忙的快 endpoint 对比下重获探测机会

        衰减是相对的：两个 endpoint 一起闲置时估计值同步下降、相对关系不变。
        翻转发生在"快的一直在服务、慢的长期没被选中"时，也就是真实负载的形态。
        """
        router = self._make_router([8, 8], latency_tau=0.05)
        slow, fast = (s.config for s in router._providers)
        router.observe(slow, 10.0)
        router.observe(fast, 1.0)

        assert router.acquire().base_url == "http://api2.com/v1"
        router.release(fast)

        time.sleep(0.4)  # 慢的那个闲置了 8 个 tau
        router.observe(fast, 1.0)  # 快的刚服务完一条，估计值是新鲜的
        assert router.acquire().base_url == "http://api1.com/v1"

    def test_in_flight_blocks_decay(self):
        """在途请求存在时不衰减——否则慢 endpoint 会被反复重新探测"""
        router = self._make_router([8], latency_tau=0.01)
        p = router._providers[0]
        router.observe(p.config, 10.0)
        p.last_active = time.time() - 100.0

        p.in_flight = 1
        assert router._estimate(p, time.time()) == 10.0
        p.in_flight = 0
        assert router._estimate(p, time.time()) < 1.0

    def test_observe_ignores_non_positive(self):
        router = self._make_router([8])
        p = router._providers[0].config
        router.observe(p, 0.0)
        router.observe(p, -1.0)
        assert router.stats["providers"][0]["ewma_latency"] is None

    def test_ewma_smooths_single_outlier(self):
        """单次长回答不应把 endpoint 判死（对称 EWMA，不取峰值）"""
        router = self._make_router([8])
        p = router._providers[0].config
        for _ in range(10):
            router.observe(p, 1.0)
        router.observe(p, 100.0)

        # 取峰值会跳到 100，对称 EWMA 只吸收 alpha 的比例
        assert router.stats["providers"][0]["ewma_latency"] < 30.0

    def test_stats_exposes_latency(self):
        router = self._make_router([8])
        p = router._providers[0].config
        router.observe(p, 3.0)
        assert router.stats["providers"][0]["ewma_latency"] == 3.0
