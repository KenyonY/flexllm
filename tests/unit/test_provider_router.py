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
        assert config.weight == 1.0
        assert config.model is None
        assert config.enabled is True

    def test_custom_values(self):
        config = ProviderConfig(
            base_url="http://api.example.com/v1",
            api_key="sk-xxx",
            weight=2.0,
            model="gpt-4",
            enabled=False,
        )
        assert config.api_key == "sk-xxx"
        assert config.weight == 2.0
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
        router = ProviderRouter(providers, strategy="round_robin")

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
        router = ProviderRouter(providers, strategy="round_robin")

        urls = [router.get_next().base_url for _ in range(100)]
        counter = Counter(urls)

        # 应该均匀分布
        assert counter["http://api1.com/v1"] == 50
        assert counter["http://api2.com/v1"] == 50


class TestWeightedStrategy:
    """测试加权策略"""

    def test_weighted_respects_weights(self):
        providers = [
            ProviderConfig(base_url="http://api1.com/v1", weight=1.0),
            ProviderConfig(base_url="http://api2.com/v1", weight=3.0),
        ]
        router = ProviderRouter(providers, strategy="weighted")

        urls = [router.get_next().base_url for _ in range(1000)]
        counter = Counter(urls)

        # weight=3 应该大约是 weight=1 的 3 倍
        ratio = counter["http://api2.com/v1"] / counter["http://api1.com/v1"]
        assert 2.0 < ratio < 4.0  # 允许一定误差


class TestRandomStrategy:
    """测试随机策略"""

    def test_random_uses_all_providers(self):
        providers = [
            ProviderConfig(base_url="http://api1.com/v1"),
            ProviderConfig(base_url="http://api2.com/v1"),
            ProviderConfig(base_url="http://api3.com/v1"),
        ]
        router = ProviderRouter(providers, strategy="random")

        urls = set(router.get_next().base_url for _ in range(100))

        # 应该使用了所有 provider
        assert len(urls) == 3


class TestFallbackStrategy:
    """测试主备策略"""

    def test_fallback_prefers_first(self):
        providers = [
            ProviderConfig(base_url="http://primary.com/v1"),
            ProviderConfig(base_url="http://backup.com/v1"),
        ]
        router = ProviderRouter(providers, strategy="fallback")

        # 应该总是返回第一个
        for _ in range(10):
            assert router.get_next().base_url == "http://primary.com/v1"

    def test_fallback_switches_on_failure(self):
        providers = [
            ProviderConfig(base_url="http://primary.com/v1"),
            ProviderConfig(base_url="http://backup.com/v1"),
        ]
        router = ProviderRouter(providers, strategy="fallback", failure_threshold=2)

        primary = router.get_next()
        assert primary.base_url == "http://primary.com/v1"

        # 标记失败 2 次
        router.mark_failed(primary)
        router.mark_failed(primary)

        # 现在应该返回 backup
        assert router.get_next().base_url == "http://backup.com/v1"


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


class TestStats:
    """测试统计信息"""

    def test_stats_structure(self):
        providers = [ProviderConfig(base_url="http://api.com/v1")]
        router = ProviderRouter(providers, strategy="round_robin")

        stats = router.stats
        assert "total" in stats
        assert "healthy" in stats
        assert "strategy" in stats
        assert "providers" in stats
        assert stats["strategy"] == "round_robin"


class TestCreateRouterFromUrls:
    """测试便捷函数"""

    def test_create_from_urls(self):
        urls = ["http://api1.com/v1", "http://api2.com/v1"]
        router = create_router_from_urls(urls, api_key="sk-xxx")

        assert router.stats["total"] == 2
        provider = router.get_next()
        assert provider.api_key == "sk-xxx"

    def test_create_with_strategy(self):
        urls = ["http://api1.com/v1", "http://api2.com/v1"]
        router = create_router_from_urls(urls, strategy="fallback")

        assert router.strategy == "fallback"
