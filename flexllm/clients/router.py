#! /usr/bin/env python3

"""
多 Provider 负载均衡和故障转移

支持多个 API endpoint 的轮询分配和自动 fallback。
"""

import time
from dataclasses import dataclass
from threading import Lock


@dataclass
class ProviderConfig:
    """
    单个 Provider 配置

    Attributes:
        base_url: API 基础 URL
        api_key: API 密钥
        model: 可选的模型覆盖
        enabled: 是否启用
        concurrency_limit: 该 endpoint 的并发上限（容量感知选路用），None 表示无上限
    """

    base_url: str
    api_key: str = "EMPTY"
    model: str | None = None
    enabled: bool = True
    concurrency_limit: int | None = None


@dataclass
class ProviderStatus:
    """Provider 运行时状态"""

    config: ProviderConfig
    failures: int = 0
    last_failure: float = 0
    is_healthy: bool = True
    in_flight: int = 0


class ProviderRouter:
    """
    Provider 路由器

    选路策略：容量感知（acquire/release，在健康且未饱和的 provider 中选负载率最低者，
    全部饱和时退回轮询）。支持健康检查和自动恢复。

    Provider 匹配语义：release/mark_failed/mark_success 按 acquire 返回的
    ProviderConfig 对象身份（is）匹配，其次按值相等（==）兜底。
    不按 base_url 匹配——相同 base_url 不同 api_key 的 endpoint 是不同 provider，
    按 base_url 匹配会导致健康状态串扰。
    """

    def __init__(
        self,
        providers: list[ProviderConfig],
        failure_threshold: int | float = float("inf"),
        recovery_time: float = 60.0,
    ):
        """
        初始化路由器

        Args:
            providers: Provider 配置列表
            failure_threshold: 连续失败多少次后标记为不健康
            recovery_time: 不健康后多久尝试恢复 (秒)
        """
        if not providers:
            raise ValueError("至少需要一个 provider")

        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time

        self._providers = [ProviderStatus(config=p) for p in providers if p.enabled]
        self._index = 0
        self._lock = Lock()

        if not self._providers:
            raise ValueError("没有可用的 provider")

    def _get_healthy_providers(self) -> list[ProviderStatus]:
        """获取健康的 provider 列表"""
        now = time.time()
        healthy = []

        for p in self._providers:
            # 尝试恢复
            if not p.is_healthy and (now - p.last_failure) > self.recovery_time:
                p.is_healthy = True
                p.failures = 0

            if p.is_healthy:
                healthy.append(p)

        return healthy if healthy else self._providers  # 全挂时返回所有

    @staticmethod
    def _matches(status: ProviderStatus, provider: ProviderConfig) -> bool:
        """按对象身份（优先）或值相等匹配 provider"""
        return status.config is provider or status.config == provider

    def get_next(self) -> ProviderConfig:
        """
        获取下一个可用的 provider（纯轮询策略，不计 in-flight）

        Returns:
            ProviderConfig
        """
        with self._lock:
            healthy = self._get_healthy_providers()
            provider = healthy[self._index % len(healthy)].config
            self._index += 1
            return provider

    def acquire(self, exclude: list[ProviderConfig] | None = None) -> ProviderConfig | None:
        """
        容量感知选路：在健康且未饱和的 provider 中选负载率最低者

        in-flight 计数 +1，请求结束后必须配对调用 release()（try/finally 保证）。

        选路规则：
        - 过滤：健康 且 不在 exclude 中（按 ProviderConfig 对象匹配，非 base_url）
        - 未饱和（in_flight < concurrency_limit）的候选按 in_flight/concurrency_limit
          比值取最低——异构限额下比绝对计数公平
        - 全部饱和时退回轮询，此时排队是正确行为

        Args:
            exclude: 需要跳过的 ProviderConfig 列表（fallback 场景传入已尝试过的 provider）

        Returns:
            选中的 ProviderConfig；无候选（健康的都被排除）时返回 None
        """
        exclude = exclude or []
        with self._lock:
            candidates = [
                p
                for p in self._get_healthy_providers()
                if not any(self._matches(p, e) for e in exclude)
            ]
            if not candidates:
                return None

            available = [
                p
                for p in candidates
                if p.config.concurrency_limit is None or p.in_flight < p.config.concurrency_limit
            ]
            if available:
                chosen = min(
                    available,
                    key=lambda p: (
                        p.in_flight / p.config.concurrency_limit
                        if p.config.concurrency_limit
                        else 0.0,
                        p.in_flight,
                    ),
                )
            else:
                chosen = candidates[self._index % len(candidates)]
                self._index += 1

            chosen.in_flight += 1
            return chosen.config

    def release(self, provider: ProviderConfig) -> None:
        """
        请求结束（正常返回或异常），in-flight 计数 -1

        与 acquire() 配对调用。

        Args:
            provider: acquire() 返回的 provider 配置
        """
        with self._lock:
            for p in self._providers:
                if self._matches(p, provider):
                    p.in_flight = max(0, p.in_flight - 1)
                    break

    def mark_failed(self, provider: ProviderConfig) -> None:
        """
        标记 provider 失败

        Args:
            provider: 失败的 provider 配置
        """
        with self._lock:
            for p in self._providers:
                if self._matches(p, provider):
                    p.failures += 1
                    p.last_failure = time.time()
                    if p.failures >= self.failure_threshold:
                        p.is_healthy = False
                    break

    def mark_success(self, provider: ProviderConfig) -> None:
        """
        标记 provider 成功，重置失败计数

        Args:
            provider: 成功的 provider 配置
        """
        with self._lock:
            for p in self._providers:
                if self._matches(p, provider):
                    p.failures = 0
                    p.is_healthy = True
                    break

    def get_all_healthy(self) -> list[ProviderConfig]:
        """获取所有健康的 provider"""
        with self._lock:
            return [p.config for p in self._get_healthy_providers()]

    @property
    def stats(self) -> dict:
        """返回路由器统计信息"""
        with self._lock:
            return {
                "total": len(self._providers),
                "healthy": sum(1 for p in self._providers if p.is_healthy),
                "providers": [
                    {
                        "base_url": p.config.base_url,
                        "healthy": p.is_healthy,
                        "failures": p.failures,
                        "in_flight": p.in_flight,
                    }
                    for p in self._providers
                ],
            }


def create_router_from_urls(
    urls: list[str],
    api_key: str = "EMPTY",
) -> ProviderRouter:
    """
    便捷函数：从 URL 列表创建路由器

    Args:
        urls: API URL 列表
        api_key: 统一的 API 密钥

    Returns:
        ProviderRouter 实例
    """
    providers = [ProviderConfig(base_url=url, api_key=api_key) for url in urls]
    return ProviderRouter(providers)
