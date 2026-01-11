#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多 Provider 负载均衡和故障转移

支持多个 API endpoint 的轮询、加权分配和自动 fallback。
"""

import time
import random
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from threading import Lock


Strategy = Literal["round_robin", "weighted", "random", "fallback"]


@dataclass
class ProviderConfig:
    """
    单个 Provider 配置

    Attributes:
        base_url: API 基础 URL
        api_key: API 密钥
        weight: 权重 (用于 weighted 策略)
        model: 可选的模型覆盖
        enabled: 是否启用
    """
    base_url: str
    api_key: str = "EMPTY"
    weight: float = 1.0
    model: Optional[str] = None
    enabled: bool = True


@dataclass
class ProviderStatus:
    """Provider 运行时状态"""
    config: ProviderConfig
    failures: int = 0
    last_failure: float = 0
    is_healthy: bool = True


class ProviderRouter:
    """
    Provider 路由器

    支持策略:
        - round_robin: 轮询
        - weighted: 加权随机
        - random: 随机
        - fallback: 主备模式 (只有主挂了才用备)
    """

    def __init__(
        self,
        providers: List[ProviderConfig],
        strategy: Strategy = "round_robin",
        failure_threshold: int = 3,
        recovery_time: float = 60.0,
    ):
        """
        初始化路由器

        Args:
            providers: Provider 配置列表
            strategy: 路由策略
            failure_threshold: 连续失败多少次后标记为不健康
            recovery_time: 不健康后多久尝试恢复 (秒)
        """
        if not providers:
            raise ValueError("至少需要一个 provider")

        self.strategy = strategy
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time

        self._providers = [
            ProviderStatus(config=p) for p in providers if p.enabled
        ]
        self._index = 0
        self._lock = Lock()

        if not self._providers:
            raise ValueError("没有可用的 provider")

    def _get_healthy_providers(self) -> List[ProviderStatus]:
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

    def get_next(self) -> ProviderConfig:
        """
        获取下一个可用的 provider

        Returns:
            ProviderConfig
        """
        with self._lock:
            healthy = self._get_healthy_providers()

            if self.strategy == "round_robin":
                self._index = (self._index + 1) % len(healthy)
                return healthy[self._index].config

            elif self.strategy == "weighted":
                weights = [p.config.weight for p in healthy]
                total = sum(weights)
                r = random.uniform(0, total)
                cumsum = 0
                for p in healthy:
                    cumsum += p.config.weight
                    if r <= cumsum:
                        return p.config
                return healthy[-1].config

            elif self.strategy == "random":
                return random.choice(healthy).config

            elif self.strategy == "fallback":
                # 优先使用第一个健康的
                return healthy[0].config

            else:
                return healthy[0].config

    def mark_failed(self, provider: ProviderConfig) -> None:
        """
        标记 provider 失败

        Args:
            provider: 失败的 provider 配置
        """
        with self._lock:
            for p in self._providers:
                if p.config.base_url == provider.base_url:
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
                if p.config.base_url == provider.base_url:
                    p.failures = 0
                    p.is_healthy = True
                    break

    def get_all_healthy(self) -> List[ProviderConfig]:
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
                "strategy": self.strategy,
                "providers": [
                    {
                        "base_url": p.config.base_url,
                        "healthy": p.is_healthy,
                        "failures": p.failures,
                    }
                    for p in self._providers
                ],
            }


def create_router_from_urls(
    urls: List[str],
    api_key: str = "EMPTY",
    strategy: Strategy = "round_robin",
) -> ProviderRouter:
    """
    便捷函数：从 URL 列表创建路由器

    Args:
        urls: API URL 列表
        api_key: 统一的 API 密钥
        strategy: 路由策略

    Returns:
        ProviderRouter 实例
    """
    providers = [ProviderConfig(base_url=url, api_key=api_key) for url in urls]
    return ProviderRouter(providers, strategy=strategy)
