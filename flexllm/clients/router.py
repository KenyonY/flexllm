#! /usr/bin/env python3

"""
多 Provider 负载均衡和故障转移

支持多个 API endpoint 的轮询分配和自动 fallback。
"""

import math
import time
from dataclasses import dataclass
from threading import Lock

# 延迟估计的 EWMA 权重：有效窗口约 1/alpha 个样本。
# 固定值而非时间感知权重——闲置衰减已经负责"信息过期"，两者叠加只会让行为更难预测。
_EWMA_ALPHA = 0.25


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
    # 服务耗时的 EWMA 估计（秒）。None = 还没有成功样本，选路时按最乐观处理。
    ewma_latency: float | None = None
    # 最后一次交互（被选中 / 拿到样本 / 请求结束）时的全局选路序号，闲置衰减以它为基准。
    last_active_seq: int = 0


class ProviderRouter:
    """
    Provider 路由器

    选路策略：延迟感知 + 容量感知。两者分工明确——

    - 容量：饱和（in_flight >= concurrency_limit）的 provider 直接出局，全部饱和时
      退回轮询，此时排队是正确行为
    - 延迟：在剩下的（未饱和，即新请求无需排队）候选里选延迟估计最低者

    这样并发度为 1 的逐条调用也能避开慢 endpoint——in-flight 在那里恒为 0，
    提供不了任何信息，只能靠延迟。并发升高后饱和过滤接手，回到容量感知。

    没有设 concurrency_limit 的 provider 无从判断饱和，改用 Finagle 的
    cost = rtt * (in_flight + 1)，让 in-flight 直接进入比较。

    还没有延迟样本的 provider cost 记 0（最乐观），保证每个 endpoint 至少被探测一次；
    cost 相同时按负载率、绝对 in-flight 决胜，即冷启动阶段行为与纯容量感知一致。

    支持健康检查和自动恢复。

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
        latency_decay_calls: float = 10.0,
    ):
        """
        初始化路由器

        Args:
            providers: Provider 配置列表
            failure_threshold: 连续失败多少次后标记为不健康
            recovery_time: 不健康后多久尝试恢复 (秒)
            latency_decay_calls: 延迟估计的闲置衰减常数，单位是"选路次数"。某个
                endpoint 每被跳过 N 次，其延迟估计衰减到约 37%，慢 endpoint 借此
                周期性地被重新探测：慢 K 倍的 endpoint 大约每 N*ln(K) 次调用被探测
                一次。用次数而非秒，是因为 endpoint 的快慢不会因为没人调用就改变，
                而探测的代价恰恰是按请求计的——按墙上时间衰减会让稀疏/交互式调用
                （每次调用时所有 endpoint 都已闲置很久）退化成轮询。设为 0 关闭衰减。
        """
        if not providers:
            raise ValueError("至少需要一个 provider")

        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.latency_decay_calls = latency_decay_calls

        self._providers = [ProviderStatus(config=p) for p in providers if p.enabled]
        self._index = 0
        self._seq = 0
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

    def _estimate(self, status: ProviderStatus) -> float | None:
        """当前延迟估计，None 表示还没有样本

        闲置按"被跳过了多少次选路"衡量，不按秒：endpoint 的快慢不会因为没人调用就
        改变，而探测的代价恰恰是按请求计的。按墙上时间衰减时，稀疏/交互式调用下每次
        选路所有 endpoint 都已闲置远超 tau，衰减量由"谁最久没被选中"主导，延迟信息
        被完全抹掉，退化成轮询——慢 endpoint 照样吃到 1/n 的流量。

        基准是"最后一次交互"（被选中 / 拿到样本 / 请求结束），且在途请求存在时不
        衰减：衰减表达的是"这份估计有多旧"，而请求进行中和刚结束时它都是最新的。
        基准若取"上次拿到样本"，慢 endpoint 的样本间隔天然更长，会被反复重新探测。
        """
        if status.ewma_latency is None:
            return None
        if status.in_flight > 0 or self.latency_decay_calls <= 0:
            return status.ewma_latency
        skipped = self._seq - status.last_active_seq
        if skipped <= 0:
            return status.ewma_latency
        return status.ewma_latency * math.exp(-skipped / self.latency_decay_calls)

    def _cost(self, status: ProviderStatus) -> float:
        """期望完成时间。无样本记 0（最乐观），保证每个 endpoint 至少被探测一次

        设了 concurrency_limit 的 endpoint 直接返回延迟估计：acquire 只在未饱和的
        候选之间比 cost，而未饱和意味着新请求无需排队，期望完成时间就是服务耗时本身。
        容量维度由"饱和即出局"承担，不需要在 cost 里再算一次。

        没有限额时无从判断饱和，用 in-flight 当排队深度的代理，即 Finagle 的
        cost = rtt * (in_flight + 1)。
        """
        estimate = self._estimate(status)
        if estimate is None:
            return 0.0
        if status.config.concurrency_limit is None:
            return estimate * (status.in_flight + 1)
        return estimate

    def observe(self, provider: ProviderConfig, service_time: float) -> None:
        """记录一次成功请求的服务耗时，更新该 provider 的延迟估计

        只喂成功样本：失败往往是快速返回的，计入会让坏 endpoint 显得更快而吸引流量，
        健康检查（mark_failed）才是处理失败的地方。

        Args:
            provider: acquire() 返回的 provider 配置
            service_time: 该 endpoint 的服务耗时 (秒)，应扣除客户端本地排队
        """
        if service_time <= 0:
            return
        with self._lock:
            for p in self._providers:
                if self._matches(p, provider):
                    if p.ewma_latency is None:
                        p.ewma_latency = service_time
                    else:
                        p.ewma_latency = (
                            p.ewma_latency * (1 - _EWMA_ALPHA) + service_time * _EWMA_ALPHA
                        )
                    p.last_active_seq = self._seq
                    break

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
        - 未饱和（in_flight < concurrency_limit）的候选按 _cost() 取最低；cost 相同
          （只可能是都还没有延迟样本）时按 in_flight/concurrency_limit 比值决胜，
          异构限额下比绝对计数公平
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

            self._seq += 1
            available = [
                p
                for p in candidates
                if p.config.concurrency_limit is None or p.in_flight < p.config.concurrency_limit
            ]
            if available:
                costs = [self._cost(p) for p in available]
                cheapest = min(costs)
                # 纯相对容差：闲置衰减是等比的，长时间闲置后各 endpoint 的估计值
                # 会一起衰减到极小，但彼此的比例不变，仍然可比。掺绝对容差会在那时
                # 把所有候选判成并列，退回负载率后又稳定选回列表首个。
                # cheapest 为 0 只可能是"都还没有样本"，那时才该让所有候选并列。
                tolerance = cheapest * (1 + 1e-9) if cheapest > 0 else 0.0
                finalists = [p for p, cost in zip(available, costs) if cost <= tolerance]
                chosen = min(
                    finalists,
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
            chosen.last_active_seq = self._seq
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
                    p.last_active_seq = self._seq
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
                        "ewma_latency": p.ewma_latency,
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
