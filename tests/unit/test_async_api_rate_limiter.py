"""
async_api 模块速率限制器测试

RateLimiter 只有 core.py 一份实现，基于 aiolimiter 漏桶算法（允许突发），
支持小数 QPS（如 0.5 表示每 2 秒 1 个请求）。
"""

import asyncio
import time

import pytest

from flexllm.async_api.core import RateLimiter as CoreRateLimiter


class TestCoreRateLimiter:
    """core.py 中 RateLimiter 的测试"""

    async def test_no_limit(self):
        """测试 max_qps=None 时不限制"""
        limiter = CoreRateLimiter(max_qps=None)

        start = time.time()
        for _ in range(10):
            await limiter.acquire()
        elapsed = time.time() - start

        # 无限制时应该几乎瞬间完成
        assert elapsed < 0.1

    async def test_qps_enforcement(self):
        """测试漏桶模式 QPS 限制"""
        max_qps = 10.0  # 每秒 10 个请求
        limiter = CoreRateLimiter(max_qps=max_qps)

        # aiolimiter 允许突发（burst），前 max_qps 个请求立即放行
        # 超过 burst 后才开始限制
        num_requests = 15  # 超过 burst 阈值

        start = time.time()
        for _ in range(num_requests):
            await limiter.acquire()
        elapsed = time.time() - start

        # 15 个请求，QPS=10，前 10 个立即放行，后 5 个需要等待
        # 理论最小时间 ~0.5s (5 * 0.1s)
        min_expected = (num_requests - max_qps) / max_qps  # 0.5s
        assert elapsed >= min_expected * 0.8  # 允许 20% 误差

    async def test_lazy_init_limiter(self):
        """测试 limiter 延迟初始化"""
        limiter = CoreRateLimiter(max_qps=10.0)

        # 创建时 _limiter 应该为 None
        assert limiter._limiter is None

        # 第一次 acquire 后应该被初始化
        await limiter.acquire()
        assert limiter._limiter is not None

    async def test_concurrent_acquire(self):
        """测试多协程并发 acquire"""
        max_qps = 10.0
        limiter = CoreRateLimiter(max_qps=max_qps)
        num_coroutines = 15  # 超过 burst 阈值

        async def worker():
            await limiter.acquire()
            return time.time()

        start = time.time()
        await asyncio.gather(*[worker() for _ in range(num_coroutines)])
        elapsed = time.time() - start

        # 15 个协程并发，QPS=10，前 10 个立即放行，后 5 个排队
        # 总时间应该 >= 0.5s
        min_expected = (num_coroutines - max_qps) / max_qps
        assert elapsed >= min_expected * 0.8

    async def test_uses_aiolimiter(self):
        """测试底层使用 aiolimiter"""
        limiter = CoreRateLimiter(max_qps=10.0)
        await limiter.acquire()

        # 应该使用 AsyncLimiter
        from aiolimiter import AsyncLimiter

        assert isinstance(limiter._limiter, AsyncLimiter)


class TestFractionalQPS:
    """max_qps < 1 的支持

    回归：AsyncLimiter(0.5, 1) 容量不足 1，acquire(1) 直接抛 ValueError，
    异常又被上层 except Exception 吞成业务错误 → 所有请求 100% 静默失败。
    现在 max_qps < 1 时换算为 AsyncLimiter(1, 1/max_qps)。
    """

    async def test_fractional_qps_acquire_succeeds(self):
        """max_qps=0.5 时 acquire 不得抛异常（修复前首个 acquire 即 ValueError）"""
        limiter = CoreRateLimiter(max_qps=0.5)
        await limiter.acquire()  # 修复前此处抛 ValueError

        # 换算后容量恰为 1：1 个请求每 2 秒
        assert limiter._limiter.max_rate == 1
        assert limiter._limiter.time_period == 2

    @pytest.mark.slow
    async def test_fractional_qps_rate_enforced(self):
        """max_qps=0.5 的速率确实是每 2 秒 1 个请求"""
        limiter = CoreRateLimiter(max_qps=0.5)

        start = time.perf_counter()
        await limiter.acquire()  # 首个立即放行（burst 容量 1）
        await limiter.acquire()  # 第二个需等待 ~2s
        elapsed = time.perf_counter() - start

        assert elapsed >= 2 * 0.8


class TestRateLimiterIsNonBlocking:
    """acquire() 不得阻塞 event loop

    回归：曾有一份 RateLimiter 副本在 `async def acquire()` 里调用同步的
    `time.sleep()`，限流期间整个 loop 冻结。现在只允许 core 这一份实现。
    """

    async def _count_heartbeats_during_acquires(self, limiter, n: int) -> tuple[float, int]:
        """在 acquire 期间跑一个 10ms 心跳，返回 (耗时, 心跳次数)"""
        stop = asyncio.Event()

        async def heartbeat():
            ticks = 0
            while not stop.is_set():
                ticks += 1
                await asyncio.sleep(0.01)
            return ticks

        hb = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.05)  # 让心跳先跑起来

        start = time.perf_counter()
        for _ in range(n):
            await limiter.acquire()
        elapsed = time.perf_counter() - start

        stop.set()
        return elapsed, await hb

    async def test_acquire_yields_to_event_loop(self):
        limiter = CoreRateLimiter(max_qps=5.0)  # burst 5，之后间隔 0.2s
        elapsed, ticks = await self._count_heartbeats_during_acquires(limiter, 9)

        assert elapsed >= 0.5, "限流应当真的生效"
        # 未阻塞时心跳约 elapsed/0.01 次；阻塞实现只能跑到个位数
        assert ticks >= (elapsed / 0.01) * 0.5, (
            f"loop 被阻塞：{elapsed:.2f}s 内只 tick 了 {ticks} 次"
        )


class TestRateLimiterEdgeCases:
    """速率限制器边界情况测试"""

    async def test_very_high_qps(self):
        """测试非常高的 QPS 设置"""
        limiter = CoreRateLimiter(max_qps=10000.0)

        start = time.time()
        for _ in range(100):
            await limiter.acquire()
        elapsed = time.time() - start

        # 高 QPS 应该快速完成
        assert elapsed < 0.5

    async def test_moderate_qps(self):
        """测试中等 QPS 设置（超过 burst 阈值）"""
        max_qps = 10.0
        limiter = CoreRateLimiter(max_qps=max_qps)

        # 发送超过 burst 阈值的请求
        num_requests = 13

        start = time.time()
        for _ in range(num_requests):
            await limiter.acquire()
        elapsed = time.time() - start

        # 13 个请求，QPS=10，前 10 个立即放行，后 3 个需要等待 ~0.3s
        min_expected = (num_requests - max_qps) / max_qps
        assert elapsed >= min_expected * 0.8

    async def test_same_loop_reuses_limiter(self):
        """同一 event loop 内应复用 limiter 实例"""
        limiter = CoreRateLimiter(max_qps=100.0)

        await limiter.acquire()
        first_limiter = limiter._limiter

        await limiter.acquire()
        assert limiter._limiter is first_limiter  # 同一 loop 应该复用


class TestRateLimiterTiming:
    """速率限制器时间精度测试"""

    @pytest.mark.slow
    async def test_timing_accuracy(self):
        """测试漏桶模式时间精度"""
        max_qps = 10.0
        limiter = CoreRateLimiter(max_qps=max_qps)
        # 需要超过 burst 阈值才能看到限制效果
        num_requests = 15

        timestamps = []
        for _ in range(num_requests):
            await limiter.acquire()
            timestamps.append(time.time())

        # 只检查超过 burst 阈值后的间隔（第 10 个请求之后）
        # aiolimiter 允许突发，前 max_qps 个请求立即放行
        post_burst_intervals = [
            timestamps[i + 1] - timestamps[i] for i in range(int(max_qps), len(timestamps) - 1)
        ]

        # 期望间隔 ~0.1s，允许一定误差
        expected_interval = 1 / max_qps
        for interval in post_burst_intervals:
            # 漏桶模式下间隔应该接近期望值
            assert interval >= expected_interval * 0.5  # 允许较大误差
