"""重试退避策略测试

规则：
- 429/5xx 响应带 Retry-After 头时，退避时长以服务端给出的值为准（封顶 max_delay）
- 无 Retry-After 时用指数退避 + equal jitter，避免并发失败请求同步重试形成惊群
- Retry-After 只向上抖动：早于服务端要求重试必然再吃一次 429
"""

import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

from aiohttp import web

from flexllm.async_api.core import ConcurrentRequester, parse_retry_after
from flexllm.utils.core import async_retry, compute_retry_delay


class TestParseRetryAfter:
    def test_delta_seconds(self):
        assert parse_retry_after("20") == 20.0
        assert parse_retry_after("  0.5  ") == 0.5
        assert parse_retry_after("0") == 0.0

    def test_negative_clamped_to_zero(self):
        """过期的时间点不该产生负等待"""
        assert parse_retry_after("-5") == 0.0

    def test_http_date(self):
        when = datetime.now(timezone.utc) + timedelta(seconds=30)
        parsed = parse_retry_after(format_datetime(when, usegmt=True))
        # HTTP-date 精度到秒，允许 2 秒误差
        assert 27 <= parsed <= 31

    def test_http_date_in_past_clamped(self):
        when = datetime.now(timezone.utc) - timedelta(seconds=60)
        assert parse_retry_after(format_datetime(when, usegmt=True)) == 0.0

    def test_unparsable_returns_none(self):
        """头部格式异常不该让请求失败，退回指数退避"""
        for bad in (None, "", "   ", "soon", "inf", "nan", "Wed, 99 Xxx 2015"):
            assert parse_retry_after(bad) is None


class TestComputeRetryDelay:
    def test_exponential_growth(self):
        """无 Retry-After：均值随 attempt 翻倍"""
        for attempt, base in ((0, 1.0), (1, 2.0), (2, 4.0)):
            samples = [compute_retry_delay(attempt, 1.0) for _ in range(200)]
            # equal jitter：落在 [base/2, base]
            assert all(base / 2 <= s <= base for s in samples)

    def test_jitter_is_not_constant(self):
        """固定延迟会让并发失败请求同步重试——必须有抖动"""
        samples = {compute_retry_delay(0, 1.0) for _ in range(50)}
        assert len(samples) > 1

    def test_capped_by_max_delay(self):
        assert compute_retry_delay(20, 1.0, max_delay=10.0) <= 10.0

    def test_retry_after_takes_precedence(self):
        """服务端说等 30 秒，就不该按本地基数 0.3 秒重试"""
        samples = [compute_retry_delay(0, 0.3, retry_after=30.0) for _ in range(200)]
        # 只向上抖动 0~25%
        assert all(30.0 <= s <= 37.5 for s in samples)

    def test_retry_after_capped(self):
        """服务端要求等 600 秒时不能把整个批量任务拖死"""
        assert compute_retry_delay(0, 1.0, max_delay=60.0, retry_after=600.0) <= 75.0


class TestAsyncRetryCallSemantics:
    async def test_zero_retry_times_still_calls_once(self):
        """retry_times<=0 时循环体不执行，但函数仍须被调用一次

        回归：把末尾的兜底调用当成死代码删除后，retry_times=0 的调用方
        （批量路径的默认配置之一）会拿不到任何结果。
        """
        calls = []

        @async_retry(retry_times=0, retry_delay=0.01)
        async def fn():
            calls.append(1)
            return "ok"

        assert await fn() == "ok"
        assert len(calls) == 1


class RateLimitedServer:
    """首次返回 429（可带 Retry-After），之后返回 200"""

    def __init__(self, retry_after: str | None):
        self.retry_after = retry_after
        self.request_count = 0
        self._runner = None
        self.url = None

    async def _handler(self, request):
        self.request_count += 1
        if self.request_count == 1:
            headers = {"Retry-After": self.retry_after} if self.retry_after else {}
            return web.json_response({"error": "rate limited"}, status=429, headers=headers)
        return web.json_response({"result": "ok"})

    async def __aenter__(self):
        app = web.Application()
        app.router.add_post("/test", self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/test"
        return self

    async def __aexit__(self, *args):
        await self._runner.cleanup()


async def _timed_request(server: RateLimitedServer, retry_delay: float) -> float:
    requester = ConcurrentRequester(concurrency_limit=1, retry_times=3, retry_delay=retry_delay)
    t0 = time.perf_counter()
    results, _ = await requester.process_requests(
        request_params=[{"json": {"q": 1}}],
        url=server.url,
        show_progress=False,
    )
    elapsed = time.perf_counter() - t0
    await requester.aclose()
    assert results[0].status == "success"
    return elapsed


class TestRetryAfterEndToEnd:
    async def test_honors_retry_after_header(self):
        """回归：Retry-After 被忽略时，重试会在本地 retry_delay 后立刻打回去再吃 429"""
        async with RateLimitedServer(retry_after="0.4") as server:
            elapsed = await _timed_request(server, retry_delay=0.01)

        assert server.request_count == 2
        assert elapsed >= 0.4, f"未遵守 Retry-After，仅等待 {elapsed:.3f}s"

    async def test_http_date_retry_after(self):
        """HTTP-date 形式的 Retry-After 同样生效

        HTTP-date 精度只到秒，格式化会截断当前时间的亚秒部分，
        因此 +2s 的实际等待落在 (1, 2] 秒。
        """
        when = datetime.now(timezone.utc) + timedelta(seconds=2)
        async with RateLimitedServer(retry_after=format_datetime(when, usegmt=True)) as server:
            elapsed = await _timed_request(server, retry_delay=0.01)

        assert server.request_count == 2
        assert elapsed >= 0.9, f"未遵守 HTTP-date 形式的 Retry-After，仅等待 {elapsed:.3f}s"

    async def test_without_header_uses_local_backoff(self):
        """无 Retry-After 时不受影响，按本地退避基数重试"""
        async with RateLimitedServer(retry_after=None) as server:
            elapsed = await _timed_request(server, retry_delay=0.02)

        assert server.request_count == 2
        assert elapsed < 0.5
