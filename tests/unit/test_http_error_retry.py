"""HTTP 错误分类重试测试

规则（async_api/core.py::_make_requests）：
- 429 / 5xx：可重试，进入 async_retry；重试耗尽后返回 error 结果并保留响应体
- 其他 4xx（400/401/404 等）：重试不可能成功，不重试，直接返回 error 结果并保留响应体
- 2xx：正常返回
"""

from aiohttp import web

from flexllm.async_api.core import ConcurrentRequester


class FlakyServer:
    """按预设状态码序列响应的本地 HTTP 服务器，记录收到的请求数"""

    def __init__(self, status_sequence: list[int]):
        self.status_sequence = status_sequence
        self.request_count = 0
        self._runner = None
        self.url = None

    async def _handler(self, request):
        idx = min(self.request_count, len(self.status_sequence) - 1)
        status = self.status_sequence[idx]
        self.request_count += 1
        if status == 200:
            return web.json_response({"result": "ok"})
        return web.json_response({"error": {"message": f"server says {status}"}}, status=status)

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


async def _request_once(server: FlakyServer, retry_times: int = 3):
    requester = ConcurrentRequester(concurrency_limit=1, retry_times=retry_times, retry_delay=0.01)
    results, _ = await requester.process_requests(
        request_params=[{"json": {"q": 1}}],
        url=server.url,
        show_progress=False,
    )
    await requester.aclose()
    return results[0]


class TestRetryableErrors:
    async def test_429_retries_until_success(self):
        """429 两次后恢复：应重试并最终成功"""
        async with FlakyServer([429, 429, 200]) as server:
            result = await _request_once(server, retry_times=3)

        assert server.request_count == 3
        assert result.status == "success"
        assert result.data == {"result": "ok"}

    async def test_500_retries_until_success(self):
        """500 一次后恢复：应重试并最终成功"""
        async with FlakyServer([500, 200]) as server:
            result = await _request_once(server, retry_times=3)

        assert server.request_count == 2
        assert result.status == "success"

    async def test_retry_exhausted_preserves_body(self):
        """持续 503 重试耗尽：error 结果保留服务器返回的响应体"""
        async with FlakyServer([503]) as server:
            result = await _request_once(server, retry_times=2)

        assert server.request_count == 2  # retry_times 即总尝试次数
        assert result.status == "error"
        assert result.data["status_code"] == 503
        assert result.data["response_data"]["error"]["message"] == "server says 503"


class TestNonRetryableErrors:
    async def test_400_does_not_retry(self):
        """400 参数错误：不应重试，且保留响应体"""
        async with FlakyServer([400, 200]) as server:
            result = await _request_once(server, retry_times=3)

        assert server.request_count == 1  # 只请求一次
        assert result.status == "error"
        assert result.data["status_code"] == 400
        assert result.data["response_data"]["error"]["message"] == "server says 400"

    async def test_401_does_not_retry(self):
        """401 认证错误：不应重试"""
        async with FlakyServer([401]) as server:
            result = await _request_once(server, retry_times=3)

        assert server.request_count == 1
        assert result.status == "error"
        assert result.data["status_code"] == 401


class TestConcurrentMixed:
    async def test_batch_mixed_statuses(self):
        """批量请求中混合成功与失败互不影响"""

        async with FlakyServer([200]) as server:
            requester = ConcurrentRequester(concurrency_limit=4, retry_times=2, retry_delay=0.01)
            results, _ = await requester.process_requests(
                request_params=[{"json": {"q": i}} for i in range(4)],
                url=server.url,
                show_progress=False,
            )
            await requester.aclose()

        assert all(r.status == "success" for r in results)
