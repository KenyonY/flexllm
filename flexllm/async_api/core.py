import asyncio
import contextvars
import itertools
import math
import time
from asyncio import Queue
from collections.abc import AsyncGenerator, Callable, Iterable
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import (
    Any,
)

from aiohttp import (
    ClientConnectionError,
    ClientPayloadError,
    ClientSession,
    ClientTimeout,
    TCPConnector,
)

from ..utils.core import async_retry
from .interface import RequestResult
from .progress import ProgressBarConfig, ProgressTracker


@dataclass
class StreamingResult:
    completed_requests: list[RequestResult]
    progress: ProgressTracker | None
    is_final: bool


class RetryableHTTPError(Exception):
    """可重试的 HTTP 错误（429 限流 / 5xx 服务端错误），携带响应体供重试耗尽后上报

    retry_after 来自响应头，async_retry 会优先用它决定退避时长。
    """

    def __init__(self, status_code: int, response_data: Any, retry_after: float | None = None):
        self.status_code = status_code
        self.response_data = response_data
        self.retry_after = retry_after
        super().__init__(f"HTTP {status_code}")


class JSONDecodeHTTPError(Exception):
    """2xx 响应但响应体不是合法 JSON（如网关返回 text/html）。

    确定性错误：重试不会改变结果，不进入重试；保留响应体文本供上层报错。
    """

    def __init__(self, status_code: int, body_text: str | None):
        self.status_code = status_code
        self.body_text = body_text
        super().__init__(f"HTTP {status_code}: response body is not valid JSON")


# 可重试的异常：网络/超时类瞬态错误 + 服务端明确可重试的状态码。
# 确定性错误（ContentTypeError、InvalidURL、TypeError 等）重试也不会成功，不在其中。
# 与 _make_requests 的 HTTP 状态码分类（429/5xx 可重试，其他 4xx 不重试）保持一致。
RETRYABLE_EXCEPTIONS = (
    RetryableHTTPError,  # 429 / 5xx（_make_requests 主动抛出）
    ClientConnectionError,  # 连接断开/重置/OS 层网络错误（含 ServerTimeoutError）
    ClientPayloadError,  # 响应体传输中断
    asyncio.TimeoutError,  # 请求超时（Py3.11+ 即内建 TimeoutError）
    TimeoutError,
)


def parse_retry_after(value: str | None) -> float | None:
    """解析 Retry-After 响应头（RFC 7231），返回需等待的秒数。

    两种合法形式：delta-seconds（`Retry-After: 20`）与 HTTP-date
    （`Retry-After: Wed, 21 Oct 2015 07:28:00 GMT`）。无法解析时返回 None，
    由调用方退回指数退避——头部格式异常不该让请求直接失败。
    """
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
        if math.isfinite(seconds):
            return max(0.0, seconds)
        return None
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        # RFC 规定 HTTP-date 为 GMT，缺时区信息时按 UTC 解释
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


# SOCKS 代理走 connector 层（aiohttp 原生的 proxy= 参数只支持 HTTP CONNECT），
# 需要可选依赖 aiohttp-socks。socks5h 是 curl 的写法，python_socks 不认，
# 但 SOCKS5 的 rdns 默认为 True（域名交由代理解析）即 socks5h 语义，故等价规范化。
SOCKS_SCHEMES = ("socks4", "socks5", "socks5h")


def is_socks_proxy(proxy: str | None) -> bool:
    """代理是否为 SOCKS（决定走 connector 层还是 aiohttp 的 proxy= 参数）"""
    if not proxy:
        return False
    return proxy.split("://", 1)[0].lower() in SOCKS_SCHEMES


def create_proxied_session(proxy: str | None, **session_kwargs) -> tuple[ClientSession, dict]:
    """为独立 session（各客户端的流式路径）构造带代理的 ClientSession。

    返回 `(session, request_kwargs)`：SOCKS 的隧道由 connector 建立，此时
    request_kwargs 为空——再传 `proxy=` 会让 aiohttp 对着 SOCKS 端口发 HTTP
    CONNECT；HTTP 代理反之，走 request 级 `proxy=` 参数。

    流式路径不走 ConcurrentRequester（各客户端自建 session），这里与
    `ConcurrentRequester._create_session` / `make_requests` 保持同一套语义。
    """
    if is_socks_proxy(proxy):
        from aiohttp_socks import ProxyConnector

        connector = ProxyConnector.from_url(proxy)
        return ClientSession(connector=connector, trust_env=True, **session_kwargs), {}
    session = ClientSession(trust_env=True, **session_kwargs)
    return session, ({"proxy": proxy} if proxy else {})


def validate_proxy(proxy: str | None) -> str | None:
    """校验并规范化正向代理 URL。

    接受 http(s):// 与 socks4/socks5/socks5h://，其余 scheme 直接报错：
    aiohttp 不校验 scheme，传未知 scheme 它会照样往该端口发 HTTP CONNECT，
    在 SOCKS 服务端上表现为莫名其妙的连接错误。与其让它在运行时以难以定位的
    方式失败，不如构造时就报错。

    SOCKS 依赖 aiohttp-socks，未安装时同样在构造时报错而非发请求时才炸。
    """
    if proxy is None:
        return None
    scheme = proxy.split("://", 1)[0].lower() if "://" in proxy else ""
    if scheme in SOCKS_SCHEMES:
        try:
            import aiohttp_socks  # noqa: F401
        except ImportError as e:
            raise ValueError(
                f"SOCKS 代理 {proxy!r} 需要额外依赖：pip install 'flexllm[socks]'"
            ) from e
        if scheme == "socks5h":
            # python_socks 不认 socks5h；SOCKS5 默认 rdns=True，语义完全等价
            return "socks5://" + proxy.split("://", 1)[1]
        return proxy
    if scheme not in ("http", "https"):
        raise ValueError(
            f"不支持的代理 scheme: {proxy!r}。支持 http://、https:// 与 "
            f"socks4://、socks5://、socks5h://。带认证用 scheme://user:pass@host:port。"
        )
    return proxy


class RateLimiter:
    """
    速率限制器（aiolimiter 漏桶算法）

    支持边界：limiter 做 lazy init 并在检测到 event loop 变化时重建，
    仅支持"串行多次 asyncio.run"场景；多个 event loop 并发使用同一实例不受支持。

    Args:
        max_qps: 每秒最大请求数，支持小数（如 0.5 表示每 2 秒 1 个请求）
    """

    def __init__(self, max_qps: float | None = None):
        self.max_qps = max_qps
        # lazy init，避免绑定错误的 event loop（多次 asyncio.run 场景）
        self._limiter = None

    def _get_limiter(self):
        """获取或创建 limiter（确保绑定到当前 event loop）

        注：_loop 是 asyncio 对象的内部属性，可能在 Python 版本间变化，
        使用 getattr 安全获取。
        """
        if not self.max_qps:
            return None
        try:
            loop = asyncio.get_running_loop()
            if self._limiter is not None:
                # aiolimiter.AsyncLimiter 内部也有 _loop 属性
                limiter_loop = getattr(self._limiter, "_loop", None)
                if limiter_loop is not None and limiter_loop is not loop:
                    self._limiter = None
            if self._limiter is None:
                from aiolimiter import AsyncLimiter

                # AsyncLimiter 要求容量 >= 单次 acquire 量（1）。
                # max_qps < 1 时 AsyncLimiter(max_qps, 1) 的容量不足 1，
                # acquire(1) 会直接抛 ValueError，因此换算为
                # "1 个请求每 1/max_qps 秒"，速率等价且容量恰为 1。
                if self.max_qps < 1:
                    self._limiter = AsyncLimiter(1, 1 / self.max_qps)
                else:
                    self._limiter = AsyncLimiter(self.max_qps, 1)
        except RuntimeError:
            # 没有运行的 event loop，不应该发生（acquire 在 async 中调用）
            pass
        return self._limiter

    async def acquire(self):
        if not self.max_qps:
            return
        await self._get_limiter().acquire()


class ConcurrencyLimiter:
    """并发上限限制器（asyncio.Semaphore 的 lazy 重绑定包装）

    lazy init 以避免绑定错误的 event loop（多次 asyncio.run 场景），与 RateLimiter 一致，
    同样仅支持"串行多次 asyncio.run"；多个 event loop 并发使用同一实例不受支持。
    多个 ConcurrentRequester 共享同一实例时，即构成跨 endpoint 的全局并发硬上限。
    """

    def __init__(self, limit: int):
        self.limit = limit
        self._semaphore: asyncio.Semaphore | None = None
        # __aenter__ 记录本次 acquire 用的信号量，__aexit__ 释放同一个对象：
        # 若两次之间发生 loop 切换导致 _semaphore 被重建，release 错对象会
        # 造成新信号量凭空多出配额。ContextVar 随 task 隔离，并发安全。
        self._held: contextvars.ContextVar[asyncio.Semaphore | None] = contextvars.ContextVar(
            "concurrency_limiter_held", default=None
        )

    def _get_semaphore(self) -> asyncio.Semaphore:
        try:
            loop = asyncio.get_running_loop()
            if self._semaphore is not None:
                # Python 3.10+ Semaphore 有 _loop 属性（内部）
                sem_loop = getattr(self._semaphore, "_loop", None)
                if sem_loop is not None and sem_loop is not loop:
                    self._semaphore = None
        except RuntimeError:
            pass
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.limit)
        return self._semaphore

    async def __aenter__(self):
        semaphore = self._get_semaphore()
        await semaphore.acquire()
        self._held.set(semaphore)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        semaphore = self._held.get()
        self._held.set(None)
        semaphore.release()


class ConcurrentRequester:
    """
    并发请求管理器

    Example
    -------

    requester = ConcurrentRequester(
        concurrency_limit=5,
        max_qps=10,
        timeout=0.7,
    )

    request_params = [
        {
            'json': {
                'messages': [{"role": "user", "content": "讲个笑话" }],
                'model': "qwen2.5:latest",
            },
            'headers': {'Content-Type': 'application/json'}
        } for i in range(10)
    ]

    # 执行并发请求
    results, tracker = await requester.process_requests(
        request_params=request_params,
        url='http://localhost:11434/v1/chat/completions',
        method='POST',
        show_progress=True
    )
    """

    def __init__(
        self,
        concurrency_limit: int,
        max_qps: float | None = None,
        timeout: float | None = None,
        retry_times: int = 3,
        retry_delay: float = 0.3,
        proxy: str | None = None,
        pool_semaphore: ConcurrencyLimiter | None = None,
        pool_rate_limiter: RateLimiter | None = None,
    ):
        self._concurrency_limit = concurrency_limit
        self._proxy = validate_proxy(proxy)
        self._proxy_is_socks = is_socks_proxy(self._proxy)
        if timeout:
            self._timeout = ClientTimeout(total=timeout, connect=min(10.0, timeout))
        else:
            self._timeout = None
        self._rate_limiter = RateLimiter(max_qps)
        self._limiter = ConcurrencyLimiter(concurrency_limit)
        # pool 级共享限制器（跨多个 requester 的全局硬上限），由 LLMClientPool 注入。
        # 获取顺序固定为 endpoint sem → pool sem：等待 pool slot 时只占着自己
        # endpoint 的 slot（非全局稀缺资源），不会出现"攥着全局 slot 在某个
        # endpoint 队列里干等"的队头阻塞。
        self._pool_semaphore = pool_semaphore
        self._pool_rate_limiter = pool_rate_limiter
        self.retry_times = retry_times
        self.retry_delay = retry_delay

        # Session 复用：避免每次请求都创建新的 session
        self._connector: TCPConnector | None = None
        self._session: ClientSession | None = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        """获取或创建 Semaphore（确保绑定到当前 event loop）"""
        return self._limiter._get_semaphore()

    def _create_session(self) -> ClientSession:
        """创建新的 session（内部使用）"""
        connector_kwargs = dict(
            limit=self._concurrency_limit + 10, limit_per_host=0, force_close=False
        )
        if self._proxy_is_socks:
            # SOCKS 必须在 connector 层建隧道：aiohttp 的 proxy= 参数只会发 HTTP
            # CONNECT。ProxyConnector 继承自 TCPConnector，其余逻辑（loop 绑定
            # 检查、关闭）无需区分。依赖已在 validate_proxy 中校验过。
            from aiohttp_socks import ProxyConnector

            self._connector = ProxyConnector.from_url(self._proxy, **connector_kwargs)
        else:
            self._connector = TCPConnector(**connector_kwargs)
        self._session = ClientSession(
            timeout=self._timeout, connector=self._connector, trust_env=True
        )
        return self._session

    def _is_session_valid(self) -> bool:
        """检查 session 是否有效（存在、未关闭、且绑定到当前 event loop）"""
        if self._session is None or self._session.closed:
            return False
        # 检查 session 的 connector 是否绑定到当前 loop
        try:
            current_loop = asyncio.get_running_loop()
            if self._connector is not None:
                connector_loop = getattr(self._connector, "_loop", None)
                if connector_loop is not None and connector_loop is not current_loop:
                    return False
        except RuntimeError:
            pass
        return True

    @asynccontextmanager
    async def _get_session(self):
        """获取或创建 session（复用模式，确保绑定到当前 event loop）

        支持边界：与 RateLimiter/ConcurrencyLimiter 一致，loop 变化时重建仅覆盖
        "串行多次 asyncio.run"；多个 event loop 并发共用同一 requester 不受支持
        （旧 loop 上的在途请求会在 session 被替换后失败）。
        """
        # 如果 session 无效（不存在、已关闭、或绑定到不同的 loop），创建新的
        if not self._is_session_valid():
            # 清理旧的 session（如果存在）
            if self._session is not None and not self._session.closed:
                try:
                    await self._session.close()
                except Exception:
                    pass
            if self._connector is not None and not self._connector.closed:
                try:
                    await self._connector.close()
                except Exception:
                    pass
            self._create_session()
        yield self._session

    async def aclose(self):
        """异步关闭 session 和 connector（推荐在异步上下文中使用）"""
        session = self._session
        connector = self._connector
        self._session = None
        self._connector = None

        if session and not session.closed:
            await session.close()
        if connector and not connector.closed:
            await connector.close()

    def close(self):
        """同步关闭 session 和 connector"""
        session = self._session
        connector = self._connector
        self._session = None
        self._connector = None

        if session and not session.closed:
            try:
                loop = asyncio.get_running_loop()
                # 在运行中的事件循环内，创建任务来关闭
                loop.create_task(self._async_close(session, connector))
            except RuntimeError:
                # 没有运行中的事件循环，创建新循环来关闭
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(self._async_close(session, connector))
                finally:
                    loop.close()

    def __del__(self):
        """析构时标记 session/connector 为已关闭，避免 'Unclosed client session' 警告。
        __del__ 中无法 await 异步 close()，标记 _closed 是 aiohttp 生态的标准做法，
        底层 TCP 连接会在进程退出时由 OS 回收。
        """
        if self._connector is not None and not self._connector.closed:
            self._connector._closed = True
        if self._session is not None and not self._session.closed:
            self._session._closed = True

    @staticmethod
    async def _async_close(session: ClientSession, connector: TCPConnector):
        """异步关闭 session 和 connector（内部使用）"""
        if session and not session.closed:
            await session.close()
        if connector and not connector.closed:
            await connector.close()

    @staticmethod
    async def _read_body(response) -> Any:
        """读取响应体，优先 JSON，失败时降级为原始文本"""
        try:
            return await response.json()
        except Exception:
            try:
                return {"raw": await response.text()}
            except Exception:
                return None

    @staticmethod
    async def _make_requests(session: ClientSession, method: str, url: str, **kwargs):
        async with session.request(method, url, **kwargs) as response:
            if response.status == 429 or response.status >= 500:
                # 限流/服务端错误：抛异常进入 async_retry 重试，携带响应体。
                # Retry-After 由服务端给出配额恢复时间，比本地指数退避准确。
                raise RetryableHTTPError(
                    response.status,
                    await ConcurrentRequester._read_body(response),
                    retry_after=parse_retry_after(response.headers.get("Retry-After")),
                )
            if response.status >= 400:
                # 其他 4xx（认证/参数/404 等）重试也不会成功：不重试，
                # 保留响应体交由调用方生成错误结果
                return response, await ConcurrentRequester._read_body(response)
            try:
                data = await response.json()
            except Exception as e:
                # 2xx 但响应体不是 JSON（如网关/代理返回 text/html）：
                # 确定性失败，不进入重试，保留响应体文本供上层报错
                try:
                    text = await response.text()
                except Exception:
                    text = None
                raise JSONDecodeHTTPError(response.status, text) from e
            return response, data

    async def make_requests(self, session: ClientSession, method: str, url: str, **kwargs):
        if self._proxy and not self._proxy_is_socks:
            # setdefault：per-request 显式传入的 proxy 优先于客户端级配置。
            # SOCKS 例外：隧道已由 connector 建立，再传 proxy= 会让 aiohttp
            # 对着 SOCKS 端口发 HTTP CONNECT。
            kwargs.setdefault("proxy", self._proxy)
        return await async_retry(
            self.retry_times, self.retry_delay, exceptions=RETRYABLE_EXCEPTIONS
        )(self._make_requests)(session, method, url, **kwargs)

    async def _send_single_request(
        self,
        session: ClientSession,
        request_id: int,
        url: str,
        method: str = "POST",
        meta: dict = None,
        **kwargs,
    ) -> RequestResult:
        """发送单个请求"""
        # 端到端计时起点：在 semaphore 外，包含排队等待
        t_enqueue = time.perf_counter()
        queue_time = 0.0
        pool_gate = self._pool_semaphore if self._pool_semaphore is not None else nullcontext()
        async with self._get_semaphore(), pool_gate:
            try:
                await self._rate_limiter.acquire()
                if self._pool_rate_limiter is not None:
                    await self._pool_rate_limiter.acquire()
                # 排队结束：semaphore 与漏桶（含 pool 级）均已放行，此后的耗时归因于服务。
                # 漏桶只在这里 acquire 一次（retry 循环在 make_requests 内部），
                # 所以 queue_time 是一次性的，重试不会再次穿过漏桶。
                queue_time = time.perf_counter() - t_enqueue
                response, data = await self.make_requests(session, method, url, **kwargs)
                latency = time.perf_counter() - t_enqueue

                # 与 _make_requests 的分类一致：>= 400 才是错误（201/204 等 2xx 是成功）
                if response.status >= 400:
                    error_info = {
                        "status_code": response.status,
                        "response_data": data,
                        "error": f"HTTP {response.status}",
                    }
                    return RequestResult(
                        request_id=request_id,
                        data=error_info,
                        status="error",
                        meta=meta,
                        latency=latency,
                        queue_time=queue_time,
                    )

                return RequestResult(
                    request_id=request_id,
                    data=data,
                    status="success",
                    meta=meta,
                    latency=latency,
                    queue_time=queue_time,
                )

            except RetryableHTTPError as e:
                # 重试耗尽的 429/5xx：结构与上面非 200 分支一致，保留响应体
                return RequestResult(
                    request_id=request_id,
                    data={
                        "status_code": e.status_code,
                        "response_data": e.response_data,
                        "error": f"HTTP {e.status_code}",
                    },
                    status="error",
                    meta=meta,
                    latency=time.perf_counter() - t_enqueue,
                    queue_time=queue_time,
                )
            except JSONDecodeHTTPError as e:
                # 2xx 但响应体不是 JSON：确定性错误，保留原始文本
                return RequestResult(
                    request_id=request_id,
                    data={
                        "status_code": e.status_code,
                        "response_data": {"raw": e.body_text},
                        "error": "Invalid JSON in response body",
                    },
                    status="error",
                    meta=meta,
                    latency=time.perf_counter() - t_enqueue,
                    queue_time=queue_time,
                )
            except asyncio.TimeoutError as e:
                return RequestResult(
                    request_id=request_id,
                    data={"error": "Timeout error", "detail": str(e)},
                    status="error",
                    meta=meta,
                    latency=time.perf_counter() - t_enqueue,
                    queue_time=queue_time,
                )
            except Exception as e:
                return RequestResult(
                    request_id=request_id,
                    data={"error": e.__class__.__name__, "detail": str(e)},
                    status="error",
                    meta=meta,
                    latency=time.perf_counter() - t_enqueue,
                    queue_time=queue_time,
                )

    async def process_with_concurrency_window(
        self,
        items: Iterable,
        process_func: Callable,
        concurrency_limit: int,
        progress: ProgressTracker | None = None,
        batch_size: int = 1,
    ) -> AsyncGenerator[StreamingResult, Any]:
        """
        使用滑动窗口方式处理并发任务，支持流式返回结果

        Args:
            items: 待处理的项目迭代器
            process_func: 处理单个项目的异步函数，接收item和项目item_id作为参数
            concurrency_limit: 并发限制数量,也是窗口大小
            progress: 可选的进度跟踪器
            batch_size: 每次yield返回的最小完成请求数量

        Yields:
             生成 StreamingResult 对象序列
        """
        completed_batch = []
        items_iter = iter(items)
        item_id = 0
        active_tasks: dict[asyncio.Task, int] = {}  # task -> item_id

        def create_task(item, idx):
            """创建并返回新任务"""
            task = asyncio.create_task(process_func(item, idx))
            active_tasks[task] = idx
            return task

        try:
            # 填满初始窗口
            for item in items_iter:
                create_task(item, item_id)
                item_id += 1
                if len(active_tasks) >= concurrency_limit:
                    break

            # 滑动窗口处理
            while active_tasks:
                # 等待任意一个任务完成
                done, _ = await asyncio.wait(
                    active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
                )

                # 处理所有完成的任务，并立即填补空位
                for task in done:
                    result = await task
                    del active_tasks[task]

                    if progress:
                        progress.update(result)
                    completed_batch.append(result)

                    # 立即创建新任务填补空位（真正的滑动窗口）
                    try:
                        next_item = next(items_iter)
                        create_task(next_item, item_id)
                        item_id += 1
                    except StopIteration:
                        pass  # 没有更多 items 了

                # 检查是否需要 yield 结果
                is_final = len(active_tasks) == 0
                if len(completed_batch) >= batch_size or (is_final and completed_batch):
                    if is_final and progress:
                        progress.summary()
                    yield StreamingResult(
                        completed_requests=sorted(completed_batch, key=lambda x: x.request_id),
                        progress=progress,
                        is_final=is_final,
                    )
                    completed_batch = []
        finally:
            # 消费端提前退出（break / 异常上抛 / 生成器被 close）时，
            # 取消窗口内在途的 HTTP 请求任务，避免后台继续发请求烧钱
            if active_tasks:
                for task in active_tasks:
                    task.cancel()
                await asyncio.gather(*active_tasks, return_exceptions=True)

    async def _stream_requests(
        self,
        queue: Queue,
        request_params: Iterable[dict[str, Any]],
        url: str,
        method: str = "POST",
        total_requests: int | None = None,
        show_progress: bool = True,
        batch_size: int | None = None,
        progress_config: ProgressBarConfig | None = None,
        model_name: str | None = None,
        input_price_per_1m: float | None = None,
        output_price_per_1m: float | None = None,
    ):
        """
        流式处理批量请求，实时返回已完成的结果

        Args:
            request_params: 请求参数列表
            url: 请求URL
            method: 请求方法
            total_requests: 总请求数量
            show_progress: 是否显示进度
            batch_size: 每次yield返回的最小完成请求数量
            progress_config: 进度条配置
            model_name: 模型名称（用于双行进度条显示）
            input_price_per_1m: 输入价格（$/1M tokens）
            output_price_per_1m: 输出价格（$/1M tokens）
        """
        try:
            progress = None
            if batch_size is None:
                batch_size = self._concurrency_limit
            if total_requests is None and show_progress:
                request_params, params_for_counting = itertools.tee(request_params)
                total_requests = sum(1 for _ in params_for_counting)

            if show_progress and total_requests is not None:
                config = progress_config or ProgressBarConfig()
                progress = ProgressTracker(
                    total_requests,
                    config=config,
                    model_name=model_name,
                    input_price_per_1m=input_price_per_1m,
                    output_price_per_1m=output_price_per_1m,
                )

            async with self._get_session() as session:
                async for result in self.process_with_concurrency_window(
                    items=request_params,
                    process_func=lambda params, request_id: self._send_single_request(
                        session=session,
                        request_id=request_id,
                        url=url,
                        method=method,
                        meta=params.pop("meta", None),
                        **params,
                    ),
                    concurrency_limit=self._concurrency_limit,
                    progress=progress,
                    batch_size=batch_size,
                ):
                    await queue.put(result)
        finally:
            # 结束哨兵必达：生产者异常时若不入队，消费者会在 queue.get() 上永久挂起，
            # 真实异常只能等 GC 时以 "Task exception was never retrieved" 出现。
            # 真实异常由消费侧 finally 中 await task 取回并传播。
            await queue.put(None)

    async def aiter_stream_requests(
        self,
        request_params: Iterable[dict[str, Any]],
        url: str,
        method: str = "POST",
        total_requests: int | None = None,
        show_progress: bool = True,
        batch_size: int | None = None,
        progress_config: ProgressBarConfig | None = None,
        model_name: str | None = None,
        input_price_per_1m: float | None = None,
        output_price_per_1m: float | None = None,
    ):
        queue = Queue()
        task = asyncio.create_task(
            self._stream_requests(
                queue,
                request_params=request_params,
                url=url,
                method=method,
                total_requests=total_requests,
                show_progress=show_progress,
                batch_size=batch_size,
                progress_config=progress_config,
                model_name=model_name,
                input_price_per_1m=input_price_per_1m,
                output_price_per_1m=output_price_per_1m,
            )
        )
        try:
            while True:
                result = await queue.get()
                if result is None:
                    break
                yield result
        finally:
            # 无论正常结束还是消费侧提前退出，都要收尾生产者任务：
            # - 提前退出：cancel 后 await，触发窗口内在途请求的取消
            # - 生产者自身异常：await 取回并向上传播，不能静默吞掉
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass  # 我们主动取消的，不向上传播
            else:
                # 已结束：正常返回或重新抛出生产者的真实异常
                await task

    async def process_requests(
        self,
        request_params: Iterable[dict[str, Any]],
        url: str,
        method: str = "POST",
        total_requests: int | None = None,
        show_progress: bool = True,
        progress_config: ProgressBarConfig | None = None,
    ) -> tuple[list[RequestResult], ProgressTracker | None]:
        """
        处理批量请求

        Returns:
            Tuple[list[RequestResult], Optional[ProgressTracker]]:
            请求结果列表和进度跟踪器（如果启用了进度显示）
        """
        progress = None
        if total_requests is None and show_progress:
            request_params, params_for_counting = itertools.tee(request_params)
            total_requests = sum(1 for _ in params_for_counting)

        if show_progress and total_requests is not None:
            config = progress_config or ProgressBarConfig()
            progress = ProgressTracker(total_requests, config=config)

        results = []
        async with self._get_session() as session:
            async for result in self.process_with_concurrency_window(
                items=request_params,
                process_func=lambda params, request_id: self._send_single_request(
                    session=session,
                    request_id=request_id,
                    url=url,
                    method=method,
                    meta=params.pop("meta", None),
                    **params,
                ),
                concurrency_limit=self._concurrency_limit,
                progress=progress,
            ):
                results.extend(result.completed_requests)
        # sort
        results = sorted(results, key=lambda x: x.request_id)
        return results, progress
