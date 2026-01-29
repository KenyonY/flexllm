import asyncio
import itertools
import time
from asyncio import Queue
from collections.abc import AsyncGenerator, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import (
    Any,
)

from aiohttp import ClientSession, ClientTimeout, TCPConnector

from ..utils.core import async_retry
from .interface import RequestResult
from .progress import ProgressBarConfig, ProgressTracker


@dataclass
class StreamingResult:
    completed_requests: list[RequestResult]
    progress: ProgressTracker | None
    is_final: bool


class RateLimiter:
    """
    速率限制器

    Args:
        max_qps: 每秒最大请求数
        use_bucket: 是否使用漏桶算法（aiolimiter），默认 True
                    False 时使用简单的锁+sleep 实现
    """

    def __init__(self, max_qps: float | None = None, use_bucket: bool = True):
        self.max_qps = max_qps
        self._use_bucket = use_bucket

        # lazy init，避免绑定错误的 event loop（多次 asyncio.run 场景）
        self._limiter = None
        self._lock: asyncio.Lock | None = None
        self._last_request_time = 0
        if max_qps and not use_bucket:
            self._min_interval = 1 / max_qps

    def _get_limiter(self):
        """获取或创建 limiter（确保绑定到当前 event loop）

        注：_loop 是 asyncio 对象的内部属性，可能在 Python 版本间变化，
        使用 getattr 安全获取。
        """
        if not self.max_qps or not self._use_bucket:
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

                self._limiter = AsyncLimiter(self.max_qps, 1)
        except RuntimeError:
            # 没有运行的 event loop，不应该发生（acquire 在 async 中调用）
            pass
        return self._limiter

    def _get_lock(self) -> asyncio.Lock:
        """获取或创建 Lock（确保绑定到当前 event loop）"""
        try:
            loop = asyncio.get_running_loop()
            if self._lock is not None:
                lock_loop = getattr(self._lock, "_loop", None)
                if lock_loop is not None and lock_loop is not loop:
                    self._lock = None
            if self._lock is None:
                self._lock = asyncio.Lock()
        except RuntimeError:
            # 没有运行的 event loop，不应该发生
            if self._lock is None:
                self._lock = asyncio.Lock()
        return self._lock

    async def acquire(self):
        if not self.max_qps:
            return
        if self._use_bucket:
            await self._get_limiter().acquire()
        else:
            async with self._get_lock():
                elapsed = time.time() - self._last_request_time
                if elapsed < self._min_interval:
                    await asyncio.sleep(self._min_interval - elapsed)
                self._last_request_time = time.time()


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
    ):
        self._concurrency_limit = concurrency_limit
        if timeout:
            self._timeout = ClientTimeout(total=timeout, connect=min(10.0, timeout))
        else:
            self._timeout = None
        self._rate_limiter = RateLimiter(max_qps)
        self._semaphore: asyncio.Semaphore | None = None  # lazy init，避免绑定错误的 event loop
        self.retry_times = retry_times
        self.retry_delay = retry_delay

        # Session 复用：避免每次请求都创建新的 session
        self._connector: TCPConnector | None = None
        self._session: ClientSession | None = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        """获取或创建 Semaphore（确保绑定到当前 event loop）"""
        try:
            loop = asyncio.get_running_loop()
            # 检查 semaphore 是否绑定到当前 loop
            if self._semaphore is not None:
                # Python 3.10+ Semaphore 有 _loop 属性（内部）
                sem_loop = getattr(self._semaphore, "_loop", None)
                if sem_loop is not None and sem_loop is not loop:
                    # 绑定到不同的 loop，需要重新创建
                    self._semaphore = None
            if self._semaphore is None:
                self._semaphore = asyncio.Semaphore(self._concurrency_limit)
        except RuntimeError:
            # 没有运行的 event loop，创建新的 semaphore
            if self._semaphore is None:
                self._semaphore = asyncio.Semaphore(self._concurrency_limit)
        return self._semaphore

    def _create_session(self) -> ClientSession:
        """创建新的 session（内部使用）"""
        self._connector = TCPConnector(
            limit=self._concurrency_limit + 10, limit_per_host=0, force_close=False
        )
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
        """获取或创建 session（复用模式，确保绑定到当前 event loop）"""
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

    @staticmethod
    async def _async_close(session: ClientSession, connector: TCPConnector):
        """异步关闭 session 和 connector（内部使用）"""
        if session and not session.closed:
            await session.close()
        if connector and not connector.closed:
            await connector.close()

    @staticmethod
    async def _make_requests(session: ClientSession, method: str, url: str, **kwargs):
        async with session.request(method, url, **kwargs) as response:
            response.raise_for_status()
            data = await response.json()
            return response, data

    async def make_requests(self, session: ClientSession, method: str, url: str, **kwargs):
        return await async_retry(self.retry_times, self.retry_delay)(self._make_requests)(
            session, method, url, **kwargs
        )

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
        start_time = time.time()  # 在 semaphore 外计时，包含等待时间
        async with self._get_semaphore():
            try:
                await self._rate_limiter.acquire()
                response, data = await self.make_requests(session, method, url, **kwargs)
                latency = time.time() - start_time

                if response.status != 200:
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
                    )

                return RequestResult(
                    request_id=request_id, data=data, status="success", meta=meta, latency=latency
                )

            except asyncio.TimeoutError as e:
                return RequestResult(
                    request_id=request_id,
                    data={"error": "Timeout error", "detail": str(e)},
                    status="error",
                    meta=meta,
                    latency=time.time() - start_time,
                )
            except Exception as e:
                return RequestResult(
                    request_id=request_id,
                    data={"error": e.__class__.__name__, "detail": str(e)},
                    status="error",
                    meta=meta,
                    latency=time.time() - start_time,
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

        # 填满初始窗口
        for item in items_iter:
            create_task(item, item_id)
            item_id += 1
            if len(active_tasks) >= concurrency_limit:
                break

        # 滑动窗口处理
        while active_tasks:
            # 等待任意一个任务完成
            done, _ = await asyncio.wait(active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)

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
                concurrency=self._concurrency_limit,
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
            if not task.done():
                task.cancel()

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
            progress = ProgressTracker(
                total_requests, concurrency=self._concurrency_limit, config=config
            )

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
