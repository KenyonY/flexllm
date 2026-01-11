import asyncio
from asyncio import Queue
import itertools

import time
from typing import Any, Dict, Iterable, List, Optional, Callable, AsyncIterator, AsyncGenerator, Tuple, Union
from aiohttp import ClientSession, TCPConnector, ClientTimeout
from contextlib import asynccontextmanager

# 临时解决方案：直接导入以避免命名冲突
try:
    from ..utils.core import async_retry
except ImportError:
    # 如果有命名冲突，可能需要更具体的导入
    def async_retry(retry_times=3, retry_delay=1.0):
        def decorator(func):
            return func
        return decorator
from .interface import RequestResult
from .progress import ProgressTracker, ProgressBarConfig

from dataclasses import dataclass


@dataclass
class StreamingResult:
    completed_requests: List[RequestResult]
    progress: Optional[ProgressTracker]
    is_final: bool


class RateLimiter:
    """
    速率限制器

    Args:
        max_qps: 每秒最大请求数
        use_bucket: 是否使用漏桶算法（aiolimiter），默认 True
                    False 时使用简单的锁+sleep 实现
    """

    def __init__(self, max_qps: Optional[float] = None, use_bucket: bool = True):
        self.max_qps = max_qps
        self._use_bucket = use_bucket

        if max_qps:
            if use_bucket:
                from aiolimiter import AsyncLimiter
                self._limiter = AsyncLimiter(max_qps, 1)
            else:
                self._lock = asyncio.Lock()
                self._last_request_time = 0
                self._min_interval = 1 / max_qps

    async def acquire(self):
        if not self.max_qps:
            return
        if self._use_bucket:
            await self._limiter.acquire()
        else:
            async with self._lock:
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
            max_qps: Optional[float] = None,
            timeout: Optional[float] = None,
            retry_times: int = 3,
            retry_delay: float = 0.3
    ):
        self._concurrency_limit = concurrency_limit
        if timeout:
            self._timeout = ClientTimeout(total=timeout, connect=min(10., timeout))
        else:
            self._timeout = None
        self._rate_limiter = RateLimiter(max_qps)
        self._semaphore = asyncio.Semaphore(concurrency_limit)
        self.retry_times = retry_times
        self.retry_delay = retry_delay

    @asynccontextmanager
    async def _get_session(self):
        connector = TCPConnector(limit=self._concurrency_limit+10, limit_per_host=0, force_close=False)
        async with ClientSession(timeout=self._timeout, connector=connector, trust_env=True) as session:
            yield session

    @staticmethod
    async def _make_requests( session: ClientSession,method: str, url: str,  **kwargs):
        async with session.request(method, url, **kwargs) as response:
            response.raise_for_status()
            data = await response.json()
            return response, data

    async def make_requests(self, session: ClientSession,method: str, url: str,  **kwargs):
        return await async_retry(self.retry_times, self.retry_delay)(self._make_requests)(session,method, url, **kwargs)

    async def _send_single_request(
            self,
            session: ClientSession,
            request_id: int,
            url: str,
            method: str = 'POST',
            meta: dict = None,
            **kwargs
    ) -> RequestResult:
        """发送单个请求"""
        async with self._semaphore:
            try:
                # todo: 速率限制也许需要优化
                await self._rate_limiter.acquire()

                start_time = time.time()
                response, data = await self.make_requests(session, method, url, **kwargs)
                latency = time.time() - start_time

                if response.status != 200:
                    error_info = {
                        'status_code': response.status,
                        'response_data': data,
                        'error': f"HTTP {response.status}"
                    }
                    return RequestResult(
                        request_id=request_id,
                        data=error_info,
                        status='error',
                        meta=meta,
                        latency=latency
                    )

                return RequestResult(
                    request_id=request_id,
                    data=data,
                    status="success",
                    meta=meta,
                    latency=latency
                )

            except asyncio.TimeoutError as e:
                return RequestResult(
                    request_id=request_id,
                    data={'error': 'Timeout error', 'detail': str(e)},
                    status='error',
                    meta=meta,
                    latency=time.time() - start_time
                )
            except Exception as e:
                return RequestResult(
                    request_id=request_id,
                    data={'error': e.__class__.__name__, 'detail': str(e)},
                    status='error',
                    meta=meta,
                    latency=time.time() - start_time
                )

    async def process_with_concurrency_window(
            self,
            items: Iterable,
            process_func: Callable,
            concurrency_limit: int,
            progress: Optional[ProgressTracker] = None,
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

        async def handle_completed_tasks(done_tasks, batch, is_final=False):
            """内部函数处理已完成的任务"""
            for task in done_tasks:
                result = await task
                if progress:
                    progress.update(result)
                batch.append(result)

            if len(batch) >= batch_size or (is_final and batch):
                if is_final and progress:
                    progress.summary()
                yield StreamingResult(
                    completed_requests=sorted(batch, key=lambda x: x.request_id),
                    progress=progress,
                    is_final=is_final
                )
                batch.clear()

        item_id = 0
        active_tasks = set()
        completed_batch = []

        # 处理输入项目
        for item in items:
            if len(active_tasks) >= concurrency_limit:
                done, active_tasks = await asyncio.wait(
                    active_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )
                async for result in handle_completed_tasks(done, completed_batch):
                    yield result

            active_tasks.add(asyncio.create_task(process_func(item, item_id)))
            item_id += 1

        # 处理剩余任务
        if active_tasks:
            done, _ = await asyncio.wait(active_tasks)
            async for result in handle_completed_tasks(done, completed_batch, is_final=True):
                yield result

    async def _stream_requests(
            self,
            queue: Queue,
            request_params: Iterable[Dict[str, Any]],
            url: str,
            method: str = 'POST',
            total_requests: Optional[int] = None,
            show_progress: bool = True,
            batch_size: Optional[int] = None
    ) :
        """
        流式处理批量请求，实时返回已完成的结果

        Args:
            request_params: 请求参数列表
            url: 请求URL
            method: 请求方法
            total_requests: 总请求数量
            show_progress: 是否显示进度
            batch_size: 每次yield返回的最小完成请求数量
        """
        progress = None
        if batch_size is None:
            batch_size = self._concurrency_limit
        if total_requests is None and show_progress:
            request_params, params_for_counting = itertools.tee(request_params)
            total_requests = sum(1 for _ in params_for_counting)

        if show_progress and total_requests is not None:
            progress = ProgressTracker(
                total_requests,
                concurrency=self._concurrency_limit,
                config=ProgressBarConfig()
            )

        async with self._get_session() as session:
            async for result in self.process_with_concurrency_window(
                    items=request_params,
                    process_func=lambda params, request_id: self._send_single_request(
                        session=session,
                        request_id=request_id,
                        url=url,
                        method=method,
                        meta=params.pop('meta', None),
                        **params
                    ),
                    concurrency_limit=self._concurrency_limit,
                    progress=progress,
                    batch_size=batch_size,
            ):
                await queue.put(result)

        await queue.put(None)

    async def aiter_stream_requests(self,
                                  request_params: Iterable[Dict[str, Any]],
                                  url: str,
                                  method: str = 'POST',
                                  total_requests: Optional[int] = None,
                                  show_progress: bool = True,
                                  batch_size: Optional[int] = None
                                  ):
        queue = Queue()
        task = asyncio.create_task(self._stream_requests(queue,
                                                         request_params=request_params,
                                                         url=url,
                                                         method=method,
                                                         total_requests=total_requests,
                                                         show_progress=show_progress,
                                                         batch_size=batch_size))
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
            request_params: Iterable[Dict[str, Any]],
            url: str,
            method: str = 'POST',
            total_requests: Optional[int] = None,
            show_progress: bool = True
    ) -> Tuple[List[RequestResult], Optional[ProgressTracker]]:
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
            progress = ProgressTracker(
                total_requests,
                concurrency=self._concurrency_limit,
                config=ProgressBarConfig()
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
                    meta=params.pop('meta', None),
                    **params
                ),
                concurrency_limit=self._concurrency_limit,
                progress=progress,
            ):
                results.extend(result.completed_requests)
        # sort
        results = sorted(results, key=lambda x: x.request_id)
        return results, progress
