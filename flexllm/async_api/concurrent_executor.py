import asyncio
import time
import itertools
import inspect
import os
import uuid
from asyncio import Queue
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Callable,
    AsyncIterator,
    AsyncGenerator,
    Tuple,
    Union,
    Awaitable,
    Protocol,
)
from dataclasses import dataclass, field
import heapq

from ..utils.core import async_retry
from .interface import RequestResult
from .progress import ProgressTracker, ProgressBarConfig

# FlaxKV2 用于检查点存储
try:
    from flaxkv2 import FlaxKV
    FLAXKV_AVAILABLE = True
except ImportError:
    FLAXKV_AVAILABLE = False


# 添加函数协议定义
class TaskFunction(Protocol):
    """任务函数协议"""

    async def __call__(self, *args, **kwargs) -> Any: ...


@dataclass
class TaskContext:
    """任务执行上下文，包含所有可能需要的信息"""

    task_id: int
    data: Any
    meta: Optional[dict] = None
    retry_count: int = 0
    executor_kwargs: Optional[dict] = None


@dataclass
class TaskItem:
    """任务项，支持优先级"""

    priority: int
    task_id: int
    data: Any
    meta: Optional[dict] = field(default_factory=dict)

    def __lt__(self, other):
        return self.priority < other.priority


@dataclass
class ExecutionResult:
    """执行结果"""

    task_id: int
    data: Any
    status: str  # 'success' or 'error'
    meta: Optional[dict] = None
    latency: float = 0.0
    error: Optional[Exception] = None
    retry_count: int = 0  # 重试次数


@dataclass
class StreamingExecutionResult:
    """流式执行结果"""

    completed_tasks: List[ExecutionResult]
    progress: Optional[ProgressTracker]
    is_final: bool


class RateLimiter:
    """速率限制器"""

    def __init__(self, max_qps: Optional[float] = None):
        self.max_qps = max_qps
        self.min_interval = 1 / max_qps if max_qps else 0
        self.last_request_time = 0

    async def acquire(self):
        if not self.max_qps:
            return

        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()


class ConcurrentExecutor:
    """
    通用并发执行器

    可以对任意的异步函数进行并发调度，支持：
    - 并发数量控制
    - QPS限制
    - 进度跟踪
    - 重试机制
    - 流式结果返回
    - 优先级调度
    - 自定义错误处理
    - 智能函数调用模式

    支持的执行方法：
    - execute_batch: 智能批量执行，自动检测函数签名
    - execute_batch_with_adapter: 使用适配器的批量执行
    - execute_batch_with_context: 上下文模式的批量执行
    - execute_batch_with_factory: 函数工厂模式的批量执行
    - execute_priority_batch: 按优先级的批量执行
    - aiter_execute_batch: 流式批量执行
    - execute_batch_sync: 同步版本的批量执行

    Example
    -------

    # 方式1: 简单函数 - 仅接收数据
    async def simple_task(data):
        await asyncio.sleep(0.1)
        return f"processed: {data}"

    # 方式2: 上下文函数 - 接收完整上下文
    async def context_task(context: TaskContext):
        return f"task_{context.task_id}: {context.data} (retry: {context.retry_count})"

    # 方式3: 通用函数 - 接受任意参数
    async def flexible_task(item, **options):
        return f"processed {item} with {options}"

    # 创建执行器
    executor = ConcurrentExecutor(
        concurrency_limit=5,
        max_qps=10,
        retry_times=3
    )

    # 智能批量执行 - 自动检测函数签名适配调用方式
    results, _ = await executor.execute_batch(
        async_func=simple_task,         # 简单函数
        tasks_data=["task1", "task2", "task3"]
    )

    results, _ = await executor.execute_batch(
        async_func=flexible_task,       # 通用函数
        tasks_data=["item1", "item2"],
        executor_kwargs={"user_id": 123, "mode": "fast"}
    )

    # 使用任务适配器（适配复杂函数签名）
    results, _ = await executor.execute_batch_with_adapter(
        async_func=complex_function,
        tasks_data=complex_data,
        task_adapter=my_adapter_function
    )

    # 使用上下文模式（获取完整执行信息）
    results, _ = await executor.execute_batch_with_context(
        async_func=context_task,
        tasks_data=["data1", "data2"]
    )
    """

    def __init__(
        self,
        concurrency_limit: int,
        max_qps: Optional[float] = None,
        retry_times: int = 3,
        retry_delay: float = 0.3,
        error_handler: Optional[Callable[[Exception, Any, int], bool]] = None,
    ):
        self._concurrency_limit = concurrency_limit
        self._rate_limiter = RateLimiter(max_qps)
        self._semaphore = asyncio.Semaphore(concurrency_limit)
        self.retry_times = retry_times
        self.retry_delay = retry_delay
        self.error_handler = error_handler  # 自定义错误处理函数

    def _inspect_function_signature(self, func: Callable) -> dict:
        """检查函数签名，返回参数信息"""
        sig = inspect.signature(func)
        params = sig.parameters

        # 更强大的TaskContext检测
        has_context_param = False
        for param in params.values():
            # 检查类型注解
            if param.annotation == TaskContext:
                has_context_param = True
                break
            # 检查字符串形式的注解
            if isinstance(param.annotation, str) and "TaskContext" in param.annotation:
                has_context_param = True
                break
            # 检查参数名称（作为备选检测方式）
            if param.name == "context" and len(params) == 1:
                has_context_param = True
                break

        return {
            "param_names": list(params.keys()),
            "has_context_param": has_context_param,
            "accepts_var_kwargs": any(p.kind == p.VAR_KEYWORD for p in params.values()),
            "param_count": len(
                [
                    p
                    for p in params.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                ]
            ),
        }

    async def _call_function_intelligently(
        self,
        func: Callable,
        task_context: TaskContext,
        executor_kwargs: Optional[dict] = None,
    ) -> Any:
        """智能调用函数，根据函数签名自动适配参数"""
        sig_info = self._inspect_function_signature(func)
        executor_kwargs = executor_kwargs or {}

        # 方式1: 如果函数接受TaskContext类型参数
        if sig_info["has_context_param"]:
            # 上下文函数不接受额外的关键字参数，所有信息都在TaskContext中
            return await func(task_context)

        # 方式2: 只接受data的简单函数
        elif sig_info["param_count"] == 1 and not sig_info["accepts_var_kwargs"]:
            return await func(task_context.data)

        # 方式3: 接受任意参数的通用函数
        else:
            return await func(task_context.data, **executor_kwargs)

    async def _execute_single_task(
        self,
        async_func: Callable[..., Awaitable[Any]],
        task_data: Any,
        task_id: int,
        meta: Optional[dict] = None,
        executor_kwargs: Optional[dict] = None,
        task_adapter: Optional[Callable] = None,
        **kwargs,
    ) -> ExecutionResult:
        """执行单个异步任务"""
        retry_count = 0
        last_error = None
        executor_kwargs = {**(executor_kwargs or {}), **kwargs}

        async with self._semaphore:
            while retry_count <= self.retry_times:
                try:
                    await self._rate_limiter.acquire()

                    start_time = time.time()

                    # 创建任务上下文
                    task_context = TaskContext(
                        task_id=task_id,
                        data=task_data,
                        meta=meta,
                        retry_count=retry_count,
                        executor_kwargs=executor_kwargs,
                    )

                    # 使用适配器或智能调用
                    if task_adapter:
                        args, kwargs_from_adapter = task_adapter(
                            task_data, task_context
                        )
                        if isinstance(args, (list, tuple)):
                            result = await async_func(
                                *args, **{**executor_kwargs, **kwargs_from_adapter}
                            )
                        else:
                            result = await async_func(
                                args, **{**executor_kwargs, **kwargs_from_adapter}
                            )
                    else:
                        result = await self._call_function_intelligently(
                            async_func, task_context, executor_kwargs
                        )

                    latency = time.time() - start_time

                    return ExecutionResult(
                        task_id=task_id,
                        data=result,
                        status="success",
                        meta=meta,
                        latency=latency,
                        retry_count=retry_count,
                    )

                except Exception as e:
                    last_error = e
                    retry_count += 1

                    # 调用自定义错误处理函数
                    if self.error_handler:
                        should_retry = self.error_handler(e, task_data, retry_count)
                        if not should_retry:
                            break

                    if retry_count <= self.retry_times:
                        await asyncio.sleep(self.retry_delay)

            # 所有重试都失败了
            return ExecutionResult(
                task_id=task_id,
                data=None,
                status="error",
                meta=meta,
                latency=time.time() - start_time if "start_time" in locals() else 0,
                error=last_error,
                retry_count=retry_count - 1,
            )

    async def _process_with_concurrency_window(
        self,
        async_func: Callable[..., Awaitable[Any]],
        tasks_data: Iterable[Any],
        progress: Optional[ProgressTracker] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> AsyncGenerator[StreamingExecutionResult, Any]:
        """
        使用滑动窗口方式处理并发任务，支持流式返回结果
        """

        async def handle_completed_tasks(done_tasks, batch, is_final=False):
            """处理已完成的任务"""
            for task in done_tasks:
                result = await task
                if progress:
                    # 将ExecutionResult转换为RequestResult以兼容ProgressTracker
                    # 对于错误情况，需要构造包含错误信息的data字典
                    progress_data = result.data
                    if result.status == "error" and result.error:
                        progress_data = {
                            "error": result.error.__class__.__name__,
                            "detail": str(result.error),
                        }

                    request_result = RequestResult(
                        request_id=result.task_id,
                        data=progress_data,
                        status=result.status,
                        meta=result.meta,
                        latency=result.latency,
                    )
                    progress.update(request_result)
                batch.append(result)

            if len(batch) >= batch_size or (is_final and batch):
                if is_final and progress:
                    progress.summary()
                yield StreamingExecutionResult(
                    completed_tasks=sorted(batch, key=lambda x: x.task_id),
                    progress=progress,
                    is_final=is_final,
                )
                batch.clear()

        task_id = 0
        active_tasks = set()
        completed_batch = []

        # 处理任务数据
        for data in tasks_data:
            if len(active_tasks) >= self._concurrency_limit:
                done, active_tasks = await asyncio.wait(
                    active_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                async for result in handle_completed_tasks(done, completed_batch):
                    yield result

            active_tasks.add(
                asyncio.create_task(
                    self._execute_single_task(async_func, data, task_id, **kwargs)
                )
            )
            task_id += 1

        # 处理剩余任务
        if active_tasks:
            done, _ = await asyncio.wait(active_tasks)
            async for result in handle_completed_tasks(
                done, completed_batch, is_final=True
            ):
                yield result

    async def execute_batch(
        self,
        async_func: Callable[..., Awaitable[Any]],
        tasks_data: Iterable[Any],
        total_tasks: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> Tuple[List[ExecutionResult], Optional[ProgressTracker]]:
        """
        批量执行异步任务

        Args:
            async_func: 要执行的异步函数，函数签名应为 async def func(data, meta=None, **kwargs)
            tasks_data: 任务数据列表
            total_tasks: 总任务数量，如果不提供会自动计算
            show_progress: 是否显示进度
            **kwargs: 传递给异步函数的额外参数

        Returns:
            (结果列表, 进度跟踪器)
        """
        progress = None

        if total_tasks is None and show_progress:
            tasks_data, data_for_counting = itertools.tee(tasks_data)
            total_tasks = sum(1 for _ in data_for_counting)

        if show_progress and total_tasks is not None:
            progress = ProgressTracker(
                total_tasks,
                concurrency=self._concurrency_limit,
                config=ProgressBarConfig(),
            )

        results = []
        async for result in self._process_with_concurrency_window(
            async_func=async_func, tasks_data=tasks_data, progress=progress, **kwargs
        ):
            results.extend(result.completed_tasks)

        # 按任务ID排序
        results = sorted(results, key=lambda x: x.task_id)
        return results, progress

    async def _stream_execute(
        self,
        queue: Queue,
        async_func: Callable[..., Awaitable[Any]],
        tasks_data: Iterable[Any],
        total_tasks: Optional[int] = None,
        show_progress: bool = True,
        batch_size: Optional[int] = None,
        **kwargs,
    ):
        """流式执行任务并将结果放入队列"""
        progress = None
        if batch_size is None:
            batch_size = self._concurrency_limit

        if total_tasks is None and show_progress:
            tasks_data, data_for_counting = itertools.tee(tasks_data)
            total_tasks = sum(1 for _ in data_for_counting)

        if show_progress and total_tasks is not None:
            progress = ProgressTracker(
                total_tasks,
                concurrency=self._concurrency_limit,
                config=ProgressBarConfig(),
            )

        async for result in self._process_with_concurrency_window(
            async_func=async_func,
            tasks_data=tasks_data,
            progress=progress,
            batch_size=batch_size,
            **kwargs,
        ):
            await queue.put(result)

        await queue.put(None)

    async def aiter_execute_batch(
        self,
        async_func: Callable[..., Awaitable[Any]],
        tasks_data: Iterable[Any],
        total_tasks: Optional[int] = None,
        show_progress: bool = True,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[StreamingExecutionResult]:
        """
        流式批量执行异步任务

        Args:
            async_func: 要执行的异步函数
            tasks_data: 任务数据列表
            total_tasks: 总任务数量
            show_progress: 是否显示进度
            batch_size: 每次返回的批次大小
            **kwargs: 传递给异步函数的额外参数

        Yields:
            StreamingExecutionResult: 包含已完成任务的结果
        """
        queue = Queue()
        task = asyncio.create_task(
            self._stream_execute(
                queue=queue,
                async_func=async_func,
                tasks_data=tasks_data,
                total_tasks=total_tasks,
                show_progress=show_progress,
                batch_size=batch_size,
                **kwargs,
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

    def execute_batch_sync(
        self,
        async_func: Callable[..., Awaitable[Any]],
        tasks_data: Iterable[Any],
        total_tasks: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> Tuple[List[ExecutionResult], Optional[ProgressTracker]]:
        """同步版本的批量执行"""
        try:
            # 检查是否已经在事件循环中
            loop = asyncio.get_running_loop()
            # 如果已经在事件循环中，使用新的线程执行
            import concurrent.futures
            import threading

            def run_in_thread():
                return asyncio.run(
                    self.execute_batch(
                        async_func=async_func,
                        tasks_data=tasks_data,
                        total_tasks=total_tasks,
                        show_progress=show_progress,
                        **kwargs,
                    )
                )

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()

        except RuntimeError:
            # 没有运行的事件循环，可以直接使用 asyncio.run
            return asyncio.run(
                self.execute_batch(
                    async_func=async_func,
                    tasks_data=tasks_data,
                    total_tasks=total_tasks,
                    show_progress=show_progress,
                    **kwargs,
                )
            )

    async def execute_priority_batch(
        self,
        async_func: Callable[..., Awaitable[Any]],
        priority_tasks: List[TaskItem],
        show_progress: bool = True,
        **kwargs,
    ) -> Tuple[List[ExecutionResult], Optional[ProgressTracker]]:
        """
        按优先级批量执行任务

        Args:
            async_func: 要执行的异步函数
            priority_tasks: 带优先级的任务列表 (优先级数字越小越优先)
            show_progress: 是否显示进度
            **kwargs: 传递给异步函数的额外参数

        Returns:
            (结果列表, 进度跟踪器)
        """
        # 创建优先级队列
        task_queue = []
        for task in priority_tasks:
            heapq.heappush(task_queue, task)

        progress = None
        if show_progress:
            progress = ProgressTracker(
                len(priority_tasks),
                concurrency=self._concurrency_limit,
                config=ProgressBarConfig(),
            )

        results = []
        active_tasks = set()

        while task_queue or active_tasks:
            # 启动新任务直到达到并发限制
            while len(active_tasks) < self._concurrency_limit and task_queue:
                task_item = heapq.heappop(task_queue)
                coroutine = self._execute_single_task(
                    async_func=async_func,
                    task_data=task_item.data,
                    task_id=task_item.task_id,
                    meta=task_item.meta,
                    **kwargs,
                )
                active_tasks.add(asyncio.create_task(coroutine))

            # 等待至少一个任务完成
            if active_tasks:
                done, active_tasks = await asyncio.wait(
                    active_tasks, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    result = await task
                    results.append(result)

                    if progress:
                        # 转换为RequestResult以兼容ProgressTracker
                        request_result = RequestResult(
                            request_id=result.task_id,
                            data=result.data,
                            status=result.status,
                            meta=result.meta,
                            latency=result.latency,
                        )
                        progress.update(request_result)

        if progress:
            progress.summary()

        # 按任务ID排序
        results = sorted(results, key=lambda x: x.task_id)
        return results, progress

    def add_custom_error_handler(self, handler: Callable[[Exception, Any, int], bool]):
        """
        添加自定义错误处理函数

        Args:
            handler: 错误处理函数，签名为 (error, task_data, retry_count) -> should_retry
        """
        self.error_handler = handler

    # === 新增的更灵活的执行方法 ===

    async def execute_batch_with_adapter(
        self,
        async_func: Callable[..., Awaitable[Any]],
        tasks_data: Iterable[Any],
        task_adapter: Callable[[Any, TaskContext], Tuple[Any, Dict]],
        executor_kwargs: Optional[dict] = None,
        total_tasks: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> Tuple[List[ExecutionResult], Optional[ProgressTracker]]:
        """
        使用任务适配器的批量执行

        Args:
            async_func: 要执行的异步函数
            tasks_data: 任务数据列表
            task_adapter: 任务适配器函数，签名为 (data, context) -> (args, kwargs)
                         返回值应为 (位置参数, 关键字参数) 的元组
            executor_kwargs: 传递给所有任务的公共参数
            total_tasks: 总任务数量
            show_progress: 是否显示进度
            **kwargs: 其他参数

        Returns:
            (结果列表, 进度跟踪器)

        Example:
            # 定义适配器函数
            def my_adapter(data, context):
                # 返回位置参数和关键字参数
                return (data['item'],), {'user_id': data['user_id'], 'batch_id': context.task_id}

            # 执行
            results, _ = await executor.execute_batch_with_adapter(
                async_func=my_custom_function,
                tasks_data=[{'item': 'a', 'user_id': 1}, {'item': 'b', 'user_id': 2}],
                task_adapter=my_adapter,
                executor_kwargs={'mode': 'fast'}
            )
        """
        progress = None

        if total_tasks is None and show_progress:
            tasks_data, data_for_counting = itertools.tee(tasks_data)
            total_tasks = sum(1 for _ in data_for_counting)

        if show_progress and total_tasks is not None:
            progress = ProgressTracker(
                total_tasks,
                concurrency=self._concurrency_limit,
                config=ProgressBarConfig(),
            )

        results = []
        async for result in self._process_with_concurrency_window(
            async_func=async_func,
            tasks_data=tasks_data,
            progress=progress,
            executor_kwargs=executor_kwargs,
            task_adapter=task_adapter,
            **kwargs,
        ):
            results.extend(result.completed_tasks)

        # 按任务ID排序
        results = sorted(results, key=lambda x: x.task_id)
        return results, progress

    async def execute_batch_with_context(
        self,
        async_func: Callable[[TaskContext], Awaitable[Any]],
        tasks_data: Iterable[Any],
        executor_kwargs: Optional[dict] = None,
        total_tasks: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> Tuple[List[ExecutionResult], Optional[ProgressTracker]]:
        """
        使用上下文模式的批量执行，函数直接接收TaskContext对象

        Args:
            async_func: 要执行的异步函数，签名应为 async def func(context: TaskContext) -> Any
            tasks_data: 任务数据列表
            executor_kwargs: 传递给TaskContext的额外参数
            total_tasks: 总任务数量
            show_progress: 是否显示进度
            **kwargs: 其他参数

        Returns:
            (结果列表, 进度跟踪器)

        Example:
            async def context_task(context: TaskContext):
                print(f"处理任务 {context.task_id}: {context.data}")
                print(f"重试次数: {context.retry_count}")
                print(f"额外参数: {context.executor_kwargs}")
                return f"结果: {context.data}"

            results, _ = await executor.execute_batch_with_context(
                async_func=context_task,
                tasks_data=["data1", "data2", "data3"],
                executor_kwargs={'user_id': 123}
            )
        """
        return await self.execute_batch(
            async_func=async_func,
            tasks_data=tasks_data,
            total_tasks=total_tasks,
            show_progress=show_progress,
            executor_kwargs=executor_kwargs,
            **kwargs,
        )

    async def execute_batch_with_factory(
        self,
        func_factory: Callable[[TaskContext], Callable[..., Awaitable[Any]]],
        tasks_data: Iterable[Any],
        total_tasks: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> Tuple[List[ExecutionResult], Optional[ProgressTracker]]:
        """
        使用函数工厂的批量执行，可以为每个任务动态创建不同的执行函数

        Args:
            func_factory: 函数工厂，根据上下文返回要执行的函数
            tasks_data: 任务数据列表
            total_tasks: 总任务数量
            show_progress: 是否显示进度
            **kwargs: 其他参数

        Returns:
            (结果列表, 进度跟踪器)

        Example:
            def task_factory(context: TaskContext):
                if context.task_id % 2 == 0:
                    return slow_processor
                else:
                    return fast_processor

            results, _ = await executor.execute_batch_with_factory(
                func_factory=task_factory,
                tasks_data=["data1", "data2", "data3"]
            )
        """

        async def factory_wrapper(context: TaskContext):
            actual_func = func_factory(context)
            return await actual_func(context.data)

        return await self.execute_batch_with_context(
            async_func=factory_wrapper,
            tasks_data=tasks_data,
            total_tasks=total_tasks,
            show_progress=show_progress,
            **kwargs,
        )


# ============== 检查点/断点续传 ==============

DEFAULT_CHECKPOINT_DIR = os.path.expanduser("~/.cache/maque/checkpoints")


@dataclass
class CheckpointConfig:
    """
    检查点配置

    Attributes:
        enabled: 是否启用检查点
        checkpoint_dir: 检查点存储目录
        checkpoint_interval: 每完成 N 个任务保存一次检查点
    """
    enabled: bool = False
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    checkpoint_interval: int = 100

    @classmethod
    def disabled(cls) -> "CheckpointConfig":
        return cls(enabled=False)

    @classmethod
    def default(cls) -> "CheckpointConfig":
        return cls(enabled=True)


class CheckpointManager:
    """
    检查点管理器

    用于在大规模批量任务中保存和恢复进度。
    使用 FlaxKV2 作为存储后端。

    Example:
        # 创建带检查点的执行
        checkpoint = CheckpointManager(CheckpointConfig.default())
        checkpoint_id = checkpoint.create("my_batch_task")

        # 执行时保存进度
        for i, result in enumerate(results):
            checkpoint.save_result(checkpoint_id, i, result)
            if i % 100 == 0:
                checkpoint.flush(checkpoint_id)

        # 中断后恢复
        completed, pending_indices = checkpoint.load(checkpoint_id, total_tasks=1000)
    """

    def __init__(self, config: Optional[CheckpointConfig] = None):
        if not FLAXKV_AVAILABLE:
            raise ImportError("检查点功能需要安装 flaxkv2: pip install flaxkv2")

        self.config = config or CheckpointConfig.disabled()
        self._db: Optional[FlaxKV] = None

        if self.config.enabled:
            self._db = FlaxKV(
                "checkpoints",
                self.config.checkpoint_dir,
                write_buffer_size=50,
                async_flush=True,
            )

    def create(self, name: str = "") -> str:
        """
        创建新的检查点

        Args:
            name: 检查点名称 (可选)

        Returns:
            检查点 ID
        """
        checkpoint_id = f"{name}_{uuid.uuid4().hex[:8]}" if name else uuid.uuid4().hex[:8]

        if self._db is not None:
            self._db[f"{checkpoint_id}:meta"] = {
                "name": name,
                "created_at": time.time(),
                "completed_count": 0,
            }

        return checkpoint_id

    def save_result(
        self,
        checkpoint_id: str,
        task_id: int,
        result: ExecutionResult,
    ) -> None:
        """保存单个任务结果"""
        if self._db is None:
            return

        # 保存结果（只保存可序列化的部分）
        self._db[f"{checkpoint_id}:result:{task_id}"] = {
            "task_id": result.task_id,
            "data": result.data,
            "status": result.status,
            "meta": result.meta,
            "latency": result.latency,
            "retry_count": result.retry_count,
        }

        # 更新计数
        meta = self._db.get(f"{checkpoint_id}:meta", {})
        meta["completed_count"] = meta.get("completed_count", 0) + 1
        meta["last_updated"] = time.time()
        self._db[f"{checkpoint_id}:meta"] = meta

    def save_pending(
        self,
        checkpoint_id: str,
        pending_data: List[Tuple[int, Any]],
    ) -> None:
        """
        保存待处理的任务数据

        Args:
            checkpoint_id: 检查点 ID
            pending_data: (task_id, data) 元组列表
        """
        if self._db is None:
            return
        self._db[f"{checkpoint_id}:pending"] = pending_data

    def load(
        self,
        checkpoint_id: str,
        total_tasks: int,
    ) -> Tuple[List[ExecutionResult], List[int]]:
        """
        加载检查点

        Args:
            checkpoint_id: 检查点 ID
            total_tasks: 总任务数

        Returns:
            (已完成的结果列表, 待处理的任务索引列表)
        """
        if self._db is None:
            return [], list(range(total_tasks))

        completed = []
        completed_ids = set()

        # 读取所有已完成的结果
        for key in self._db.keys():
            if key.startswith(f"{checkpoint_id}:result:"):
                data = self._db[key]
                result = ExecutionResult(
                    task_id=data["task_id"],
                    data=data["data"],
                    status=data["status"],
                    meta=data.get("meta"),
                    latency=data.get("latency", 0),
                    retry_count=data.get("retry_count", 0),
                )
                completed.append(result)
                completed_ids.add(data["task_id"])

        # 计算待处理的任务
        pending_indices = [i for i in range(total_tasks) if i not in completed_ids]

        return sorted(completed, key=lambda x: x.task_id), pending_indices

    def delete(self, checkpoint_id: str) -> None:
        """删除检查点"""
        if self._db is None:
            return

        keys_to_delete = [
            key for key in self._db.keys()
            if key.startswith(f"{checkpoint_id}:")
        ]
        for key in keys_to_delete:
            del self._db[key]

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """列出所有检查点"""
        if self._db is None:
            return []

        checkpoints = []
        for key in self._db.keys():
            if key.endswith(":meta"):
                checkpoint_id = key.replace(":meta", "")
                meta = self._db[key]
                checkpoints.append({
                    "id": checkpoint_id,
                    **meta,
                })

        return sorted(checkpoints, key=lambda x: x.get("created_at", 0), reverse=True)

    def close(self):
        """关闭检查点管理器"""
        if self._db is not None:
            self._db.close()
            self._db = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
