import asyncio
import time
from typing import Callable, Any, List, Union, Sequence
from dataclasses import dataclass
from flexllm.async_api.progress import ProgressTracker, ProgressBarConfig
from flexllm.async_api.interface import RequestResult


@dataclass
class FunctionArgs:
    args: tuple = ()
    kwargs: dict = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


async def concurrent_executor(
        async_func: Callable,
        func_args: Sequence[Union[tuple, dict, FunctionArgs]],
        concurrency_limit: int,
        show_progress: bool = True
) -> List[Any]:
    """
    并发执行异步函数的控制器

    Args:
        async_func: 要执行的异步函数
        func_args: 函数参数列表，支持多种参数形式:
            - tuple: 按位置传参
            - dict: 按关键字传参
            - FunctionArgs: 同时包含位置参数和关键字参数
        concurrency_limit: 并发上限
        show_progress: 是否显示进度条

    Returns:
        list: 所有任务的执行结果列表
    """
    semaphore = asyncio.Semaphore(concurrency_limit)

    def normalize_args(arg) -> FunctionArgs:
        if isinstance(arg, FunctionArgs):
            return arg
        elif isinstance(arg, tuple):
            return FunctionArgs(args=arg)
        elif isinstance(arg, dict):
            return FunctionArgs(kwargs=arg)
        else:
            return FunctionArgs(args=(arg,))

    async def wrapped_func(func_arg: FunctionArgs, task_id: int):
        async with semaphore:
            try:
                start_time = time.time()
                result = await async_func(*func_arg.args, **func_arg.kwargs)
                status = 'success'
            except Exception as e:
                result = e
                status = 'error'

            if progress:
                progress.update(RequestResult(
                    request_id=task_id,
                    data=result,
                    status=status,
                    # meta=None,
                    latency=time.time() - start_time
                ))
            return task_id, result

    # 标准化所有参数
    normalized_args = [normalize_args(arg) for arg in func_args]
    total_tasks = len(normalized_args)

    progress = None
    if show_progress:
        progress = ProgressTracker(
            total_tasks,
            concurrency=concurrency_limit,
            config=ProgressBarConfig()
        )

    # 创建任务列表
    tasks = [
        asyncio.create_task(wrapped_func(arg, i))
        for i, arg in enumerate(normalized_args)
    ]

    # 等待所有任务完成
    completed_results = await asyncio.gather(*tasks)

    # 按任务ID排序结果
    sorted_results = sorted(completed_results, key=lambda x: x[0])
    results = [result for _, result in sorted_results]

    if progress:
        progress.summary()

    return results
