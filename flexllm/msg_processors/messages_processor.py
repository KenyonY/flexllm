from __future__ import annotations
import asyncio
import aiohttp
import time
from copy import deepcopy
from typing import Optional, Callable, TYPE_CHECKING
from loguru import logger


if TYPE_CHECKING:
    from .image_processor import ImageCacheConfig
else:
    ImageCacheConfig = None  # 避免mypy等类型检查器报错

try:
    from tqdm.asyncio import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


async def process_content_recursive(
    content, session, cache_config: Optional["ImageCacheConfig"] = None, **kwargs
):
    """Recursively process a content dictionary, replacing any URL with its Base64 equivalent.

    Args:
        content: Content dictionary to process
        session: aiohttp.ClientSession for async URL fetching
        cache_config: Image cache configuration, if None or disabled, no caching will be used
        **kwargs: Additional arguments to pass to image processing functions
    """
    from .image_processor import encode_image_to_base64

    if isinstance(content, dict):
        for key, value in content.items():
            if key == "url" and isinstance(value, str):  # Detect URL fields
                base64_data = await encode_image_to_base64(
                    value,
                    session,
                    max_width=kwargs.get("max_width"),
                    max_height=kwargs.get("max_height"),
                    max_pixels=kwargs.get("max_pixels"),
                    cache_config=cache_config,
                )
                if base64_data:
                    content[key] = base64_data
            else:
                await process_content_recursive(
                    value, session, cache_config=cache_config, **kwargs
                )
    elif isinstance(content, list):
        for item in content:
            await process_content_recursive(
                item, session, cache_config=cache_config, **kwargs
            )


async def messages_preprocess(
    messages, inplace=False, cache_config: Optional["ImageCacheConfig"] = None, **kwargs
):
    """Process a list of messages, converting URLs in any type of content to Base64.

    Args:
        messages: List of messages to process
        inplace: Whether to modify the messages in-place or create a copy
        cache_config: Image cache configuration object
        **kwargs: Additional arguments to pass to image processing functions
    """
    if not inplace:
        messages = deepcopy(messages)

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_content_recursive(
                message, session, cache_config=cache_config, **kwargs
            )
            for message in messages
        ]
        await asyncio.gather(*tasks)
    return messages


async def batch_messages_preprocess(
    messages_list,
    max_concurrent=5,
    inplace=False,
    cache_config: Optional["ImageCacheConfig"] = None,
    as_iterator=False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    show_progress: bool = False,
    progress_desc: str = "处理消息",
    **kwargs,
):
    """Process multiple lists of messages in batches.

    Args:
        messages_list: List, iterator or async iterator of message lists to process
        max_concurrent: Maximum number of concurrent batches to process
        inplace: Whether to modify the messages in-place
        cache_config: Image cache configuration object
        as_iterator: Whether to return an async iterator instead of a list
        progress_callback: Optional callback function to report progress (current, total)
        show_progress: Whether to show a progress bar using tqdm
        progress_desc: Description for the progress bar
        **kwargs: Additional arguments to pass to image processing functions

    Returns:
        List of processed message lists or an async iterator yielding processed message lists
    """

    # 创建处理单个消息列表的函数
    async def process_single_batch(messages, semaphore, index=None):
        async with semaphore:
            try:
                processed_messages = await messages_preprocess(
                    messages, inplace=inplace, cache_config=cache_config, **kwargs
                )
            except Exception as e:
                logger.error(f"{e=}\n")
                processed_messages = messages
            return processed_messages, index

    # 进度报告函数
    def report_progress(current: int, total: int, start_time: float = None):
        if progress_callback:
            try:
                # 计算时间信息
                elapsed_time = time.time() - start_time if start_time else 0

                # 创建扩展的进度信息
                progress_info = {
                    "current": current,
                    "total": total,
                    "percentage": (current / total * 100) if total > 0 else 0,
                    "elapsed_time": elapsed_time,
                    "estimated_total_time": (elapsed_time / current * total)
                    if current > 0
                    else 0,
                    "estimated_remaining_time": (
                        elapsed_time / current * (total - current)
                    )
                    if current > 0
                    else 0,
                    "rate": current / elapsed_time if elapsed_time > 0 else 0,
                }

                # 如果回调函数接受单个参数，传递扩展信息；否则保持兼容性
                import inspect

                sig = inspect.signature(progress_callback)
                if len(sig.parameters) == 1:
                    progress_callback(progress_info)
                else:
                    progress_callback(current, total)

            except Exception as e:
                logger.warning(f"进度回调函数执行失败: {e}")

    # 如果要求返回迭代器
    if as_iterator:

        async def process_iterator():
            semaphore = asyncio.Semaphore(max_concurrent)

            # 检查是否为异步迭代器
            is_async_iterator = hasattr(messages_list, "__aiter__")

            processed_count = 0
            total_count = None
            messages_to_process = messages_list  # 使用新变量名避免作用域问题

            # 如果可以获取总数，先计算总数
            if not is_async_iterator and hasattr(messages_list, "__len__"):
                total_count = len(messages_list)
            elif not is_async_iterator:
                # 对于迭代器，先转换为列表获取长度
                messages_list_converted = list(messages_list)
                total_count = len(messages_list_converted)
                messages_to_process = iter(messages_list_converted)  # 使用新变量名

            # 创建进度条
            pbar = None
            start_time = time.time()
            if show_progress and TQDM_AVAILABLE and total_count:
                # 自定义进度条格式，显示时间信息
                bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                pbar = tqdm(
                    total=total_count,
                    desc=progress_desc,
                    unit="批次",
                    bar_format=bar_format,
                    ncols=100,  # 控制进度条宽度
                    miniters=1,  # 每次更新都显示
                )

            try:
                # 处理异步迭代器
                if is_async_iterator:
                    pending_tasks = []
                    task_index = 0
                    async for messages in messages_to_process:
                        # 如果已经达到最大并发数，等待一个任务完成
                        if len(pending_tasks) >= max_concurrent:
                            done, pending_tasks = await asyncio.wait(
                                pending_tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                            for task in done:
                                result, _ = await task
                                processed_count += 1
                                if pbar:
                                    pbar.update(1)
                                report_progress(
                                    processed_count,
                                    total_count or processed_count,
                                    start_time,
                                )
                                yield result

                        # 创建新任务
                        task = asyncio.create_task(
                            process_single_batch(messages, semaphore, task_index)
                        )
                        pending_tasks.append(task)
                        task_index += 1

                    # 等待所有剩余任务完成
                    if pending_tasks:
                        for task in asyncio.as_completed(pending_tasks):
                            result, _ = await task
                            processed_count += 1
                            if pbar:
                                pbar.update(1)
                            report_progress(
                                processed_count,
                                total_count or processed_count,
                                start_time,
                            )
                            yield result

                # 处理同步迭代器或列表
                else:
                    # 转换为列表以避免消耗迭代器
                    if not isinstance(messages_to_process, (list, tuple)):
                        messages_list_converted = list(messages_to_process)
                    else:
                        messages_list_converted = messages_to_process

                    if not total_count:
                        total_count = len(messages_list_converted)
                        if pbar:
                            pbar.total = total_count

                    # 分批处理
                    for i in range(0, len(messages_list_converted), max_concurrent):
                        batch = messages_list_converted[i : i + max_concurrent]
                        tasks = [
                            process_single_batch(messages, semaphore, i + j)
                            for j, messages in enumerate(batch)
                        ]
                        results = await asyncio.gather(*tasks)

                        for result, _ in results:
                            processed_count += 1
                            if pbar:
                                pbar.update(1)
                            report_progress(processed_count, total_count, start_time)
                            yield result
            finally:
                if pbar:
                    pbar.close()

        return process_iterator()

    # 原始实现，返回列表
    else:
        semaphore = asyncio.Semaphore(max_concurrent)

        # 检查是否为异步迭代器
        is_async_iterator = hasattr(messages_list, "__aiter__")

        # 转换为列表
        if is_async_iterator:
            messages_list_converted = []
            async for messages in messages_list:
                messages_list_converted.append(messages)
        elif not isinstance(messages_list, (list, tuple)):
            messages_list_converted = list(messages_list)
        else:
            messages_list_converted = messages_list

        if not messages_list_converted:
            return []

        total_count = len(messages_list_converted)
        processed_count = 0

        # 创建进度条
        pbar = None
        start_time = time.time()
        if show_progress and TQDM_AVAILABLE:
            # 自定义进度条格式，显示时间信息
            bar_format = (
                "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
            pbar = tqdm(
                total=total_count,
                desc=progress_desc,
                unit=" items",
                bar_format=bar_format,
                ncols=100,  # 控制进度条宽度
                miniters=1,  # 每次更新都显示
            )

        try:
            # 分批处理以实现进度更新
            results = []
            for i in range(0, len(messages_list_converted), max_concurrent):
                batch = messages_list_converted[i : i + max_concurrent]
                tasks = [
                    process_single_batch(messages, semaphore, i + j)
                    for j, messages in enumerate(batch)
                ]
                batch_results = await asyncio.gather(*tasks)

                for result, _ in batch_results:
                    results.append(result)
                    processed_count += 1
                    if pbar:
                        pbar.update(1)
                    report_progress(processed_count, total_count, start_time)

            return results
        finally:
            if pbar:
                pbar.close()


# 为了向后兼容，提供别名
batch_process_messages = batch_messages_preprocess
