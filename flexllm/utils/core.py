"""Core utilities for flexllm"""

import asyncio
import logging
from contextvars import ContextVar
from functools import wraps

# 用于在 async_retry 重试时通知外部（如进度条）
retry_callback: ContextVar[callable] = ContextVar("retry_callback", default=None)


def async_retry(
    retry_times: int = 3,
    retry_delay: float = 1.0,
    exceptions: tuple = (Exception,),
    logger=None,
):
    """
    Async retry decorator

    Args:
        retry_times: Maximum retry count
        retry_delay: Delay between retries (seconds)
        exceptions: Exception types to retry on
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(retry_times):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == retry_times - 1:
                        raise
                    logger.debug(f"Attempt {attempt + 1} failed: {str(e)}")
                    # 通知外部重试（如更新进度条）
                    callback = retry_callback.get()
                    if callback:
                        callback()
                    await asyncio.sleep(retry_delay)
            return await func(*args, **kwargs)

        return wrapper

    return decorator
