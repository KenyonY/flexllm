"""Core utilities for flexllm"""

from functools import wraps
import asyncio
import logging


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
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    await asyncio.sleep(retry_delay)
            return await func(*args, **kwargs)

        return wrapper

    return decorator
