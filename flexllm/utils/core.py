"""Core utilities for flexllm"""

import asyncio
import logging
import re
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


def safe_repr_source(source: str, max_length: int = 100) -> str:
    """安全地表示图像源，避免输出大量base64字符串"""
    if not source:
        return "空源"

    # 检查是否是base64数据URI
    if source.startswith("data:image/") and ";base64," in source:
        parts = source.split(";base64,", 1)
        if len(parts) == 2:
            mime_type = parts[0].replace("data:", "")
            base64_data = parts[1]
            return f"[{mime_type} base64数据 长度:{len(base64_data)}]"

    # 检查是否是纯base64字符串（很长且只包含base64字符）
    if len(source) > 100 and all(
        c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in source
    ):
        return f"[base64数据 长度:{len(source)}]"

    # 普通字符串，截断显示
    if len(source) <= max_length:
        return source
    else:
        return source[:max_length] + "..."


def safe_repr_error(error_msg: str, max_length: int = 200) -> str:
    """安全地表示错误信息，避免输出大量base64字符串"""
    if not error_msg:
        return error_msg

    # 检查错误信息中是否包含data:image的base64数据
    if "data:image/" in error_msg and ";base64," in error_msg:
        # 使用正则表达式替换base64数据URI
        pattern = r"data:image/[^;]+;base64,[A-Za-z0-9+/]+=*"

        def replace_base64(match):
            full_uri = match.group(0)
            parts = full_uri.split(";base64,", 1)
            if len(parts) == 2:
                mime_type = parts[0].replace("data:", "")
                base64_data = parts[1]
                return f"[{mime_type} base64数据 长度:{len(base64_data)}]"
            return full_uri

        error_msg = re.sub(pattern, replace_base64, error_msg)

    # 截断过长的错误信息
    if len(error_msg) <= max_length:
        return error_msg
    else:
        return error_msg[:max_length] + "..."
