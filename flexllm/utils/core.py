"""Core utilities for flexllm"""

import asyncio
import logging
import random
import re
from contextvars import ContextVar
from functools import wraps

# 用于在 async_retry 重试时通知外部（如进度条）
retry_callback: ContextVar[callable] = ContextVar("retry_callback", default=None)

# 退避上限：服务端给出的 Retry-After 可能是几百秒，批量任务不能被单个请求拖死
DEFAULT_MAX_RETRY_DELAY = 60.0


def compute_retry_delay(
    attempt: int,
    base_delay: float,
    max_delay: float = DEFAULT_MAX_RETRY_DELAY,
    retry_after: float | None = None,
) -> float:
    """计算第 attempt 次失败后的等待时长（attempt 从 0 开始）。

    - 服务端给出 Retry-After 时以它为准（它知道配额何时恢复，比本地猜测准），
      并向上抖动 0~25%：批量场景下成百上千个请求会收到相同的 Retry-After，
      不抖动则会在同一时刻齐发，再次把服务端打挂。只向上抖是因为早于服务端
      要求重试必然再吃一次 429。基数先压到 max_delay/1.25，使抖动后的结果
      仍不超过 max_delay——max_delay 是硬上限，同时抖动在任何情况下都不失效。
    - 没有 Retry-After 时用指数退避 + equal jitter（delay/2 + rand(0, delay/2)）：
      固定延迟会让并发失败的请求同步重试，形成惊群。
    """
    if retry_after is not None:
        return min(retry_after, max_delay / 1.25) * (1 + random.random() * 0.25)
    # 指数先封顶再乘：2**attempt 是任意精度整数，attempt>=1024 时与 float
    # 相乘会 OverflowError，中断重试并吞掉真实的 HTTP 错误
    delay = min(base_delay * 2 ** min(attempt, 32), max_delay)
    return delay / 2 + random.random() * delay / 2


def async_retry(
    retry_times: int = 3,
    retry_delay: float = 1.0,
    exceptions: tuple = (Exception,),
    logger=None,
    max_delay: float = DEFAULT_MAX_RETRY_DELAY,
):
    """
    Async retry decorator

    重试间隔为指数退避 + 抖动；异常若带 `retry_after` 属性（如 HTTP 429 的
    Retry-After 响应头），则以该值为准。

    Args:
        retry_times: 最大尝试次数（含首次调用）
        retry_delay: 退避基数（秒），实际间隔为 retry_delay * 2**attempt 加抖动
        exceptions: 需要重试的异常类型
        logger: Logger instance
        max_delay: 单次退避上限（秒）
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
                    delay = compute_retry_delay(
                        attempt,
                        retry_delay,
                        max_delay,
                        retry_after=getattr(e, "retry_after", None),
                    )
                    await asyncio.sleep(delay)
            # retry_times<=0 时上面的循环体不执行，仍需保证至少调用一次
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
