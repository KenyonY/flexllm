"""
LLMClient - 统一的 LLM 客户端（现在是 LLMClientPool 的别名）

为了向后兼容，LLMClient 现在指向 LLMClientPool。
LLMClientPool 已经支持单 endpoint 模式，行为与原 LLMClient 完全一致。

Example (单 endpoint):
    >>> client = LLMClient(
    ...     provider="openai",
    ...     base_url="https://api.openai.com/v1",
    ...     api_key="your-key",
    ...     model="gpt-4",
    ... )
    >>> result = await client.chat_completions(messages)

Example (多 endpoint):
    >>> client = LLMClient(
    ...     endpoints=[
    ...         {"base_url": "http://api1.com/v1", "model": "qwen"},
    ...         {"base_url": "http://api2.com/v1", "model": "qwen"},
    ...     ],
    ...     load_balance="round_robin",
    ... )
    >>> result = await client.chat_completions(messages)
"""

from .pool import LLMClientPool

# LLMClient 现在是 LLMClientPool 的别名
# 单 endpoint 时行为完全一致，零额外开销
LLMClient = LLMClientPool

__all__ = ["LLMClient"]
