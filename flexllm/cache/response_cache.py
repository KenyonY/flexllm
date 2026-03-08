#! /usr/bin/env python3

"""
LLM 响应缓存模块

使用 FlaxKV2 (LMDB) 作为存储后端，提供高性能缓存。
LMDB 原生支持多进程并发读写，无需 IPC 中转。
"""

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

from ..pricing.token_counter import messages_hash

if TYPE_CHECKING:
    from flaxkv2 import FlaxKV


DEFAULT_CACHE_DIR = os.path.expanduser("~/.flexllm/cache/response")


@dataclass
class ResponseCacheConfig:
    """
    响应缓存配置

    Attributes:
        enabled: 是否启用缓存
        cache_dir: 缓存目录
        ttl: 缓存过期时间(秒)，0 表示永不过期
    """

    enabled: bool = False
    cache_dir: str = DEFAULT_CACHE_DIR
    ttl: int = 86400  # 24小时

    @classmethod
    def disabled(cls) -> "ResponseCacheConfig":
        """禁用缓存"""
        return cls(enabled=False)

    @classmethod
    def default(cls) -> "ResponseCacheConfig":
        """默认配置：禁用缓存"""
        return cls(enabled=False)

    @classmethod
    def with_ttl(cls, ttl: int = 3600, cache_dir: str = None) -> "ResponseCacheConfig":
        """
        启用缓存，自定义 TTL

        Args:
            ttl: 过期时间（秒）
            cache_dir: 缓存目录
        """
        return cls(
            enabled=True,
            ttl=ttl,
            cache_dir=cache_dir or DEFAULT_CACHE_DIR,
        )

    @classmethod
    def persistent(cls, cache_dir: str = DEFAULT_CACHE_DIR) -> "ResponseCacheConfig":
        """持久缓存：永不过期"""
        return cls(enabled=True, cache_dir=cache_dir, ttl=0)


class ResponseCache:
    """
    LLM 响应缓存

    使用 FlaxKV2 (LMDB) 存储，支持 TTL 过期、多进程并发读写。
    """

    def __init__(self, config: ResponseCacheConfig | None = None):
        self.config = config or ResponseCacheConfig.disabled()
        self._stats = {"hits": 0, "misses": 0}
        self._db: FlaxKV | None = None

        if self.config.enabled:
            try:
                from flaxkv2 import FlaxKV
            except ImportError:
                raise ImportError("缓存功能需要安装 flaxkv2。请运行: pip install flexllm[cache]")

            ttl = self.config.ttl if self.config.ttl > 0 else None

            logger.debug(f"使用 LMDB 缓存: cache_dir={self.config.cache_dir}")
            self._db = FlaxKV(
                "llm_cache",
                self.config.cache_dir,
                default_ttl=ttl,
                write_buffer_size=100,
                async_flush=True,
                auto_nested=False,
            )

    def _make_key(self, messages: list[dict], model: str, **kwargs) -> str:
        """生成缓存键"""
        return messages_hash(messages, model, **kwargs)

    def get(self, messages: list[dict], model: str = "", **kwargs) -> Any | None:
        """
        获取缓存的响应

        Args:
            messages: 消息列表
            model: 模型名称
            **kwargs: 其他参数 (temperature, max_tokens 等)

        Returns:
            缓存的响应，未命中返回 None
        """
        if self._db is None:
            return None

        cache_key = self._make_key(messages, model, **kwargs)
        result = self._db.get(cache_key)

        if result is not None:
            self._stats["hits"] += 1
        else:
            self._stats["misses"] += 1

        return result

    def set(self, messages: list[dict], response: Any, model: str = "", **kwargs) -> None:
        """
        存储响应到缓存

        Args:
            messages: 消息列表
            response: API 响应
            model: 模型名称
            **kwargs: 其他参数
        """
        if self._db is None:
            return

        cache_key = self._make_key(messages, model, **kwargs)
        self._db[cache_key] = response

    def get_batch(
        self, messages_list: list[list[dict]], model: str = "", **kwargs
    ) -> tuple[list[Any | None], list[int]]:
        """
        批量获取缓存

        Returns:
            (cached_responses, uncached_indices)
        """
        if self._db is None:
            return [None] * len(messages_list), list(range(len(messages_list)))

        cache_keys = [self._make_key(msgs, model, **kwargs) for msgs in messages_list]
        results = self._db.batch_get(cache_keys)

        cached = []
        uncached_indices = []
        for i, result in enumerate(results):
            cached.append(result)
            if result is not None:
                self._stats["hits"] += 1
            else:
                self._stats["misses"] += 1
                uncached_indices.append(i)
        return cached, uncached_indices

    def set_batch(
        self, messages_list: list[list[dict]], responses: list[Any], model: str = "", **kwargs
    ) -> None:
        """批量存储缓存"""
        if self._db is None:
            return
        items = {}
        for messages, response in zip(messages_list, responses):
            if response is not None:
                items[self._make_key(messages, model, **kwargs)] = response
        if items:
            self._db.batch_set(items)

    def clear(self) -> int:
        """清空缓存"""
        if self._db is None:
            return 0
        keys = list(self._db.keys())
        count = len(keys)
        for key in keys:
            del self._db[key]
        return count

    def close(self):
        """关闭缓存"""
        if self._db is not None:
            self._db.close()
            self._db = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def stats(self) -> dict[str, Any]:
        """返回缓存统计"""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        return {
            **self._stats,
            "total": total,
            "hit_rate": round(hit_rate, 4),
        }
