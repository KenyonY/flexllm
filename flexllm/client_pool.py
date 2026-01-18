"""
LLMClientPool - 多 Endpoint 客户端池

提供多个 LLM endpoint 的负载均衡和故障转移能力，接口与 LLMClient 一致。

Example:
    # 方式1：传入 endpoints 配置
    pool = LLMClientPool(
        endpoints=[
            {"base_url": "http://api1.com/v1", "api_key": "key1", "model": "qwen"},
            {"base_url": "http://api2.com/v1", "api_key": "key2", "model": "qwen"},
        ],
        load_balance="round_robin",
        fallback=True,
    )

    # 方式2：传入已有的 clients
    pool = LLMClientPool(
        clients=[client1, client2],
        load_balance="round_robin",
        fallback=True,
    )

    # 接口与 LLMClient 一致
    result = await pool.chat_completions(messages)
    results = await pool.chat_completions_batch(messages_list)
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger

from .base_client import ChatCompletionResult
from .llm_client import LLMClient
from .provider_router import ProviderConfig, ProviderRouter, Strategy


@dataclass
class EndpointConfig:
    """Endpoint 配置"""

    base_url: str
    api_key: str = "EMPTY"
    model: str = None
    provider: Literal["openai", "gemini", "auto"] = "auto"
    weight: float = 1.0
    # 其他 LLMClient 参数
    extra: dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


class LLMClientPool:
    """
    多 Endpoint 客户端池

    功能：
    - 负载均衡：round_robin, weighted, random
    - 故障转移：fallback=True 时自动尝试其他 endpoint
    - 健康检查：自动标记失败的 endpoint，一段时间后尝试恢复
    - 统一接口：与 LLMClient 完全一致的调用方式

    Attributes:
        load_balance: 负载均衡策略
        fallback: 是否启用故障转移
        max_fallback_attempts: 最大故障转移尝试次数
    """

    def __init__(
        self,
        endpoints: list[dict | EndpointConfig] = None,
        clients: list[LLMClient] = None,
        load_balance: Strategy = "round_robin",
        fallback: bool = True,
        max_fallback_attempts: int = None,
        failure_threshold: int = 3,
        recovery_time: float = 60.0,
        # 共享的 LLMClient 参数（仅当使用 endpoints 时生效）
        concurrency_limit: int = 10,
        max_qps: int = 1000,
        timeout: int = 120,
        retry_times: int = 3,
        **kwargs,
    ):
        """
        初始化客户端池

        Args:
            endpoints: Endpoint 配置列表，每个元素可以是 dict 或 EndpointConfig
            clients: 已创建的 LLMClient 列表（与 endpoints 二选一）
            load_balance: 负载均衡策略
                - "round_robin": 轮询
                - "weighted": 加权随机
                - "random": 随机
                - "fallback": 主备模式
            fallback: 是否启用故障转移（某个 endpoint 失败时尝试其他）
            max_fallback_attempts: 最大故障转移次数，默认为 endpoint 数量
            failure_threshold: 连续失败多少次后标记为不健康
            recovery_time: 不健康后多久尝试恢复（秒）
            concurrency_limit: 每个 client 的并发限制
            max_qps: 每个 client 的 QPS 限制
            timeout: 请求超时时间
            retry_times: 重试次数
            **kwargs: 其他传递给 LLMClient 的参数
        """
        if not endpoints and not clients:
            raise ValueError("必须提供 endpoints 或 clients")
        if endpoints and clients:
            raise ValueError("endpoints 和 clients 只能二选一")

        self._fallback = fallback
        self._load_balance = load_balance

        if clients:
            # 使用已有的 clients
            self._clients = clients
            self._endpoints = [
                EndpointConfig(
                    base_url=c._client._base_url,
                    api_key=c._client._api_key or "EMPTY",
                    model=c._model,
                )
                for c in clients
            ]
        else:
            # 从 endpoints 创建 clients
            self._endpoints = []
            self._clients = []

            for ep in endpoints:
                if isinstance(ep, dict):
                    ep = EndpointConfig(**ep)
                self._endpoints.append(ep)

                # 合并参数
                client_kwargs = {
                    "provider": ep.provider,
                    "base_url": ep.base_url,
                    "api_key": ep.api_key,
                    "model": ep.model,
                    "concurrency_limit": concurrency_limit,
                    "max_qps": max_qps,
                    "timeout": timeout,
                    "retry_times": retry_times,
                    **kwargs,
                    **(ep.extra or {}),
                }
                self._clients.append(LLMClient(**client_kwargs))

        # 创建路由器
        provider_configs = [
            ProviderConfig(
                base_url=ep.base_url,
                api_key=ep.api_key,
                weight=ep.weight,
                model=ep.model,
            )
            for ep in self._endpoints
        ]
        self._router = ProviderRouter(
            providers=provider_configs,
            strategy=load_balance,
            failure_threshold=failure_threshold,
            recovery_time=recovery_time,
        )

        # endpoint -> client 映射
        self._client_map = {
            ep.base_url: client for ep, client in zip(self._endpoints, self._clients)
        }

        self._max_fallback_attempts = max_fallback_attempts or len(self._clients)

    def _get_client(self) -> tuple[LLMClient, ProviderConfig]:
        """获取下一个可用的 client"""
        provider = self._router.get_next()
        client = self._client_map[provider.base_url]
        return client, provider

    async def chat_completions(
        self,
        messages: list[dict],
        model: str = None,
        return_raw: bool = False,
        return_usage: bool = False,
        **kwargs,
    ) -> str | ChatCompletionResult:
        """
        单条聊天完成（支持故障转移）

        Args:
            messages: 消息列表
            model: 模型名称（可选，使用 endpoint 配置的默认值）
            return_raw: 是否返回原始响应
            return_usage: 是否返回包含 usage 的结果
            **kwargs: 其他参数

        Returns:
            与 LLMClient.chat_completions 返回值一致
        """
        last_error = None
        tried_providers = set()

        for attempt in range(self._max_fallback_attempts):
            client, provider = self._get_client()

            # 避免重复尝试同一个 provider
            if provider.base_url in tried_providers:
                # 如果所有 provider 都试过了，退出
                if len(tried_providers) >= len(self._clients):
                    break
                continue

            tried_providers.add(provider.base_url)

            try:
                result = await client.chat_completions(
                    messages=messages,
                    model=model or provider.model,
                    return_raw=return_raw,
                    return_usage=return_usage,
                    **kwargs,
                )

                # 检查是否返回了 RequestResult（表示失败）
                if hasattr(result, "status") and result.status != "success":
                    raise RuntimeError(f"请求失败: {getattr(result, 'error', result)}")

                self._router.mark_success(provider)
                return result

            except Exception as e:
                last_error = e
                self._router.mark_failed(provider)
                logger.warning(f"Endpoint {provider.base_url} 失败: {e}")

                if not self._fallback:
                    raise

        raise last_error or RuntimeError("所有 endpoint 都失败了")

    def chat_completions_sync(
        self,
        messages: list[dict],
        model: str = None,
        return_raw: bool = False,
        return_usage: bool = False,
        **kwargs,
    ) -> str | ChatCompletionResult:
        """同步版本的聊天完成"""
        return asyncio.run(
            self.chat_completions(
                messages=messages,
                model=model,
                return_raw=return_raw,
                return_usage=return_usage,
                **kwargs,
            )
        )

    async def chat_completions_batch(
        self,
        messages_list: list[list[dict]],
        model: str = None,
        return_raw: bool = False,
        return_usage: bool = False,
        show_progress: bool = True,
        return_summary: bool = False,
        output_jsonl: str | None = None,
        flush_interval: float = 1.0,
        distribute: bool = True,
        **kwargs,
    ) -> list[str] | list[ChatCompletionResult] | tuple:
        """
        批量聊天完成（支持负载均衡和故障转移）

        Args:
            messages_list: 消息列表的列表
            model: 模型名称
            return_raw: 是否返回原始响应
            return_usage: 是否返回包含 usage 的结果
            show_progress: 是否显示进度条
            return_summary: 是否返回统计摘要
            output_jsonl: 输出文件路径（JSONL）
            flush_interval: 文件刷新间隔（秒）
            distribute: 是否将请求分散到多个 endpoint（True）
                        False 时使用单个 endpoint + fallback
            **kwargs: 其他参数

        Returns:
            与 LLMClient.chat_completions_batch 返回值一致
        """
        # output_jsonl 扩展名校验
        if output_jsonl and not output_jsonl.endswith(".jsonl"):
            raise ValueError(f"output_jsonl 必须使用 .jsonl 扩展名，当前: {output_jsonl}")

        if not distribute or len(self._clients) == 1:
            # 单 endpoint 模式：使用 fallback
            return await self._batch_with_fallback(
                messages_list=messages_list,
                model=model,
                return_raw=return_raw,
                return_usage=return_usage,
                show_progress=show_progress,
                return_summary=return_summary,
                output_jsonl=output_jsonl,
                flush_interval=flush_interval,
                **kwargs,
            )
        else:
            # 多 endpoint 分布式模式
            return await self._batch_distributed(
                messages_list=messages_list,
                model=model,
                return_raw=return_raw,
                return_usage=return_usage,
                show_progress=show_progress,
                return_summary=return_summary,
                output_jsonl=output_jsonl,
                flush_interval=flush_interval,
                **kwargs,
            )

    async def _batch_with_fallback(
        self,
        messages_list: list[list[dict]],
        model: str = None,
        return_raw: bool = False,
        return_usage: bool = False,
        show_progress: bool = True,
        return_summary: bool = False,
        output_jsonl: str | None = None,
        flush_interval: float = 1.0,
        **kwargs,
    ):
        """使用单个 endpoint + fallback 的批量调用"""
        last_error = None
        tried_providers = set()

        for attempt in range(self._max_fallback_attempts):
            client, provider = self._get_client()

            if provider.base_url in tried_providers:
                if len(tried_providers) >= len(self._clients):
                    break
                continue

            tried_providers.add(provider.base_url)

            try:
                result = await client.chat_completions_batch(
                    messages_list=messages_list,
                    model=model or provider.model,
                    return_raw=return_raw,
                    return_usage=return_usage,
                    show_progress=show_progress,
                    return_summary=return_summary,
                    output_jsonl=output_jsonl,
                    flush_interval=flush_interval,
                    **kwargs,
                )
                self._router.mark_success(provider)
                return result

            except Exception as e:
                last_error = e
                self._router.mark_failed(provider)
                logger.warning(f"Endpoint {provider.base_url} 批量调用失败: {e}")

                if not self._fallback:
                    raise

        raise last_error or RuntimeError("所有 endpoint 都失败了")

    async def _batch_distributed(
        self,
        messages_list: list[list[dict]],
        model: str = None,
        return_raw: bool = False,
        return_usage: bool = False,
        show_progress: bool = True,
        return_summary: bool = False,
        output_jsonl: str | None = None,
        flush_interval: float = 1.0,
        **kwargs,
    ):
        """
        动态分配：多个 worker 从共享队列取任务

        每个 client 启动 concurrency_limit 个 worker，所有 worker 从同一个队列
        竞争取任务。快的 client 会自动处理更多任务，实现动态负载均衡。
        """
        import json
        import time
        from pathlib import Path

        from tqdm import tqdm

        n = len(messages_list)
        results = [None] * n
        success_count = 0
        failed_count = 0
        cached_count = 0
        start_time = time.time()

        # 断点续传：读取已完成的记录
        completed_indices = set()
        if output_jsonl:
            output_path = Path(output_jsonl)
            if output_path.exists():
                records = []
                with open(output_path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            record = json.loads(line.strip())
                            if record.get("status") == "success" and "input" in record:
                                idx = record.get("index")
                                if 0 <= idx < n:
                                    records.append(record)
                        except (json.JSONDecodeError, KeyError, TypeError):
                            continue

                # 首尾校验
                file_valid = True
                if records:
                    first, last = records[0], records[-1]
                    if first["input"] != messages_list[first["index"]]:
                        file_valid = False
                    elif len(records) > 1 and last["input"] != messages_list[last["index"]]:
                        file_valid = False

                if file_valid:
                    for record in records:
                        idx = record["index"]
                        completed_indices.add(idx)
                        results[idx] = record["output"]
                    if completed_indices:
                        logger.info(f"从文件恢复: 已完成 {len(completed_indices)}/{n}")
                        cached_count = len(completed_indices)
                else:
                    raise ValueError(
                        f"文件校验失败: {output_jsonl} 中的 input 与当前 messages_list 不匹配。"
                        f"请删除或重命名该文件后重试。"
                    )

        # 共享任务队列（跳过已完成的）
        queue = asyncio.Queue()
        for idx, msg in enumerate(messages_list):
            if idx not in completed_indices:
                queue.put_nowait((idx, msg))

        pending_count = queue.qsize()
        if pending_count == 0:
            logger.info("所有任务已完成，无需执行")
            if return_summary:
                return results, {
                    "total": n,
                    "success": n,
                    "failed": 0,
                    "cached": cached_count,
                    "elapsed": 0,
                }
            return results

        logger.info(f"待执行: {pending_count}/{n}")

        # 进度条
        pbar = tqdm(total=pending_count, desc="Processing", disable=not show_progress)

        # 文件写入相关
        file_writer = None
        file_buffer = []
        last_flush_time = time.time()

        if output_jsonl:
            file_writer = open(output_jsonl, "a", encoding="utf-8")

        # 用于统计和线程安全更新
        lock = asyncio.Lock()

        def flush_to_file():
            """刷新缓冲区到文件"""
            nonlocal file_buffer, last_flush_time
            if file_writer and file_buffer:
                for record in file_buffer:
                    file_writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                file_writer.flush()
                file_buffer = []
                last_flush_time = time.time()

        async def worker(client_idx: int):
            """单个 worker：循环从队列取任务并执行"""
            nonlocal success_count, failed_count, last_flush_time

            client = self._clients[client_idx]
            provider = self._router._providers[client_idx].config
            effective_model = model or provider.model

            while True:
                try:
                    idx, msg = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                try:
                    result = await client.chat_completions(
                        messages=msg,
                        model=effective_model,
                        return_raw=return_raw,
                        return_usage=return_usage,
                        **kwargs,
                    )

                    # 检查是否返回了 RequestResult（表示失败）
                    if hasattr(result, "status") and result.status != "success":
                        raise RuntimeError(f"请求失败: {getattr(result, 'error', result)}")

                    results[idx] = result
                    self._router.mark_success(provider)

                    async with lock:
                        success_count += 1
                        pbar.update(1)

                        # 写入文件
                        if file_writer:
                            file_buffer.append(
                                {
                                    "index": idx,
                                    "output": result,
                                    "status": "success",
                                    "input": msg,
                                }
                            )
                            if time.time() - last_flush_time >= flush_interval:
                                flush_to_file()

                except Exception as e:
                    logger.warning(f"Task {idx} failed on {provider.base_url}: {e}")
                    results[idx] = None
                    self._router.mark_failed(provider)

                    async with lock:
                        failed_count += 1
                        pbar.update(1)

                        # 写入失败记录
                        if file_writer:
                            file_buffer.append(
                                {
                                    "index": idx,
                                    "output": None,
                                    "status": "error",
                                    "error": str(e),
                                    "input": msg,
                                }
                            )
                            if time.time() - last_flush_time >= flush_interval:
                                flush_to_file()

        try:
            # 启动所有 worker
            # 每个 client 启动 concurrency_limit 个 worker
            workers = []
            for client_idx, client in enumerate(self._clients):
                # 获取 client 的并发限制
                concurrency = getattr(client._client, "_concurrency_limit", 10)
                for _ in range(concurrency):
                    workers.append(worker(client_idx))

            # 并发执行所有 worker
            await asyncio.gather(*workers)

        finally:
            # 确保最后的数据写入
            flush_to_file()
            if file_writer:
                file_writer.close()
            pbar.close()

        if return_summary:
            summary = {
                "total": n,
                "success": success_count + cached_count,
                "failed": failed_count,
                "cached": cached_count,
                "elapsed": time.time() - start_time,
            }
            return results, summary

        return results

    def chat_completions_batch_sync(
        self,
        messages_list: list[list[dict]],
        model: str = None,
        return_raw: bool = False,
        return_usage: bool = False,
        show_progress: bool = True,
        return_summary: bool = False,
        output_jsonl: str | None = None,
        flush_interval: float = 1.0,
        distribute: bool = True,
        **kwargs,
    ) -> list[str] | list[ChatCompletionResult] | tuple:
        """同步版本的批量聊天完成"""
        return asyncio.run(
            self.chat_completions_batch(
                messages_list=messages_list,
                model=model,
                return_raw=return_raw,
                return_usage=return_usage,
                show_progress=show_progress,
                return_summary=return_summary,
                output_jsonl=output_jsonl,
                flush_interval=flush_interval,
                distribute=distribute,
                **kwargs,
            )
        )

    async def chat_completions_stream(
        self,
        messages: list[dict],
        model: str = None,
        return_usage: bool = False,
        **kwargs,
    ):
        """
        流式聊天完成（支持故障转移）

        Args:
            messages: 消息列表
            model: 模型名称
            return_usage: 是否返回 usage 信息
            **kwargs: 其他参数

        Yields:
            与 LLMClient.chat_completions_stream 一致
        """
        last_error = None
        tried_providers = set()

        for attempt in range(self._max_fallback_attempts):
            client, provider = self._get_client()

            if provider.base_url in tried_providers:
                if len(tried_providers) >= len(self._clients):
                    break
                continue

            tried_providers.add(provider.base_url)

            try:
                async for chunk in client.chat_completions_stream(
                    messages=messages,
                    model=model or provider.model,
                    return_usage=return_usage,
                    **kwargs,
                ):
                    yield chunk
                self._router.mark_success(provider)
                return

            except Exception as e:
                last_error = e
                self._router.mark_failed(provider)
                logger.warning(f"Endpoint {provider.base_url} 流式调用失败: {e}")

                if not self._fallback:
                    raise

        raise last_error or RuntimeError("所有 endpoint 都失败了")

    @property
    def stats(self) -> dict:
        """返回池的统计信息"""
        return {
            "load_balance": self._load_balance,
            "fallback": self._fallback,
            "num_endpoints": len(self._clients),
            "router_stats": self._router.stats,
        }

    def close(self):
        """关闭所有客户端"""
        for client in self._clients:
            client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return (
            f"LLMClientPool(endpoints={len(self._clients)}, "
            f"load_balance='{self._load_balance}', fallback={self._fallback})"
        )
