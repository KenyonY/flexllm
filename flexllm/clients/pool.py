"""
LLMClientPool - 统一的 LLM 客户端

支持单 endpoint 和多 endpoint 两种模式：
- 单 endpoint：直接使用底层客户端（OpenAI/Gemini/Claude），零额外开销
- 多 endpoint：负载均衡 + 故障转移

Example:
    # 单 endpoint 模式（等价于原 LLMClient）
    client = LLMClientPool(
        base_url="https://api.openai.com/v1",
        api_key="your-key",
        model="gpt-4",
    )

    # 多 endpoint 模式（负载均衡 + 故障转移）
    pool = LLMClientPool(
        endpoints=[
            {"base_url": "http://api1.com/v1", "api_key": "key1", "model": "qwen"},
            {"base_url": "http://api2.com/v1", "api_key": "key2", "model": "qwen"},
        ],
        load_balance="round_robin",
        fallback=True,
    )

    # 接口完全一致
    result = await client.chat_completions(messages)
    results = await pool.chat_completions_batch(messages_list)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Union

logger = logging.getLogger(__name__)

from ..async_api.interface import RequestResult
from ..async_api.progress import ProgressBarConfig, ProgressTracker
from ..cache import ResponseCacheConfig
from ..pricing import get_model_pricing
from ..utils.core import retry_callback
from .base import ChatCompletionResult, LLMClientBase
from .claude import ClaudeClient
from .gemini import GeminiClient
from .openai import OpenAIClient
from .router import ProviderConfig, ProviderRouter, Strategy

if TYPE_CHECKING:
    from ..async_api.interface import RequestResult


@dataclass
class EndpointConfig:
    """Endpoint 配置"""

    base_url: str
    api_key: str = "EMPTY"
    model: str = None
    provider: Literal["openai", "gemini", "auto"] = "auto"
    weight: float = 1.0
    # endpoint 级别的 rate limit 配置（None 表示使用全局配置）
    concurrency_limit: int = None
    max_qps: int = None
    # 其他 LLMClient 参数
    extra: dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


class LLMClientPool:
    """
    统一的 LLM 客户端（支持单/多 endpoint）

    功能：
    - 单 endpoint：直接使用底层客户端，零额外开销
    - 多 endpoint：负载均衡 + 故障转移
    - 统一接口：所有模式API完全一致

    Attributes:
        load_balance: 负载均衡策略（仅多 endpoint 模式）
        fallback: 是否启用故障转移
        max_fallback_attempts: 最大故障转移尝试次数
    """

    def __init__(
        self,
        # 单 endpoint 参数（与原 LLMClient 兼容）
        provider: Literal["openai", "gemini", "claude", "auto"] = "auto",
        base_url: str = None,
        api_key: str = None,
        model: str = None,
        # 多 endpoint 参数
        endpoints: list[dict | EndpointConfig] = None,
        clients: list = None,  # 已废弃的参数，保留向后兼容
        load_balance: Strategy = "round_robin",
        fallback: bool = True,
        max_fallback_attempts: int = None,
        failure_threshold: int | float = float("inf"),
        recovery_time: float = 60.0,
        # 共享参数
        concurrency_limit: int = 10,
        max_qps: int = None,
        timeout: int = 120,
        retry_times: int = None,
        cache_image: bool = False,
        cache_dir: str = "image_cache",
        # Gemini/Vertex AI 专用
        use_vertex_ai: bool = False,
        project_id: str = None,
        location: str = "us-central1",
        credentials=None,
        # 响应缓存配置
        cache: ResponseCacheConfig | None = None,
        **kwargs,
    ):
        """
        初始化统一 LLM 客户端

        Args:
            # 单 endpoint 模式参数
            provider: Provider 类型（"openai", "gemini", "claude", "auto"）
            base_url: API 基础 URL（单 endpoint 模式）
            api_key: API 密钥（单 endpoint 模式）
            model: 默认模型名称

            # 多 endpoint 模式参数
            endpoints: Endpoint 配置列表，每个元素可以是 dict 或 EndpointConfig
            clients: （已废弃）已创建的客户端列表
            load_balance: 负载均衡策略
                - "round_robin": 轮询
                - "weighted": 加权随机
                - "random": 随机
                - "fallback": 主备模式
            fallback: 是否启用故障转移（某个 endpoint 失败时尝试其他）
            max_fallback_attempts: 最大故障转移次数，默认为 endpoint 数量
            failure_threshold: 连续失败多少次后标记为不健康
            recovery_time: 不健康后多久尝试恢复（秒）

            # 共享参数
            concurrency_limit: 并发请求限制
            max_qps: 最大 QPS（openai 默认 1000，gemini 默认 60）
            timeout: 请求超时时间
            retry_times: 重试次数。fallback=True 时表示总重试次数（会在多个 endpoint 间分配），默认为 0；
                fallback=False 时为单 client 重试次数，默认为 3
            cache_image: 是否缓存图片
            cache_dir: 图片缓存目录
            use_vertex_ai: 是否使用 Vertex AI（仅 Gemini）
            project_id: GCP 项目 ID（仅 Vertex AI）
            location: GCP 区域（仅 Vertex AI）
            credentials: Google Cloud 凭证（仅 Vertex AI）
            cache: 响应缓存配置
            **kwargs: 其他传递给底层客户端的参数
        """
        # 判断是单 endpoint 还是多 endpoint 模式
        # 单模式：提供了 base_url，或者 provider 是 gemini/claude（它们不需要 base_url）
        # 多模式：提供了 endpoints 或 clients
        # 无参数：抛出错误
        has_multi_endpoint = endpoints is not None or clients is not None

        # 如果没有提供多 endpoint 参数，检查是否是单 endpoint 模式
        if not has_multi_endpoint:
            # 单 endpoint 模式的条件：
            # 1. 提供了 base_url，或
            # 2. provider 是 gemini/claude（它们不需要 base_url），或
            # 3. 提供了 api_key（可能是 gemini/claude）
            has_single_endpoint = (
                base_url is not None
                or provider in ("gemini", "claude")
                or (api_key is not None and provider != "openai")  # openai 必须有 base_url
            )
        else:
            has_single_endpoint = base_url is not None

        if not has_single_endpoint and not has_multi_endpoint:
            raise ValueError("必须提供 base_url（单 endpoint）或 endpoints/clients（多 endpoint）")

        if has_single_endpoint and has_multi_endpoint:
            raise ValueError(
                "不能同时提供 base_url 和 endpoints/clients，请选择单或多 endpoint 模式"
            )

        if has_single_endpoint:
            # ========== 单 endpoint 模式 ==========
            self._init_single_mode(
                provider=provider,
                base_url=base_url,
                api_key=api_key,
                model=model,
                concurrency_limit=concurrency_limit,
                max_qps=max_qps,
                timeout=timeout,
                retry_times=retry_times if retry_times is not None else 3,
                cache_image=cache_image,
                cache_dir=cache_dir,
                use_vertex_ai=use_vertex_ai,
                project_id=project_id,
                location=location,
                credentials=credentials,
                cache=cache,
                **kwargs,
            )
        else:
            # ========== 多 endpoint 模式 ==========
            if not endpoints and not clients:
                raise ValueError("多 endpoint 模式必须提供 endpoints 或 clients")
            if endpoints and clients:
                raise ValueError("endpoints 和 clients 只能二选一")

            self._init_multi_mode(
                endpoints=endpoints,
                clients=clients,
                load_balance=load_balance,
                fallback=fallback,
                max_fallback_attempts=max_fallback_attempts,
                failure_threshold=failure_threshold,
                recovery_time=recovery_time,
                concurrency_limit=concurrency_limit,
                max_qps=max_qps,
                timeout=timeout,
                retry_times=retry_times,
                cache_image=cache_image,
                cache_dir=cache_dir,
                cache=cache,
                **kwargs,
            )

    @staticmethod
    def _infer_provider(base_url: str, use_vertex_ai: bool) -> str:
        """根据 base_url 推断 provider"""
        if use_vertex_ai:
            return "gemini"
        if base_url:
            url_lower = base_url.lower()
            if "generativelanguage.googleapis.com" in url_lower:
                return "gemini"
            if "aiplatform.googleapis.com" in url_lower:
                return "gemini"
            if "anthropic.com" in url_lower:
                return "claude"
        return "openai"

    def _create_base_client(
        self,
        provider: str,
        base_url: str = None,
        api_key: str = None,
        model: str = None,
        concurrency_limit: int = 10,
        max_qps: int = None,
        timeout: int = 120,
        retry_times: int = 3,
        cache_image: bool = False,
        cache_dir: str = "image_cache",
        use_vertex_ai: bool = False,
        project_id: str = None,
        location: str = "us-central1",
        credentials=None,
        cache: ResponseCacheConfig | None = None,
        **kwargs,
    ) -> LLMClientBase:
        """创建底层客户端（OpenAI/Gemini/Claude）"""
        if provider == "gemini":
            return GeminiClient(
                api_key=api_key,
                model=model,
                base_url=base_url,
                concurrency_limit=concurrency_limit,
                max_qps=max_qps if max_qps is not None else 60,
                timeout=timeout,
                retry_times=retry_times,
                cache_image=cache_image,
                cache_dir=cache_dir,
                cache=cache,
                use_vertex_ai=use_vertex_ai,
                project_id=project_id,
                location=location,
                credentials=credentials,
                **kwargs,
            )
        elif provider == "claude":
            if not api_key:
                raise ValueError("Claude provider 需要提供 api_key")
            return ClaudeClient(
                api_key=api_key,
                model=model,
                base_url=base_url,
                concurrency_limit=concurrency_limit,
                max_qps=max_qps if max_qps is not None else 60,
                timeout=timeout,
                retry_times=retry_times,
                cache_image=cache_image,
                cache_dir=cache_dir,
                cache=cache,
                **kwargs,
            )
        else:  # openai
            if not base_url:
                raise ValueError("OpenAI provider 需要提供 base_url")
            return OpenAIClient(
                base_url=base_url,
                api_key=api_key or "EMPTY",
                model=model,
                concurrency_limit=concurrency_limit,
                max_qps=max_qps if max_qps is not None else 1000,
                timeout=timeout,
                retry_times=retry_times,
                cache_image=cache_image,
                cache_dir=cache_dir,
                cache=cache,
                **kwargs,
            )

    def _init_single_mode(
        self,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        concurrency_limit: int,
        max_qps: int,
        timeout: int,
        retry_times: int,
        cache_image: bool,
        cache_dir: str,
        use_vertex_ai: bool,
        project_id: str,
        location: str,
        credentials,
        cache: ResponseCacheConfig,
        **kwargs,
    ):
        """初始化单 endpoint 模式"""
        self._mode = "single"
        self._model = model
        self._fallback = False
        self._load_balance = None
        self._router = None
        self._clients = None
        self._endpoints = None
        self._client_map = None
        self._max_fallback_attempts = 1

        # 自动推断 provider
        if provider == "auto":
            provider = self._infer_provider(base_url, use_vertex_ai)

        self._provider = provider

        # 直接创建底层客户端（跳过 LLMClient 中间层）
        self._single_client = self._create_base_client(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            model=model,
            concurrency_limit=concurrency_limit,
            max_qps=max_qps,
            timeout=timeout,
            retry_times=retry_times,
            cache_image=cache_image,
            cache_dir=cache_dir,
            use_vertex_ai=use_vertex_ai,
            project_id=project_id,
            location=location,
            credentials=credentials,
            cache=cache,
            **kwargs,
        )

    def _init_multi_mode(
        self,
        endpoints: list,
        clients: list,
        load_balance: str,
        fallback: bool,
        max_fallback_attempts: int,
        failure_threshold: float,
        recovery_time: float,
        concurrency_limit: int,
        max_qps: int,
        timeout: int,
        retry_times: int,
        cache_image: bool,
        cache_dir: str,
        cache: ResponseCacheConfig,
        **kwargs,
    ):
        """初始化多 endpoint 模式"""
        self._mode = "multi"
        self._fallback = fallback
        self._load_balance = load_balance
        self._single_client = None
        self._provider = None
        self._model = None

        if clients:
            # 使用已有的 clients（向后兼容，但已废弃）
            # 需要从 LLMClient 包装器中提取底层客户端
            self._clients = []
            self._endpoints = []
            for c in clients:
                # 检查是否是 LLMClient（有 _client 属性）
                if hasattr(c, "_client"):
                    self._clients.append(c._client)  # 提取底层客户端
                    self._endpoints.append(
                        EndpointConfig(
                            base_url=c._client._base_url,
                            api_key=c._client._api_key or "EMPTY",
                            model=c._model if hasattr(c, "_model") else None,
                        )
                    )
                else:
                    # 已经是底层客户端
                    self._clients.append(c)
                    self._endpoints.append(
                        EndpointConfig(
                            base_url=c._base_url,
                            api_key=c._api_key or "EMPTY",
                            model=getattr(c, "_model", None),
                        )
                    )
        else:
            # 从 endpoints 创建底层 clients
            self._endpoints = []
            self._clients = []

            num_endpoints = len(endpoints)

            # 确定有效的 client retry_times
            # fallback 模式下，用户指定的 retry_times 是"总重试次数"，会在多个 endpoint 间分配
            if fallback:
                user_retry_times = retry_times if retry_times is not None else 0
                effective_retry_times = user_retry_times // num_endpoints
            else:
                # 非 fallback 模式
                effective_retry_times = retry_times if retry_times is not None else 3

            for ep in endpoints:
                if isinstance(ep, dict):
                    ep = EndpointConfig(**ep)
                self._endpoints.append(ep)

                # 确定 rate limit 配置（endpoint 级别优先）
                ep_concurrency = (
                    ep.concurrency_limit if ep.concurrency_limit is not None else concurrency_limit
                )
                ep_max_qps = ep.max_qps if ep.max_qps is not None else max_qps

                # 自动推断 provider
                provider = ep.provider
                if provider == "auto":
                    provider = self._infer_provider(ep.base_url, False)

                # 合并参数
                client_kwargs = {
                    "provider": provider,
                    "base_url": ep.base_url,
                    "api_key": ep.api_key,
                    "model": ep.model,
                    "concurrency_limit": ep_concurrency,
                    "max_qps": ep_max_qps,
                    "timeout": timeout,
                    "retry_times": effective_retry_times,
                    "cache_image": cache_image,
                    "cache_dir": cache_dir,
                    "cache": cache,
                    **kwargs,
                    **(ep.extra or {}),
                }
                # 直接创建底层客户端
                self._clients.append(self._create_base_client(**client_kwargs))

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

    def _get_client(self) -> tuple[LLMClientBase, ProviderConfig]:
        """获取下一个可用的 client（返回底层客户端）"""
        provider = self._router.get_next()
        client = self._client_map[provider.base_url]
        return client, provider

    async def chat_completions(
        self,
        messages: list[dict],
        model: str = None,
        return_raw: bool = False,
        return_usage: bool = False,
        show_progress: bool = False,
        preprocess_msg: bool = False,
        **kwargs,
    ) -> Union[str, ChatCompletionResult, "RequestResult"]:
        """
        单条聊天完成（支持故障转移）

        Args:
            messages: 消息列表
            model: 模型名称（可选，使用 endpoint 配置的默认值）
            return_raw: 是否返回原始响应
            return_usage: 是否返回包含 usage 的结果
            show_progress: 是否显示进度
            preprocess_msg: 是否预处理消息（图片转 base64）
            **kwargs: 其他参数

        Returns:
            与 LLMClient.chat_completions 返回值一致
        """
        # 单 endpoint 模式：直接调用底层客户端
        if self._mode == "single":
            return await self._single_client.chat_completions(
                messages=messages,
                model=model,
                return_raw=return_raw,
                return_usage=return_usage,
                show_progress=show_progress,
                preprocess_msg=preprocess_msg,
                **kwargs,
            )

        # 多 endpoint 模式：使用 fallback
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
                    show_progress=show_progress,
                    preprocess_msg=preprocess_msg,
                    **kwargs,
                )

                # 检查是否返回了 RequestResult（表示失败）
                if hasattr(result, "status") and result.status != "success":
                    # 从 result.data 中提取错误信息
                    error_msg = "unknown"
                    if hasattr(result, "data") and isinstance(result.data, dict):
                        error_msg = result.data.get("error", "unknown")
                    raise RuntimeError(error_msg)

                self._router.mark_success(provider)
                return result

            except Exception as e:
                last_error = e
                self._router.mark_failed(provider)
                logger.debug(f"Endpoint {provider.base_url} 失败: {e}")

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
    ) -> Union[str, ChatCompletionResult, "RequestResult"]:
        """同步版本的聊天完成"""
        # 单 endpoint 模式：使用底层客户端的 sync 方法
        if self._mode == "single":
            return self._single_client.chat_completions_sync(
                messages=messages,
                model=model,
                return_raw=return_raw,
                return_usage=return_usage,
                **kwargs,
            )

        # 多 endpoint 模式：运行异步方法
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
        track_cost: bool = False,
        preprocess_msg: bool = False,
        output_jsonl: str | None = None,
        flush_interval: float = 1.0,
        distribute: bool = True,
        metadata_list: list[dict] | None = None,
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
            track_cost: 是否在进度条中显示实时成本
            preprocess_msg: 是否预处理消息
            output_jsonl: 输出文件路径（JSONL）
            flush_interval: 文件刷新间隔（秒）
            distribute: 是否将请求分散到多个 endpoint（True）
                        False 时使用单个 endpoint + fallback
            metadata_list: 元数据列表，与 messages_list 等长，每个元素保存到对应输出记录
            **kwargs: 其他参数

        Returns:
            与 LLMClient.chat_completions_batch 返回值一致
        """
        # 单 endpoint 模式：直接调用底层客户端
        if self._mode == "single":
            return await self._single_client.chat_completions_batch(
                messages_list=messages_list,
                model=model,
                return_raw=return_raw,
                return_usage=return_usage,
                show_progress=show_progress,
                return_summary=return_summary,
                preprocess_msg=preprocess_msg,
                output_jsonl=output_jsonl,
                flush_interval=flush_interval,
                metadata_list=metadata_list,
                **kwargs,
            )

        # 多 endpoint 模式：参数校验
        # track_cost 需要 usage 信息
        if track_cost:
            return_usage = True

        # metadata_list 长度校验
        if metadata_list is not None and len(metadata_list) != len(messages_list):
            raise ValueError(
                f"metadata_list 长度 ({len(metadata_list)}) 必须与 messages_list 长度 ({len(messages_list)}) 一致"
            )

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
                track_cost=track_cost,
                output_jsonl=output_jsonl,
                flush_interval=flush_interval,
                metadata_list=metadata_list,
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
                track_cost=track_cost,
                output_jsonl=output_jsonl,
                flush_interval=flush_interval,
                metadata_list=metadata_list,
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
        track_cost: bool = False,
        output_jsonl: str | None = None,
        flush_interval: float = 1.0,
        metadata_list: list[dict] | None = None,
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
                    track_cost=track_cost,
                    output_jsonl=output_jsonl,
                    flush_interval=flush_interval,
                    metadata_list=metadata_list,
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
        track_cost: bool = False,
        output_jsonl: str | None = None,
        flush_interval: float = 1.0,
        metadata_list: list[dict] | None = None,
        **kwargs,
    ):
        """
        动态分配：多个 worker 从共享队列取任务

        每个 client 启动 concurrency_limit 个 worker，所有 worker 从同一个队列
        竞争取任务。快的 client 会自动处理更多任务，实现动态负载均衡。

        支持：
        - Fallback 重试：任务失败时自动尝试其他 endpoint
        - 响应缓存：复用 LLMClient 的缓存能力
        """
        import json
        from pathlib import Path

        n = len(messages_list)
        results = [None] * n
        cached_count = 0
        file_restored_count = 0
        start_time = time.time()

        # 获取所有 endpoint 的 base_url 集合（用于 fallback 判断）
        all_endpoints = {ep.base_url for ep in self._endpoints}
        num_endpoints = len(all_endpoints)

        # 获取响应缓存（如果有的话，使用第一个 client 的缓存）
        # 缓存已支持存储 usage 信息，return_usage 时也可使用缓存
        response_cache = None
        for client in self._clients:
            cache = getattr(client._client, "_response_cache", None)
            if cache is not None:
                response_cache = cache
                break

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
                        file_restored_count = len(completed_indices)
                else:
                    raise ValueError(
                        f"文件校验失败: {output_jsonl} 中的 input 与当前 messages_list 不匹配。"
                        f"请删除或重命名该文件后重试。"
                    )

        # 检查缓存命中（如果启用了缓存）
        effective_model = model or self._endpoints[0].model
        if response_cache is not None:
            for idx, msg in enumerate(messages_list):
                if idx in completed_indices:
                    continue
                cached_result = response_cache.get(msg, model=effective_model, **kwargs)
                if cached_result is not None:
                    # 缓存格式为 {"content": ..., "usage": ...}
                    if return_usage:
                        results[idx] = ChatCompletionResult(
                            content=cached_result["content"],
                            usage=cached_result.get("usage"),
                        )
                    else:
                        results[idx] = cached_result["content"]
                    completed_indices.add(idx)
                    cached_count += 1
            if cached_count > 0:
                logger.info(f"缓存命中: {cached_count}/{n}")

        # 共享任务队列（跳过已完成的）
        # 队列元素: (idx, msg, tried_endpoints: set)
        queue = asyncio.Queue()
        for idx, msg in enumerate(messages_list):
            if idx not in completed_indices:
                queue.put_nowait((idx, msg, set()))

        pending_count = queue.qsize()
        if pending_count == 0:
            logger.info("所有任务已完成，无需执行")
            if return_summary:
                return results, {
                    "total": n,
                    "success": n,
                    "failed": 0,
                    "cached": cached_count + file_restored_count,
                    "elapsed": 0,
                }
            return results

        logger.info(f"待执行: {pending_count}/{n}")

        # 计算总并发数
        total_concurrency = sum(
            getattr(client._client, "_concurrency_limit", 10) for client in self._clients
        )

        # 进度条配置（支持成本显示）
        progress_config = ProgressBarConfig(show_cost=track_cost) if show_progress else None

        # 获取第一个 endpoint 的模型用于显示
        first_model = model or self._endpoints[0].model
        pricing = get_model_pricing(first_model) if track_cost else None
        input_price = pricing["input"] * 1e6 if pricing else None
        output_price = pricing["output"] * 1e6 if pricing else None

        # 创建进度追踪器
        tracker = (
            ProgressTracker(
                total_requests=pending_count,
                concurrency=total_concurrency,
                config=progress_config,
                model_name=first_model if track_cost else None,
                input_price_per_1m=input_price,
                output_price_per_1m=output_price,
            )
            if show_progress
            else None
        )

        # 文件写入相关
        file_writer = None
        file_buffer = []
        last_flush_time = time.time()

        if output_jsonl:
            file_writer = open(output_jsonl, "a", encoding="utf-8")

        # 用于统计和线程安全更新
        lock = asyncio.Lock()
        # 活跃任务计数（用于判断 worker 是否应该退出）
        active_tasks = 0
        all_done = asyncio.Event()

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
            """单个 worker：循环从队列取任务并执行，支持 fallback 重试"""
            nonlocal last_flush_time, active_tasks

            client = self._clients[client_idx]
            provider = self._router._providers[client_idx].config
            my_endpoint = provider.base_url
            worker_model = model or provider.model

            while not all_done.is_set():
                try:
                    idx, msg, tried_endpoints = queue.get_nowait()
                except asyncio.QueueEmpty:
                    # 队列为空，检查是否还有活跃任务
                    async with lock:
                        if active_tasks == 0 and queue.empty():
                            all_done.set()
                            break
                    # 等待一小段时间后重试（可能有任务被放回队列）
                    await asyncio.sleep(0.05)
                    continue

                # 增加活跃任务计数
                async with lock:
                    active_tasks += 1

                # 如果已尝试过当前 endpoint，放回队列让其他 worker 处理
                if my_endpoint in tried_endpoints:
                    # 检查是否所有 endpoint 都已尝试
                    if len(tried_endpoints) >= num_endpoints:
                        # 所有 endpoint 都失败了，标记最终失败
                        async with lock:
                            active_tasks -= 1
                            if tracker:
                                req_result = RequestResult(
                                    request_id=idx,
                                    data={"error": "All endpoints failed"},
                                    status="error",
                                    latency=0,
                                )
                                tracker.update(req_result)
                            if file_writer:
                                record = {
                                    "index": idx,
                                    "output": None,
                                    "status": "error",
                                    "error": f"All {num_endpoints} endpoints failed",
                                    "input": msg,
                                }
                                if metadata_list is not None:
                                    record["metadata"] = metadata_list[idx]
                                file_buffer.append(record)
                        continue
                    # 放回队列，让其他 endpoint 的 worker 处理
                    await queue.put((idx, msg, tried_endpoints))
                    async with lock:
                        active_tasks -= 1
                    await asyncio.sleep(0.01)  # 短暂让出，避免死循环
                    continue

                task_start = time.time()
                try:
                    # 设置重试回调，让 async_retry 重试时更新进度条
                    if tracker:
                        retry_callback.set(tracker.increment_retry)
                    result = await client.chat_completions(
                        messages=msg,
                        model=worker_model,
                        return_raw=return_raw,
                        return_usage=return_usage,
                        **kwargs,
                    )

                    # 检查是否返回了 RequestResult（表示失败）
                    if hasattr(result, "status") and result.status != "success":
                        # 从 result.data 中提取错误信息
                        error_type = "unknown"
                        error_detail = ""
                        if hasattr(result, "data") and isinstance(result.data, dict):
                            error_type = result.data.get("error", "unknown")
                            error_detail = result.data.get("detail", "")
                        # 构造包含类型和详情的错误消息
                        error_msg = f"{error_type}: {error_detail}" if error_detail else error_type
                        raise RuntimeError(error_msg)

                    latency = time.time() - task_start
                    results[idx] = result
                    self._router.mark_success(provider)

                    # 写入缓存（存储 content 和 usage）
                    if response_cache is not None:
                        if hasattr(result, "content"):
                            cache_data = {
                                "content": result.content,
                                "usage": getattr(result, "usage", None),
                            }
                        else:
                            cache_data = {"content": result, "usage": None}
                        response_cache.set(msg, cache_data, model=worker_model, **kwargs)

                    async with lock:
                        active_tasks -= 1
                        # 更新进度条
                        if tracker:
                            req_result = RequestResult(
                                request_id=idx,
                                data=result,
                                status="success",
                                latency=latency,
                            )
                            tracker.update(req_result)

                            # 更新成本信息
                            if track_cost and hasattr(result, "usage") and result.usage:
                                usage = result.usage
                                input_tokens = usage.get("prompt_tokens", 0)
                                output_tokens = usage.get("completion_tokens", 0)
                                cost = 0.0
                                if pricing:
                                    cost = (
                                        input_tokens * pricing["input"]
                                        + output_tokens * pricing["output"]
                                    )
                                tracker.update_cost(input_tokens, output_tokens, cost)

                        # 写入文件
                        if file_writer:
                            # 处理 ChatCompletionResult 对象的序列化
                            if hasattr(result, "content"):
                                output_content = result.content
                                output_usage = getattr(result, "usage", None)
                            else:
                                output_content = result
                                output_usage = None

                            record = {
                                "index": idx,
                                "output": output_content,
                                "status": "success",
                                "input": msg,
                            }
                            if metadata_list is not None:
                                record["metadata"] = metadata_list[idx]
                            if output_usage:
                                record["usage"] = output_usage
                            file_buffer.append(record)
                            if time.time() - last_flush_time >= flush_interval:
                                flush_to_file()

                except Exception as e:
                    latency = time.time() - task_start
                    self._router.mark_failed(provider)

                    # 记录已尝试的 endpoint
                    tried_endpoints = tried_endpoints | {my_endpoint}

                    # 检查是否还有其他 endpoint 可以重试
                    if self._fallback and len(tried_endpoints) < num_endpoints:
                        # 放回队列，让其他 endpoint 重试（进度条会显示 retry 计数）
                        await queue.put((idx, msg, tried_endpoints))
                        async with lock:
                            active_tasks -= 1
                            # 更新进度条的 retry 计数
                            if tracker:
                                tracker.increment_retry()
                    else:
                        # 所有 endpoint 都失败了，或者未启用 fallback
                        results[idx] = None

                        async with lock:
                            active_tasks -= 1
                            # 更新进度条
                            if tracker:
                                # 直接使用异常消息作为错误信息
                                req_result = RequestResult(
                                    request_id=idx,
                                    data={"error": str(e)},
                                    status="error",
                                    latency=latency,
                                )
                                tracker.update(req_result)

                            # 写入失败记录
                            if file_writer:
                                record = {
                                    "index": idx,
                                    "output": None,
                                    "status": "error",
                                    "error": str(e),
                                    "input": msg,
                                }
                                if metadata_list is not None:
                                    record["metadata"] = metadata_list[idx]
                                file_buffer.append(record)
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
                # 自动 compact：去重并按 index 排序
                if output_jsonl:
                    self._clients[0]._client._compact_output_file(output_jsonl)
            # 打印最终统计
            if tracker:
                tracker.summary(print_to_console=True)

        if return_summary:
            total_cached = cached_count + file_restored_count
            summary = {
                "total": n,
                "success": (tracker.success_count if tracker else 0) + total_cached,
                "failed": tracker.error_count if tracker else 0,
                "cached": total_cached,
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
        track_cost: bool = False,
        output_jsonl: str | None = None,
        flush_interval: float = 1.0,
        distribute: bool = True,
        metadata_list: list[dict] | None = None,
        **kwargs,
    ) -> list[str] | list[ChatCompletionResult] | tuple:
        """同步版本的批量聊天完成"""
        # 单 endpoint 模式：使用底层客户端的 sync 方法
        if self._mode == "single":
            return self._single_client.chat_completions_batch_sync(
                messages_list=messages_list,
                model=model,
                return_raw=return_raw,
                return_usage=return_usage,
                show_progress=show_progress,
                return_summary=return_summary,
                output_jsonl=output_jsonl,
                flush_interval=flush_interval,
                metadata_list=metadata_list,
                **kwargs,
            )

        # 多 endpoint 模式：运行异步方法
        return asyncio.run(
            self.chat_completions_batch(
                messages_list=messages_list,
                model=model,
                return_raw=return_raw,
                return_usage=return_usage,
                show_progress=show_progress,
                return_summary=return_summary,
                track_cost=track_cost,
                output_jsonl=output_jsonl,
                flush_interval=flush_interval,
                distribute=distribute,
                metadata_list=metadata_list,
                **kwargs,
            )
        )

    async def chat_completions_stream(
        self,
        messages: list[dict],
        model: str = None,
        return_usage: bool = False,
        preprocess_msg: bool = False,
        timeout: int = None,
        **kwargs,
    ):
        """
        流式聊天完成（支持故障转移）

        Args:
            messages: 消息列表
            model: 模型名称
            return_usage: 是否返回 usage 信息
            preprocess_msg: 是否预处理消息
            timeout: 超时时间（秒）
            **kwargs: 其他参数

        Yields:
            与 LLMClient.chat_completions_stream 一致
        """
        # 单 endpoint 模式：直接调用底层客户端
        if self._mode == "single":
            async for chunk in self._single_client.chat_completions_stream(
                messages=messages,
                model=model,
                return_usage=return_usage,
                preprocess_msg=preprocess_msg,
                timeout=timeout,
                **kwargs,
            ):
                yield chunk
            return

        # 多 endpoint 模式：使用 fallback
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
                    preprocess_msg=preprocess_msg,
                    timeout=timeout,
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

    async def iter_chat_completions_batch(
        self,
        messages_list: list[list[dict]],
        model: str = None,
        return_raw: bool = False,
        return_usage: bool = False,
        show_progress: bool = True,
        preprocess_msg: bool = False,
        output_jsonl: str | None = None,
        flush_interval: float = 1.0,
        metadata_list: list[dict] | None = None,
        batch_size: int = None,
        **kwargs,
    ):
        """
        迭代式批量聊天完成（边请求边返回结果）

        Args:
            messages_list: 消息列表的列表
            model: 模型名称
            return_raw: 是否返回原始响应
            return_usage: 是否在 result 对象上添加 usage 属性
            show_progress: 是否显示进度条
            preprocess_msg: 是否预处理消息
            output_jsonl: 输出文件路径（JSONL）
            flush_interval: 文件刷新间隔（秒）
            metadata_list: 元数据列表
            batch_size: 每批返回的数量
            **kwargs: 其他参数

        Yields:
            result: 包含 content、usage、original_idx 等属性的结果对象
        """
        # 单 endpoint 模式：直接调用底层客户端
        if self._mode == "single":
            async for result in self._single_client.iter_chat_completions_batch(
                messages_list=messages_list,
                model=model,
                return_raw=return_raw,
                return_usage=return_usage,
                show_progress=show_progress,
                preprocess_msg=preprocess_msg,
                output_jsonl=output_jsonl,
                flush_interval=flush_interval,
                metadata_list=metadata_list,
                batch_size=batch_size,
                **kwargs,
            ):
                yield result
            return

        # 多 endpoint 模式：暂不支持，使用批量方法
        # TODO: 未来可以实现分布式迭代
        raise NotImplementedError("多 endpoint 模式暂不支持 iter_chat_completions_batch")

    def model_list(self) -> list[str]:
        """获取可用模型列表"""
        if self._mode == "single":
            return self._single_client.model_list()
        else:
            # 多 endpoint 模式：返回第一个客户端的模型列表
            return self._clients[0].model_list() if self._clients else []

    def parse_thoughts(self, response_data: dict) -> dict:
        """
        从响应中解析思考内容和答案

        Args:
            response_data: 原始响应数据（通过 return_raw=True 获取）

        Returns:
            dict: {"thought": str, "answer": str}
        """
        if self._mode == "single":
            # 单模式：根据 provider 选择解析方法
            if self._provider == "gemini":
                return GeminiClient.parse_thoughts(response_data)
            elif self._provider == "claude":
                return ClaudeClient.parse_thoughts(response_data)
            else:
                return OpenAIClient.parse_thoughts(response_data)
        else:
            # 多模式：使用第一个客户端的方法
            if isinstance(self._clients[0], GeminiClient):
                return GeminiClient.parse_thoughts(response_data)
            elif isinstance(self._clients[0], ClaudeClient):
                return ClaudeClient.parse_thoughts(response_data)
            else:
                return OpenAIClient.parse_thoughts(response_data)

    @property
    def provider(self) -> str:
        """返回当前使用的 provider"""
        if self._mode == "single":
            return self._provider
        else:
            # 多模式：返回 "multi"
            return "multi"

    @property
    def client(self) -> LLMClientBase:
        """返回底层客户端实例（单模式）或第一个客户端（多模式）"""
        if self._mode == "single":
            return self._single_client
        else:
            return self._clients[0] if self._clients else None

    @property
    def _client(self) -> LLMClientBase:
        """向后兼容属性：返回底层客户端"""
        return self.client

    @property
    def stats(self) -> dict:
        """返回池的统计信息"""
        if self._mode == "single":
            return {
                "mode": "single",
                "provider": self._provider,
                "model": self._model,
            }
        else:
            return {
                "mode": "multi",
                "load_balance": self._load_balance,
                "fallback": self._fallback,
                "num_endpoints": len(self._clients),
                "router_stats": self._router.stats,
            }

    async def aclose(self):
        """异步关闭所有客户端（推荐在异步上下文中使用）"""
        if self._mode == "single":
            await self._single_client.aclose()
        else:
            for client in self._clients:
                await client.aclose()

    def close(self):
        """同步关闭所有客户端"""
        if self._mode == "single":
            self._single_client.close()
        else:
            for client in self._clients:
                client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    def __repr__(self) -> str:
        if self._mode == "single":
            return f"LLMClientPool(provider='{self._provider}', model='{self._model}')"
        else:
            return (
                f"LLMClientPool(endpoints={len(self._clients)}, "
                f"load_balance='{self._load_balance}', fallback={self._fallback})"
            )

    def __getattr__(self, name):
        """自动委托未显式定义的方法给底层客户端（仅单模式）"""
        if self._mode == "single":
            return getattr(self._single_client, name)
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}' "
                f"(仅单 endpoint 模式支持自动委托)"
            )
