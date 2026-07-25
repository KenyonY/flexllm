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

from ..async_api.core import ConcurrencyLimiter, RateLimiter
from ..async_api.interface import RequestResult
from ..async_api.progress import ProgressBarConfig, ProgressTracker
from ..cache import ResponseCacheConfig
from ..pricing import get_model_pricing
from ..utils.core import retry_callback
from .base import ChatCompletionResult, LLMClientBase
from .batch_helpers import (
    JsonlWriter,
    build_gen_params_list,
    validate_batch_params,
)
from .claude import ClaudeClient
from .gemini import GeminiClient
from .openai import OpenAIClient
from .router import ProviderConfig, ProviderRouter

if TYPE_CHECKING:
    from ..async_api.interface import RequestResult


@dataclass
class EndpointConfig:
    """Endpoint 配置"""

    base_url: str
    api_key: str = "EMPTY"
    model: str = None
    provider: Literal["openai", "gemini", "auto"] = "auto"
    # endpoint 级别的 rate limit 配置（None 表示使用全局配置）
    concurrency_limit: int = None
    max_qps: int = None
    # endpoint 级别的正向代理（None 表示使用 pool 顶层 proxy）
    # 用于部分 endpoint 需经网关、部分直连的场景——这是环境变量做不到的
    proxy: str = None
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
    - 多 endpoint：轮询分发 + 故障转移
    - 统一接口：所有模式API完全一致

    Attributes:
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
        fallback: bool = True,
        max_fallback_attempts: int = None,
        failure_threshold: int | float = float("inf"),
        recovery_time: float = 60.0,
        # 共享参数
        concurrency_limit: int = 10,
        max_qps: int = None,
        total_concurrency_limit: int = None,
        total_max_qps: float = None,
        timeout: int = 120,
        retry_times: int = None,
        cache_image: bool = False,
        cache_dir: str | None = None,
        proxy: str | None = None,
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
            fallback: 是否启用故障转移（某个 endpoint 失败时尝试其他）
            max_fallback_attempts: 最大故障转移次数，默认为 endpoint 数量
            failure_threshold: 连续失败多少次后标记为不健康
            recovery_time: 不健康后多久尝试恢复（秒）

            # 共享参数
            concurrency_limit: 并发请求限制（多 endpoint 模式下为每个 endpoint 的默认值）
            max_qps: 最大 QPS（openai 默认 1000，gemini 默认 60；多 endpoint 模式下为每个
                endpoint 的默认值）
            total_concurrency_limit: 跨所有 endpoint 的全局并发硬上限。与 per-endpoint
                的 concurrency_limit 同时生效（哪个先触发就卡在哪）。不传时无全局上限，
                行为与之前完全一致。不约束流式接口（per-endpoint 限制同样不约束流式）。
            total_max_qps: 跨所有 endpoint 的全局 QPS 硬上限，语义同上。
                QPS 令牌按 wire 请求计：fallback 换 endpoint 重试会再取一次令牌。
            timeout: 请求超时时间
            retry_times: 重试次数。fallback=True 时表示总重试次数（会在多个 endpoint 间分配），默认为 0；
                fallback=False 时为单 client 重试次数，默认为 3
            cache_image: 是否缓存图片
            cache_dir: 图片缓存目录
            proxy: 正向代理 URL（http://gateway:8080 或 socks5://gateway:1080，
                均支持 scheme://user:pass@host:port）。
                多 endpoint 模式下为各 endpoint 的默认值，可被 EndpointConfig.proxy 覆盖，
                从而做到"部分 endpoint 经网关、部分直连"——这是进程级环境变量做不到的。
                SOCKS 需额外依赖：pip install 'flexllm[socks]'。
            use_vertex_ai: 是否使用 Vertex AI（仅 Gemini）
            project_id: GCP 项目 ID（仅 Vertex AI）
            location: GCP 区域（仅 Vertex AI）
            credentials: Google Cloud 凭证（仅 Vertex AI）
            cache: 响应缓存配置
            **kwargs: 其他传递给底层客户端的参数
        """
        # from_config() 使用的配置属性
        self._config_system: str | None = None
        self._config_params: dict = {}
        self._config_user_template: str | None = None

        # 根据 api_key 前缀自动推断 provider（当 provider="auto" 且无 base_url 时）
        if provider == "auto" and not base_url and api_key:
            if isinstance(api_key, str) and "sk-ant-oat" in api_key:
                provider = "claude"

        # 判断是单 endpoint 还是多 endpoint 模式
        # 单模式：提供了 base_url，或者 provider 是 gemini/claude（它们不需要 base_url）
        # 多模式：提供了 endpoints
        # 无参数：抛出错误
        has_multi_endpoint = endpoints is not None

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
            raise ValueError("必须提供 base_url（单 endpoint）或 endpoints（多 endpoint）")

        if has_single_endpoint and has_multi_endpoint:
            raise ValueError("不能同时提供 base_url 和 endpoints，请选择单或多 endpoint 模式")

        # pool 级全局限制器：注入到每个底层客户端的 ConcurrentRequester 中，
        # 在唯一的请求执行点（_send_single_request）生效，天然覆盖单条/批量/迭代
        # 全部路径（流式除外，流式不经过 ConcurrentRequester）
        if total_concurrency_limit is not None and total_concurrency_limit < 1:
            raise ValueError(f"total_concurrency_limit 必须 ≥ 1，当前为 {total_concurrency_limit}")
        if total_max_qps is not None and total_max_qps <= 0:
            raise ValueError(f"total_max_qps 必须 > 0，当前为 {total_max_qps}")
        self._pool_limiter = (
            ConcurrencyLimiter(total_concurrency_limit) if total_concurrency_limit else None
        )
        self._pool_rate_limiter = RateLimiter(total_max_qps) if total_max_qps else None

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
                proxy=proxy,
                use_vertex_ai=use_vertex_ai,
                project_id=project_id,
                location=location,
                credentials=credentials,
                cache=cache,
                **kwargs,
            )
        else:
            # ========== 多 endpoint 模式 ==========
            if not endpoints:
                raise ValueError("多 endpoint 模式必须提供 endpoints")

            self._init_multi_mode(
                endpoints=endpoints,
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
                proxy=proxy,
                cache=cache,
                **kwargs,
            )

    @classmethod
    def from_config(
        cls,
        config: str = None,
        *,
        model: str = None,
        **overrides,
    ) -> "LLMClientPool":
        """从配置文件创建客户端

        Args:
            config: 配置文件路径，None 则按默认路径搜索
            model: 模型 name 或 id，None 则用配置文件中的默认模型
            **overrides: 覆盖配置中的参数（base_url, api_key, concurrency_limit 等）

        Returns:
            配置好的 LLMClientPool 实例

        Example:
            # 默认配置文件 + 默认模型
            client = LLMClient.from_config()

            # 指定配置文件
            client = LLMClient.from_config("path/to/config.yaml")

            # 默认配置文件 + 指定模型
            client = LLMClient.from_config(model="qwen-plus")

            # 指定配置文件 + 指定模型
            client = LLMClient.from_config("path/to/config.yaml", model="qwen-plus")
        """
        from ..cli.config import FlexLLMConfig, get_config

        cfg = FlexLLMConfig(config) if config else get_config()
        model_config = cfg.get_model_config(model)
        if not model_config:
            raise ValueError(
                f"未找到模型配置: {model or '(默认)'}，"
                "请检查 ~/.flexllm/config.yaml 或设置环境变量 FLEXLLM_BASE_URL"
            )

        # 模型 ID 为空时，自动从 /v1/models 获取（仅 openai provider）
        model_id = model_config.get("id")
        if not model_id and model_config.get("provider", "openai") == "openai":
            base_url = model_config.get("base_url")
            if base_url:
                from ..cli.utils import _fetch_model_id

                model_id = _fetch_model_id(base_url, model_config.get("api_key", "EMPTY"))

        # 构造 LLMClientPool 的参数
        init_kwargs = {
            "model": model_id,
            "base_url": model_config.get("base_url"),
            "api_key": model_config.get("api_key", "EMPTY"),
        }
        if "provider" in model_config:
            init_kwargs["provider"] = model_config["provider"]
        # 每个模型可单独配代理：CLI 走 from_config，没有这条就只能靠进程级环境变量，
        # 而"仅此 endpoint 需经网关"正是环境变量表达不了的场景
        if "proxy" in model_config:
            init_kwargs["proxy"] = model_config["proxy"]

        # overrides 覆盖配置值
        init_kwargs.update(overrides)

        instance = cls(**init_kwargs)

        # 设置配置中的 system prompt、user_template 和模型参数
        resolved_name = model or cfg.config.get("default")
        instance._config_system = cfg.get_system(resolved_name)
        instance._config_user_template = cfg.get_user_template(resolved_name)
        instance._config_params = cfg.get_model_params(resolved_name)

        return instance

    def _merge_config_params(self, kwargs: dict) -> dict:
        """合并 from_config() 的配置参数，用户显式传入的优先"""
        if self._config_params:
            return {**self._config_params, **kwargs}
        return kwargs

    def _prepare_messages(self, messages: str | list[dict]) -> list[dict]:
        """准备 messages：字符串转换 + user_template + system 注入

        支持字符串快捷方式：
            client.chat_completions("你好")
            等价于 client.chat_completions([{"role": "user", "content": "你好"}])
        """
        if isinstance(messages, str):
            content = messages
            if self._config_user_template:
                content = self._config_user_template.format(content=content)
            messages = [{"role": "user", "content": content}]

        # 注入 system prompt（messages 中没有 system 时）
        if self._config_system and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": self._config_system}] + messages

        return messages

    def _prepare_messages_batch(self, messages_list: list[str | list[dict]]) -> list[list[dict]]:
        """批量版本的 _prepare_messages"""
        return [self._prepare_messages(msgs) for msgs in messages_list]

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
        cache_dir: str | None = None,
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
        cache_dir: str | None,
        proxy: str | None,
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
            proxy=proxy,
            use_vertex_ai=use_vertex_ai,
            project_id=project_id,
            location=location,
            credentials=credentials,
            cache=cache,
            **kwargs,
        )
        self._inject_pool_limits(self._single_client)

    def _init_multi_mode(
        self,
        endpoints: list,
        fallback: bool,
        max_fallback_attempts: int,
        failure_threshold: float,
        recovery_time: float,
        concurrency_limit: int,
        max_qps: int,
        timeout: int,
        retry_times: int,
        cache_image: bool,
        cache_dir: str | None,
        proxy: str | None,
        cache: ResponseCacheConfig,
        **kwargs,
    ):
        """初始化多 endpoint 模式"""
        self._mode = "multi"
        self._fallback = fallback
        self._single_client = None
        self._provider = None
        self._model = None

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
            # endpoint 级 proxy 覆盖 pool 顶层默认
            ep_proxy = ep.proxy if ep.proxy is not None else proxy

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
                "proxy": ep_proxy,
                "cache": cache,
                **kwargs,
                **(ep.extra or {}),
            }
            # 直接创建底层客户端
            self._clients.append(self._create_base_client(**client_kwargs))

        # 创建路由器
        # concurrency_limit 直接取底层 client 的实际值（ground truth 在 client 上）
        provider_configs = [
            ProviderConfig(
                base_url=ep.base_url,
                api_key=ep.api_key,
                model=ep.model,
                concurrency_limit=client._concurrency_limit,
            )
            for ep, client in zip(self._endpoints, self._clients)
        ]
        self._router = ProviderRouter(
            providers=provider_configs,
            failure_threshold=failure_threshold,
            recovery_time=recovery_time,
        )

        # provider -> client 映射：以 ProviderConfig 对象身份为键。
        # 不能用 base_url 作键：相同 base_url 不同 api_key 的 endpoint 会坍缩成一个
        self._client_map = {id(pc): client for pc, client in zip(provider_configs, self._clients)}

        for client in self._clients:
            self._inject_pool_limits(client)

        self._max_fallback_attempts = max_fallback_attempts or len(self._clients)

    def _inject_pool_limits(self, client: LLMClientBase) -> None:
        """把 pool 级共享限制器注入底层客户端的 ConcurrentRequester

        统一用事后注入而非构造参数透传：新建 endpoint、单 endpoint、
        legacy clients= 三条路径共用同一机制。
        """
        if self._pool_limiter is None and self._pool_rate_limiter is None:
            return
        client._client._pool_semaphore = self._pool_limiter
        client._client._pool_rate_limiter = self._pool_rate_limiter

    def _acquire_client(
        self, tried: list[ProviderConfig]
    ) -> tuple[LLMClientBase, ProviderConfig] | tuple[None, None]:
        """容量感知地获取一个未尝试过的 client

        in-flight 计数 +1，调用方必须在请求结束后 release（try/finally 保证）。
        健康的 endpoint 都已尝试过时返回 (None, None)。
        """
        provider = self._router.acquire(exclude=tried)
        if provider is None:
            return None, None
        return self._client_map[id(provider)], provider

    async def chat_completions(
        self,
        messages: str | list[dict],
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
            与 LLMClient.chat_completions 返回值一致。
            请求失败时（所有可用 endpoint 都失败）与单模式一致：
            返回最后一次失败的 RequestResult（status="error"），不抛异常。
            仅当失败源于本地异常（非 HTTP 请求失败）时才向上抛出。
        """
        kwargs = self._merge_config_params(kwargs)
        messages = self._prepare_messages(messages)

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

        # 多 endpoint 模式：容量感知选路 + fallback
        last_error = None
        last_error_result = None  # 请求失败时的 RequestResult（区别于本地异常）
        tried_providers: list[ProviderConfig] = []

        for _ in range(self._max_fallback_attempts):
            client, provider = self._acquire_client(tried_providers)
            if provider is None:
                break  # 健康的 endpoint 都已尝试过

            tried_providers.append(provider)

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

                # 检查是否返回了 RequestResult（表示请求失败）
                if hasattr(result, "status") and result.status != "success":
                    last_error_result = result
                    self._router.mark_failed(provider)
                    logger.debug(f"Endpoint {provider.base_url} 请求失败: {result.data}")
                    if not self._fallback:
                        return result
                    continue  # 尝试下一个 endpoint

                self._router.mark_success(provider)
                return result

            except Exception as e:
                last_error = e
                self._router.mark_failed(provider)
                logger.debug(f"Endpoint {provider.base_url} 失败: {e}")

                if not self._fallback:
                    raise
            finally:
                self._router.release(provider)

        # 所有 endpoint 都失败：与单模式行为一致，请求失败返回 RequestResult 而非抛异常
        if last_error_result is not None:
            logger.warning("所有 endpoint 都失败了，返回最后一次失败的 RequestResult")
            return last_error_result
        raise last_error or RuntimeError("所有 endpoint 都失败了")

    def chat_completions_sync(
        self,
        messages: str | list[dict],
        model: str = None,
        return_raw: bool = False,
        return_usage: bool = False,
        **kwargs,
    ) -> Union[str, ChatCompletionResult, "RequestResult"]:
        """同步版本的聊天完成"""
        kwargs = self._merge_config_params(kwargs)
        messages = self._prepare_messages(messages)

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
        messages_list: list[str | list[dict]],
        model: str = None,
        return_raw: bool = False,
        return_usage: bool = False,
        show_progress: bool = True,
        return_summary: bool = False,
        return_cost_report: bool = False,
        track_cost: bool = False,
        preprocess_msg: bool = False,
        output_jsonl: str | None = None,
        flush_interval: float = 1.0,
        distribute: bool = True,
        metadata_list: list[dict] | None = None,
        save_input: bool | str = True,
        params_list: list[dict | None] | None = None,
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
            params_list: per-record 参数列表（与 messages_list 等长，dict 或 None），
                每条覆盖全局 kwargs 并参与缓存键；带 params 的行回显到输出。
            **kwargs: 其他参数

        Returns:
            与 LLMClient.chat_completions_batch 返回值一致
        """
        kwargs = self._merge_config_params(kwargs)
        messages_list = self._prepare_messages_batch(messages_list)

        # 单 endpoint 模式：直接调用底层客户端
        if self._mode == "single":
            return await self._single_client.chat_completions_batch(
                messages_list=messages_list,
                model=model,
                return_raw=return_raw,
                return_usage=return_usage,
                show_progress=show_progress,
                return_summary=return_summary,
                return_cost_report=return_cost_report,
                track_cost=track_cost,
                preprocess_msg=preprocess_msg,
                output_jsonl=output_jsonl,
                flush_interval=flush_interval,
                metadata_list=metadata_list,
                save_input=save_input,
                params_list=params_list,
                **kwargs,
            )

        # 多 endpoint 模式：参数校验
        # track_cost 需要 usage 信息
        if track_cost:
            return_usage = True

        validate_batch_params(messages_list, metadata_list, output_jsonl, params_list)

        if not distribute or len(self._clients) == 1:
            # 单 endpoint 模式：使用 fallback
            return await self._batch_with_fallback(
                messages_list=messages_list,
                model=model,
                return_raw=return_raw,
                return_usage=return_usage,
                show_progress=show_progress,
                return_summary=return_summary,
                return_cost_report=return_cost_report,
                track_cost=track_cost,
                preprocess_msg=preprocess_msg,
                output_jsonl=output_jsonl,
                flush_interval=flush_interval,
                metadata_list=metadata_list,
                save_input=save_input,
                params_list=params_list,
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
                return_cost_report=return_cost_report,
                track_cost=track_cost,
                preprocess_msg=preprocess_msg,
                output_jsonl=output_jsonl,
                flush_interval=flush_interval,
                metadata_list=metadata_list,
                save_input=save_input,
                params_list=params_list,
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
        return_cost_report: bool = False,
        track_cost: bool = False,
        preprocess_msg: bool = False,
        output_jsonl: str | None = None,
        flush_interval: float = 1.0,
        metadata_list: list[dict] | None = None,
        save_input: bool | str = True,
        params_list: list[dict | None] | None = None,
        **kwargs,
    ):
        """使用单个 endpoint + fallback 的批量调用"""
        last_error = None
        tried_providers: list[ProviderConfig] = []

        for _ in range(self._max_fallback_attempts):
            # 整批算 1 个 in-flight：计数不精确，但能让单条调用避开正在跑批的 endpoint
            client, provider = self._acquire_client(tried_providers)
            if provider is None:
                break  # 健康的 endpoint 都已尝试过

            tried_providers.append(provider)

            try:
                result = await client.chat_completions_batch(
                    messages_list=messages_list,
                    model=model or provider.model,
                    return_raw=return_raw,
                    return_usage=return_usage,
                    show_progress=show_progress,
                    return_summary=return_summary,
                    return_cost_report=return_cost_report,
                    track_cost=track_cost,
                    preprocess_msg=preprocess_msg,
                    output_jsonl=output_jsonl,
                    flush_interval=flush_interval,
                    metadata_list=metadata_list,
                    save_input=save_input,
                    params_list=params_list,
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
            finally:
                self._router.release(provider)

        raise last_error or RuntimeError("所有 endpoint 都失败了")

    async def _batch_distributed(
        self,
        messages_list: list[list[dict]],
        model: str = None,
        return_raw: bool = False,
        return_usage: bool = False,
        show_progress: bool = True,
        return_summary: bool = False,
        return_cost_report: bool = False,
        track_cost: bool = False,
        preprocess_msg: bool = False,
        output_jsonl: str | None = None,
        flush_interval: float = 1.0,
        metadata_list: list[dict] | None = None,
        save_input: bool | str = True,
        params_list: list[dict | None] | None = None,
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
        n = len(messages_list)
        results = [None] * n
        cached_count = 0
        start_time = time.perf_counter()

        # 消息预处理（图片/视频转 base64 等）：与单模式一致，在缓存查询之前做，
        # 保证缓存键基于预处理后的 messages
        messages_list = await self._clients[0]._preprocess_messages_batch(
            messages_list, preprocess_msg
        )

        # per-record 生成参数（剥除消息构造类键），用于覆盖 kwargs 与缓存键
        gen_params_list = build_gen_params_list(params_list)

        # fallback 判断以 endpoint（ProviderConfig 对象）为粒度：
        # 相同 base_url 不同 api_key 是不同 endpoint，不能用 base_url 集合
        num_endpoints = len(self._clients)

        # 获取响应缓存（如果有的话，使用第一个 client 的缓存）
        response_cache = None
        for client in self._clients:
            cache = getattr(client, "_response_cache", None)
            if cache is not None:
                response_cache = cache
                break

        # 使用 JsonlWriter 管理文件输出和断点续传（params_list 用于回显 per-record params）
        writer = JsonlWriter(
            output_jsonl,
            messages_list,
            save_input,
            metadata_list,
            flush_interval,
            params_list=params_list,
        )
        completed_indices = set(writer.completed_indices)

        # 恢复已完成的记录到 results（断点续传）。
        # 恢复项按当前调用的 return_usage 语义包装，保证返回列表类型一致
        # （文件里的 output 已含 prefix，usage 有则恢复，没有为 None）
        file_restored_count = len(completed_indices)
        if output_jsonl and completed_indices:
            from .batch_helpers import resume_from_jsonl

            _, records = resume_from_jsonl(output_jsonl, messages_list, save_input)
            for record in records:
                if return_usage and not return_raw:
                    results[record["index"]] = ChatCompletionResult(
                        content=record["output"], usage=record.get("usage")
                    )
                else:
                    results[record["index"]] = record["output"]

        # 检查缓存命中（如果启用了缓存）—— 批量查询，单次 IPC 往返
        # 多 endpoint 不同模型且未指定 model 时，跳过 pool 级缓存（键无法统一），
        # 由各 base client 的 chat_completions 各自处理缓存。
        # return_raw 跳过缓存（缓存只存提取后的 content，与 base client 行为一致）
        effective_model = model or self._endpoints[0].model
        all_same_model = model or len({ep.model for ep in self._endpoints}) == 1
        if response_cache is not None and all_same_model and not return_raw:
            # 过滤出未完成的 messages
            pending = [
                (idx, msg) for idx, msg in enumerate(messages_list) if idx not in completed_indices
            ]
            if pending:
                pending_indices, pending_msgs = zip(*pending)
                pending_gen_params = (
                    [gen_params_list[i] for i in pending_indices] if gen_params_list else None
                )
                cached_responses, _ = response_cache.get_batch(
                    list(pending_msgs),
                    model=effective_model,
                    params_list=pending_gen_params,
                    **kwargs,
                )
                for idx, cached_result in zip(pending_indices, cached_responses):
                    if cached_result is not None:
                        # 缓存存的 content 不含 prefill 前缀，与 base client 一致在此拼回
                        content = cached_result["content"]
                        prefix = LLMClientBase._trailing_assistant_prefix(messages_list[idx])
                        if prefix and content is not None:
                            content = prefix + content
                        if return_usage:
                            results[idx] = ChatCompletionResult(
                                content=content,
                                usage=cached_result.get("usage"),
                            )
                        else:
                            results[idx] = content
                        completed_indices.add(idx)
                        cached_count += 1
                        # 写入输出文件（断点续传）
                        writer.write_result(
                            idx,
                            content,
                            usage=cached_result.get("usage"),
                        )
            if cached_count > 0:
                logger.info(f"缓存命中: {cached_count}/{n}")

        # 共享任务队列（跳过已完成的）
        queue = asyncio.Queue()
        for idx, msg in enumerate(messages_list):
            if idx not in completed_indices:
                queue.put_nowait((idx, msg, set()))

        pending_count = queue.qsize()
        if pending_count == 0:
            logger.info("所有任务已完成，无需执行")
            writer.close()
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
                config=progress_config,
                model_name=first_model if track_cost else None,
                input_price_per_1m=input_price,
                output_price_per_1m=output_price,
            )
            if show_progress
            else None
        )

        # 用于统计和线程安全更新
        lock = asyncio.Lock()
        active_tasks = 0
        all_done = asyncio.Event()

        async def worker(client_idx: int):
            """单个 worker：循环从队列取任务并执行，支持 fallback 重试"""
            nonlocal active_tasks

            client = self._clients[client_idx]
            provider = self._router._providers[client_idx].config
            my_endpoint = provider.base_url
            worker_model = model or provider.model

            while not all_done.is_set():
                # claim(取任务)与 active_tasks 自增必须在同一把锁内原子完成：
                # 否则"取走末个任务但尚未计数"的窗口会被其他 worker 的空队列检查
                # 误判为全部完成 → all_done 提前置位 → fallback 重新入队的任务
                # 再无消费者而永久丢失（results[idx] 恒 None 且无错误记录）。
                async with lock:
                    try:
                        idx, msg, tried_endpoints = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        if active_tasks == 0:
                            all_done.set()
                            break
                        idx = None
                    else:
                        active_tasks += 1
                if idx is None:
                    await asyncio.sleep(0.05)
                    continue

                # 如果已尝试过当前 endpoint，放回队列让其他 worker 处理
                if my_endpoint in tried_endpoints:
                    if len(tried_endpoints) >= num_endpoints:
                        # 所有 endpoint 都失败了
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
                            writer.write_result(
                                idx,
                                None,
                                "error",
                                f"All {num_endpoints} endpoints failed",
                            )
                        continue
                    await queue.put((idx, msg, tried_endpoints))
                    async with lock:
                        active_tasks -= 1
                    await asyncio.sleep(0.01)
                    continue

                task_start = time.perf_counter()
                try:
                    if tracker:
                        retry_callback.set(tracker.increment_retry)
                    row_extra = gen_params_list[idx] if gen_params_list else None
                    # 内部强制 return_usage=True 以拿到 queue_time（return_raw 时
                    # RequestResult 本身就带），返回前按用户要求解包
                    result = await client.chat_completions(
                        messages=msg,
                        model=worker_model,
                        return_raw=return_raw,
                        return_usage=return_usage or not return_raw,
                        **({**kwargs, **row_extra} if row_extra else kwargs),
                    )

                    # 检查是否返回了 RequestResult（表示失败）
                    if hasattr(result, "status") and result.status != "success":
                        error_type = "unknown"
                        error_detail = ""
                        if hasattr(result, "data") and isinstance(result.data, dict):
                            error_type = result.data.get("error", "unknown")
                            error_detail = result.data.get("detail", "")
                        error_msg = f"{error_type}: {error_detail}" if error_detail else error_type
                        raise RuntimeError(error_msg)

                    latency = time.perf_counter() - task_start
                    queue_time = getattr(result, "queue_time", None)
                    if not return_usage and not return_raw and hasattr(result, "content"):
                        result = result.content  # 用户没要 usage，解包回 str
                    results[idx] = result
                    self._router.mark_success(provider)

                    # 缓存写入由 base client 的 chat_completions 内部处理，
                    # 无需在 pool 层重复写入

                    async with lock:
                        active_tasks -= 1
                        if tracker:
                            req_result = RequestResult(
                                request_id=idx,
                                data=result,
                                status="success",
                                latency=latency,
                                queue_time=queue_time or 0.0,
                            )
                            tracker.update(req_result)

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
                        if hasattr(result, "content"):
                            output_content = result.content
                            output_usage = getattr(result, "usage", None)
                        else:
                            output_content = result
                            output_usage = None
                        writer.write_result(idx, output_content, usage=output_usage)

                except Exception as e:
                    latency = time.perf_counter() - task_start
                    self._router.mark_failed(provider)

                    tried_endpoints = tried_endpoints | {my_endpoint}

                    if self._fallback and len(tried_endpoints) < num_endpoints:
                        await queue.put((idx, msg, tried_endpoints))
                        async with lock:
                            active_tasks -= 1
                            if tracker:
                                tracker.increment_retry()
                    else:
                        results[idx] = None
                        async with lock:
                            active_tasks -= 1
                            if tracker:
                                req_result = RequestResult(
                                    request_id=idx,
                                    data={"error": str(e)},
                                    status="error",
                                    latency=latency,
                                )
                                tracker.update(req_result)
                            writer.write_result(idx, None, "error", str(e))

        try:
            workers = []
            for client_idx, client in enumerate(self._clients):
                concurrency = getattr(client, "_concurrency_limit", 10)
                for _ in range(concurrency):
                    workers.append(worker(client_idx))
            await asyncio.gather(*workers)

        finally:
            writer.close()
            if tracker:
                tracker.summary(print_to_console=True)

        if return_summary:
            total_cached = cached_count + file_restored_count
            summary = {
                "total": n,
                "success": (tracker.success_count if tracker else 0) + total_cached,
                "failed": tracker.error_count if tracker else 0,
                "cached": total_cached,
                "elapsed": time.perf_counter() - start_time,
            }
            return results, summary

        return results

    def chat_completions_batch_sync(
        self,
        messages_list: list[str | list[dict]],
        model: str = None,
        return_raw: bool = False,
        return_usage: bool = False,
        show_progress: bool = True,
        return_summary: bool = False,
        return_cost_report: bool = False,
        track_cost: bool = False,
        output_jsonl: str | None = None,
        flush_interval: float = 1.0,
        distribute: bool = True,
        metadata_list: list[dict] | None = None,
        save_input: bool | str = True,
        **kwargs,
    ) -> list[str] | list[ChatCompletionResult] | tuple:
        """同步版本的批量聊天完成"""
        kwargs = self._merge_config_params(kwargs)
        messages_list = self._prepare_messages_batch(messages_list)

        # 单 endpoint 模式：使用底层客户端的 sync 方法
        if self._mode == "single":
            return self._single_client.chat_completions_batch_sync(
                messages_list=messages_list,
                model=model,
                return_raw=return_raw,
                return_usage=return_usage,
                show_progress=show_progress,
                return_summary=return_summary,
                return_cost_report=return_cost_report,
                track_cost=track_cost,
                output_jsonl=output_jsonl,
                flush_interval=flush_interval,
                metadata_list=metadata_list,
                save_input=save_input,
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
                return_cost_report=return_cost_report,
                track_cost=track_cost,
                output_jsonl=output_jsonl,
                flush_interval=flush_interval,
                distribute=distribute,
                metadata_list=metadata_list,
                save_input=save_input,
                **kwargs,
            )
        )

    async def chat_completions_stream(
        self,
        messages: str | list[dict],
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
        kwargs = self._merge_config_params(kwargs)
        messages = self._prepare_messages(messages)

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

        # 多 endpoint 模式：容量感知选路 + fallback
        # 流式不经过 ConcurrentRequester、不占 endpoint semaphore，但占服务端资源，
        # 因此计入 in-flight（覆盖整个流的生命周期）
        last_error = None
        tried_providers: list[ProviderConfig] = []

        for _ in range(self._max_fallback_attempts):
            client, provider = self._acquire_client(tried_providers)
            if provider is None:
                break  # 健康的 endpoint 都已尝试过

            tried_providers.append(provider)

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
            finally:
                self._router.release(provider)

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
        save_input: bool | str = True,
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
            save_input: 控制输出 JSONL 中 input 字段的保存策略（同 chat_completions_batch）
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
                save_input=save_input,
                **kwargs,
            ):
                yield result
            return

        # 多 endpoint 模式：分布式迭代，边完成边 yield
        from types import SimpleNamespace

        n = len(messages_list)
        all_endpoints_set = {ep.base_url for ep in self._endpoints}
        num_endpoints = len(all_endpoints_set)

        # 结果队列：worker 完成一条就 put 一条，主循环 get 并 yield
        result_queue: asyncio.Queue = asyncio.Queue()

        # 共享任务队列
        task_queue: asyncio.Queue = asyncio.Queue()
        for idx, msg in enumerate(messages_list):
            task_queue.put_nowait((idx, msg, set()))

        total_pending = n
        active_tasks = 0
        lock = asyncio.Lock()
        all_done = asyncio.Event()

        async def worker(client_idx: int):
            nonlocal active_tasks

            client = self._clients[client_idx]
            provider = self._router._providers[client_idx].config
            my_endpoint = provider.base_url
            worker_model = model or provider.model

            while not all_done.is_set():
                # claim(取任务)与 active_tasks 自增在同一把锁内原子完成，
                # 杜绝末个任务被取走但尚未计数的窗口被误判为全部完成而丢任务
                # （详见 _batch_distributed 中同款修复）。
                async with lock:
                    try:
                        idx, msg, tried = task_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        if active_tasks == 0:
                            all_done.set()
                            break
                        idx = None
                    else:
                        active_tasks += 1
                if idx is None:
                    await asyncio.sleep(0.05)
                    continue

                if my_endpoint in tried:
                    if len(tried) >= num_endpoints:
                        # 所有 endpoint 都失败
                        await result_queue.put(
                            SimpleNamespace(
                                content=None,
                                error="All endpoints failed",
                                original_idx=idx,
                                latency=0.0,
                                queue_time=None,
                                status="error",
                                data=None,
                                usage=None,
                                summary=None,
                            )
                        )
                        async with lock:
                            active_tasks -= 1
                        continue
                    await task_queue.put((idx, msg, tried))
                    async with lock:
                        active_tasks -= 1
                    await asyncio.sleep(0.01)
                    continue

                task_start = time.perf_counter()
                try:
                    # 内部强制 return_usage=True 以拿到 queue_time（return_raw 时
                    # RequestResult 本身就带），返回前按用户要求解包
                    result = await client.chat_completions(
                        messages=msg,
                        model=worker_model,
                        return_raw=return_raw,
                        return_usage=return_usage or not return_raw,
                        **kwargs,
                    )

                    if hasattr(result, "status") and result.status != "success":
                        error_detail = ""
                        if hasattr(result, "data") and isinstance(result.data, dict):
                            error_detail = result.data.get("error", "unknown")
                        raise RuntimeError(error_detail or "unknown error")

                    latency = time.perf_counter() - task_start
                    queue_time = getattr(result, "queue_time", None)
                    self._router.mark_success(provider)

                    # 提取 content；用户没要 usage 时把 data 解包回 str，保持返回形状不变
                    if hasattr(result, "content"):
                        content = result.content
                        usage = getattr(result, "usage", None) if return_usage else None
                        if not return_usage and not return_raw:
                            result = content
                    else:
                        content = result
                        usage = None

                    await result_queue.put(
                        SimpleNamespace(
                            content=content,
                            error=None,
                            original_idx=idx,
                            latency=latency,
                            queue_time=queue_time,
                            status="success",
                            data=result,
                            usage=usage,
                            summary=None,
                        )
                    )
                    async with lock:
                        active_tasks -= 1

                except Exception as e:
                    latency = time.perf_counter() - task_start
                    self._router.mark_failed(provider)
                    tried = tried | {my_endpoint}

                    if self._fallback and len(tried) < num_endpoints:
                        await task_queue.put((idx, msg, tried))
                        async with lock:
                            active_tasks -= 1
                    else:
                        await result_queue.put(
                            SimpleNamespace(
                                content=None,
                                error=str(e),
                                original_idx=idx,
                                latency=latency,
                                queue_time=None,
                                status="error",
                                data=None,
                                usage=None,
                                summary=None,
                            )
                        )
                        async with lock:
                            active_tasks -= 1

        # 启动所有 worker
        workers = []
        for client_idx, client in enumerate(self._clients):
            concurrency = getattr(client, "_concurrency_limit", 10)
            for _ in range(concurrency):
                workers.append(asyncio.ensure_future(worker(client_idx)))

        # 边完成边 yield
        yielded = 0
        success_count = 0
        start_time = time.perf_counter()
        total_latency = 0.0

        try:
            while yielded < total_pending:
                try:
                    result = await asyncio.wait_for(result_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if all_done.is_set() and result_queue.empty():
                        break
                    continue

                yielded += 1
                if result.status == "success":
                    success_count += 1
                    total_latency += result.latency

                # 最后一条附带 summary
                if yielded == total_pending:
                    elapsed = time.perf_counter() - start_time
                    result.summary = {
                        "total": n,
                        "success": success_count,
                        "failed": n - success_count,
                        "cached": 0,
                        "elapsed": elapsed,
                        "avg_latency": total_latency / success_count if success_count else 0,
                    }

                yield result

        finally:
            all_done.set()
            await asyncio.gather(*workers, return_exceptions=True)

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
            return f"LLMClientPool(endpoints={len(self._clients)}, fallback={self._fallback})"

    def __getattr__(self, name):
        """自动委托未显式定义的方法给底层客户端（仅单模式）"""
        if self._mode == "single":
            return getattr(self._single_client, name)
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}' "
                f"(仅单 endpoint 模式支持自动委托)"
            )
