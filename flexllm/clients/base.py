"""
LLMClientBase - LLM 客户端抽象基类

提供通用的方法实现，子类只需实现核心的差异化方法。
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

logger = logging.getLogger(__name__)

from ..async_api import ConcurrentRequester, validate_proxy
from ..async_api.progress import ProgressBarConfig
from ..cache import ResponseCache, ResponseCacheConfig
from ..msg_processors.image_processor import ImageCacheConfig
from ..msg_processors.unified_processor import batch_process_messages as optimized_batch_preprocess
from ..msg_processors.unified_processor import unified_messages_preprocess
from ..pricing import estimate_cost, get_model_pricing
from ..pricing.cost_tracker import BudgetExceededError, CostReport, CostTracker, CostTrackerConfig
from .batch_helpers import (
    JsonlWriter,
    build_gen_params_list,
    compact_output_file,
    extract_save_input,
    resume_from_jsonl,
    validate_batch_params,
)

if TYPE_CHECKING:
    from ..async_api.interface import RequestResult


# 向后兼容的别名（pool.py 和 test_resume_jsonl.py 都 import 这些）
_extract_save_input = extract_save_input
_resume_from_jsonl = resume_from_jsonl


@dataclass
class ToolCall:
    """工具调用信息"""

    id: str
    type: str  # "function"
    function: dict  # {"name": "...", "arguments": "..."}


@dataclass
class ChatCompletionResult:
    """聊天完成的结果，包含内容和 token 用量信息"""

    content: str
    usage: dict | None = None  # {"prompt_tokens": x, "completion_tokens": y, "total_tokens": z}
    reasoning_content: str | None = None  # 思考内容（DeepSeek-R1、Qwen3 等）
    tool_calls: list["ToolCall"] | None = None  # 工具调用列表


@dataclass
class BatchResultItem:
    """批量请求中单条结果，包含索引、内容和 usage"""

    index: int
    content: str | None
    usage: dict | None = None
    status: str = "success"  # success, error, cached
    error: str | None = None
    latency: float = 0.0


class LLMClientBase(ABC):
    """
    LLM 客户端抽象基类

    子类只需实现 4 个核心方法：
    - _get_url(model, stream) -> str
    - _get_headers() -> dict
    - _build_request_body(messages, model, **kwargs) -> dict
    - _extract_content(response_data) -> str

    可选覆盖：
    - _extract_stream_content(data) -> str
    - _get_stream_url(model) -> str
    """

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model: str = None,
        concurrency_limit: int = 10,
        max_qps: int = 1000,
        timeout: int = 120,
        retry_times: int = 3,
        retry_delay: float = 1.0,
        cache_image: bool = False,
        cache_dir: str = "image_cache",
        cache: bool | ResponseCacheConfig | None = None,
        cost_tracker: bool | CostTrackerConfig | None = None,
        proxy: str | None = None,
        **kwargs,
    ):
        """
        Args:
            base_url: API 基础 URL
            api_key: API 密钥
            model: 默认模型名称
            concurrency_limit: 并发请求数限制
            max_qps: 最大 QPS
            timeout: 请求超时时间（秒）
            retry_times: 重试次数
            retry_delay: 重试延迟（秒）
            cache_image: 是否缓存图片
            cache_dir: 图片缓存目录
            proxy: 正向代理 URL，形如 http://gateway:8080 或
                   http://user:pass@gateway:8080（仅支持 http(s)://，不支持 SOCKS）。
                   用于目标 base_url 仅经某网关可达的场景。不传时仍会沿用
                   HTTP_PROXY/HTTPS_PROXY/NO_PROXY 环境变量（trust_env=True）。
            cache: 响应缓存配置
                   - True: 启用缓存（默认 24小时 TTL）
                   - False/None: 禁用缓存（默认）
                   - ResponseCacheConfig: 自定义配置
            cost_tracker: 成本追踪配置
                   - True: 启用成本追踪（仅追踪，不限预算）
                   - False/None: 禁用成本追踪（默认）
                   - CostTrackerConfig: 自定义配置（含预算控制）
        """
        self._base_url = base_url.rstrip("/") if base_url else None
        self._api_key = api_key
        self._model = model
        self._concurrency_limit = concurrency_limit
        self._timeout = timeout
        self._proxy = validate_proxy(proxy)

        self._client = ConcurrentRequester(
            concurrency_limit=concurrency_limit,
            max_qps=max_qps,
            timeout=timeout,
            retry_times=retry_times,
            retry_delay=retry_delay,
            proxy=proxy,
        )

        self._cache_config = ImageCacheConfig(
            enabled=cache_image,
            cache_dir=cache_dir,
            force_refresh=False,
            retry_failed=False,
        )

        # 响应缓存
        if cache is True:
            cache = ResponseCacheConfig.with_ttl()
        elif cache is None or cache is False:
            cache = ResponseCacheConfig.disabled()
        self._response_cache = ResponseCache(cache) if cache.enabled else None

        # 成本追踪
        if cost_tracker is True:
            cost_tracker = CostTrackerConfig.tracking_only()
        elif cost_tracker is None or cost_tracker is False:
            cost_tracker = CostTrackerConfig.disabled()
        self._cost_tracker = CostTracker(cost_tracker) if cost_tracker.enabled else None

    # ========== 核心抽象方法（子类必须实现）==========

    @abstractmethod
    def _get_url(self, model: str, stream: bool = False) -> str: ...

    @abstractmethod
    def _get_headers(self) -> dict: ...

    @abstractmethod
    def _build_request_body(
        self, messages: list[dict], model: str, stream: bool = False, **kwargs
    ) -> dict: ...

    @abstractmethod
    def _extract_content(self, response_data: dict) -> str | None: ...

    def _extract_usage(self, response_data: dict) -> dict | None:
        """提取 usage 信息（子类可覆盖）"""
        if not response_data:
            return None
        return response_data.get("usage")

    def _extract_tool_calls(self, response_data: dict) -> list[ToolCall] | None:
        """提取工具调用信息（子类可覆盖）"""
        return None

    # ========== 可选覆盖的钩子方法 ==========

    def _extract_stream_content(self, data: dict) -> str | None:
        return self._extract_content(data)

    def _extract_stream_thinking(self, data: dict) -> str | None:
        """从流式响应中提取思考内容，子类按需重写"""
        return None

    def _extract_stream_tool_calls(self, data: dict) -> list[dict] | None:
        """从流式 chunk 中提取 tool_call delta，子类可覆盖

        Returns:
            tool_call delta 列表或 None
            每个 delta 格式: {"index": 0, "id": "...", "type": "function",
                             "function": {"name": "...", "arguments": "..."}}
        """
        return None

    def _extract_stream_usage(self, data: dict) -> dict | None:
        """从流式 chunk 中提取 usage 信息，子类可覆盖

        Returns:
            usage dict 或 None（表示此 chunk 不含 usage）
        """
        if "usage" in data and data["usage"]:
            return data["usage"]
        return None

    def _prepare_stream_body(self, body: dict, return_usage: bool) -> dict:
        """流式请求体的额外处理，子类可覆盖

        OpenAI 格式需要添加 stream_options，其他 API 不需要。
        """
        if return_usage:
            body["stream_options"] = {"include_usage": True}
        return body

    def _get_stream_url(self, model: str) -> str:
        return self._get_url(model, stream=True)

    # ========== 通用工具方法 ==========

    def _get_effective_model(self, model: str = None) -> str:
        effective_model = model or self._model
        if not effective_model:
            raise ValueError("必须提供 model 参数或在初始化时指定 model")
        return effective_model

    @staticmethod
    def _trailing_assistant_prefix(messages: list[dict]) -> str | None:
        """若 messages 末尾是 assistant message 且 content 为字符串,返回该 content,否则 None。

        用于识别 prefill 场景下需要拼回输出的前缀。
        """
        if not messages:
            return None
        last = messages[-1]
        if last.get("role") != "assistant":
            return None
        content = last.get("content")
        return content if isinstance(content, str) else None

    async def _preprocess_messages(
        self, messages: list[dict], preprocess_msg: bool = False
    ) -> list[dict]:
        """消息预处理（图片/视频/音频转 base64 等）"""
        if preprocess_msg:
            return await unified_messages_preprocess(messages)
        return messages

    async def _preprocess_messages_batch(
        self, messages_list: list[list[dict]], preprocess_msg: bool = False
    ) -> list[list[dict]]:
        """批量消息预处理"""
        if preprocess_msg:
            return await optimized_batch_preprocess(
                messages_list,
                max_concurrent=self._concurrency_limit,
                cache_config=self._cache_config,
            )
        return messages_list

    # ========== 通用接口实现 ==========

    async def chat_completions(
        self,
        messages: list[dict],
        model: str = None,
        return_raw: bool = False,
        return_usage: bool = False,
        show_progress: bool = False,
        preprocess_msg: bool = False,
        url: str = None,
        prefix: str | None = None,
        include_prefix: bool = True,
        **kwargs,
    ) -> Union[str, ChatCompletionResult, "RequestResult"]:
        """
        单条聊天完成

        Args:
            messages: 消息列表
            model: 模型名称
            return_raw: 是否返回原始响应（RequestResult）
            return_usage: 是否返回包含 usage 的结果（ChatCompletionResult）
            show_progress: 是否显示进度条
            preprocess_msg: 是否预处理消息
            url: 自定义请求 URL，默认使用 _get_url() 生成
            prefix: 预设回复开头(prefill),仅 OpenAI 兼容客户端(含 vLLM/Ollama 等)实际生效;
                若传入则自动追加为末尾 assistant message,等同于直接在 messages 末尾加 assistant。
            include_prefix: prefill 场景下返回值是否拼接 prefix。
                默认 True,返回 "prefix + 续写";
                设为 False 仅返回模型续写部分(与底层 API 原始行为一致)。

        Returns:
            - return_raw=True: RequestResult 原始响应
            - return_usage=True: ChatCompletionResult(content, usage, reasoning_content)
            - 默认: str 内容文本
            - 请求失败时: 返回 RequestResult（status="error"），不会抛异常。
              如果需要失败时抛异常，请使用 chat_completions_or_raise()。

        Note:
            缓存由初始化时的 cache 参数控制，return_raw 时自动跳过缓存。
            缓存只存模型续写部分(不含 prefix),拼接由返回路径完成。
        """
        effective_model = self._get_effective_model(model)
        messages = await self._preprocess_messages(messages, preprocess_msg)

        # prefix 显式参数等价于在 messages 末尾追加 assistant message
        if prefix is not None:
            messages = list(messages) + [{"role": "assistant", "content": prefix}]

        # 拼接用前缀(仅在 include_prefix 时生效);return_raw 路径不参与拼接
        effective_prefix = (
            self._trailing_assistant_prefix(messages) if include_prefix and not return_raw else None
        )

        # 检查缓存
        use_cache = self._response_cache is not None and not return_raw
        if use_cache:
            cached = self._response_cache.get(messages, model=effective_model, **kwargs)
            if cached is not None:
                cached_content = cached["content"]
                if effective_prefix and cached_content is not None:
                    cached_content = effective_prefix + cached_content
                if return_usage:
                    return ChatCompletionResult(
                        content=cached_content,
                        usage=cached.get("usage"),
                    )
                return cached_content

        body = self._build_request_body(messages, effective_model, stream=False, **kwargs)
        request_params = {"json": body, "headers": self._get_headers()}
        effective_url = url or self._get_url(effective_model, stream=False)

        results, _ = await self._client.process_requests(
            request_params=[request_params],
            url=effective_url,
            method="POST",
            show_progress=show_progress,
        )

        data = results[0]
        if return_raw:
            return data
        if data.status == "success":
            content = self._extract_content(data.data)
            usage = self._extract_usage(data.data)

            # 写入缓存（始终存储 usage; content 不含 prefix）
            if use_cache and content is not None:
                self._response_cache.set(
                    messages, {"content": content, "usage": usage}, model=effective_model, **kwargs
                )

            if effective_prefix and content is not None:
                content = effective_prefix + content

            if return_usage:
                tool_calls = self._extract_tool_calls(data.data)
                if self._cost_tracker:
                    self._cost_tracker.record(usage, effective_model)
                return ChatCompletionResult(content=content, usage=usage, tool_calls=tool_calls)
            return content
        logger.warning("chat_completions 请求失败: %s, 返回 RequestResult 而非 str", data.data)
        return data

    async def chat_completions_or_raise(
        self,
        messages: list[dict],
        model: str = None,
        return_usage: bool = False,
        **kwargs,
    ) -> Union[str, ChatCompletionResult]:
        """
        单条聊天完成（失败时抛异常）

        与 chat_completions() 行为相同，但请求失败时抛出异常而非返回 RequestResult。

        Raises:
            RuntimeError: 请求失败时，包含错误信息
        """
        result = await self.chat_completions(
            messages=messages,
            model=model,
            return_usage=return_usage,
            **kwargs,
        )
        # 导入放在这里避免循环引用
        from ..async_api.interface import RequestResult

        if isinstance(result, RequestResult):
            raise RuntimeError(f"LLM 请求失败: status={result.status}, data={result.data}")
        return result

    def chat_completions_sync(
        self,
        messages: list[dict],
        model: str = None,
        return_raw: bool = False,
        return_usage: bool = False,
        **kwargs,
    ) -> Union[str, ChatCompletionResult, "RequestResult"]:
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
        return_cost_report: bool = False,
        track_cost: bool = False,
        preprocess_msg: bool = False,
        output_jsonl: str | None = None,
        flush_interval: float = 1.0,
        metadata_list: list[dict] | None = None,
        url: str = None,
        save_input: bool | str = True,
        include_prefix: bool = True,
        params_list: list[dict | None] | None = None,
        **kwargs,
    ) -> list[str] | list[ChatCompletionResult] | tuple:
        """
        批量聊天完成（支持断点续传）

        Args:
            messages_list: 消息列表
            model: 模型名称
            return_raw: 是否返回原始响应
            return_usage: 是否返回包含 usage 的结果（ChatCompletionResult 列表）
            show_progress: 是否显示进度条
            return_summary: 是否返回执行摘要
            return_cost_report: 是否返回成本报告（需要启用 cost_tracker）
            track_cost: 是否在进度条中显示实时成本（自动启用 return_usage）
            preprocess_msg: 是否预处理消息
            output_jsonl: 输出文件路径（JSONL 格式），用于持久化保存结果
            flush_interval: 文件刷新间隔（秒），默认 1 秒
            metadata_list: 元数据列表，与 messages_list 等长，每个元素保存到对应输出记录
            url: 自定义请求 URL，默认使用 _get_url() 生成
            save_input: 控制输出 JSONL 中 input 字段的保存策略
                - True（默认）: 保存完整 messages，断点续传时做首尾 input 校验
                - "last": 仅保存最后一个 user message 的 content
                - False: 不保存 input 字段，断点续传仅基于 index 恢复
            params_list: per-record 参数列表（与 messages_list 等长，元素为 dict 或 None）。
                每条的有效请求参数 = {**全局kwargs, **params_list[i]}，覆盖全局 kwargs，
                并参与各自的缓存键计算。带 params 的行会把 params 原样回显到输出 JSONL。
                注：断点续传校验只看 messages，不感知 params 变化（与改 kwargs 不影响续传一致）。

        Returns:
            - return_usage=True: List[ChatCompletionResult] 或 (List[ChatCompletionResult], summary)
            - return_cost_report=True: 返回元组 (results, cost_report)
            - 默认: List[str] 或 (List[str], summary)

        Note:
            缓存由初始化时的 cache 参数控制。
            切换 save_input 模式可能导致断点续传校验失败，这是预期行为。

        Raises:
            BudgetExceededError: 当超过预算硬限制时（由 cost_tracker 配置）
        """
        # track_cost 需要 usage 信息
        if track_cost:
            return_usage = True
        effective_model = self._get_effective_model(model)
        effective_url = url or self._get_url(effective_model, stream=False)
        headers = self._get_headers()

        validate_batch_params(messages_list, metadata_list, output_jsonl, params_list)
        messages_list = await self._preprocess_messages_batch(messages_list, preprocess_msg)

        # per-record 生成参数：剥除消息构造类键（system/user_template，已在上层消费），
        # 其余作为该行覆盖全局 kwargs 的生成参数，并参与各自缓存键。
        gen_params_list = build_gen_params_list(params_list)

        def merged_kwargs(idx: int) -> dict:
            """该行有效请求参数 = 全局 kwargs 叠加 per-record 生成参数"""
            extra = gen_params_list[idx] if gen_params_list else None
            return {**kwargs, **extra} if extra else kwargs

        use_cache = self._response_cache is not None

        # Prefill 拼接前缀:逐条样本预计算,缓存与底层返回均不含 prefix,只在输出与返回处拼回
        prefix_list = (
            [self._trailing_assistant_prefix(m) for m in messages_list]
            if include_prefix
            else [None] * len(messages_list)
        )

        def with_prefix(idx: int, content):
            """若该样本有 prefix 且 content 非空,返回拼接后的内容"""
            p = prefix_list[idx]
            if p and content is not None:
                return p + content
            return content

        def extractor(result):
            """提取 content 和 usage（用于缓存存储, content 不含 prefix）"""
            content = self._extract_content(result.data)
            usage = self._extract_usage(result.data)
            return {"content": content, "usage": usage}

        def to_chat_result(extracted, idx: int):
            """转换为 ChatCompletionResult, content 自动拼接 prefix"""
            return ChatCompletionResult(
                content=with_prefix(idx, extracted["content"]),
                usage=extracted.get("usage"),
                tool_calls=None,  # 缓存不存储 tool_calls
            )

        # 进度条配置（支持成本显示）
        progress_config = ProgressBarConfig(show_cost=track_cost) if show_progress else None

        # 提前获取定价信息（用于双行进度条显示）
        pricing = get_model_pricing(effective_model) if track_cost else None
        input_price = pricing["input"] * 1e6 if pricing else None
        output_price = pricing["output"] * 1e6 if pricing else None

        # 使用 JsonlWriter 管理文件输出（params_list 用于带 params 的行回显 params）
        writer = JsonlWriter(
            output_jsonl,
            messages_list,
            save_input,
            metadata_list,
            flush_interval,
            params_list=params_list,
        )
        completed_indices = writer.completed_indices

        try:
            # 计算实际需要执行的索引（排除文件中已完成的）
            if completed_indices:
                logger.info(f"从文件恢复跳过: {len(completed_indices)}/{len(messages_list)}")

            # 带缓存执行
            if use_cache and self._response_cache:
                # 查询缓存（传递 kwargs + per-record params 以确保不同参数使用不同缓存键）
                cached_responses, uncached_indices = self._response_cache.get_batch(
                    messages_list, model=effective_model, params_list=gen_params_list, **kwargs
                )

                # 将缓存命中的写入文件（如果文件中没有）
                for i, resp in enumerate(cached_responses):
                    if resp is not None and i not in completed_indices:
                        writer.write_result(
                            i, with_prefix(i, resp["content"]), usage=resp.get("usage")
                        )

                # 过滤掉文件中已完成的
                actual_uncached = [i for i in uncached_indices if i not in completed_indices]
                cache_hit_count = len(messages_list) - len(uncached_indices)

                progress = None
                if cache_hit_count > 0:
                    logger.info(f"缓存命中: {cache_hit_count}/{len(messages_list)}")
                if actual_uncached:
                    logger.info(f"待执行: {len(actual_uncached)}/{len(messages_list)}")

                    request_params = [
                        {
                            "json": self._build_request_body(
                                messages_list[i], effective_model, **merged_kwargs(i)
                            ),
                            "headers": headers,
                        }
                        for i in actual_uncached
                    ]

                    async for batch in self._client.aiter_stream_requests(
                        request_params=request_params,
                        url=effective_url,
                        method="POST",
                        show_progress=show_progress,
                        total_requests=len(actual_uncached),
                        progress_config=progress_config,
                        model_name=effective_model if track_cost else None,
                        input_price_per_1m=input_price,
                        output_price_per_1m=output_price,
                    ):
                        for result in batch.completed_requests:
                            original_idx = actual_uncached[result.request_id]
                            if result.status != "success":
                                error_msg = (
                                    result.data.get("error", "Unknown error")
                                    if isinstance(result.data, dict)
                                    else str(result.data)
                                )
                                logger.debug(f"请求失败: {error_msg}")
                                cached_responses[original_idx] = None
                                writer.write_result(original_idx, None, "error", error_msg)
                                continue
                            try:
                                extracted = extractor(result)
                                cached_responses[original_idx] = extracted
                                # 写入缓存（per-record 参数纳入键，与 get_batch 对齐）
                                self._response_cache.set(
                                    messages_list[original_idx],
                                    extracted,
                                    model=effective_model,
                                    **merged_kwargs(original_idx),
                                )
                                # 文件输出
                                writer.write_result(
                                    original_idx,
                                    with_prefix(original_idx, extracted["content"]),
                                    usage=extracted.get("usage"),
                                )
                                # 记录成本
                                if self._cost_tracker and extracted.get("usage"):
                                    self._cost_tracker.record(extracted["usage"], effective_model)
                                # 更新进度条的成本显示
                                if track_cost and batch.progress and extracted.get("usage"):
                                    usage = extracted["usage"]
                                    input_tokens = usage.get("prompt_tokens", 0)
                                    output_tokens = usage.get("completion_tokens", 0)
                                    cost = estimate_cost(
                                        input_tokens, output_tokens, effective_model
                                    )
                                    batch.progress.update_cost(input_tokens, output_tokens, cost)
                            except BudgetExceededError:
                                logger.warning("预算超限，停止批量处理")
                                raise
                            except Exception as e:
                                logger.warning(f"提取结果失败: {e}")
                                cached_responses[original_idx] = None
                                writer.write_result(original_idx, None, "error", str(e))
                        if batch.is_final:
                            progress = batch.progress

                responses = cached_responses
            else:
                # 不使用缓存，直接批量执行（流式处理以支持增量保存）
                indices_to_run = [
                    i for i in range(len(messages_list)) if i not in completed_indices
                ]
                responses = [None] * len(messages_list)

                progress = None
                if indices_to_run:
                    request_params = [
                        {
                            "json": self._build_request_body(
                                messages_list[i], effective_model, **merged_kwargs(i)
                            ),
                            "headers": headers,
                        }
                        for i in indices_to_run
                    ]
                    async for batch in self._client.aiter_stream_requests(
                        request_params=request_params,
                        url=effective_url,
                        method="POST",
                        show_progress=show_progress,
                        total_requests=len(indices_to_run),
                        progress_config=progress_config,
                        model_name=effective_model if track_cost else None,
                        input_price_per_1m=input_price,
                        output_price_per_1m=output_price,
                    ):
                        for result in batch.completed_requests:
                            original_idx = indices_to_run[result.request_id]
                            if result.status != "success":
                                error_msg = (
                                    result.data.get("error", "Unknown error")
                                    if isinstance(result.data, dict)
                                    else str(result.data)
                                )
                                logger.debug(f"请求失败: {error_msg}")
                                responses[original_idx] = None
                                writer.write_result(original_idx, None, "error", error_msg)
                                continue
                            try:
                                extracted = extractor(result)
                                responses[original_idx] = extracted
                                writer.write_result(
                                    original_idx,
                                    with_prefix(original_idx, extracted["content"]),
                                    usage=extracted.get("usage"),
                                )
                                if self._cost_tracker and extracted.get("usage"):
                                    self._cost_tracker.record(extracted["usage"], effective_model)
                                if track_cost and batch.progress and extracted.get("usage"):
                                    usage = extracted["usage"]
                                    input_tokens = usage.get("prompt_tokens", 0)
                                    output_tokens = usage.get("completion_tokens", 0)
                                    cost = estimate_cost(
                                        input_tokens, output_tokens, effective_model
                                    )
                                    batch.progress.update_cost(input_tokens, output_tokens, cost)
                            except BudgetExceededError:
                                logger.warning("预算超限，停止批量处理")
                                raise
                            except Exception as e:
                                logger.warning(f"Error: {e}, set content to None")
                                responses[original_idx] = None
                                writer.write_result(original_idx, None, "error", str(e))
                        if batch.is_final:
                            progress = batch.progress

        except BudgetExceededError:
            # 预算超限时也要正常结束处理
            pass

        finally:
            writer.close()

        summary = progress.summary(print_to_console=False) if progress else None

        # 转换返回值格式（prefill 场景统一在此拼接 prefix）
        if return_usage:
            final_responses = [
                to_chat_result(r, i) if r is not None else None for i, r in enumerate(responses)
            ]
        else:
            final_responses = [
                with_prefix(i, r["content"]) if r is not None else None
                for i, r in enumerate(responses)
            ]

        # 构建返回值
        result = final_responses
        if return_summary:
            result = (final_responses, summary)
        if return_cost_report and self._cost_tracker:
            cost_report = self._cost_tracker.get_report()
            if return_summary:
                result = (final_responses, summary, cost_report)
            else:
                result = (final_responses, cost_report)
        return result

    def _compact_output_file(self, file_path: str):
        """去重输出文件，保留每个 index 的最新成功记录（委托给 batch_helpers）"""
        compact_output_file(file_path)

    def chat_completions_batch_sync(
        self,
        messages_list: list[list[dict]],
        model: str = None,
        return_raw: bool = False,
        return_usage: bool = False,
        show_progress: bool = True,
        return_summary: bool = False,
        return_cost_report: bool = False,
        track_cost: bool = False,
        output_jsonl: str | None = None,
        flush_interval: float = 1.0,
        metadata_list: list[dict] | None = None,
        save_input: bool | str = True,
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
                return_cost_report=return_cost_report,
                track_cost=track_cost,
                output_jsonl=output_jsonl,
                flush_interval=flush_interval,
                metadata_list=metadata_list,
                save_input=save_input,
                **kwargs,
            )
        )

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
        url: str = None,
        save_input: bool | str = True,
        include_prefix: bool = True,
        **kwargs,
    ):
        """
        迭代式批量聊天完成（边请求边返回结果）

        与 chat_completions_batch 功能相同，但以流式方式逐条返回结果，
        适合处理大批量数据时节省内存。

        Args:
            messages_list: 消息列表
            model: 模型名称
            return_raw: 是否返回原始响应（影响 result.content 的内容）
            return_usage: 是否在 result 对象上添加 usage 属性
            show_progress: 是否显示进度条
            preprocess_msg: 是否预处理消息
            output_jsonl: 输出文件路径（JSONL 格式），用于持久化保存结果
            flush_interval: 文件刷新间隔（秒），默认 1 秒
            metadata_list: 元数据列表，与 messages_list 等长，每个元素保存到对应输出记录
            batch_size: 每批返回的数量（传递给底层请求器）
            url: 自定义请求 URL，默认使用 _get_url() 生成
            save_input: 控制输出 JSONL 中 input 字段的保存策略（同 chat_completions_batch）

        Yields:
            result: 包含以下属性的结果对象
                - content: 提取后的内容 (str | dict)
                - usage: token 用量信息（仅当 return_usage=True 时）
                - original_idx: 原始索引
                - latency: 请求延迟（秒）
                - status: 状态 ('success', 'error', 'cached')
                - error: 错误信息（如果有）
                - data: 原始响应数据
                - summary: 最后一个 result 包含整体统计 (dict)，其他为 None
                    - total: 总请求数
                    - success: 成功数
                    - failed: 失败数
                    - cached: 缓存命中数
                    - elapsed: 总耗时（秒）
                    - avg_latency: 平均延迟（秒）

        Note:
            缓存由初始化时的 cache 参数控制
        """
        effective_model = self._get_effective_model(model)
        effective_url = url or self._get_url(effective_model, stream=False)
        headers = self._get_headers()

        validate_batch_params(messages_list, metadata_list, output_jsonl)
        messages_list = await self._preprocess_messages_batch(messages_list, preprocess_msg)

        use_cache = self._response_cache is not None and not return_raw

        # Prefill 拼接前缀(同 chat_completions_batch)
        prefix_list = (
            [self._trailing_assistant_prefix(m) for m in messages_list]
            if include_prefix
            else [None] * len(messages_list)
        )

        def with_prefix(idx: int, content):
            p = prefix_list[idx]
            if p and content is not None:
                return p + content
            return content

        # 使用 JsonlWriter 管理文件输出
        writer = JsonlWriter(output_jsonl, messages_list, save_input, metadata_list, flush_interval)
        completed_indices = writer.completed_indices

        try:
            # 统计信息
            total_count = len(messages_list)
            yielded_count = 0
            success_count = 0
            cached_count = 0
            start_time = time.time()
            total_latency = 0.0

            # 查询缓存
            cached_responses = [None] * len(messages_list)
            uncached_indices = list(range(len(messages_list)))

            if use_cache and self._response_cache:
                cached_responses, uncached_indices = self._response_cache.get_batch(
                    messages_list, model=effective_model, **kwargs
                )

                # 先 yield 缓存命中的结果
                for i, resp in enumerate(cached_responses):
                    if resp is not None:
                        merged_content = with_prefix(i, resp["content"])
                        if i not in completed_indices:
                            writer.write_result(i, merged_content, usage=resp.get("usage"))
                        from types import SimpleNamespace

                        yielded_count += 1
                        cached_count += 1
                        success_count += 1
                        is_last = yielded_count == total_count

                        cached_result = SimpleNamespace(
                            content=merged_content,
                            usage=resp.get("usage"),
                            original_idx=i,
                            latency=0.0,
                            status="cached",
                            error=None,
                            data=None,
                            summary=None,
                        )
                        if is_last:
                            cached_result.summary = {
                                "total": total_count,
                                "success": success_count,
                                "failed": total_count - success_count,
                                "cached": cached_count,
                                "elapsed": time.time() - start_time,
                                "avg_latency": total_latency / max(yielded_count - cached_count, 1),
                            }
                        yield cached_result

            # 过滤掉文件中已完成的
            actual_uncached = [i for i in uncached_indices if i not in completed_indices]

            if actual_uncached:
                logger.info(f"待执行: {len(actual_uncached)}/{len(messages_list)}")

                uncached_messages = [messages_list[i] for i in actual_uncached]
                request_params = [
                    {
                        "json": self._build_request_body(m, effective_model, **kwargs),
                        "headers": headers,
                    }
                    for m in uncached_messages
                ]

                async for batch in self._client.aiter_stream_requests(
                    request_params=request_params,
                    url=effective_url,
                    method="POST",
                    show_progress=show_progress,
                    batch_size=batch_size,
                    total_requests=len(uncached_messages),
                ):
                    for result in batch.completed_requests:
                        original_idx = actual_uncached[result.request_id]
                        yielded_count += 1
                        is_last = yielded_count == total_count

                        # 检查请求状态
                        if result.status != "success":
                            error_msg = (
                                result.data.get("error", "Unknown error")
                                if isinstance(result.data, dict)
                                else str(result.data)
                            )
                            logger.debug(f"请求失败: {error_msg}")
                            writer.write_result(original_idx, None, "error", error_msg)
                            result.content = None
                            result.usage = None
                            result.original_idx = original_idx
                            result.error = error_msg
                        else:
                            try:
                                content = (
                                    self._extract_content(result.data) if result.data else None
                                )
                                usage = self._extract_usage(result.data)
                                # 写入缓存(存原始,不含 prefix)
                                if use_cache and self._response_cache and content is not None:
                                    self._response_cache.set(
                                        messages_list[original_idx],
                                        {"content": content, "usage": usage},
                                        model=effective_model,
                                        **kwargs,
                                    )
                                merged_content = with_prefix(original_idx, content)
                                writer.write_result(original_idx, merged_content, usage=usage)
                                result.content = merged_content
                                result.usage = usage if return_usage else None
                                result.original_idx = original_idx
                                success_count += 1
                                total_latency += result.latency
                            except Exception as e:
                                logger.warning(f"提取结果失败: {e}")
                                writer.write_result(original_idx, None, "error", str(e))
                                result.content = None
                                result.usage = None
                                result.original_idx = original_idx

                        # 最后一个 result 添加 summary
                        result.summary = None
                        if is_last:
                            result.summary = {
                                "total": total_count,
                                "success": success_count,
                                "failed": total_count - success_count,
                                "cached": cached_count,
                                "elapsed": time.time() - start_time,
                                "avg_latency": total_latency / max(yielded_count - cached_count, 1),
                            }
                        yield result

        finally:
            writer.close()

    async def chat_completions_stream(
        self,
        messages: list[dict],
        model: str = None,
        return_usage: bool = False,
        preprocess_msg: bool = False,
        url: str = None,
        timeout: int = None,
        **kwargs,
    ):
        """
        流式聊天完成

        Args:
            messages: 消息列表
            model: 模型名称
            return_usage: 是否返回 usage 信息。当为 True 时，yield 的是 dict:
                - {"type": "content", "content": "..."} 表示内容片段
                - {"type": "usage", "usage": {...}} 表示 token 用量（最后一条）
                当为 False 时（默认），yield 的是 str 内容片段
            preprocess_msg: 是否预处理消息
            url: 自定义请求 URL，默认使用 _get_stream_url() 生成
            timeout: 超时时间（秒），默认使用客户端配置

        Yields:
            - return_usage=False: str 内容片段
            - return_usage=True: dict，包含 type 和对应数据
        """
        import json

        import aiohttp

        effective_model = self._get_effective_model(model)
        messages = await self._preprocess_messages(messages, preprocess_msg)

        body = self._build_request_body(messages, effective_model, stream=True, **kwargs)
        body = self._prepare_stream_body(body, return_usage)

        effective_url = url or self._get_stream_url(effective_model)
        headers = self._get_headers()

        effective_timeout = timeout if timeout is not None else self._timeout
        aio_timeout = aiohttp.ClientTimeout(total=effective_timeout)

        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.post(
                effective_url,
                json=body,
                headers=headers,
                timeout=aio_timeout,
                proxy=self._proxy,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

                _thinking_started = False
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)

                            # 检查是否包含 usage
                            if return_usage:
                                usage = self._extract_stream_usage(data)
                                if usage:
                                    yield {"type": "usage", "usage": usage}
                                    continue

                            # 提取思考内容
                            thinking = self._extract_stream_thinking(data)
                            if thinking:
                                if return_usage:
                                    yield {"type": "thinking", "content": thinking}
                                else:
                                    if not _thinking_started:
                                        yield "<think>\n"
                                        _thinking_started = True
                                    yield thinking
                                continue

                            # 提取 tool_call delta
                            tool_call_deltas = self._extract_stream_tool_calls(data)
                            if tool_call_deltas:
                                if return_usage:
                                    yield {
                                        "type": "tool_call_delta",
                                        "tool_calls": tool_call_deltas,
                                    }
                                continue

                            content = self._extract_stream_content(data)
                            if content:
                                if _thinking_started:
                                    yield "</think>"
                                    _thinking_started = False
                                if return_usage:
                                    yield {"type": "content", "content": content}
                                else:
                                    yield content
                        except json.JSONDecodeError:
                            continue

    def model_list(self) -> list[str]:
        raise NotImplementedError("子类需要实现 model_list 方法")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self._model}')"

    # ========== 资源管理 ==========

    async def aclose(self):
        """异步关闭客户端，释放资源（推荐在异步上下文中使用）"""
        if self._response_cache is not None:
            self._response_cache.close()
            self._response_cache = None
        if self._client is not None:
            await self._client.aclose()

    def close(self):
        """同步关闭客户端，释放资源（如缓存连接、HTTP session）"""
        if self._response_cache is not None:
            self._response_cache.close()
            self._response_cache = None
        if self._client is not None:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()
