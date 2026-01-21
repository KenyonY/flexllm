"""
LLMClientBase - LLM 客户端抽象基类

提供通用的方法实现，子类只需实现核心的差异化方法。
"""

import asyncio
import json
import time
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from loguru import logger

from ..async_api import ConcurrentRequester
from ..async_api.progress import ProgressBarConfig
from ..cache import ResponseCache, ResponseCacheConfig
from ..msg_processors.image_processor import ImageCacheConfig
from ..msg_processors.messages_processor import messages_preprocess
from ..msg_processors.unified_processor import batch_process_messages as optimized_batch_preprocess
from ..pricing import estimate_cost, get_model_pricing
from ..pricing.cost_tracker import BudgetExceededError, CostReport, CostTracker, CostTrackerConfig

if TYPE_CHECKING:
    from ..async_api.interface import RequestResult


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
            cache: 响应缓存配置
                   - True: 启用缓存（默认 IPC 模式，24小时 TTL）
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

        self._client = ConcurrentRequester(
            concurrency_limit=concurrency_limit,
            max_qps=max_qps,
            timeout=timeout,
            retry_times=retry_times,
            retry_delay=retry_delay,
        )

        self._cache_config = ImageCacheConfig(
            enabled=cache_image,
            cache_dir=cache_dir,
            force_refresh=False,
            retry_failed=False,
        )

        # 响应缓存
        if cache is True:
            cache = ResponseCacheConfig.ipc()  # 默认 IPC 模式，24小时 TTL
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

    def _get_url(self, model: str, stream: bool = False) -> str:
        raise NotImplementedError

    def _get_headers(self) -> dict:
        raise NotImplementedError

    def _build_request_body(
        self, messages: list[dict], model: str, stream: bool = False, **kwargs
    ) -> dict:
        raise NotImplementedError

    def _extract_content(self, response_data: dict) -> str | None:
        raise NotImplementedError

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

    def _get_stream_url(self, model: str) -> str:
        return self._get_url(model, stream=True)

    # ========== 通用工具方法 ==========

    def _get_effective_model(self, model: str = None) -> str:
        effective_model = model or self._model
        if not effective_model:
            raise ValueError("必须提供 model 参数或在初始化时指定 model")
        return effective_model

    async def _preprocess_messages(
        self, messages: list[dict], preprocess_msg: bool = False
    ) -> list[dict]:
        """消息预处理（图片转 base64 等）"""
        if preprocess_msg:
            return await messages_preprocess(
                messages, preprocess_msg=preprocess_msg, cache_config=self._cache_config
            )
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

        Returns:
            - return_raw=True: RequestResult 原始响应
            - return_usage=True: ChatCompletionResult(content, usage, reasoning_content)
            - 默认: str 内容文本

        Note:
            缓存由初始化时的 cache 参数控制，return_raw/return_usage 时自动跳过缓存
        """
        effective_model = self._get_effective_model(model)
        messages = await self._preprocess_messages(messages, preprocess_msg)

        # 检查缓存（缓存不包含 usage 信息，return_raw/return_usage 时跳过缓存）
        use_cache = self._response_cache is not None and not return_raw and not return_usage
        if use_cache:
            cached = self._response_cache.get(messages, model=effective_model, **kwargs)
            if cached is not None:
                return cached

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
            # 写入缓存
            if use_cache and content is not None:
                self._response_cache.set(messages, content, model=effective_model, **kwargs)

            if return_usage:
                usage = self._extract_usage(data.data)
                tool_calls = self._extract_tool_calls(data.data)
                # 记录成本
                if self._cost_tracker:
                    self._cost_tracker.record(usage, effective_model)
                return ChatCompletionResult(content=content, usage=usage, tool_calls=tool_calls)
            return content
        return data

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

        Returns:
            - return_usage=True: List[ChatCompletionResult] 或 (List[ChatCompletionResult], summary)
            - return_cost_report=True: 返回元组 (results, cost_report)
            - 默认: List[str] 或 (List[str], summary)

        Note:
            缓存由初始化时的 cache 参数控制，return_usage=True 时自动跳过缓存

        Raises:
            BudgetExceededError: 当超过预算硬限制时（由 cost_tracker 配置）
        """
        # track_cost 需要 usage 信息
        if track_cost:
            return_usage = True
        effective_model = self._get_effective_model(model)
        effective_url = url or self._get_url(effective_model, stream=False)
        headers = self._get_headers()

        # metadata_list 长度校验
        if metadata_list is not None and len(metadata_list) != len(messages_list):
            raise ValueError(
                f"metadata_list 长度 ({len(metadata_list)}) 必须与 messages_list 长度 ({len(messages_list)}) 一致"
            )

        # output_jsonl 扩展名校验
        if output_jsonl and not output_jsonl.endswith(".jsonl"):
            raise ValueError(f"output_jsonl 必须使用 .jsonl 扩展名，当前: {output_jsonl}")

        messages_list = await self._preprocess_messages_batch(messages_list, preprocess_msg)

        # return_usage 时跳过缓存（缓存不包含 usage 信息）
        use_cache = self._response_cache is not None and not return_usage

        def extractor(result):
            return self._extract_content(result.data)

        def extractor_with_usage(result):
            content = self._extract_content(result.data)
            usage = self._extract_usage(result.data)
            tool_calls = self._extract_tool_calls(result.data)
            return ChatCompletionResult(content=content, usage=usage, tool_calls=tool_calls)

        # 进度条配置（支持成本显示）
        progress_config = ProgressBarConfig(show_cost=track_cost) if show_progress else None

        # 文件输出相关状态
        file_writer = None
        file_buffer = []
        last_flush_time = time.time()
        completed_indices = set()

        # 如果指定了输出文件，读取已完成的索引（断点续传）
        if output_jsonl:
            output_path = Path(output_jsonl)
            if output_path.exists():
                # 读取所有有效记录
                records = []
                with open(output_path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            record = json.loads(line.strip())
                            if record.get("status") == "success" and "input" in record:
                                idx = record.get("index")
                                if 0 <= idx < len(messages_list):
                                    records.append(record)
                        except (json.JSONDecodeError, KeyError, TypeError):
                            continue

                # 首尾校验：只比较第一条和最后一条的 input
                file_valid = True
                if records:
                    first, last = records[0], records[-1]
                    if first["input"] != messages_list[first["index"]]:
                        file_valid = False
                    elif len(records) > 1 and last["input"] != messages_list[last["index"]]:
                        file_valid = False

                if file_valid:
                    completed_indices = {r["index"] for r in records}
                    if completed_indices:
                        logger.info(
                            f"从文件恢复: 已完成 {len(completed_indices)}/{len(messages_list)}"
                        )
                else:
                    raise ValueError(
                        f"文件校验失败: {output_jsonl} 中的 input 与当前 messages_list 不匹配。"
                        f"请删除或重命名该文件后重试。"
                    )

            file_writer = open(output_path, "a", encoding="utf-8")

        def flush_to_file():
            """刷新缓冲区到文件"""
            nonlocal file_buffer, last_flush_time
            if file_writer and file_buffer:
                for record in file_buffer:
                    file_writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                file_writer.flush()
                file_buffer = []
                last_flush_time = time.time()

        def on_file_result(
            original_idx: int,
            content: Any,
            status: str = "success",
            error: str = None,
            usage: dict = None,
        ):
            """文件输出回调"""
            nonlocal last_flush_time
            if file_writer is None:
                return
            record = {
                "index": original_idx,
                "output": content,
                "status": status,
                "input": messages_list[original_idx],
            }
            if metadata_list is not None:
                record["metadata"] = metadata_list[original_idx]
            if usage is not None:
                record["usage"] = usage
            if error:
                record["error"] = error
            file_buffer.append(record)
            # 基于时间刷新
            if time.time() - last_flush_time >= flush_interval:
                flush_to_file()

        try:
            # 计算实际需要执行的索引（排除文件中已完成的）
            all_indices = set(range(len(messages_list)))
            indices_to_skip = completed_indices & all_indices
            if indices_to_skip:
                logger.info(f"从文件恢复跳过: {len(indices_to_skip)}/{len(messages_list)}")

            # 带缓存执行
            if use_cache and self._response_cache:
                # 查询缓存（传递 kwargs 以确保不同参数配置使用不同缓存键）
                cached_responses, uncached_indices = self._response_cache.get_batch(
                    messages_list, model=effective_model, **kwargs
                )

                # 将缓存命中的写入文件（如果文件中没有）
                for i, resp in enumerate(cached_responses):
                    if resp is not None and i not in completed_indices:
                        on_file_result(i, resp)

                # 过滤掉文件中已完成的
                actual_uncached = [i for i in uncached_indices if i not in completed_indices]

                progress = None
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

                    # 选择提取器
                    extract_fn = extractor_with_usage if return_usage else extractor

                    _pricing_set = False
                    async for batch in self._client.aiter_stream_requests(
                        request_params=request_params,
                        url=effective_url,
                        method="POST",
                        show_progress=show_progress,
                        total_requests=len(uncached_messages),
                        progress_config=progress_config,
                    ):
                        # 设置模型定价信息（仅在首次且开启成本追踪时）
                        if not _pricing_set and track_cost and batch.progress:
                            pricing = get_model_pricing(effective_model)
                            if pricing:
                                batch.progress.set_model_pricing(
                                    effective_model,
                                    pricing["input"] * 1e6,  # 转换回 $/1M
                                    pricing["output"] * 1e6,
                                )
                            _pricing_set = True
                        for result in batch.completed_requests:
                            original_idx = actual_uncached[result.request_id]
                            # 检查请求状态
                            if result.status != "success":
                                error_msg = (
                                    result.data.get("error", "Unknown error")
                                    if isinstance(result.data, dict)
                                    else str(result.data)
                                )
                                logger.warning(f"请求失败: {error_msg}")
                                cached_responses[original_idx] = None
                                on_file_result(original_idx, None, "error", error_msg)
                                continue
                            try:
                                extracted = extract_fn(result)
                                cached_responses[original_idx] = extracted
                                # 写入缓存（仅当不需要 usage 时，因为缓存不存储 usage）
                                if not return_usage:
                                    self._response_cache.set(
                                        messages_list[original_idx],
                                        extracted,
                                        model=effective_model,
                                        **kwargs,
                                    )
                                # 文件输出（存储 content 和 usage）
                                if return_usage:
                                    on_file_result(
                                        original_idx, extracted.content, usage=extracted.usage
                                    )
                                    # 记录成本
                                    if self._cost_tracker:
                                        self._cost_tracker.record(extracted.usage, effective_model)
                                    # 更新进度条的成本显示
                                    if track_cost and batch.progress and extracted.usage:
                                        input_tokens = extracted.usage.get("prompt_tokens", 0)
                                        output_tokens = extracted.usage.get("completion_tokens", 0)
                                        cost = estimate_cost(
                                            input_tokens, output_tokens, effective_model
                                        )
                                        batch.progress.update_cost(
                                            input_tokens, output_tokens, cost
                                        )
                                else:
                                    on_file_result(original_idx, extracted)
                            except BudgetExceededError:
                                # 预算超限，优雅停止
                                logger.warning("预算超限，停止批量处理")
                                raise
                            except Exception as e:
                                logger.warning(f"提取结果失败: {e}")
                                cached_responses[original_idx] = None
                                on_file_result(original_idx, None, "error", str(e))
                        if batch.is_final:
                            progress = batch.progress

                responses = cached_responses
            else:
                # 不使用缓存，直接批量执行（流式处理以支持增量保存）
                indices_to_run = [
                    i for i in range(len(messages_list)) if i not in completed_indices
                ]
                responses = [None] * len(messages_list)

                # 选择提取器
                extract_fn = extractor_with_usage if return_usage else extractor

                progress = None
                if indices_to_run:
                    messages_to_run = [messages_list[i] for i in indices_to_run]
                    request_params = [
                        {
                            "json": self._build_request_body(m, effective_model, **kwargs),
                            "headers": headers,
                        }
                        for m in messages_to_run
                    ]
                    # 使用流式处理，每完成一个请求就写入文件
                    _pricing_set = False
                    async for batch in self._client.aiter_stream_requests(
                        request_params=request_params,
                        url=effective_url,
                        method="POST",
                        show_progress=show_progress,
                        total_requests=len(messages_to_run),
                        progress_config=progress_config,
                    ):
                        # 设置模型定价信息（仅在首次且开启成本追踪时）
                        if not _pricing_set and track_cost and batch.progress:
                            pricing = get_model_pricing(effective_model)
                            if pricing:
                                batch.progress.set_model_pricing(
                                    effective_model,
                                    pricing["input"] * 1e6,  # 转换回 $/1M
                                    pricing["output"] * 1e6,
                                )
                            _pricing_set = True
                        for result in batch.completed_requests:
                            original_idx = indices_to_run[result.request_id]
                            # 检查请求状态
                            if result.status != "success":
                                error_msg = (
                                    result.data.get("error", "Unknown error")
                                    if isinstance(result.data, dict)
                                    else str(result.data)
                                )
                                logger.warning(f"请求失败: {error_msg}")
                                responses[original_idx] = None
                                on_file_result(original_idx, None, "error", error_msg)
                                continue
                            try:
                                extracted = extract_fn(result)
                                responses[original_idx] = extracted
                                # 文件输出（存储 content 和 usage）
                                if return_usage:
                                    on_file_result(
                                        original_idx, extracted.content, usage=extracted.usage
                                    )
                                    # 记录成本
                                    if self._cost_tracker:
                                        self._cost_tracker.record(extracted.usage, effective_model)
                                    # 更新进度条的成本显示
                                    if track_cost and batch.progress and extracted.usage:
                                        input_tokens = extracted.usage.get("prompt_tokens", 0)
                                        output_tokens = extracted.usage.get("completion_tokens", 0)
                                        cost = estimate_cost(
                                            input_tokens, output_tokens, effective_model
                                        )
                                        batch.progress.update_cost(
                                            input_tokens, output_tokens, cost
                                        )
                                else:
                                    on_file_result(original_idx, extracted)
                            except BudgetExceededError:
                                # 预算超限，优雅停止
                                logger.warning("预算超限，停止批量处理")
                                raise
                            except Exception as e:
                                logger.warning(f"Error: {e}, set content to None")
                                responses[original_idx] = None
                                on_file_result(original_idx, None, "error", str(e))
                        if batch.is_final:
                            progress = batch.progress

        except BudgetExceededError:
            # 预算超限时也要正常结束处理
            pass

        finally:
            # 确保最后的数据写入
            flush_to_file()
            if file_writer:
                file_writer.close()
                # 自动 compact：去重，保留每个 index 的最新成功记录
                self._compact_output_file(output_jsonl)

        summary = progress.summary(print_to_console=False) if progress else None

        # 构建返回值
        result = responses
        if return_summary:
            result = (responses, summary)
        if return_cost_report and self._cost_tracker:
            cost_report = self._cost_tracker.get_report()
            if return_summary:
                result = (responses, summary, cost_report)
            else:
                result = (responses, cost_report)
        return result

    def _compact_output_file(self, file_path: str):
        """去重输出文件，保留每个 index 的最新成功记录"""
        import os

        tmp_path = file_path + ".tmp"
        try:
            records = {}
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    idx = r.get("index")
                    if idx is None:
                        continue
                    # 成功记录优先，或者该 index 还没有记录
                    if r.get("status") == "success" or idx not in records:
                        records[idx] = r

            # 先写入临时文件
            with open(tmp_path, "w", encoding="utf-8") as f:
                for r in sorted(records.values(), key=lambda x: x["index"]):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            # 原子替换（同一文件系统上 replace 是原子操作）
            os.replace(tmp_path, file_path)
        except Exception as e:
            logger.warning(f"Compact 输出文件失败: {e}")
            # 清理可能残留的临时文件
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

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
            缓存由初始化时的 cache 参数控制，return_usage=True 时自动跳过缓存
        """
        effective_model = self._get_effective_model(model)
        effective_url = url or self._get_url(effective_model, stream=False)
        headers = self._get_headers()

        # metadata_list 长度校验
        if metadata_list is not None and len(metadata_list) != len(messages_list):
            raise ValueError(
                f"metadata_list 长度 ({len(metadata_list)}) 必须与 messages_list 长度 ({len(messages_list)}) 一致"
            )

        # output_jsonl 扩展名校验
        if output_jsonl and not output_jsonl.endswith(".jsonl"):
            raise ValueError(f"output_jsonl 必须使用 .jsonl 扩展名，当前: {output_jsonl}")

        messages_list = await self._preprocess_messages_batch(messages_list, preprocess_msg)

        # return_usage 时跳过缓存
        use_cache = self._response_cache is not None and not return_usage

        def extractor(result):
            if return_raw:
                return result.data
            return self._extract_content(result.data) if result.data else None

        # 文件输出相关状态
        file_writer = None
        file_buffer = []
        last_flush_time = time.time()
        completed_indices = set()

        # 如果指定了输出文件，读取已完成的索引（断点续传）
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
                                if 0 <= idx < len(messages_list):
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
                    completed_indices = {r["index"] for r in records}
                    if completed_indices:
                        logger.info(
                            f"从文件恢复: 已完成 {len(completed_indices)}/{len(messages_list)}"
                        )
                else:
                    raise ValueError(
                        f"文件校验失败: {output_jsonl} 中的 input 与当前 messages_list 不匹配。"
                        f"请删除或重命名该文件后重试。"
                    )

            file_writer = open(output_path, "a", encoding="utf-8")

        def flush_to_file():
            nonlocal file_buffer, last_flush_time
            if file_writer and file_buffer:
                for record in file_buffer:
                    file_writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                file_writer.flush()
                file_buffer = []
                last_flush_time = time.time()

        def on_file_result(
            original_idx: int,
            content: Any,
            status: str = "success",
            error: str = None,
            usage: dict = None,
        ):
            nonlocal last_flush_time
            if file_writer is None:
                return
            record = {
                "index": original_idx,
                "output": content,
                "status": status,
                "input": messages_list[original_idx],
            }
            if metadata_list is not None:
                record["metadata"] = metadata_list[original_idx]
            if usage is not None:
                record["usage"] = usage
            if error:
                record["error"] = error
            file_buffer.append(record)
            if time.time() - last_flush_time >= flush_interval:
                flush_to_file()

        try:
            # 统计信息
            total_count = len(messages_list)
            yielded_count = 0
            success_count = 0
            cached_count = 0
            start_time = time.time()
            total_latency = 0.0
            last_progress = None

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
                        if i not in completed_indices:
                            on_file_result(i, resp)
                        # 缓存命中时创建结果对象
                        from types import SimpleNamespace

                        yielded_count += 1
                        cached_count += 1
                        success_count += 1
                        is_last = yielded_count == total_count

                        cached_result = SimpleNamespace(
                            content=resp,
                            usage=None,  # 缓存不包含 usage 信息
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
                            logger.warning(f"请求失败: {error_msg}")
                            on_file_result(original_idx, None, "error", error_msg)
                            result.content = None
                            result.usage = None
                            result.original_idx = original_idx
                            result.error = error_msg
                        else:
                            try:
                                content = extractor(result)
                                usage = self._extract_usage(result.data) if return_usage else None
                                # 写入缓存
                                if (
                                    use_cache
                                    and self._response_cache
                                    and content is not None
                                    and not return_raw
                                ):
                                    self._response_cache.set(
                                        messages_list[original_idx],
                                        content,
                                        model=effective_model,
                                        **kwargs,
                                    )
                                on_file_result(original_idx, content, usage=usage)
                                # 在 result 对象上添加属性
                                result.content = content
                                result.usage = usage
                                result.original_idx = original_idx
                                success_count += 1
                                total_latency += result.latency
                            except Exception as e:
                                logger.warning(f"提取结果失败: {e}")
                                on_file_result(original_idx, None, "error", str(e))
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
                    if batch.is_final:
                        last_progress = batch.progress

        finally:
            flush_to_file()
            if file_writer:
                file_writer.close()
                # 自动 compact：去重，保留每个 index 的最新成功记录
                self._compact_output_file(output_jsonl)

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

        # 当需要 usage 时，添加 stream_options（OpenAI 格式）
        if return_usage:
            body["stream_options"] = {"include_usage": True}

        effective_url = url or self._get_stream_url(effective_model)
        headers = self._get_headers()

        effective_timeout = timeout if timeout is not None else self._timeout
        aio_timeout = aiohttp.ClientTimeout(total=effective_timeout)

        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.post(
                effective_url, json=body, headers=headers, timeout=aio_timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)

                            # 检查是否包含 usage（流式响应的最后一个 chunk）
                            if return_usage and "usage" in data and data["usage"]:
                                yield {"type": "usage", "usage": data["usage"]}
                                continue

                            content = self._extract_stream_content(data)
                            if content:
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

    def close(self):
        """关闭客户端，释放资源（如缓存连接）"""
        if self._response_cache is not None:
            self._response_cache.close()
            self._response_cache = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.close()
