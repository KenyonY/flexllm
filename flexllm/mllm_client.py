#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MLLM client
"""

import asyncio
from typing import List, Callable, Optional, Any
from rich import print
from .openaiclient import OpenAIClient
from .response_cache import ResponseCacheConfig

from .processors.unified_processor import (
    batch_process_messages,
    UnifiedProcessorConfig,
    UnifiedImageProcessor,
)


from .processors.image_processor import ImageCacheConfig
from abc import ABC, abstractmethod


class MllmClientBase(ABC):
    """
    MLLM客户端抽象基类
    定义了所有MLLM客户端必须实现的核心接口
    """

    @abstractmethod
    async def call_llm(
        self,
        messages_list,
        model=None,
        temperature=0.1,
        max_tokens=2000,
        top_p=0.95,
        safety={
            "input_level": "none",
            "input_image_level": "none",
        },
        **kwargs,
    ):
        """
        调用LLM的抽象方法

        Args:
            messages_list: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成token数
            top_p: top_p参数
            safety: 安全级别
            **kwargs: 其他参数

        Returns:
            response_list: 响应列表
        """
        pass


class MllmClient(MllmClientBase):
    """
    MLLM客户端实现类
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key="EMPTY",
        concurrency_limit=10,
        preprocess_concurrency=16,
        max_qps=50,
        timeout=60,
        retry_times=3,
        retry_delay=0.55,
        processor_config: Optional[UnifiedProcessorConfig] = None,
        cache: Optional[ResponseCacheConfig] = None,
        **kwargs,
    ):
        """
        初始化MLLM客户端

        Args:
            model: 模型名称
            base_url: API基础URL
            api_key: API密钥
            concurrency_limit: 并发限制
            preprocess_concurrency: 预处理并发数
            max_qps: 最大QPS
            timeout: 超时时间（秒）
            retry_times: 重试次数
            retry_delay: 重试延迟（秒）
            processor_config: 统一处理器配置，如为None则使用高性能默认配置
            cache: 响应缓存配置，默认启用（24小时TTL）
            **kwargs: 其他参数
        """
        self.client = OpenAIClient(
            api_key=api_key,
            base_url=base_url,
            concurrency_limit=concurrency_limit,
            max_qps=max_qps,
            timeout=timeout,
            retry_times=retry_times,
            retry_delay=retry_delay,
            cache=cache,
            **kwargs,
        )
        self.model = model
        self.preprocess_concurrency = preprocess_concurrency

        # 创建处理器配置和实例（关键改进）
        self.processor_config = (
            processor_config or UnifiedProcessorConfig.high_performance()
        )
        # 创建并持有处理器实例，保持缓存效果
        self.processor_instance = UnifiedImageProcessor(self.processor_config)

        # 延迟导入避免循环引用
        from .table_processor import MllmTableProcessor
        from .folder_processor import MllmFolderProcessor

        self.table = MllmTableProcessor(self)
        self.folder = MllmFolderProcessor(self)

    def call_llm_sync(
        self,
        messages_list,
        model=None,
        temperature=0.1,
        max_tokens=2000,
        top_p=0.95,
        safety={
            "input_level": "none",
            "input_image_level": "none",
        },
        **kwargs,
    ):
        return asyncio.run(
            self.call_llm(
                messages_list, model, temperature, max_tokens, top_p, safety, **kwargs
            )
        )

    async def call_llm(
        self,
        messages_list,
        model=None,
        temperature=0.1,
        max_tokens=2000,
        top_p=0.95,
        safety={
            "input_level": "none",
            "input_image_level": "none",
        },
        show_progress=True,
        **kwargs,
    ):
        """
        调用LLM

        Args:
            messages_list: 消息列表
            model: 模型名称，默认使用初始化时指定的模型
            temperature: 温度参数
            max_tokens: 最大生成token数
            top_p: top_p参数
            safety: 安全级别
            show_progress: 是否显示每一步的进度条和统计信息
            **kwargs: 其他参数

        Returns:
            response_list: 响应列表
        """
        if model is None:
            model = self.model

        # 使用持有的处理器实例进行预处理，保持缓存效果
        messages_list = await self._preprocess_messages_with_instance(
            messages_list,
            show_progress=show_progress,
        )
        # print(f"{messages_list[-1]=}")
        response_list, _ = await self.client.chat_completions_batch(
            messages_list=messages_list,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            return_summary=True,
            safety=safety,
            show_progress=show_progress,
            **kwargs,
        )
        return response_list

    async def call_llm_stream(
        self,
        messages: list,
        model=None,
        temperature=0.1,
        max_tokens=2000,
        top_p=0.95,
        safety={
            "input_level": "none",
            "input_image_level": "none",
        },
        **kwargs,
    ):
        """
        流式调用LLM - 逐token返回响应，适合单条对话

        Args:
            messages: 单条消息列表 (不是messages_list)
            model: 模型名称，默认使用初始化时指定的模型
            temperature: 温度参数
            max_tokens: 最大生成token数
            top_p: top_p参数
            safety: 安全级别
            **kwargs: 其他参数

        Yields:
            str: 流式返回的token片段
        """
        if model is None:
            model = self.model

        # 预处理消息（不显示进度条，因为只有一条消息）
        processed_messages = await self._preprocess_messages_with_instance(
            [messages], show_progress=False
        )

        # 使用OpenAIClient的流式方法
        async for token in self.client.chat_completions_stream(
            messages=processed_messages[0],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs,
        ):
            yield token

    async def _preprocess_messages_with_instance(
        self,
        messages_list,
        show_progress=True,
    ):
        """
        使用持有的处理器实例进行消息预处理
        这样可以保持缓存效果，避免重复初始化开销
        """
        from .processors.unified_processor import process_content_recursive
        import aiohttp
        from copy import deepcopy
        from tqdm.asyncio import tqdm
        import asyncio

        # 创建消息副本，避免修改原始数据
        messages_list = deepcopy(messages_list)

        # 创建进度条
        pbar = None
        if show_progress:
            try:
                pbar = tqdm(
                    total=len(messages_list),
                    desc="处理图片",
                    unit=" items",
                    ncols=100,
                    miniters=1,
                )
            except ImportError:
                pbar = None

        try:
            # 使用HTTP会话和持有的处理器实例
            async with aiohttp.ClientSession() as session:
                # 创建信号量控制并发
                semaphore = asyncio.Semaphore(self.preprocess_concurrency)

                async def process_single_messages(messages):
                    async with semaphore:
                        for message in messages:
                            await process_content_recursive(
                                message,
                                session,
                                self.processor_instance,  # 使用持有的实例！
                            )
                        if pbar:
                            pbar.update(1)
                        return messages

                # 并发处理所有消息组
                tasks = [
                    process_single_messages(messages) for messages in messages_list
                ]
                processed_messages_list = await asyncio.gather(*tasks)

            return processed_messages_list

        finally:
            if pbar:
                pbar.close()

    async def call_llm_with_selection(
        self,
        messages_list,
        n_predictions: int = 3,
        selector_fn: Optional[Callable[[List[Any]], Any]] = None,
        model=None,
        temperature=0.1,
        max_tokens=2000,
        top_p=0.95,
        safety={
            "input_level": "none",
            "input_image_level": "none",
        },
        show_progress=True,
        **kwargs,
    ):
        """
        增强版LLM调用方法，对每条消息进行n次预测，并使用选择函数选择最佳结果

        Args:
            messages_list: 消息列表
            n_predictions: 每条消息预测次数
            selector_fn: 选择函数，接收n个响应列表，返回选中的响应
                         如果为None，默认返回第一个响应
            model: 模型名称，默认使用初始化时指定的模型
            temperature: 温度参数
            max_tokens: 最大生成token数
            top_p: top_p参数
            safety: 安全级别
            show_progress: 是否显示进度条
            **kwargs: 其他参数

        Returns:
            response_list: 选择后的响应列表
        """
        if model is None:
            model = self.model

        # 默认选择函数(如果未提供)，简单返回第一个响应
        if selector_fn is None:
            selector_fn = lambda responses: responses[0]

        # 为每条消息创建n个副本
        expanded_messages_list = []
        for messages in messages_list:
            for _ in range(n_predictions):
                expanded_messages_list.append(messages)

        # 调用模型获取所有响应 - 使用持有的处理器实例
        messages_list = await self._preprocess_messages_with_instance(
            expanded_messages_list,
            show_progress=show_progress,
        )
        all_responses, _ = await self.client.chat_completions_batch(
            messages_list=messages_list,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            return_summary=True,
            safety=safety,
            show_progress=show_progress,
            **kwargs,
        )

        # 重组响应并应用选择函数
        selected_responses = []
        for i in range(0, len(all_responses), n_predictions):
            message_responses = all_responses[i : i + n_predictions]
            # 安全打印，避免打印可能包含base64的响应数据
            print(
                f"[cyan]处理第 {i // n_predictions + 1} 组响应（包含 {len(message_responses)} 个预测）[/cyan]"
            )
            selected_response = selector_fn(message_responses)
            selected_responses.append(selected_response)

        return selected_responses

    async def call_llm_nested(
        self,
        messages_list_list,
        model=None,
        temperature=0.1,
        max_tokens=2000,
        top_p=0.95,
        safety={
            "input_level": "none",
            "input_image_level": "none",
        },
        **kwargs,
    ):
        """
        处理嵌套的messages_list_list结构
        将messages_list_list展平为messages_list，调用call_llm获取结果，再重组为response_list_list
        这样做可以提高整体调用性能

        Args:
            messages_list_list: 嵌套的消息列表列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成token数
            top_p: top_p参数
            safety: 安全级别
            **kwargs: 其他参数

        Returns:
            response_list_list: 嵌套的响应列表列表，与输入结构对应
        """
        # 记录每个子列表的长度，用于之后重组结果
        lengths = [len(messages_list) for messages_list in messages_list_list]

        # 展平messages_list_list
        flattened_messages_list = []
        for messages_list in messages_list_list:
            flattened_messages_list.extend(messages_list)

        # 调用call_llm获取展平后的response_list
        flattened_response_list = await self.call_llm(
            flattened_messages_list,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            safety=safety,
            **kwargs,
        )

        # 根据之前记录的长度，将展平的response_list重组为response_list_list
        response_list_list = []
        start_idx = 0
        for length in lengths:
            response_list_list.append(
                flattened_response_list[start_idx : start_idx + length]
            )
            start_idx += length

        return response_list_list

    async def call_llm_nested_with_selection(
        self,
        messages_list_list,
        n_predictions: int = 3,
        selector_fn: Optional[Callable[[List[Any]], Any]] = None,
        model=None,
        temperature=0.1,
        max_tokens=2000,
        top_p=0.95,
        safety={
            "input_level": "none",
            "input_image_level": "none",
        },
        **kwargs,
    ):
        """
        处理嵌套的messages_list_list结构，并对每条消息进行多次预测和选择

        Args:
            messages_list_list: 嵌套的消息列表列表
            n_predictions: 每条消息预测次数
            selector_fn: 选择函数，接收n个响应列表，返回选中的响应
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成token数
            top_p: top_p参数
            safety: 安全级别
            **kwargs: 其他参数

        Returns:
            response_list_list: 嵌套的响应列表列表，与输入结构对应
        """
        # 记录每个子列表的长度，用于之后重组结果
        lengths = [len(messages_list) for messages_list in messages_list_list]

        # 展平messages_list_list
        flattened_messages_list = []
        for messages_list in messages_list_list:
            flattened_messages_list.extend(messages_list)

        # 调用enhanced_call_llm获取展平后的response_list
        flattened_response_list = await self.call_llm_with_selection(
            flattened_messages_list,
            n_predictions=n_predictions,
            selector_fn=selector_fn,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            safety=safety,
            **kwargs,
        )

        # 根据之前记录的长度，将展平的response_list重组为response_list_list
        response_list_list = []
        start_idx = 0
        for length in lengths:
            response_list_list.append(
                flattened_response_list[start_idx : start_idx + length]
            )
            start_idx += length

        return response_list_list

    def cleanup(self):
        """
        清理资源，释放处理器实例和客户端连接
        """
        if hasattr(self, "processor_instance") and self.processor_instance:
            self.processor_instance.cleanup()
            self.processor_instance = None

        # 清理客户端资源（包括响应缓存）
        if hasattr(self, "client") and self.client:
            self.client.close()

    def close(self):
        """关闭客户端，释放资源（别名方法，与 LLMClientBase 接口一致）"""
        self.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.cleanup()

    def __del__(self):
        """
        析构函数，确保资源被释放
        """
        try:
            self.cleanup()
        except Exception:
            pass  # 析构函数中避免抛出异常

    # 所有table和dataframe相关方法已移至TableProcessor类
