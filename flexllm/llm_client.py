"""
LLMClient - 统一的 LLM 客户端封装

自动根据配置选择 OpenAIClient、GeminiClient 或 ClaudeClient，提供统一接口。
"""

from typing import TYPE_CHECKING, List, Union, Optional, Literal

from .base_client import LLMClientBase, ChatCompletionResult
from .openaiclient import OpenAIClient
from .geminiclient import GeminiClient
from .claudeclient import ClaudeClient
from .response_cache import ResponseCacheConfig

if TYPE_CHECKING:
    from flexllm.async_api.interface import RequestResult


class LLMClient:
    """
    统一的 LLM 客户端，支持 OpenAI 兼容 API、Gemini API 和 Claude API

    根据 provider 参数自动选择底层客户端：
    - "openai": 使用 OpenAIClient（适用于 OpenAI、vLLM、Ollama 等）
    - "gemini": 使用 GeminiClient（适用于 Google Gemini）
    - "claude": 使用 ClaudeClient（适用于 Anthropic Claude）

    所有 LLMClientBase 的方法都可以直接调用，会自动委托给底层客户端。

    Example (OpenAI 兼容):
        >>> client = LLMClient(
        ...     provider="openai",
        ...     base_url="https://api.openai.com/v1",
        ...     api_key="your-key",
        ...     model="gpt-4",
        ... )
        >>> result = await client.chat_completions(messages)

    Example (Gemini):
        >>> client = LLMClient(
        ...     provider="gemini",
        ...     api_key="your-google-key",
        ...     model="gemini-3-flash-preview",
        ... )
        >>> result = await client.chat_completions(messages)

    Example (Claude):
        >>> client = LLMClient(
        ...     provider="claude",
        ...     api_key="your-anthropic-key",
        ...     model="claude-3-5-sonnet-20241022",
        ... )
        >>> result = await client.chat_completions(messages)

    Example (thinking 参数 - 统一的思考控制):
        >>> # 禁用思考（快速响应）
        >>> result = client.chat_completions_sync(
        ...     messages=[{"role": "user", "content": "1+1=?"}],
        ...     thinking=False,
        ... )
        >>> # 启用思考并获取思考内容
        >>> result = client.chat_completions_sync(
        ...     messages=[{"role": "user", "content": "1+1=?"}],
        ...     thinking=True,
        ...     return_raw=True,
        ... )
        >>> parsed = client.parse_thoughts(result.data)
        >>> print("思考:", parsed["thought"])
        >>> print("答案:", parsed["answer"])

    thinking 参数值:
        - False: 禁用思考
        - True: 启用思考并返回思考内容
        - "minimal"/"low"/"medium"/"high": 设置思考深度（仅 Gemini）
        - int: 设置 budget_tokens（仅 Claude）
        - None: 使用模型默认行为
    """

    _client: LLMClientBase

    def __init__(
        self,
        provider: Literal["openai", "gemini", "claude", "auto"] = "auto",
        # 通用参数
        base_url: str = None,
        api_key: str = None,
        model: str = None,
        concurrency_limit: int = 10,
        max_qps: int = None,  # 不同 provider 有不同默认值
        timeout: int = 120,
        retry_times: int = 3,
        retry_delay: float = 1.0,
        cache_image: bool = False,
        cache_dir: str = "image_cache",
        # Gemini/Vertex AI 专用
        use_vertex_ai: bool = False,
        project_id: str = None,
        location: str = "us-central1",
        credentials=None,
        # 响应缓存配置
        cache: Optional[ResponseCacheConfig] = None,
        **kwargs,
    ):
        """
        初始化统一 LLM 客户端

        Args:
            provider: 指定使用的 provider
                - "openai": OpenAI 兼容 API
                - "gemini": Google Gemini API
                - "claude": Anthropic Claude API
                - "auto": 根据 base_url 自动推断
            base_url: API 基础 URL
            api_key: API 密钥
            model: 默认模型名称
            concurrency_limit: 并发请求限制
            max_qps: 最大 QPS（openai 默认 1000，gemini 默认 60）
            timeout: 请求超时时间
            retry_times: 重试次数
            retry_delay: 重试延迟
            cache_image: 是否缓存图片
            cache_dir: 图片缓存目录
            use_vertex_ai: 是否使用 Vertex AI（仅 Gemini）
            project_id: GCP 项目 ID（仅 Vertex AI）
            location: GCP 区域（仅 Vertex AI）
            credentials: Google Cloud 凭证（仅 Vertex AI）
            cache: 响应缓存配置，默认启用（24小时TTL）
        """
        # 自动推断 provider
        if provider == "auto":
            provider = self._infer_provider(base_url, use_vertex_ai)

        self._provider = provider
        self._model = model

        if provider == "gemini":
            self._client = GeminiClient(
                api_key=api_key,
                model=model,
                base_url=base_url,
                concurrency_limit=concurrency_limit,
                max_qps=max_qps if max_qps is not None else 60,
                timeout=timeout,
                retry_times=retry_times,
                retry_delay=retry_delay,
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
            self._client = ClaudeClient(
                api_key=api_key,
                model=model,
                base_url=base_url,
                concurrency_limit=concurrency_limit,
                max_qps=max_qps if max_qps is not None else 60,
                timeout=timeout,
                retry_times=retry_times,
                retry_delay=retry_delay,
                cache_image=cache_image,
                cache_dir=cache_dir,
                cache=cache,
                **kwargs,
            )
        else:  # openai
            if not base_url:
                raise ValueError("OpenAI provider 需要提供 base_url")
            self._client = OpenAIClient(
                base_url=base_url,
                api_key=api_key or "EMPTY",
                model=model,
                concurrency_limit=concurrency_limit,
                max_qps=max_qps if max_qps is not None else 1000,
                timeout=timeout,
                retry_times=retry_times,
                retry_delay=retry_delay,
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

    def __getattr__(self, name):
        """自动委托未显式定义的方法给底层客户端"""
        return getattr(self._client, name)

    @property
    def provider(self) -> str:
        """返回当前使用的 provider"""
        return self._provider

    @property
    def client(self) -> LLMClientBase:
        """返回底层客户端实例（用于访问特定功能）"""
        return self._client

    # ========== 显式定义常用方法（用于 IDE 代码提示）==========

    async def chat_completions(
        self,
        messages: List[dict],
        model: str = None,
        return_raw: bool = False,
        return_usage: bool = False,
        show_progress: bool = False,
        preprocess_msg: bool = False,
        **kwargs,
    ) -> Union[str, ChatCompletionResult, "RequestResult"]:
        """
        单条聊天完成

        Args:
            messages: 消息列表（OpenAI 格式）
            model: 模型名称（可选，使用初始化时的默认值）
            return_raw: 是否返回原始响应（RequestResult）
            return_usage: 是否返回包含 usage 的结果（ChatCompletionResult）
            show_progress: 是否显示进度
            preprocess_msg: 是否预处理消息（图片转 base64）
            **kwargs: 其他参数（max_tokens, temperature, thinking 等）

        Returns:
            - return_raw=True: RequestResult 原始响应
            - return_usage=True: ChatCompletionResult(content, usage, reasoning_content)
            - 默认: str 内容文本

        Note:
            缓存由初始化时的 cache 参数控制，return_raw/return_usage 时自动跳过缓存
        """
        return await self._client.chat_completions(
            messages=messages,
            model=model,
            return_raw=return_raw,
            return_usage=return_usage,
            show_progress=show_progress,
            preprocess_msg=preprocess_msg,
            **kwargs,
        )

    def chat_completions_sync(
        self,
        messages: List[dict],
        model: str = None,
        return_raw: bool = False,
        return_usage: bool = False,
        **kwargs,
    ) -> Union[str, ChatCompletionResult, "RequestResult"]:
        """
        同步版本的聊天完成

        Args:
            messages: 消息列表（OpenAI 格式）
            model: 模型名称
            return_raw: 是否返回原始响应
            return_usage: 是否返回包含 usage 的结果
            **kwargs: 其他参数（max_tokens, temperature, thinking 等）
        """
        return self._client.chat_completions_sync(
            messages=messages,
            model=model,
            return_raw=return_raw,
            return_usage=return_usage,
            **kwargs,
        )

    async def chat_completions_batch(
        self,
        messages_list: List[List[dict]],
        model: str = None,
        return_raw: bool = False,
        return_usage: bool = False,
        show_progress: bool = True,
        return_summary: bool = False,
        preprocess_msg: bool = False,
        output_jsonl: Optional[str] = None,
        flush_interval: float = 1.0,
        metadata_list: Optional[List[dict]] = None,
        **kwargs,
    ) -> Union[List[str], List[ChatCompletionResult], tuple]:
        """
        批量聊天完成（支持断点续传）

        Args:
            messages_list: 消息列表的列表
            model: 模型名称
            return_raw: 是否返回原始响应
            return_usage: 是否返回包含 usage 的结果（ChatCompletionResult 列表）
            show_progress: 是否显示进度条
            return_summary: 是否返回统计摘要
            preprocess_msg: 是否预处理消息
            output_jsonl: 输出文件路径（JSONL），用于断点续传和持久化
            flush_interval: 文件刷新间隔（秒）
            metadata_list: 元数据列表，与 messages_list 等长，每个元素保存到对应输出记录
            **kwargs: 其他参数（max_tokens, temperature, thinking 等）

        Returns:
            - return_usage=True: List[ChatCompletionResult] 或 (List[ChatCompletionResult], summary)
            - 默认: List[str] 或 (List[str], summary)

        Note:
            缓存由初始化时的 cache 参数控制，return_usage=True 时自动跳过缓存
        """
        return await self._client.chat_completions_batch(
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

    def chat_completions_batch_sync(
        self,
        messages_list: List[List[dict]],
        model: str = None,
        return_raw: bool = False,
        return_usage: bool = False,
        show_progress: bool = True,
        return_summary: bool = False,
        output_jsonl: Optional[str] = None,
        flush_interval: float = 1.0,
        metadata_list: Optional[List[dict]] = None,
        **kwargs,
    ) -> Union[List[str], List[ChatCompletionResult], tuple]:
        """同步版本的批量聊天完成"""
        return self._client.chat_completions_batch_sync(
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

    async def iter_chat_completions_batch(
        self,
        messages_list: List[List[dict]],
        model: str = None,
        return_raw: bool = False,
        return_usage: bool = False,
        show_progress: bool = True,
        preprocess_msg: bool = False,
        output_jsonl: Optional[str] = None,
        flush_interval: float = 1.0,
        metadata_list: Optional[List[dict]] = None,
        batch_size: int = None,
        **kwargs,
    ):
        """
        迭代式批量聊天完成（边请求边返回结果）

        与 chat_completions_batch 功能相同，但以流式方式逐条返回结果，
        适合处理大批量数据时节省内存。

        Args:
            messages_list: 消息列表的列表
            model: 模型名称
            return_raw: 是否返回原始响应（影响 result.content 的内容）
            return_usage: 是否在 result 对象上添加 usage 属性
            show_progress: 是否显示进度条
            preprocess_msg: 是否预处理消息
            output_jsonl: 输出文件路径（JSONL）
            flush_interval: 文件刷新间隔（秒）
            metadata_list: 元数据列表，与 messages_list 等长，每个元素保存到对应输出记录
            batch_size: 每批返回的数量
            **kwargs: 其他参数（max_tokens, temperature, thinking 等）

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
        async for result in self._client.iter_chat_completions_batch(
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

    async def chat_completions_stream(
        self,
        messages: List[dict],
        model: str = None,
        return_usage: bool = False,
        preprocess_msg: bool = False,
        timeout: int = None,
        **kwargs,
    ):
        """
        流式聊天完成 - 逐 token 返回响应

        Args:
            messages: 消息列表
            model: 模型名称
            return_usage: 是否返回 usage 信息。当为 True 时，yield 的是 dict:
                - {"type": "content", "content": "..."} 表示内容片段
                - {"type": "usage", "usage": {...}} 表示 token 用量（最后一条）
                当为 False 时（默认），yield 的是 str 内容片段
            preprocess_msg: 是否预处理消息
            timeout: 超时时间（秒）
            **kwargs: 其他参数（max_tokens, temperature, thinking 等）

        Yields:
            - return_usage=False: str 内容片段
            - return_usage=True: dict，包含 type 和对应数据
        """
        async for chunk in self._client.chat_completions_stream(
            messages=messages,
            model=model,
            return_usage=return_usage,
            preprocess_msg=preprocess_msg,
            timeout=timeout,
            **kwargs,
        ):
            yield chunk

    def model_list(self) -> List[str]:
        """获取可用模型列表"""
        return self._client.model_list()

    def parse_thoughts(self, response_data: dict) -> dict:
        """
        从响应中解析思考内容和答案

        根据 provider 自动选择正确的解析方法。

        Args:
            response_data: 原始响应数据（通过 return_raw=True 获取）

        Returns:
            dict: {
                "thought": str,  # 思考过程（可能为空）
                "answer": str,   # 最终答案
            }

        Example:
            >>> result = client.chat_completions_sync(
            ...     messages=[...],
            ...     thinking=True,
            ...     return_raw=True,
            ... )
            >>> parsed = client.parse_thoughts(result.data)
            >>> print("思考:", parsed["thought"])
            >>> print("答案:", parsed["answer"])
        """
        if self._provider == "gemini":
            return GeminiClient.parse_thoughts(response_data)
        elif self._provider == "claude":
            return ClaudeClient.parse_thoughts(response_data)
        else:
            return OpenAIClient.parse_thoughts(response_data)

    def close(self):
        """关闭客户端，释放资源"""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return f"LLMClient(provider='{self._provider}', model='{self._model}')"
