"""
flexllm - High-performance LLM client

Batch processing, caching, and checkpoint recovery for LLM APIs.

Example:
    # =====================================================
    # 1. LLMClient - Unified Client (Recommended)
    # =====================================================
    from flexllm import LLMClient

    # 自动识别 provider（根据 base_url 推断）
    client = LLMClient(
        base_url="https://api.openai.com/v1",  # 或 vLLM/Ollama/DeepSeek 地址
        api_key="your-key",
        model="gpt-4",
        concurrency_limit=10,
        retry_times=3,
    )

    # 同步调用（简单场景）
    result = client.chat_completions_sync(
        messages=[{"role": "user", "content": "Hello!"}]
    )

    # 异步批量调用 + 断点续传
    results = await client.chat_completions_batch(
        messages_list,
        show_progress=True,
        output_jsonl="results.jsonl",  # 增量写入，中断后自动恢复
    )

    # 流式输出
    async for chunk in client.chat_completions_stream(messages):
        print(chunk, end="", flush=True)

    # 使用 Gemini
    gemini_client = LLMClient(
        provider="gemini",
        api_key="your-google-key",
        model="gemini-2.5-flash",
    )

    # =====================================================
    # 2. OpenAIClient - OpenAI 兼容 API（vLLM、Ollama 等）
    # =====================================================
    from flexllm import OpenAIClient, ResponseCacheConfig

    client = OpenAIClient(
        base_url="https://api.example.com/v1",
        api_key="your-key",
        model="qwen-vl-plus",
        concurrency_limit=10,  # 并发数
        max_qps=50,            # QPS 限制
        retry_times=3,         # 自动重试
        cache=ResponseCacheConfig(enabled=True),  # 启用响应缓存（默认1小时TTL）
    )

    # 单条调用
    result = await client.chat_completions(messages)

    # 批量调用 + 断点续传（中断后自动从缓存/文件恢复）
    results = await client.chat_completions_batch(
        messages_list,
        show_progress=True,
        output_jsonl="results.jsonl",  # 增量写入文件（断点续传）
        flush_interval=1.0,           # 每秒刷新到磁盘
    )

    # 流式输出
    async for chunk in client.chat_completions_stream(messages):
        print(chunk, end="", flush=True)

    # =====================================================
    # 3. GeminiClient - Google Gemini（Developer API / Vertex AI）
    # =====================================================
    from flexllm import GeminiClient

    # Gemini Developer API
    gemini = GeminiClient(
        api_key="your-google-api-key",
        model="gemini-2.5-flash",
        concurrency_limit=10,
    )
    result = await gemini.chat_completions(messages)

    # Vertex AI 模式
    gemini_vertex = GeminiClient(
        project_id="your-project-id",
        location="us-central1",
        model="gemini-2.5-flash",
        use_vertex_ai=True,
    )

    # Gemini 思考模式
    result = await gemini.chat_completions(
        messages,
        thinking="high",  # False, True, "minimal", "low", "medium", "high"
    )

    # =====================================================
    # 4. 多 Endpoint 负载均衡和故障转移（推荐）
    # =====================================================
    from flexllm import LLMClientPool

    # 创建客户端池（轮询 + 故障转移）
    pool = LLMClientPool(
        endpoints=[
            {"base_url": "http://host1:8000/v1", "api_key": "key1", "model": "qwen"},
            {"base_url": "http://host2:8000/v1", "api_key": "key2", "model": "qwen"},
        ],
        load_balance="round_robin",  # round_robin, weighted, random, fallback
        fallback=True,  # 失败时自动切换到其他 endpoint
    )

    # 接口与 LLMClient 完全一致
    result = await pool.chat_completions(messages)
    results = await pool.chat_completions_batch(messages_list)

    # 批量调用可分散到多个 endpoint 并行处理
    results = await pool.chat_completions_batch(messages_list, distribute=True)

    # =====================================================
    # 5. 底层 Provider 路由器（高级用法）
    # =====================================================
    from flexllm import ProviderRouter, ProviderConfig, create_router_from_urls

    # 快速创建（多个 URL 轮询）
    router = create_router_from_urls(
        urls=["http://host1:8000/v1", "http://host2:8000/v1"],
        api_key="EMPTY",
        strategy="round_robin",
    )

    # 获取下一个可用 provider
    provider = router.get_next()
    client = OpenAIClient(base_url=provider.base_url, api_key=provider.api_key)

    # 请求成功/失败时更新状态（自动 fallback）
    router.mark_success(provider)  # 或 router.mark_failed(provider)

    # =====================================================
    # 6. 响应缓存配置
    # =====================================================
    from flexllm import ResponseCacheConfig

    cache = ResponseCacheConfig()                         # 默认禁用
    cache = ResponseCacheConfig(enabled=True)             # 启用（默认1天 TTL）
    cache = ResponseCacheConfig(enabled=True, ttl=0)      # 启用（永不过期）
    cache = ResponseCacheConfig(enabled=False)            # 显式禁用
    cache = ResponseCacheConfig(enabled=True, ttl=3600)   # 自定义 TTL（秒）
"""

__version__ = "0.2.4"

# 多模态模型功能
from .mllm_client import MllmClient
from .table_processor import MllmTableProcessor
from .folder_processor import MllmFolderProcessor

# LLM基础功能
from .base_client import LLMClientBase, ChatCompletionResult, BatchResultItem, ToolCall
from .openaiclient import OpenAIClient
from .geminiclient import GeminiClient
from .claudeclient import ClaudeClient
from .llm_client import LLMClient
from .llm_parser import *

# Token 计数和成本估算
from .token_counter import (
    count_tokens,
    count_messages_tokens,
    estimate_cost,
    estimate_batch_cost,
    messages_hash,
    MODEL_PRICING,
)

# 响应缓存
from .response_cache import ResponseCache, ResponseCacheConfig

# Provider 路由
from .provider_router import ProviderRouter, ProviderConfig, create_router_from_urls

# 客户端池
from .client_pool import LLMClientPool, EndpointConfig

# Chain of Thought
from .chain_of_thought_client import ChainOfThoughtClient, Step

__all__ = [
    # 客户端
    'LLMClientBase',
    'MllmClient',
    'MllmTableProcessor',
    'MllmFolderProcessor',
    'OpenAIClient',
    'GeminiClient',
    'ClaudeClient',
    'LLMClient',
    # 结果类型
    'ChatCompletionResult',
    'BatchResultItem',
    'ToolCall',
    # Token 计数
    'count_tokens',
    'count_messages_tokens',
    'estimate_cost',
    'estimate_batch_cost',
    'messages_hash',
    'MODEL_PRICING',
    # 缓存
    'ResponseCache',
    'ResponseCacheConfig',
    # Provider 路由
    'ProviderRouter',
    'ProviderConfig',
    'create_router_from_urls',
    # 客户端池
    'LLMClientPool',
    'EndpointConfig',
    # Chain of Thought
    'ChainOfThoughtClient',
    'Step',
]
