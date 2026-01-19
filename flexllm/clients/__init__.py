"""
flexllm.clients - LLM 客户端实现

包含所有 LLM 客户端类：
- LLMClientBase: 抽象基类
- LLMClient: 统一入口（推荐）
- OpenAIClient: OpenAI 兼容 API
- GeminiClient: Google Gemini API
- ClaudeClient: Anthropic Claude API
- MllmClient: 多模态客户端
- LLMClientPool: 多 Endpoint 负载均衡
- ChainOfThoughtClient: 链式推理
"""

from .base import BatchResultItem, ChatCompletionResult, LLMClientBase, ToolCall
from .chain_of_thought import ChainOfThoughtClient, Step
from .claude import ClaudeClient
from .gemini import GeminiClient
from .llm import LLMClient
from .mllm import MllmClient
from .openai import OpenAIClient
from .pool import EndpointConfig, LLMClientPool
from .router import ProviderConfig, ProviderRouter, create_router_from_urls

__all__ = [
    # 基础类
    "LLMClientBase",
    "ChatCompletionResult",
    "BatchResultItem",
    "ToolCall",
    # 客户端
    "LLMClient",
    "OpenAIClient",
    "GeminiClient",
    "ClaudeClient",
    "MllmClient",
    # 客户端池
    "LLMClientPool",
    "EndpointConfig",
    # Provider 路由
    "ProviderRouter",
    "ProviderConfig",
    "create_router_from_urls",
    # Chain of Thought
    "ChainOfThoughtClient",
    "Step",
]
