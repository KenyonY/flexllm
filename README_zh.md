<h1 align="center">flexllm</h1>

<p align="center">
    <strong>高性能 LLM 客户端，支持批量处理、响应缓存和断点续传</strong>
</p>

<p align="center">
    <a href="https://pypi.org/project/flexllm/">
        <img src="https://img.shields.io/pypi/v/flexllm?color=brightgreen&style=flat-square" alt="PyPI version">
    </a>
    <a href="https://github.com/KenyonY/flexllm/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/KenyonY/flexllm.svg?color=blue&style=flat-square">
    </a>
    <a href="https://pypistats.org/packages/flexllm">
        <img alt="pypi downloads" src="https://img.shields.io/pypi/dm/flexllm?style=flat-square">
    </a>
</p>

<p align="center">
    <a href="README.md">English</a> | 中文
</p>

---

## 为什么选择 flexllm？

- **批量处理无忧**：断点续传 + 响应缓存，跑一半挂了不丢进度，重复请求不花钱
- **统一接口**：一套代码适配 OpenAI、Gemini、Claude 及自托管模型
- **开箱即用**：合理默认值，最少配置

## 特性

- **多 Provider 支持**：OpenAI、Gemini、Claude 及任何 OpenAI 兼容 API（vLLM、Ollama、DeepSeek...）
- **批量处理**：并发请求 + QPS 控制 + 断点续传
- **响应缓存**：避免重复 API 调用，支持 TTL
- **异步优先**：基于 asyncio 构建，性能最大化

## 安装

```bash
pip install flexllm[all]
```

## 快速开始

```python
from flexllm import LLMClient

client = LLMClient(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key"
)

# 单次请求
response = await client.chat_completions([
    {"role": "user", "content": "Hello!"}
])

# 批量处理 + 断点续传（中断后重新运行会自动恢复）
results = await client.chat_completions_batch(
    messages_list,
    output_file="results.jsonl",
)
```

### 响应缓存

```python
from flexllm import LLMClient, ResponseCacheConfig

client = LLMClient(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    cache=ResponseCacheConfig(enabled=True, ttl=3600),
)

result1 = await client.chat_completions(messages)  # API 调用
result2 = await client.chat_completions(messages)  # 缓存命中
```

### 负载均衡

```python
from flexllm import LLMClientPool

pool = LLMClientPool(
    endpoints=[
        {"base_url": "http://host1:8000/v1", "api_key": "key1", "model": "qwen"},
        {"base_url": "http://host2:8000/v1", "api_key": "key2", "model": "qwen"},
    ],
    load_balance="round_robin",
    fallback=True,
)

results = await pool.chat_completions_batch(messages_list, distribute=True)
```

### Gemini / Claude

```python
from flexllm import GeminiClient, ClaudeClient

# Gemini
client = GeminiClient(model="gemini-2.5-flash", api_key="your-key")

# Claude
client = ClaudeClient(model="claude-sonnet-4-20250514", api_key="your-key")
```

## CLI

```bash
flexllm ask "What is Python?"                # 快速问答
flexllm batch input.jsonl -o output.jsonl    # 批量处理
flexllm chat                                 # 交互式聊天
flexllm test                                 # 测试连接
```

## 许可证

[Apache 2.0](LICENSE)
