<h1 align="center">flexllm</h1>

<p align="center">
    <strong>High-performance LLM client with batch processing, caching, and checkpoint recovery</strong>
</p>

<p align="center">
    <a href="https://pypi.org/project/flexllm/">
        <img src="https://img.shields.io/pypi/v/flexllm?color=brightgreen&style=flat-square" alt="PyPI version">
    </a>
    <a href="https://github.com/KenyonY/flexllm/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/KenyonY/flexllm.svg?color=blue&style=flat-square">
    </a>
</p>

---

## Features

- **Multi-Provider**: OpenAI, Gemini, Claude, and any OpenAI-compatible API (vLLM, Ollama, DeepSeek...)
- **Batch Processing**: Concurrent requests with QPS control and checkpoint recovery
- **Response Caching**: Avoid duplicate API calls with TTL support
- **Async-First**: Built on asyncio for maximum performance

## Installation

```bash
pip install flexllm[all]
```

## Quick Start

```python
from flexllm import LLMClient

client = LLMClient(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key"
)

# Single request
response = await client.chat_completions([
    {"role": "user", "content": "Hello!"}
])

# Batch with checkpoint recovery (re-run resumes from where it stopped)
results = await client.chat_completions_batch(
    messages_list,
    output_file="results.jsonl",
)
```

### Response Caching

```python
from flexllm import LLMClient, ResponseCacheConfig

client = LLMClient(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    cache=ResponseCacheConfig(enabled=True, ttl=3600),
)

result1 = await client.chat_completions(messages)  # API call
result2 = await client.chat_completions(messages)  # Cache hit
```

### Load Balancing

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
flexllm ask "What is Python?"      # Quick ask
flexllm chat                       # Interactive chat
flexllm test                       # Test connection
flexllm init                       # Initialize config
```

## License

[Apache 2.0](LICENSE)
