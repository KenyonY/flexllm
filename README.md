<h1 align="center">flexllm</h1>

<p align="center">
    <strong>Production-grade LLM client with checkpoint recovery, response caching, and multi-provider support</strong>
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

---

## Features

| Feature | Description |
|---------|-------------|
| **Checkpoint Recovery** | Batch jobs auto-resume from interruption - process millions of requests without losing progress |
| **Response Caching** | Built-in intelligent caching with TTL and IPC multi-process sharing - avoid duplicate API calls |
| **Multi-Provider** | One interface for OpenAI, Gemini, Claude, and any OpenAI-compatible API (vLLM, Ollama, etc.) |
| **High-Performance Async** | Fine-grained concurrency control, QPS limiting, and streaming batch results |
| **Load Balancing** | Multi-endpoint distribution with automatic failover (round_robin/weighted/random/fallback) |

---

## Core Strengths

### 1. Checkpoint Recovery - Never Lose Progress

Process millions of requests without fear of interruption. When your batch job crashes at 3 AM, just restart it - flexllm picks up exactly where it left off.

```python
# Process 100,000 requests - if interrupted, resume automatically
results = await client.chat_completions_batch(
    messages_list,
    output_jsonl="results.jsonl",  # Progress saved here
)
# Ctrl+C at 50,000? No problem. Re-run and it continues from 50,001.
```

### 2. Response Caching - Save Money, Save Time

Built-in intelligent caching avoids duplicate API calls. Same question? Instant answer from cache.

```python
client = LLMClient(
    model="gpt-4",
    cache=ResponseCacheConfig.with_ttl(3600),  # 1 hour cache
)

# First call: API request (~2s, ~$0.01)
result1 = await client.chat_completions(messages)

# Second call: Cache hit (~0.001s, $0)
result2 = await client.chat_completions(messages)
```

Supports multi-process cache sharing via IPC - perfect for distributed workloads.

### 3. One Interface, All Providers

Write once, run everywhere. Switch between OpenAI, Gemini, Claude, or self-hosted models without changing your code.

```python
# OpenAI
client = LLMClient(provider="openai", base_url="https://api.openai.com/v1", ...)

# Gemini
client = LLMClient(provider="gemini", api_key="...", model="gemini-2.0-flash")

# Claude
client = LLMClient(provider="claude", api_key="...", model="claude-sonnet-4-20250514")

# Self-hosted (vLLM, Ollama, etc.)
client = LLMClient(base_url="http://localhost:8000/v1", model="qwen2.5")

# Same API for all:
result = await client.chat_completions(messages)
```

### 4. High-Performance Async Engine

Maximize throughput with fine-grained concurrency control and QPS limiting.

```python
client = LLMClient(
    concurrency_limit=100,  # 100 concurrent requests
    max_qps=50,             # Rate limit: 50 req/sec
    retry_times=3,          # Auto-retry on failure
)

# Process 10,000 requests with optimal parallelism
results = await client.chat_completions_batch(messages_list, show_progress=True)
```

Streaming results - process results as they complete, don't wait for all:

```python
async for result in client.iter_chat_completions_batch(messages_list):
    process(result)  # Handle each result immediately
```

### 5. Load Balancing & Failover

Distribute workloads across multiple endpoints with automatic failover.

```python
pool = LLMClientPool(
    endpoints=[
        {"base_url": "http://gpu1:8000/v1", "model": "qwen"},
        {"base_url": "http://gpu2:8000/v1", "model": "qwen"},
        {"base_url": "http://gpu3:8000/v1", "model": "qwen"},
    ],
    load_balance="round_robin",  # or "weighted", "random", "fallback"
    fallback=True,               # Auto-switch on failure
)

# Requests automatically distributed across healthy endpoints
results = await pool.chat_completions_batch(messages_list, distribute=True)
```

### 6. Thinking Mode Support

Unified interface for reasoning models - DeepSeek-R1, Qwen3, Claude extended thinking, Gemini thinking.

```python
result = await client.chat_completions(
    messages,
    thinking=True,      # Enable thinking
    return_raw=True,
)

# Unified parsing across all providers
parsed = client.parse_thoughts(result.data)
print("Thinking:", parsed["thought"])
print("Answer:", parsed["answer"])
```

---

## Installation

```bash
pip install flexllm

# With caching support
pip install flexllm[cache]

# With CLI
pip install flexllm[cli]

# All features
pip install flexllm[all]
```

## Quick Start

### Single Request

```python
from flexllm import LLMClient

client = LLMClient(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key"
)

# Async
response = await client.chat_completions([
    {"role": "user", "content": "Hello!"}
])

# Sync
response = client.chat_completions_sync([
    {"role": "user", "content": "Hello!"}
])
```

### Batch Processing with Checkpoint Recovery

```python
from flexllm import LLMClient

client = LLMClient(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    concurrency_limit=50,
    max_qps=100,
)

messages_list = [
    [{"role": "user", "content": f"Question {i}"}]
    for i in range(10000)
]

# If interrupted, re-running resumes from where it stopped
results = await client.chat_completions_batch(
    messages_list,
    output_jsonl="results.jsonl",
    show_progress=True,
)
```

### Response Caching

```python
from flexllm import LLMClient, ResponseCacheConfig

client = LLMClient(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    cache=ResponseCacheConfig.with_ttl(3600),  # 1 hour TTL
)

# Duplicate requests hit cache automatically
result1 = await client.chat_completions(messages)  # API call
result2 = await client.chat_completions(messages)  # Cache hit (instant)

# Multi-process cache sharing (IPC mode - default)
cache = ResponseCacheConfig.ipc(ttl=86400)  # 24h, shared across processes
```

### Streaming Response

```python
async for chunk in client.chat_completions_stream(messages):
    print(chunk, end="", flush=True)
```

### Multi-Modal (Vision)

```python
from flexllm import MllmClient

client = MllmClient(
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    model="gpt-4o",
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "path/to/image.jpg"}}
        ]
    }
]

response = await client.call_llm([messages])
```

### Load Balancing with Failover

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

# Single request with automatic failover
result = await pool.chat_completions(messages)

# Batch requests distributed across endpoints
results = await pool.chat_completions_batch(messages_list, distribute=True)
```

### Gemini Client

```python
from flexllm import GeminiClient

# Gemini Developer API
client = GeminiClient(
    model="gemini-2.0-flash",
    api_key="your-gemini-api-key"
)

# With thinking mode
response = await client.chat_completions(
    messages,
    thinking="high",  # False, True, "minimal", "low", "medium", "high"
)

# Vertex AI mode
client = GeminiClient(
    model="gemini-2.0-flash",
    project_id="your-project-id",
    location="us-central1",
    use_vertex_ai=True,
)
```

### Claude Client

```python
from flexllm import LLMClient, ClaudeClient

# Using unified LLMClient (recommended)
client = LLMClient(
    provider="claude",
    api_key="your-anthropic-key",
    model="claude-sonnet-4-20250514",
)

response = await client.chat_completions([
    {"role": "user", "content": "Hello, Claude!"}
])

# With extended thinking
result = await client.chat_completions(
    messages,
    thinking=True,
    return_raw=True,
)
parsed = client.parse_thoughts(result.data)
```

### Function Calling (Tool Use)

```python
from flexllm import LLMClient

client = LLMClient(
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    model="gpt-4",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }
]

result = await client.chat_completions(
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    return_usage=True,
)

if result.tool_calls:
    for tool_call in result.tool_calls:
        print(f"Function: {tool_call.function['name']}")
        print(f"Arguments: {tool_call.function['arguments']}")
```

## CLI Usage

```bash
# Quick ask
flexllm ask "What is Python?"
flexllm ask "Explain this" -s "You are a code expert"
echo "long text" | flexllm ask "Summarize"

# Interactive chat
flexllm chat
flexllm chat --model=gpt-4 "Hello"

# Batch processing with checkpoint recovery
flexllm batch input.jsonl -o output.jsonl

# List models
flexllm models           # Remote models
flexllm list_models      # Configured models

# Test connection
flexllm test

# Initialize config
flexllm init
```

### CLI Configuration

Create `~/.flexllm/config.yaml`:

```yaml
default: "gpt-4"

models:
  - id: gpt-4
    name: gpt-4
    provider: openai
    base_url: https://api.openai.com/v1
    api_key: your-api-key

  - id: local
    name: local-ollama
    provider: openai
    base_url: http://localhost:11434/v1
    api_key: EMPTY
```

Or use environment variables:

```bash
export FLEXLLM_BASE_URL="https://api.openai.com/v1"
export FLEXLLM_API_KEY="your-key"
export FLEXLLM_MODEL="gpt-4"
```

## API Reference

### LLMClient

```python
LLMClient(
    provider: str = "auto",        # "auto", "openai", "gemini", "claude"
    model: str,                    # Model name
    base_url: str,                 # API base URL
    api_key: str = "EMPTY",        # API key
    cache: ResponseCacheConfig,    # Cache config
    concurrency_limit: int = 10,   # Max concurrent requests
    max_qps: float = None,         # Max requests per second
    retry_times: int = 3,          # Retry count on failure
    retry_delay: float = 1.0,      # Delay between retries
    timeout: int = 120,            # Request timeout (seconds)
)
```

### Methods

| Method | Description |
|--------|-------------|
| `chat_completions(messages)` | Single async request |
| `chat_completions_sync(messages)` | Single sync request |
| `chat_completions_batch(messages_list)` | Batch async with checkpoint |
| `chat_completions_batch_sync(messages_list)` | Batch sync with checkpoint |
| `iter_chat_completions_batch(messages_list)` | Streaming batch results |
| `chat_completions_stream(messages)` | Token-by-token streaming |
| `parse_thoughts(response_data)` | Parse thinking content |

### ResponseCacheConfig

```python
# Shortcuts
ResponseCacheConfig.with_ttl(3600)     # 1 hour TTL
ResponseCacheConfig.persistent()        # Never expire
ResponseCacheConfig.ipc(ttl=86400)      # Multi-process shared (default)
ResponseCacheConfig.local(ttl=86400)    # Single process only

# Full config
ResponseCacheConfig(
    enabled: bool = False,
    ttl: int = 86400,              # Time-to-live in seconds
    cache_dir: str = "~/.cache/flexllm/llm_response",
    use_ipc: bool = True,          # Multi-process cache sharing
)
```

### Token Counting

```python
from flexllm import count_tokens, estimate_cost, estimate_batch_cost

tokens = count_tokens("Hello world", model="gpt-4")
cost = estimate_cost(tokens, model="gpt-4", is_input=True)
total_cost = estimate_batch_cost(messages_list, model="gpt-4")
```

## Architecture

```
LLMClient (Unified entry point)
    ├── OpenAIClient (OpenAI-compatible APIs)
    ├── GeminiClient (Google Gemini)
    └── ClaudeClient (Anthropic Claude)
            │
            └── LLMClientBase (Abstract base - 4 methods to implement)
                    │
                    ├── ConcurrentRequester (Async engine with QPS control)
                    ├── ResponseCache (FlaxKV2-based caching with IPC)
                    └── ImageProcessor (Multi-modal support)

LLMClientPool (Multi-endpoint load balancing)
    └── ProviderRouter (round_robin / weighted / random / fallback)
```

## License

Apache 2.0
