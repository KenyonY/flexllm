<h1 align="center">flexllm</h1>

<p align="center">
    <strong>One Client, All LLMs</strong><br>
    <em>Production-grade LLM client with checkpoint recovery, response caching, and multi-provider support</em>
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

## Design Philosophy

**One unified entry point for all LLM providers.**

```python
from flexllm import LLMClient

# That's all you need to import. Everything else is configuration.
```

flexllm follows the **"Single Interface, Multiple Backends"** principle. Whether you're calling OpenAI, Gemini, Claude, or a self-hosted model, the API stays the same. Provider differences are abstracted away - you focus on your application logic, not on SDK quirks.

```python
# OpenAI GPT-4
client = LLMClient(base_url="https://api.openai.com/v1", model="gpt-4", api_key="...")

# Google Gemini
client = LLMClient(provider="gemini", model="gemini-2.0-flash", api_key="...")

# Anthropic Claude
client = LLMClient(provider="claude", model="claude-sonnet-4-20250514", api_key="...")

# Self-hosted (vLLM, Ollama, etc.)
client = LLMClient(base_url="http://localhost:8000/v1", model="qwen2.5")

# The API is identical for all:
result = await client.chat_completions(messages)
results = await client.chat_completions_batch(messages_list)
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Unified Interface** | One `LLMClient` for OpenAI, Gemini, Claude, and any OpenAI-compatible API |
| **Checkpoint Recovery** | Batch jobs auto-resume from interruption - process millions of requests safely |
| **Response Caching** | Built-in caching with TTL and IPC multi-process sharing |
| **Cost Tracking** | Real-time cost monitoring with budget control |
| **High-Performance Async** | Fine-grained concurrency control, QPS limiting, and streaming |
| **Load Balancing** | Multi-endpoint distribution with automatic failover |

---

## Installation

```bash
pip install flexllm

# With all features
pip install flexllm[all]
```

---

## Quick Start

### Basic Usage

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

Process millions of requests safely. If interrupted, just restart - it continues from where it left off.

```python
messages_list = [
    [{"role": "user", "content": f"Question {i}"}]
    for i in range(100000)
]

# Interrupted at 50,000? Re-run and it continues from 50,001.
results = await client.chat_completions_batch(
    messages_list,
    output_jsonl="results.jsonl",  # Progress saved here
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
    cache=ResponseCacheConfig(enabled=True, ttl=3600),  # 1 hour TTL
)

# First call: API request (~2s, ~$0.01)
result1 = await client.chat_completions(messages)

# Second call: Cache hit (~0.001s, $0)
result2 = await client.chat_completions(messages)
```

### Cost Tracking

```python
# Track costs during batch processing
results, cost_report = await client.chat_completions_batch(
    messages_list,
    return_cost_report=True,
)
print(f"Total cost: ${cost_report.total_cost:.4f}")

# Real-time cost display in progress bar
results = await client.chat_completions_batch(
    messages_list,
    track_cost=True,  # Shows 💰 $0.0012 in progress bar
)
```

### Streaming

```python
# Token-by-token streaming
async for chunk in client.chat_completions_stream(messages):
    print(chunk, end="", flush=True)

# Batch streaming - process results as they complete
async for result in client.iter_chat_completions_batch(messages_list):
    process(result)
```

### Multi-Provider Support

```python
from flexllm import LLMClient

# OpenAI (auto-detected from base_url)
client = LLMClient(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="gpt-4o",
)

# Gemini
client = LLMClient(
    provider="gemini",
    api_key="your-gemini-key",
    model="gemini-2.0-flash",
)

# Claude
client = LLMClient(
    provider="claude",
    api_key="your-anthropic-key",
    model="claude-sonnet-4-20250514",
)

# Self-hosted (vLLM, Ollama, etc.)
client = LLMClient(
    base_url="http://localhost:8000/v1",
    model="qwen2.5",
)
```

### Thinking Mode (Reasoning Models)

Unified interface for DeepSeek-R1, Qwen3, Claude extended thinking, Gemini thinking.

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

### Load Balancing

```python
from flexllm import LLMClientPool

pool = LLMClientPool(
    endpoints=[
        {"base_url": "http://gpu1:8000/v1", "model": "qwen"},
        {"base_url": "http://gpu2:8000/v1", "model": "qwen"},
    ],
    load_balance="round_robin",  # or "weighted", "random", "fallback"
    fallback=True,               # Auto-switch on failure
)

# Requests automatically distributed
results = await pool.chat_completions_batch(messages_list, distribute=True)
```

---

## CLI

```bash
# Quick ask
flexllm ask "What is Python?"

# Interactive chat
flexllm chat

# Batch processing with cost tracking
flexllm batch input.jsonl -o output.jsonl --track-cost

# Model management
flexllm list        # Configured models
flexllm models      # Remote available models
flexllm test        # Test connection
```

---

## Architecture

```
flexllm/
├── clients/           # All client implementations
│   ├── base.py        # Abstract base class (LLMClientBase)
│   ├── llm.py         # Unified entry point (LLMClient)
│   ├── openai.py      # OpenAI-compatible backend
│   ├── gemini.py      # Google Gemini backend
│   ├── claude.py      # Anthropic Claude backend
│   ├── pool.py        # Multi-endpoint load balancer
│   └── router.py      # Provider routing strategies
├── pricing/           # Cost estimation and tracking
│   ├── cost_tracker.py
│   └── token_counter.py
├── cache/             # Response caching with IPC
├── async_api/         # High-performance async engine
└── msg_processors/    # Multi-modal message processing
```

The architecture follows a simple layered design:

```
LLMClient (Unified entry point - recommended)
    │
    ├── Provider auto-detection or explicit selection
    │
    └── Backend Clients (internal)
            ├── OpenAIClient
            ├── GeminiClient
            └── ClaudeClient
                    │
                    └── LLMClientBase (Abstract - 4 methods to implement)
                            │
                            ├── ConcurrentRequester (Async engine)
                            ├── ResponseCache (Caching layer)
                            └── CostTracker (Cost monitoring)
```

---

## API Reference

### LLMClient

```python
LLMClient(
    provider: str = "auto",        # "auto", "openai", "gemini", "claude"
    model: str,                    # Model name
    base_url: str = None,          # API base URL (required for openai)
    api_key: str = "EMPTY",        # API key
    cache: ResponseCacheConfig,    # Cache config
    concurrency_limit: int = 10,   # Max concurrent requests
    max_qps: float = None,         # Max requests per second
    retry_times: int = 3,          # Retry count on failure
    timeout: int = 120,            # Request timeout (seconds)
)
```

### Main Methods

| Method | Description |
|--------|-------------|
| `chat_completions(messages)` | Single async request |
| `chat_completions_sync(messages)` | Single sync request |
| `chat_completions_batch(messages_list)` | Batch async with checkpoint |
| `iter_chat_completions_batch(messages_list)` | Streaming batch results |
| `chat_completions_stream(messages)` | Token-by-token streaming |

---

## License

Apache 2.0
