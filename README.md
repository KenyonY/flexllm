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
    <a href="https://pypistats.org/packages/flexllm">
        <img alt="pypi downloads" src="https://img.shields.io/pypi/dm/flexllm?style=flat-square">
    </a>
</p>

---

## Why flexllm?

**flexllm** is designed for production LLM applications where reliability and efficiency matter:

- **One Interface, Multiple Providers**: Write code once, switch between OpenAI, Gemini, Claude, or self-hosted models (vLLM, Ollama) without changing your application logic.

- **Production-Ready**: Built-in retry, timeout, QPS limiting, and checkpoint recovery - handle API failures gracefully without losing progress on large batch jobs.

- **Simple by Design**: KISS principle - minimal configuration, sensible defaults, and a clean API that stays out of your way.

## Features

- **Batch Processing**: Process thousands of requests concurrently with QPS control
- **Response Caching**: Built-in caching with TTL support, avoid duplicate API calls
- **Checkpoint Recovery**: Resume interrupted batch jobs automatically
- **Multi-Provider**: OpenAI, Gemini, Claude, and any OpenAI-compatible API (vLLM, Ollama, DeepSeek, Qwen...)
- **Function Calling**: Unified tool use support across providers
- **Multi-Modal**: Image + text processing with automatic base64 encoding
- **Load Balancing**: Multi-endpoint client pool with failover
- **Async-First**: Built on asyncio for maximum performance
- **CLI Tool**: Quick ask, chat, and test commands

## Installation

```bash
pip install flexllm

# With caching support
pip install flexllm[cache]

# With CLI support
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
    [{"role": "user", "content": "What is 1+1?"}],
    [{"role": "user", "content": "What is 2+2?"}],
    # ... thousands more
]

# Batch processing with checkpoint recovery
# If interrupted, re-running will resume from where it stopped
results = await client.chat_completions_batch(
    messages_list,
    output_file="results.jsonl",  # Auto-save progress
    show_progress=True,
)
```

### Response Caching

```python
from flexllm import LLMClient, ResponseCacheConfig

# Enable caching (avoid duplicate API calls)
client = LLMClient(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    cache=ResponseCacheConfig(enabled=True, ttl=3600),  # 1 hour TTL
)

# Duplicate requests hit cache automatically
result1 = await client.chat_completions(messages)  # API call
result2 = await client.chat_completions(messages)  # Cache hit (instant)
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
            {"type": "image_url", "image_url": {"url": "path/to/image.jpg"}}  # Local path or URL
        ]
    }
]

response = await client.call_llm([messages])
```

### Load Balancing with Failover

```python
from flexllm import LLMClientPool

# Create client pool with multiple endpoints
pool = LLMClientPool(
    endpoints=[
        {"base_url": "http://host1:8000/v1", "api_key": "key1", "model": "qwen"},
        {"base_url": "http://host2:8000/v1", "api_key": "key2", "model": "qwen"},
    ],
    load_balance="round_robin",  # round_robin, weighted, random, fallback
    fallback=True,  # Auto switch on failure
)

# Same API as LLMClient
result = await pool.chat_completions(messages)

# Distribute batch requests across endpoints
results = await pool.chat_completions_batch(messages_list, distribute=True)
```

### Gemini Client

```python
from flexllm import GeminiClient

# Gemini Developer API
client = GeminiClient(
    model="gemini-3-flash-preview",
    api_key="your-gemini-api-key"
)

# With thinking mode
response = await client.chat_completions(
    messages,
    thinking="high",  # False, True, "minimal", "low", "medium", "high"
)

# Vertex AI mode
client = GeminiClient(
    model="gemini-3-flash-preview",
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
    model="claude-3-5-sonnet-20241022",
)

response = await client.chat_completions([
    {"role": "user", "content": "Hello, Claude!"}
])

# Or use ClaudeClient directly
client = ClaudeClient(
    api_key="your-anthropic-key",
    model="claude-3-5-sonnet-20241022",
)

# With extended thinking
response = await client.chat_completions(
    messages,
    thinking=True,  # or budget_tokens as int
    return_raw=True,
)
parsed = ClaudeClient.parse_thoughts(response.data)
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

# Returns ChatCompletionResult with tool_calls
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

### Thinking Mode (DeepSeek, etc.)

```python
from flexllm import OpenAIClient

client = OpenAIClient(
    base_url="https://api.deepseek.com/v1",
    api_key="your-key",
    model="deepseek-reasoner",
)

# Enable thinking
result = await client.chat_completions(
    messages,
    thinking=True,
    return_raw=True,
)

# Parse thinking content
parsed = OpenAIClient.parse_thoughts(result.data)
print("Thinking:", parsed["thought"])
print("Answer:", parsed["answer"])
```

## CLI Usage

```bash
# Quick ask (for scripts/agents)
flexllm ask "What is Python?"
flexllm ask "Explain this" -s "You are a code expert"
echo "long text" | flexllm ask "Summarize"

# Interactive chat
flexllm chat
flexllm chat "Hello"
flexllm chat --model=gpt-4 "Hello"

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

Main unified client for all providers.

```python
LLMClient(
    model: str,                    # Model name
    base_url: str,                 # API base URL
    api_key: str = "EMPTY",        # API key
    provider: str = "auto",        # "auto", "openai", "gemini", "claude"
    cache: ResponseCacheConfig = None,  # Cache config
    concurrency_limit: int = 50,   # Max concurrent requests
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
| `chat_completions_batch(messages_list)` | Batch async requests |
| `chat_completions_batch_sync(messages_list)` | Batch sync requests |
| `chat_completions_stream(messages)` | Streaming response |

### ResponseCacheConfig

```python
ResponseCacheConfig(
    enabled: bool = False,         # Enable caching
    ttl: int = 86400,              # Time-to-live in seconds (default 24h)
    cache_dir: str = "~/.cache/flexllm/llm_response",
    use_ipc: bool = True,          # Use IPC for multi-process sharing
)

# Shortcuts
ResponseCacheConfig.with_ttl(3600)     # 1 hour TTL
ResponseCacheConfig.persistent()        # Never expire
```

### Token Counting

```python
from flexllm import count_tokens, estimate_cost, estimate_batch_cost

# Count tokens
tokens = count_tokens("Hello world", model="gpt-4")

# Estimate cost
cost = estimate_cost(tokens, model="gpt-4", is_input=True)

# Estimate batch cost
total_cost = estimate_batch_cost(messages_list, model="gpt-4")
```

## Architecture

```
flexllm/
├── flexllm/
│   ├── llm_client.py          # Unified client (recommended)
│   ├── openaiclient.py        # OpenAI-compatible API
│   ├── geminiclient.py        # Google Gemini
│   ├── claudeclient.py        # Anthropic Claude
│   ├── mllm_client.py         # Multi-modal client
│   ├── client_pool.py         # Load balancing pool
│   ├── response_cache.py      # Response caching
│   ├── token_counter.py       # Token counting & cost
│   ├── async_api/             # Async engine
│   └── processors/            # Image & message processing
```

## License

MIT
