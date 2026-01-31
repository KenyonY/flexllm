<h1 align="center">flexllm</h1>

<p align="center">
    <strong>High-Performance LLM Client for Production</strong><br>
    <em>Batch processing with checkpoint recovery, response caching, load balancing, and cost tracking</em>
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

**Built for production batch processing at scale.**

```python
from flexllm import LLMClient

client = LLMClient(base_url="https://api.openai.com/v1", model="gpt-4", api_key="...")

# Process 100k requests with automatic checkpoint recovery
# Interrupted at 50k? Just restart - it continues from 50,001
results = await client.chat_completions_batch(
    messages_list,
    output_jsonl="results.jsonl",  # Progress saved here
    show_progress=True,
    track_cost=True,  # Real-time cost display
)
```

**Scale out across multiple endpoints with zero code change.**

```python
from flexllm import LLMClientPool

# Same API, multiple GPU nodes — faster endpoints automatically handle more tasks
pool = LLMClientPool(
    endpoints=[
        {"base_url": "http://gpu1:8000/v1", "model": "qwen", "concurrency_limit": 50},
        {"base_url": "http://gpu2:8000/v1", "model": "qwen", "concurrency_limit": 20},
        {"base_url": "http://gpu3:8000/v1", "model": "qwen"},
    ],
    fallback=True,  # Auto-switch on endpoint failure
)

results = await pool.chat_completions_batch(messages_list, output_jsonl="results.jsonl")
```

---

## Features

| Feature                          | Description                                                                     |
| -------------------------------- | ------------------------------------------------------------------------------- |
| **Checkpoint Recovery**    | Batch jobs auto-resume from interruption - process millions of requests safely  |
| **Multi-Endpoint Pool**   | Distribute tasks across GPU nodes with shared-queue dynamic balancing and automatic failover |
| **Response Caching**       | Built-in caching with TTL and IPC multi-process sharing                         |
| **Cost Tracking**          | Real-time cost monitoring with budget control                                   |
| **High-Performance Async** | Fine-grained concurrency control, QPS limiting, and streaming                   |
| **Multi-Provider**         | Supports OpenAI-compatible APIs, Gemini, Claude                                 |

---

## Installation

```bash
pip install flexllm

# With all features
pip install flexllm[all]
```

### Claude Code Integration

Enable Claude Code to use flexllm for LLM API calls, batch processing, and more:

```bash
flexllm install-skill
```

After installation, Claude Code gains the ability to use flexllm across all your projects.

---

## Quick Start

### Basic Usage

```python
from flexllm import LLMClient

# Recommended: use context manager for proper resource cleanup
async with LLMClient(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key"
) as client:
    # Async call
    response = await client.chat_completions([
        {"role": "user", "content": "Hello!"}
    ])

# Sync version (also supports context manager)
with LLMClient(model="gpt-4", base_url="...", api_key="...") as client:
    response = client.chat_completions_sync([
        {"role": "user", "content": "Hello!"}
    ])

# Get token usage
result = await client.chat_completions(
    messages=[{"role": "user", "content": "Hello!"}],
    return_usage=True,  # Returns ChatCompletionResult with usage info
)
print(f"Tokens: {result.usage}")  # {'prompt_tokens': 10, 'completion_tokens': 5, ...}
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

### Multi-Endpoint Pool

Distribute batch tasks across multiple GPU nodes / API endpoints. Faster endpoints automatically handle more tasks via a shared queue model, with automatic failover and health monitoring.

> `LLMClient` and `LLMClientPool` share the same API. Single endpoint → use `LLMClient`; multiple endpoints → use `LLMClientPool`.

```python
from flexllm import LLMClientPool

pool = LLMClientPool(
    endpoints=[
        # Each endpoint can have independent rate limits
        {"base_url": "http://gpu1:8000/v1", "model": "qwen", "concurrency_limit": 50, "max_qps": 100},
        {"base_url": "http://gpu2:8000/v1", "model": "qwen", "concurrency_limit": 20, "max_qps": 50},
        {"base_url": "http://gpu3:8000/v1", "model": "qwen"},
    ],
    fallback=True,               # Auto-switch on endpoint failure
    failure_threshold=3,         # Mark unhealthy after 3 consecutive failures
    recovery_time=60.0,          # Try to recover after 60 seconds
)

# Single request — automatic failover across endpoints
result = await pool.chat_completions(messages)

# Distributed batch — shared queue, dynamic load balancing, checkpoint recovery
results = await pool.chat_completions_batch(
    messages_list,
    distribute=True,
    output_jsonl="results.jsonl",
    track_cost=True,
)

# Streaming with failover
async for chunk in pool.chat_completions_stream(messages):
    print(chunk, end="", flush=True)
```

**Highlights:**
- **Shared Queue**: Faster endpoints automatically pull more tasks — no manual tuning needed
- **Automatic Failover**: Failed requests retry on healthy endpoints; unhealthy nodes auto-recover
- **Per-Endpoint Config**: Independent `concurrency_limit` and `max_qps` for each endpoint
- **Full Feature Support**: Checkpoint recovery, caching, cost tracking all work with Pool

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

### Tool Calls (Function Calling)

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}]

result = await client.chat_completions(
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    return_usage=True,
)

if result.tool_calls:
    for call in result.tool_calls:
        print(f"Call: {call.function['name']}({call.function['arguments']})")
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
flexllm batch input.jsonl -o output.jsonl -n 5           # First 5 records only
flexllm batch data.jsonl -o out.jsonl -uf text -sf sys   # Custom field names

# Model management
flexllm list              # Configured models
flexllm models            # Remote available models
flexllm set-model gpt-4   # Set default model
flexllm test              # Test connection
flexllm init              # Initialize config file

# Utilities
flexllm pricing gpt-4     # Query model pricing
flexllm credits           # Check API key balance
flexllm mock              # Start mock LLM server for testing
```

### Configuration

Config file location: `~/.flexllm/config.yaml`

See [config.example.yaml](config.example.yaml) for a comprehensive configuration example with all available options, or [config.quickstart.yaml](config.quickstart.yaml) for a minimal quick-start template.

```yaml
# Default model
default: "gpt-4"

# Global system prompt (applied to all commands unless overridden)
system: "You are a helpful assistant."

# Global user content template (applied to all user messages unless overridden)
# Use {content} as placeholder for original user content
# user_template: "{content}/detail"

# Model list
models:
  - id: gpt-4
    name: gpt-4
    provider: openai
    base_url: https://api.openai.com/v1
    api_key: your-api-key
    system: "You are a GPT-4 assistant."  # Model-specific system prompt (optional)

  - id: local-finetuned
    name: local-finetuned
    provider: openai
    base_url: http://localhost:8000/v1
    api_key: EMPTY
    user_template: "{content}/detail"  # Model-specific user template for fine-tuned models (optional)

  - id: local-ollama
    name: local-ollama
    provider: openai
    base_url: http://localhost:11434/v1
    api_key: EMPTY

# Batch command config (optional)
batch:
  concurrency: 20
  cache: true
  track_cost: true
  system: "You are a batch processing assistant."  # Batch-specific system prompt (optional)
  # user_template: "[INST]{content}[/INST]"  # Batch-specific user template (optional)
```

**System prompt priority** (higher priority overrides lower):
1. CLI argument (`-s/--system`)
2. Batch config (`batch.system`)
3. Model config (`models[].system`)
4. Global config (`system`)

**User template priority** (higher priority overrides lower):
1. CLI argument (`--user-template`)
2. Batch config (`batch.user_template`)
3. Model config (`models[].user_template`)
4. Global config (`user_template`)

User template uses `{content}` as placeholder for original user content. Useful for fine-tuned models requiring specific prompt formats (e.g., `"{content}/detail"`, `"[INST]{content}[/INST]"`).

Environment variables (higher priority than config file):

- `FLEXLLM_BASE_URL` / `OPENAI_BASE_URL`
- `FLEXLLM_API_KEY` / `OPENAI_API_KEY`
- `FLEXLLM_MODEL` / `OPENAI_MODEL`

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
LLMClient (single endpoint)  /  LLMClientPool (multi-endpoint)
    │                                  │
    │                                  ├── ProviderRouter (round_robin)
    │                                  ├── Health Monitor (failure threshold + auto recovery)
    │                                  └── Shared Task Queue (dynamic load balancing)
    │                                  │
    └──────────── Backend Clients ─────┘
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

| Method                                         | Description                 |
| ---------------------------------------------- | --------------------------- |
| `chat_completions(messages)`                 | Single async request        |
| `chat_completions_sync(messages)`            | Single sync request         |
| `chat_completions_batch(messages_list)`      | Batch async with checkpoint |
| `iter_chat_completions_batch(messages_list)` | Streaming batch results     |
| `chat_completions_stream(messages)`          | Token-by-token streaming    |

---

## License

Apache 2.0
