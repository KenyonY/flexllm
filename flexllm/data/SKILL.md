---
name: flexllm
description: LLM API 客户端 - 批量处理、断点续传、响应缓存、负载均衡、成本追踪
---

# flexllm - 生产级高性能 LLM 客户端

## 核心特性

| 特性 | 说明 |
|------|------|
| 断点续传 | 批处理自动恢复，百万请求安全处理 |
| 响应缓存 | 内置缓存，支持 TTL 和 IPC 多进程共享 |
| 负载均衡 | 多 Endpoint 分发，自动故障转移 |
| 成本追踪 | 实时成本监控，预算控制 |
| 高性能异步 | 并发控制、QPS 限制、流式输出 |
| 多 Provider | 支持 OpenAI 兼容 API、Gemini、Claude |

## Python API

### 基本用法

```python
from flexllm import LLMClient

# 推荐：使用上下文管理器
async with LLMClient(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key"
) as client:
    response = await client.chat_completions([
        {"role": "user", "content": "Hello!"}
    ])

# 同步版本
with LLMClient(model="gpt-4", base_url="...", api_key="...") as client:
    response = client.chat_completions_sync([
        {"role": "user", "content": "Hello!"}
    ])

# 获取 token 用量
result = await client.chat_completions(
    messages=[{"role": "user", "content": "Hello!"}],
    return_usage=True,
)
print(f"Tokens: {result.usage}")
```

### 多 Provider 支持

```python
# OpenAI（从 base_url 自动检测）
client = LLMClient(base_url="https://api.openai.com/v1", model="gpt-4o", api_key="...")

# Gemini
client = LLMClient(provider="gemini", model="gemini-2.5-flash", api_key="...")

# Claude
client = LLMClient(provider="claude", model="claude-sonnet-4-20250514", api_key="...")

# 自托管（vLLM, Ollama 等）
client = LLMClient(base_url="http://localhost:8000/v1", model="qwen2.5")
```

### 批处理 + 断点续传

```python
messages_list = [
    [{"role": "user", "content": f"Question {i}"}]
    for i in range(100000)
]

# 中断后重新运行会自动从断点继续
results = await client.chat_completions_batch(
    messages_list,
    output_jsonl="results.jsonl",  # 进度保存在这里
    show_progress=True,
    track_cost=True,  # 实时显示成本
)
```

### 响应缓存

```python
from flexllm import LLMClient, ResponseCacheConfig

client = LLMClient(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    cache=ResponseCacheConfig(enabled=True, ttl=3600),  # 1 小时 TTL
)

result1 = await client.chat_completions(messages)  # API 请求
result2 = await client.chat_completions(messages)  # 缓存命中，$0
```

### 成本追踪

```python
results, cost_report = await client.chat_completions_batch(
    messages_list,
    return_cost_report=True,
)
print(f"Total cost: ${cost_report.total_cost:.4f}")

# 进度条中实时显示成本
results = await client.chat_completions_batch(messages_list, track_cost=True)
```

### 流式输出

```python
# Token 级流式
async for chunk in client.chat_completions_stream(messages):
    print(chunk, end="", flush=True)

# 批处理流式（结果完成即返回）
async for result in client.iter_chat_completions_batch(messages_list):
    process(result)
```

### Tool Calls (Function Calling)

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather info",
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

### Thinking Mode（推理模型）

统一接口支持 DeepSeek-R1、Qwen3、Claude extended thinking、Gemini thinking：

```python
result = await client.chat_completions(messages, thinking=True, return_raw=True)

parsed = client.parse_thoughts(result.data)
print("Thinking:", parsed["thought"])
print("Answer:", parsed["answer"])
```

### 负载均衡 (LLMClientPool)

```python
from flexllm import LLMClientPool

pool = LLMClientPool(
    endpoints=[
        {"base_url": "http://gpu1:8000/v1", "model": "qwen", "concurrency_limit": 50},
        {"base_url": "http://gpu2:8000/v1", "model": "qwen", "concurrency_limit": 20},
        {"base_url": "http://gpu3:8000/v1", "model": "qwen", "weight": 2.0},
    ],
    load_balance="round_robin",  # 或 "weighted", "random", "fallback"
    fallback=True,               # 失败自动切换
    failure_threshold=3,         # 连续失败 3 次后标记不健康（默认不启用）
    recovery_time=60.0,          # 60 秒后尝试恢复
)

# 单次请求
result = await pool.chat_completions(messages)

# 批量处理，动态负载均衡
results = await pool.chat_completions_batch(
    messages_list,
    distribute=True,
    output_jsonl="results.jsonl",
    track_cost=True,
)

# 流式输出
async for chunk in pool.chat_completions_stream(messages):
    print(chunk, end="", flush=True)
```

## LLMClient 参数

```python
LLMClient(
    provider: str = "auto",        # "auto", "openai", "gemini", "claude"
    model: str,                    # 模型名
    base_url: str = None,          # API 地址（openai 必需）
    api_key: str = "EMPTY",        # API Key
    cache: ResponseCacheConfig,    # 缓存配置
    concurrency_limit: int = 10,   # 最大并发
    max_qps: float = None,         # 最大 QPS
    retry_times: int = 3,          # 重试次数
    timeout: int = 120,            # 超时（秒）
)
```

## CLI 命令

```bash
# 快速问答
flexllm ask "What is Python?"
flexllm ask "总结一下" < article.txt   # 管道输入

# 交互式聊天
flexllm chat
flexllm chat --model gpt-4o

# 批处理（支持断点续传）
flexllm batch input.jsonl -o output.jsonl
flexllm batch input.jsonl -o output.jsonl --track-cost  # 成本追踪

# 模型管理
flexllm list              # 已配置模型
flexllm models            # 远程可用模型
flexllm test              # 测试连接
flexllm set-model gpt-4   # 设置默认模型
flexllm init              # 初始化配置文件

# 定价与余额
flexllm pricing                 # 列出所有模型定价
flexllm pricing gpt-4o          # 查询特定模型定价
flexllm pricing --update        # 从 OpenRouter 更新定价表
flexllm credits                 # 查询当前 API Key 余额
flexllm credits -m grok-4       # 查询指定模型的 Key 余额

# Mock 服务器（测试/开发用）
flexllm mock                    # 启动 Mock 服务，端口 8001
flexllm mock -p 8080            # 指定端口
flexllm mock -d 0.5             # 固定延迟 0.5s
flexllm mock -d 1-5             # 随机延迟 1-5s
flexllm mock --error-rate 0.1   # 10% 请求返回错误

# 版本
flexllm version
```

## 配置

配置文件：`~/.flexllm/config.yaml`

```yaml
default: "gpt-4"

models:
  - id: gpt-4
    name: gpt-4
    provider: openai
    base_url: https://api.openai.com/v1
    api_key: your-api-key

  - id: local-ollama
    name: local-ollama
    provider: openai
    base_url: http://localhost:11434/v1
    api_key: EMPTY

batch:
  concurrency: 20
  cache: true
  track_cost: true
```

环境变量：
- `FLEXLLM_BASE_URL` / `OPENAI_BASE_URL`
- `FLEXLLM_API_KEY` / `OPENAI_API_KEY`
- `FLEXLLM_MODEL` / `OPENAI_MODEL`
