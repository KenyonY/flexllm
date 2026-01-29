<h1 align="center">flexllm</h1>

<p align="center">
    <strong>生产级高性能 LLM 客户端</strong><br>
    <em>批量处理 + 断点续传、响应缓存、负载均衡、成本追踪</em>
</p>

<p align="center">
    <a href="https://pypi.org/project/flexllm/">
        <img src="https://img.shields.io/pypi/v/flexllm?color=brightgreen&style=flat-square" alt="PyPI version">
    </a>
    <a href="https://github.com/KenyonY/flexllm/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/KenyonY/flexllm.svg?color=blue&style=flat-square">
    </a>
</p>

<p align="center">
    <a href="README.md">English</a> | 中文
</p>

---

## 为什么选择 flexllm？

**专为大规模生产级批量处理而设计。**

```python
from flexllm import LLMClient

client = LLMClient(base_url="https://api.openai.com/v1", model="gpt-4", api_key="...")

# 处理 10 万条请求，支持自动断点续传
# 50000 条时中断？重新运行，从 50001 继续
results = await client.chat_completions_batch(
    messages_list,
    output_jsonl="results.jsonl",  # 进度保存在此
    show_progress=True,
    track_cost=True,  # 实时显示成本
)
```

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **断点续传** | 批量任务自动恢复，百万级请求安全处理 |
| **响应缓存** | 内置缓存，支持 TTL 和 IPC 多进程共享 |
| **负载均衡** | 多 Endpoint 动态分发，自动故障转移 |
| **成本追踪** | 实时成本监控，支持预算控制 |
| **高性能异步** | 精细并发控制、QPS 限流、流式处理 |
| **多 Provider** | 支持 OpenAI 兼容 API、Gemini、Claude |

---

## 安装

```bash
pip install flexllm

# 完整功能
pip install flexllm[all]
```

### Claude Code 集成

让 Claude Code 学会使用 flexllm 进行 LLM API 调用、批量处理等操作：

```bash
flexllm install-skill
```

安装后，Claude Code 在任何项目中都能使用 flexllm。

---

## 快速开始

### 基本用法

```python
from flexllm import LLMClient

# 推荐：使用上下文管理器自动管理资源
async with LLMClient(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key"
) as client:
    # 异步调用
    response = await client.chat_completions([
        {"role": "user", "content": "你好！"}
    ])

# 同步版本（同样支持上下文管理器）
with LLMClient(model="gpt-4", base_url="...", api_key="...") as client:
    response = client.chat_completions_sync([
        {"role": "user", "content": "你好！"}
    ])

# 获取 token 用量
result = await client.chat_completions(
    messages=[{"role": "user", "content": "你好！"}],
    return_usage=True,  # 返回包含 usage 信息的 ChatCompletionResult
)
print(f"Token 用量: {result.usage}")  # {'prompt_tokens': 10, 'completion_tokens': 5, ...}
```

### 批量处理 + 断点续传

安全处理百万级请求。中断后重启，自动从断点继续。

```python
messages_list = [
    [{"role": "user", "content": f"问题 {i}"}]
    for i in range(100000)
]

# 50000 条时中断？重新运行，从 50001 继续
results = await client.chat_completions_batch(
    messages_list,
    output_jsonl="results.jsonl",  # 进度保存在此
    show_progress=True,
)
```

### 响应缓存

```python
from flexllm import LLMClient, ResponseCacheConfig

client = LLMClient(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    cache=ResponseCacheConfig(enabled=True, ttl=3600),  # 1小时 TTL
)

# 首次调用：API 请求 (~2秒, ~$0.01)
result1 = await client.chat_completions(messages)

# 再次调用：缓存命中 (~0.001秒, $0)
result2 = await client.chat_completions(messages)
```

### 成本追踪

```python
# 批量处理时追踪成本
results, cost_report = await client.chat_completions_batch(
    messages_list,
    return_cost_report=True,
)
print(f"总成本: ${cost_report.total_cost:.4f}")

# 进度条实时显示成本
results = await client.chat_completions_batch(
    messages_list,
    track_cost=True,  # 进度条显示 💰 $0.0012
)
```

### 流式输出

```python
# 逐 token 流式输出
async for chunk in client.chat_completions_stream(messages):
    print(chunk, end="", flush=True)

# 批量流式 - 结果完成即返回
async for result in client.iter_chat_completions_batch(messages_list):
    process(result)
```

### 思考模式（推理模型）

统一接口支持 DeepSeek-R1、Qwen3、Claude 扩展思考、Gemini 思考模式。

```python
result = await client.chat_completions(
    messages,
    thinking=True,      # 启用思考
    return_raw=True,
)

# 跨 Provider 统一解析
parsed = client.parse_thoughts(result.data)
print("思考过程:", parsed["thought"])
print("答案:", parsed["answer"])
```

### 工具调用（Function Calling）

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取天气信息",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}]

result = await client.chat_completions(
    messages=[{"role": "user", "content": "东京天气怎么样？"}],
    tools=tools,
    return_usage=True,
)

if result.tool_calls:
    for call in result.tool_calls:
        print(f"调用: {call.function['name']}({call.function['arguments']})")
```

### 负载均衡（LLMClientPool）

多 Endpoint 负载均衡，支持自动故障转移、健康检查和动态任务分配。

```python
from flexllm import LLMClientPool

pool = LLMClientPool(
    endpoints=[
        # 每个 endpoint 可独立配置限流参数
        {"base_url": "http://gpu1:8000/v1", "model": "qwen", "concurrency_limit": 50, "max_qps": 100},
        {"base_url": "http://gpu2:8000/v1", "model": "qwen", "concurrency_limit": 20, "max_qps": 50},
        {"base_url": "http://gpu3:8000/v1", "model": "qwen"},
    ],
    fallback=True,               # endpoint 故障时自动切换
    failure_threshold=3,         # 连续失败 3 次后标记为不健康
    recovery_time=60.0,          # 60 秒后尝试恢复
)

# 单次请求，自动故障转移
result = await pool.chat_completions(messages)

# 批量处理，动态负载均衡
# 快的 endpoint 自动处理更多任务（共享队列模型）
results = await pool.chat_completions_batch(
    messages_list,
    distribute=True,      # 启用分布式处理
    output_jsonl="results.jsonl",  # 支持断点续传
    track_cost=True,
)

# 流式输出，支持故障转移
async for chunk in pool.chat_completions_stream(messages):
    print(chunk, end="", flush=True)

# 查看池统计信息
print(pool.stats)  # {'num_endpoints': 3, 'router_stats': {...}}
```

**核心特性：**
- **动态负载均衡**：共享队列模型，快的 endpoint 自动处理更多任务
- **自动故障转移**：失败请求自动在其他健康 endpoint 重试
- **健康监控**：不健康的 endpoint 在 `recovery_time` 后自动恢复
- **独立配置**：每个 endpoint 可独立设置 `concurrency_limit`、`max_qps`
- **完整功能支持**：断点续传、响应缓存、成本追踪均可在 Pool 中使用

---

## CLI

```bash
# 快速问答
flexllm ask "Python 是什么？"

# 交互对话
flexllm chat

# 批量处理 + 成本追踪
flexllm batch input.jsonl -o output.jsonl --track-cost

# 模型管理
flexllm list              # 已配置模型
flexllm models            # 远程可用模型
flexllm set-model gpt-4   # 设置默认模型
flexllm test              # 测试连接
flexllm init              # 初始化配置文件

# 实用工具
flexllm pricing gpt-4     # 查询模型定价
flexllm credits           # 查询 API Key 余额
flexllm mock              # 启动 Mock 服务器（测试用）
```

### 配置文件

配置文件位置：`~/.flexllm/config.yaml`

```yaml
# 默认模型
default: "gpt-4"

# 模型列表
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

# batch 命令配置（可选）
batch:
  concurrency: 20
  cache: true
  track_cost: true
```

环境变量（优先级高于配置文件）：
- `FLEXLLM_BASE_URL` / `OPENAI_BASE_URL`
- `FLEXLLM_API_KEY` / `OPENAI_API_KEY`
- `FLEXLLM_MODEL` / `OPENAI_MODEL`

---

## 架构

```
flexllm/
├── clients/           # 所有客户端实现
│   ├── base.py        # 抽象基类 (LLMClientBase)
│   ├── llm.py         # 统一入口 (LLMClient)
│   ├── openai.py      # OpenAI 兼容后端
│   ├── gemini.py      # Google Gemini 后端
│   ├── claude.py      # Anthropic Claude 后端
│   ├── pool.py        # 多 Endpoint 负载均衡
│   └── router.py      # Provider 路由策略
├── pricing/           # 成本估算和追踪
├── cache/             # 响应缓存 (支持 IPC)
├── async_api/         # 高性能异步引擎
└── msg_processors/    # 多模态消息处理
```

分层设计：

```
LLMClient (统一入口 - 推荐使用)
    │
    ├── Provider 自动识别或显式指定
    │
    └── 后端客户端 (内部)
            ├── OpenAIClient
            ├── GeminiClient
            └── ClaudeClient
                    │
                    └── LLMClientBase (抽象基类 - 只需实现4个方法)
                            │
                            ├── ConcurrentRequester (异步引擎)
                            ├── ResponseCache (缓存层)
                            └── CostTracker (成本监控)
```

---

## 许可证

[Apache 2.0](LICENSE)
