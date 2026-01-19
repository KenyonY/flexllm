<h1 align="center">flexllm</h1>

<p align="center">
    <strong>一个客户端，所有大模型</strong><br>
    <em>生产级 LLM 客户端，支持断点续传、响应缓存、多 Provider 统一接口</em>
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

## 设计理念

**一个统一入口，适配所有 LLM 服务商。**

```python
from flexllm import LLMClient

# 只需导入这一个类，其他都是配置。
```

flexllm 遵循 **"单一接口，多后端"** 原则。无论调用 OpenAI、Gemini、Claude 还是自建模型，API 完全一致。Provider 差异被抽象封装，你只需关注业务逻辑。

```python
# OpenAI GPT-4
client = LLMClient(base_url="https://api.openai.com/v1", model="gpt-4", api_key="...")

# Google Gemini
client = LLMClient(provider="gemini", model="gemini-2.0-flash", api_key="...")

# Anthropic Claude
client = LLMClient(provider="claude", model="claude-sonnet-4-20250514", api_key="...")

# 自建服务 (vLLM, Ollama 等)
client = LLMClient(base_url="http://localhost:8000/v1", model="qwen2.5")

# API 完全一致：
result = await client.chat_completions(messages)
results = await client.chat_completions_batch(messages_list)
```

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **统一接口** | 一个 `LLMClient` 适配 OpenAI、Gemini、Claude 及所有 OpenAI 兼容 API |
| **断点续传** | 批量任务自动恢复，百万级请求安全处理 |
| **响应缓存** | 内置缓存，支持 TTL 和 IPC 多进程共享 |
| **成本追踪** | 实时成本监控，支持预算控制 |
| **高性能异步** | 精细并发控制、QPS 限流、流式处理 |
| **负载均衡** | 多 Endpoint 分发，自动故障转移 |

---

## 安装

```bash
pip install flexllm

# 完整功能
pip install flexllm[all]
```

---

## 快速开始

### 基本用法

```python
from flexllm import LLMClient

client = LLMClient(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key"
)

# 异步
response = await client.chat_completions([
    {"role": "user", "content": "你好！"}
])

# 同步
response = client.chat_completions_sync([
    {"role": "user", "content": "你好！"}
])
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

### 负载均衡

```python
from flexllm import LLMClientPool

pool = LLMClientPool(
    endpoints=[
        {"base_url": "http://gpu1:8000/v1", "model": "qwen"},
        {"base_url": "http://gpu2:8000/v1", "model": "qwen"},
    ],
    load_balance="round_robin",  # 或 "weighted", "random", "fallback"
    fallback=True,               # 故障自动切换
)

# 请求自动分发
results = await pool.chat_completions_batch(messages_list, distribute=True)
```

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
flexllm list        # 已配置模型
flexllm models      # 远程可用模型
flexllm test        # 测试连接
```

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
