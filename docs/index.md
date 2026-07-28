# flexllm 文档

高性能 LLM 客户端库，支持批量处理、响应缓存和断点续传。

## 文档目录

```
docs/
├── index.md              # 本文档（主入口）
├── api.md                # API 详细参考
├── advanced.md           # 高级用法（多模态/成本/正向代理等）
├── audio.md              # 语音能力（转录/合成/音频输入）
├── roadmap.md            # 开发路线图
└── plans/                # 实现计划
```

> Agent 模块已从 flexllm 移除，独立为 openagent 包。

> 安装、CLI 用法、配置文件等基础内容见项目根目录 [README.md](../README.md)。

## 核心概念

### 1. 客户端层次

```
LLMClient (统一入口，LLMClientPool 的别名)
    ├── 单 endpoint 模式：自动创建底层客户端
    │   ├── OpenAIClient (OpenAI 兼容 API)
    │   ├── GeminiClient (Google Gemini)
    │   └── ClaudeClient (Anthropic Claude)
    │
    └── 多 endpoint 模式：负载均衡
        └── ProviderRouter (容量感知，全饱和时退回轮询)
```

### 2. 请求模式

| 模式 | 方法 | 说明 |
|------|------|------|
| 单条同步 | `chat_completions_sync()` | 简单场景 |
| 单条异步 | `chat_completions()` | 高性能场景 |
| 批量异步 | `chat_completions_batch()` | 大规模处理 |
| 流式输出 | `chat_completions_stream()` | 实时显示 |

### 3. 缓存机制

```python
from flexllm import ResponseCacheConfig

# 启用缓存（1小时 TTL）
cache = ResponseCacheConfig(enabled=True, ttl=3600)

# 永久缓存
cache = ResponseCacheConfig(enabled=True, ttl=0)
```

缓存基于消息内容的 hash，相同请求自动命中缓存。

### 4. 成本追踪

```python
# 简单启用成本追踪
results, cost_report = await client.chat_completions_batch(
    messages_list,
    return_cost_report=True,
)
print(f"总成本: ${cost_report.total_cost:.4f}")

# 进度条实时显示成本
results = await client.chat_completions_batch(
    messages_list,
    track_cost=True,  # 进度条中显示 💰 $0.0012
)
```

详见 [高级用法 - 成本追踪](advanced.md#成本追踪)。

### 5. 断点续传

```python
results = await client.chat_completions_batch(
    messages_list,
    output_jsonl="results.jsonl",  # 关键：指定输出文件
)
```

- 结果增量写入文件
- 程序中断后，重新运行自动跳过已完成的请求
- 配合缓存使用效果更好

## 下一步

- [API 详细参考](api.md) - 完整的 API 文档
- [高级用法](advanced.md) - 负载均衡、多模态、链式推理等
- [语音能力](audio.md) - 语音转录、语音合成、对话音频输入
