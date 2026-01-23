# 高级用法

## 多模态处理

### MllmClient

处理图文混合内容：

```python
from flexllm import MllmClient

client = MllmClient(
    base_url="https://api.openai.com/v1",
    api_key="your-key",
    model="gpt-4o",
)

# 构建多模态消息
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "描述这张图片"},
            {"type": "image_url", "image_url": {"url": "/path/to/image.jpg"}}
        ]
    }
]

# 单条调用
result = await client.call_llm([messages])

# 批量调用
results = await client.call_llm(messages_list)
```

**支持的图像源：**
- 本地文件路径（自动转 base64）
- HTTP/HTTPS URL（自动下载转 base64）
- base64 编码字符串
- PIL Image 对象

### 图像处理器

```python
from flexllm.msg_processors import (
    encode_image_to_base64,
    ImageCacheConfig,
    unified_batch_process_messages,
)

# 单张图片编码
base64_data = await encode_image_to_base64("/path/to/image.jpg")

# 批量消息预处理（高性能）
processed = await unified_batch_process_messages(
    messages_list,
    show_progress=True,
)
```

---

## 表格和文件夹处理

### MllmTableProcessor

处理 CSV/Excel 表格数据：

```python
from flexllm import MllmClient, MllmTableProcessor

client = MllmClient(...)
processor = MllmTableProcessor(client)

# 加载数据
df = processor.load_dataframe("data.xlsx", sheet_name=0)

# 批量处理
results = await processor.process_dataframe(
    df,
    prompt_template="分析这条数据: {row}",
    show_progress=True,
)
```

### MllmFolderProcessor

批量处理文件夹中的图像：

```python
from flexllm import MllmClient, MllmFolderProcessor

client = MllmClient(...)
processor = MllmFolderProcessor(client)

# 扫描图像
images = processor.scan_folder_images(
    "/path/to/images",
    recursive=True,
    extensions={'.jpg', '.png'},
)

# 批量处理
results = await processor.process_folder(
    "/path/to/images",
    prompt="描述这张图片",
)
```

---

## 链式推理

### ChainOfThoughtClient

多步骤推理任务：

```python
from flexllm.chain_of_thought_client import ChainOfThoughtClient, Step

client = ChainOfThoughtClient(llm_client=base_client)

# 定义推理步骤
steps = [
    Step(
        name="分析问题",
        prepare_messages=lambda ctx: [
            {"role": "user", "content": f"分析问题: {ctx['query']}"}
        ],
        decide_next=lambda response, ctx: "综合" if "需要" in response else None,
    ),
    Step(
        name="综合",
        prepare_messages=lambda ctx: [
            {"role": "user", "content": f"基于分析给出答案: {ctx['analysis']}"}
        ],
        is_final=True,
    ),
]

# 执行推理链
result = await client.execute_chain(
    query="复杂问题",
    steps=steps,
)
```

---

## 负载均衡策略

### 多 Endpoint 配置

```python
from flexllm import LLMClientPool

pool = LLMClientPool(
    endpoints=[
        {
            "base_url": "http://fast-host:8000/v1",
            "api_key": "key1",
            "model": "qwen",
            "weight": 2,              # 权重（用于 weighted 策略）
            "concurrency_limit": 50,  # endpoint 级别并发（可选）
            "max_qps": 500,           # endpoint 级别 QPS（可选）
        },
        {
            "base_url": "http://slow-host:8000/v1",
            "api_key": "key2",
            "model": "qwen",
            "weight": 1,
            "concurrency_limit": 5,   # 较慢服务使用更低的并发
            "max_qps": 50,
        },
    ],
    load_balance="weighted",
    fallback=True,
    failure_threshold=3,   # 连续失败 3 次标记为不健康
    recovery_time=60.0,    # 60 秒后尝试恢复
    concurrency_limit=10,  # 全局默认值（未指定 endpoint 级别配置时使用）
    max_qps=100,           # 全局默认值
)
```

### 策略说明

| 策略 | 说明 |
|------|------|
| `round_robin` | 轮询，依次使用每个 endpoint |
| `weighted` | 加权随机，按 weight 比例分配 |
| `random` | 完全随机 |
| `fallback` | 主备模式，优先使用第一个，失败后切换 |

### Endpoint 级别 Rate Limit

每个 endpoint 可以独立配置 `concurrency_limit` 和 `max_qps`，以适应异构 endpoint 场景（不同服务性能差异大）：

```python
from flexllm import LLMClientPool, EndpointConfig

# 方式1：使用 EndpointConfig（推荐）
pool = LLMClientPool(
    endpoints=[
        EndpointConfig(
            base_url="http://fast-api.com/v1",
            api_key="key1",
            model="qwen",
            concurrency_limit=50,  # 高性能服务
            max_qps=500,
        ),
        EndpointConfig(
            base_url="http://slow-api.com/v1",
            api_key="key2",
            model="qwen",
            concurrency_limit=5,   # 低性能服务
            max_qps=50,
        ),
    ],
)

# 方式2：使用 dict 配置
pool = LLMClientPool(
    endpoints=[
        {"base_url": "http://fast.com/v1", "concurrency_limit": 50, "max_qps": 500},
        {"base_url": "http://slow.com/v1", "concurrency_limit": 5, "max_qps": 50},
    ],
    concurrency_limit=10,  # 全局默认值
    max_qps=100,           # 全局默认值
)
```

**配置优先级**：endpoint 级别配置 > 全局配置 > 默认值

**CLI 配置方式**（`~/.flexllm/config.yaml`）：

```yaml
batch:
  concurrency: 10       # 全局默认并发
  max_qps: 100          # 全局默认 QPS
  endpoints:
    - base_url: http://fast-api.com/v1
      api_key: key1
      model: qwen
      concurrency_limit: 50
      max_qps: 500
    - base_url: http://slow-api.com/v1
      api_key: key2
      model: qwen
      concurrency_limit: 5
      max_qps: 50
  fallback: true
```

**CLI 优先级**：`-m 参数` > `batch.endpoints` > 默认模型

- 指定 `-m model`：使用指定的模型配置
- 未指定 `-m` 且配置了 `batch.endpoints`：自动使用 `LLMClientPool`
- 都没有：使用默认模型

**使用场景**：
- 混合部署：本地 GPU 服务（高并发）+ 云 API（受限）
- 成本优化：付费 API（低并发）+ 免费 API（高并发）
- 性能适配：快速服务处理更多请求，慢速服务不被压垮

### Fallback 重试机制

当启用 `fallback=True` 时，重试次数会在多个 endpoint 间分配，避免单个 endpoint 超时导致的长时间等待：

```python
pool = LLMClientPool(
    endpoints=[...],  # 假设 3 个 endpoint
    fallback=True,
    retry_times=6,    # 总重试次数
)
# 每个 endpoint 实际重试 6 // 3 = 2 次
# 单个请求最多尝试 3 个 endpoint × 2 次 = 6 次

# 不指定 retry_times 时，fallback 模式默认为 0（快速切换）
pool = LLMClientPool(endpoints=[...], fallback=True)
# 每个 endpoint 尝试 1 次即切换到下一个
```

### 分布式批量请求

```python
# 将请求分散到多个 endpoint 并行处理
results = await pool.chat_completions_batch(
    messages_list,
    distribute=True,  # 启用分布式
)
```

---

## 性能优化

### 并发控制

```python
client = LLMClient(
    concurrency_limit=100,  # 最大并发请求数
    max_qps=50,             # 每秒最大请求数
    timeout=120,            # 单请求超时
)
```

### 缓存优化

```python
from flexllm import ResponseCacheConfig

# IPC 模式（多进程共享，推荐）
cache = ResponseCacheConfig(
    enabled=True,
    ttl=3600,
    use_ipc=True,
)

# 本地模式（单进程，更快）
cache = ResponseCacheConfig(
    enabled=True,
    ttl=3600,
    use_ipc=False,
)
```

### 批量处理最佳实践

```python
# 1. 使用输出文件（断点续传）
results = await client.chat_completions_batch(
    messages_list,
    output_file="results.jsonl",
)

# 2. 使用 metadata_list 保存额外信息
# 适合需要追踪数据来源的场景
metadata_list = [
    {"id": "001", "source": "data.jsonl", "line": 1},
    {"id": "002", "source": "data.jsonl", "line": 2},
]
results = await client.chat_completions_batch(
    messages_list,
    metadata_list=metadata_list,  # 元数据会保存到输出文件
    output_file="results.jsonl",
)
# 输出文件格式：{"index": 0, "output": "...", "status": "success", "input": [...], "metadata": {"id": "001", ...}}

# 3. 配合缓存使用
client = LLMClient(
    cache=ResponseCacheConfig(enabled=True),
)

# 4. 迭代式处理（内存友好）
async for batch_result in client.iter_chat_completions_batch(
    messages_list,
    batch_size=100,
):
    process(batch_result)
```

---

## Thinking 模式

### OpenAI 兼容（DeepSeek 等）

```python
from flexllm import OpenAIClient

client = OpenAIClient(
    base_url="https://api.deepseek.com/v1",
    api_key="your-key",
    model="deepseek-reasoner",
)

# 启用思考
result = await client.chat_completions(
    messages,
    thinking=True,
    return_raw=True,
)

# 解析思考内容
parsed = OpenAIClient.parse_thoughts(result.data)
print("思考过程:", parsed["thought"])
print("最终答案:", parsed["answer"])
```

### Gemini

```python
from flexllm import GeminiClient

client = GeminiClient(
    api_key="your-key",
    model="gemini-2.5-flash",
)

# 思考级别控制
result = await client.chat_completions(
    messages,
    thinking="high",  # "minimal", "low", "medium", "high"
)
```

---

## 错误处理

### 自动重试

```python
client = LLMClient(
    retry_times=3,      # 重试次数
    retry_delay=1.0,    # 重试间隔
)
```

### 进度条状态显示

批量处理时，进度条会实时显示重试和失败信息：

```
[▉▉▉▉▉▉▉▉▉▉          ] 50.0% (500/1000) ⚡ 25.3 req/s avg: 0.04s 💰 $0.0012 ↻12 ✗2
```

| 标记 | 说明 |
|------|------|
| `↻N` | 总重试次数（包括内部重试和 fallback 重试） |
| `✗N` | 最终失败的请求数 |

**错误警告**：首次遇到新错误类型时，会打印一次警告：
```
⚠️  新错误类型: timeout: Request timed out after 120s
```
相同错误类型后续出现不会重复打印。

### 批量处理错误

```python
results, summary = await client.chat_completions_batch(
    messages_list,
    return_summary=True,
)

print(f"成功: {summary['success']}")
print(f"失败: {summary['failed']}")
print(f"缓存命中: {summary['cached']}")
```

### 手动错误处理

```python
from flexllm import BatchResultItem

results = await client.chat_completions_batch(
    messages_list,
    return_raw=True,
)

for item in results:
    if item.status == "success":
        print(item.content)
    elif item.status == "error":
        print(f"错误: {item.error}")
    elif item.status == "cached":
        print(f"缓存: {item.content}")
```

---

## 上下文管理

```python
# 推荐：使用 async with 自动清理资源
async with LLMClient(...) as client:
    result = await client.chat_completions(messages)

# 手动清理
client = LLMClient(...)
try:
    result = await client.chat_completions(messages)
finally:
    await client.close()
```

---

## 成本追踪

### 基本用法

批量处理时追踪成本：

```python
from flexllm import LLMClient

client = LLMClient(...)

# 方式1：获取成本报告
results, cost_report = await client.chat_completions_batch(
    messages_list,
    return_cost_report=True,
)
print(f"总成本: ${cost_report.total_cost:.4f}")
print(f"总 tokens: {cost_report.total_tokens:,}")
print(f"平均成本/请求: ${cost_report.avg_cost_per_request:.6f}")

# 方式2：进度条实时显示成本
results = await client.chat_completions_batch(
    messages_list,
    track_cost=True,  # 进度条显示 💰 $0.0012
)
```

### CostReport 属性

| 属性 | 说明 |
|------|------|
| `total_cost` | 总成本（美元） |
| `total_input_tokens` | 总输入 tokens |
| `total_output_tokens` | 总输出 tokens |
| `total_tokens` | 总 tokens |
| `request_count` | 请求数 |
| `avg_cost_per_request` | 平均成本/请求 |
| `avg_input_tokens` | 平均输入 tokens |
| `avg_output_tokens` | 平均输出 tokens |

### 预算控制

使用 `CostTrackerConfig` 设置预算限制：

```python
from flexllm import LLMClient, CostTrackerConfig

# 带预算控制的客户端
client = LLMClient(
    ...,
    cost_tracker=CostTrackerConfig.with_budget(
        limit=5.0,        # 硬限制：超过 $5 自动停止
        warning=4.0,      # 软限制：超过 $4 触发警告
        on_warning=lambda current, total: print(f"⚠️ 预算警告: ${current:.2f}/{total:.2f}")
    )
)

try:
    results = await client.chat_completions_batch(messages_list)
except BudgetExceededError as e:
    print(f"预算超限: {e}")
```

### 配置方式

```python
from flexllm import CostTrackerConfig

# 方式1：仅追踪（不限制预算）
config = CostTrackerConfig.tracking_only()

# 方式2：带预算控制
config = CostTrackerConfig.with_budget(
    limit=10.0,
    warning=8.0,
    on_warning=my_warning_handler,
)

# 方式3：禁用
config = CostTrackerConfig.disabled()

# 应用到客户端
client = LLMClient(..., cost_tracker=config)
```

### CLI 用法

```bash
# 进度条默认显示实时成本（track_cost=True）
flexllm batch input.jsonl -o output.jsonl

# 输出示例：
# [▉▉▉▉▉▉▉▉▉▉          ] 50.0% (50/100) ⚡ 2.5 req/s avg: 0.8s 💰 $0.0012
```

### 成本估算

成本基于 `flexllm/pricing.py` 中的模型定价表估算。支持的模型包括：

- OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo 等
- Anthropic: claude-3.5-sonnet, claude-3-opus 等
- Google: gemini-2.0-flash, gemini-1.5-pro 等
- DeepSeek: deepseek-chat, deepseek-reasoner 等
- 其他: qwen, yi, llama 等主流模型

未在定价表中的模型会使用默认估算价格。
