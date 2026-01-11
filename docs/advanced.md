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
from flexllm.processors import (
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
            "base_url": "http://host1:8000/v1",
            "api_key": "key1",
            "model": "qwen",
            "weight": 2,  # 权重（用于 weighted 策略）
        },
        {
            "base_url": "http://host2:8000/v1",
            "api_key": "key2",
            "model": "qwen",
            "weight": 1,
        },
    ],
    load_balance="weighted",
    fallback=True,
    failure_threshold=3,   # 连续失败 3 次标记为不健康
    recovery_time=60.0,    # 60 秒后尝试恢复
)
```

### 策略说明

| 策略 | 说明 |
|------|------|
| `round_robin` | 轮询，依次使用每个 endpoint |
| `weighted` | 加权随机，按 weight 比例分配 |
| `random` | 完全随机 |
| `fallback` | 主备模式，优先使用第一个，失败后切换 |

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
