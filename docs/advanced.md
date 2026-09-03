# 高级用法

## 多模态处理

### 消息预处理

`messages_preprocess` 自动将消息中的本地文件路径和 URL 转换为 base64，支持图片、视频和音频：

```python
from flexllm.msg_processors import messages_preprocess

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "/path/to/image.png"}},
            {"type": "video_url", "video_url": {"url": "/path/to/video.mp4"}},
            {"type": "input_audio", "input_audio": {"data": "/path/to/audio.wav", "format": "wav"}},
            {"type": "text", "text": "描述你看到和听到的内容"},
        ],
    }
]

# 预处理：本地路径/URL → base64
processed = await messages_preprocess(messages)
result = await client.chat_completions(processed)
```

**支持的内容类型和处理方式：**

| `type` 字段 | 数据字段 | 处理方式 | 输出格式 |
|---|---|---|---|
| `image_url` | `image_url.url` | PIL/OpenCV 管道（支持缩放） | `data:image/...;base64,...` |
| `video_url` | `video_url.url` | 原始字节 base64 | `data:video/...;base64,...` |
| `audio_url` | `audio_url.url` | 原始字节 base64 | `data:audio/...;base64,...` |
| `input_audio` | `input_audio.data` | 原始字节 base64 | 纯 base64（无 data: 前缀，OpenAI 格式） |

**支持的来源：** 本地路径、`file://` URI、HTTP/HTTPS URL、`data:` URI（直接透传）。

**跨 Provider 格式转换：** Claude 和 Gemini 客户端会自动将 OpenAI 格式转换为各自原生格式：
- Claude: `video_url`/`audio_url` → `document` 类型，`input_audio` → `document` 类型
- Gemini: 统一转换为 `inline_data` 格式

### 通用媒体编码

对于单个文件的 base64 编码，可直接使用 `encode_media_to_base64`：

```python
from flexllm.msg_processors import encode_media_to_base64

# 本地文件 → data URI
data_uri = await encode_media_to_base64("/path/to/video.mp4")
# "data:video/mp4;base64,..."

# 不带 MIME 前缀（适用于 input_audio.data）
raw_b64 = await encode_media_to_base64("/path/to/audio.wav", return_with_mime=False)
# "UklGRiT6AABX..."
```

### MllmClient

处理图文混合内容的高级客户端：

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

# 单条调用（call_llm 返回列表）
results = await client.call_llm([messages])
result = results[0]

# 批量调用
messages_list = [[msg1], [msg2], ...]  # 每个元素是一组消息
results = await client.call_llm(messages_list)
```

### 图像处理器

```python
from flexllm.msg_processors import (
    encode_image_to_base64,
    unified_batch_process_messages,
)

# 单张图片编码（支持本地路径/URL，支持缩放）
base64_data = await encode_image_to_base64("/path/to/image.jpg", max_width=1024)

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

client = MllmClient(base_url="...", api_key="...", model="gpt-4o")
processor = MllmTableProcessor(client)

# 加载数据
df = processor.load_dataframe("data.xlsx", sheet_name=0, max_num=100)

# 方式1：直接处理表格文件（推荐）
results = await processor.call_table(
    table_path="data.xlsx",
    text_col="question",      # 文本列名
    image_col="image_path",   # 图像列名（可选，None 表示纯文本）
)

# 方式2：处理 DataFrame
results = await processor.call_dataframe(
    df,
    text_col="question",
    image_col=None,  # 纯文本模式
)

# 方式3：批量处理表格中的图像
results = await processor.call_table_images(
    table_path="images.xlsx",
    image_col="image_path",
    text_prompt="描述这张图片",
)
```

### MllmFolderProcessor

批量处理文件夹中的图像：

```python
from flexllm import MllmClient, MllmFolderProcessor

client = MllmClient(base_url="...", api_key="...", model="gpt-4o")
processor = MllmFolderProcessor(client)

# 扫描图像
images = processor.scan_folder_images(
    "/path/to/images",
    recursive=True,
    max_num=100,
    extensions={'.jpg', '.png'},
)

# 批量处理文件夹中的图像
results = await processor.call_folder_images(
    "/path/to/images",
    text_prompt="描述这张图片",
    system_prompt="你是一个图像分析助手",
    recursive=True,
)

# 或处理指定的图像文件列表
results = await processor.call_image_files(
    image_files=["/path/to/img1.jpg", "/path/to/img2.png"],
    text_prompt="这张图片中有什么？",
)
```

---

## 链式推理

### ChainOfThoughtClient

多步骤推理任务：

```python
from flexllm import OpenAIClient
from flexllm.clients.chain_of_thought import ChainOfThoughtClient, Step

# 创建底层客户端
base_client = OpenAIClient(base_url="...", api_key="...", model="gpt-4")

# 创建链式推理客户端
client = ChainOfThoughtClient(openai_client=base_client)

# 定义推理步骤
steps = [
    Step(
        name="分析问题",
        prepare_messages_fn=lambda ctx: [
            {"role": "user", "content": f"分析问题: {ctx.query}"}
        ],
        get_next_step_fn=lambda response, ctx: "综合" if "需要" in response else None,
    ),
    Step(
        name="综合",
        prepare_messages_fn=lambda ctx: [
            {"role": "user", "content": f"基于分析给出答案: {ctx.get('analysis')}"}
        ],
        get_next_step_fn=lambda response, ctx: None,  # 返回 None 表示结束
    ),
]

# 注册步骤
client.add_steps(steps)

# 执行推理链
context = await client.execute_chain(
    initial_step_name="分析问题",
    initial_context={"query": "复杂问题"},
)
print(context.final_response)
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
            "concurrency_limit": 50,  # endpoint 级别并发（可选）
            "max_qps": 500,           # endpoint 级别 QPS（可选）
        },
        {
            "base_url": "http://slow-host:8000/v1",
            "api_key": "key2",
            "model": "qwen",
            "concurrency_limit": 5,   # 较慢服务使用更低的并发
            "max_qps": 50,
        },
    ],
    fallback=True,
    failure_threshold=3,   # 连续失败 3 次标记为不健康
    recovery_time=60.0,    # 60 秒后尝试恢复
    concurrency_limit=10,  # 全局默认值（未指定 endpoint 级别配置时使用）
    max_qps=100,           # 全局默认值
)
```

选路策略：

- **单条调用**（`chat_completions` / `chat_completions_stream`）：容量感知选路——在健康且未饱和的 endpoint 中选负载率（in-flight / concurrency_limit）最低者，全部饱和时退回轮询。异构 endpoint 下慢节点饱和后，流量自动流向快节点。
- **批量调用**（`distribute=True`）：worker 模型——每个 endpoint 的 worker 数等于其并发上限，所有 worker 从共享队列抢任务，快 endpoint 周转快自然多拿任务。

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

### Pool 级全局硬上限（total_concurrency_limit / total_max_qps）

per-endpoint 限制是**叠加**的：3 个 endpoint 各 `concurrency_limit=30`，峰值并发可达 90。
当 API 配额按总 QPS 计费、或下游系统有总并发上限时，用 pool 级参数设全局硬上限：

```python
pool = LLMClientPool(
    endpoints=[...],              # 3 个 endpoint
    concurrency_limit=30,         # 每 endpoint 软上限（单点保护）
    total_concurrency_limit=50,   # 跨所有 endpoint 的并发硬上限
    total_max_qps=100,            # 跨所有 endpoint 的 QPS 硬上限
)
```

两层同时生效，哪个先触发就卡在哪（类比 K8s 的 LimitRange + ResourceQuota）。语义细节：

- **不传时行为完全不变**；单 endpoint 模式下同样生效（total 小于 endpoint 限制时以 total 为准）
- 全局闸门在底层请求执行点生效，获取顺序为 endpoint 并发 → 全局并发 → endpoint QPS → 全局 QPS。
  等全局 slot 的请求不占任何全局稀缺资源，不存在"慢 endpoint 拖住快 endpoint"的队头阻塞
- QPS 令牌按 wire 请求计：fallback 换 endpoint 重试会再取一次令牌
- **流式接口不受约束**（`chat_completions_stream` 不经过并发引擎，per-endpoint 限制同样不约束它）
- 排队等待计入 `queue_time`（`return_usage=True` 时可从 `ChatCompletionResult.queue_time` 读取）

**CLI 配置方式**（`~/.flexllm/config.yaml`）：

```yaml
batch:
  concurrency: 10          # 每 endpoint 默认并发（未单独配置时的回退值）
  max_qps: 100             # 每 endpoint 默认 QPS
  total_concurrency: 50    # 跨所有 endpoint 的并发硬上限（可选）
  total_max_qps: 100       # 跨所有 endpoint 的 QPS 硬上限（可选）
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

cache = ResponseCacheConfig(
    enabled=True,
    ttl=3600,
)

# 或使用快捷方法
cache = ResponseCacheConfig.with_ttl(3600)
cache = ResponseCacheConfig.persistent()  # 永不过期
```

### 批量处理最佳实践

```python
# 1. 使用输出文件（断点续传）
results = await client.chat_completions_batch(
    messages_list,
    output_jsonl="results.jsonl",
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
    output_jsonl="results.jsonl",
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

#### 断点续传的三条语义

1. **返回列表始终是全量**：续跑时，`output_jsonl` 中已完成的样本不会重新请求，但会回填到
   `chat_completions_batch` 的返回列表，直接用返回值即可，不必读回文件。
   （例外：`iter_chat_completions_batch` 是流式接口，恢复项不 yield，续跑要全量结果就读文件。）

2. **按 index 对齐，顺序不能变**：checkpoint 用列表位置对齐。重跑时逐条校验文件里的
   `input` 与当前 `messages_list`，顺序变了会直接抛 `ValueError`，不会静默错位。
   样本集合本身会变动（增删、换序）时，用 `metadata_list` 带上业务主键，回读时按 key 匹配。
   `save_input=False` 时无 input 可校验，此时顺序完全由调用方保证。

3. **只跳过 `status == "success"` 的记录**：失败条目（status 为 error）续跑时会自动重试。
   注意"成功"指调用成功——模型返回了内容但 JSON 格式跑歪，flexllm 视作已完成，不会重跑。
   需要按解析结果重跑时，自行校验后把坏样本挑出来，写到另一个 `output_jsonl` 里跑修复轮。
   另外 `retry_times` 只覆盖网络/429/5xx 等传输层错误，不会因内容不合预期而重试。

### Per-record 参数（参数扫描）

`chat_completions_batch` 接受 `params_list`（与 `messages_list` 等长，元素为 dict 或 None），
让每条记录单独覆盖全局生成参数，用于参数扫描（同一 prompt 跑不同 `temperature` / `stop` 对比）。

```python
results = await client.chat_completions_batch(
    messages_list=[msgs, msgs],          # 相同 messages
    params_list=[
        {"temperature": 0.2, "stop": ["\n\n"]},
        {"temperature": 0.9},
    ],
    output_jsonl="sweep.jsonl",
)
# 每条有效参数 = {**全局kwargs, **params_list[i]}；缓存键随各自参数区分，互不命中。
# 带 params 的行会把该行 params 原样回显到输出（output 多出 "params" 字段）。
```

CLI 中通过 JSONL 每行的 `params` 字段使用（嵌套，不平铺）：

```jsonl
{"messages": [{"role":"user","content":"解释量子纠缠"}], "params": {"temperature": 0.2, "stop": ["\n\n"]}}
{"q": "解释量子纠缠", "params": {"system": "你是物理老师", "user_template": "用初中生能懂的话：{content}", "temperature": 0.9}}
```

`params` 内有两类键，消费位置不同：

- **`system` / `user_template`**：消息构造类，作为该行的有效 system / 模板（行内 > 配置），
  不会作为 API 参数下发。
- **其余键**（`temperature` / `stop` / `max_tokens` / `response_format` …）：作为该行生成参数覆盖全局。

优先级（行内 > 配置）：

- system：`messages 内显式 system` > `params.system` > CLI `-s` > 配置 `models.system`
- user_template：`params.user_template` > CLI `--user-template` > 配置 `user_template`
- 生成参数：`params.<key>` > CLI 对应参数 > 配置 `models` 节

> 注意：断点续传校验只看 messages，不感知 `params` 变化（与"改 kwargs 不影响续传判定"一致）。
> 若改了某行的 `params` 想重跑，需先删除输出文件中对应记录。

---

## Thinking 模式

### OpenAI 兼容（DeepSeek、GLM 等）

```python
from flexllm import OpenAIClient

client = OpenAIClient(
    base_url="https://api.deepseek.com/v1",
    api_key="your-key",
    model="deepseek-reasoner",
)

# 透传 provider 原生 thinking；reasoning_effort 等其他参数也会原样传递
result = await client.chat_completions(
    messages,
    thinking={"type": "enabled"},
    reasoning_effort="low",
    return_raw=True,
)

# 解析思考内容
parsed = OpenAIClient.parse_thoughts(result.data)
print("思考过程:", parsed["thought"])
print("最终答案:", parsed["answer"])
```

### Claude

```python
from flexllm import ClaudeClient

client = ClaudeClient(
    api_key="your-key",
    model="claude-sonnet-4-6",
)

# Claude 4.6+：自动转换为 adaptive thinking + output_config.effort
result = await client.chat_completions(
    messages,
    reasoning_effort="low",
    return_raw=True,
)

# 解析思考内容
parsed = ClaudeClient.parse_thoughts(result.data)
print("思考过程:", parsed["thought"])
print("最终答案:", parsed["answer"])
```

`thinking="low"` 与 `reasoning_effort="low"` 等价。Claude 3.7/4.0-4.5 会把强度映射为
`budget_tokens`；4.6 默认 adaptive，但仍兼容整数预算；4.7+ 只接受 adaptive，整数预算会
明确报错。Claude 3.5 及更早版本不支持 extended thinking，省略参数即可正常调用。
Fable/Mythos 5 的 adaptive thinking 始终开启，不能用 `thinking=False`，应通过 effort
控制开销。

工具回合请把 `return_usage=True` 返回的 `result.assistant_message` 原样放入下一轮消息，
再追加工具结果。该字段保留 DeepSeek 的 `reasoning_content` 以及 Claude 带签名的
thinking blocks；用 `content + tool_calls` 自行重建会丢失必要的 provider 状态。

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
    retry_times=3,      # 最大尝试次数（含首次调用）
    retry_delay=1.0,    # 退避基数（秒）
)
```

**可重试的错误**：429 限流、5xx 服务端错误、连接中断、超时。其他 4xx
（400/401/404 等）重试也不会成功，直接返回错误结果并保留响应体。

**退避策略**：

| 情况 | 等待时长 |
|------|----------|
| 响应带 `Retry-After` 头 | 以服务端给出的值为准，向上抖动 0~25% |
| 无 `Retry-After` | 指数退避 `retry_delay * 2**n`，equal jitter |

两种情况都以 60 秒为硬上限（抖动后也不越过），避免服务端返回 `Retry-After: 600`
时把整个批量任务拖死。

`Retry-After` 同时支持秒数（`Retry-After: 20`）与 HTTP-date 两种形式，格式
无法解析时退回指数退避。服务端明确告知配额恢复时间时按它等待，比本地猜测准确——
否则固定短间隔的几次重试会在 1 秒内打光，必然全部再吃 429。

抖动是必要的：批量场景下成百上千个并发请求会同时失败并收到相同的
`Retry-After`，不抖动则会在同一时刻齐发，把服务端再打挂一次。`Retry-After`
只向上抖动，因为早于服务端要求重试必然再吃一次限流。

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

# 同步版本使用 with
with LLMClient(...) as client:
    result = client.chat_completions_sync(messages)

# 手动清理（异步）
client = LLMClient(...)
try:
    result = await client.chat_completions(messages)
finally:
    await client.aclose()

# 手动清理（同步）
client = LLMClient(...)
try:
    result = client.chat_completions_sync(messages)
finally:
    client.close()
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

## 正向代理

目标 `base_url` 仅经某网关可达时（例如网关常驻 VPN），可显式指定正向代理。

不设 `proxy` 时，仍会沿用 `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` 环境变量
（底层 `trust_env=True`）。`proxy` 参数解决的是环境变量表达不了的场景：
**按 client / 按 endpoint 分别决定走不走代理**。

```python
from flexllm import LLMClient

# 单 endpoint
client = LLMClient(
    base_url="https://api.example.com/v1",
    api_key="sk-xxx",
    proxy="http://gateway:8080",
)

# 带认证（凭据由 aiohttp 从 URL 中提取，转为 Proxy-Authorization 头）
client = LLMClient(base_url="...", proxy="http://user:pass@gateway:8080")

# 多 endpoint：部分直连、部分经网关
client = LLMClient(
    endpoints=[
        {"base_url": "http://local-vllm:8000/v1"},                           # 直连
        {"base_url": "https://vpn-only/v1", "proxy": "http://gateway:8080"}, # 经网关
    ],
)

# 顶层 proxy 作为各 endpoint 的默认值，endpoint 级可覆盖
client = LLMClient(
    endpoints=[
        {"base_url": "https://a/v1"},                                  # 用 gw:8080
        {"base_url": "https://b/v1", "proxy": "http://gw2:9090"},      # 用 gw2:9090
    ],
    proxy="http://gw:8080",
)
```

配置文件中可为每个模型单独指定（`flexllm ask/batch` 等 CLI 命令走此路径）：

```yaml
models:
  - name: vpn-only-model
    base_url: https://vpn-only/v1
    api_key: sk-xxx
    proxy: http://gateway:8080
```

### SOCKS 代理

SOCKS 需要额外依赖：

```bash
pip install 'flexllm[socks]'   # 装 aiohttp-socks
```

```python
client = LLMClient(base_url="...", proxy="socks5://gateway:1080")
client = LLMClient(base_url="...", proxy="socks5://user:pass@gateway:1080")

# ssh -D 起的动态转发同样可用
client = LLMClient(base_url="...", proxy="socks5://127.0.0.1:1080")
```

| Scheme | 说明 |
|--------|------|
| `http://` `https://` | aiohttp 原生，无需额外依赖 |
| `socks5://` `socks4://` | 需 `flexllm[socks]` |
| `socks5h://` | 等价于 `socks5://`，自动规范化 |

**域名由代理解析**：SOCKS5 默认 `rdns=True`，目标域名原样交给代理侧解析，
不在本地解析。VPN 网关场景下这是关键——目标域名往往只有网关那侧能解析，
这也是 `socks5h://` 的语义，所以两者等价。

**未安装依赖时在构造时报错**，而不是发请求时才炸：

```python
LLMClient(base_url="...", proxy="socks5://gw:1080")
# ValueError: SOCKS 代理 'socks5://gw:1080' 需要额外依赖：pip install 'flexllm[socks]'
```

其他 scheme（`socks6://`、`ftp://` 等）一律在构造时拒绝。aiohttp **不校验
scheme**，给它未知 scheme 它会照样往该端口发 HTTP `CONNECT`，表现为难以定位的
连接错误——与其运行时以令人困惑的方式失败，不如构造时就拒绝。

> 实现差异：HTTP 代理走 aiohttp 的 per-request `proxy=` 参数，SOCKS 必须在
> connector 层建隧道。因此 SOCKS 的粒度是 per-client / per-endpoint（每个
> client 独占一个 connector），无法 per-request——这正好覆盖了按 endpoint
> 区分走不走代理的需求。

两种代理对流式（`chat_completions_stream`）与非流式请求同样生效。流式路径不走
`ConcurrentRequester`（各客户端自建 session），但通过 `create_proxied_session()`
共用同一套代理语义。

**多模态图片/媒体下载也走代理**：`preprocess_msg=True` 时，消息里的图片/音频/
视频 URL 会被预处理下载并转 base64，这条下载链同样经 client 的 `proxy`。VPN
网关场景下目标模型与图床往往在同一内网，二者共用一个代理即可：

```python
client = LLMClient(base_url="https://vpn-only/v1", api_key="...", proxy="socks5://gateway:1080")
# 图片 URL 的下载也经 gateway:1080
await client.chat_completions(messages=[
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "http://intranet/pic.png"}},
    ]},
], preprocess_msg=True)
```

实现上，图片下载 session 由预处理入口 `unified_messages_preprocess` 用
`create_proxied_session()` 建立，per-request 代理参数挂在 session 上
（`session_proxy_kwargs()`）贯穿传给各下载点，中间层无需改动。边界：直接调用底层
预处理函数且不传 `proxy` 时，退回 `HTTP_PROXY` 等环境变量。

### 图片/媒体磁盘缓存

下载的图片/媒体 URL 可缓存到本地磁盘，同一 URL 跨调用/跨进程复用、避免重复下载：

```python
# 默认不缓存（每次重新下载）
client = LLMClient(base_url="...", api_key="...")

# 开启磁盘缓存，默认路径 ~/.flexllm/cache/image_cache
client = LLMClient(base_url="...", api_key="...", cache_image=True)

# 自定义缓存目录
client = LLMClient(base_url="...", api_key="...", cache_image=True, cache_dir="/data/img_cache")
```

`cache_image` 默认 `False`（不缓存）；`cache_dir` 仅在开启时生效，`None` 时用默认
路径。client 持有并复用一个按此配置构建的处理器实例，批量预处理时跨消息共享内存/
磁盘缓存。
