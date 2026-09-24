# API 参考

## 客户端类

### LLMClient

统一的 LLM 客户端，自动选择底层实现。

```python
from flexllm import LLMClient

client = LLMClient(
    model: str,                          # 模型名称
    base_url: str = None,                # API 地址
    api_key: str = "EMPTY",              # API 密钥
    provider: str = "auto",              # "auto", "openai", "gemini"
    cache: ResponseCacheConfig = None,   # 缓存配置
    concurrency_limit: int = 10,         # 最大并发数
    max_qps: float = None,               # QPS 限制
    retry_times: int = 3,                # 最大尝试次数（含首次调用）
    retry_delay: float = 1.0,            # 退避基数（秒），指数退避+抖动
    timeout: int = 120,                  # 请求超时（秒）；非流式=单请求总时长，流式=相邻 chunk 最长间隔
)
```

**方法：**

#### complete / complete_batch（推荐接口）

```python
result = await client.complete(messages, model=None, **gen_kwargs)
results = await client.complete_batch(messages_list, model=None, **kwargs)
```

**单条 `complete()`** 总是返回 `ChatCompletionResult`，失败抛结构化异常：

| 字段 | 说明 |
| --- | --- |
| `content` | 文本内容 |
| `usage` | `{"prompt_tokens", "completion_tokens", "total_tokens"}` |
| `reasoning_content` | 思考内容（DeepSeek-R1 / Qwen3 / Claude / Gemini） |
| `tool_calls` | `list[ToolCall]` |
| `finish_reason` | `"stop"` / `"length"` / `"tool_calls"` …（各 provider 统一到这套值） |
| `assistant_message` | 下一轮需原样回传的 assistant 消息（带签名的 thinking / tool_use 时非 None） |
| `extra` | 响应里标准信封之外的顶层字段（网关挂的带外信息） |
| `raw_response` | provider 原始响应 |
| `cached` / `queue_time` | 是否缓存命中 / 客户端排队耗时 |
| `error` / `ok` | 仅批量失败项使用，见下 |

失败抛 `LLMRequestError` 及其子类（`LLMHTTPError` / `LLMConnectionError` /
`LLMTimeoutError` / `LLMResponseError`），带 `status_code`、`response_data`、
`request_id`、`retryable`。

**批量 `complete_batch()` 不抛异常**：批量里单条失败是预期结果的一种，不是控制流事件。
返回等长同构的 `BatchResult`，失败项是 `content=None` 且带 `.error` 的同一种结果对象：

```python
results = await client.complete_batch(messages_list, output_jsonl="out.jsonl")

for r in results:            # BatchResult 可迭代、可索引、可 len()
    if r.ok:
        use(r.content)
    elif r.error.retryable:
        retry_later(r)

results.errors            # {index: LLMRequestError}，按原始 index 对齐
results.ok                # 成功项
results.failed            # 失败项
results.success_count / results.failed_count / results.cached_count
results.cost              # CostReport 或 None
results.elapsed           # 秒
results.raise_for_errors()  # 需要 fail-fast 时显式调用
```

只有整批开不了工（参数非法等）才抛异常。`complete_sync` / `complete_batch_sync`
是同步版本。这两个入口不接受 `return_raw` / `return_usage` / `return_summary` /
`return_cost_report` / `raise_on_error`——返回形状是固定的，传了会直接报错。

批量常用参数：`output_jsonl`（断点续传）、`show_progress`、`track_cost`、
`metadata_list`、`params_list`、`save_input`、`flush_interval`、`save_raw`。
其中 `save_raw` 默认 `False`：checkpoint 的 `result` 字段不保存 `raw_response`
（`content`/`usage` 已在记录顶层），百万级批量时这份冗余是实打实的磁盘开销；
需要事后按原始响应分析时再打开。

#### chat_completions（已弃用，0.18.0 移除）

```python
async def chat_completions(
    messages: List[dict],
    model: str = None,
    return_raw: bool = False,
    return_usage: bool = False,
    raise_on_error: bool = False,
    skip_cache: bool = False,
    **kwargs
) -> str | ChatCompletionResult | RequestResult
```

单条异步请求。返回形状由 `return_raw` / `return_usage` 决定；默认失败返回
`RequestResult` 而不抛异常，`raise_on_error=True` 才抛结构化异常。
使用旧返回形状时发出 `LegacyResponseWarning`。新代码用 `complete()`。

`chat_completions_sync`、`chat_completions_or_raise` 同样已弃用。

#### chat_completions_batch（已弃用，0.18.0 移除）

```python
async def chat_completions_batch(
    messages_list: List[List[dict]],
    output_jsonl: str = None,
    show_progress: bool = True,
    return_summary: bool = False,
    return_cost_report: bool = False,
    **kwargs
) -> List[str] | Tuple[List[str], dict]
```

批量异步请求，支持断点续传。失败项为 `None`——想知道每条为何失败，用 `complete_batch()`。

#### 旧接口的兼容承诺（到 0.18.0）

`chat_completions*` 的返回形状、参数顺序、失败时的返回值与 0.16.x 逐字一致，只在末尾
追加了默认 `False` 的 `raise_on_error`。用 v0.16.6 的代码跑同一组调用逐行比对过，
包括这些容易被忽略的角落：

- 单条失败返回 `RequestResult` 而不抛异常；批量失败项为 `None`
- `return_summary` 在单 endpoint 返回统计字符串、在 pool 返回 dict（两者本就不同）
- `return_cost_report` 只在启用了 `cost_tracker` 时才多返回一项；**多 endpoint
  分布式批量下不返回**（0.16.x 的分布式路径直接丢弃该参数，这个形状一并保留，
  但会打一条 warning 指向 `complete_batch().cost`）
- checkpoint JSONL 的字段集合不变；旧文件能被新版续跑，新文件也能被旧版读
  （`resume_from_jsonl` 只挑它认识的键）
- `transcribe` / `speech` 及其 batch 版本的失败项仍是 `RequestResult`

唯一的行为差异：`chat_completions_or_raise` 失败时抛的是 `LLMHTTPError` 等子类而非
`LLMRequestError` 基类，`except LLMRequestError` 照常捕获。

#### complete_stream（推荐流式接口）

```python
async for event in client.complete_stream(messages, model=None, **kwargs):
    ...
```

事件恒为 dict，不受任何开关影响（思考内容只走 `thinking` 事件，不会以 `<think>` 混进正文）：

| type | 字段 | 说明 |
|---|---|---|
| `thinking` | `content` | 思考片段 |
| `content` | `content` | 正文片段 |
| `tool_call_delta` | `tool_calls` | 工具调用增量（OpenAI 形态，按 `index` 合并） |
| `extra` | `extra` | 网关带外字段 |
| `result` | `result` | **最后一条，成功时恰好一次**：`ChatCompletionResult`，与 `complete()` 同构 |

`result` 里 `content` / `reasoning_content` / `tool_calls` 已累加好，`finish_reason` / `usage` /
`assistant_message`（下一轮需原样回传的续接状态，如 Claude 带签名的 thinking block）也都在上面，
调用方不需要自己拼。失败抛 typed error，与 `complete()` 相同。

与 `complete()` 的差异（均为有意）：
- `content` 只含正文。OpenAI 兼容端点在 `thinking=True` 时，`complete()` 的 `content`
  带 `<think>…</think>` 前缀，这里思考只在 `reasoning_content`；没有正文时为 `None`。
- Gemini 的 tool call id 是本地合成的（Gemini 不返回 id）：流式按 functionCall 顺序编号
  `call_0, call_1…`，非流式按 part 下标编号，二者不保证相同，只保证单次响应内唯一。

```python
async for event in client.complete_stream(messages, tools=tools):
    if event["type"] == "content":
        print(event["content"], end="", flush=True)
    elif event["type"] == "result":
        result = event["result"]   # result.tool_calls / result.assistant_message ...
```

多 endpoint（pool）时，故障转移**只发生在首个事件之前**：流已开始输出后失败直接抛异常——
已送出的内容收不回来，换 endpoint 从头再流只会产生重复输出。

#### chat_completions_stream

```python
async def chat_completions_stream(
    messages: List[dict],
    return_usage: bool = False,
    timeout: int = None,
    **kwargs
) -> AsyncIterator[str | dict]
```

流式响应。默认逐段 yield `str`；`return_usage=True` 时 yield 事件 dict，按序为：

| type | 字段 | 说明 |
|---|---|---|
| `thinking` | `content` | 思考片段（reasoning 模型） |
| `content` | `content` | 正文片段 |
| `tool_call_delta` | `tool_calls` | 工具调用增量 |
| `assistant_message` | `message` | 工具回合结束时可原样放入下一轮的 provider 续接消息；仅在存在思考或工具状态时发送 |
| `finish` | `reason` | 模型停止原因，OpenAI 语义：`stop` / `length`（被 max_tokens 截断）/ `tool_calls` / `content_filter` / …；provider 不给时为 `None`。**流末尾必发一次** |
| `usage` | `usage` | token 用量，最后一条（provider 给了才有） |

新代码用 `complete_stream()`：它事件形状固定，并在结尾给出汇总好的结果。

流式的 `timeout` 是**空闲超时**（两个 chunk 之间的最长间隔），不限制整条流的总时长——
长思考模型一轮可能持续数分钟，只要还在吐 token 就不算卡死。

---

### OpenAIClient

OpenAI 兼容 API 客户端。

```python
from flexllm import OpenAIClient

client = OpenAIClient(
    base_url: str,
    api_key: str = "EMPTY",
    model: str = None,
    # ... 其他参数同 LLMClient
)
```

**额外参数：**
- `thinking`: 思考模式控制
  - `False`: 禁用思考
  - `True`: 启用思考
  - `dict`: 透传 provider 原生 `thinking` 配置
  - `None`: 使用模型默认行为

**静态方法：**

```python
@staticmethod
def parse_thoughts(response_data: dict) -> dict
```

解析思考内容，返回 `{"thought": str, "answer": str}`。

---

### ClaudeClient

Anthropic Claude 客户端。

```python
from flexllm import ClaudeClient

client = ClaudeClient(
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    base_url: str = "https://api.anthropic.com/v1",
    api_version: str = "2023-06-01",
)
```

**thinking 参数：**
- `False`: 禁用扩展思考；不支持 thinking 的旧模型会省略参数
- `True`: Claude 4.6+ 使用 adaptive thinking；Claude 3.7/4.0-4.5 使用默认 budget_tokens
- `str`: 统一强度 `minimal/low/medium/high/xhigh/max/ultra`
- `int`: 为 Claude 3.7/4.0-4.6 启用并指定 budget_tokens；4.7+ 会明确报错
- `dict`: 直接传入 Anthropic 原生 thinking 配置
- `None`: 使用模型默认行为

也可传 `reasoning_effort` 作为统一强度入口。Claude 4.6+ 会生成
`thinking={"type": "adaptive"}` 与 `output_config.effort`；Claude 3.7/4.0-4.5 自动换算为
token 预算；Claude 3.5 及更早版本不支持该参数。
Fable/Mythos 5 的 adaptive thinking 始终开启，传 `thinking=False` 会明确报错。

**静态方法：**

```python
@staticmethod
def parse_thoughts(response_data: dict) -> dict
```

解析思考内容，返回 `{"thought": str, "answer": str}`。

---

### GeminiClient

Google Gemini 客户端。

```python
from flexllm import GeminiClient

# Developer API 模式
client = GeminiClient(
    api_key: str,
    model: str = "gemini-2.5-flash",
)

# Vertex AI 模式
client = GeminiClient(
    project_id: str,
    location: str = "us-central1",
    model: str = "gemini-2.5-flash",
    use_vertex_ai: bool = True,
)
```

**thinking 参数：**
- `False`: 禁用
- `True`: 启用
- `"minimal"`, `"low"`, `"medium"`, `"high"`: 思考级别

---

### LLMClientPool

多 Endpoint 客户端池，支持容量感知的负载均衡和故障转移。

```python
from flexllm import LLMClientPool

pool = LLMClientPool(
    endpoints: List[dict] = None,        # Endpoint 配置列表
    clients: List[LLMClient] = None,     # 或直接传入客户端
    fallback: bool = True,               # 故障转移
    failure_threshold: int = 3,          # 失败阈值
    recovery_time: float = 60.0,         # 恢复时间（秒）
)
```

**方法：** 与 LLMClient 完全一致。

---

## 数据类

### ChatCompletionResult

```python
@dataclass
class ChatCompletionResult:
    content: str                          # 响应内容
    usage: Optional[dict] = None          # Token 使用情况
    reasoning_content: Optional[str] = None  # 思考内容
    tool_calls: Optional[list[ToolCall]] = None
    queue_time: Optional[float] = None    # 客户端排队耗时（semaphore + QPS 漏桶）；缓存命中为 None
    finish_reason: Optional[str] = None   # 停止原因（OpenAI 语义，见 chat_completions_stream 的 finish 事件）；缓存命中为 None
    assistant_message: Optional[dict] = None  # 下一轮应原样回传的 provider 续接消息
    latency: Optional[float] = None       # 端到端耗时；缓存命中为 None
```

`latency = queue_time + service_time`，只覆盖真实请求，不含消息预处理（图片下载转
base64）。想知道某个 endpoint 自己有多快，用 `latency - queue_time`，多 endpoint
选路正是据此判断快慢（见[高级用法 · 负载均衡策略](advanced.md#负载均衡策略)）。

`finish_reason == "length"` 表示输出被 `max_tokens` 截断。注意 reasoning 模型的思考
tokens 也计入 `max_tokens`：预算过小时可能 `content` 为空而 `finish_reason == "length"`。
当工具调用伴随独立思考状态时，下一轮应回传 `assistant_message`，不要只从正文和
`tool_calls` 重建 assistant 消息。

### BatchResultItem

```python
@dataclass
class BatchResultItem:
    index: int                    # 请求索引
    content: Optional[str]        # 响应内容
    usage: Optional[dict]         # Token 使用
    status: str                   # "success", "error", "cached"
    error: Optional[str]          # 错误信息
    latency: float                # 延迟（秒）
```

### TranscriptionResult

```python
@dataclass
class TranscriptionResult:
    text: str                     # 转录文本
    language: Optional[str]       # 识别出的语言
    duration: Optional[float]     # 音频时长（秒）
    segments: list[dict]          # 分段（含 start/end/text）
    raw: Optional[dict]           # 服务端原始响应

    def to_srt(self) -> str: ...  # 渲染 SRT 字幕
    def to_vtt(self) -> str: ...  # 渲染 WebVTT 字幕
```

---

## 语音端点

OpenAI 兼容客户端（含 `LLMClient`）提供转录与合成，详见 [语音能力](audio.md)。

```python
# 转录 /audio/transcriptions
text = client.transcribe_sync("a.wav", model="glm-asr")
result = client.transcribe_sync("a.wav", model="glm-asr", return_details=True)
texts = client.transcribe_batch_sync(["a.wav", "b.wav"], model="glm-asr")

# 合成 /audio/speech
audio: bytes = client.speech_sync("你好", model="glm-tts", voice="tongtong")
path = client.speech_sync("你好", model="glm-tts", output="hello.wav")
paths = client.speech_batch_sync(["一", "二"], model="glm-tts", outputs=["1.wav", "2.wav"])
```

去掉 `_sync` 后缀即为异步版本。并发与限流沿用客户端的 `concurrency_limit` / `max_qps`。

---

## 缓存配置

### ResponseCacheConfig

```python
from flexllm import ResponseCacheConfig

config = ResponseCacheConfig(
    enabled: bool = False,
    ttl: int = 86400,                    # TTL（秒），0 表示永不过期
    cache_dir: str = "~/.flexllm/cache/response",
)

# 快捷方法
ResponseCacheConfig.with_ttl(3600)       # 1 小时
ResponseCacheConfig.persistent()          # 永久
```

---

## Token 计数

```python
from flexllm import (
    count_tokens,
    count_messages_tokens,
    estimate_cost,
    estimate_batch_cost,
    messages_hash,
    MODEL_PRICING,
)

# 计数
tokens = count_tokens("Hello world", model="gpt-4")
tokens = count_messages_tokens(messages, model="gpt-4")

# 成本估算
cost = estimate_cost(tokens, model="gpt-4", is_input=True)
total = estimate_batch_cost(messages_list, model="gpt-4")

# 消息哈希（用于缓存 key）
hash_str = messages_hash(messages)
```

**支持的模型定价：**

```python
MODEL_PRICING = {
    "gpt-4o": {"input": 2.5/1e6, "output": 10/1e6},
    "gpt-4": {"input": 30/1e6, "output": 60/1e6},
    "gpt-3.5-turbo": {"input": 0.5/1e6, "output": 1.5/1e6},
    "claude-3-5-sonnet": {"input": 3/1e6, "output": 15/1e6},
    "deepseek-chat": {"input": 0.14/1e6, "output": 0.28/1e6},
    "qwen-max": {"input": 2/1e6, "output": 6/1e6},
    # ...
}
```

---

## 响应解析

```python
from flexllm import extract_code_snippets, parse_to_obj, parse_to_code

# 提取代码片段
snippets = extract_code_snippets(text)
# 返回: [{"language": "python", "code": "..."}, ...]

# 解析为 Python 对象
obj = parse_to_obj(text)

# 提取代码字符串
code = parse_to_code(text)
```

---

## Mock 服务器

用于测试和开发的轻量级 Mock LLM 服务器，支持 OpenAI / Claude / Gemini 三种 API 格式。

### 基本用法

```python
from flexllm.mock import MockLLMServer, MockServerConfig

config = MockServerConfig(
    port=8001,              # 端口号
    delay_min=0.1,          # 最小延迟（秒）
    delay_max=0.1,          # 最大延迟
    model="mock-model",     # 模型名称
    response_min_len=10,    # 响应最小长度（字符）
    response_max_len=1000,  # 响应最大长度
    rps=0,                  # RPS 限制，0 不限制
    token_rate=0,           # 流式 token 速率，0 不限制
    error_rate=0,           # 错误率 (0-1)
    thinking=False,         # 是否返回思考内容
    qa_path="qa.jsonl",     # QA 数据集路径，匹配输入时返回确定性回复
)

# 上下文管理器方式（后台进程）
with MockLLMServer(config) as server:
    # OpenAI / Claude: server.url -> "http://localhost:8001/v1"
    # Gemini: server.gemini_url -> "http://localhost:8001"
    pass
```

### 支持的 API 端点

| 格式 | 端点 | base_url |
|------|------|----------|
| OpenAI | `POST /v1/chat/completions` | `server.url` |
| Claude | `POST /v1/messages` | `server.url` |
| Gemini | `POST /models/{model}:generateContent` | `server.gemini_url` |
| Gemini 流式 | `POST /models/{model}:streamGenerateContent` | `server.gemini_url` |

### 思考内容触发方式

通过 `MockServerConfig(thinking=True)` 全局启用，或通过请求参数动态触发：

| 格式 | 请求参数 |
|------|----------|
| OpenAI | `"think": true` |
| Claude | `"thinking": {"type": "enabled", "budget_tokens": 10000}` |
| Gemini | `"generationConfig": {"thinkingConfig": {"includeThoughts": true}}` |

### QA 数据集（确定性回复）

通过 `--qa` 指定 JSONL 文件，当用户输入精确匹配时返回预设回复，未匹配则回退随机生成：

```jsonl
{"input": "你好", "output": "你好！有什么可以帮你的？"}
{"input": "1+1等于几", "output": "2"}
```

### 请求日志

通过 `--log` 记录每个请求的输入输出到 JSONL 文件，base64 图片自动替换为占位符（如 `<image:32.1KB>`）：

```bash
flexllm mock --log requests.jsonl
```

日志格式：
```json
{"timestamp": "...", "api_format": "openai", "request": {...}, "output": "回复文本", "prompt_tokens": 10, "completion_tokens": 50}
```

### CLI

```bash
flexllm mock                          # 默认配置
flexllm mock --thinking               # 启用思考内容
flexllm mock -p 8080 -d 0.1-0.5      # 自定义端口和延迟
flexllm mock --error-rate 0.3         # 30% 错误率
flexllm mock --qa qa.jsonl            # 使用 QA 数据集确定性回复
flexllm mock --log requests.jsonl     # 记录请求日志
```

---

## Provider 路由

```python
from flexllm import ProviderRouter, ProviderConfig, create_router_from_urls

# 快速创建
router = create_router_from_urls(
    urls=["http://host1:8000/v1", "http://host2:8000/v1"],
    api_key="EMPTY",
)

# 获取下一个 provider
provider = router.get_next()

# 更新状态
router.mark_success(provider)
router.mark_failed(provider)
```
