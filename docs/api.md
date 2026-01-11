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
    concurrency_limit: int = 50,         # 最大并发数
    max_qps: float = None,               # QPS 限制
    retry_times: int = 3,                # 重试次数
    retry_delay: float = 1.0,            # 重试延迟（秒）
    timeout: int = 120,                  # 请求超时（秒）
)
```

**方法：**

#### chat_completions

```python
async def chat_completions(
    messages: List[dict],
    model: str = None,
    return_raw: bool = False,
    return_usage: bool = False,
    skip_cache: bool = False,
    **kwargs
) -> str | ChatCompletionResult
```

单条异步请求。

**参数：**
- `messages`: 消息列表
- `model`: 覆盖默认模型
- `return_raw`: 返回原始响应对象
- `return_usage`: 返回 token 使用情况
- `skip_cache`: 跳过缓存

#### chat_completions_sync

```python
def chat_completions_sync(messages, **kwargs) -> str | ChatCompletionResult
```

单条同步请求（内部使用 asyncio.run）。

#### chat_completions_batch

```python
async def chat_completions_batch(
    messages_list: List[List[dict]],
    output_file: str = None,
    show_progress: bool = True,
    return_summary: bool = False,
    flush_interval: float = 1.0,
    metadata_list: List[dict] = None,
    **kwargs
) -> List[str] | Tuple[List[str], dict]
```

批量异步请求，支持断点续传。

**参数：**
- `messages_list`: 消息列表的列表
- `output_file`: 输出文件路径（启用断点续传）
- `show_progress`: 显示进度条
- `return_summary`: 返回统计摘要
- `flush_interval`: 写入磁盘间隔
- `metadata_list`: 元数据列表，与 `messages_list` 等长，每条记录的元数据会保存到输出文件

#### chat_completions_stream

```python
async def chat_completions_stream(
    messages: List[dict],
    **kwargs
) -> AsyncIterator[str]
```

流式响应。

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
  - `None`: 使用模型默认行为

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

多 Endpoint 客户端池，支持负载均衡和故障转移。

```python
from flexllm import LLMClientPool

pool = LLMClientPool(
    endpoints: List[dict] = None,        # Endpoint 配置列表
    clients: List[LLMClient] = None,     # 或直接传入客户端
    load_balance: str = "round_robin",   # 负载均衡策略
    fallback: bool = True,               # 故障转移
    failure_threshold: int = 3,          # 失败阈值
    recovery_time: float = 60.0,         # 恢复时间（秒）
)
```

**负载均衡策略：**
- `round_robin`: 轮询
- `weighted`: 加权随机
- `random`: 随机
- `fallback`: 主备模式

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
```

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

---

## 缓存配置

### ResponseCacheConfig

```python
from flexllm import ResponseCacheConfig

config = ResponseCacheConfig(
    enabled: bool = False,
    ttl: int = 86400,                    # TTL（秒），0 表示永不过期
    cache_dir: str = "~/.cache/flexllm/llm_response",
    use_ipc: bool = True,                # IPC 模式（多进程共享）
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

## Provider 路由

```python
from flexllm import ProviderRouter, ProviderConfig, create_router_from_urls

# 快速创建
router = create_router_from_urls(
    urls=["http://host1:8000/v1", "http://host2:8000/v1"],
    api_key="EMPTY",
    strategy="round_robin",
)

# 获取下一个 provider
provider = router.get_next()

# 更新状态
router.mark_success(provider)
router.mark_failed(provider)
```
