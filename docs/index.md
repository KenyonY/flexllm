# flexllm 文档

高性能 LLM 客户端库，支持批量处理、响应缓存和断点续传。

## 文档目录

```
docs/
├── index.md              # 本文档（主入口）
├── api.md                # API 详细参考
└── advanced.md           # 高级用法
```

## 快速开始

### 安装

```bash
pip install flexllm[all]
```

### 基本使用

```python
from flexllm import LLMClient

client = LLMClient(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key"
)

# 同步调用
result = client.chat_completions_sync([
    {"role": "user", "content": "Hello!"}
])
print(result)
```

## 核心概念

### 1. 客户端层次

```
LLMClient (推荐，统一入口)
    ├── OpenAIClient (OpenAI 兼容 API)
    └── GeminiClient (Google Gemini)

LLMClientPool (多 Endpoint 负载均衡)
    └── 内部管理多个 LLMClient
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

### 4. 断点续传

批量处理支持自动断点续传：

```python
results = await client.chat_completions_batch(
    messages_list,
    output_file="results.jsonl",  # 关键：指定输出文件
)
```

- 结果增量写入文件
- 程序中断后，重新运行自动跳过已完成的请求
- 配合缓存使用效果更好

## 支持的 Provider

| Provider | 客户端 | 说明 |
|----------|--------|------|
| OpenAI | LLMClient/OpenAIClient | GPT 系列 |
| DeepSeek | LLMClient/OpenAIClient | 支持 thinking 模式 |
| Qwen | LLMClient/OpenAIClient | 通义千问 |
| vLLM | LLMClient/OpenAIClient | 本地部署 |
| Ollama | LLMClient/OpenAIClient | 本地部署 |
| Gemini | LLMClient/GeminiClient | Google AI |
| Vertex AI | GeminiClient | GCP 托管 |

## CLI 工具

flexllm 提供命令行工具（别名 `xllm`），支持 Tab 自动补全。

```bash
# 安装自动补全（可选）
flexllm --install-completion
```

### 命令一览

| 命令 | 说明 |
|------|------|
| `ask` | 快速问答（支持管道输入） |
| `chat` | 交互式对话 |
| `batch` | 批量处理 JSONL（支持断点续传） |
| `list` | 列出本地配置的模型 |
| `set-model` | 设置默认模型 |
| `models` | 列出远程服务器的可用模型 |
| `test` | 测试 LLM 服务连接 |
| `init` | 初始化配置文件 |
| `version` | 显示版本信息 |

### 快速示例

```bash
# 快速问答
flexllm ask "什么是 Python?"
flexllm ask "解释代码" -s "你是代码专家"
echo "长文本" | flexllm ask "总结一下"

# 交互对话
flexllm chat
flexllm chat "你好" -m gpt-4

# 模型管理
flexllm list                      # 查看已配置模型
flexllm set-model gpt-4           # 设置默认模型

# 批量处理（支持断点续传）
flexllm batch input.jsonl -o output.jsonl

# 测试连接
flexllm test

# 初始化配置
flexllm init
```

### 配置文件

配置文件位置：`~/.flexllm/config.yaml`（运行 `flexllm init` 创建）

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
```

也支持环境变量配置：`FLEXLLM_BASE_URL`、`FLEXLLM_API_KEY`、`FLEXLLM_MODEL`

### batch 命令

批量处理 JSONL 文件，自动检测输入格式，支持断点续传。

**支持的输入格式：**

| 格式 | 检测规则 | 示例 |
|------|----------|------|
| openai_chat | 存在 `messages` 字段 | `{"messages": [{"role": "user", "content": "..."}]}` |
| alpaca | 存在 `instruction` 字段 | `{"instruction": "翻译", "input": "Hello"}` |
| simple | 存在 `q`/`question`/`prompt` 字段 | `{"q": "什么是AI?", "system": "你是专家"}` |

**使用示例：**

```bash
# 基本用法
flexllm batch input.jsonl -o output.jsonl

# 指定模型和并发数
flexllm batch input.jsonl -o output.jsonl -m gpt-4 -c 20

# 与 dtflow 配合（管道输入）
dt transform qa.jsonl --preset=openai_chat | flexllm batch -o output.jsonl

# 全局 system prompt
flexllm batch input.jsonl -o output.jsonl --system "你是翻译专家"

# 断点续传（默认行为，中断后重新运行即可继续）
flexllm batch input.jsonl -o output.jsonl
```

**输出格式：**

```jsonl
{"index": 0, "output": "LLM响应", "status": "success", "input": [...], "metadata": {"id": "001"}}
```

- `output`: LLM 响应内容
- `input`: 转换后的 messages 格式
- `metadata`: 输入文件中除 messages 外的其他字段

## 下一步

- [API 详细参考](api.md) - 完整的 API 文档
- [高级用法](advanced.md) - 负载均衡、多模态、链式推理等
