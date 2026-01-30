---
name: flexllm
description: LLM API 客户端 - 批量处理、断点续传、响应缓存、负载均衡、成本追踪
---

# flexllm

支持 OpenAI 兼容 API（vLLM/Ollama/DeepSeek 等）、Gemini、Claude。

## Python API

```python
from flexllm import LLMClient, LLMClientPool

# 单条请求
async with LLMClient(model="gpt-4", base_url="https://api.openai.com/v1", api_key="...") as client:
    result = await client.chat_completions([{"role": "user", "content": "Hello"}])
    # 同步: client.chat_completions_sync(messages)
    # 流式: async for chunk in client.chat_completions_stream(messages)
    # usage: await client.chat_completions(messages, return_usage=True) → ChatCompletionResult
    # thinking: await client.chat_completions(messages, thinking=True, return_raw=True)

# 批处理（支持断点续传、成本追踪）
results = await client.chat_completions_batch(
    messages_list,
    output_jsonl="results.jsonl",   # 断点续传文件
    save_input=False,               # True(默认) | "last" | False
    show_progress=True,
    track_cost=True,
)

# 响应缓存
from flexllm import ResponseCacheConfig
client = LLMClient(..., cache=ResponseCacheConfig(enabled=True, ttl=3600))

# 多 Provider（自动检测: base_url 含 googleapis→gemini, anthropic→claude, 其他→openai）
LLMClient(provider="gemini", model="gemini-2.5-flash", api_key="...")
LLMClient(provider="claude", model="claude-sonnet-4-20250514", api_key="...")

# 负载均衡
pool = LLMClientPool(
    endpoints=[
        {"base_url": "http://gpu1:8000/v1", "model": "qwen", "concurrency_limit": 50},
        {"base_url": "http://gpu2:8000/v1", "model": "qwen"},
    ],
    fallback=True,
)
results = await pool.chat_completions_batch(messages_list, output_jsonl="results.jsonl")
```

## CLI

```bash
flexllm ask "问题"                     # 快速问答
flexllm chat                           # 交互式聊天
flexllm list                           # 已配置模型
flexllm test                           # 测试连接
flexllm pricing gpt-4o                 # 查询定价
flexllm credits                        # 查询余额
flexllm mock                           # Mock 服务器（测试用）
```

### batch 命令（核心）

```bash
flexllm batch input.jsonl -o output.jsonl               # 基本用法
flexllm batch input.jsonl -o output.jsonl -n 5           # 只处理前 5 条（快速试跑）
flexllm batch input.jsonl -o output.jsonl -m gpt-4 -c 20 # 指定模型和并发
flexllm batch data.jsonl -o out.jsonl -uf text -sf sys_prompt  # 指定任意字段名
flexllm batch input.jsonl -o output.jsonl -s "You are helpful" # 全局 system prompt
flexllm batch input.jsonl -o output.jsonl --save-input false   # 不保存 input
```

**参数速查：**

| 参数 | 缩写 | 说明 |
|------|------|------|
| `--output` | `-o` | 输出文件路径（必需） |
| `--limit` | `-n` | 只处理前 N 条（快速试跑） |
| `--model` | `-m` | 模型名称 |
| `--concurrency` | `-c` | 并发数 |
| `--system` | `-s` | 全局 system prompt |
| `--user-field` | `-uf` | 指定 user content 的字段名 |
| `--system-field` | `-sf` | 指定 system prompt 的字段名 |
| `--save-input` | | 输出 input 保存策略：true/last/false |
| `--track-cost` | | 进度条显示实时成本 |
| `--cache/--no-cache` | | 启用/禁用响应缓存 |
| `--temperature` | `-t` | 采样温度 |
| `--max-tokens` | | 最大生成 token 数 |

**输入格式自动检测（按优先级）：**

| 格式 | 识别字段 | 转换规则 |
|------|---------|---------|
| openai_chat | `messages` | 直接使用 |
| alpaca | `instruction` (+可选`input`,`system`) | user=`instruction\n\ninput` |
| simple | `q`/`question`/`prompt`/`input`/`user` (+可选`system`) | 直接作为 user content |
| custom | 通过 `-uf`/`-sf` 指定 | 跳过自动检测 |

未识别的字段自动保留为 metadata。`-s` 全局 system prompt 优先级高于记录级 system。

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
batch:
  concurrency: 20
  cache: true
  track_cost: true
```

环境变量：`FLEXLLM_BASE_URL`/`OPENAI_BASE_URL`, `FLEXLLM_API_KEY`/`OPENAI_API_KEY`, `FLEXLLM_MODEL`/`OPENAI_MODEL`
