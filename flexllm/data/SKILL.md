---
name: flexllm
description: LLM API 客户端 - 批量处理、断点续传、响应缓存、负载均衡、成本追踪、Agent tool-use
---

# flexllm

支持 OpenAI 兼容 API（vLLM/Ollama/DeepSeek 等）、Gemini、Claude。

## Python API

```python
from flexllm import LLMClient

# 单条请求
async with LLMClient(model="gpt-4", base_url="https://api.openai.com/v1", api_key="...") as client:
    result = await client.chat_completions([{"role": "user", "content": "Hello"}])
    # 同步: client.chat_completions_sync(messages)
    # 流式: async for chunk in client.chat_completions_stream(messages)
    # usage: await client.chat_completions(messages, return_usage=True) → ChatCompletionResult
    # thinking: await client.chat_completions(messages, thinking=True, return_raw=True)
    # response_format: await client.chat_completions(messages, response_format={"type":"json_object"})

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

# 负载均衡（多 endpoint，同一个 LLMClient）
client = LLMClient(
    endpoints=[
        {"base_url": "http://gpu1:8000/v1", "model": "qwen", "concurrency_limit": 50},
        {"base_url": "http://gpu2:8000/v1", "model": "qwen"},
    ],
    fallback=True,
)
results = await client.chat_completions_batch(messages_list, output_jsonl="results.jsonl")
```

### AgentClient（tool-use 循环）

组合 LLMClient，自动执行 tool-calling 循环：LLM 调用 → 执行工具 → 回传结果 → 循环直到完成。

```python
from flexllm import AgentClient, LLMClient

client = LLMClient(model="gpt-4", base_url="...", api_key="...")
agent = AgentClient(
    client=client,
    system="你是一个助手",
    tools=[{"type":"function","function":{"name":"get_weather","parameters":{...}}}],
    tool_executor=my_fn,    # (name, arguments_json) -> result（同步/异步均可）
    max_rounds=10,
)

# 单次任务（无状态）
result = await agent.run("查北京天气")
# result.content, result.rounds, result.tool_calls, result.usage

# 多轮对话（有状态，自动维护历史）
r1 = await agent.chat("你好")
r2 = await agent.chat("帮我查天气")  # 带 r1 上下文
agent.reset()

# Structured output（Pydantic 绑定）
from pydantic import BaseModel
class Decision(BaseModel):
    action: str
    reason: str
result = await agent.run("分析需求", response_format=Decision)
result.parsed  # Decision(action="approve", reason="...")

# 事件回调
agent.on_tool_call = lambda name, args: print(f"调用: {name}")
agent.on_tool_result = lambda name, result: print(f"结果: {result[:100]}")
```

## CLI

```bash
flexllm ask "问题"                     # 快速问答
flexllm chat                           # 交互式聊天
flexllm chat --tools shell             # Agent 模式（带 shell 工具）
flexllm agent --tools shell "查CPU使用率" # Agent 非交互式执行
flexllm list                           # 已配置模型
flexllm test                           # 测试连接
flexllm pricing gpt-4o                 # 查询定价
flexllm credits                        # 查询余额
flexllm mock                           # Mock 服务器（测试用）
flexllm serve                          # 启动 HTTP API 服务
```

### serve 命令

将 LLM 包装为 HTTP API，适用于微调模型部署。固定 system prompt + user template，调用方只需发送 content。

```bash
flexllm serve -m qwen-finetuned -s "你是助手" --user-template "[INST]{content}[/INST]"
flexllm serve --thinking true -p 8000 -v       # 思考模式 + 请求日志
# API: POST /api/generate {"content":"文本"} → {"content":"...", "thinking":"...", "usage":{...}}
#      POST /api/generate/stream (SSE), POST /api/generate/batch, GET /health
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
    # 模型调用参数（除 id/name/provider/base_url/api_key/system/user_template 外
    # 的所有字段都会自动透传给 LLM API）
    # max_tokens: 512
    # temperature: 0.3
    # top_p: 0.9
    # thinking: true
batch:
  concurrency: 20
  cache: true
  track_cost: true
```

模型调用参数优先级：CLI 参数 > batch 配置 > 模型配置 > 命令默认值

环境变量：`FLEXLLM_BASE_URL`/`OPENAI_BASE_URL`, `FLEXLLM_API_KEY`/`OPENAI_API_KEY`, `FLEXLLM_MODEL`/`OPENAI_MODEL`
