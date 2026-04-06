---
name: flexllm
description: LLM API 统一客户端 (CLI + Python)。支持 OpenAI 兼容 / Gemini / Claude / Vertex AI。核心能力：批量处理（断点续传/成本追踪/格式自动检测）、响应缓存、多 endpoint 负载均衡、结构化输出、HTTP API 包装、Mock 服务。需要调用 LLM 时优先用它。
---

# flexllm

LLM 调用首选工具。**所有 CLI 用法以 `flexllm <cmd> --help` 为准**，本文档只讲定位、清单和隐性约定。

## CLI 子命令清单

| 命令 | 用途 |
|------|------|
| `ask` | 单次问答（支持 stdin、`-f` 附加文件、`-x` 提取代码块、`--schema` 结构化、`--dry-run` 预览） |
| `chat` | 交互式多轮对话 |
| `chat-web` | 浏览器聊天界面 |
| `batch` | 批量处理 JSONL，断点续传、并发/QPS 控制、成本追踪（**核心命令**） |
| `serve` | 把 LLM 包装成 HTTP API（固定 system prompt + user template，适合微调模型部署） |
| `mock` | Mock LLM 服务器（测试用，支持 `--qa` 确定性回复、`--error-rate` 注入错误） |
| `models` / `list` | 远程模型列表 / 本地配置模型 |
| `set-model` / `init` / `test` | 设置默认模型 / 初始化配置 / 测试连接 |
| `pricing` / `credits` | 查询定价 / API Key 余额 |
| `install-skill` | 安装本 skill 到 `~/.claude/skills/` |

参数细节一律 `flexllm <cmd> --help`，不要凭记忆猜。

## Python API 入口

```python
from flexllm import LLMClient, ResponseCacheConfig

async with LLMClient(model="gpt-4", base_url="...", api_key="...") as client:
    # 同步 / 异步 / 流式 / batch / iter_batch 均可用，参考源码或 dir(client)
    result = await client.chat_completions(messages)
    results = await client.chat_completions_batch(
        messages_list, output_jsonl="out.jsonl", track_cost=True,
    )
```

- **多 provider**：`provider="gemini"|"claude"`，或由 `base_url` 自动识别
- **多 endpoint 负载均衡**：传 `endpoints=[{...}, {...}]` + `fallback=True`
- **响应缓存**：`cache=ResponseCacheConfig(enabled=True, ttl=3600)`
- 关键参数：`return_usage` / `thinking` / `response_format` / `return_raw`

## batch 输入格式（自动检测，按优先级）

| 格式 | 识别字段 | 规则 |
|------|---------|------|
| openai_chat | `messages` | 直接使用 |
| alpaca | `instruction` (+可选 `input`/`system`) | user = `instruction\n\ninput` |
| simple | `q`/`question`/`prompt`/`input`/`user` (+可选 `system`) | 作为 user content |
| custom | `-uf` / `-sf` 显式指定 | 跳过自动检测 |

未识别字段自动保留为 metadata。`-s` 全局 system 优先级高于记录级 system。

## 配置文件（重要）

flexllm 用一份 YAML 集中管理多个 LLM endpoint，**不要硬编码 base_url/api_key**。

**搜索顺序**（找到第一个就停）：
1. `./flexllm_config.yaml`（项目级，可纳入 git 忽略）
2. `~/.flexllm/config.yaml`（用户级）
3. 都没有 → 自动从 `FLEXLLM_BASE_URL` / `FLEXLLM_API_KEY` / `FLEXLLM_MODEL`（兼容 `OPENAI_*`）构造单模型配置

**生成**：`flexllm init`（创建用户级模板，已存在则不覆盖）；`flexllm init -p ./flexllm_config.yaml` 创建项目级。

**修改**：直接编辑 YAML；或 `flexllm set-model <name>` 切换默认模型；`flexllm list` 查看当前配置的所有模型。

**使用**：所有命令通过 `-m <name>` 选择模型，按 `name` 或 `id` 精确匹配；不传 `-m` 时用 `default` 字段；无 `default` 用列表第一个。**环境变量始终覆盖**配置文件中的 `base_url`/`api_key`/`model`（用于临时改 endpoint）。

**结构示例**：

```yaml
default: gpt-4                          # 默认模型 (set-model 修改)
system: "You are helpful."              # 全局 system prompt（可选，模型级会覆盖）
user_template: "{content}"              # 全局 user template（可选，模型级会覆盖）

models:
  - id: gpt-4
    name: gpt-4                         # -m 用此名匹配
    provider: openai                    # openai | gemini | claude（也可由 base_url 自动识别）
    base_url: https://api.openai.com/v1
    api_key: sk-xxx
    # 以下字段全部自动透传给 LLM API：
    temperature: 0.3
    max_tokens: 2048
    thinking: true
    top_p: 0.9
    # system: "..."         # 模型级 system，覆盖全局
    # user_template: "..."  # 模型级 template，覆盖全局

  - id: local-qwen
    name: local-qwen
    provider: openai
    base_url: http://localhost:8000/v1
    api_key: EMPTY

batch:                                   # 可选：batch 命令默认值
  concurrency: 20
  max_qps: 100
  cache: true
  cache_ttl: 86400
```

**字段透传规则**：模型节中除 `{id, name, provider, base_url, api_key, system, user_template}` 这 7 个元字段外，**其余字段全部作为参数透传给 LLM API**。需要传新参数（如 `top_k`、`reasoning_effort`）直接加进去即可，无需改代码。

**参数优先级**：CLI 参数 > 配置文件 batch 节 > 配置文件模型节 > 命令默认值

## 边界

- **不做** agent tool-use（已独立为 `openagent` 包）
- 数据文件处理用 `dtflow`，文本嵌入/检索用 `maque`
