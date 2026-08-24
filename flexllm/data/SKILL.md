---
name: flexllm
description: LLM API 统一客户端 (CLI + Python)。支持 OpenAI 兼容 / Gemini / Claude / Vertex AI。核心能力：批量处理（断点续传/成本追踪/格式自动检测）、响应缓存、多 endpoint 负载均衡、结构化输出、语音转录/合成、HTTP API 包装、Mock 服务。需要调用 LLM 时优先用它。
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
| `transcribe` | 语音转录（音频转文字），多文件并发，`-f text/json/srt/vtt`（字幕本地由 segments 渲染） |
| `speak` | 语音合成（文字转音频），支持 stdin 输入、`-o -` 写 stdout 接管道 |
| `models` / `list` | 远程模型列表 / 本地配置模型 |
| `set-model` / `init` / `test` | 设置默认模型 / 初始化配置 / 测试连接 |
| `pricing` / `credits` | 查询定价 / API Key 余额 |
| `version` / `--version` | 版本信息 |
| `install-skill` | 安装本 skill；默认 Claude Code，`--target codex` 安装到 Codex |

参数细节一律 `flexllm <cmd> --help`，不要凭记忆猜。

## Agent 友好约定

- **JSON 输出**：`ask` / `chat` / `batch` 支持 `--format json`，stdout 为结构化数据，stderr 为进度/日志；非 TTY 下错误自动以 JSON 输出到 stderr。`transcribe -f json` 输出含 segments/language/duration。
- **退出码**（跨版本稳定）：0 成功 / 1 通用错误 / 2 参数用法错误 / 3 资源未找到 / 4 认证失败 / 5 冲突 / 6 网络错误（通常可重试）/ 7 依赖缺失 / 8 文件 IO 错误 / 10 dry-run 成功。
- **`--dry-run`** 几乎所有执行类命令都支持，先预览再跑。
- **`-f` 含义因命令而异**：`ask` 的 `-f` 是 `--file`（附加文件），`transcribe`/`speak` 的 `-f` 是 `--format`（输出格式）。别跨命令套用短参。

## Python API 入口

```python
from flexllm import LLMClient, ResponseCacheConfig

async with LLMClient(model="gpt-4", base_url="...", api_key="...") as client:
    result = await client.chat_completions(messages)
    results = await client.chat_completions_batch(
        messages_list, output_jsonl="out.jsonl", track_cost=True,
    )
    async for chunk in client.chat_completions_stream(messages): ...
```

- **多 provider**：`provider="gemini"|"claude"`，或由 `base_url` 自动识别
- **多 endpoint 负载均衡**：传 `endpoints=[{...}, {...}]` + `fallback=True`（`LLMClientPool`，`distribute=True` 可把 batch 分散到多 endpoint）
- **响应缓存**：`cache=ResponseCacheConfig(enabled=True, ttl=3600)`（LMDB 后端，多进程安全，默认目录 `~/.flexllm/cache/response`）；图片缓存另开关 `cache_image=True`（目录 `~/.flexllm/cache/image_cache`）
- **代理**：客户端级 `proxy="http://gateway:8080"`（也支持 socks5），LLM 请求与多模态图片/媒体 URL 下载都走它；不传则遵循环境变量
- 核心方法：`chat_completions` / `chat_completions_sync` / `chat_completions_batch` / `chat_completions_stream`
- **语音**：`transcribe` / `transcribe_batch` / `speech` / `speech_batch`（各有 `_sync` 版本），返回 `TranscriptionResult`，配 `segments_to_srt` / `segments_to_vtt` 生成字幕
- 关键参数：`return_usage` / `thinking`（跨厂商统一：DeepSeek-R1/Qwen3/Claude/Gemini，取值 `True|False|"minimal"|"low"|"medium"|"high"`）/ `response_format` / `return_raw`
- 其余 kwargs 原样透传给 API（如 `tools`）；响应中的 `tool_calls` 会解析进 `ChatCompletionResult.tool_calls`（注意：缓存不存储 tool_calls）

## Python API 高级能力

`from flexllm import ...` 除 `LLMClient` 外还提供：

| 类 | 定位 | 典型入口 |
|---|---|---|
| `MllmClient` | 多模态（图片/视频抽帧/音频 `input_audio`），自动预处理（base64/URL/本地路径） | `await mllm.call_llm(messages_list)` |
| `ChainOfThoughtClient` | 多步推理链，可根据上一步结果动态决定下一步模型/prompt，支持批量并发 | `add_step(Step(...))` / `create_linear_chain([LinearStep(...)])` / `await execute_chain(...)` / `await execute_chains_batch(...)` |
| `MllmTableProcessor` | 表格批处理（需 pandas），通过 `MllmClient.table` 属性访问 | `mllm.table.load_dataframe(...)` + `mllm.table.call_llm(...)` |
| `MllmFolderProcessor` | 文件夹批处理（扫描图片目录），通过 `MllmClient.folder` 属性访问 | `mllm.folder.scan_folder_images(...)` + `mllm.folder.call_llm(...)` |

**成本追踪**：`from flexllm import CostTracker, estimate_batch_cost, count_tokens, MODEL_PRICING`，支持预算上限（超出抛 `BudgetExceededError`）。

**语音**：`from flexllm import TranscriptionResult, segments_to_srt, segments_to_vtt`；音频消息预处理需可选依赖组 `flexllm[audio]`。

**工具函数**：`from flexllm.utils import extract_code_snippets, parse_to_code, parse_to_obj`（对应 CLI 的 `-x` 和 `--schema` 后处理）。

## batch 输入格式（自动检测，按优先级）

| 格式 | 识别字段 | 规则 |
|------|---------|------|
| openai_chat | `messages` | 直接使用 |
| alpaca | `instruction` (+可选 `input`/`system`) | user = `instruction\n\ninput` |
| simple | `q`/`question`/`prompt`/`input`/`user` (+可选 `system`) | 作为 user content |
| custom | `-uf` / `-sf` 显式指定 | 跳过自动检测 |

未识别字段自动保留为 metadata。`input` 只在没有 `instruction` 时才当作 simple 格式的 user content。

**system 是兜底不是覆盖**：`-s` 指定的全局 system 只在记录内没有 system 时才插入，行内 system 始终优先。`--user-template` 只套用到最后一条 content 为 str 的 user 消息，其余 user 消息不动。

## 配置文件（重要）

flexllm 用一份 YAML 集中管理多个 LLM endpoint，**不要硬编码 base_url/api_key**。

**搜索顺序**（找到第一个就停）：
1. `./flexllm_config.yaml`（项目级，可纳入 git 忽略）
2. `~/.flexllm/config.yaml`（用户级）
3. 都没有 → 自动从 `FLEXLLM_BASE_URL` / `FLEXLLM_API_KEY` / `FLEXLLM_MODEL`（兼容 `OPENAI_*`）构造单模型配置

**生成**：`flexllm init`（创建用户级模板，已存在则不覆盖）；`flexllm init -p ./flexllm_config.yaml` 创建项目级。

**修改**：直接编辑 YAML；或 `flexllm set-model <name>` 切换默认模型；`flexllm list` 查看当前配置的所有模型。

**使用**：所有命令通过 `-m <name>` 选择模型，按 `name` 或 `id` 精确匹配；不传 `-m` 时用 `default` 字段；无 `default` 用列表第一个。

**优先级链（高 → 低）**：
1. CLI 参数（`--base-url` / `--api-key`）
2. 配置文件具名模型（`-m` 命中的条目**原样使用，环境变量不覆盖它**）
3. 环境变量（`FLEXLLM_*` / `OPENAI_*`，仅在未显式选中模型、或 `-m` 名称未命中配置文件时生效）
4. 配置文件 `default` 指向的模型，其次列表第一个

即：显式 `-m` 命中 > 环境变量。想用环境变量临时改 endpoint，就别同时传一个已在配置里的 `-m` 名字。

加载配置时会自动 `load_dotenv()` 读取 `.env`。

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

batch:                                   # 可选：batch 命令的调度/IO 默认值
  model: gpt-4                           # 单 endpoint 模式；与 endpoints 二选一
  concurrency: 20
  max_qps: 100
  timeout: 120
  retry_times: 3
  cache: true
  cache_ttl: 86400
  return_usage: true
  track_cost: true
  # endpoints: [...]                     # 多 endpoint 模式（与 model 二选一）
  # fallback: true
  # total_concurrency: 100               # pool 级全局硬上限（跨所有 endpoint）
  # total_max_qps: 200
```

**字段透传规则**：模型节中除 `{id, name, provider, base_url, api_key, system, user_template, endpoints, fallback}` 这 9 个元字段外，**其余字段全部作为参数透传给 LLM API**。需要传新参数（如 `top_k`、`reasoning_effort`）直接加进去即可，无需改代码。

**batch 节只放调度/IO 参数**：模型行为参数（`temperature`/`top_p`/`top_k`/`max_tokens`/`thinking`）必须写在 `models` 节，临时覆盖用 CLI 参数。

**参数优先级**：CLI 参数 > 配置文件 batch 节 > 配置文件模型节 > 命令默认值

## 边界

- **不做** agent 循环（tool-use 编排已独立为 `openagent` 包）；单次请求可透传 `tools` 并拿到 `tool_calls`，但工具执行/多轮回灌不由本库负责
- 数据文件处理用 `dtflow`，文本嵌入/检索用 `maque`
