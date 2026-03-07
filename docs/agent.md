# Agent 模块

v0.6.0 新增。将 flexllm 从 LLM 客户端工具升级为 Agent 基础设施：既能被 agent 调用，也能让 agent 通过它调用外部世界。

## 架构总览

```
外部 MCP Server          其他 Agent / 前端系统
  (GitHub, DB,             (通过 HTTP API 调用)
   Browser...)                     |
       |                           |
       v                           v
+-----------------------------------------------------------------------------+
|                           flexllm Agent 层                                   |
|                                                                              |
|  +----------+  +-----------+  +------------+  +---------+                   |
|  | MCP Client|  | Agent HTTP|  | Project Inst|  | Skills  |                  |
|  | (消费工具) |  | Serve     |  | (.flexllm.md)|  |         |                 |
|  +-----+----+  +-----+-----+  +------+-----+  +----+----+                  |
|        |              |               |              |                       |
|        v              v               v              v                       |
|  +----------------------------------------------------------------+          |
|  |                    AgentClient (核心)                            |          |
|  |  tool-use 循环 | ToolRegistry | 审批机制 | Structured Output    |          |
|  +-----+----------------------------------------------------------+          |
|        |                                                                      |
|  +-----+----------------------------------------------------------+          |
|  | Memory  |  Tracing  |  Validators  |  Context Compression       |          |
|  +-----+----------------------------------------------------------+          |
|        |                                                                      |
|  +-----+----------------------------------------------------------+          |
|  |              基础设施层 (已有)                                    |          |
|  |  LLMClient | ConcurrentRequester | ResponseCache                |          |
|  +----------------------------------------------------------------+          |
+------------------------------------------------------------------------------+
```

## 安装

```bash
pip install flexllm[agent]    # MCP 支持（mcp SDK）
pip install flexllm[memory]   # 持久化记忆（flaxkv2）
pip install flexllm[all]      # 全部功能
```

---

## 1. 项目指令 (.flexllm.md)

在项目根目录放置 `.flexllm.md` 文件，agent 启动时自动加载到 system prompt 中，为 agent 提供项目上下文。

### 用法

在项目根目录创建 `.flexllm.md`：

```markdown
这是 flexllm 项目，一个高性能 LLM 客户端库。

代码规范：
- Python 3.10+，使用 ruff 格式化
- 用中文写注释
- 遵循 KISS 原则

项目结构：
- flexllm/clients/ — LLM 客户端实现
- flexllm/agent/ — Agent 模块
- tests/unit/ — 单元测试
```

agent 启动时会从当前目录向上搜索，找到第一个 `.flexllm.md` 即加载。子目录中运行时也能自动找到父目录的配置。

### 效果

项目指令会以 `# Project Instructions` 追加到 system prompt 中：

```
[基础 system prompt]

# Project Instructions

[.flexllm.md 内容]
```

---

## 2. Skills — 可复用 Prompt 模板

将常用的 prompt 模板保存为 skill 文件，通过 `--skill` 参数快速加载。

### Skill 格式

与 Claude Code 的 skill 格式一致，使用 YAML frontmatter + Markdown 正文：

```
~/.flexllm/skills/
├── code-review/
│   └── SKILL.md          # 目录模式（推荐）
├── translate.md           # 扁平模式（也支持）
```

### 创建 Skill

```bash
mkdir -p ~/.flexllm/skills/code-review

cat > ~/.flexllm/skills/code-review/SKILL.md << 'EOF'
---
name: code-review
description: 代码审核。审查代码质量、检查最佳实践、发现潜在问题。
---

你是一位资深代码审查专家。审查代码时请关注：

1. **安全性**：SQL注入、XSS、命令注入等安全漏洞
2. **性能**：不必要的循环、内存泄漏、N+1 查询
3. **可读性**：命名规范、函数长度、注释质量

输出格式：按严重程度分类（Critical / Warning / Suggestion），给出具体行号和修改建议。
EOF
```

### Frontmatter 字段

| 字段 | 说明 |
|------|------|
| `name` | skill 名称 |
| `description` | skill 描述（用于展示） |

### 使用

```bash
# agent 模式
flexllm agent --tools code --skill code-review "审查 main.py"

# 交互式对话
flexllm chat --tools code --skill code-review
```

skill 内容会以 `# Skill: {name}` 追加到 system prompt 中，与项目指令叠加：

```
[基础 system prompt]

# Project Instructions
[.flexllm.md 内容]

# Skill: code-review
[skill 正文内容]
```

---

## 3. ToolRegistry — 动态工具注册

运行时动态管理工具集，是 MCP 集成和自定义工具的基础。

### 基本用法

```python
from flexllm import ToolRegistry, ToolDef, AgentClient, LLMClient

# 从内置工具加载
registry = ToolRegistry.from_global(["read", "edit", "bash"])

# 注册自定义工具
registry.register(ToolDef(
    name="search_docs",
    description="搜索文档库",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    executor=lambda query: my_search(query),
    readonly=True,
))

# 使用 ToolRegistry 创建 agent
async with LLMClient(model="gpt-4", base_url="...") as client:
    agent = AgentClient(client=client, tool_registry=registry)
    result = await agent.run("帮我查一下认证相关的文档")
```

### API

| 方法 | 说明 |
|------|------|
| `ToolRegistry.from_global(names)` | 从内置工具加载 |
| `register(tool_def)` | 注册工具 |
| `unregister(name)` | 移除工具 |
| `merge(other)` | 合并另一个 registry |
| `get_tool_defs()` | 输出 OpenAI 格式定义 |
| `execute(name, args_json)` | 同步执行 |
| `execute_async(name, args_json)` | 异步执行 |
| `is_readonly(name)` | 检查是否只读 |
| `names` | 工具名称列表 |
| `readonly_tools` | 只读工具名称列表 |

### 向后兼容

旧的 `tools` + `tool_executor` 参数仍然可用：

```python
# 旧方式（仍然支持）
agent = AgentClient(client=client, tools=tool_defs, tool_executor=my_executor)

# 新方式（推荐）
agent = AgentClient(client=client, tool_registry=registry)
```

---

## 4. MCP Client — 连接外部工具生态

连接任意 MCP server，让 agent 能调用 GitHub、数据库、浏览器等外部工具。

### 配置文件方式（推荐）

在全局或项目配置中声明 MCP servers，agent 启动时自动连接，无需每次手动输入 `--mcp`。

**全局配置** `~/.flexllm/config.yaml`：

```yaml
agent:
  mcp_servers:
    # command + args 分开（Claude Code 风格，推荐）
    github:
      command: npx
      args: ["-y", "@mcp/server-github"]
      env:
        GITHUB_TOKEN: "ghp_xxxx"
    # command 完整字符串（简写）
    filesystem:
      command: "npx @modelcontextprotocol/server-filesystem /home/user"
    # HTTP/SSE 模式
    remote:
      url: "http://localhost:8080/sse"
```

**项目配置** `.flexllm/settings.yaml`（从 cwd 向上搜索）：

```yaml
mcp_servers:
  project-db:
    command: npx
    args: ["@mcp/server-postgres"]
    env:
      DATABASE_URL: "postgres://localhost/mydb"
```

合并规则：项目级覆盖全局同名 server，CLI `--mcp` 追加在最后。

### CLI

```bash
# 使用配置文件中的 MCP servers（自动连接）
flexllm agent --tools code "查看最近的 PR"

# 临时添加 MCP server（与配置文件合并）
flexllm agent --mcp "npx @mcp/server-github" --tools code "查看最近的 PR"

# 连接多个 MCP server
flexllm agent --mcp "npx @mcp/server-github" --mcp "npx @mcp/server-postgres" "分析数据"

# 交互式对话 + MCP
flexllm chat --tools code --mcp "npx @mcp/server-github"
```

### Python API

```python
from flexllm.agent.mcp import MCPConnection, mcp_tools_to_registry

# stdio 模式
async with MCPConnection(command="npx @mcp/server-github") as conn:
    registry = await mcp_tools_to_registry([conn])
    agent = AgentClient(client=llm, tool_registry=registry)
    result = await agent.run("列出最近的 issue")

# SSE 模式
async with MCPConnection(url="http://localhost:8080/sse") as conn:
    tools = await conn.list_tools()
    result = await conn.call_tool("search", {"query": "test"})

# 多个 server 合并
async with MCPConnection(command="npx @mcp/server-github") as github, \
         MCPConnection(command="npx @mcp/server-postgres") as pg:
    registry = await mcp_tools_to_registry([github, pg])
    # 工具名冲突时自动加 "{server_name}." 前缀
```

### MCPConnection 参数

| 参数 | 说明 |
|------|------|
| `command` | stdio 模式，启动子进程命令 |
| `url` | SSE 模式，远程 server URL |
| `env` | 环境变量（stdio 模式） |
| `name` | 连接名称（用于日志和工具前缀，自动推断） |

---

## 5. Agent HTTP 服务

通过 HTTP API 暴露 agent 能力，让其他系统/agent 直接调用。

### 启动

```bash
flexllm serve -m gpt-4 --tools code -s "你是代码助手" -p 8000
```

### 端点

启用 `--tools` 后新增 3 个端点：

#### POST /api/agent/run — 单次执行

```bash
curl -X POST http://localhost:8000/api/agent/run \
  -H "Content-Type: application/json" \
  -d '{"content": "读取 main.py 并分析结构"}'
```

响应：
```json
{
  "content": "main.py 包含...",
  "rounds": 2,
  "tool_calls": [
    {"name": "read", "arguments": "{\"file_path\": \"main.py\"}", "result": "..."}
  ],
  "usage": {"prompt_tokens": 1000, "completion_tokens": 500}
}
```

#### POST /api/agent/run/stream — 流式执行 (SSE)

```bash
curl -X POST http://localhost:8000/api/agent/run/stream \
  -H "Content-Type: application/json" \
  -d '{"content": "分析这个项目"}'
```

SSE 事件流：
```
data: {"type": "tool_call", "name": "read", "arguments": "..."}
data: {"type": "tool_result", "name": "read", "result": "..."}
data: {"type": "done", "content": "分析完成...", "rounds": 2, ...}
```

#### POST /api/agent/chat — 多轮对话

```bash
# 第一轮
curl -X POST http://localhost:8000/api/agent/chat \
  -d '{"content": "你好", "session_id": "abc-123"}'

# 第二轮（自动保持上下文）
curl -X POST http://localhost:8000/api/agent/chat \
  -d '{"content": "帮我看看 main.py", "session_id": "abc-123"}'
```

### 请求参数

| 参数 | 说明 | 必填 |
|------|------|------|
| `content` | 任务/消息内容 | 是 |
| `system` | 系统提示词（覆盖服务端默认值） | 否 |
| `max_rounds` | 最大 tool-use 轮数 | 否 |
| `session_id` | 会话 ID（仅 /chat 端点） | /chat 必填 |

---

## 6. 持久化记忆

Agent 跨 session 保留对话历史和知识事实。基于 flaxkv2 存储。

### 基本用法

```python
from flexllm.agent.memory import MemoryStore

store = MemoryStore("~/.flexllm/agent_memory")

# 配合 AgentClient 自动管理
agent = AgentClient(
    client=llm,
    tool_registry=registry,
    memory=store,
    session_id="project-review",
)
await agent.chat("帮我看看 main.py")   # 自动保存对话历史
await agent.chat("再看看 utils.py")    # 自动加载上轮对话
```

### 独立使用

```python
store = MemoryStore("/path/to/db")

# 对话历史
store.save("session-1", [{"role": "user", "content": "你好"}])
history = store.load("session-1")
sessions = store.list_sessions()

# 事实存储
store.save_fact("user_preference", "喜欢简洁的代码风格")
store.save_fact("project_lang", "Python")
facts = store.get_facts()  # 返回所有事实
value = store.get_fact("user_preference")
```

---

## 7. 结构化可观测性 (Tracing)

追踪 agent 执行过程中每一步的耗时和 token 消耗。

### 使用

```python
from flexllm.agent.tracing import ConsoleExporter, JsonFileExporter, CallbackExporter

# 控制台输出
agent = AgentClient(client=llm, tool_registry=registry, trace_exporter=ConsoleExporter())

# 导出到 JSON 文件
agent = AgentClient(client=llm, tool_registry=registry, trace_exporter=JsonFileExporter("traces/"))

# 自定义回调
agent = AgentClient(
    client=llm,
    trace_exporter=CallbackExporter(lambda trace: print(f"总 token: {trace.total_tokens}"))
)
```

### Trace 结构

```
[Trace abc123] agent.run  2.3s
  +-- llm.call  1.2s  in:500 out:200
  +-- tool.read  0.1s  success=True
  +-- tool.grep  0.05s  success=True
  +-- llm.call  0.9s  in:800 out:150
Total: 2 rounds, 1650 tokens
```

### 导出器

| 导出器 | 说明 |
|--------|------|
| `ConsoleExporter` | 打印到控制台 |
| `JsonFileExporter(dir)` | 每次 trace 写一个 JSON 文件 |
| `CallbackExporter(fn)` | 自定义回调函数 |

---

## 8. Human-in-the-loop — 操作审批

危险操作（写文件、执行命令）前要求确认。

### CLI

```bash
# 手动审批模式（只读操作自动通过，写操作终端询问）
flexllm agent --tools code --approve manual "重构这个模块"
```

### Python API

```python
from flexllm.agent import console_approval, auto_approve

# 内置：终端询问（只读操作自动通过）
agent = AgentClient(client=llm, tool_registry=registry, approval_handler=console_approval)

# 内置：全部自动通过（无人值守）
agent = AgentClient(client=llm, tool_registry=registry, approval_handler=auto_approve)

# 自定义审批逻辑
def my_approval(name: str, args: str, readonly: bool) -> bool:
    if name == "bash":
        return False  # 禁止 bash
    if readonly:
        return True   # 只读操作自动通过
    return input(f"允许 {name}? (y/n) ") == "y"

agent = AgentClient(client=llm, tool_registry=registry, approval_handler=my_approval)
```

### approval_handler 签名

```python
def handler(name: str, args: str, readonly: bool) -> bool:
    """
    Args:
        name: 工具名称
        args: 参数 JSON 字符串
        readonly: 工具是否只读（ToolDef.readonly）
    Returns:
        True 允许执行，False 拒绝
    """
```

---

## AgentClient 完整参数

```python
agent = AgentClient(
    client=llm,                     # LLMClient 实例（必填）
    system="你是助手",               # 系统提示词
    tool_registry=registry,         # ToolRegistry（推荐）
    tools=tool_defs,                # OpenAI 格式工具定义（旧接口）
    tool_executor=my_fn,            # 工具执行函数（旧接口）
    max_rounds=10,                  # 最大 tool-use 轮数
    max_context_tokens=8000,        # 上下文窗口限制（智能压缩）
    max_tool_result_chars=10000,    # 单个工具输出最大字符数
    approval_handler=console_approval,  # 操作审批
    trace_exporter=ConsoleExporter(),   # 可观测性
    memory=store,                   # 持久化记忆
    session_id="my-session",        # 会话 ID
    enable_subagent=True,           # 启用子代理（task 工具）
)

# 流式输出（通过 on_llm_token 回调实时输出）
agent.on_llm_token = lambda token: print(token, end="", flush=True)
result = await agent.run("分析代码", stream=True)
```

## 配置文件结构

```
~/.flexllm/
├── config.yaml                # 全局配置（模型、agent MCP servers）
└── skills/                    # Skills 目录
    ├── code-review/
    │   └── SKILL.md           # 目录模式 skill
    └── translate.md           # 扁平模式 skill

项目根目录/
├── .flexllm.md                # 项目指令（自动注入 system prompt）
└── .flexllm/
    └── settings.yaml          # 项目级配置（MCP servers 等）
```

### MCP servers 配置格式

支持 Claude Code 风格的 `command` + `args` 分开格式：

```yaml
# ~/.flexllm/config.yaml
agent:
  mcp_servers:
    github:
      command: npx
      args: ["-y", "@mcp/server-github"]
      env:
        GITHUB_TOKEN: "ghp_xxxx"
```

项目级 `.flexllm/settings.yaml` 中的同名 server 会覆盖全局配置。
