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
|  +----------+  +-----------+                                                 |
|  | MCP Client|  | Agent HTTP|                                                |
|  | (消费工具) |  | Serve     |                                               |
|  +-----+----+  +-----+-----+                                                |
|        |              |                                                      |
|        v              v                                                      |
|  +----------------------------------------------------------------+          |
|  |                    AgentClient (核心)                            |          |
|  |  tool-use 循环 | ToolRegistry | 审批机制 | Structured Output    |          |
|  +-----+----------------------------------------------------------+          |
|        |                                                                      |
|  +-----+----------------------------------------------------------+          |
|  | Memory  |  Tracing  |  Validators                               |          |
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

## 1. ToolRegistry — 动态工具注册

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

## 2. MCP Client — 连接外部工具生态

连接任意 MCP server，让 agent 能调用 GitHub、数据库、浏览器等外部工具。

### CLI

```bash
# 连接单个 MCP server
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

## 3. Agent HTTP 服务

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

## 4. 持久化记忆

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

## 5. 结构化可观测性 (Tracing)

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

## 6. Human-in-the-loop — 操作审批

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
    max_context_tokens=8000,        # 上下文窗口限制
    approval_handler=console_approval,  # 操作审批
    trace_exporter=ConsoleExporter(),   # 可观测性
    memory=store,                   # 持久化记忆
    session_id="my-session",        # 会话 ID
)
```
