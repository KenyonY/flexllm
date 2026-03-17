# Agent 模块架构文档

面向开发者的 AgentClient 内部架构说明。用户文档见 [agent.md](agent.md)。

## 架构分层

```
┌─────────────────────────────────────────────┐
│                  CLI Layer                   │
│  commands.py → chat_helpers.py               │
│  (agent/chat 命令, 参数解析, AgentConsole UI) │
├─────────────────────────────────────────────┤
│              AgentClient Layer               │
│  client.py                                   │
│  (tool-use 循环, 流式/非流式, 验证闭环)       │
├──────────┬──────────┬───────────┬────────────┤
│ Tools    │ MCP      │ Memory    │ Tracing    │
│ Registry │ Client   │ Store     │ Exporter   │
├──────────┼──────────┼───────────┼────────────┤
│ TaskMgr  │ TodoTrack│ Subagent  │ Validators │
│ (持久化) │ (内存级) │ (子代理)  │ (代码验证) │
├──────────┴──────────┴───────────┴────────────┤
│              LLMClient Layer                  │
│  OpenAIClient / GeminiClient / ClaudeClient  │
└─────────────────────────────────────────────┘
```

## 核心循环流程

```
AgentClient.run(user_input)
    │
    ▼
_build_messages(system + history + user_input)
    │
    ▼
┌─► LLM 调用 (chat_completions / stream)
│       │
│       ▼
│   有 tool_calls?
│       │ 否 ──► 返回 AgentResult
│       │ 是
│       ▼
│   构建 assistant_msg (含 tool_calls)
│       │
│       ▼
│   并行执行工具 (_execute_tool)
│   ├── 审批检查 (approval_handler)
│   ├── ToolRegistry.execute_async()
│   └── 截断过大输出
│       │
│       ▼
│   追加 tool result messages
│       │
│       ▼
│   Todo 提醒注入 (如启用)
│       │
│       ▼
│   上下文压缩 (如配置 max_context_tokens)
│       │
│       ▼
│   rounds++ < max_rounds?
│       │ 是
└───────┘
```

## 模块职责

| 文件 | 职责 |
|------|------|
| `client.py` | AgentClient 主类，tool-use 循环、流式输出、验证闭环、子代理 |
| `types.py` | AgentResult、ToolCallRecord 数据类 |
| `agent_types.py` | 预定义 agent 类型配置（explore/code/plan） |
| `task_manager.py` | 文件持久化的任务管理，创建/更新/依赖/删除 |
| `todo_tracker.py` | 内存级 todo 追踪，定时纠缠提醒 |
| `tools/base.py` | ToolDef、ToolRegistry、@register_tool 装饰器、全局 TOOL_REGISTRY |
| `tools/file_tools.py` | read/write/edit 文件操作工具 |
| `tools/search_tools.py` | glob/grep 搜索工具 |
| `tools/shell_tool.py` | bash 命令执行工具 |
| `tools/task_tool.py` | 子代理 task 工具的 schema 定义 |
| `tools/task_tools.py` | 任务管理工具注册（task_create/update/list/get + todo） |
| `mcp/` | MCP 协议客户端，连接外部 MCP Server |
| `memory.py` | 持久化记忆存储 |
| `tracing.py` | Trace/Span 可观测性导出 |
| `validators/` | 代码验证器（语法/lint/类型/测试） |

## 工具系统设计

```
全局 TOOL_REGISTRY (dict)         ToolRegistry (实例级)
  │                                   │
  │ @register_tool 装饰器注册         │ .from_global() 加载内置工具
  │ ├── read                          │ ├── 内置工具子集
  │ ├── write                         │ ├── MCP 工具 (动态)
  │ ├── edit                          │ ├── task 工具 (子代理)
  │ ├── glob                          │ ├── task_create/update/list/get
  │ ├── grep                          │ └── todo 工具
  │ └── bash                          │
  │                                   │ .get_tool_defs() → OpenAI 格式
  │                                   │ .execute_async() → 执行
```

**工具注入方式：**
- 内置工具：`@register_tool` → 全局 `TOOL_REGISTRY` → `ToolRegistry.from_global()`
- MCP 工具：运行时连接 MCP Server → `mcp_tools_to_registry()` → `registry.merge()`
- 子代理工具：`AgentClient._inject_task_tool()` → 闭包绑定父实例
- 任务工具：`register_task_tools(registry, tm)` → 闭包绑定 TaskManager
- Todo 工具：`register_todo_tool(registry, tracker)` → 闭包绑定 TodoTracker

## 扩展点

| 扩展点 | 方式 | 示例 |
|--------|------|------|
| 自定义工具 | `ToolRegistry.register(ToolDef(...))` | 注册业务工具 |
| MCP 集成 | `--mcp` 参数或 config.yaml | 连接外部服务 |
| 验证器 | 继承 `Validator` 基类 | 自定义代码检查 |
| 子代理 | `register_agent_type()` | 自定义 agent 类型 |
| 任务系统 | `enable_tasks=True` | 持久化任务追踪 |
| Todo 追踪 | `enable_todo=True` | 进度纠缠提醒 |
| 记忆 | `MemoryStore` | 跨会话记忆 |
| 可观测性 | `TraceExporter` | 追踪导出 |
| 审批 | `approval_handler` | 自定义审批逻辑 |

## TaskManager 数据流

```
AgentClient(enable_tasks=True)
    │
    ▼
_inject_task_tools(tasks_dir)
    │
    ├── TaskManager(tasks_dir) ──► .tasks/task_{id}.json
    │
    └── register_task_tools(registry, tm)
            │
            ├── task_create  → tm.create()
            ├── task_update  → tm.update()  ──► 自动清理依赖
            ├── task_list    → tm.list_all()
            └── task_get     → tm.get()
```

## TodoTracker 数据流

```
AgentClient(enable_todo=True)
    │
    ├── TodoTracker(nag_interval=3) ──► 内存 items[]
    │
    └── _run_loop 中每轮:
            │
            ├── LLM 调用了 todo 工具? → notify_used() 重置计数
            │
            └── tick() → 超过 nag_interval 未更新?
                    │ 是
                    └── 注入提醒 message 到 messages[]
```
