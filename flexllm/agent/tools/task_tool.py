"""Task 工具 — 生成子代理的 schema 定义

只定义 schema，不注册到全局 TOOL_REGISTRY。
task 工具的 executor 需要绑定父 AgentClient，在运行时由 AgentClient._inject_task_tool() 动态创建。
"""

TASK_TOOL_SCHEMA = {
    "name": "task",
    "description": (
        "生成子代理处理聚焦的子任务。子代理在独立上下文中运行，完成后返回摘要。\n\n"
        "Agent 类型:\n"
        "- explore: 只读探索，搜索和分析代码\n"
        "- code: 完整实现，可读写所有文件\n"
        "- plan: 规划分析，输出实现方案"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "短任务名（3-5 词），用于进度显示",
            },
            "prompt": {
                "type": "string",
                "description": "给子代理的详细指令",
            },
            "agent_type": {
                "type": "string",
                "enum": ["explore", "code", "plan"],
                "description": "代理类型",
            },
        },
        "required": ["description", "prompt", "agent_type"],
    },
}
