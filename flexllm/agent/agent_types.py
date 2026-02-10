"""Agent 类型注册表 — 定义不同角色的工具集和系统提示"""

AGENT_TYPES: dict[str, dict] = {
    "explore": {
        "description": "只读探索代理，搜索和分析代码",
        "tools": ["read", "glob", "grep", "bash"],
        "prompt": "你是探索代理。搜索和分析，不要修改文件。返回简明摘要。",
    },
    "code": {
        "description": "完整实现代理，可读写所有文件",
        "tools": "*",
        "prompt": "你是编码代理。高效实现请求的变更。",
    },
    "plan": {
        "description": "规划代理，分析并输出实现方案",
        "tools": ["read", "glob", "grep", "bash"],
        "prompt": "你是规划代理。分析代码库并输出编号方案。不要修改文件。",
    },
}


def register_agent_type(name: str, description: str, tools: list[str] | str, prompt: str):
    """注册自定义 Agent 类型"""
    AGENT_TYPES[name] = {"description": description, "tools": tools, "prompt": prompt}


def get_agent_type(name: str) -> dict:
    """获取 Agent 类型配置，不存在时抛异常"""
    if name not in AGENT_TYPES:
        available = ", ".join(AGENT_TYPES.keys())
        raise ValueError(f"未知 agent 类型: {name}，可用: {available}")
    return AGENT_TYPES[name]
