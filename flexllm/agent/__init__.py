"""flexllm.agent - Agent 客户端"""

from .client import AgentClient
from .types import AgentResult, ToolCallRecord

__all__ = ["AgentClient", "AgentResult", "ToolCallRecord"]


# 延迟导入 tools 模块，避免循环依赖
def __getattr__(name):
    if name == "tools":
        from . import tools

        return tools
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
