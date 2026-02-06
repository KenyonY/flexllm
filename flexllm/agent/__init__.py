"""flexllm.agent - Agent 客户端"""

from .client import AgentClient, auto_approve, console_approval
from .types import AgentResult, ToolCallRecord

__all__ = ["AgentClient", "AgentResult", "ToolCallRecord", "auto_approve", "console_approval"]


# 延迟导入 tools 和 validators 模块，避免循环依赖
def __getattr__(name):
    import importlib
    import sys

    if name == "tools":
        mod = importlib.import_module(".tools", __name__)
        sys.modules[f"{__name__}.tools"] = mod
        globals()["tools"] = mod
        return mod
    if name == "validators":
        mod = importlib.import_module(".validators", __name__)
        sys.modules[f"{__name__}.validators"] = mod
        globals()["validators"] = mod
        return mod
    if name == "memory":
        mod = importlib.import_module(".memory", __name__)
        sys.modules[f"{__name__}.memory"] = mod
        globals()["memory"] = mod
        return mod
    if name == "MemoryStore":
        from .memory import MemoryStore

        globals()["MemoryStore"] = MemoryStore
        return MemoryStore
    if name == "mcp":
        mod = importlib.import_module(".mcp", __name__)
        sys.modules[f"{__name__}.mcp"] = mod
        globals()["mcp"] = mod
        return mod
    if name == "tracing":
        mod = importlib.import_module(".tracing", __name__)
        sys.modules[f"{__name__}.tracing"] = mod
        globals()["tracing"] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
