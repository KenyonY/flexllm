"""flexllm.agent - Agent 客户端"""

from .client import AgentClient
from .types import AgentResult, ToolCallRecord

__all__ = ["AgentClient", "AgentResult", "ToolCallRecord"]


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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
