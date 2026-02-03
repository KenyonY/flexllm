"""flexllm.agent.tools - Agent 内置工具集

提供细粒度的文件操作和搜索工具：
- read: 读取文件（带行号、分页）
- write: 创建或覆盖文件
- edit: 精确字符串替换
- glob: 按模式匹配文件路径
- grep: 在文件中搜索内容
- bash: 执行 shell 命令
"""

from .base import TOOL_REGISTRY, ToolDef, get_tool_defs, make_tool_executor, register_tool
from .file_tools import edit_file, read_file, write_file
from .search_tools import glob_files, grep_content
from .shell_tool import bash_exec

__all__ = [
    # 注册器
    "TOOL_REGISTRY",
    "ToolDef",
    "register_tool",
    "get_tool_defs",
    "make_tool_executor",
    # 工具函数
    "read_file",
    "write_file",
    "edit_file",
    "glob_files",
    "grep_content",
    "bash_exec",
]
