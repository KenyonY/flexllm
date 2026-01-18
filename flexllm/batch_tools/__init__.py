"""
批处理工具模块

- MllmFolderProcessor: 文件夹批量处理
- MllmTableProcessor: 表格数据批量处理
"""

from .folder_processor import MllmFolderProcessor

# MllmTableProcessor 需要 pandas（可选依赖）
try:
    from .table_processor import MllmTableProcessor
except ImportError:
    MllmTableProcessor = None

__all__ = ["MllmFolderProcessor", "MllmTableProcessor"]
