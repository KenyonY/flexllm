"""flexllm CLI 模块"""

from __future__ import annotations

import sys

try:
    from typer import Typer

    app = Typer(
        name="flexllm",
        help="flexllm - 高性能 LLM 客户端命令行工具",
        add_completion=True,
        no_args_is_help=True,
    )
    HAS_TYPER = True
except ImportError:
    HAS_TYPER = False
    app = None


def _fallback_cli():
    """没有 typer 时的简单 CLI"""
    args = sys.argv[1:]

    if not args or args[0] in ["-h", "--help", "help"]:
        print("flexllm CLI")
        print("\n命令:")
        print("  ask <prompt>      快速问答")
        print("  chat              交互对话")
        print("  batch             批量处理 JSONL 文件")
        print("  mock              启动 Mock LLM 服务器")
        print("  models            列出远程模型")
        print("  list              列出配置模型")
        print("  set-model <name>  设置默认模型")
        print("  test              测试连接")
        print("  init              初始化配置")
        print("  agent             Agent 模式")
        print("  version           显示版本")
        print("\n安装 typer 获得更好的 CLI 体验: pip install typer")
        return

    print("错误: 需要安装 typer: pip install typer", file=sys.stderr)
    print("或者: pip install flexllm[cli]", file=sys.stderr)


def main():
    """CLI 入口点"""
    if HAS_TYPER:
        from .commands import register_commands

        register_commands(app)
        app()
    else:
        _fallback_cli()
