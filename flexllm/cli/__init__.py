"""flexllm CLI 模块"""

from __future__ import annotations

import sys

try:
    from typing import Annotated

    from typer import Option, Typer

    app = Typer(
        name="flexllm",
        help="""flexllm - 高性能 LLM 客户端命令行工具

\b
Quick Start:
  flexllm ask "什么是 Python"             # 快速问答
  flexllm ask -f code.py "解释这段代码"   # 附加文件
  flexllm chat                            # 交互式对话
  flexllm batch in.jsonl -o out.jsonl     # 批量处理（支持断点续传）
  flexllm list                            # 查看本地配置的模型
  flexllm test                            # 测试 LLM 连接

配置: ~/.flexllm/config.yaml (运行 flexllm init 创建)
环境变量: FLEXLLM_BASE_URL, FLEXLLM_API_KEY, FLEXLLM_MODEL

\b
Exit Codes (Agent-friendly, cross-version stable):
  0   成功
  1   通用错误
  2   参数/用法错误（非法值、缺少必选）
  3   资源未找到（模型/配置/文件）
  4   认证失败（API Key 无效、额度不足）
  5   冲突（资源已存在）
  6   网络错误（常为 retryable）
  7   依赖缺失（缺 pip 包）
  8   文件 IO 错误
  10  Dry-run 成功（非实际执行）

\b
Agent-friendly JSON 输出:
  核心命令支持 --format json（ask/chat/batch），stdout 为结构化数据，
  stderr 为进度/日志；错误在非 TTY 自动以 JSON 输出到 stderr。""",
        add_completion=True,
        no_args_is_help=True,
    )

    def _version_callback(value: bool):
        if value:
            try:
                from flexllm import __version__

                v = __version__
            except Exception:
                v = "unknown"
            print(f"flexllm {v}")
            raise SystemExit(0)

    @app.callback()
    def _main_callback(
        version: Annotated[
            bool | None,
            Option("--version", "-V", help="显示版本号", callback=_version_callback, is_eager=True),
        ] = None,
    ):
        pass

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
