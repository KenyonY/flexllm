"""Agent CLI 输出管理

三种显示模式：
- verbose: 完整展示每步输入/输出，方便调试优化（-v）
- compact: 每步简要输出（TTY 默认）
- quiet: 静默（管道环境）

中间过程输出到 stderr，最终 result.content 输出到 stdout，支持管道。
"""

from __future__ import annotations

import json
import sys

from rich.console import Console
from rich.rule import Rule


def _first_arg_preview(arguments: str, max_len: int = 80) -> str:
    """提取第一个参数值作为简要预览"""
    try:
        args = json.loads(arguments) if arguments else {}
        if not args:
            return ""
        val = str(next(iter(args.values())))
        if len(val) > max_len:
            val = val[:max_len] + "..."
        return val
    except (json.JSONDecodeError, StopIteration):
        return ""


class AgentConsole:
    """Agent 输出管理器

    compact: 每步简要输出 — 思考文本 + 工具名 + 结果预览
    verbose: 每步完整输入/输出，帮助理解 agent 行为并优化
    quiet: 静默，仅最终结果到 stdout
    """

    def __init__(self, verbose: bool = False):
        self._console = Console(stderr=True, highlight=False)
        self._stdout = Console()
        self._verbose = verbose
        self._round = 0

    @property
    def mode(self) -> str:
        if self._verbose:
            return "verbose"
        if sys.stderr.isatty():
            return "compact"
        return "quiet"

    # ── 生命周期 ──

    def begin(self, label: str = "Agent 运行中..."):
        self._round = 0

    def end(self):
        pass

    # ── 回调方法 ──

    def on_tool_call(self, name: str, arguments: str):
        mode = self.mode
        if mode == "verbose":
            self._console.print(Rule(f" {name} ", style="cyan"))
            self._console.print("  [bold]Input:[/bold]")
            try:
                args = json.loads(arguments) if arguments else {}
                for k, v in args.items():
                    v_str = str(v)
                    if len(v_str) > 2000:
                        v_str = v_str[:2000] + f"... ({len(str(v))} chars)"
                    self._console.print(f"    {k}: {v_str}", style="dim")
            except json.JSONDecodeError:
                self._console.print(f"    {arguments[:2000]}", style="dim")
        elif mode == "compact":
            preview = _first_arg_preview(arguments)
            if preview:
                self._console.print(f"  [dim]▸ {name}:[/] {preview}")
            else:
                self._console.print(f"  [dim]▸ {name}[/]")

    def on_tool_result(self, name: str, result: str):
        mode = self.mode
        if mode == "verbose":
            result_str = str(result)
            total = len(result_str)
            self._console.print("  [bold]Output:[/bold]", style="green")
            if total > 3000:
                self._console.print(f"    {result_str[:3000]}", style="green")
                self._console.print(f"    ... ({total} chars total)", style="dim green")
            else:
                for line in result_str.split("\n")[:50]:
                    self._console.print(f"    {line}", style="green")
                if result_str.count("\n") > 50:
                    self._console.print(
                        f"    ... ({result_str.count(chr(10))} lines total)", style="dim green"
                    )
        elif mode == "compact":
            result_str = str(result)
            preview = result_str[:200] + "..." if len(result_str) > 200 else result_str
            preview = preview.replace("\n", " ").strip()
            if preview:
                self._console.print(f"    [dim green]{preview}[/]")

    def on_llm_token(self, token: str):
        """流式 token 实时输出"""
        mode = self.mode
        if mode == "quiet":
            return
        self._console.print(token, end="", highlight=False)
        self._streaming = True

    def on_llm_response(self, response):
        mode = self.mode
        self._round += 1
        # 流式模式下，token 已经输出过了，换行结束
        if getattr(self, "_streaming", False):
            self._console.print()
            self._streaming = False
            return

        if mode == "verbose":
            self._console.print(Rule(f" Round {self._round} ", style="bold blue"))

            if response.content:
                self._console.print("  [bold]Content:[/bold]")
                self._console.print(f"    {response.content}")

            if response.tool_calls:
                n = len(response.tool_calls)
                self._console.print(f"  [bold]Plan:[/bold] {n} tool call(s)")
                for i, tc in enumerate(response.tool_calls, 1):
                    fn = tc.function
                    name = fn.get("name", "?")
                    args = fn.get("arguments", "{}")
                    try:
                        parsed = json.loads(args) if args else {}
                        brief = ", ".join(f"{k}={repr(v)[:60]}" for k, v in parsed.items())
                    except json.JSONDecodeError:
                        brief = args[:60] + "..."
                    self._console.print(f"    [{i}] {name}({brief})", style="dim")

            if response.usage:
                u = response.usage
                pin = u.get("prompt_tokens", "?")
                pout = u.get("completion_tokens", "?")
                self._console.print(f"  [bold]Tokens:[/bold] in={pin} out={pout}", style="dim")

        elif mode == "compact":
            # 展示 agent 的思考文本
            if response.content and response.tool_calls:
                self._console.print(f"\n{response.content}\n", style="dim")

    def on_validation(self, files: list[str], results: list):
        mode = self.mode
        if mode == "verbose":
            self._console.print(Rule(" Validation ", style="yellow"))
            self._console.print(f"  Files: {files}", style="dim")
            for r in results:
                if r.success:
                    self._console.print(f"  [{r.validator_name}] OK", style="green")
                else:
                    self._console.print(
                        f"  [{r.validator_name}] {len(r.errors)} error(s)", style="red"
                    )
                    for e in r.errors[:5]:
                        self._console.print(f"    {e}", style="red")
                    if len(r.errors) > 5:
                        self._console.print(
                            f"    ... and {len(r.errors) - 5} more", style="dim red"
                        )
        elif mode == "compact":
            ok = all(r.success for r in results)
            mark = "[green]✓[/]" if ok else "[red]✗[/]"
            self._console.print(f"  [dim]▸ validation {mark}[/]")

    # ── 输出方法 ──

    def print_result(self, result):
        """result.content → stdout"""
        if result.content:
            self._stdout.print(result.content, highlight=False)

    def print_summary(self, result):
        """执行统计"""
        if self.mode == "verbose":
            self._console.print(Rule(" Done ", style="bold green"))
            parts = [f"Rounds: {result.rounds}", f"Tools: {len(result.tool_calls)}"]
            if result.usage:
                total = result.usage.get("total_tokens")
                if total:
                    parts.append(f"Tokens: {total}")
            self._console.print("  " + "  |  ".join(parts), style="dim")
        elif self.mode == "compact" and result.tool_calls:
            self._console.print(
                f"  [dim]({len(result.tool_calls)} calls, {result.rounds} rounds)[/]"
            )

    def print_header(self, model, tools_name, mcp_servers=None, validators=None, message=None):
        """verbose: 启动信息"""
        if self.mode != "verbose":
            return

        self._console.print(Rule(" Agent ", style="bold green"))
        self._console.print(f"  Model: {model}")
        self._console.print(f"  Tools: {tools_name}")
        if mcp_servers:
            self._console.print(f"  MCP: {', '.join(mcp_servers)}")
        if validators:
            self._console.print(f"  Validators: {[v.name for v in validators]}")
        if message:
            self._console.print(f"  Task: {message}", style="dim")

    def print_chat_header(self, model, tools_name, mcp_servers=None, verbose=False):
        """交互模式 header"""
        self._console.print()
        self._console.print(Rule("Agent Chat", style="bold"))
        self._console.print(f"  Model: {model}")
        self._console.print(f"  Tools: {tools_name}")
        if mcp_servers:
            self._console.print(f"  MCP: {', '.join(mcp_servers)}")
        if verbose:
            self._console.print("  Mode: verbose")
        self._console.print("  输入 'quit' 或 Ctrl+C 退出")
