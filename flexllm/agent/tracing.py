"""Agent 执行追踪（可观测性）

追踪 agent 执行过程中每个步骤的耗时、token 消耗、错误。

Example:
    from flexllm.agent.tracing import ConsoleExporter

    agent = AgentClient(
        client=llm,
        trace_exporter=ConsoleExporter(),
    )
    result = await agent.run("分析代码")
    # 自动输出 trace 信息
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Span:
    """执行过程中的一个步骤"""

    span_id: str
    name: str  # "agent.run", "llm.call", "tool.read", "tool.bash"
    start_time: float
    end_time: float = 0
    attributes: dict = field(default_factory=dict)
    status: str = "ok"  # "ok" / "error"
    error: str | None = None
    children: list[Span] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """耗时（毫秒）"""
        if self.end_time <= 0:
            return 0
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict:
        """序列化为字典"""
        d = {
            "span_id": self.span_id,
            "name": self.name,
            "duration_ms": round(self.duration_ms, 1),
            "status": self.status,
        }
        if self.attributes:
            d["attributes"] = self.attributes
        if self.error:
            d["error"] = self.error
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


@dataclass
class Trace:
    """一次完整的 agent 执行追踪"""

    trace_id: str
    root_span: Span
    total_tokens: int = 0
    total_cost: float = 0
    total_rounds: int = 0

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "total_rounds": self.total_rounds,
            "root": self.root_span.to_dict(),
        }


def _new_span_id() -> str:
    return uuid.uuid4().hex[:8]


class SpanContext:
    """Span 上下文管理器，自动计时"""

    def __init__(self, name: str, parent: Span | None = None, **attributes):
        self.span = Span(
            span_id=_new_span_id(),
            name=name,
            start_time=time.time(),
            attributes=dict(attributes),
        )
        self._parent = parent
        if parent is not None:
            parent.children.append(self.span)

    def __enter__(self) -> Span:
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.span.end_time = time.time()
        if exc_type is not None:
            self.span.status = "error"
            self.span.error = str(exc_val)
        return False  # 不吞异常


# ========== Exporters ==========


class TraceExporter:
    """导出器基类"""

    def export(self, trace: Trace):
        raise NotImplementedError


class ConsoleExporter(TraceExporter):
    """控制台输出，树形结构展示"""

    def export(self, trace: Trace):
        print(f"\n{'─' * 60}")
        print(f"Trace: {trace.trace_id}")
        print(f"Rounds: {trace.total_rounds} | Tokens: {trace.total_tokens}")
        print(f"{'─' * 60}")
        self._print_span(trace.root_span, indent=0)
        print(f"{'─' * 60}")

    def _print_span(self, span: Span, indent: int):
        prefix = "  " * indent
        status_icon = "✓" if span.status == "ok" else "✗"
        duration = f"{span.duration_ms:.0f}ms" if span.duration_ms > 0 else ""
        print(f"{prefix}{status_icon} {span.name} {duration}")

        # 显示关键属性
        for key in ("model", "input_tokens", "output_tokens", "file_path", "command"):
            if key in span.attributes:
                val = str(span.attributes[key])
                if len(val) > 80:
                    val = val[:80] + "..."
                print(f"{prefix}  {key}: {val}")

        if span.error:
            print(f"{prefix}  error: {span.error}")

        for child in span.children:
            self._print_span(child, indent + 1)


class JsonFileExporter(TraceExporter):
    """JSON 文件输出"""

    def __init__(self, path: str):
        self.path = path

    def export(self, trace: Trace):
        with open(self.path, "a") as f:
            f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")


class CallbackExporter(TraceExporter):
    """自定义回调"""

    def __init__(self, callback: Callable[[Trace], None]):
        self._callback = callback

    def export(self, trace: Trace):
        self._callback(trace)
