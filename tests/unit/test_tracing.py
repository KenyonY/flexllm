"""Test agent tracing - Span/Trace/SpanContext/Exporters 及 AgentClient 集成"""

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from flexllm.agent.tracing import (
    CallbackExporter,
    ConsoleExporter,
    JsonFileExporter,
    Span,
    SpanContext,
    Trace,
    _new_span_id,
)


class TestSpan:
    """Span 基础功能"""

    def test_create_span(self):
        span = Span(span_id="abc", name="test", start_time=100.0)
        assert span.span_id == "abc"
        assert span.name == "test"
        assert span.status == "ok"
        assert span.error is None
        assert span.children == []

    def test_duration_ms(self):
        span = Span(span_id="a", name="t", start_time=1.0, end_time=1.5)
        assert span.duration_ms == 500.0

    def test_duration_ms_no_end(self):
        span = Span(span_id="a", name="t", start_time=1.0)
        assert span.duration_ms == 0

    def test_to_dict_minimal(self):
        span = Span(span_id="a1", name="llm.call", start_time=1.0, end_time=1.1)
        d = span.to_dict()
        assert d["span_id"] == "a1"
        assert d["name"] == "llm.call"
        assert d["status"] == "ok"
        assert d["duration_ms"] == 100.0
        assert "attributes" not in d
        assert "error" not in d
        assert "children" not in d

    def test_to_dict_with_attributes_and_error(self):
        span = Span(
            span_id="a2",
            name="tool.bash",
            start_time=1.0,
            end_time=2.0,
            attributes={"command": "ls"},
            status="error",
            error="permission denied",
        )
        d = span.to_dict()
        assert d["attributes"] == {"command": "ls"}
        assert d["error"] == "permission denied"

    def test_to_dict_with_children(self):
        parent = Span(span_id="p", name="agent.run", start_time=1.0, end_time=3.0)
        child = Span(span_id="c", name="llm.call", start_time=1.0, end_time=2.0)
        parent.children.append(child)
        d = parent.to_dict()
        assert len(d["children"]) == 1
        assert d["children"][0]["name"] == "llm.call"


class TestTrace:
    """Trace 序列化"""

    def test_to_dict(self):
        root = Span(span_id="r", name="agent.run", start_time=1.0, end_time=2.0)
        trace = Trace(
            trace_id="t1", root_span=root, total_tokens=100, total_cost=0.01, total_rounds=2
        )
        d = trace.to_dict()
        assert d["trace_id"] == "t1"
        assert d["total_tokens"] == 100
        assert d["total_cost"] == 0.01
        assert d["total_rounds"] == 2
        assert d["root"]["name"] == "agent.run"


class TestSpanContext:
    """SpanContext 上下文管理器"""

    def test_auto_timing(self):
        with SpanContext("test.op") as span:
            time.sleep(0.01)
        assert span.end_time > span.start_time
        assert span.duration_ms >= 10
        assert span.status == "ok"

    def test_parent_link(self):
        parent = Span(span_id="p", name="root", start_time=1.0)
        with SpanContext("child", parent=parent) as child:
            pass
        assert len(parent.children) == 1
        assert parent.children[0] is child

    def test_error_status(self):
        try:
            with SpanContext("fail") as span:
                raise ValueError("boom")
        except ValueError:
            pass
        assert span.status == "error"
        assert span.error == "boom"
        assert span.end_time > 0

    def test_no_exception_swallow(self):
        with pytest.raises(RuntimeError, match="test"):
            with SpanContext("op"):
                raise RuntimeError("test")


class TestNewSpanId:
    def test_format(self):
        sid = _new_span_id()
        assert len(sid) == 8
        assert sid.isalnum()

    def test_unique(self):
        ids = {_new_span_id() for _ in range(100)}
        assert len(ids) == 100


class TestConsoleExporter:
    def test_output(self, capsys):
        root = Span(span_id="r", name="agent.run", start_time=1.0, end_time=2.0)
        child = Span(
            span_id="c",
            name="llm.call",
            start_time=1.0,
            end_time=1.5,
            attributes={"input_tokens": 10, "output_tokens": 5},
        )
        root.children.append(child)
        trace = Trace(trace_id="t1", root_span=root, total_tokens=15, total_rounds=1)

        ConsoleExporter().export(trace)
        out = capsys.readouterr().out

        assert "t1" in out
        assert "agent.run" in out
        assert "llm.call" in out
        assert "input_tokens: 10" in out
        assert "Rounds: 1" in out

    def test_error_span_output(self, capsys):
        root = Span(
            span_id="r",
            name="agent.run",
            start_time=1.0,
            end_time=2.0,
        )
        err_child = Span(
            span_id="e",
            name="tool.bash",
            start_time=1.0,
            end_time=1.5,
            status="error",
            error="command failed",
        )
        root.children.append(err_child)
        trace = Trace(trace_id="t2", root_span=root)

        ConsoleExporter().export(trace)
        out = capsys.readouterr().out
        assert "✗" in out
        assert "command failed" in out


class TestJsonFileExporter:
    def test_write(self, tmp_path):
        path = str(tmp_path / "traces.jsonl")
        root = Span(span_id="r", name="agent.run", start_time=1.0, end_time=2.0)
        trace = Trace(trace_id="t1", root_span=root, total_tokens=50)

        exporter = JsonFileExporter(path)
        exporter.export(trace)
        exporter.export(trace)  # 追加模式

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        data = json.loads(lines[0])
        assert data["trace_id"] == "t1"
        assert data["total_tokens"] == 50


class TestCallbackExporter:
    def test_callback(self):
        collected = []
        exporter = CallbackExporter(lambda t: collected.append(t))
        root = Span(span_id="r", name="agent.run", start_time=1.0, end_time=2.0)
        trace = Trace(trace_id="t1", root_span=root)

        exporter.export(trace)
        assert len(collected) == 1
        assert collected[0].trace_id == "t1"


class TestAgentClientTracingIntegration:
    """AgentClient 与 tracing 集成测试"""

    @pytest.mark.asyncio
    async def test_no_exporter_no_impact(self):
        """trace_exporter=None 时不影响正常执行"""
        from flexllm import AgentClient
        from flexllm.clients.base import ChatCompletionResult

        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            return_value=ChatCompletionResult(
                content="hello",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                tool_calls=None,
            )
        )

        agent = AgentClient(client=mock_client, trace_exporter=None)
        result = await agent.run("hi")
        assert result.content == "hello"
        assert result.rounds == 0

    @pytest.mark.asyncio
    async def test_exporter_receives_trace(self):
        """有 exporter 时正确生成并导出 trace"""
        from flexllm import AgentClient
        from flexllm.clients.base import ChatCompletionResult, ToolCall

        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            side_effect=[
                ChatCompletionResult(
                    content=None,
                    usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            type="function",
                            function={"name": "read", "arguments": '{"path": "a.py"}'},
                        ),
                    ],
                ),
                ChatCompletionResult(
                    content="done",
                    usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                    tool_calls=None,
                ),
            ]
        )

        collected = []
        exporter = CallbackExporter(lambda t: collected.append(t))

        agent = AgentClient(
            client=mock_client,
            tools=[{"type": "function", "function": {"name": "read"}}],
            tool_executor=lambda n, a: "file content",
            trace_exporter=exporter,
        )
        result = await agent.run("read file")

        assert result.content == "done"
        assert result.rounds == 1
        assert len(collected) == 1

        trace = collected[0]
        assert trace.total_rounds == 1
        assert trace.total_tokens == 45
        assert trace.root_span.name == "agent.run"
        assert trace.root_span.end_time > 0

        # 应有 2 个 llm.call span + 1 个 tool.read span
        children = trace.root_span.children
        assert len(children) == 3
        assert children[0].name == "llm.call"
        assert children[0].attributes["input_tokens"] == 10
        assert children[1].name == "tool.read"
        assert children[1].attributes["success"] is True
        assert children[2].name == "llm.call"
        assert children[2].attributes["input_tokens"] == 20

    @pytest.mark.asyncio
    async def test_exporter_no_tool_calls(self):
        """无工具调用时 trace 只包含一个 llm.call span"""
        from flexllm import AgentClient
        from flexllm.clients.base import ChatCompletionResult

        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            return_value=ChatCompletionResult(
                content="hi",
                usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                tool_calls=None,
            )
        )

        collected = []
        exporter = CallbackExporter(lambda t: collected.append(t))

        agent = AgentClient(client=mock_client, trace_exporter=exporter)
        await agent.run("hello")

        trace = collected[0]
        assert trace.total_rounds == 0
        assert trace.total_tokens == 8
        assert len(trace.root_span.children) == 1
        assert trace.root_span.children[0].name == "llm.call"
