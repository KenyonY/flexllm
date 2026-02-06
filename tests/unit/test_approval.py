"""Test Human-in-the-loop approval mechanism"""

from unittest.mock import AsyncMock

import pytest

from flexllm import AgentClient
from flexllm.agent.client import auto_approve, console_approval
from flexllm.clients.base import ChatCompletionResult, ToolCall


class TestApprovalHandler:
    """审批机制测试"""

    def test_auto_approve(self):
        assert auto_approve("bash", '{"command": "rm -rf /"}', False) is True
        assert auto_approve("read", '{"file_path": "a.py"}', True) is True

    def test_console_approval_readonly(self):
        # readonly 操作自动通过
        assert console_approval("read", '{"file_path": "a.py"}', True) is True

    @pytest.mark.asyncio
    async def test_approval_denied(self):
        """handler 拒绝时返回 denied 消息"""
        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            side_effect=[
                ChatCompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            type="function",
                            function={"name": "bash", "arguments": '{"command": "rm -rf /"}'},
                        ),
                    ],
                ),
                ChatCompletionResult(content="ok", tool_calls=None),
            ]
        )

        def deny_all(name, args, readonly):
            return False

        agent = AgentClient(
            client=mock_client,
            tools=[{"type": "function", "function": {"name": "bash"}}],
            tool_executor=lambda n, a: "executed",
            approval_handler=deny_all,
        )
        result = await agent.run("test")
        assert "denied" in result.tool_calls[0].result.lower()

    @pytest.mark.asyncio
    async def test_no_handler_default(self):
        """无 handler 时直接执行"""
        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            side_effect=[
                ChatCompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            type="function",
                            function={"name": "tool", "arguments": "{}"},
                        ),
                    ],
                ),
                ChatCompletionResult(content="done", tool_calls=None),
            ]
        )

        agent = AgentClient(
            client=mock_client,
            tools=[{"type": "function", "function": {"name": "tool"}}],
            tool_executor=lambda n, a: "result",
        )
        result = await agent.run("test")
        assert result.tool_calls[0].result == "result"

    @pytest.mark.asyncio
    async def test_approval_selective(self):
        """选择性审批：readonly 通过，write 拒绝"""
        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            side_effect=[
                ChatCompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            type="function",
                            function={"name": "read", "arguments": '{"file_path": "a.py"}'},
                        ),
                        ToolCall(
                            id="c2",
                            type="function",
                            function={
                                "name": "write",
                                "arguments": '{"file_path": "a.py", "content": "x"}',
                            },
                        ),
                    ],
                ),
                ChatCompletionResult(content="done", tool_calls=None),
            ]
        )

        def selective(name, args, readonly):
            return readonly  # 只通过 readonly 工具

        agent = AgentClient(
            client=mock_client,
            tools=[
                {"type": "function", "function": {"name": "read"}},
                {"type": "function", "function": {"name": "write"}},
            ],
            tool_executor=lambda n, a: f"{n}_result",
            approval_handler=selective,
        )
        result = await agent.run("test")
        # read (readonly) 应该执行成功
        # write (not readonly) 应该被拒绝
        assert len(result.tool_calls) == 2
