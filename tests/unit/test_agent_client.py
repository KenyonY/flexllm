"""Test AgentClient - tool-use 循环、上下文裁剪、structured output"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from flexllm import AgentClient, AgentResult, ToolCallRecord
from flexllm.clients.base import ChatCompletionResult, ToolCall


class TestAgentClientBasic:
    """基本功能测试"""

    def test_init(self):
        """测试 AgentClient 初始化"""
        client = MagicMock()
        agent = AgentClient(client=client, system="你好")
        assert agent.client is client
        assert agent.system == "你好"
        assert agent.max_rounds == 10
        assert agent._history == []

    def test_reset(self):
        """测试清空对话历史"""
        agent = AgentClient(client=MagicMock())
        agent._history = [{"role": "user", "content": "test"}]
        agent.reset()
        assert agent._history == []


class TestAgentRunNoTools:
    """无工具调用时的 run 测试"""

    @pytest.mark.asyncio
    async def test_run_simple(self):
        """LLM 直接返回内容，不调用工具"""
        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            return_value=ChatCompletionResult(
                content="你好！",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=None,
            )
        )

        agent = AgentClient(client=mock_client, system="你是助手")
        result = await agent.run("你好")

        assert result.content == "你好！"
        assert result.rounds == 0
        assert result.tool_calls == []
        assert result.usage["prompt_tokens"] == 10

    @pytest.mark.asyncio
    async def test_run_passes_kwargs(self):
        """验证 kwargs 被传递给 client.chat_completions"""
        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            return_value=ChatCompletionResult(content="ok", tool_calls=None)
        )

        agent = AgentClient(client=mock_client)
        await agent.run("test", temperature=0.5)

        call_kwargs = mock_client.chat_completions_or_raise.call_args
        assert call_kwargs.kwargs["temperature"] == 0.5


class TestAgentRunWithTools:
    """工具调用循环测试"""

    @pytest.mark.asyncio
    async def test_single_tool_call(self):
        """单次工具调用 -> LLM 返回最终结果"""
        mock_client = AsyncMock()

        # 第一次：LLM 返回 tool_call
        tool_call_result = ChatCompletionResult(
            content=None,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function={"name": "get_weather", "arguments": '{"city": "北京"}'},
                )
            ],
        )
        # 第二次：LLM 返回最终内容
        final_result = ChatCompletionResult(
            content="北京今天晴，25°C",
            usage={"prompt_tokens": 20, "completion_tokens": 10},
            tool_calls=None,
        )
        mock_client.chat_completions_or_raise = AsyncMock(
            side_effect=[tool_call_result, final_result]
        )

        def my_tool(name, arguments):
            args = json.loads(arguments)
            return f"{args['city']}今天晴，25°C"

        agent = AgentClient(
            client=mock_client,
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_executor=my_tool,
        )
        result = await agent.run("查天气")

        assert result.content == "北京今天晴，25°C"
        assert result.rounds == 1
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].result == "北京今天晴，25°C"
        # 累计 usage
        assert result.usage["prompt_tokens"] == 30
        assert result.usage["completion_tokens"] == 15

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self):
        """多次工具调用（2轮）"""
        mock_client = AsyncMock()

        results = [
            # 第一轮：调用工具 A
            ChatCompletionResult(
                content=None,
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=[
                    ToolCall(
                        id="call_1", type="function", function={"name": "step1", "arguments": "{}"}
                    ),
                ],
            ),
            # 第二轮：调用工具 B
            ChatCompletionResult(
                content=None,
                usage={"prompt_tokens": 20, "completion_tokens": 5},
                tool_calls=[
                    ToolCall(
                        id="call_2", type="function", function={"name": "step2", "arguments": "{}"}
                    ),
                ],
            ),
            # 最终返回
            ChatCompletionResult(
                content="done",
                usage={"prompt_tokens": 30, "completion_tokens": 10},
                tool_calls=None,
            ),
        ]
        mock_client.chat_completions_or_raise = AsyncMock(side_effect=results)

        agent = AgentClient(
            client=mock_client,
            tools=[
                {"type": "function", "function": {"name": "step1"}},
                {"type": "function", "function": {"name": "step2"}},
            ],
            tool_executor=lambda name, args: f"{name}_result",
        )
        result = await agent.run("multi step")

        assert result.content == "done"
        assert result.rounds == 2
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "step1"
        assert result.tool_calls[1].name == "step2"

    @pytest.mark.asyncio
    async def test_max_rounds_limit(self):
        """达到 max_rounds 限制时停止"""
        mock_client = AsyncMock()
        # 永远返回 tool_call
        mock_client.chat_completions_or_raise = AsyncMock(
            return_value=ChatCompletionResult(
                content=None,
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=[
                    ToolCall(
                        id="call_x", type="function", function={"name": "loop", "arguments": "{}"}
                    ),
                ],
            )
        )

        agent = AgentClient(
            client=mock_client,
            tools=[{"type": "function", "function": {"name": "loop"}}],
            tool_executor=lambda name, args: "result",
            max_rounds=3,
        )
        result = await agent.run("infinite loop")

        assert result.rounds == 3
        assert len(result.tool_calls) == 3

    @pytest.mark.asyncio
    async def test_async_tool_executor(self):
        """异步 tool_executor 支持"""
        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            side_effect=[
                ChatCompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            type="function",
                            function={"name": "async_tool", "arguments": "{}"},
                        ),
                    ],
                ),
                ChatCompletionResult(content="done", tool_calls=None),
            ]
        )

        async def async_executor(name, arguments):
            return "async_result"

        agent = AgentClient(
            client=mock_client,
            tools=[{"type": "function", "function": {"name": "async_tool"}}],
            tool_executor=async_executor,
        )
        result = await agent.run("test async")
        assert result.tool_calls[0].result == "async_result"

    @pytest.mark.asyncio
    async def test_tool_executor_error(self):
        """tool_executor 抛出异常时返回错误信息"""
        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            side_effect=[
                ChatCompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            type="function",
                            function={"name": "bad_tool", "arguments": "{}"},
                        ),
                    ],
                ),
                ChatCompletionResult(content="handled", tool_calls=None),
            ]
        )

        def bad_executor(name, arguments):
            raise RuntimeError("tool crashed")

        agent = AgentClient(
            client=mock_client,
            tools=[{"type": "function", "function": {"name": "bad_tool"}}],
            tool_executor=bad_executor,
        )
        result = await agent.run("test error")
        assert "tool crashed" in result.tool_calls[0].result

    @pytest.mark.asyncio
    async def test_no_tool_executor(self):
        """未配置 tool_executor 时返回错误"""
        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            side_effect=[
                ChatCompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            type="function",
                            function={"name": "some_tool", "arguments": "{}"},
                        ),
                    ],
                ),
                ChatCompletionResult(content="done", tool_calls=None),
            ]
        )

        agent = AgentClient(
            client=mock_client,
            tools=[{"type": "function", "function": {"name": "some_tool"}}],
            # 没有 tool_executor
        )
        result = await agent.run("test")
        assert "No tool_executor" in result.tool_calls[0].result


class TestAgentChat:
    """多轮对话测试"""

    @pytest.mark.asyncio
    async def test_chat_maintains_history(self):
        """chat() 自动维护对话历史"""
        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            return_value=ChatCompletionResult(content="回复", tool_calls=None)
        )

        agent = AgentClient(client=mock_client, system="sys")

        await agent.chat("第一条")
        assert len(agent._history) == 2  # user + assistant

        await agent.chat("第二条")
        assert len(agent._history) == 4  # 2 轮 * (user + assistant)

        # 验证第二次调用时 messages 包含历史
        second_call_messages = mock_client.chat_completions_or_raise.call_args_list[1].args[0]
        # system + history(user+assistant) + new user
        assert len(second_call_messages) == 4
        assert second_call_messages[0]["role"] == "system"
        assert second_call_messages[1]["content"] == "第一条"
        assert second_call_messages[2]["content"] == "回复"
        assert second_call_messages[3]["content"] == "第二条"

    @pytest.mark.asyncio
    async def test_chat_reset(self):
        """reset() 清空历史"""
        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            return_value=ChatCompletionResult(content="ok", tool_calls=None)
        )

        agent = AgentClient(client=mock_client)
        await agent.chat("hello")
        assert len(agent._history) == 2

        agent.reset()
        assert len(agent._history) == 0


class TestContextTruncation:
    """上下文裁剪测试"""

    @pytest.mark.asyncio
    async def test_truncate_long_history(self):
        """超过 max_context_tokens 时裁剪旧消息"""
        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            return_value=ChatCompletionResult(content="ok", tool_calls=None)
        )

        agent = AgentClient(
            client=mock_client,
            system="system prompt",
            max_context_tokens=100,  # 很小的限制
        )

        # 塞入大量历史
        for i in range(20):
            agent._history.append({"role": "user", "content": f"message {i} " + "x" * 50})
            agent._history.append({"role": "assistant", "content": f"reply {i} " + "y" * 50})

        await agent.run("new message")

        # 验证实际发送的 messages 被裁剪了
        sent_messages = mock_client.chat_completions_or_raise.call_args.args[0]
        # 应该只包含 system + 少量最近消息
        assert len(sent_messages) < 42  # 远少于 40 历史 + system + new

    def test_truncate_preserves_system(self):
        """裁剪时始终保留 system message"""
        agent = AgentClient(client=MagicMock(), system="important system", max_context_tokens=50)
        agent._history = [
            {"role": "user", "content": "a" * 200},
            {"role": "assistant", "content": "b" * 200},
        ]
        messages = agent._build_messages("new")
        # system 必须存在
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "important system"


class TestStructuredOutput:
    """Structured output (Pydantic 绑定) 测试"""

    @pytest.mark.asyncio
    async def test_pydantic_response_format(self):
        """Pydantic model 自动转换为 JSON schema 并解析"""
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("pydantic not installed")

        class Decision(BaseModel):
            action: str
            reason: str

        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            return_value=ChatCompletionResult(
                content='{"action": "approve", "reason": "符合规范"}',
                tool_calls=None,
            )
        )

        agent = AgentClient(client=mock_client)
        result = await agent.run("分析需求", response_format=Decision)

        assert result.parsed is not None
        assert result.parsed.action == "approve"
        assert result.parsed.reason == "符合规范"

        # 验证传递了 response_format 到 client
        call_kwargs = mock_client.chat_completions_or_raise.call_args.kwargs
        assert "response_format" in call_kwargs
        assert call_kwargs["response_format"]["type"] == "json_schema"

    @pytest.mark.asyncio
    async def test_dict_response_format_passthrough(self):
        """普通 dict response_format 直接透传"""
        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            return_value=ChatCompletionResult(content='{"key": "value"}', tool_calls=None)
        )

        agent = AgentClient(client=mock_client)
        result = await agent.run("test", response_format={"type": "json_object"})

        call_kwargs = mock_client.chat_completions_or_raise.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert result.parsed is None  # 没有 Pydantic model，不解析

    @pytest.mark.asyncio
    async def test_invalid_json_graceful(self):
        """LLM 返回无效 JSON 时不崩溃"""
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("pydantic not installed")

        class MyModel(BaseModel):
            name: str

        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            return_value=ChatCompletionResult(content="not json at all", tool_calls=None)
        )

        agent = AgentClient(client=mock_client)
        result = await agent.run("test", response_format=MyModel)

        assert result.parsed is None
        assert result.content == "not json at all"


class TestEventCallbacks:
    """事件回调测试"""

    @pytest.mark.asyncio
    async def test_on_tool_call_callback(self):
        """on_tool_call 回调被触发"""
        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            side_effect=[
                ChatCompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            type="function",
                            function={"name": "my_tool", "arguments": '{"a":1}'},
                        ),
                    ],
                ),
                ChatCompletionResult(content="done", tool_calls=None),
            ]
        )

        called_with = []

        agent = AgentClient(
            client=mock_client,
            tools=[{"type": "function", "function": {"name": "my_tool"}}],
            tool_executor=lambda n, a: "ok",
        )
        agent.on_tool_call = lambda name, args: called_with.append((name, args))

        await agent.run("test")
        assert len(called_with) == 1
        assert called_with[0][0] == "my_tool"

    @pytest.mark.asyncio
    async def test_on_tool_result_callback(self):
        """on_tool_result 回调被触发"""
        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            side_effect=[
                ChatCompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="c1", type="function", function={"name": "t", "arguments": "{}"}
                        ),
                    ],
                ),
                ChatCompletionResult(content="done", tool_calls=None),
            ]
        )

        results = []

        agent = AgentClient(
            client=mock_client,
            tools=[{"type": "function", "function": {"name": "t"}}],
            tool_executor=lambda n, a: "tool_output",
        )
        agent.on_tool_result = lambda name, result: results.append((name, result))

        await agent.run("test")
        assert len(results) == 1
        assert results[0] == ("t", "tool_output")

    @pytest.mark.asyncio
    async def test_on_llm_response_callback(self):
        """on_llm_response 回调被触发"""
        mock_client = AsyncMock()
        mock_client.chat_completions_or_raise = AsyncMock(
            return_value=ChatCompletionResult(content="hello", tool_calls=None)
        )

        responses = []

        agent = AgentClient(client=mock_client)
        agent.on_llm_response = lambda r: responses.append(r)

        await agent.run("test")
        assert len(responses) == 1
        assert responses[0].content == "hello"


class TestDataTypes:
    """数据类型测试"""

    def test_agent_result_defaults(self):
        """AgentResult 默认值"""
        result = AgentResult(content="test")
        assert result.content == "test"
        assert result.rounds == 0
        assert result.tool_calls == []
        assert result.usage is None
        assert result.parsed is None

    def test_tool_call_record(self):
        """ToolCallRecord 创建"""
        record = ToolCallRecord(name="fn", arguments='{"a":1}', result="ok")
        assert record.name == "fn"
        assert record.arguments == '{"a":1}'
        assert record.result == "ok"


class TestBuildAssistantMsg:
    """_build_assistant_msg 测试"""

    def test_with_content_and_tool_calls(self):
        """有 content 和 tool_calls"""
        result = ChatCompletionResult(
            content="thinking...",
            tool_calls=[
                ToolCall(id="c1", type="function", function={"name": "fn", "arguments": "{}"}),
            ],
        )
        msg = AgentClient._build_assistant_msg(result)
        assert msg["role"] == "assistant"
        assert msg["content"] == "thinking..."
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["id"] == "c1"

    def test_with_no_content(self):
        """content 为 None"""
        result = ChatCompletionResult(
            content=None,
            tool_calls=[
                ToolCall(id="c1", type="function", function={"name": "fn", "arguments": "{}"}),
            ],
        )
        msg = AgentClient._build_assistant_msg(result)
        assert msg["content"] is None

    def test_no_tool_calls(self):
        """无 tool_calls"""
        result = ChatCompletionResult(content="hello", tool_calls=None)
        msg = AgentClient._build_assistant_msg(result)
        assert "tool_calls" not in msg


class TestPrepareResponseFormat:
    """_prepare_response_format 测试"""

    def test_dict_passthrough(self):
        """dict 格式直接透传"""
        fmt, model = AgentClient._prepare_response_format({"type": "json_object"})
        assert fmt == {"type": "json_object"}
        assert model is None

    def test_pydantic_model(self):
        """Pydantic model 转换为 json_schema"""
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("pydantic not installed")

        class MyModel(BaseModel):
            name: str
            age: int

        fmt, model = AgentClient._prepare_response_format(MyModel)
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "MyModel"
        assert "properties" in fmt["json_schema"]["schema"]
        assert model is MyModel

    def test_none(self):
        """None 值"""
        fmt, model = AgentClient._prepare_response_format(None)
        assert fmt is None
        assert model is None


class TestMergeUsage:
    """测试 _merge_usage 累加逻辑"""

    def test_none_plus_none(self):
        """两个 None 返回 None"""
        from flexllm.agent.client import _merge_usage

        assert _merge_usage(None, None) is None

    def test_none_plus_dict(self):
        """total=None 时返回 new 的副本"""
        from flexllm.agent.client import _merge_usage

        new = {"prompt_tokens": 10, "completion_tokens": 5}
        result = _merge_usage(None, new)
        assert result == new
        assert result is not new  # 是副本

    def test_dict_plus_none(self):
        """new=None 时返回 total 原值"""
        from flexllm.agent.client import _merge_usage

        total = {"prompt_tokens": 10}
        result = _merge_usage(total, None)
        assert result is total

    def test_dict_plus_dict(self):
        """两个 dict 累加数值字段"""
        from flexllm.agent.client import _merge_usage

        total = {"prompt_tokens": 10, "completion_tokens": 5}
        new = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        result = _merge_usage(total, new)
        assert result["prompt_tokens"] == 30
        assert result["completion_tokens"] == 15
        assert result["total_tokens"] == 30

    def test_non_numeric_fields_ignored(self):
        """非数值字段不累加"""
        from flexllm.agent.client import _merge_usage

        total = {"prompt_tokens": 10}
        new = {"prompt_tokens": 5, "model": "gpt-4"}
        result = _merge_usage(total, new)
        assert result["prompt_tokens"] == 15
        assert "model" not in result
