"""
AgentClient - 基于 LLMClient 的 Agent 客户端

支持 tool-use 循环、多轮对话、structured output 和事件回调。
组合 LLMClient，不继承，不修改现有客户端。

Example:
    from flexllm import AgentClient, LLMClient

    client = LLMClient(model="gpt-4", base_url="...", api_key="...")
    agent = AgentClient(
        client=client,
        system="你是一个助手",
        tools=[{
            "type": "function",
            "function": {"name": "get_weather", "parameters": {...}}
        }],
        tool_executor=my_tool_fn,
    )

    # 单次任务
    result = await agent.run("查一下北京天气")

    # 多轮对话
    r1 = await agent.chat("你好")
    r2 = await agent.chat("帮我查天气")
    agent.reset()
"""

import asyncio
import inspect
import json
import logging
from typing import TYPE_CHECKING, Any, Callable

from .types import AgentResult, ToolCallRecord

if TYPE_CHECKING:
    from ..clients.base import LLMClientBase

logger = logging.getLogger(__name__)


def _merge_usage(total: dict | None, new: dict | None) -> dict | None:
    """累加 token 用量"""
    if new is None:
        return total
    if total is None:
        return dict(new)
    for key in new:
        if isinstance(new[key], (int, float)):
            total[key] = total.get(key, 0) + new[key]
    return total


class AgentClient:
    """
    Agent 客户端，组合 LLMClient，支持 tool-use 循环。

    Args:
        client: LLMClient 实例
        system: 系统提示词
        tools: OpenAI 格式的 tool definitions
        tool_executor: 工具执行函数 (name, arguments) -> result
        max_rounds: 单次 run 最大 tool-calling 轮数
        max_context_tokens: 可选，上下文窗口限制（粗略按字符估算）
    """

    def __init__(
        self,
        client: "LLMClientBase",
        system: str = None,
        tools: list[dict] = None,
        tool_executor: Callable[[str, str], str] = None,
        max_rounds: int = 10,
        max_context_tokens: int | None = None,
    ):
        self.client = client
        self.system = system
        self.tools = tools
        self.tool_executor = tool_executor
        self.max_rounds = max_rounds
        self.max_context_tokens = max_context_tokens

        # 多轮对话历史
        self._history: list[dict] = []

        # 事件回调（可选）
        self.on_tool_call: Callable[[str, str], Any] | None = None
        self.on_tool_result: Callable[[str, str], Any] | None = None
        self.on_llm_response: Callable[[Any], Any] | None = None

    def reset(self):
        """清空对话历史"""
        self._history.clear()

    async def run(self, user_input: str, **kwargs) -> AgentResult:
        """
        单次任务（无状态），执行 tool-use 循环直到 LLM 不再调用工具。

        Args:
            user_input: 用户输入
            **kwargs: 传递给 LLMClient.chat_completions 的额外参数
                      如果传入 response_format 为 Pydantic model，会自动转换为 JSON schema
        Returns:
            AgentResult
        """
        messages = self._build_messages(user_input)
        return await self._run_loop(messages, **kwargs)

    async def chat(self, user_input: str, **kwargs) -> AgentResult:
        """
        多轮对话（有状态），自动维护 messages 历史。

        Args:
            user_input: 用户输入
            **kwargs: 传递给 LLMClient.chat_completions 的额外参数
        Returns:
            AgentResult
        """
        messages = self._build_messages(user_input)
        result = await self._run_loop(messages, **kwargs)

        # 更新历史：添加本轮对话
        self._history.append({"role": "user", "content": user_input})
        if result.content:
            self._history.append({"role": "assistant", "content": result.content})

        return result

    async def _run_loop(self, messages: list[dict], **kwargs) -> AgentResult:
        """tool-use 核心循环"""
        rounds = 0
        tool_history = []
        total_usage = None

        # 处理 structured output: Pydantic model -> response_format
        pydantic_model = None
        response_format = kwargs.pop("response_format", None)
        if response_format is not None:
            response_format, pydantic_model = self._prepare_response_format(response_format)
            if response_format is not None:
                kwargs["response_format"] = response_format

        # 准备 tools 参数
        call_kwargs = dict(kwargs)
        if self.tools:
            call_kwargs["tools"] = self.tools

        last_content = None

        while rounds < self.max_rounds:
            result = await self.client.chat_completions(messages, return_usage=True, **call_kwargs)

            # 触发 on_llm_response 回调
            if self.on_llm_response:
                self._fire_callback(self.on_llm_response, result)

            total_usage = _merge_usage(total_usage, result.usage)
            last_content = result.content

            # 没有 tool_calls，结束循环
            if not result.tool_calls:
                break

            # 追加 assistant message（含 tool_calls）
            assistant_msg = self._build_assistant_msg(result)
            messages.append(assistant_msg)

            # 执行每个 tool call
            for tc in result.tool_calls:
                fn_name = tc.function["name"]
                fn_args = tc.function.get("arguments", "{}")

                # 触发 on_tool_call 回调
                if self.on_tool_call:
                    self._fire_callback(self.on_tool_call, fn_name, fn_args)

                output = await self._execute_tool(fn_name, fn_args)
                tool_history.append(ToolCallRecord(name=fn_name, arguments=fn_args, result=output))

                # 触发 on_tool_result 回调
                if self.on_tool_result:
                    self._fire_callback(self.on_tool_result, fn_name, output)

                # 追加 tool result message
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    }
                )

            rounds += 1

        # Structured output 解析
        parsed = None
        if pydantic_model and last_content:
            try:
                parsed = pydantic_model.model_validate_json(last_content)
            except Exception as e:
                logger.warning(f"Structured output 解析失败: {e}")

        return AgentResult(
            content=last_content,
            rounds=rounds,
            tool_calls=tool_history,
            usage=total_usage,
            parsed=parsed,
        )

    def _build_messages(self, user_input: str) -> list[dict]:
        """构建 messages 列表（system + history + user_input）"""
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.extend(self._history)
        messages.append({"role": "user", "content": user_input})

        if self.max_context_tokens:
            messages = self._truncate(messages)

        return messages

    def _truncate(self, messages: list[dict]) -> list[dict]:
        """上下文裁剪：保留 system + 尽可能多的最近消息"""
        # 粗略估算：1 token ≈ 2 个字符（中英文混合场景的折中）
        max_chars = self.max_context_tokens * 2

        system_msg = None
        other_msgs = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg
            else:
                other_msgs.append(msg)

        # system 消息始终保留
        used_chars = len(json.dumps(system_msg, ensure_ascii=False)) if system_msg else 0

        # 从后往前保留消息
        kept = []
        for msg in reversed(other_msgs):
            msg_chars = len(json.dumps(msg, ensure_ascii=False))
            if used_chars + msg_chars > max_chars:
                break
            kept.append(msg)
            used_chars += msg_chars

        kept.reverse()
        result = []
        if system_msg:
            result.append(system_msg)
        result.extend(kept)
        return result

    @staticmethod
    def _build_assistant_msg(result) -> dict:
        """从 ChatCompletionResult 构建 assistant message（含 tool_calls）"""
        msg = {"role": "assistant"}
        if result.content:
            msg["content"] = result.content
        else:
            msg["content"] = None

        if result.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": tc.function,
                }
                for tc in result.tool_calls
            ]
        return msg

    async def _execute_tool(self, name: str, arguments: str) -> str:
        """执行工具调用"""
        if not self.tool_executor:
            return json.dumps({"error": f"No tool_executor configured for tool '{name}'"})

        try:
            result = self.tool_executor(name, arguments)
            # 支持异步 tool_executor
            if inspect.isawaitable(result):
                result = await result
            return str(result)
        except Exception as e:
            logger.warning(f"Tool '{name}' 执行失败: {e}")
            return json.dumps({"error": str(e)})

    @staticmethod
    def _prepare_response_format(response_format):
        """
        处理 response_format 参数。

        如果是 Pydantic model class，提取 JSON schema 转换为 API 格式。
        否则直接透传（如 {"type": "json_object"}）。

        Returns:
            (response_format_for_api, pydantic_model_or_none)
        """
        # 检查是否是 Pydantic model class
        if isinstance(response_format, type):
            try:
                schema = response_format.model_json_schema()
                return (
                    {
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_format.__name__,
                            "schema": schema,
                        },
                    },
                    response_format,
                )
            except AttributeError:
                # 不是 Pydantic model，直接透传
                pass

        return response_format, None

    def _fire_callback(self, callback, *args):
        """触发回调（同步或异步均安全）"""
        try:
            result = callback(*args)
            if inspect.isawaitable(result):
                # 如果回调是异步的，创建 task 但不等待
                asyncio.ensure_future(result)
        except Exception as e:
            logger.warning(f"Callback 执行失败: {e}")
