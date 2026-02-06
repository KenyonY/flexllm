"""
AgentClient - 基于 LLMClient 的 Agent 客户端

支持 tool-use 循环、多轮对话、structured output、事件回调和代码验证闭环。
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

    # 带验证的任务（自动修复错误）
    from flexllm.agent.validators import PythonSyntaxValidator, PythonLintValidator
    result = await agent.run_with_validation(
        "帮我修复这个 bug",
        validators=[PythonSyntaxValidator(), PythonLintValidator()],
    )
"""

import asyncio
import inspect
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Callable

from .types import AgentResult, ToolCallRecord

if TYPE_CHECKING:
    from ..clients.base import LLMClientBase
    from .memory import MemoryStore
    from .tracing import TraceExporter
    from .validators.base import ValidationResult, Validator

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
        tools: OpenAI 格式的 tool definitions（旧接口）
        tool_executor: 工具执行函数 (name, arguments) -> result（旧接口）
        tool_registry: ToolRegistry 实例（新接口，与 tools+tool_executor 二选一）
        max_rounds: 单次 run 最大 tool-calling 轮数
        max_context_tokens: 可选，上下文窗口限制（粗略按字符估算）
        approval_handler: 工具执行审批函数 (name, args, readonly) -> bool
        trace_exporter: Trace 导出器（可观测性）
        memory: MemoryStore 实例（持久化记忆）
        session_id: 会话 ID，配合 memory 使用
    """

    def __init__(
        self,
        client: "LLMClientBase",
        system: str = None,
        tools: list[dict] = None,
        tool_executor: Callable[[str, str], str] = None,
        tool_registry: Any | None = None,
        max_rounds: int = 10,
        max_context_tokens: int | None = None,
        approval_handler: Callable[[str, str, bool], bool] | None = None,
        trace_exporter: "TraceExporter | None" = None,
        memory: "MemoryStore | None" = None,
        session_id: str | None = None,
    ):
        self.client = client
        self.system = system
        self.max_rounds = max_rounds
        self.max_context_tokens = max_context_tokens
        self.approval_handler = approval_handler
        self.trace_exporter = trace_exporter
        self.memory = memory
        self.session_id = session_id

        # 工具系统：优先使用 tool_registry
        self._tool_registry = tool_registry
        if tool_registry is not None:
            self.tools = tool_registry.get_tool_defs()
            self.tool_executor = None  # 由 registry 管理执行
        else:
            self.tools = tools
            self.tool_executor = tool_executor

        # 多轮对话历史（如果有 memory + session_id，从记忆加载）
        if memory and session_id:
            self._history: list[dict] = memory.load(session_id)
        else:
            self._history: list[dict] = []

        # 事件回调（可选）
        self.on_tool_call: Callable[[str, str], Any] | None = None
        self.on_tool_result: Callable[[str, str], Any] | None = None
        self.on_llm_response: Callable[[Any], Any] | None = None
        self.on_validation: Callable[[list[str], list], Any] | None = None  # (files, results)

    @property
    def tool_registry(self) -> Any | None:
        """获取当前工具注册表"""
        return self._tool_registry

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

        # 持久化到 memory
        if self.memory and self.session_id:
            self.memory.save(self.session_id, self._history)

        return result

    async def _run_loop(self, messages: list[dict], **kwargs) -> AgentResult:
        """tool-use 核心循环"""
        rounds = 0
        tool_history = []
        total_usage = None

        # Tracing setup (only if exporter configured)
        root_span = None
        trace = None
        if self.trace_exporter:
            from .tracing import Span, SpanContext, Trace, _new_span_id

            root_span = Span(
                span_id=_new_span_id(),
                name="agent.run",
                start_time=time.time(),
            )
            trace = Trace(
                trace_id=_new_span_id(),
                root_span=root_span,
            )

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
            # LLM 调用 tracing
            if root_span:
                llm_span_ctx = SpanContext("llm.call", parent=root_span)
                llm_span = llm_span_ctx.__enter__()

            result = await self.client.chat_completions_or_raise(
                messages, return_usage=True, **call_kwargs
            )

            if root_span:
                if result.usage:
                    llm_span.attributes.update(
                        {
                            "input_tokens": result.usage.get("prompt_tokens", 0),
                            "output_tokens": result.usage.get("completion_tokens", 0),
                        }
                    )
                llm_span_ctx.__exit__(None, None, None)

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

            # 并行执行所有 tool calls
            tool_calls_info = [
                (tc, tc.function["name"], tc.function.get("arguments", "{}"))
                for tc in result.tool_calls
            ]

            # 触发 on_tool_call 回调（所有工具）
            for tc, fn_name, fn_args in tool_calls_info:
                if self.on_tool_call:
                    self._fire_callback(self.on_tool_call, fn_name, fn_args)

            # 并行执行工具
            if len(tool_calls_info) > 1:
                tasks = [
                    self._execute_tool(fn_name, fn_args) for _, fn_name, fn_args in tool_calls_info
                ]
                outputs = await asyncio.gather(*tasks)
            else:
                outputs = [await self._execute_tool(tool_calls_info[0][1], tool_calls_info[0][2])]

            # 处理结果
            for (tc, fn_name, fn_args), output in zip(tool_calls_info, outputs):
                # 工具调用 tracing
                if root_span:
                    tool_span_ctx = SpanContext(f"tool.{fn_name}", parent=root_span)
                    tool_span = tool_span_ctx.__enter__()
                    tool_span.attributes["success"] = not output.startswith("[error:")
                    tool_span_ctx.__exit__(None, None, None)

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

        # 导出 trace
        if self.trace_exporter and trace and root_span:
            root_span.end_time = time.time()
            trace.total_rounds = rounds
            if total_usage:
                trace.total_tokens = total_usage.get("total_tokens", 0)
            self.trace_exporter.export(trace)

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
        """执行工具调用（支持 ToolRegistry 或旧式 tool_executor）"""
        # 审批检查
        if self.approval_handler:
            readonly = False
            if self._tool_registry and name in self._tool_registry:
                readonly = self._tool_registry.is_readonly(name)
            else:
                from .tools.base import TOOL_REGISTRY

                tool_def = TOOL_REGISTRY.get(name)
                readonly = tool_def.readonly if tool_def else False
            if not self.approval_handler(name, arguments, readonly):
                return "[denied: tool execution was denied by user]"

        # 优先使用 ToolRegistry
        if self._tool_registry:
            return await self._tool_registry.execute_async(name, arguments)

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

    # ========== 验证闭环 ==========

    async def run_with_validation(
        self,
        user_input: str,
        validators: list["Validator"] | None = None,
        max_fix_attempts: int = 3,
        **kwargs,
    ) -> AgentResult:
        """
        带验证闭环的任务执行。

        修改代码后自动运行验证器，如果验证失败则让 LLM 修复并重试。

        Args:
            user_input: 用户输入
            validators: 验证器列表，如 [PythonSyntaxValidator(), PythonLintValidator()]
            max_fix_attempts: 最大修复尝试次数
            **kwargs: 传递给 run() 的额外参数

        Returns:
            AgentResult（最后一次执行的结果）
        """
        if not validators:
            # 无验证器，直接执行
            return await self.run(user_input, **kwargs)

        # 首次执行
        result = await self.run(user_input, **kwargs)

        for attempt in range(max_fix_attempts):
            # 收集本轮修改的文件
            changed_files = self._get_changed_files(result.tool_calls)
            if not changed_files:
                logger.debug("No files changed, skipping validation")
                break

            # 运行所有验证器
            validation_results = await self._run_validators(validators, changed_files)

            # 触发验证回调
            if self.on_validation:
                self._fire_callback(self.on_validation, changed_files, validation_results)

            # 检查是否全部通过
            failed_results = [r for r in validation_results if not r.success]
            if not failed_results:
                logger.debug(f"All {len(validators)} validators passed")
                break

            logger.info(
                f"Validation failed ({len(failed_results)}/{len(validators)}), "
                f"attempt {attempt + 1}/{max_fix_attempts}"
            )

            # 构建修复提示
            fix_prompt = self._build_fix_prompt(failed_results)

            # 让 LLM 修复
            result = await self.run(fix_prompt, **kwargs)

        return result

    def _get_changed_files(self, tool_calls: list[ToolCallRecord]) -> list[str]:
        """从工具调用记录中提取被修改的文件路径"""
        changed = set()
        write_tools = {"write", "edit"}

        for tc in tool_calls:
            if tc.name in write_tools:
                try:
                    args = json.loads(tc.arguments)
                    file_path = args.get("file_path") or args.get("path")
                    if file_path:
                        changed.add(file_path)
                except json.JSONDecodeError:
                    pass

        return list(changed)

    async def _run_validators(
        self,
        validators: list["Validator"],
        changed_files: list[str],
    ) -> list["ValidationResult"]:
        """并行运行所有验证器"""
        tasks = [v.validate(changed_files) for v in validators]
        return await asyncio.gather(*tasks)

    def _build_fix_prompt(self, failed_results: list["ValidationResult"]) -> str:
        """构建修复提示"""
        error_sections = []
        for result in failed_results:
            error_sections.append(str(result))

        return f"""代码修改后验证失败，请修复以下错误：

{chr(10).join(error_sections)}

请直接修复代码，不需要解释。"""


def console_approval(name: str, args: str, readonly: bool) -> bool:
    """CLI 模式：在终端询问用户。只读操作自动通过。"""
    if readonly:
        return True
    args_preview = args[:200] + "..." if len(args) > 200 else args
    print(f"[Agent] 即将执行: {name}({args_preview})")
    return input("是否允许? (y/n) ").strip().lower() in ("y", "yes")


def auto_approve(name: str, args: str, readonly: bool) -> bool:
    """全部自动通过（无人值守模式）"""
    return True
