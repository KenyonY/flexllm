"""Structured completion entry points shared by providers and the endpoint pool."""

import asyncio
import sys
import time
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BatchResult, ChatCompletionResult


class LegacyResponseWarning(FutureWarning):
    """A legacy return shape is being used; these shapes go away in 0.18.0.

    刻意用 FutureWarning 而不是 DeprecationWarning：后者默认对终端用户静默，
    而这里需要调用方真的看见。
    """


_SUPPRESS_LEGACY_WARNING = ContextVar("flexllm_suppress_legacy_warning", default=False)


@contextmanager
def suppress_legacy_response_warning():
    """Avoid duplicate warnings when a public pool API calls a provider internally."""
    token = _SUPPRESS_LEGACY_WARNING.set(True)
    try:
        yield
    finally:
        _SUPPRESS_LEGACY_WARNING.reset(token)


def warn_legacy_response(*, return_raw: bool, return_usage: bool, raise_on_error: bool = False):
    if _SUPPRESS_LEGACY_WARNING.get() or (return_usage and not return_raw and raise_on_error):
        return
    # Attribute warnings to the consumer even through pool and sync wrappers.
    level = 2
    frame = sys._getframe(1)
    while frame and frame.f_globals.get("__name__", "").startswith(("flexllm.", "asyncio.")):
        level += 1
        frame = frame.f_back
    warnings.warn(
        "Legacy completion return shapes are deprecated and will be removed in flexllm 0.18.0; "
        "use complete()/complete_batch() (or their _sync variants). complete() returns a "
        "ChatCompletionResult and raises typed errors; complete_batch() returns an equal-length "
        "BatchResult whose failed items carry .error instead of raising. "
        "Read .content for text and .raw_response for the provider response. "
        "Existing return values stay unchanged until 0.18.0.",
        LegacyResponseWarning,
        stacklevel=level,
    )


def merge_tool_call_delta(tool_calls: dict[int, dict], delta: dict) -> None:
    """把一条 OpenAI 形态的流式 tool_call 增量按 index 合并进 tool_calls。

    id/type/name 首次出现即定值，arguments 逐段拼接。provider 各自的流式实现都产出
    这种形态（Gemini/Claude 已转换），所以累加只有这一份。
    """
    current = tool_calls.setdefault(
        delta.get("index", 0),
        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
    )
    if delta.get("id"):
        current["id"] = delta["id"]
    if delta.get("type"):
        current["type"] = delta["type"]
    function = delta.get("function", {})
    if function.get("name"):
        current["function"]["name"] = function["name"]
    if "arguments" in function:
        current["function"]["arguments"] += function["arguments"]


class CompletionMixin:
    """New entry points always return structured results or raise typed errors."""

    @staticmethod
    def _validate_completion_options(kwargs):
        reserved = {
            "return_raw",
            "return_usage",
            "raise_on_error",
            "return_summary",
            "return_cost_report",
        }.intersection(kwargs)
        if reserved:
            raise ValueError(
                "Structured completions do not accept return-shape options: "
                + ", ".join(sorted(reserved))
            )

    async def complete(self, messages, model=None, **kwargs) -> "ChatCompletionResult":
        """Return content, usage, tools, reasoning and the raw response in one object."""
        self._validate_completion_options(kwargs)
        return await self.chat_completions(
            messages, model=model, return_usage=True, raise_on_error=True, **kwargs
        )

    def complete_sync(self, messages, model=None, **kwargs) -> "ChatCompletionResult":
        """Synchronous counterpart of complete()."""
        return asyncio.run(self.complete(messages, model=model, **kwargs))

    async def complete_stream(self, messages, model=None, **kwargs):
        """流式完成：边生成边产出增量事件，结尾给出与 complete() 同构的完整结果。

        事件恒为 dict（不受任何开关影响，思考内容不会混进正文）：
            {"type": "content", "content": str}
            {"type": "thinking", "content": str}
            {"type": "tool_call_delta", "tool_calls": [...]}  OpenAI 形态的原始增量
            {"type": "extra", "extra": dict}                  网关带外字段
            {"type": "result", "result": ChatCompletionResult} 最后一条，成功时恰好一次

        result 里 content/reasoning_content/tool_calls 已累加好，finish_reason/usage/
        assistant_message（下一轮需原样回传的续接状态）也都在上面，调用方无需自己拼。
        失败抛 typed error，与 complete() 相同；多 endpoint 时只在首个事件前故障转移。
        """
        from .base import ChatCompletionResult, ToolCall

        self._validate_completion_options(kwargs)
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: dict[int, dict] = {}
        extra: dict = {}
        assistant_message = finish_reason = usage = None
        start = time.perf_counter()

        async for event in self.chat_completions_stream(
            messages, model=model, return_usage=True, **kwargs
        ):
            kind = event["type"]
            # 汇总类事件只进 result，不单独产出
            if kind == "assistant_message":
                assistant_message = event["message"]
                continue
            if kind == "finish":
                finish_reason = event["reason"]
                continue
            if kind == "usage":
                usage = event["usage"]
                continue

            if kind == "content":
                content_parts.append(event["content"])
            elif kind == "thinking":
                thinking_parts.append(event["content"])
            elif kind == "tool_call_delta":
                for delta in event["tool_calls"]:
                    merge_tool_call_delta(tool_calls, delta)
            elif kind == "extra":
                extra.update(event["extra"])
            yield event

        yield {
            "type": "result",
            "result": ChatCompletionResult(
                content="".join(content_parts) or None,
                usage=usage,
                reasoning_content="".join(thinking_parts) or None,
                tool_calls=[ToolCall(**tc) for _, tc in sorted(tool_calls.items())] or None,
                finish_reason=finish_reason,
                extra=extra or None,
                assistant_message=assistant_message,
                latency=time.perf_counter() - start,
            ),
        }

    async def complete_batch(self, messages_list, model=None, **kwargs) -> "BatchResult":
        """批量完成：等长同构的 BatchResult，单条失败不抛异常。

        批量里"某条失败"是预期结果的一种，不是控制流事件——把它升级成异常会逼调用方
        用 try/except 走正常分支，还会让已完成的结果需要从异常对象里捞。失败项是
        content=None 且带 error 的 ChatCompletionResult，用 .ok 分流即可；需要
        fail-fast 调用 result.raise_for_errors()。
        """
        self._validate_completion_options(kwargs)
        run = await self._run_batch(messages_list, model=model, return_usage=True, **kwargs)
        return run.to_batch_result()

    def complete_batch_sync(self, messages_list, model=None, **kwargs) -> "BatchResult":
        """Synchronous counterpart of complete_batch()."""
        return asyncio.run(self.complete_batch(messages_list, model=model, **kwargs))
