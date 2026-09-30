"""Structured completion entry points shared by providers and the endpoint pool."""

import asyncio
import sys
import time
import warnings
from contextlib import aclosing, contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BatchResult, ChatCompletionResult, _BatchRun


class LegacyResponseWarning(FutureWarning):
    """A legacy return shape or entry point is being used; both go away in 0.18.0.

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


def _warn_legacy(message: str):
    if _SUPPRESS_LEGACY_WARNING.get():
        return
    # Attribute warnings to the consumer even through pool and sync wrappers.
    level = 2
    frame = sys._getframe(1)
    while frame and frame.f_globals.get("__name__", "").startswith(("flexllm.", "asyncio.")):
        level += 1
        frame = frame.f_back
    warnings.warn(message, LegacyResponseWarning, stacklevel=level)


def warn_legacy_response(*, return_raw: bool, return_usage: bool, raise_on_error: bool = False):
    if return_usage and not return_raw and raise_on_error:
        return
    _warn_legacy(
        "Legacy completion return shapes are deprecated and will be removed in flexllm 0.18.0; "
        "use chat()/chat_batch() (or their _sync variants). chat() returns a "
        "ChatCompletionResult and raises typed errors; chat_batch() returns an equal-length "
        "BatchResult whose failed items carry .error instead of raising. "
        "Read .content for text and .raw_response for the provider response. "
        "Existing return values stay unchanged until 0.18.0."
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


async def think_tagged_text(events, on_result=None):
    """把 chat_stream 的事件还原成纯文本片段，思考内容包在 <think>…</think> 里。

    给只认文本流的消费方（CLI 终端输出、MllmClient 的 token 流）用。需要结构化结果的
    （如把回复写回对话历史——那里不该混进思考文本）传 on_result 接住结尾的
    ChatCompletionResult。
    """
    thinking = False
    async for event in events:
        if event["type"] == "result" and on_result is not None:
            on_result(event["result"])
        elif event["type"] == "thinking":
            if not thinking:
                yield "<think>\n"
                thinking = True
            yield event["content"]
        elif event["type"] == "content":
            if thinking:
                yield "</think>"
                thinking = False
            yield event["content"]
    if thinking:
        yield "</think>"


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

    async def chat(self, messages, model=None, **kwargs) -> "ChatCompletionResult":
        """Return content, usage, tools, reasoning and the raw response in one object."""
        self._validate_completion_options(kwargs)
        return await self.chat_completions(
            messages, model=model, return_usage=True, raise_on_error=True, **kwargs
        )

    def chat_sync(self, messages, model=None, **kwargs) -> "ChatCompletionResult":
        """Synchronous counterpart of chat()."""
        return asyncio.run(self.chat(messages, model=model, **kwargs))

    async def chat_stream(self, messages, model=None, **kwargs):
        """流式完成：边生成边产出增量事件，结尾给出与 chat() 同构的完整结果。

        事件恒为 dict（不受任何开关影响，思考内容不会混进正文）：
            {"type": "content", "content": str}
            {"type": "thinking", "content": str}
            {"type": "tool_call_delta", "tool_calls": [...]}  OpenAI 形态的原始增量
            {"type": "extra", "extra": dict}                  网关带外字段
            {"type": "result", "result": ChatCompletionResult} 最后一条，成功时恰好一次

        result 里 content/reasoning_content/tool_calls 已累加好，finish_reason/usage/
        assistant_message（下一轮需原样回传的续接状态）也都在上面，调用方无需自己拼。
        失败抛 typed error，与 chat() 相同；多 endpoint 时只在首个事件前故障转移。
        """
        from .base import ChatCompletionResult, ToolCall

        self._validate_completion_options(kwargs)
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: dict[int, dict] = {}
        extra: dict = {}
        assistant_message = finish_reason = usage = None
        start = time.perf_counter()

        async for event in self._stream(messages, model=model, return_usage=True, **kwargs):
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

    async def chat_batch(self, messages_list, model=None, **kwargs) -> "BatchResult":
        """批量完成：等长同构的 BatchResult，单条失败不抛异常。

        批量里"某条失败"是预期结果的一种，不是控制流事件——把它升级成异常会逼调用方
        用 try/except 走正常分支，还会让已完成的结果需要从异常对象里捞。失败项是
        content=None 且带 error 的 ChatCompletionResult，用 .ok 分流即可；需要
        fail-fast 调用 result.raise_for_errors()。
        """
        self._validate_completion_options(kwargs)
        run = await self._run_batch(messages_list, model=model, return_usage=True, **kwargs)
        return run.to_batch_result()

    def chat_batch_sync(self, messages_list, model=None, **kwargs) -> "BatchResult":
        """Synchronous counterpart of chat_batch()."""
        return asyncio.run(self.chat_batch(messages_list, model=model, **kwargs))

    async def chat_batch_iter(self, messages_list, model=None, **kwargs):
        """边完成边产出的 chat_batch：逐条 yield ``(index, ChatCompletionResult)``。

        与 chat_batch 是同一份执行（缓存/断点续传/成本/故障转移都一样），差别只在交付方式：
        按完成顺序逐条交出，不在内存里攒整批。每个 index 恰好出现一次——断点续传恢复的、
        缓存命中的也会产出（排在最前），所以把它收集起来按 index 排好就等于 chat_batch。
        失败项与 chat_batch 一致：content=None 且带 error，不抛异常。

        提前停止：生成器关闭时取消在途请求并关闭 checkpoint 文件，已落盘的下次可续跑。
        直接 break 出 async for 时，关闭由事件循环稍后调度（Python async generator 的
        语义）；要在 break 之后立刻读 checkpoint，用 ``contextlib.aclosing`` 包住::

            async with aclosing(client.chat_batch_iter(msgs, output_jsonl=path)) as items:
                async for index, result in items:
                    ...
        """
        from .base import ChatCompletionResult

        self._validate_completion_options(kwargs)
        async with aclosing(
            self._iter_batch(messages_list, model=model, return_usage=True, **kwargs)
        ) as items:
            async for index, value, error in items:
                yield index, (value if error is None else ChatCompletionResult._from_error(error))

    async def _run_batch(self, messages_list, **kwargs) -> "_BatchRun":
        """把 _iter_batch 收集成整批结果：chat_batch 与旧 chat_completions_batch 共用。"""
        from .base import _BatchReport, _BatchRun

        started = time.perf_counter()
        report = _BatchReport()
        responses: list = [None] * len(messages_list)
        errors: dict = {}
        async with aclosing(self._iter_batch(messages_list, report=report, **kwargs)) as items:
            async for index, value, error in items:
                responses[index] = value
                if error is not None:
                    errors[index] = error
        return _BatchRun(
            responses=responses,
            errors=errors,
            summary=report.summary,
            cost=self._batch_cost(),
            elapsed=time.perf_counter() - started,
        )

    async def chat_completions_stream(self, *args, **kwargs):
        """流式聊天完成（旧接口）。

        .. deprecated:: 0.17.8
            用 chat_stream()：事件形状固定，结尾给出汇总好的 ChatCompletionResult。
            本方法在 0.18.0 移除。参数与产出与 0.17.7 完全一致。
        """
        _warn_legacy(
            "chat_completions_stream() is deprecated and will be removed in flexllm 0.18.0; "
            "use chat_stream(), which yields fixed-shape event dicts and ends with a "
            '{"type": "result"} event carrying the aggregated ChatCompletionResult.'
        )
        async for chunk in self._stream(*args, **kwargs):
            yield chunk

    async def iter_chat_completions_batch(self, *args, **kwargs):
        """迭代式批量聊天完成（旧接口）。

        .. deprecated:: 0.17.8
            用 chat_batch_iter()：逐条产出 (index, ChatCompletionResult)，与 chat_batch
            共用同一份执行。本方法在 0.18.0 移除。参数与产出与 0.17.7 完全一致。
        """
        _warn_legacy(
            "iter_chat_completions_batch() is deprecated and will be removed in flexllm 0.18.0; "
            "use chat_batch_iter(), which yields (index, ChatCompletionResult) pairs and shares "
            "chat_batch()'s caching, checkpointing and failover."
        )
        async for result in self._legacy_iter_batch(*args, **kwargs):
            yield result
