"""Structured completion entry points shared by providers and the endpoint pool."""

import asyncio
import sys
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
