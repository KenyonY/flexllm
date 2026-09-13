"""Structured completion entry points shared by providers and the endpoint pool."""

import asyncio
import sys
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import ChatCompletionResult


class LegacyResponseWarning(FutureWarning):
    """A legacy return shape is being used during the migration period."""


def warn_legacy_response(*, return_raw: bool, return_usage: bool, raise_on_error: bool):
    if return_usage and not return_raw and raise_on_error:
        return
    # Attribute warnings to the consumer even through pool and sync wrappers.
    level = 2
    frame = sys._getframe(1)
    while frame and frame.f_globals.get("__name__", "").startswith(("flexllm.", "asyncio.")):
        level += 1
        frame = frame.f_back
    warnings.warn(
        "Legacy completion return shapes are deprecated; use complete()/complete_batch() "
        "(or their _sync variants) for ChatCompletionResult with typed exceptions. "
        "Read .content for text and .raw_response for the provider response. "
        "Existing return values remain unchanged during the migration period.",
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

    async def complete_batch(
        self, messages_list, model=None, **kwargs
    ) -> list["ChatCompletionResult"]:
        """Return ordered structured results; partial failures raise BatchRequestError."""
        from .base import BatchRequestError, LLMRequestError

        self._validate_completion_options(kwargs)
        results = await self.chat_completions_batch(
            messages_list, model=model, return_usage=True, raise_on_error=True, **kwargs
        )
        errors = {
            i: LLMRequestError("Batch item was not completed")
            for i, result in enumerate(results)
            if result is None
        }
        if errors:
            raise BatchRequestError(
                "Batch stopped before completion", results=results, errors=errors
            )
        return results

    def complete_batch_sync(
        self, messages_list, model=None, **kwargs
    ) -> list["ChatCompletionResult"]:
        """Synchronous counterpart of complete_batch()."""
        return asyncio.run(self.complete_batch(messages_list, model=model, **kwargs))
