"""Reasoning policy, endpoint capabilities and lossless wire adapters.

Capabilities are declarations for one endpoint/model pair, not inferred from a
model name. None means unknown; empty effort_levels / False budget_tokens mean
explicitly unsupported. This module never makes network requests.
"""

from dataclasses import asdict, dataclass
from typing import Literal
from urllib.parse import urlparse


@dataclass(frozen=True)
class Reasoning:
    """Request intent. An empty object explicitly selects server defaults."""

    enabled: bool | None = None
    effort: str | None = None
    budget_tokens: int | None = None

    def __post_init__(self):
        if self.enabled is not None and not isinstance(self.enabled, bool):
            raise ValueError("reasoning.enabled must be a boolean")
        if self.effort is not None and (
            not isinstance(self.effort, str) or not self.effort.strip() or self.effort == "none"
        ):
            raise ValueError("reasoning.effort must be a level name; use enabled=False to disable")
        if self.budget_tokens is not None and (
            type(self.budget_tokens) is not int or self.budget_tokens <= 0
        ):
            raise ValueError("reasoning.budget_tokens must be a positive integer")
        if self.enabled is False and (self.effort is not None or self.budget_tokens is not None):
            raise ValueError("disabled reasoning cannot specify effort or budget_tokens")

    @classmethod
    def parse(cls, value):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            try:
                return cls(**value)
            except TypeError as exc:
                raise ValueError(f"Invalid reasoning fields: {', '.join(map(str, value))}") from exc
        raise ValueError("reasoning must be a Reasoning object, mapping or None")

    def to_dict(self):
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class TokenBudget:
    min: int
    max: int

    def __post_init__(self):
        if type(self.min) is not int or type(self.max) is not int or not 0 < self.min <= self.max:
            raise ValueError("budget_tokens range requires 0 < min <= max")


@dataclass(frozen=True)
class ReasoningCapabilities:
    """Model capabilities. Unknown values are not promises of support."""

    can_enable: bool | None = None
    can_disable: bool | None = None
    effort_levels: tuple[str, ...] | None = None
    budget_tokens: TokenBudget | Literal[False] | None = None
    supports_effort_and_budget: bool = False

    def __post_init__(self):
        for key in ("can_enable", "can_disable"):
            value = getattr(self, key)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{key} must be boolean or None")
        if not isinstance(self.supports_effort_and_budget, bool):
            raise ValueError("supports_effort_and_budget must be boolean")
        levels = self.effort_levels
        if levels is not None:
            if not isinstance(levels, (list, tuple)) or any(
                not isinstance(x, str) or not x.strip() or x == "none" for x in levels
            ):
                raise ValueError("effort_levels must be a list of level names or None")
            object.__setattr__(self, "effort_levels", tuple(levels))
        budget = self.budget_tokens
        if isinstance(budget, dict):
            try:
                budget = TokenBudget(**budget)
            except TypeError as exc:
                raise ValueError("budget_tokens requires min and max") from exc
            object.__setattr__(self, "budget_tokens", budget)
        if budget is not None and budget is not False and not isinstance(budget, TokenBudget):
            raise ValueError("budget_tokens must be a TokenBudget, min/max mapping, False or None")

    @classmethod
    def parse(cls, value):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            try:
                return cls(**value)
            except TypeError as exc:
                raise ValueError("Invalid reasoning_capabilities fields") from exc
        raise ValueError("reasoning_capabilities must be a mapping or ReasoningCapabilities")

    def validate(self, policy: Reasoning):
        if policy.enabled is False and self.can_disable is False:
            raise ValueError("This endpoint/model cannot disable reasoning")
        if (
            policy.enabled is True
            and policy.effort is None
            and policy.budget_tokens is None
            and self.can_enable is False
        ):
            raise ValueError("This endpoint/model does not support an explicit reasoning on switch")
        if policy.effort is not None and self.effort_levels is not None:
            if policy.effort not in self.effort_levels:
                raise ValueError(
                    f"Unsupported reasoning effort {policy.effort!r}; "
                    f"supported levels: {', '.join(self.effort_levels) or '(none)'}"
                )
        if policy.budget_tokens is not None:
            if self.budget_tokens is False:
                raise ValueError("This endpoint/model does not support reasoning budget_tokens")
            if isinstance(self.budget_tokens, TokenBudget):
                if not self.budget_tokens.min <= policy.budget_tokens <= self.budget_tokens.max:
                    raise ValueError(
                        f"reasoning budget_tokens must be between {self.budget_tokens.min} "
                        f"and {self.budget_tokens.max}"
                    )
        if policy.effort is not None and policy.budget_tokens is not None:
            if not self.supports_effort_and_budget:
                raise ValueError("This endpoint/model does not declare combined effort and budget")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ModelCapabilities:
    reasoning: ReasoningCapabilities


ADAPTERS = ("openai", "siliconflow", "deepseek", "vllm", "claude", "gemini")
NATIVE_REASONING_KEYS = frozenset(
    {"thinking", "reasoning_effort", "enable_thinking", "thinking_budget", "think"}
)


def resolve_adapter(adapter, provider, base_url):
    if adapter is not None:
        if adapter not in ADAPTERS:
            raise ValueError(f"reasoning_adapter must be one of {ADAPTERS}")
        allowed = (
            {"claude"}
            if provider == "claude"
            else {"gemini"}
            if provider == "gemini"
            else set(ADAPTERS) - {"claude", "gemini"}
        )
        if adapter not in allowed:
            raise ValueError(f"reasoning_adapter={adapter!r} does not match provider={provider!r}")
        return adapter
    if provider in ("claude", "gemini"):
        return provider
    host = urlparse(base_url or "").hostname
    if host in ("api.siliconflow.cn", "api.siliconflow.com"):
        return "siliconflow"
    if host == "api.deepseek.com":
        return "deepseek"
    return "openai"


def compile_reasoning(policy: Reasoning, adapter: str) -> dict:
    """Translate fields without inventing effort levels or budget conversions."""
    enabled, effort, budget = policy.enabled, policy.effort, policy.budget_tokens
    if not policy.to_dict():
        return {}
    if adapter == "openai":
        if budget is not None or (enabled is True and effort is None):
            raise ValueError(
                "OpenAI reasoning requires an effort level; no on switch or token budget"
            )
        return {"reasoning_effort": "none" if enabled is False else effort}
    if adapter in ("siliconflow", "deepseek", "vllm"):
        if budget is not None and adapter != "siliconflow":
            raise ValueError(f"{adapter} reasoning adapter does not support budget_tokens")
        on = enabled if enabled is not None else True
        params = {}
        if adapter == "siliconflow":
            params["enable_thinking"] = on
            if budget is not None:
                params["thinking_budget"] = budget
        elif adapter == "deepseek":
            params["thinking"] = {"type": "enabled" if on else "disabled"}
        else:
            params["chat_template_kwargs"] = {"enable_thinking": on}
        if effort is not None:
            params["reasoning_effort"] = effort
        return params
    if adapter == "claude":
        # Consumed by ClaudeClient before serialization: new policies use the
        # endpoint capability declaration, not the legacy model-name heuristics.
        params = {"_reasoning_validated": True}
        if enabled is False:
            params["thinking"] = {"type": "disabled"}
        elif budget is not None:
            params["thinking"] = {"type": "enabled", "budget_tokens": budget}
        else:
            params["thinking"] = {"type": "adaptive"}
        if effort is not None:
            params["output_config"] = {"effort": effort}
        return params
    if adapter == "gemini":
        if effort is not None and budget is not None:
            raise ValueError("Gemini reasoning adapter cannot combine effort and budget_tokens")
        config = {"includeThoughts": True}
        if enabled is False:
            config = {"thinkingBudget": 0}
        elif budget is not None:
            config["thinkingBudget"] = budget
        elif effort is not None:
            config["thinkingLevel"] = effort
        elif enabled is True:
            raise ValueError("Gemini reasoning requires an effort level or token budget")
        return {"thinking_config": config}
    raise ValueError(f"Unknown reasoning adapter: {adapter}")


def apply_reasoning(kwargs, *, default, capabilities, adapter):
    """Resolve an entire policy before cache lookup and request serialization."""
    if "reasoning" not in kwargs and default is None:
        return kwargs
    params = dict(kwargs)
    policy = Reasoning.parse(params.pop("reasoning", default))
    conflicts = set(NATIVE_REASONING_KEYS.intersection(params))
    for key, field in (("output_config", "effort"), ("chat_template_kwargs", "enable_thinking")):
        if isinstance(params.get(key), dict) and field in params[key]:
            conflicts.add(key)
    if "thinking_config" in params or "thinkingConfig" in params:
        conflicts.add("thinking_config")
    if conflicts:
        raise ValueError(
            "Do not combine reasoning with native controls: " + ", ".join(sorted(conflicts))
        )
    capabilities.validate(policy)
    for key, value in compile_reasoning(policy, adapter).items():
        params[key] = (
            {**params[key], **value} if isinstance(value, dict) and key in params else value
        )
    return params
