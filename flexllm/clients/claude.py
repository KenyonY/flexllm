"""
Anthropic Claude API Client

支持 Claude 系列模型（claude-3-opus, claude-3-sonnet, claude-3-haiku 等）
"""

import json
import logging
import re
from copy import deepcopy

import aiohttp

logger = logging.getLogger(__name__)

from ..async_api import create_proxied_session
from ..cache import ResponseCacheConfig
from .base import LLMClientBase, LLMRequestError, ToolCall, _decode_error_body

# Anthropic Messages 流式事件的全集（含本客户端不处理但属于规范的 ping / *_stop / error）。
# 这个集合之外的事件被当作带外信息透出，而不是静默丢弃。
_KNOWN_STREAM_EVENTS = frozenset(
    {
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "ping",
        "error",
    }
)

# Anthropic Messages 非流式响应的标准信封字段（对应 base 的 _OPENAI_ENVELOPE_KEYS）
_ANTHROPIC_ENVELOPE_KEYS = frozenset(
    {"id", "type", "role", "content", "model", "stop_reason", "stop_sequence", "usage"}
)

_CLAUDE_LEGACY_THINKING_BUDGETS = {
    "minimal": 1024,
    "low": 4000,
    "medium": 8000,
    "high": 16000,
    "xhigh": 32000,
    "max": 32000,
    "ultra": 32000,
}
_CLAUDE_ADAPTIVE_EFFORTS = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "max",
    "max": "max",
    "ultra": "max",
}


def _claude_version(model: str) -> tuple[int, int | None] | None:
    """Extract the Claude generation from common direct/Bedrock-style ids."""
    normalized = model.lower().replace("_", "-").replace(".", "-")
    match = re.search(
        r"claude-(?:opus-|sonnet-|haiku-|fable-|mythos-)?(\d+)(?:-(\d+))?",
        normalized,
    )
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)) if match.group(2) else None


def _claude_thinking_mode(model: str) -> str:
    """Return ``unsupported``, ``manual``, or ``adaptive`` for a Claude id."""
    if "claude-mythos-preview" in model.lower().replace("_", "-"):
        return "adaptive"
    version = _claude_version(model)
    if version is None:
        return "unsupported"
    major, minor = version
    if major == 3:
        return "manual" if minor == 7 else "unsupported"
    if major == 4:
        return "manual" if minor is None or minor <= 5 else "adaptive"
    return "adaptive" if major >= 5 else "unsupported"


def _allows_manual_thinking(model: str) -> bool:
    """Claude 4.6 still accepts manual budgets; 4.7+ rejects them."""
    mode = _claude_thinking_mode(model)
    if mode == "manual":
        return True
    return _claude_version(model) == (4, 6)


def _always_uses_adaptive_thinking(model: str) -> bool:
    """Current Fable/Mythos 5 models cannot disable adaptive thinking."""
    normalized = model.lower().replace("_", "-").replace(".", "-")
    return bool(
        "claude-mythos-preview" in normalized
        or re.search(r"claude-(?:fable|mythos)-5(?:-|$)", normalized)
    )


def _is_claude_model(model: str) -> bool:
    return "claude" in model.lower()


def _normalize_reasoning_effort(
    effort: str, *, adaptive: bool, model: str | None = None
) -> str | int:
    normalized = effort.strip().lower()
    mapping = dict(_CLAUDE_ADAPTIVE_EFFORTS) if adaptive else _CLAUDE_LEGACY_THINKING_BUDGETS
    version = _claude_version(model or "")
    if adaptive and (
        (version is not None and (version[0] >= 5 or version >= (4, 7)))
        and "mythos-preview" not in (model or "").lower()
    ):
        mapping["xhigh"] = "xhigh"
    try:
        return mapping[normalized]
    except KeyError as exc:
        supported = ", ".join(mapping)
        raise ValueError(
            f"Unsupported Claude reasoning effort {effort!r}; choose: {supported}"
        ) from exc


class ClaudeClient(LLMClientBase):
    """
    Anthropic Claude API 客户端

    Example:
        >>> client = ClaudeClient(
        ...     api_key="your-anthropic-key",
        ...     model="claude-3-5-sonnet-20241022",
        ... )
        >>> result = await client.chat_completions(messages)

    Example (thinking 参数 - 扩展思考模式):
        >>> # 启用扩展思考
        >>> result = client.chat_completions_sync(
        ...     messages=[{"role": "user", "content": "复杂推理问题"}],
        ...     thinking=True,
        ...     return_raw=True,
        ... )
        >>> parsed = ClaudeClient.parse_thoughts(result.data)
        >>> print("思考:", parsed["thought"])
        >>> print("答案:", parsed["answer"])

    thinking 参数值:
        - False: 禁用扩展思考（不支持 thinking 的旧模型等价于省略参数）
        - True: Claude 4.6+ 使用 adaptive thinking，3.7/4.0-4.5 使用默认 token 预算
        - str: 设置思考强度（minimal/low/medium/high/xhigh/max/ultra）
        - int: 为旧版 Claude 启用扩展思考并指定 budget_tokens
        - dict: 直接使用 Anthropic 原生 thinking 配置
        - None: 使用模型默认行为
    """

    DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
    DEFAULT_API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str,
        model: str = None,
        base_url: str = None,
        api_version: str = None,
        concurrency_limit: int = 10,
        max_qps: int = 60,
        timeout: int = 120,
        retry_times: int = 3,
        retry_delay: float = 1.0,
        cache_image: bool = False,
        cache_dir: str | None = None,
        cache: ResponseCacheConfig | None = None,
        **kwargs,
    ):
        self._api_version = api_version or self.DEFAULT_API_VERSION

        super().__init__(
            base_url=base_url or self.DEFAULT_BASE_URL,
            api_key=api_key,
            model=model,
            concurrency_limit=concurrency_limit,
            max_qps=max_qps,
            timeout=timeout,
            retry_times=retry_times,
            retry_delay=retry_delay,
            cache_image=cache_image,
            cache_dir=cache_dir,
            cache=cache,
            **kwargs,
        )

    # ========== 实现基类核心方法 ==========

    def _get_url(self, model: str, stream: bool = False) -> str:
        return f"{self._base_url}/messages"

    def _is_oauth_token(self) -> bool:
        return isinstance(self._api_key, str) and "sk-ant-oat" in self._api_key

    def _get_headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self._api_version,
        }
        if self._is_oauth_token():
            headers["Authorization"] = f"Bearer {self._api_key}"
            headers["anthropic-beta"] = ",".join(
                [
                    "oauth-2025-04-20",
                    "claude-code-20250219",
                    "fine-grained-tool-streaming-2025-05-14",
                    "interleaved-thinking-2025-05-14",
                ]
            )
        else:
            headers["x-api-key"] = self._api_key
        return headers

    def _build_request_body(
        self,
        messages: list[dict],
        model: str,
        stream: bool = False,
        max_tokens: int = 4096,  # Claude 必需参数
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        thinking: bool | str | int | dict | None = None,
        reasoning_effort: str | None = None,
        response_format: dict = None,
        **kwargs,
    ) -> dict:
        """
        构建 Claude API 请求体

        Args:
            thinking: 扩展思考控制参数
                - False: 禁用扩展思考
                - True: 启用扩展思考（默认 budget_tokens=10000）
                - str: 使用统一强度（minimal/low/medium/high/xhigh/max/ultra）
                - int: 启用扩展思考并指定 budget_tokens（旧版 Claude）
                - dict: 直接使用 Anthropic 原生 thinking 配置
                - None: 使用模型默认行为

                Claude 4.6+ 的 True/str 会转换为 adaptive thinking；旧版 Claude
                使用 budget_tokens。reasoning_effort 是 str 强度的等价统一入口。

                Claude API 要求 max_tokens > thinking.budget_tokens。启用思考且
                max_tokens ≤ budget_tokens 时（含默认 max_tokens=4096 的情况），
                自动抬高 max_tokens = budget_tokens + 4096，保证请求合法且
                思考后仍有输出空间。
            response_format: 响应格式控制
                - {"type": "json_object"}: 输出 JSON
                - {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}:
                  按 JSON schema 输出
                Claude 不原生支持 response_format，通过 system prompt 注入 JSON 指令实现
        """
        # 分离 system message
        system_content = None
        user_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                # 合并多个 system messages
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") for p in content if p.get("type") == "text"
                    )
                system_content = (system_content + "\n" + content) if system_content else content
            else:
                user_messages.append(self._convert_message(msg))

        # response_format: 通过 system prompt 注入 JSON 指令
        if response_format:
            json_instruction = self._build_json_instruction(response_format)
            if json_instruction:
                system_content = (
                    (system_content + "\n\n" + json_instruction)
                    if system_content
                    else json_instruction
                )

        is_claude = _is_claude_model(model)
        thinking_mode = _claude_thinking_mode(model) if is_claude else "unsupported"
        adaptive = thinking_mode == "adaptive"
        allows_manual = _allows_manual_thinking(model) if is_claude else False
        always_adaptive = _always_uses_adaptive_thinking(model) if is_claude else False
        effort = thinking if isinstance(thinking, str) else reasoning_effort
        native_thinking = dict(thinking) if isinstance(thinking, dict) else None
        if effort is not None and not is_claude:
            # Anthropic-compatible endpoints may expose other model families.
            # Preserve their explicit wire parameter instead of applying Claude budgets.
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            effort = None

        control_requested = (
            thinking is True
            or isinstance(thinking, str)
            or (isinstance(thinking, int) and not isinstance(thinking, bool) and thinking > 0)
            or (native_thinking is not None and native_thinking.get("type") != "disabled")
            or reasoning_effort is not None
        )
        if is_claude and thinking_mode == "unsupported" and control_requested:
            raise ValueError(
                f"Claude model {model!r} does not support extended thinking; "
                "use Claude 3.7 or Claude 4+"
            )
        if (
            is_claude
            and thinking_mode == "unsupported"
            and native_thinking is not None
            and native_thinking.get("type") == "disabled"
        ):
            native_thinking = None
        if (
            is_claude
            and always_adaptive
            and (
                thinking is False
                or (native_thinking is not None and native_thinking.get("type") == "disabled")
            )
        ):
            raise ValueError(f"Claude model {model!r} has always-on adaptive thinking")
        if (
            is_claude
            and not allows_manual
            and (
                (isinstance(thinking, int) and not isinstance(thinking, bool) and thinking > 0)
                or (native_thinking is not None and native_thinking.get("type") == "enabled")
            )
        ):
            raise ValueError(
                f"Claude model {model!r} requires adaptive thinking and does not accept "
                "manual budget_tokens"
            )

        # Claude 扩展思考模式：旧版 API 要求 max_tokens > budget_tokens，
        # 不满足时自动抬高 max_tokens = budget_tokens + 4096（见 docstring）。
        budget_tokens = None
        if is_claude and thinking is True:
            if adaptive:
                native_thinking = {"type": "adaptive"}
            else:
                budget_tokens = 10000
        elif (
            is_claude
            and isinstance(thinking, int)
            and not isinstance(thinking, bool)
            and thinking > 0
        ):
            budget_tokens = thinking
        elif is_claude and effort is not None and (thinking is None or isinstance(thinking, str)):
            normalized_effort = _normalize_reasoning_effort(effort, adaptive=adaptive, model=model)
            if adaptive:
                native_thinking = {"type": "adaptive"}
            else:
                budget_tokens = normalized_effort

        if native_thinking and native_thinking.get("type") == "enabled":
            native_budget = native_thinking.get("budget_tokens")
            if isinstance(native_budget, int) and native_budget > 0:
                budget_tokens = native_budget
        if budget_tokens is not None and max_tokens <= budget_tokens:
            max_tokens = budget_tokens + 4096

        body = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": user_messages,
        }

        if system_content:
            body["system"] = system_content
        if stream:
            body["stream"] = True
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if top_k is not None:
            body["top_k"] = top_k

        if native_thinking is not None:
            body["thinking"] = native_thinking
        elif budget_tokens is not None:
            body["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
        elif thinking is False and thinking_mode != "unsupported":
            body["thinking"] = {"type": "disabled"}

        if adaptive and effort is not None and body.get("thinking", {}).get("type") == "adaptive":
            output_config = dict(kwargs.pop("output_config", {}) or {})
            output_config.setdefault(
                "effort", _normalize_reasoning_effort(effort, adaptive=True, model=model)
            )
            body["output_config"] = output_config

        # 透传其他参数，但排除 response_format（已通过 prompt 注入）
        kwargs.pop("response_format", None)

        # 将 OpenAI 格式的 tools 转换为 Claude 格式
        if "tools" in kwargs:
            kwargs["tools"] = self._convert_tools(kwargs["tools"])

        body.update(kwargs)
        return body

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        """将 OpenAI 格式的 tools 转换为 Claude 格式

        OpenAI: [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]
        Claude: [{"name": ..., "description": ..., "input_schema": ...}]
        """
        converted = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                converted.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    }
                )
            else:
                # 已经是 Claude 格式，直接透传
                converted.append(tool)
        return converted

    @staticmethod
    def _build_json_instruction(response_format: dict) -> str | None:
        """将 response_format 转换为 system prompt 中的 JSON 输出指令"""
        fmt_type = response_format.get("type", "")
        if fmt_type == "json_object":
            return "You must respond with valid JSON only. No other text or explanation."
        elif fmt_type == "json_schema":
            schema = response_format.get("json_schema", {}).get("schema")
            if schema:
                schema_str = json.dumps(schema, ensure_ascii=False)
                return (
                    f"You must respond with valid JSON that conforms to this schema:\n"
                    f"{schema_str}\n"
                    f"Output only the JSON object, no other text."
                )
        return None

    def _convert_message(self, msg: dict) -> dict:
        """转换消息格式（处理多模态内容、tool_calls、tool result）"""
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # tool result 消息 → Claude tool_result content block
        if role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": content,
                    }
                ],
            }

        # flexllm 的响应续接消息会保留 Claude 原生 content blocks。带签名的
        # thinking/redacted_thinking 必须逐字节回传；若再从统一 tool_calls 重建，
        # 会丢签名并重复 tool_use。
        if (
            role == "assistant"
            and isinstance(content, list)
            and any(
                isinstance(item, dict)
                and item.get("type") in {"thinking", "redacted_thinking", "tool_use"}
                for item in content
            )
        ):
            return {"role": "assistant", "content": deepcopy(content)}

        # assistant 消息中包含 tool_calls → Claude tool_use content block
        if role == "assistant" and msg.get("tool_calls"):
            claude_content = []
            if content:
                claude_content.append({"type": "text", "text": content})
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                arguments = func.get("arguments", "{}")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (json.JSONDecodeError, TypeError):
                        arguments = {}
                claude_content.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "input": arguments,
                    }
                )
            return {"role": "assistant", "content": claude_content}

        # Claude 格式: role 只能是 "user" 或 "assistant"
        claude_role = "assistant" if role == "assistant" else "user"

        # 处理多模态内容
        if isinstance(content, list):
            claude_content = []
            for item in content:
                if isinstance(item, str):
                    claude_content.append({"type": "text", "text": item})
                elif isinstance(item, dict):
                    item_type = item.get("type", "text")
                    if item_type == "text":
                        claude_content.append({"type": "text", "text": item.get("text", "")})
                    elif item_type == "image_url":
                        # 转换 OpenAI 图片格式到 Claude 格式
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            # base64 格式
                            match = re.match(r"data:([^;]+);base64,(.+)", url)
                            if match:
                                claude_content.append(
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": match.group(1),
                                            "data": match.group(2),
                                        },
                                    }
                                )
                        else:
                            # URL 格式
                            claude_content.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": url,
                                    },
                                }
                            )
                    elif item_type in ("video_url", "audio_url"):
                        # 转换视频/音频到 Claude document 格式
                        media_key = item_type  # "video_url" 或 "audio_url"
                        url = item.get(media_key, {}).get("url", "")
                        if url.startswith("data:"):
                            match = re.match(r"data:([^;]+);base64,(.+)", url)
                            if match:
                                claude_content.append(
                                    {
                                        "type": "document",
                                        "source": {
                                            "type": "base64",
                                            "media_type": match.group(1),
                                            "data": match.group(2),
                                        },
                                    }
                                )
                        else:
                            claude_content.append(
                                {
                                    "type": "document",
                                    "source": {"type": "url", "url": url},
                                }
                            )
                    elif item_type == "input_audio":
                        # 转换 OpenAI input_audio 到 Claude document 格式
                        audio_data = item.get("input_audio", {})
                        data = audio_data.get("data", "")
                        fmt = audio_data.get("format", "wav")
                        if data:
                            claude_content.append(
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": f"audio/{fmt}",
                                        "data": data,
                                    },
                                }
                            )
            return {"role": claude_role, "content": claude_content}

        return {"role": claude_role, "content": content}

    def _extract_content(self, response_data: dict, **gen_kwargs) -> str | None:
        """提取 Claude 响应中的文本内容"""
        try:
            content_blocks = response_data.get("content", [])
            texts = []
            for block in content_blocks:
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
            return "".join(texts) if texts else None
        except Exception as e:
            logger.warning(f"Failed to extract content: {e}")
            return None

    def _extract_reasoning_content(self, response_data: dict) -> str | None:
        thoughts = [
            block.get("thinking", "")
            for block in (response_data or {}).get("content", [])
            if block.get("type") == "thinking"
        ]
        return "".join(thoughts) or None

    def _extract_assistant_message(self, response_data: dict) -> dict | None:
        content = (response_data or {}).get("content")
        if not isinstance(content, list) or not any(
            isinstance(block, dict)
            and block.get("type") in {"thinking", "redacted_thinking", "tool_use"}
            for block in content
        ):
            return None
        message = {"role": "assistant", "content": deepcopy(content)}
        tool_calls = self._extract_tool_calls(response_data)
        if tool_calls:
            message["tool_calls"] = [
                {"id": call.id, "type": call.type, "function": deepcopy(call.function)}
                for call in tool_calls
            ]
        return message

    def _extract_extra(self, data: dict) -> dict | None:
        """Anthropic 的响应信封与 OpenAI 不同，不能用基类那套字段名判定。"""
        extra = {k: v for k, v in data.items() if k not in _ANTHROPIC_ENVELOPE_KEYS}
        return extra or None

    def _extract_usage(self, response_data: dict) -> dict | None:
        """提取 Claude usage 信息并转换为统一格式"""
        if not response_data:
            return None
        usage = response_data.get("usage")
        if not usage:
            return None
        return {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        }

    _STOP_REASON_MAP = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "refusal": "content_filter",
        # 4.5+ 模型：input + 生成触到窗口上限，语义上同样是被截断
        "model_context_window_exceeded": "length",
    }

    def _extract_finish_reason(self, response_data: dict) -> str | None:
        """Claude stop_reason → OpenAI 语义"""
        reason = (response_data or {}).get("stop_reason")
        if not reason:
            return None
        return self._STOP_REASON_MAP.get(reason, reason)

    def _extract_tool_calls(self, response_data: dict) -> list[ToolCall] | None:
        """提取 Claude tool_use 信息"""
        try:
            content_blocks = response_data.get("content", [])
            tool_calls = []
            for block in content_blocks:
                if block.get("type") == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.get("id", ""),
                            type="function",
                            function={
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        )
                    )
            return tool_calls if tool_calls else None
        except Exception:
            return None

    # ========== 流式响应 ==========
    # ClaudeClient 完整覆写 chat_completions_stream，
    # 基类的 _extract_stream_* / _prepare_stream_body 钩子不会被调用，无需覆写

    async def chat_completions_stream(
        self,
        messages: list[dict],
        model: str = None,
        return_usage: bool = False,
        preprocess_msg: bool = False,
        url: str = None,
        timeout: int = None,
        extra_headers: dict[str, str] | None = None,
        **kwargs,
    ):
        """Claude 流式聊天完成"""
        effective_model = self._get_effective_model(model)
        messages = await self._preprocess_messages(messages, preprocess_msg)

        body = self._build_request_body(messages, effective_model, stream=True, **kwargs)
        effective_url = url or self._get_url(effective_model, stream=True)
        headers = self._merge_headers(extra_headers)

        effective_timeout = timeout if timeout is not None else self._timeout
        # 流式：空闲超时语义（见基类 chat_completions_stream 说明）
        aio_timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=min(30, effective_timeout) if effective_timeout else None,
            sock_read=effective_timeout,
        )

        session, proxy_kwargs = create_proxied_session(self._proxy)
        async with session:
            async with session.post(
                effective_url,
                json=body,
                headers=headers,
                timeout=aio_timeout,
                **proxy_kwargs,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMRequestError(
                        f"HTTP {response.status}: {error_text}",
                        status_code=response.status,
                        response_data=_decode_error_body(error_text),
                    )

                usage_data = None
                finish_reason = None
                # 同时保留 UI 所需增量和下一轮必须原样回传的 Claude content blocks。
                content_blocks: dict[int, dict] = {}
                tool_use_blocks = {}  # {block_index: {"id", "name", "arguments"}}
                current_block_index = -1

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            event_type = data.get("type")

                            # Anthropic 规范要求客户端忽略未知事件类型，网关正是靠这一点
                            # 挂带外信息。这里不解释内容，原样透出给调用方。
                            if return_usage and event_type not in _KNOWN_STREAM_EVENTS:
                                yield {"type": "extra", "extra": data}
                                continue

                            # content_block_start: 新 block 开始
                            if event_type == "content_block_start":
                                current_block_index = data.get("index", 0)
                                block = data.get("content_block", {})
                                if isinstance(block, dict):
                                    content_blocks[current_block_index] = deepcopy(block)
                                if block.get("type") == "tool_use":
                                    tool_use_blocks[current_block_index] = {
                                        "id": block.get("id", ""),
                                        "name": block.get("name", ""),
                                        "arguments": "",
                                    }
                                    if return_usage:
                                        yield {
                                            "type": "tool_call_delta",
                                            "tool_calls": [
                                                {
                                                    "index": current_block_index,
                                                    "id": block.get("id", ""),
                                                    "type": "function",
                                                    "function": {
                                                        "name": block.get("name", ""),
                                                        "arguments": "",
                                                    },
                                                }
                                            ],
                                        }
                                continue

                            # content_block_delta
                            if event_type == "content_block_delta":
                                idx = data.get("index", current_block_index)
                                delta = data.get("delta", {})
                                delta_type = delta.get("type")

                                # 思考内容
                                if delta_type == "thinking_delta":
                                    thinking = delta.get("thinking")
                                    if thinking and idx in content_blocks:
                                        content_blocks[idx]["thinking"] = (
                                            content_blocks[idx].get("thinking", "") + thinking
                                        )
                                    if thinking and return_usage:
                                        yield {"type": "thinking", "content": thinking}
                                    continue

                                # thinking block 的签名是下一轮工具调用请求的必需状态。
                                if delta_type == "signature_delta":
                                    signature = delta.get("signature", "")
                                    if signature and idx in content_blocks:
                                        content_blocks[idx]["signature"] = (
                                            content_blocks[idx].get("signature", "") + signature
                                        )
                                    continue

                                # tool_use 的 input_json_delta
                                if delta_type == "input_json_delta" and idx in tool_use_blocks:
                                    partial = delta.get("partial_json", "")
                                    tool_use_blocks[idx]["arguments"] += partial
                                    if return_usage:
                                        yield {
                                            "type": "tool_call_delta",
                                            "tool_calls": [
                                                {
                                                    "index": idx,
                                                    "function": {"arguments": partial},
                                                }
                                            ],
                                        }
                                    continue

                                # 文本内容
                                if delta_type == "text_delta":
                                    text = delta.get("text")
                                    if text:
                                        if idx in content_blocks:
                                            content_blocks[idx]["text"] = (
                                                content_blocks[idx].get("text", "") + text
                                            )
                                        if return_usage:
                                            yield {"type": "content", "content": text}
                                        else:
                                            yield text
                                continue

                            # message_delta 中的 usage：只带 output_tokens（增量事件
                            # 不含 input_tokens），必须与 message_start 的记录合并，
                            # 整体覆盖会把 prompt_tokens 归零
                            if event_type == "message_delta":
                                delta_reason = self._extract_finish_reason(data.get("delta"))
                                if delta_reason:
                                    finish_reason = delta_reason
                                usage = data.get("usage")
                                if usage:
                                    prompt_tokens = usage.get("input_tokens") or (
                                        usage_data.get("prompt_tokens", 0) if usage_data else 0
                                    )
                                    completion_tokens = usage.get("output_tokens", 0)
                                    usage_data = {
                                        "prompt_tokens": prompt_tokens,
                                        "completion_tokens": completion_tokens,
                                        "total_tokens": prompt_tokens + completion_tokens,
                                    }

                            # message_start 中的 usage（输入 tokens）
                            if event_type == "message_start":
                                msg_usage = data.get("message", {}).get("usage", {})
                                if msg_usage:
                                    usage_data = {
                                        "prompt_tokens": msg_usage.get("input_tokens", 0),
                                        "completion_tokens": msg_usage.get("output_tokens", 0),
                                        "total_tokens": msg_usage.get("input_tokens", 0)
                                        + msg_usage.get("output_tokens", 0),
                                    }

                        except json.JSONDecodeError:
                            continue

                if return_usage:
                    continuation_blocks = [
                        deepcopy(block) for _, block in sorted(content_blocks.items())
                    ]
                    if any(
                        block.get("type") in {"thinking", "redacted_thinking", "tool_use"}
                        for block in continuation_blocks
                    ):
                        for idx, block in sorted(content_blocks.items()):
                            if block.get("type") != "tool_use" or idx not in tool_use_blocks:
                                continue
                            arguments = tool_use_blocks[idx]["arguments"]
                            if arguments:
                                try:
                                    block["input"] = json.loads(arguments)
                                except json.JSONDecodeError:
                                    # 保留不完整 JSON 只能制造一个下一轮必然失败的请求；
                                    # 流结束时工具参数应完整，因此明确暴露协议错误。
                                    raise LLMRequestError(
                                        f"Invalid streamed Claude tool input: {arguments!r}"
                                    )
                        continuation_blocks = [
                            deepcopy(block) for _, block in sorted(content_blocks.items())
                        ]
                        assistant_message = {
                            "role": "assistant",
                            "content": continuation_blocks,
                        }
                        if tool_use_blocks:
                            assistant_message["tool_calls"] = [
                                {
                                    "id": block["id"],
                                    "type": "function",
                                    "function": {
                                        "name": block["name"],
                                        "arguments": block["arguments"],
                                    },
                                }
                                for _, block in sorted(tool_use_blocks.items())
                            ]
                        yield {
                            "type": "assistant_message",
                            "message": assistant_message,
                        }
                    yield {"type": "finish", "reason": finish_reason}
                    if usage_data:
                        yield {"type": "usage", "usage": usage_data}

    @staticmethod
    def parse_thoughts(response_data: dict) -> dict:
        """
        从响应中解析思考内容和答案

        当使用 thinking=True 时，可以用此方法解析响应。

        Args:
            response_data: 原始响应数据（通过 return_raw=True 获取）

        Returns:
            dict: {
                "thought": str,  # 思考过程（可能为空）
                "answer": str,   # 最终答案
            }
        """
        try:
            content_blocks = response_data.get("content", [])
            thoughts = []
            answers = []

            for block in content_blocks:
                block_type = block.get("type", "")
                if block_type == "thinking":
                    thoughts.append(block.get("thinking", ""))
                elif block_type == "text":
                    answers.append(block.get("text", ""))

            return {
                "thought": "\n".join(thoughts),
                "answer": "".join(answers),
            }
        except Exception as e:
            logger.warning(f"Failed to parse thoughts: {e}")
            return {"thought": "", "answer": ""}

    # ========== Claude 特有方法 ==========

    def model_list(self) -> list[str]:
        """返回 Claude 模型列表（静态）"""
        return [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
        ]
