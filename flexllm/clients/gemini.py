"""
Gemini API Client - Google Gemini 模型的批量调用客户端

与 OpenAIClient 保持相同的接口，方便上层代码无缝切换。
"""

import asyncio
import json
import logging
import re
from copy import deepcopy
from typing import Any

logger = logging.getLogger(__name__)

from ..async_api import create_proxied_session
from ..cache import ResponseCacheConfig
from .base import (
    LLMClientBase,
    LLMConnectionError,
    LLMHTTPError,
    LLMTimeoutError,
    _decode_error_body,
)
from .message_images import has_non_text_parts, move_tool_images_to_user

# Gemini 响应的标准信封字段；之外的顶层字段才是带外信息（见 ChatCompletionResult.extra）
_GEMINI_ENVELOPE_KEYS = frozenset(
    {"candidates", "usageMetadata", "modelVersion", "responseId", "promptFeedback", "createTime"}
)
_THINKING_LEVELS = ("minimal", "low", "medium", "high")


def _is_gemini_2(model: str) -> bool:
    """Gemini 2.x 与 3 的协议差异（thinkingLevel、多模态 functionResponse）按主版本分派。
    认不出版本的名字（自定义部署、mock）按当前代处理。"""
    match = re.search(r"gemini-(\d+)", model or "")
    return match is not None and int(match.group(1)) < 3


# 注入的 functionCall（从统一 tool_calls 重建、没有模型签发的签名）用的占位签名。
# Gemini 3 对当前轮缺签名的 functionCall 直接 400；这个值是官方文档给出的跳过校验方式，
# 代价是模型看不到自己之前的推理上下文。能拿到原生 parts 时总是优先原样回传。
_SKIP_SIGNATURE = "skip_thought_signature_validator"


class GeminiClient(LLMClientBase):
    """
    Google Gemini API 客户端

    支持 Gemini Developer API 和 Vertex AI。

    Example (Gemini Developer API):
        >>> client = GeminiClient(api_key="your-key", model="gemini-3-flash-preview")
        >>> result = await client.complete(messages)

    Example (Vertex AI):
        >>> client = GeminiClient(
        ...     project_id="your-project-id",
        ...     location="us-central1",
        ...     model="gemini-3-flash-preview",
        ...     use_vertex_ai=True,
        ... )

    Example (thinking 参数 - 统一的思考控制):
        >>> # 禁用思考（最快响应）
        >>> result = client.complete_sync(
        ...     messages=[{"role": "user", "content": "1+1=?"}],
        ...     thinking=False,
        ... )
        >>> # 启用思考并返回思考内容
        >>> result = client.complete_sync(
        ...     messages=[{"role": "user", "content": "复杂推理问题"}],
        ...     thinking=True,
        ...     return_raw=True,
        ... )
        >>> parsed = GeminiClient.parse_thoughts(result.data)

    thinking 参数值:
        - False: 禁用思考（thinkingBudget=0）
        - True: 启用思考并返回思考内容（includeThoughts=True）
        - "minimal"/"low"/"medium"/"high": 设置思考深度（Gemini 2.x 换算为预算；xhigh/max/ultra → high）
        - int: 思考 token 预算（thinkingBudget）
        - None: 使用模型默认行为
    """

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    VERTEX_AI_URL_TEMPLATE = "https://{location}-aiplatform.googleapis.com/v1"

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        base_url: str = None,
        concurrency_limit: int = 10,
        max_qps: int = 60,
        timeout: int = 120,
        retry_times: int = 3,
        retry_delay: float = 1.0,
        cache_image: bool = False,
        cache_dir: str | None = None,
        cache: ResponseCacheConfig | None = None,
        use_vertex_ai: bool = False,
        project_id: str = None,
        location: str = "us-central1",
        credentials: Any = None,
        **kwargs,
    ):
        self._use_vertex_ai = use_vertex_ai
        self._project_id = project_id
        self._location = location
        self._credentials = credentials
        self._access_token = None
        self._token_expiry = None

        if use_vertex_ai:
            if not project_id:
                raise ValueError("Vertex AI 模式需要提供 project_id")
            effective_base_url = base_url or self.VERTEX_AI_URL_TEMPLATE.format(location=location)
        else:
            if not api_key:
                raise ValueError("Gemini Developer API 模式需要提供 api_key")
            effective_base_url = base_url or self.DEFAULT_BASE_URL

        super().__init__(
            base_url=effective_base_url,
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
        action = "streamGenerateContent" if stream else "generateContent"
        if self._use_vertex_ai:
            return (
                f"{self._base_url}/projects/{self._project_id}"
                f"/locations/{self._location}/publishers/google/models/{model}:{action}"
            )
        return f"{self._base_url}/models/{model}:{action}?key={self._api_key}"

    def _get_stream_url(self, model: str) -> str:
        """Gemini 流式需要添加 alt=sse 参数"""
        url = self._get_url(model, stream=True)
        return url + ("&alt=sse" if "?" in url else "?alt=sse")

    def _get_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._use_vertex_ai:
            headers["Authorization"] = f"Bearer {self._get_access_token()}"
        return headers

    def _build_request_body(
        self,
        messages: list[dict],
        model: str,
        stream: bool = False,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        stop_sequences: list[str] = None,
        safety_settings: list[dict] = None,
        thinking: bool | str | None = None,
        response_format: dict = None,
        tools: list[dict] = None,
        tool_choice: str | dict = None,
        **kwargs,
    ) -> dict:
        """
        构建请求体

        Args:
            thinking: 统一的思考控制参数
                - False: 禁用思考（thinkingBudget=0；Pro 系列不能关思考，API 会拒绝）
                - True: 启用思考并返回思考内容（includeThoughts=True）
                - "minimal"/"low"/"medium"/"high": 设置思考深度（Gemini 3 用 thinkingLevel；
                  Gemini 2.x 不支持 level，换算成 thinkingBudget）；"xhigh"/"max"/"ultra" 取最高档
                - int: 思考 token 预算（thinkingBudget，Gemini 2.5 与 3 均支持）
                - None: 使用模型默认行为
            response_format: 响应格式控制
                - {"type": "json_object"}: 输出 JSON
                - {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}:
                  按 JSON schema 输出（转换为 responseJsonSchema：它收完整 JSON Schema，
                  responseSchema 只收 OpenAPI 子集，strict 模式必带的 additionalProperties 会被 400）
            tools: OpenAI 格式（{"type": "function", ...}）转为 functionDeclarations；
                Gemini 原生格式原样透传
            tool_choice: OpenAI 语义（"auto"/"none"/"required"/指定函数）转为 toolConfig
        """
        messages = self._vision_messages(messages)
        if _is_gemini_2(model):
            # 2.x 不支持多模态 functionResponse（400），而图片与 functionResponse 同一条消息时
            # 模型看不见它（实测）；只有单独一条紧随其后的 user 消息能被看到
            messages = move_tool_images_to_user(messages)
        contents, system_obj = self._convert_messages_to_contents(messages)
        body = {"contents": contents}

        if system_obj:
            body["systemInstruction"] = system_obj
        if tools:
            body["tools"] = self._convert_tools(tools)
        if tool_choice is not None:
            body["toolConfig"] = {"functionCallingConfig": self._convert_tool_choice(tool_choice)}

        gen_config = {}
        if max_tokens is not None:
            gen_config["maxOutputTokens"] = max_tokens
        if temperature is not None:
            gen_config["temperature"] = temperature
        if top_p is not None:
            gen_config["topP"] = top_p
        if top_k is not None:
            gen_config["topK"] = top_k
        if stop_sequences:
            gen_config["stopSequences"] = stop_sequences

        # response_format 转换为 Gemini 格式
        if response_format:
            fmt_type = response_format.get("type", "")
            if fmt_type == "json_object":
                gen_config["responseMimeType"] = "application/json"
            elif fmt_type == "json_schema":
                gen_config["responseMimeType"] = "application/json"
                schema = response_format.get("json_schema", {}).get("schema")
                if schema:
                    gen_config["responseJsonSchema"] = schema

        # 构建 thinkingConfig
        # 实测：thinkingLevel 在 Gemini 2.5 上 400，而 thinkingBudget=0 在 2.5/3 的 Flash 上
        # 都能关思考（minimal 在 2.5 上不行）；level 只认 minimal/low/medium/high
        thinking_config = {}
        if thinking is False:
            thinking_config["thinkingBudget"] = 0
        elif thinking is True:
            thinking_config["includeThoughts"] = True
        elif isinstance(thinking, int):
            thinking_config["thinkingBudget"] = thinking
            thinking_config["includeThoughts"] = True
        elif isinstance(thinking, str):
            level = "high" if thinking in ("xhigh", "max", "ultra") else thinking
            if level not in _THINKING_LEVELS:
                raise ValueError(
                    f"Gemini 不支持的 thinking 级别: {thinking!r}，"
                    "可选 minimal/low/medium/high/xhigh/max/ultra、bool 或 int 预算"
                )
            if _is_gemini_2(model):
                # 2.5 不认 thinkingLevel（实测 400），按级别换算成预算
                thinking_config["thinkingBudget"] = self._GEMINI_2_BUDGETS[level]
            else:
                thinking_config["thinkingLevel"] = level
            thinking_config["includeThoughts"] = True
        # thinking=None 时不设置，使用默认行为

        if thinking_config:
            gen_config["thinkingConfig"] = thinking_config

        # 透传其余 kwargs（与 OpenAI/Claude 客户端行为一致，不静默丢弃）：
        # 顶层键直接进 body，生成参数（snake_case 自动转 camelCase）进 generationConfig，
        # 无法映射的丢弃并给出 warning
        for key, value in kwargs.items():
            camel_key = self._snake_to_camel(key)
            if camel_key in self._TOP_LEVEL_KEYS:
                body[camel_key] = value
            elif camel_key in self._GENERATION_CONFIG_KEYS:
                gen_config[camel_key] = value
            else:
                logger.warning(f"GeminiClient 无法映射参数 {key!r} 到 Gemini API，已忽略")

        if gen_config:
            body["generationConfig"] = gen_config
        if safety_settings:
            body["safetySettings"] = safety_settings

        return body

    # Gemini 2.5 的级别→预算（Flash 上限 24576）
    _GEMINI_2_BUDGETS = {"minimal": 512, "low": 2048, "medium": 8192, "high": 24576}

    # Gemini generateContent 请求体的合法键（用于 **kwargs 透传映射）
    _TOP_LEVEL_KEYS = frozenset(
        {"tools", "toolConfig", "safetySettings", "systemInstruction", "cachedContent", "labels"}
    )
    _GENERATION_CONFIG_KEYS = frozenset(
        {
            "candidateCount",
            "stopSequences",
            "maxOutputTokens",
            "temperature",
            "topP",
            "topK",
            "seed",
            "presencePenalty",
            "frequencyPenalty",
            "responseLogprobs",
            "logprobs",
            "responseMimeType",
            "responseSchema",
            "responseJsonSchema",
            "responseModalities",
            "thinkingConfig",
            "mediaResolution",
            "speechConfig",
        }
    )

    @staticmethod
    def _snake_to_camel(name: str) -> str:
        """snake_case → camelCase（已是 camelCase 的原样返回）"""
        if "_" not in name:
            return name
        head, *rest = name.split("_")
        return head + "".join(part.capitalize() for part in rest)

    def _extract_content(self, response_data: dict, **gen_kwargs) -> str | None:
        try:
            candidates = response_data.get("candidates", [])
            if not candidates:
                if "promptFeedback" in response_data:
                    block_reason = response_data["promptFeedback"].get("blockReason", "UNKNOWN")
                    logger.warning(f"Request blocked by Gemini: {block_reason}")
                return None

            parts = candidates[0].get("content", {}).get("parts", [])
            # 只提取非 thought 部分的文本（即最终答案）
            texts = [p.get("text", "") for p in parts if "text" in p and not p.get("thought")]
            return "".join(texts) if texts else None
        except Exception as e:
            logger.warning(f"Failed to extract response text: {e}")
            return None

    _FINISH_REASON_MAP = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "PROHIBITED_CONTENT": "content_filter",
        "BLOCKLIST": "content_filter",
        "SPII": "content_filter",
    }

    def _extract_finish_reason(self, response_data: dict) -> str | None:
        """Gemini candidates[0].finishReason → OpenAI 语义（非流式与流式 chunk 结构相同）

        Gemini 发起工具调用时 finishReason 仍是 STOP；OpenAI 语义下这是 "tool_calls"，
        调用方靠它判断要不要执行工具。流式的 functionCall 与 finishReason 不在同一个
        chunk，那条路径在流末尾单独修正。
        """
        candidates = (response_data or {}).get("candidates")
        if not candidates:
            if (response_data or {}).get("promptFeedback", {}).get("blockReason"):
                return "content_filter"
            return None
        reason = candidates[0].get("finishReason")
        if not reason:
            return None
        mapped = self._FINISH_REASON_MAP.get(reason, reason.lower())
        if mapped == "stop" and any("functionCall" in p for p in self._parts(response_data)):
            return "tool_calls"
        return mapped

    @staticmethod
    def _parts(response_data: dict) -> list[dict]:
        candidates = (response_data or {}).get("candidates")
        if not candidates:
            return []
        return candidates[0].get("content", {}).get("parts", [])

    @staticmethod
    def _needs_continuation(parts: list) -> bool:
        """parts 里有下一轮必须原样回传的状态：签名或函数调用。

        提取（是否产出 assistant_message）与回传（是否按原生 parts 透传）共用这一个判定，
        两边不一致会让原生 parts 被当成 OpenAI content 转换——thought part 丢掉 thought
        标记，思考摘要被当成说过的话发回去。纯思考摘要不是续接状态，官方也不要求回传。
        """
        return any(
            isinstance(p, dict) and ("functionCall" in p or "thoughtSignature" in p) for p in parts
        )

    def _extract_extra(self, data: dict) -> dict | None:
        """Gemini 的信封与 OpenAI 不同，不能用基类的字段名判定，否则整个响应都成了 extra"""
        extra = {k: v for k, v in data.items() if k not in _GEMINI_ENVELOPE_KEYS}
        return extra or None

    @staticmethod
    def _tool_call_from_part(fc: dict, fallback_id: str) -> dict:
        """functionCall → OpenAI 形态的 tool_call。Gemini 3 返回真实 id，老模型/Vertex
        可能没有，此时用本地合成的 fallback_id。"""
        return {
            "id": fc.get("id") or fallback_id,
            "type": "function",
            "function": {"name": fc.get("name", ""), "arguments": json.dumps(fc.get("args", {}))},
        }

    def _extract_reasoning_content(self, response_data: dict) -> str | None:
        thoughts = [
            p["text"] for p in self._parts(response_data) if p.get("thought") and "text" in p
        ]
        return "".join(thoughts) or None

    def _extract_assistant_message(self, response_data: dict) -> dict | None:
        """原生 parts 原样保留：Gemini 3 要求回传 functionCall 上的 thoughtSignature，
        从统一 tool_calls 重建会丢掉它。content 放原生 parts（与 Claude 放原生 blocks 同理），
        tool_calls 仍给 OpenAI 形态，供调用方执行工具、按 id 回填结果。"""
        parts = self._parts(response_data)
        if not self._needs_continuation(parts):
            return None
        message = {"role": "assistant", "content": deepcopy(parts)}
        tool_calls = self._extract_tool_calls(response_data)
        if tool_calls:
            message["tool_calls"] = [
                {"id": c.id, "type": c.type, "function": deepcopy(c.function)} for c in tool_calls
            ]
        return message

    def _extract_usage(self, response_data: dict) -> dict | None:
        """
        提取 Gemini API 的 usage 信息

        Gemini 响应格式:
        {
            "candidates": [...],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50,
                "totalTokenCount": 150
            }
        }

        转换为统一格式:
        {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
        """
        if not response_data:
            return None

        usage_metadata = response_data.get("usageMetadata")
        if not usage_metadata:
            return None

        # candidatesTokenCount 不含思考 token，而思考按输出计费；OpenAI 语义下
        # completion_tokens 包含 reasoning tokens，不加上会让成本和用量都少算
        thoughts = usage_metadata.get("thoughtsTokenCount", 0)
        usage = {
            "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
            "completion_tokens": usage_metadata.get("candidatesTokenCount", 0) + thoughts,
            "total_tokens": usage_metadata.get("totalTokenCount", 0),
        }
        if thoughts:
            usage["completion_tokens_details"] = {"reasoning_tokens": thoughts}
        if cached := usage_metadata.get("cachedContentTokenCount"):
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        return usage

    def _extract_tool_calls(self, response_data: dict):
        """
        提取 Gemini 格式的 function calls

        Gemini 响应格式:
        {
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"location": "Tokyo"}
                        }
                    }]
                }
            }]
        }
        """
        from .base import ToolCall

        tool_calls = [
            ToolCall(**self._tool_call_from_part(part["functionCall"], f"call_{i}"))
            for i, part in enumerate(self._parts(response_data))
            if "functionCall" in part
        ]
        return tool_calls or None

    @staticmethod
    def parse_thoughts(response_data: dict) -> dict:
        """
        从响应中解析思考内容和答案

        当使用 thinking=True 时，可以用此方法解析响应。

        Args:
            response_data: 原始响应数据（通过 return_raw=True 获取）

        Returns:
            dict: {
                "thought": str,  # 思考过程摘要（可能为空）
                "answer": str,   # 最终答案
            }

        Example:
            >>> result = await client.complete(
            ...     messages=[...],
            ...     thinking=True,
            ...     return_raw=True,
            ... )
            >>> parsed = GeminiClient.parse_thoughts(result.data)
            >>> print("思考:", parsed["thought"])
            >>> print("答案:", parsed["answer"])
        """
        thought_parts = []
        answer_parts = []

        try:
            candidates = response_data.get("candidates", [])
            if not candidates:
                return {"thought": "", "answer": ""}

            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                text = part.get("text", "")
                if not text:
                    continue
                if part.get("thought"):
                    thought_parts.append(text)
                else:
                    answer_parts.append(text)

            return {
                "thought": "".join(thought_parts),
                "answer": "".join(answer_parts),
            }
        except Exception as e:
            logger.warning(f"Failed to parse thoughts: {e}")
            return {"thought": "", "answer": ""}

    @staticmethod
    def _append_stream_part(parts: list[dict], part: dict) -> None:
        """把流式 part 拼回与非流式等价的 parts：同类文本片段合并，签名随合并落在
        这段文本上（Gemini 把签名放在一段文本的末尾片段，常是一个空文本 part）。
        functionCall 各自独立，不合并。"""
        last = parts[-1] if parts else None
        text_keys = {"text", "thought", "thoughtSignature"}
        if (
            last is not None
            and "text" in part
            and "text" in last
            and set(part) <= text_keys
            and set(last) <= text_keys
            and bool(part.get("thought")) == bool(last.get("thought"))
            and "thoughtSignature" not in last
        ):
            last["text"] += part["text"]
            if "thoughtSignature" in part:
                last["thoughtSignature"] = part["thoughtSignature"]
            return
        parts.append(deepcopy(part))

    def _extract_stream_usage(self, data: dict) -> dict | None:
        """从 Gemini 流式 chunk 中提取 usage（usageMetadata 字段）"""
        if "usageMetadata" in data:
            return self._extract_usage(data)
        return None

    def _prepare_stream_body(self, body: dict, return_usage: bool) -> dict:
        """Gemini 不需要 stream_options"""
        return body

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
        """Gemini 流式聊天完成

        Gemini 的 thinking 和 content 可能在同一个 chunk 中（不 continue），
        因此需要覆写基类的 stream 方法。

        已知缺口：这条独立实现**不透出** `{"type": "extra"}` 带外字段（基类与
        ClaudeClient 都透出）。Gemini 把 api_key 拼在 query string 里，本来就不该经
        网关转发，所以带外信号在这条链路上没有来源。真需要时照基类那段补即可。
        """

        import aiohttp

        effective_model = self._get_effective_model(model)
        messages = await self._preprocess_messages(messages, preprocess_msg)

        body = self._build_request_body(messages, effective_model, stream=True, **kwargs)

        effective_url = url or self._get_stream_url(effective_model)
        headers = self._merge_headers(extra_headers)

        effective_timeout = timeout if timeout is not None else self._timeout
        # 流式：空闲超时语义（见基类 chat_completions_stream 说明）
        aio_timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=min(30, effective_timeout) if effective_timeout else None,
            sock_read=effective_timeout,
        )

        session, proxy_kwargs = create_proxied_session(self._proxy)
        try:
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
                        raise LLMHTTPError(
                            f"HTTP {response.status}: {error_text}",
                            status_code=response.status,
                            response_data=_decode_error_body(error_text),
                            retryable=response.status == 429 or response.status >= 500,
                        )

                    _last_usage = None
                    _finish_reason = None
                    # 拼回的原生 parts：流末尾作为 assistant_message 发出，保留签名
                    _parts: list[dict] = []
                    _tool_calls: list[dict] = []
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)

                                # 一个 chunk 可以有多个 part（思考、正文、函数调用混排），
                                # 按顺序逐个处理，任何一个都不能漏
                                for part in self._parts(data):
                                    self._append_stream_part(_parts, part)
                                    if "functionCall" in part:
                                        # functionCall 整条到达、没有跨 chunk 的 index，
                                        # 按到达顺序编号；Gemini 3 自带真实 id
                                        call = self._tool_call_from_part(
                                            part["functionCall"], f"call_{len(_tool_calls)}"
                                        )
                                        call["index"] = len(_tool_calls)
                                        _tool_calls.append(call)
                                        if return_usage:
                                            yield {"type": "tool_call_delta", "tool_calls": [call]}
                                    elif part.get("text"):
                                        if part.get("thought"):
                                            if return_usage:
                                                yield {"type": "thinking", "content": part["text"]}
                                        elif return_usage:
                                            yield {"type": "content", "content": part["text"]}
                                        else:
                                            yield part["text"]

                                # Gemini 每个 chunk 都带 usageMetadata（累计值），
                                # 只记录最新值，流结束后统一 yield，保证 usage 事件唯一且在最后
                                if return_usage:
                                    usage = self._extract_stream_usage(data)
                                    if usage:
                                        _last_usage = usage
                                    reason = self._extract_finish_reason(data)
                                    if reason:
                                        _finish_reason = reason

                            except json.JSONDecodeError:
                                continue

                    if return_usage:
                        if self._needs_continuation(_parts):
                            message = {"role": "assistant", "content": _parts}
                            if _tool_calls:
                                message["tool_calls"] = [
                                    {k: v for k, v in c.items() if k != "index"}
                                    for c in _tool_calls
                                ]
                            yield {"type": "assistant_message", "message": message}
                        # functionCall 与 finishReason 不在同一个 chunk，这里补上非流式的修正
                        if _tool_calls and _finish_reason == "stop":
                            _finish_reason = "tool_calls"
                        yield {"type": "finish", "reason": _finish_reason}
                        if _last_usage:
                            yield {"type": "usage", "usage": _last_usage}

        except aiohttp.ClientConnectorError as e:
            raise LLMConnectionError(f"LLM 流式连接失败: {e}", cause=e, retryable=True) from e
        except asyncio.TimeoutError as e:
            raise LLMTimeoutError("LLM 流式请求超时", cause=e, retryable=True) from e
        except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError) as e:
            raise LLMConnectionError(f"LLM 流式连接失败: {e}", cause=e, retryable=True) from e

    # ========== Gemini 特有方法 ==========

    def _get_access_token(self) -> str:
        """获取 Vertex AI 的 Access Token"""
        import time

        if self._access_token and self._token_expiry and time.time() < self._token_expiry - 60:
            return self._access_token

        try:
            import google.auth
            import google.auth.transport.requests

            credentials = (
                self._credentials
                or google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])[0]
            )

            request = google.auth.transport.requests.Request()
            credentials.refresh(request)

            self._access_token = credentials.token
            self._token_expiry = time.time() + 3600
            return self._access_token
        except ImportError:
            raise ImportError("Vertex AI 模式需要安装 google-auth: pip install google-auth")
        except Exception as e:
            raise RuntimeError(f"获取 Vertex AI 访问令牌失败: {e}")

    def _convert_messages_to_contents(
        self, messages: list[dict], system_instruction: str = None
    ) -> tuple[list[dict], dict | None]:
        """将 OpenAI 格式的 messages 转换为 Gemini 格式

        - assistant 的 tool_calls → functionCall parts；flexllm 返回的 assistant_message
          （content 是原生 parts）原样回传，保住 thoughtSignature
        - tool 消息 → functionResponse；Gemini 按函数名配对，名字从前面 assistant 的
          tool_calls 按 tool_call_id 查。同一步的多条结果并进同一条 user 消息
        - 多条 system 消息按顺序拼接
        """
        contents = []
        system_texts = [system_instruction] if system_instruction else []
        tool_names: dict[str, str] = {}

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    system_texts.append(content)
                elif isinstance(content, list):
                    system_texts.append("\n".join(self._texts_of(content)))
                continue

            if role == "tool":
                part = self._convert_tool_result(msg, tool_names)
                last = contents[-1] if contents else None
                if last and last["role"] == "user" and "functionResponse" in last["parts"][0]:
                    last["parts"].append(part)
                else:
                    contents.append({"role": "user", "parts": [part]})
                continue

            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    # 合成 id（call_0…）每轮都会重复，后出现的覆盖先出现的，
                    # 保证 tool 消息配到离它最近的那次调用
                    tool_names[tc.get("id")] = tc.get("function", {}).get("name", "")
                parts = self._convert_assistant_parts(msg)
            else:
                parts = self._convert_content_to_parts(content)

            if parts:
                contents.append(
                    {"role": "model" if role == "assistant" else "user", "parts": parts}
                )

        system_texts = [t for t in system_texts if t]
        system_obj = {"parts": [{"text": "\n\n".join(system_texts)}]} if system_texts else None
        return contents, system_obj

    @staticmethod
    def _texts_of(content: list) -> list[str]:
        return [
            p if isinstance(p, str) else p.get("text", "")
            for p in content
            if isinstance(p, str) or p.get("type", "text") == "text"
        ]

    def _convert_assistant_parts(self, msg: dict) -> list[dict]:
        content = msg.get("content")
        # flexllm 保留下来的原生 parts（见 _extract_assistant_message）原样回传
        if isinstance(content, list) and self._needs_continuation(content):
            return deepcopy(content)

        parts = self._convert_content_to_parts(content)
        for i, tc in enumerate(msg.get("tool_calls") or []):
            func = tc.get("function", {})
            arguments = func.get("arguments") or "{}"
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    # 与 Claude 一致：历史里坏掉的参数降级为空对象，不让整段对话发不出去
                    arguments = {}
            call = {"name": func.get("name", ""), "args": arguments}
            if tc.get("id"):
                call["id"] = tc["id"]
            part = {"functionCall": call}
            # 签名只要求挂在每一步的第一个 functionCall 上
            if i == 0:
                part["thoughtSignature"] = _SKIP_SIGNATURE
            parts.append(part)
        return parts

    def _convert_tool_result(self, msg: dict, tool_names: dict[str, str]) -> dict:
        """一条 tool 消息 → functionResponse；媒体放 functionResponse.parts（Gemini 3 的
        多模态函数结果。2.x 的图片已在 _build_request_body 里挪成单独的 user 消息）"""
        call_id = msg.get("tool_call_id")
        name = msg.get("name") or tool_names.get(call_id)
        if not name:
            raise ValueError(
                f"tool 消息 tool_call_id={call_id!r} 找不到对应的 assistant tool_call："
                "Gemini 的 functionResponse 必须带函数名"
            )
        content = msg.get("content")
        response = {"name": name}
        if call_id:
            response["id"] = call_id
        if isinstance(content, list):
            response["response"] = {"result": "".join(self._texts_of(content))}
            if has_non_text_parts(content):
                response["parts"] = [
                    p for p in self._convert_content_to_parts(content) if "text" not in p
                ]
        else:
            # response 必须是对象（protobuf Struct），纯字符串会被 400
            response["response"] = {"result": content or ""}
        return {"functionResponse": response}

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        """OpenAI 格式的函数合并成一个 functionDeclarations；原生格式原样透传。

        参数 schema 用 parametersJsonSchema：它接受完整 JSON Schema，而 parameters
        只收 OpenAPI 子集，OpenAI 常见的 additionalProperties 会被 400。
        """
        declarations, native = [], []
        for tool in tools:
            if tool.get("type") != "function":
                native.append(tool)
                continue
            func = tool["function"]
            declaration = {"name": func["name"]}
            if func.get("description"):
                declaration["description"] = func["description"]
            if func.get("parameters"):
                declaration["parametersJsonSchema"] = func["parameters"]
            declarations.append(declaration)
        return ([{"functionDeclarations": declarations}] if declarations else []) + native

    @staticmethod
    def _convert_tool_choice(tool_choice: str | dict) -> dict:
        modes = {"auto": "AUTO", "none": "NONE", "required": "ANY"}
        if isinstance(tool_choice, str) and tool_choice in modes:
            return {"mode": modes[tool_choice]}
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            return {"mode": "ANY", "allowedFunctionNames": [tool_choice["function"]["name"]]}
        raise ValueError(f"不支持的 tool_choice: {tool_choice!r}")

    def _convert_content_to_parts(self, content: Any) -> list[dict]:
        """将 OpenAI 格式的 content 转换为 Gemini 格式的 parts"""
        if content is None:
            return []
        if isinstance(content, str):
            return [{"text": content}]

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append({"text": item})
                elif isinstance(item, dict):
                    item_type = item.get("type", "text")
                    if item_type == "text" and item.get("text"):
                        parts.append({"text": item["text"]})
                    elif item_type == "image_url":
                        if img := self._convert_image_url(item.get("image_url", {})):
                            parts.append(img)
                    elif item_type in ("video_url", "audio_url"):
                        media_obj = item.get(item_type, {})
                        if part := self._convert_media_url(media_obj):
                            parts.append(part)
                    elif item_type == "input_audio":
                        audio_obj = item.get("input_audio", {})
                        if part := self._convert_input_audio(audio_obj):
                            parts.append(part)
                    elif item_type == "image":
                        if img := self._convert_image_direct(item):
                            parts.append(img)
            return parts
        return []

    _MEDIA_PART_TYPES = ("image_url", "video_url", "audio_url")

    @classmethod
    def _has_remote_media(cls, messages: list[dict]) -> bool:
        """是否含 http(s) 媒体 URL。Gemini 不会替你拉取外部 URL（fileData 给外链直接
        400 "Cannot fetch content"），不转 base64 这张图就只能被丢掉。"""
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") in cls._MEDIA_PART_TYPES:
                    url = item.get(item["type"], {}).get("url", "")
                    if url.startswith(("http://", "https://")):
                        return True
        return False

    async def _preprocess_messages(self, messages, preprocess_msg: bool = False):
        return await super()._preprocess_messages(
            messages, preprocess_msg or self._has_remote_media(messages)
        )

    async def _preprocess_messages_batch(self, messages_list, preprocess_msg: bool = False):
        return await super()._preprocess_messages_batch(
            messages_list,
            preprocess_msg or any(self._has_remote_media(m) for m in messages_list),
        )

    def _convert_image_url(self, image_url_obj: dict) -> dict | None:
        """将 OpenAI 的 image_url 格式转换为 Gemini 的 inline_data 格式"""
        url = image_url_obj.get("url", "")
        if not url:
            return None

        if url.startswith("data:"):
            match = re.match(r"data:([^;]+);base64,(.+)", url)
            if match:
                return {"inline_data": {"mime_type": match.group(1), "data": match.group(2)}}

        logger.warning(f"Gemini API 不直接支持外部 URL，请先转换为 base64: {url[:50]}...")
        return None

    def _convert_media_url(self, media_url_obj: dict) -> dict | None:
        """将 video_url/audio_url 格式转换为 Gemini 的 inline_data 格式"""
        url = media_url_obj.get("url", "")
        if not url:
            return None

        if url.startswith("data:"):
            match = re.match(r"data:([^;]+);base64,(.+)", url)
            if match:
                return {"inline_data": {"mime_type": match.group(1), "data": match.group(2)}}

        logger.warning(f"Gemini API 不直接支持外部 URL，请先转换为 base64: {url[:50]}...")
        return None

    def _convert_input_audio(self, audio_obj: dict) -> dict | None:
        """将 OpenAI input_audio 格式转换为 Gemini 的 inline_data 格式"""
        data = audio_obj.get("data", "")
        fmt = audio_obj.get("format", "wav")
        if data:
            return {"inline_data": {"mime_type": f"audio/{fmt}", "data": data}}
        return None

    def _convert_image_direct(self, image_obj: dict) -> dict | None:
        """处理直接的图片数据"""
        data = image_obj.get("data", "")
        if data:
            return {
                "inline_data": {"mime_type": image_obj.get("mime_type", "image/jpeg"), "data": data}
            }
        return None

    def model_list(self) -> list[str]:
        """获取可用模型列表"""
        import requests

        if self._use_vertex_ai:
            url = f"{self._base_url}/projects/{self._project_id}/locations/{self._location}/publishers/google/models"
            response = requests.get(url, headers=self._get_headers())
        else:
            response = requests.get(f"{self._base_url}/models?key={self._api_key}")

        if response.status_code == 200:
            models = response.json().get("models", [])
            return [m.get("name", "").replace("models/", "") for m in models]
        logger.error(f"Failed to fetch model list: {response.text}")
        return []
