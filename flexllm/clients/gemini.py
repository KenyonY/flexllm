"""
Gemini API Client - Google Gemini 模型的批量调用客户端

与 OpenAIClient 保持相同的接口，方便上层代码无缝切换。
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

from ..cache import ResponseCacheConfig
from .base import LLMClientBase


class GeminiClient(LLMClientBase):
    """
    Google Gemini API 客户端

    支持 Gemini Developer API 和 Vertex AI。

    Example (Gemini Developer API):
        >>> client = GeminiClient(api_key="your-key", model="gemini-3-flash-preview")
        >>> result = await client.chat_completions(messages)

    Example (Vertex AI):
        >>> client = GeminiClient(
        ...     project_id="your-project-id",
        ...     location="us-central1",
        ...     model="gemini-3-flash-preview",
        ...     use_vertex_ai=True,
        ... )

    Example (thinking 参数 - 统一的思考控制):
        >>> # 禁用思考（最快响应）
        >>> result = client.chat_completions_sync(
        ...     messages=[{"role": "user", "content": "1+1=?"}],
        ...     thinking=False,
        ... )
        >>> # 启用思考并返回思考内容
        >>> result = client.chat_completions_sync(
        ...     messages=[{"role": "user", "content": "复杂推理问题"}],
        ...     thinking=True,
        ...     return_raw=True,
        ... )
        >>> parsed = GeminiClient.parse_thoughts(result.data)

    thinking 参数值:
        - False: 禁用思考（thinkingLevel=minimal）
        - True: 启用思考并返回思考内容（includeThoughts=True）
        - "minimal"/"low"/"medium"/"high": 设置思考深度（仅 Gemini 3）
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
        cache_dir: str = "image_cache",
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
        **kwargs,
    ) -> dict:
        """
        构建请求体

        Args:
            thinking: 统一的思考控制参数
                - False: 禁用思考（thinkingLevel=minimal）
                - True: 启用思考并返回思考内容（includeThoughts=True）
                - "minimal"/"low"/"medium"/"high": 设置思考深度
                - None: 使用模型默认行为
        """
        contents, system_obj = self._convert_messages_to_contents(messages)
        body = {"contents": contents}

        if system_obj:
            body["systemInstruction"] = system_obj

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

        # 构建 thinkingConfig
        thinking_config = {}
        if thinking is False:
            # 禁用思考
            thinking_config["thinkingLevel"] = "minimal"
        elif thinking is True:
            # 启用思考并返回思考内容
            thinking_config["includeThoughts"] = True
        elif isinstance(thinking, str):
            # 设置思考深度
            thinking_config["thinkingLevel"] = thinking
            thinking_config["includeThoughts"] = True
        # thinking=None 时不设置，使用默认行为

        if thinking_config:
            gen_config["thinkingConfig"] = thinking_config

        if gen_config:
            body["generationConfig"] = gen_config
        if safety_settings:
            body["safetySettings"] = safety_settings

        return body

    def _extract_content(self, response_data: dict) -> str | None:
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

        return {
            "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
            "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
            "total_tokens": usage_metadata.get("totalTokenCount", 0),
        }

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
        import json

        from .base import ToolCall

        try:
            candidates = response_data.get("candidates", [])
            if not candidates:
                return None

            parts = candidates[0].get("content", {}).get("parts", [])
            tool_calls = []
            for i, part in enumerate(parts):
                if "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{i}",  # Gemini 没有 id，生成一个
                            type="function",
                            function={
                                "name": fc.get("name", ""),
                                "arguments": json.dumps(fc.get("args", {})),
                            },
                        )
                    )
            return tool_calls if tool_calls else None
        except Exception:
            return None

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
            >>> result = await client.chat_completions(
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

    def _extract_stream_content(self, data: dict) -> str | None:
        """
        从 Gemini 流式响应中提取文本内容

        Gemini 流式响应格式：
        {
            "candidates": [{
                "content": {
                    "parts": [{"text": "部分文本"}],
                    "role": "model"
                }
            }]
        }
        """
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                return None

            # 获取第一个候选的内容
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])

            # 提取所有文本部分
            for part in parts:
                if "text" in part:
                    return part["text"]
            return None
        except Exception:
            return None

    async def chat_completions_stream(
        self,
        messages: list[dict],
        model: str = None,
        return_usage: bool = False,
        preprocess_msg: bool = False,
        url: str = None,
        timeout: int = None,
        **kwargs,
    ):
        """
        流式聊天完成（Gemini 专用实现）

        Gemini 流式响应中的 usage 信息在最后一个 chunk 的 usageMetadata 字段中。

        Args:
            messages: 消息列表
            model: 模型名称
            return_usage: 是否返回 usage 信息
            preprocess_msg: 是否预处理消息
            url: 自定义请求 URL
            timeout: 超时时间（秒）

        Yields:
            - return_usage=False: str 内容片段
            - return_usage=True: dict，包含 type 和对应数据
        """
        import json

        import aiohttp

        effective_model = self._get_effective_model(model)
        messages = await self._preprocess_messages(messages, preprocess_msg)

        body = self._build_request_body(messages, effective_model, stream=True, **kwargs)
        # 注意：Gemini 不需要 stream_options，usage 自动包含在最后一个 chunk 中

        effective_url = url or self._get_stream_url(effective_model)
        headers = self._get_headers()

        effective_timeout = timeout if timeout is not None else self._timeout
        aio_timeout = aiohttp.ClientTimeout(total=effective_timeout)

        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.post(
                effective_url, json=body, headers=headers, timeout=aio_timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)

                            # 提取内容
                            content = self._extract_stream_content(data)
                            if content:
                                if return_usage:
                                    yield {"type": "content", "content": content}
                                else:
                                    yield content

                            # 检查是否包含 usage（Gemini 在最后一个 chunk 中包含 usageMetadata）
                            if return_usage and "usageMetadata" in data:
                                usage = self._extract_usage(data)
                                if usage:
                                    yield {"type": "usage", "usage": usage}

                        except json.JSONDecodeError:
                            continue

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
        """将 OpenAI 格式的 messages 转换为 Gemini 格式"""
        contents = []
        extracted_system = system_instruction

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    extracted_system = content
                elif isinstance(content, list):
                    texts = [p.get("text", "") for p in content if p.get("type") == "text"]
                    extracted_system = "\n".join(texts)
                continue

            gemini_role = "model" if role == "assistant" else "user"
            parts = self._convert_content_to_parts(content)

            if parts:
                contents.append({"role": gemini_role, "parts": parts})

        system_obj = {"parts": [{"text": extracted_system}]} if extracted_system else None
        return contents, system_obj

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
                    elif item_type == "image":
                        if img := self._convert_image_direct(item):
                            parts.append(img)
            return parts
        return []

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
