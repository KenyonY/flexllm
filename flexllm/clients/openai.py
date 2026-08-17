"""
OpenAI 兼容 API 客户端

支持 OpenAI、vLLM、通义千问、DeepSeek 等兼容 OpenAI API 的服务。
"""

import logging
import re

logger = logging.getLogger(__name__)

from ..cache import ResponseCacheConfig
from ..msg_processors.audio_processor import normalize_audio_format
from .audio import AudioMixin
from .base import LLMClientBase


class OpenAIClient(AudioMixin, LLMClientBase):
    """
    OpenAI 兼容 API 客户端

    支持 OpenAI、vLLM、Ollama、DeepSeek 等兼容 OpenAI API 的服务。

    Example:
        >>> client = OpenAIClient(
        ...     base_url="https://api.openai.com/v1",
        ...     api_key="your-key",
        ...     model="gpt-4",
        ... )
        >>> result = await client.chat_completions(messages)

    Example (Ollama/vLLM 本地模型):
        >>> client = OpenAIClient(
        ...     base_url="http://localhost:11434/v1",  # Ollama
        ...     model="qwen3:4b",
        ... )

    Example (thinking 参数 - 统一的思考控制):
        >>> # 禁用思考（快速响应）
        >>> result = client.chat_completions_sync(
        ...     messages=[{"role": "user", "content": "1+1=?"}],
        ...     thinking=False,
        ... )
        >>> # 启用思考并获取思考内容
        >>> result = client.chat_completions_sync(
        ...     messages=[{"role": "user", "content": "1+1=?"}],
        ...     thinking=True,
        ...     return_raw=True,
        ... )
        >>> parsed = OpenAIClient.parse_thoughts(result.data)
        >>> print("思考:", parsed["thought"])
        >>> print("答案:", parsed["answer"])

    thinking 参数值:
        - False: 禁用思考（Ollama: think=False, vLLM/Qwen3: /no_think）
        - True: 启用思考（Ollama: think=True）
        - None: 使用模型默认行为
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        model: str = None,
        concurrency_limit: int = 10,
        max_qps: int = 1000,
        timeout: int = 100,
        retry_times: int = 3,
        retry_delay: float = 0.55,
        cache_image: bool = False,
        cache_dir: str | None = None,
        cache: ResponseCacheConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
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
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    # ========== 实现基类核心方法 ==========

    def _get_url(self, model: str, stream: bool = False) -> str:
        return f"{self._base_url}/chat/completions"

    def _get_headers(self) -> dict:
        return self._headers

    @staticmethod
    def _convert_audio_url_to_input_audio(messages: list[dict]) -> list[dict]:
        """将 audio_url 类型转换为 OpenAI 标准的 input_audio 类型。

        audio_url 是 flexllm 的便捷格式，OpenAI API 不支持。
        类似 Claude/Gemini client 中对 audio_url 的转换。
        """
        result = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                result.append(msg)
                continue
            new_content = []
            changed = False
            for part in content:
                if part.get("type") == "audio_url":
                    url = part.get("audio_url", {}).get("url", "")
                    if url.startswith("data:"):
                        match = re.match(r"data:audio/([^;]+);base64,(.+)", url, re.DOTALL)
                        if match:
                            new_content.append(
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": match.group(2),
                                        "format": normalize_audio_format(match.group(1)),
                                    },
                                }
                            )
                            changed = True
                            continue
                    # 非 data URI 的 audio_url 无法转换，保持原样
                    new_content.append(part)
                else:
                    new_content.append(part)
            result.append({**msg, "content": new_content} if changed else msg)
        return result

    def _build_request_body(
        self,
        messages: list[dict],
        model: str,
        stream: bool = False,
        max_tokens: int = None,
        thinking: bool | None = None,
        **kwargs,
    ) -> dict:
        """
        构建请求体

        Args:
            thinking: 统一的思考控制参数
                - False: 禁用思考（Ollama: think=False, vLLM: enable_thinking=False）
                - True: 启用思考（Ollama: think=True, vLLM: enable_thinking=True）
                - None: 使用模型默认行为

        Note:
            think / chat_template_kwargs 是 Ollama / vLLM 的非标准扩展字段，
            官方 OpenAI 端点（api.openai.com）严格校验请求体会返回 400，
            因此对官方端点不注入这两个字段；其他端点维持现状。
        """
        processed_messages = self._convert_audio_url_to_input_audio(messages)

        body = {"messages": processed_messages, "model": model, "stream": stream}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        # 思考模式控制：同时发送多种格式，由服务端选择性处理。
        # 官方 OpenAI 端点不认识这些字段且严格校验（400），跳过注入。
        # 提取阶段是否保留思考内容不在此处记录状态（并发/批量下实例状态会互相污染），
        # 而是由 _extract_content 按各自请求的 thinking 参数决定
        is_official_openai = bool(self._base_url) and "api.openai.com" in self._base_url
        if thinking is not None and not is_official_openai:
            body["think"] = thinking  # Ollama
            body["chat_template_kwargs"] = {"enable_thinking": thinking}  # vLLM

        # Prefill: messages 末尾是 assistant message → 让模型从该 content 继续生成
        # vLLM 需要同时关闭 add_generation_prompt 并启用 continue_final_message
        # 仅在 kwargs 未显式提供这两个参数时自动设置，给用户保留 override 能力
        if (
            processed_messages
            and processed_messages[-1].get("role") == "assistant"
            and "continue_final_message" not in kwargs
            and "add_generation_prompt" not in kwargs
        ):
            body["continue_final_message"] = True
            body["add_generation_prompt"] = False

        body.update(kwargs)
        return body

    @staticmethod
    def _strip_think_tags(content: str) -> str:
        """剥离 vLLM/Qwen 思考模型返回的 <think>...</think> 内容"""
        import re

        return re.sub(r"^.*?</think>\s*", "", content, count=1, flags=re.DOTALL)

    def _extract_content(
        self, response_data: dict, thinking: bool | None = None, **gen_kwargs
    ) -> str | None:
        """提取内容；thinking=True 的请求保留思考内容（<think> 包裹），否则剥离。

        thinking 来自该请求自身的生成参数（由基类透传），不依赖实例状态，
        批量/并发下各请求互不影响。
        """
        try:
            message = response_data["choices"][0]["message"]
            content = message.get("content") or ""
            keep_thinking = thinking is True

            # reasoning-parser 模式：思考内容在独立的 reasoning 字段
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if reasoning:
                if not content and self._extract_finish_reason(response_data) != "length":
                    # parser 误判：正式回答被放入 reasoning（如 Qwen3.5 默认不思考时）。
                    # 但 finish_reason=length 且 content 为空是另一回事——思考把
                    # max_tokens 吃光了，此时 reasoning 是半截思维链，不能冒充回答。
                    return reasoning
                if keep_thinking:
                    return f"<think>\n{reasoning}\n</think>\n\n{content.strip()}"
                return content.strip() or content

            # 无 reasoning-parser：思考内容内嵌在 content 中
            if content and "</think>" in content and not keep_thinking:
                content = self._strip_think_tags(content)

            return content
        except (KeyError, IndexError) as e:
            logger.warning(f"Failed to extract content: {e}")
            return None

    def _extract_finish_reason(self, response_data: dict) -> str | None:
        """非流式响应与流式 chunk 结构一致：choices[0].finish_reason"""
        choices = (response_data or {}).get("choices")
        if choices:
            return choices[0].get("finish_reason") or None
        return None

    def _extract_stream_content(self, data: dict) -> str | None:
        choices = data.get("choices")
        if choices:
            return choices[0].get("delta", {}).get("content")
        return None

    def _extract_stream_thinking(self, data: dict) -> str | None:
        choices = data.get("choices")
        if choices:
            delta = choices[0].get("delta", {})
            return delta.get("reasoning") or delta.get("reasoning_content")
        return None

    def _extract_stream_tool_calls(self, data: dict) -> list[dict] | None:
        """从 OpenAI 流式 chunk 中提取 tool_call delta"""
        choices = data.get("choices")
        if choices:
            return choices[0].get("delta", {}).get("tool_calls")
        return None

    def _extract_tool_calls(self, response_data: dict):
        """提取 OpenAI 格式的 tool_calls"""
        from .base import ToolCall

        if not response_data:
            return None

        try:
            message = response_data["choices"][0]["message"]
            tool_calls_data = message.get("tool_calls")
            if not tool_calls_data:
                return None
            return [
                ToolCall(id=tc["id"], type=tc["type"], function=tc["function"])
                for tc in tool_calls_data
            ]
        except (KeyError, IndexError):
            return None

    @staticmethod
    def parse_thoughts(response_data: dict) -> dict:
        """
        从响应中解析思考内容和答案

        支持两种格式：
        1. reasoning 字段格式（Ollama DeepSeek-R1/Qwen3 等）
        2. 内嵌标签格式（vLLM Qwen3 等）：<think>...</think> 标签

        Args:
            response_data: 原始响应数据（通过 return_raw=True 获取）

        Returns:
            dict: {
                "thought": str,  # 思考过程（可能为空）
                "answer": str,   # 最终答案
            }

        Example:
            >>> result = client.chat_completions_sync(
            ...     messages=[...],
            ...     thinking=True,
            ...     return_raw=True,
            ... )
            >>> parsed = OpenAIClient.parse_thoughts(result.data)
            >>> print("思考:", parsed["thought"])
            >>> print("答案:", parsed["answer"])
        """
        import re

        try:
            message = response_data.get("choices", [{}])[0].get("message", {})
            content = message.get("content", "")
            # 与 _extract_content 一致：reasoning（Ollama 等）与 reasoning_content
            # （DeepSeek/vLLM reasoning-parser 等）两种字段都支持
            reasoning = message.get("reasoning") or message.get("reasoning_content") or ""

            # 如果有 reasoning 字段，直接使用
            if reasoning:
                return {
                    "thought": reasoning,
                    "answer": content,
                }

            # 否则尝试解析内嵌的 <think>...</think> 标签（Qwen3 格式）
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if think_match:
                thought = think_match.group(1).strip()
                # 移除 <think> 标签后的内容作为答案
                answer = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
                return {
                    "thought": thought,
                    "answer": answer,
                }

            # 没有思考内容
            return {
                "thought": "",
                "answer": content,
            }
        except Exception as e:
            logger.warning(f"Failed to parse thoughts: {e}")
            return {"thought": "", "answer": ""}

    # ========== OpenAI 特有方法 ==========

    def model_list(self) -> list[str]:
        """获取可用模型列表"""
        import requests

        response = requests.get(
            f"{self._base_url}/models",
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        response.raise_for_status()
        return [m["id"] for m in response.json()["data"]]
