"""
OpenAI 兼容 API 客户端

支持 OpenAI、vLLM、通义千问、DeepSeek 等兼容 OpenAI API 的服务。
"""

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

from ..cache import ResponseCacheConfig
from .base import LLMClientBase

if TYPE_CHECKING:
    pass


class OpenAIClient(LLMClientBase):
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
        cache_dir: str = "image_cache",
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
                - False: 禁用思考（Ollama: think=False, vLLM/Qwen3: /no_think）
                - True: 启用思考（Ollama: think=True）
                - None: 使用模型默认行为
        """
        processed_messages = messages

        # 禁用思考时，添加 /no_think 标签（vLLM/Qwen3 格式）
        if thinking is False:
            processed_messages = [m.copy() for m in messages]
            for i in range(len(processed_messages) - 1, -1, -1):
                if processed_messages[i].get("role") == "user":
                    content = processed_messages[i].get("content", "")
                    if isinstance(content, str) and "/no_think" not in content:
                        processed_messages[i]["content"] = content + " /no_think"
                    break

        body = {"messages": processed_messages, "model": model, "stream": stream}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        # Ollama 格式：think 参数
        if thinking is True:
            body["think"] = True
        elif thinking is False:
            body["think"] = False

        body.update(kwargs)
        return body

    def _extract_content(self, response_data: dict) -> str | None:
        try:
            return response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.warning(f"Failed to extract content: {e}")
            return None

    def _extract_stream_content(self, data: dict) -> str | None:
        try:
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0].get("delta", {}).get("content")
        except Exception:
            pass
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
            reasoning = message.get("reasoning", "")

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
