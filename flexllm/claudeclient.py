"""
Anthropic Claude API Client

支持 Claude 系列模型（claude-3-opus, claude-3-sonnet, claude-3-haiku 等）
"""

import json
import re
from typing import List, Optional, Union

import aiohttp
from loguru import logger

from .base_client import LLMClientBase, ToolCall
from .cache import ResponseCacheConfig


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
        - False: 禁用扩展思考
        - True: 启用扩展思考（默认 budget_tokens=10000）
        - int: 启用扩展思考并指定 budget_tokens
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
        cache_dir: str = "image_cache",
        cache: Optional[ResponseCacheConfig] = None,
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

    def _get_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": self._api_version,
        }

    def _build_request_body(
        self,
        messages: List[dict],
        model: str,
        stream: bool = False,
        max_tokens: int = 4096,  # Claude 必需参数
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        thinking: Union[bool, int, None] = None,
        **kwargs,
    ) -> dict:
        """
        构建 Claude API 请求体

        Args:
            thinking: 扩展思考控制参数
                - False: 禁用扩展思考
                - True: 启用扩展思考（默认 budget_tokens=10000）
                - int: 启用扩展思考并指定 budget_tokens
                - None: 使用模型默认行为
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
                system_content = (
                    (system_content + "\n" + content) if system_content else content
                )
            else:
                user_messages.append(self._convert_message(msg))

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

        # Claude 扩展思考模式
        if thinking is True:
            body["thinking"] = {"type": "enabled", "budget_tokens": 10000}
        elif isinstance(thinking, int) and thinking > 0:
            body["thinking"] = {"type": "enabled", "budget_tokens": thinking}
        elif thinking is False:
            body["thinking"] = {"type": "disabled"}

        # 透传其他参数（如 tools）
        body.update(kwargs)
        return body

    def _convert_message(self, msg: dict) -> dict:
        """转换消息格式（处理多模态内容）"""
        role = msg.get("role", "user")
        content = msg.get("content", "")

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
                        claude_content.append(
                            {"type": "text", "text": item.get("text", "")}
                        )
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
            return {"role": claude_role, "content": claude_content}

        return {"role": claude_role, "content": content}

    def _extract_content(self, response_data: dict) -> Optional[str]:
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

    def _extract_usage(self, response_data: dict) -> Optional[dict]:
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

    def _extract_tool_calls(self, response_data: dict) -> Optional[List[ToolCall]]:
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

    def _extract_stream_content(self, data: dict) -> Optional[str]:
        """从 Claude 流式响应中提取内容"""
        # Claude 流式格式：event: content_block_delta, data: {"delta": {"text": "..."}}
        if data.get("type") == "content_block_delta":
            delta = data.get("delta", {})
            if delta.get("type") == "text_delta":
                return delta.get("text")
        return None

    async def chat_completions_stream(
        self,
        messages: List[dict],
        model: str = None,
        return_usage: bool = False,
        preprocess_msg: bool = False,
        url: str = None,
        timeout: int = None,
        **kwargs,
    ):
        """Claude 流式聊天完成"""
        effective_model = self._get_effective_model(model)
        messages = await self._preprocess_messages(messages, preprocess_msg)

        body = self._build_request_body(
            messages, effective_model, stream=True, **kwargs
        )
        effective_url = url or self._get_url(effective_model, stream=True)
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

                usage_data = None
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

                            # 检查 message_delta 中的 usage
                            if data.get("type") == "message_delta":
                                usage = data.get("usage")
                                if usage:
                                    usage_data = {
                                        "prompt_tokens": usage.get("input_tokens", 0),
                                        "completion_tokens": usage.get(
                                            "output_tokens", 0
                                        ),
                                        "total_tokens": usage.get("input_tokens", 0)
                                        + usage.get("output_tokens", 0),
                                    }

                            # 检查 message_start 中的 usage（输入 tokens）
                            if data.get("type") == "message_start":
                                msg_usage = data.get("message", {}).get("usage", {})
                                if msg_usage:
                                    usage_data = {
                                        "prompt_tokens": msg_usage.get(
                                            "input_tokens", 0
                                        ),
                                        "completion_tokens": 0,
                                        "total_tokens": msg_usage.get("input_tokens", 0),
                                    }

                        except json.JSONDecodeError:
                            continue

                # 最后返回 usage
                if return_usage and usage_data:
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

    def model_list(self) -> List[str]:
        """返回 Claude 模型列表（静态）"""
        return [
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]
