#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Token 计数和成本估算模块

支持使用 tiktoken 精确计算，或在缺失时使用估算方法。
"""

import hashlib
import json
from typing import Union, List, Dict, Any, Optional

# tiktoken 是可选依赖
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


# 主流模型定价表 (单位: $/token)
# 更新于 2025-01
MODEL_PRICING = {
    # OpenAI - GPT 系列
    "gpt-4o": {"input": 2.5 / 1e6, "output": 10 / 1e6},
    "gpt-4o-mini": {"input": 0.15 / 1e6, "output": 0.6 / 1e6},
    "gpt-4.1": {"input": 2 / 1e6, "output": 8 / 1e6},
    "gpt-4.1-mini": {"input": 0.4 / 1e6, "output": 1.6 / 1e6},
    "gpt-4.1-nano": {"input": 0.1 / 1e6, "output": 0.4 / 1e6},
    "gpt-4-turbo": {"input": 10 / 1e6, "output": 30 / 1e6},
    "gpt-4": {"input": 30 / 1e6, "output": 60 / 1e6},
    "gpt-3.5-turbo": {"input": 0.5 / 1e6, "output": 1.5 / 1e6},
    # OpenAI - o 系列推理模型
    "o1": {"input": 15 / 1e6, "output": 60 / 1e6},
    "o1-mini": {"input": 3 / 1e6, "output": 12 / 1e6},
    "o3": {"input": 2 / 1e6, "output": 8 / 1e6},
    "o3-mini": {"input": 1.1 / 1e6, "output": 4.4 / 1e6},
    "o4-mini": {"input": 1.1 / 1e6, "output": 4.4 / 1e6},
    # Claude
    "claude-opus-4-5": {"input": 5 / 1e6, "output": 25 / 1e6},
    "claude-opus-4": {"input": 15 / 1e6, "output": 75 / 1e6},
    "claude-sonnet-4-5": {"input": 3 / 1e6, "output": 15 / 1e6},
    "claude-sonnet-4": {"input": 3 / 1e6, "output": 15 / 1e6},
    "claude-haiku-4-5": {"input": 1 / 1e6, "output": 5 / 1e6},
    "claude-haiku-3-5": {"input": 0.8 / 1e6, "output": 4 / 1e6},
    "claude-haiku-3": {"input": 0.25 / 1e6, "output": 1.25 / 1e6},
    # Gemini
    "gemini-2.5-pro": {"input": 1.25 / 1e6, "output": 10 / 1e6},
    "gemini-2.5-flash": {"input": 0.15 / 1e6, "output": 0.6 / 1e6},
    "gemini-2.5-flash-lite": {"input": 0.1 / 1e6, "output": 0.4 / 1e6},
    "gemini-2.0-flash": {"input": 0.1 / 1e6, "output": 0.4 / 1e6},
    "gemini-2.0-flash-lite": {"input": 0.075 / 1e6, "output": 0.3 / 1e6},
    "gemini-1.5-pro": {"input": 1.25 / 1e6, "output": 5 / 1e6},
    "gemini-1.5-flash": {"input": 0.075 / 1e6, "output": 0.3 / 1e6},
    # DeepSeek (V3.2 统一定价)
    "deepseek-chat": {"input": 0.28 / 1e6, "output": 0.42 / 1e6},
    "deepseek-reasoner": {"input": 0.28 / 1e6, "output": 0.42 / 1e6},
    # Qwen
    "qwen-turbo": {"input": 0.05 / 1e6, "output": 0.2 / 1e6},
    "qwen-plus": {"input": 0.4 / 1e6, "output": 1.2 / 1e6},
    "qwen-max": {"input": 2 / 1e6, "output": 6 / 1e6},
    "qwen2.5-max": {"input": 1.6 / 1e6, "output": 6.4 / 1e6},
    "qwen3-max": {"input": 1.2 / 1e6, "output": 6 / 1e6},
}

# 模型到 tiktoken 编码器的映射
MODEL_TO_ENCODING = {
    # GPT-4o 系列
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    # GPT-4.1 系列
    "gpt-4.1": "o200k_base",
    "gpt-4.1-mini": "o200k_base",
    "gpt-4.1-nano": "o200k_base",
    # GPT-4 系列
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    # o 系列推理模型
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o3": "o200k_base",
    "o3-mini": "o200k_base",
    "o4-mini": "o200k_base",
}

# 编码器缓存
_encoding_cache: Dict[str, Any] = {}


def _get_encoding(model: str):
    """获取模型对应的 tiktoken 编码器"""
    if not TIKTOKEN_AVAILABLE:
        return None

    encoding_name = MODEL_TO_ENCODING.get(model, "cl100k_base")
    if encoding_name not in _encoding_cache:
        _encoding_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _encoding_cache[encoding_name]


def _estimate_tokens_simple(text: str) -> int:
    """简单估算：中文约 2 字符/token，英文约 4 字符/token"""
    if not text:
        return 0
    # 粗略统计中文字符
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return chinese_chars // 2 + other_chars // 4 + 1


def count_tokens(
    content: Union[str, List[Dict], Dict],
    model: str = "gpt-4o"
) -> int:
    """
    计算 token 数量

    Args:
        content: 文本字符串或 messages 列表
        model: 模型名称，用于选择正确的 tokenizer

    Returns:
        token 数量
    """
    # 处理 messages 格式
    if isinstance(content, list):
        total = 0
        for msg in content:
            if isinstance(msg, dict):
                # 每条消息有固定开销
                total += 4  # role + content 标记
                for key, value in msg.items():
                    if isinstance(value, str):
                        total += count_tokens(value, model)
                    elif isinstance(value, list):
                        # 处理多模态内容
                        for item in value:
                            if isinstance(item, dict) and "text" in item:
                                total += count_tokens(item["text"], model)
                            elif isinstance(item, dict) and "image_url" in item:
                                # 图像 token 估算 (低分辨率约 85，高分辨率约 170*tiles)
                                total += 85
        return total + 2  # 结束标记

    if isinstance(content, dict):
        return count_tokens(json.dumps(content, ensure_ascii=False), model)

    # 文本处理
    text = str(content)
    encoding = _get_encoding(model)

    if encoding:
        return len(encoding.encode(text))
    else:
        return _estimate_tokens_simple(text)


def count_messages_tokens(
    messages_list: List[List[Dict]],
    model: str = "gpt-4o"
) -> int:
    """
    批量计算 messages 的 token 总数

    Args:
        messages_list: messages 列表的列表
        model: 模型名称

    Returns:
        总 token 数量
    """
    return sum(count_tokens(msgs, model) for msgs in messages_list)


def estimate_cost(
    input_tokens: int,
    output_tokens: int = 0,
    model: str = "gpt-4o"
) -> float:
    """
    估算 API 调用成本

    Args:
        input_tokens: 输入 token 数
        output_tokens: 输出 token 数 (如果未知可传 0)
        model: 模型名称

    Returns:
        估算成本 (美元)
    """
    # 尝试匹配模型名称
    pricing = None
    for key in MODEL_PRICING:
        if key in model.lower():
            pricing = MODEL_PRICING[key]
            break

    if not pricing:
        # 默认使用 gpt-4o-mini 的价格作为保守估计
        pricing = MODEL_PRICING["gpt-4o-mini"]

    return input_tokens * pricing["input"] + output_tokens * pricing["output"]


def estimate_batch_cost(
    messages_list: List[List[Dict]],
    model: str = "gpt-4o",
    avg_output_tokens: int = 500
) -> Dict[str, Any]:
    """
    估算批量处理的成本

    Args:
        messages_list: messages 列表的列表
        model: 模型名称
        avg_output_tokens: 预估每条请求的平均输出 token 数

    Returns:
        包含详细估算信息的字典
    """
    input_tokens = count_messages_tokens(messages_list, model)
    output_tokens = len(messages_list) * avg_output_tokens
    cost = estimate_cost(input_tokens, output_tokens, model)

    return {
        "count": len(messages_list),
        "input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": round(cost, 4),
        "model": model,
    }


def _normalize_message_for_hash(message: Dict) -> Dict:
    """
    规范化消息用于 hash 计算，将 base64 图片替换为其内容 hash

    这样做的好处：
    1. 减少 hash 计算的数据量（base64 可能有几 MB）
    2. 同一张图片即使重新编码也会产生相同的缓存键
    """
    if not isinstance(message, dict):
        return message

    result = {}
    for key, value in message.items():
        if key == "content" and isinstance(value, list):
            # 处理多模态内容（OpenAI 格式）
            normalized_content = []
            for item in value:
                if isinstance(item, dict) and "image_url" in item:
                    image_url = item["image_url"]
                    if isinstance(image_url, dict):
                        url = image_url.get("url", "")
                    else:
                        url = str(image_url)

                    # 检查是否是 base64 数据
                    if url.startswith("data:image"):
                        # 提取 base64 部分并计算 hash
                        base64_data = url.split(",", 1)[-1] if "," in url else url
                        img_hash = hashlib.md5(base64_data.encode()).hexdigest()[:16]
                        # 用短 hash 替代完整 base64
                        normalized_item = {
                            "type": item.get("type", "image_url"),
                            "image_url": {"url": f"img_hash:{img_hash}"}
                        }
                        normalized_content.append(normalized_item)
                    else:
                        # URL 类型保持不变
                        normalized_content.append(item)
                else:
                    normalized_content.append(item)
            result[key] = normalized_content
        else:
            result[key] = value
    return result


def messages_hash(
    messages: List[Dict],
    model: str = "",
    **kwargs
) -> str:
    """
    生成 messages 的唯一哈希值，用于缓存键

    Args:
        messages: 消息列表
        model: 模型名称
        **kwargs: 其他影响结果的参数 (temperature, max_tokens 等)

    Returns:
        MD5 哈希字符串
    """
    # 规范化消息（优化 base64 图片的处理）
    normalized_messages = [_normalize_message_for_hash(m) for m in messages]

    # 构建要哈希的内容
    cache_key_data = {
        "messages": normalized_messages,
        "model": model,
        **{k: v for k, v in kwargs.items() if v is not None}
    }
    content = json.dumps(cache_key_data, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(content.encode()).hexdigest()
