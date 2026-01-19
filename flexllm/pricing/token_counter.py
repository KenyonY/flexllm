#! /usr/bin/env python3

"""
Token 计数和成本估算模块

支持使用 tiktoken 精确计算，或在缺失时使用估算方法。
定价数据从 pricing/data.json 加载，可通过 `flexllm pricing --update` 更新。
"""

import hashlib
import json
from typing import Any

# tiktoken 是可选依赖
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# 延迟导入 pricing 模块功能，避免循环引用
_pricing_module = None


def _get_pricing_module():
    global _pricing_module
    if _pricing_module is None:
        from . import estimate_cost as ec
        from . import get_pricing as gp

        _pricing_module = {"get_pricing": gp, "estimate_cost": ec}
    return _pricing_module


# 兼容旧 API：MODEL_PRICING 现在是动态获取的
def _get_model_pricing():
    return _get_pricing_module()["get_pricing"]()


# 使用类属性实现延迟加载
class _LazyModelPricing:
    _cache = None

    def __getitem__(self, key):
        if self._cache is None:
            self._cache = _get_model_pricing()
        return self._cache.get(key)

    def __iter__(self):
        if self._cache is None:
            self._cache = _get_model_pricing()
        return iter(self._cache)

    def get(self, key, default=None):
        if self._cache is None:
            self._cache = _get_model_pricing()
        return self._cache.get(key, default)

    def items(self):
        if self._cache is None:
            self._cache = _get_model_pricing()
        return self._cache.items()

    def keys(self):
        if self._cache is None:
            self._cache = _get_model_pricing()
        return self._cache.keys()

    def __contains__(self, key):
        if self._cache is None:
            self._cache = _get_model_pricing()
        return key in self._cache


# 为了向后兼容，保留 MODEL_PRICING 变量（延迟加载）
MODEL_PRICING = _LazyModelPricing()

# 模型到 tiktoken 编码器的映射
MODEL_TO_ENCODING = {
    # GPT-5 系列
    "gpt-5": "o200k_base",
    "gpt-5.1": "o200k_base",
    "gpt-5.1-codex": "o200k_base",
    "gpt-5.2": "o200k_base",
    "gpt-5.2-pro": "o200k_base",
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
    "o1-pro": "o200k_base",
    "o3": "o200k_base",
    "o3-mini": "o200k_base",
    "o4-mini": "o200k_base",
}

# 编码器缓存
_encoding_cache: dict[str, Any] = {}


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
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_chars = len(text) - chinese_chars
    return chinese_chars // 2 + other_chars // 4 + 1


def count_tokens(content: str | list[dict] | dict, model: str = "gpt-4o") -> int:
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


def count_messages_tokens(messages_list: list[list[dict]], model: str = "gpt-4o") -> int:
    """
    批量计算 messages 的 token 总数

    Args:
        messages_list: messages 列表的列表
        model: 模型名称

    Returns:
        总 token 数量
    """
    return sum(count_tokens(msgs, model) for msgs in messages_list)


def estimate_cost(input_tokens: int, output_tokens: int = 0, model: str = "gpt-4o") -> float:
    """
    估算 API 调用成本

    Args:
        input_tokens: 输入 token 数
        output_tokens: 输出 token 数 (如果未知可传 0)
        model: 模型名称

    Returns:
        估算成本 (美元)
    """
    return _get_pricing_module()["estimate_cost"](input_tokens, output_tokens, model)


def estimate_batch_cost(
    messages_list: list[list[dict]], model: str = "gpt-4o", avg_output_tokens: int = 500
) -> dict[str, Any]:
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


def _normalize_message_for_hash(message: dict) -> dict:
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
                            "image_url": {"url": f"img_hash:{img_hash}"},
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


def messages_hash(messages: list[dict], model: str = "", **kwargs) -> str:
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
        **{k: v for k, v in kwargs.items() if v is not None},
    }
    content = json.dumps(cache_key_data, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(content.encode()).hexdigest()
