"""
模型定价模块

提供模型定价数据加载和成本估算功能。
定价数据存储在 data.json 中，可通过 `flexllm pricing --update` 更新。
"""

import json
from pathlib import Path
from typing import Dict, Optional

# 定价文件路径
PRICING_FILE = Path(__file__).parent / "data.json"

# 缓存的定价数据
_pricing_cache: dict[str, dict[str, float]] | None = None


def _load_pricing() -> dict[str, dict[str, float]]:
    """
    从 data.json 加载定价数据

    Returns:
        {model_name: {"input": price_per_token, "output": price_per_token}}
    """
    if not PRICING_FILE.exists():
        return {}

    try:
        with open(PRICING_FILE, encoding="utf-8") as f:
            data = json.load(f)

        models = data.get("models", {})
        # 将 $/1M tokens 转换为 $/token
        return {
            name: {"input": p["input"] / 1e6, "output": p["output"] / 1e6}
            for name, p in models.items()
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def get_pricing() -> dict[str, dict[str, float]]:
    """获取定价数据（带缓存）"""
    global _pricing_cache
    if _pricing_cache is None:
        _pricing_cache = _load_pricing()
    return _pricing_cache


def reload_pricing():
    """重新加载定价数据（用于更新后刷新）"""
    global _pricing_cache
    _pricing_cache = _load_pricing()


def get_model_pricing(model: str) -> dict[str, float] | None:
    """
    获取指定模型的定价

    Args:
        model: 模型名称（支持模糊匹配）

    Returns:
        {"input": price_per_token, "output": price_per_token} 或 None
    """
    pricing = get_pricing()

    # 精确匹配
    if model in pricing:
        return pricing[model]

    # 模糊匹配：检查模型名是否包含在 key 中
    model_lower = model.lower()
    for key in pricing:
        if model_lower in key.lower():
            return pricing[key]

    return None


def estimate_cost(input_tokens: int, output_tokens: int = 0, model: str = "gpt-4o") -> float:
    """
    估算 API 调用成本

    Args:
        input_tokens: 输入 token 数
        output_tokens: 输出 token 数
        model: 模型名称

    Returns:
        估算成本 (美元)
    """
    pricing = get_model_pricing(model)

    if not pricing:
        # 默认使用 gpt-4o-mini 的价格
        pricing = get_model_pricing("gpt-4o-mini") or {"input": 0.15e-6, "output": 0.6e-6}

    return input_tokens * pricing["input"] + output_tokens * pricing["output"]
