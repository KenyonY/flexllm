"""
模型定价模块

提供模型定价数据加载、成本估算和成本追踪功能。
定价数据存储在 data.json 中，可通过 `flexllm pricing --update` 更新。

加载优先级：~/.flexllm/pricing_data.json > 打包的 data.json
每天首次调用 get_pricing() 时自动后台更新。
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path

# 导出 cost_tracker 模块
from .cost_tracker import BudgetExceededError, CostReport, CostTracker, CostTrackerConfig

# 导出 token_counter 模块
from .token_counter import (
    MODEL_PRICING,
    count_messages_tokens,
    count_tokens,
    estimate_batch_cost,
    messages_hash,
)

logger = logging.getLogger(__name__)

# 打包的定价文件（fallback）
_BUNDLED_PRICING_FILE = Path(__file__).parent / "data.json"

# 用户数据目录的定价文件（优先）
_USER_PRICING_FILE = Path.home() / ".flexllm" / "pricing_data.json"

# 缓存的定价数据
_pricing_cache: dict[str, dict[str, float]] | None = None


def _get_pricing_file() -> Path:
    """返回应读取的定价文件路径（用户目录优先）"""
    if _USER_PRICING_FILE.exists():
        return _USER_PRICING_FILE
    return _BUNDLED_PRICING_FILE


def _load_pricing_from_file(path: Path) -> dict[str, dict[str, float]]:
    """从指定文件加载定价数据"""
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        models = data.get("models", {})
        return {
            name: {"input": p["input"] / 1e6, "output": p["output"] / 1e6}
            for name, p in models.items()
        }
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # 文件损坏（如并发写坏）不阻断使用，但必须留下线索，否则全部模型静默变为无定价
        logger.warning("定价文件 %s 解析失败，本次按无定价数据处理: %s", path, e)
        return {}


def _load_pricing() -> dict[str, dict[str, float]]:
    """加载定价数据（用户目录优先，打包文件兜底）"""
    return _load_pricing_from_file(_get_pricing_file())


def _get_updated_date() -> str | None:
    """获取当前定价数据的更新日期"""
    path = _get_pricing_file()
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("_meta", {}).get("updated_at")
    except (json.JSONDecodeError, KeyError):
        return None


def _auto_update_if_stale():
    """若定价数据非今天更新，则后台静默更新"""
    try:
        updated = _get_updated_date()
        today = datetime.now().strftime("%Y-%m-%d")
        if updated == today:
            return

        from .updater import collect_pricing, save_pricing_data

        pricing_map = collect_pricing()
        if pricing_map:
            save_pricing_data(pricing_map, _USER_PRICING_FILE)
            # 刷新缓存
            global _pricing_cache
            _pricing_cache = _load_pricing()
            logger.debug("定价数据已自动更新 (%d 个模型)", len(pricing_map))
    except Exception as e:
        # 定价自动更新失败不影响正常使用，但要留下线索
        logger.debug("定价数据自动更新失败: %s", e)


def get_pricing() -> dict[str, dict[str, float]]:
    """获取定价数据（带缓存，每日自动更新）"""
    global _pricing_cache
    if _pricing_cache is None:
        _pricing_cache = _load_pricing()
        # 后台检查是否需要更新
        threading.Thread(target=_auto_update_if_stale, daemon=True).start()
    return _pricing_cache


def reload_pricing():
    """重新加载定价数据（用于更新后刷新）"""
    global _pricing_cache
    _pricing_cache = _load_pricing()


def _normalize_model_name(model: str) -> str:
    """
    规范化模型名称（取斜杠后部分）

    例如: qwen/qwen3-32b -> qwen3-32b
          Qwen/Qwen3-32B -> qwen3-32b
    """
    # 去除首尾斜杠
    model = model.strip("/")

    # 取斜杠后部分
    if "/" in model:
        model = model.split("/", 1)[1]

    return model.lower()


def get_model_pricing(model: str) -> dict[str, float] | None:
    """
    获取指定模型的定价

    Args:
        model: 模型名称（支持规范化匹配）

    Returns:
        {"input": price_per_token, "output": price_per_token} 或 None
    """
    pricing = get_pricing()

    # 去除首尾斜杠
    model = model.strip("/")

    # Step 1: 精确匹配（大小写不敏感）
    model_lower = model.lower()
    for key in pricing:
        if model_lower == key.lower():
            return pricing[key]

    # Step 2: 规范化后匹配（取斜杠后部分）
    normalized = _normalize_model_name(model)
    for key in pricing:
        if normalized == key.lower():
            return pricing[key]

    return None


# 已提示过"未计费"的未知模型，避免批量调用时刷屏
_unknown_models_logged: set[str] = set()


def estimate_cost(input_tokens: int, output_tokens: int = 0, model: str = "gpt-4o") -> float:
    """
    估算 API 调用成本

    查不到定价的模型（如自建/免费模型）返回 0.0，不做任何 fallback 计费：
    按别的模型的价格记假账会积累出虚假成本，甚至触发预算熔断。

    Args:
        input_tokens: 输入 token 数
        output_tokens: 输出 token 数
        model: 模型名称

    Returns:
        估算成本 (美元)；无定价数据时为 0.0
    """
    pricing = get_model_pricing(model)

    if not pricing:
        if model not in _unknown_models_logged:
            _unknown_models_logged.add(model)
            logger.info("模型 %s 无定价数据，成本按 0 计（不计费）", model)
        return 0.0

    return input_tokens * pricing["input"] + output_tokens * pricing["output"]


__all__ = [
    # 成本追踪
    "CostTracker",
    "CostTrackerConfig",
    "CostReport",
    "BudgetExceededError",
    # Token 计数
    "count_tokens",
    "count_messages_tokens",
    "estimate_cost",
    "estimate_batch_cost",
    "messages_hash",
    "MODEL_PRICING",
    # 定价数据
    "get_pricing",
    "reload_pricing",
    "get_model_pricing",
]
