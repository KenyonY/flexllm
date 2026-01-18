#!/usr/bin/env python3

"""
模型定价自动更新脚本

从 OpenRouter API 获取所有模型定价，更新到 pricing.json

使用方法:
    # 预览更新内容
    python -m flexllm.pricing.updater

    # 直接更新 pricing.json
    python -m flexllm.pricing.updater --apply

    # 输出 JSON 格式
    python -m flexllm.pricing.updater --json
"""

import argparse
import json
import re
import urllib.request
from datetime import datetime
from pathlib import Path

# OpenRouter API 端点
OPENROUTER_API = "https://openrouter.ai/api/v1/models"

# 定价文件路径
PRICING_FILE = Path(__file__).parent / "data.json"

# 排除的模型模式
EXCLUDE_PATTERNS = [
    r":free$",  # 免费模型
    r":floor$",  # floor 模型
    r":extended$",  # extended 模型
]


def fetch_models() -> list[dict]:
    """从 OpenRouter API 获取模型列表"""
    req = urllib.request.Request(
        OPENROUTER_API, headers={"User-Agent": "flexllm-pricing-updater/1.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("data", [])


def should_exclude(model_id: str) -> bool:
    """检查模型是否应该排除"""
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, model_id):
            return True
    return False


def normalize_model_id(model_id: str) -> str | None:
    """
    将 OpenRouter 模型 ID 规范化

    规则: 直接取斜杠后面的部分
    例如: openai/gpt-4o -> gpt-4o
          anthropic/claude-3.5-sonnet -> claude-3.5-sonnet
    """
    if should_exclude(model_id):
        return None

    # 移除 :thinking 等后缀
    clean_id = re.sub(r":\w+$", "", model_id)

    # 取斜杠后面的部分
    if "/" in clean_id:
        return clean_id.split("/", 1)[1]

    return clean_id


def parse_pricing(model: dict) -> tuple[float, float] | None:
    """
    解析模型定价信息

    Returns:
        (input_price, output_price) 单位: $/1M tokens
    """
    pricing = model.get("pricing", {})

    prompt_price = pricing.get("prompt", "0")
    completion_price = pricing.get("completion", "0")

    try:
        # OpenRouter 返回的是 $/token，转换为 $/1M tokens
        input_price = float(prompt_price) * 1e6
        output_price = float(completion_price) * 1e6

        # 过滤免费模型和价格异常的模型
        if input_price == 0 and output_price == 0:
            return None
        if input_price < 0 or output_price < 0:
            return None

        return (round(input_price, 4), round(output_price, 4))
    except (ValueError, TypeError):
        return None


def collect_pricing() -> dict[str, dict[str, float]]:
    """
    收集所有模型的定价信息

    Returns:
        {model_name: {"input": price_per_1m, "output": price_per_1m}}
    """
    models = fetch_models()
    pricing_map = {}

    for model in models:
        model_id = model.get("id", "")
        name = normalize_model_id(model_id)

        if not name:
            continue

        pricing = parse_pricing(model)
        if not pricing:
            continue

        input_price, output_price = pricing

        # 如果同一模型有多个版本，保留价格最低的
        if name in pricing_map:
            existing = pricing_map[name]
            if input_price >= existing["input"]:
                continue

        pricing_map[name] = {
            "input": input_price,
            "output": output_price,
        }

    return pricing_map


def update_pricing_file(pricing_map: dict[str, dict[str, float]]) -> bool:
    """
    更新 data.json 文件

    Args:
        pricing_map: {model_name: {"input": price, "output": price}}

    Returns:
        是否更新成功
    """
    # 按模型名排序
    sorted_models = dict(sorted(pricing_map.items()))

    data = {
        "_meta": {
            "updated_at": datetime.now().strftime("%Y-%m-%d"),
            "source": "OpenRouter API (https://openrouter.ai/api/v1/models)",
            "unit": "$/1M tokens",
        },
        "models": sorted_models,
    }

    try:
        with open(PRICING_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error writing {PRICING_FILE}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="从 OpenRouter API 更新模型定价表")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="直接更新 data.json（默认只预览）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出 JSON 格式",
    )
    args = parser.parse_args()

    print("Fetching models from OpenRouter API...")
    pricing_map = collect_pricing()
    print(f"Found {len(pricing_map)} models with pricing info")

    if args.json:
        print(json.dumps(pricing_map, indent=2, ensure_ascii=False))
        return

    if args.apply:
        print(f"\nUpdating {PRICING_FILE}...")
        if update_pricing_file(pricing_map):
            print(f"✓ Successfully updated data.json ({len(pricing_map)} models)")
        else:
            print("✗ Failed to update data.json")
            exit(1)
    else:
        print("\n" + "=" * 60)
        print("Preview (use --apply to update data.json):")
        print("=" * 60 + "\n")

        for name in sorted(pricing_map.keys())[:30]:
            p = pricing_map[name]
            print(f"  {name:<40} ${p['input']:<10.4f} / ${p['output']:<10.4f}")

        if len(pricing_map) > 30:
            print(f"  ... and {len(pricing_map) - 30} more models")


if __name__ == "__main__":
    main()
