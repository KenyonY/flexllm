"""CLI 辅助工具函数"""

from __future__ import annotations

import json
import sys


def resolve_model_config(
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    required: bool = False,
) -> tuple[str | None, str | None, str | None]:
    """从配置解析模型连接参数，返回 (model, base_url, api_key)

    Args:
        model: 模型名称（CLI 传入）
        base_url: base_url（CLI 传入，优先级高于配置）
        api_key: api_key（CLI 传入，优先级高于配置）
        required: 为 True 时，配置不存在则打印错误并 raise typer.Exit(1)

    Returns:
        (model, base_url, api_key) 三元组，配置中的值合并 CLI 传入值
    """
    from .config import get_config

    config = get_config()
    model_config = config.get_model_config(model)

    if not model_config:
        if required:
            import typer

            print("错误: 未找到模型配置，使用 'flexllm list' 查看可用模型", file=sys.stderr)
            print(
                "提示: 设置环境变量 FLEXLLM_BASE_URL, FLEXLLM_API_KEY, FLEXLLM_MODEL"
                " 或创建 ~/.flexllm/config.yaml",
                file=sys.stderr,
            )
            raise typer.Exit(1)
        return model, base_url, api_key

    resolved_base_url = base_url or model_config.get("base_url")
    resolved_api_key = api_key or model_config.get("api_key", "EMPTY")

    # Claude provider 不需要用户提供 base_url，自动填充默认值
    if not resolved_base_url and model_config.get("provider") == "claude":
        resolved_base_url = "https://api.anthropic.com/v1"

    # model id 为空时，自动从 /v1/models 获取（仅 openai provider）
    resolved_model = model_config.get("id")
    if not resolved_model and resolved_base_url:
        provider = model_config.get("provider", "openai")
        if provider == "openai":
            resolved_model = _fetch_model_id(resolved_base_url, resolved_api_key)
    resolved_model = resolved_model or model

    return resolved_model, resolved_base_url, resolved_api_key


def _fetch_model_id(base_url: str, api_key: str = "EMPTY") -> str | None:
    """从 /v1/models 接口获取第一个可用的模型 ID"""
    import urllib.request

    url = f"{base_url.rstrip('/')}/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = data.get("data", [])
            if models:
                model_id = models[0].get("id")
                print(f"[auto-detect] 从 {base_url} 获取到模型: {model_id}", file=sys.stderr)
                return model_id
    except Exception as e:
        print(f"[auto-detect] 从 {base_url}/models 获取模型失败: {e}", file=sys.stderr)
    return None


def apply_user_template(content: str, template: str | None) -> str:
    """应用 user content 模板"""
    if not template:
        return content
    return template.format(content=content)


# ========== 输入格式处理 ==========


def detect_input_format(record: dict) -> tuple[str, list[str]]:
    """检测输入记录的格式类型

    支持的格式及优先级:
    1. openai_chat: 包含 "messages" 字段
    2. alpaca: 包含 "instruction" 字段（可选 "input"）
    3. simple: 包含 q/question/prompt/input/user 之一（可选 "system"）
       注意: "input" 仅在没有 "instruction" 时作为 simple 格式的 user content
    """
    if "messages" in record:
        return "openai_chat", ["messages"]
    if "instruction" in record:
        return "alpaca", ["instruction", "input"]
    for field in ["q", "question", "prompt", "input", "user"]:
        if field in record:
            return "simple", [field, "system"]
    return "unknown", []


def convert_to_messages(
    record: dict,
    format_type: str,
    message_fields: list[str],
    global_system: str = None,
    user_template: str = None,
) -> tuple[list[dict], dict]:
    """将输入记录转换为 messages 格式"""
    messages = []
    used_fields = set()

    if format_type == "openai_chat":
        messages = record["messages"]
        used_fields.add("messages")
        if user_template:
            messages = [
                {**msg, "content": apply_user_template(msg["content"], user_template)}
                if msg.get("role") == "user" and isinstance(msg.get("content"), str)
                else msg
                for msg in messages
            ]

    elif format_type == "alpaca":
        instruction = record.get("instruction", "")
        input_text = record.get("input", "")
        used_fields.update(["instruction", "input", "output"])

        system = record.get("system")
        if system:
            used_fields.add("system")
            messages.append({"role": "system", "content": system})

        content = instruction
        if input_text:
            content = f"{instruction}\n\n{input_text}"
        content = apply_user_template(content, user_template)
        messages.append({"role": "user", "content": content})

    elif format_type == "simple":
        prompt_field = None
        for field in ["q", "question", "prompt", "input", "user"]:
            if field in record:
                prompt_field = field
                break

        if prompt_field:
            used_fields.add(prompt_field)
            system = global_system or record.get("system")
            if "system" in record:
                used_fields.add("system")

            if system:
                messages.append({"role": "system", "content": system})
            user_content = apply_user_template(record[prompt_field], user_template)
            messages.append({"role": "user", "content": user_content})

    elif format_type == "custom":
        user_field, system_field = message_fields[0], message_fields[1]
        used_fields.add(user_field)

        system = None
        if system_field and system_field in record:
            system = record[system_field]
            used_fields.add(system_field)

        if system:
            messages.append({"role": "system", "content": system})
        user_content = apply_user_template(record[user_field], user_template)
        messages.append({"role": "user", "content": user_content})

    if global_system and format_type != "openai_chat":
        messages = [m for m in messages if m.get("role") != "system"]
        messages.insert(0, {"role": "system", "content": global_system})

    metadata = {k: v for k, v in record.items() if k not in used_fields}
    return messages, metadata


def load_jsonl_records(input_path: str = None) -> list[dict]:
    """从 JSONL 文件或 stdin 加载记录"""
    records = []

    if input_path:
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        for line in sys.stdin:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError("输入为空")

    return records


def parse_batch_input(
    input_path: str = None, skip_format_detection: bool = False
) -> tuple[list[dict], str, list[str]]:
    """解析批量输入文件或 stdin"""
    records = load_jsonl_records(input_path)

    if skip_format_detection:
        return records, "", []

    format_type, message_fields = detect_input_format(records[0])

    if format_type == "unknown":
        available_fields = list(records[0].keys())
        raise ValueError(
            f"无法识别输入格式，未找到以下字段之一：\n"
            f"  - messages (openai_chat 格式)\n"
            f"  - instruction (alpaca 格式)\n"
            f"  - q/question/prompt/input/user (simple 格式)\n\n"
            f"发现的字段: {available_fields}\n"
            f"提示: 使用 dtflow 转换格式: dt transform data.jsonl --preset=openai_chat"
        )

    return records, format_type, message_fields


def parse_thinking(value: str | None) -> bool | str | int | None:
    """解析 --thinking 参数值

    支持: true/false/low/medium/high/minimal 或整数(budget_tokens)
    """
    if value is None:
        return None
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("low", "medium", "high", "minimal"):
        return low
    try:
        return int(value)
    except ValueError:
        print(f"错误: --thinking 参数无效: {value}", file=sys.stderr)
        raise SystemExit(1)


def _detect_provider_from_key(api_key: str) -> str | None:
    """根据 API Key 前缀推断 provider 的 base_url"""
    prefix_map = {
        "sk-or-v1-": "https://openrouter.ai/api/v1",
        "sk-ant-": "https://api.anthropic.com",
    }
    for prefix, base_url in prefix_map.items():
        if api_key.startswith(prefix):
            return base_url
    return None


# 所有支持余额查询的 provider base_url（用于逐个尝试）
_CREDIT_PROVIDER_URLS = [
    "https://openrouter.ai/api/v1",
    "https://api.siliconflow.cn/v1",
    "https://api.deepseek.com",
    "https://aimlapi.com/v1",
    "https://api.openai.com/v1",
]


def query_credits_by_key(api_key: str) -> dict | None:
    """直接通过 API Key 查询余额，自动检测 provider"""
    # 先尝试根据 key 前缀检测
    detected_url = _detect_provider_from_key(api_key)
    if detected_url:
        result = query_credits(detected_url, api_key)
        if result is not None and "error" not in result:
            return result

    # 逐个尝试所有 provider
    for base_url in _CREDIT_PROVIDER_URLS:
        if base_url == detected_url:
            continue
        result = query_credits(base_url, api_key)
        if result is not None and "error" not in result:
            return result

    return None


def query_credits(base_url: str, api_key: str) -> dict | None:
    """查询 API Key 余额"""
    import requests

    headers = {"Authorization": f"Bearer {api_key}"}
    timeout = 15

    try:
        # OpenRouter
        if "openrouter.ai" in base_url:
            resp = requests.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers=headers,
                timeout=timeout,
            )
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

            data = resp.json().get("data", {})
            return {
                "provider": "OpenRouter",
                "data": {
                    "剩余额度": f"${data.get('limit_remaining') or 0:.2f}",
                    "总额度上限": f"${data.get('limit') or 0:.2f}",
                    "已使用": f"${data.get('usage') or 0:.2f}",
                    "今日消费": f"${data.get('usage_daily') or 0:.4f}",
                    "本月消费": f"${data.get('usage_monthly') or 0:.2f}",
                },
            }

        # SiliconFlow
        if "siliconflow.cn" in base_url:
            resp = requests.get(
                "https://api.siliconflow.cn/v1/user/info",
                headers=headers,
                timeout=timeout,
            )
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

            data = resp.json().get("data", {})
            return {
                "provider": "SiliconFlow",
                "data": {
                    "总余额": f"¥{data.get('totalBalance') or '0'}",
                    "充值余额": f"¥{data.get('chargeBalance') or '0'}",
                    "赠送余额": f"¥{data.get('balance') or '0'}",
                    "用户名": data.get("name") or "N/A",
                    "账户状态": data.get("status") or "N/A",
                },
            }

        # DeepSeek
        if "deepseek.com" in base_url:
            resp = requests.get(
                "https://api.deepseek.com/user/balance",
                headers=headers,
                timeout=timeout,
            )
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

            data = resp.json()
            balance_infos = data.get("balance_infos") or []
            if balance_infos:
                info = balance_infos[0]
                return {
                    "provider": "DeepSeek",
                    "data": {
                        "总余额": f"{info.get('currency') or 'CNY'} {info.get('total_balance') or '0'}",
                        "赠送余额": f"{info.get('currency') or 'CNY'} {info.get('granted_balance') or '0'}",
                        "充值余额": f"{info.get('currency') or 'CNY'} {info.get('topped_up_balance') or '0'}",
                        "余额充足": "是" if data.get("is_available") else "否",
                    },
                }
            return {
                "provider": "DeepSeek",
                "data": {"余额充足": "是" if data.get("is_available") else "否"},
            }

        # AI/ML API
        if "aimlapi.com" in base_url:
            resp = requests.get(
                "https://billing.aimlapi.com/v1/billing/balance",
                headers=headers,
                timeout=timeout,
            )
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

            data = resp.json()
            credits = data.get("balance") or 0
            usd_value = credits / 2_000_000
            return {
                "provider": "AI/ML API",
                "data": {
                    "Credits": f"{credits:,.0f}",
                    "折合美元": f"${usd_value:.4f}",
                },
            }

        # OpenAI
        if "api.openai.com" in base_url:
            resp = requests.get(
                "https://api.openai.com/v1/dashboard/billing/credit_grants",
                headers=headers,
                timeout=timeout,
            )
            if resp.status_code == 404:
                return {"error": "OpenAI 余额查询 API 不可用（可能已弃用）"}
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

            data = resp.json()
            grants = (data.get("grants") or {}).get("data") or []
            total_granted = sum((g.get("grant_amount") or 0) for g in grants) / 100
            total_used = sum((g.get("used_amount") or 0) for g in grants) / 100
            total_available = (data.get("total_available") or 0) / 100
            total_used_all = (data.get("total_used") or 0) / 100

            return {
                "provider": "OpenAI",
                "data": {
                    "可用余额": f"${total_available:.2f}",
                    "已使用": f"${total_used_all:.2f}",
                    "总授予额度": f"${total_granted:.2f}",
                },
            }

        return None

    except requests.exceptions.RequestException as e:
        return {"error": f"请求失败: {e}"}
    except Exception as e:
        return {"error": f"解析失败: {e}"}
