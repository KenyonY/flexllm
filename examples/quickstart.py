"""
flexllm 快速入门示例
===================
两分钟掌握核心功能

安装: pip install flexllm[all]
"""

import asyncio
from flexllm import LLMClient, GeminiClient, LLMClientPool, ResponseCacheConfig

# ============================================================
# 1. 基础用法 - 单次请求
# ============================================================

client = LLMClient(
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
)

messages = [{"role": "user", "content": "你好，用一句话介绍自己"}]

# 同步调用
# response = client.chat_completions_sync(messages)
# print(response)


# 异步调用
async def single_request():
    response = await client.chat_completions(messages)
    print(response)


# ============================================================
# 2. 批量处理 + 断点续传
# ============================================================
# 核心特性：中断后重新运行，自动从断点恢复

async def batch_processing():
    client = LLMClient(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="your-api-key",
        concurrency_limit=50,  # 并发数
        max_qps=100,           # QPS 限制
    )

    messages_list = [
        [{"role": "user", "content": f"{i}+{i}等于多少？"}]
        for i in range(100)
    ]

    results = await client.chat_completions_batch(
        messages_list,
        output_file="results.jsonl",  # 指定文件即支持断点续传
        show_progress=True,
    )
    return results


# ============================================================
# 3. 响应缓存 - 避免重复调用
# ============================================================

async def with_cache():
    client = LLMClient(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="your-api-key",
        cache=ResponseCacheConfig(enabled=True, ttl=3600),  # 1小时缓存
    )

    # 第一次调用 -> API 请求
    r1 = await client.chat_completions(messages)

    # 第二次调用 -> 直接读缓存（瞬间返回）
    r2 = await client.chat_completions(messages)


# ============================================================
# 4. 流式输出
# ============================================================

async def streaming():
    async for chunk in client.chat_completions_stream(messages):
        print(chunk, end="", flush=True)


# ============================================================
# 5. 多 Provider 支持
# ============================================================

# Gemini
gemini = GeminiClient(model="gemini-2.5-flash", api_key="your-gemini-key")

# Claude (通过 provider 参数)
claude = LLMClient(
    provider="claude",
    model="claude-3-5-sonnet-20241022",
    api_key="your-anthropic-key",
)

# 本地模型 (Ollama/vLLM)
local = LLMClient(
    model="qwen2.5:7b",
    base_url="http://localhost:11434/v1",
    api_key="EMPTY",
)


# ============================================================
# 6. 负载均衡 - 多节点故障转移
# ============================================================

async def load_balancing():
    pool = LLMClientPool(
        endpoints=[
            {"base_url": "http://host1:8000/v1", "api_key": "k1", "model": "qwen"},
            {"base_url": "http://host2:8000/v1", "api_key": "k2", "model": "qwen"},
        ],
        load_balance="round_robin",  # round_robin/weighted/random/fallback
        fallback=True,               # 节点故障自动切换
    )

    # 和 LLMClient API 完全一致
    result = await pool.chat_completions(messages)

    # 批量请求自动分发到多节点
    messages_list = [[{"role": "user", "content": f"问题{i}"}] for i in range(10)]
    results = await pool.chat_completions_batch(messages_list, distribute=True)


# ============================================================
# 7. 工具调用 (Function Calling)
# ============================================================

async def tool_use():
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    }]

    result = await client.chat_completions(
        messages=[{"role": "user", "content": "东京天气怎么样？"}],
        tools=tools,
    )

    if result.tool_calls:
        for call in result.tool_calls:
            print(f"调用: {call.function['name']}({call.function['arguments']})")


# ============================================================
# CLI 使用 (命令行)
# ============================================================
"""
# 快速问答
flexllm ask "什么是Python？"

# 交互聊天
flexllm chat

# 批量处理
flexllm batch input.jsonl -o output.jsonl

# 配置管理
flexllm init          # 初始化配置
flexllm list          # 列出模型
flexllm set-model gpt-4  # 设置默认模型
flexllm test          # 测试连接
"""


# ============================================================
# 运行示例
# ============================================================
if __name__ == "__main__":
    # 取消注释以运行示例
    # asyncio.run(single_request())
    # asyncio.run(batch_processing())
    # asyncio.run(with_cache())
    # asyncio.run(streaming())
    pass
