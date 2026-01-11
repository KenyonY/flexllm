"""
批量请求 + 元数据保存示例

演示如何使用 metadata_list 参数保存额外信息到输出文件，
适用于需要追踪数据来源的批量处理场景。
"""

import asyncio
import json
from pathlib import Path

from flexllm import LLMClient


async def main():
    # 初始化客户端
    client = LLMClient(
        model="Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        base_url="http://localhost:8000/v1",
        api_key="your-api-key",
    )

    # 模拟从文件读取的数据
    source_data = [
        {"id": "q001", "question": "10+10等于多少？", "category": "数学"},
        {"id": "q002", "question": "20+20等于多少？", "category": "数学"},
        {"id": "q003", "question": "30+30等于多少？", "category": "数学"},
    ]

    # 构造 messages_list
    messages_list = [[{"role": "user", "content": item["question"]}] for item in source_data]

    # 构造 metadata_list（保存原始数据的 id 和 category）
    metadata_list = [{"id": item["id"], "category": item["category"]} for item in source_data]

    output_file = "batch_results.jsonl"

    # 批量请求
    results = await client.chat_completions_batch(
        messages_list=messages_list,
        metadata_list=metadata_list,
        output_file=output_file,
        show_progress=True,
    )
    

    print(f"\n完成 {len(results)} 条请求")
    print(f"结果已保存到: {output_file}")

    # 读取并展示输出文件内容
    print("\n输出文件内容示例:")
    print("-" * 50)
    with open(output_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 2:  # 只展示前2条
                print("...")
                break
            record = json.loads(line)
            print(f"ID: {record['metadata']['id']}")
            print(f"Category: {record['metadata']['category']}")
            output = record['output'][:100] if record['output'] else "(error)"
            print(f"Output: {output}...")
            print("-" * 50)

    # 清理
    # Path(output_file).unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
