"""flexllm batch CLI 工作流示例

流程：本脚本生成数据集 -> `flexllm batch` 批量调用 -> 读取输出 JSONL。

用法（均在 examples/batch_cli/ 目录下执行）：
    python generate_dataset.py     # 生成单轮 + 多轮两份数据集
    flexllm batch data/demo_dataset.jsonl -o data/demo_output.jsonl
    flexllm batch data/demo_dataset_multiturn.jsonl -o data/demo_output_multiturn.jsonl

两份数据集演示 batch 自动识别的两种格式：
    demo_dataset.jsonl            simple 格式，question 字段，结构上只能是单轮
    demo_dataset_multiturn.jsonl  openai_chat 格式，messages 数组原样透传，可多轮
两者的 id/category 等未被消费的字段都会自动透传到输出的 metadata 中。
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATASET = DATA_DIR / "demo_dataset.jsonl"
DATASET_MULTITURN = DATA_DIR / "demo_dataset_multiturn.jsonl"

# 每条样本可单独指定 system；不指定时由 CLI 的 -s 或配置文件提供
SYSTEM = "你是一个简洁的助手，回答控制在 50 字以内。"

SAMPLES = [
    {"id": "s01", "category": "常识", "question": "用一句话解释什么是光合作用。"},
    {"id": "s02", "category": "数学", "question": "123 * 45 等于多少？只输出数字。"},
    {"id": "s03", "category": "编程", "question": "用 Python 写一行代码反转字符串 s。"},
    {"id": "s04", "category": "翻译", "question": "把「今天天气不错」翻译成英文。"},
    {"id": "s05", "category": "常识", "question": "水的沸点在标准大气压下是多少摄氏度？"},
    {"id": "s06", "category": "写作", "question": "给一款降噪耳机写一句 15 字以内的广告语。"},
    {"id": "s07", "category": "编程", "question": "Python 中列表和元组最主要的区别是什么？"},
    {"id": "s08", "category": "地理", "question": "长江和黄河哪条更长？只回答河流名称。"},
    {"id": "s09", "category": "逻辑", "question": "小明比小红高，小红比小刚高，谁最矮？"},
    {"id": "s10", "category": "总结", "question": "用三个关键词概括「深度学习」。"},
]

# 多轮：messages 为完整对话历史，最后一条必须是 user，模型续写下一条 assistant
MULTITURN_SAMPLES = [
    {
        "id": "m01",
        "category": "编程",
        "turns": [
            ("user", "我想学 Python，从哪开始？"),
            ("assistant", "先掌握基础语法：变量、条件、循环、函数。"),
            ("user", "这些学完之后，下一步学什么？"),
        ],
    },
    {
        "id": "m02",
        "category": "旅行",
        "turns": [
            ("user", "推荐一个适合秋天去的城市。"),
            ("assistant", "推荐京都，红叶季节景色很好。"),
            ("user", "那里待三天的话，怎么安排？"),
        ],
    },
    {
        "id": "m03",
        "category": "数学",
        "turns": [
            ("user", "一个长方形长 8 宽 5，面积是多少？"),
            ("assistant", "面积是 40。"),
            ("user", "如果长和宽都翻倍，面积变成多少？"),
        ],
    },
    {
        "id": "m04",
        "category": "写作",
        "turns": [
            ("user", "帮我给读书笔记起个标题。"),
            ("assistant", "可以叫《读书的痕迹》。"),
            ("user", "换一个更活泼的风格。"),
        ],
    },
    {
        "id": "m05",
        "category": "调试",
        "turns": [
            ("user", "我的 Python 代码报 IndexError 是什么原因？"),
            ("assistant", "通常是访问了超出序列长度的下标。"),
            ("user", "遍历列表时怎么避免这个错误？"),
        ],
    },
]


def main():
    DATA_DIR.mkdir(exist_ok=True)

    with DATASET.open("w", encoding="utf-8") as f:
        for sample in SAMPLES:
            f.write(json.dumps({**sample, "system": SYSTEM}, ensure_ascii=False) + "\n")

    with DATASET_MULTITURN.open("w", encoding="utf-8") as f:
        for sample in MULTITURN_SAMPLES:
            messages = [{"role": "system", "content": SYSTEM}]
            messages += [{"role": role, "content": content} for role, content in sample["turns"]]
            record = {"id": sample["id"], "category": sample["category"], "messages": messages}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"已生成 {len(SAMPLES)} 条单轮样本: {DATASET}")
    print(f"已生成 {len(MULTITURN_SAMPLES)} 条多轮样本: {DATASET_MULTITURN}")
    print("\n下一步（在本目录下执行，会自动读取 ./flexllm_config.yaml）:")
    print("  flexllm batch data/demo_dataset.jsonl -o data/demo_output.jsonl")
    print("  flexllm batch data/demo_dataset_multiturn.jsonl -o data/demo_output_multiturn.jsonl")
    print("  flexllm batch data/demo_dataset.jsonl -m glm52 -c 5   # 指定模型和并发")
    print("  flexllm batch data/demo_dataset.jsonl --dry-run       # 只预览处理计划")


if __name__ == "__main__":
    main()
