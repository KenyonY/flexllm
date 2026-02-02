"""
flexllm CLI - LLM 客户端命令行工具

提供简洁的 LLM 调用命令:
    flexllm ask "你的问题"
    flexllm chat
    flexllm batch input.jsonl -o output.jsonl
    flexllm models
    flexllm test
    flexllm agent "任务描述" --tools shell
"""

from .cli import main

if __name__ == "__main__":
    main()
