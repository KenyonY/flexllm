"""Mock LLM Server for testing

这是 flexllm.mock 模块的便捷入口，支持作为独立脚本运行。

用法:
    python tests/mock_server.py 8001              # 固定延迟 0.1s
    python tests/mock_server.py 8001 --delay 0.5  # 固定延迟 0.5s
    python tests/mock_server.py 8001 --delay 5-10 # 随机延迟 5-10s
    python tests/mock_server.py 8001 --response-len 10-1000  # 响应长度
    python tests/mock_server.py 8001 --rps 10     # RPS 限制
    python tests/mock_server.py 8001 --token-rate 50  # 流式 token 速率

推荐使用 CLI:
    flexllm mock -p 8001 -d 0.1 -l 10-1000 --rps 10 --token-rate 50
"""

import argparse

# Re-export from flexllm.mock for backward compatibility
from flexllm.mock import (
    MockLLMServer,
    MockLLMServerGroup,
    MockServerConfig,
    RPSLimiter,
    parse_range,
)

__all__ = [
    "MockLLMServer",
    "MockLLMServerGroup",
    "MockServerConfig",
    "RPSLimiter",
    "parse_range",
]


def main():
    parser = argparse.ArgumentParser(description="Mock LLM Server")
    parser.add_argument("port", type=int, nargs="?", default=8001, help="端口号")
    parser.add_argument(
        "--delay", type=str, default="0.1", help="延迟时间，支持 '0.5' 或 '5-10' 格式"
    )
    parser.add_argument(
        "--response-len",
        type=str,
        default="10-1000",
        help="响应长度（字符），支持 '100' 或 '10-1000' 格式",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=0,
        help="每秒最大请求数，0 表示不限制",
    )
    parser.add_argument(
        "--token-rate",
        type=float,
        default=0,
        help="流式返回时每秒 token 数，0 表示不限制",
    )
    parser.add_argument(
        "--error-rate",
        type=float,
        default=0,
        help="请求失败率 (0-1)，0 表示不失败",
    )
    args = parser.parse_args()

    delay_min, delay_max = parse_range(args.delay, float)
    response_min_len, response_max_len = parse_range(args.response_len, int)

    config = MockServerConfig(
        port=args.port,
        delay_min=delay_min,
        delay_max=delay_max,
        response_min_len=response_min_len,
        response_max_len=response_max_len,
        rps=args.rps,
        token_rate=args.token_rate,
        error_rate=args.error_rate,
    )

    print(f"Mock LLM Server starting on port {args.port}")
    print(f"  Delay: {delay_min}-{delay_max}s")
    print(f"  Response length: {response_min_len}-{response_max_len} chars")
    if args.rps > 0:
        print(f"  RPS limit: {args.rps}")
    if args.token_rate > 0:
        print(f"  Token rate: {args.token_rate}/s (streaming)")
    if args.error_rate > 0:
        print(f"  Error rate: {args.error_rate * 100:.1f}%")
    print(f"  URL: http://localhost:{args.port}/v1")

    server = MockLLMServer(config)
    server.run()


if __name__ == "__main__":
    main()
