"""测试 vllm 服务调用"""

from maque import MeasureTime

from flexllm import LLMClient, LLMClientPool

client = LLMClient(
    **{"base_url": "http://100.95.177.11:8000/qwen25/v1", "model": "Qwen2.5-32B-Instruct"},
    api_key="dummy",
    concurrency_limit=10,
)

pool = LLMClientPool(
    endpoints=[
        {"base_url": "http://100.95.177.11:8000/qwen3/v1", "model": "Qwen3-32B"},
        {"base_url": "http://100.95.177.11:8000/qwen25/v1", "model": "Qwen2.5-32B-Instruct"},
    ],
    load_balance="round_robin",  # or "weighted", "random", "fallback"
    fallback=True,  # Auto-switch on failure
    concurrency_limit=10,
)

# 单次调用
# messages = [{"role": "user", "content": "用一句话介绍自己"}]
# print("单次调用:", client.chat_completions_sync(messages, thinking=False))

# 批量调用
batch_messages = [[{"role": "user", "content": "用一句话介绍自己"}] for _ in range(5)]
mt = MeasureTime()
results = client.chat_completions_batch_sync(
    batch_messages,
    # output_jsonl="output.jsonl",
    track_cost=True,
    thinking=False,
)
mt.show_interval()
print(results[0])
# results_v2 = client.chat_completions_batch_sync(batch_messages, thinking=False,
#                                                 track_cost=True,
#                                                 )
# mt.show_interval()
# print(results_v2[0])
print("批量调用完成，结果已保存到 .jsonl")
