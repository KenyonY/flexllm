from dataclasses import dataclass
from typing import Any


@dataclass
class RequestResult:
    """请求结果的数据类

    延迟按"归因"拆分，而非按代码阶段：

        latency = queue_time + service_time

    - queue_time：客户端自己造成的等待（并发 semaphore + max_qps 漏桶排队）
    - service_time：决定发出去之后拿到答案的耗时，含重试与重试间隔
      （重试由服务报错触发，归因于服务）；可由 latency - queue_time 推导，不单独存储

    latency 保持"端到端"语义不变：ETA、吞吐量以及写入 JSONL 的输出格式都依赖它。
    """

    request_id: int
    data: Any
    status: str
    latency: float
    meta: dict = None
    queue_time: float = 0.0
