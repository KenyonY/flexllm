# Roadmap

记录 flexllm 后续计划实现的特性。

## 自适应速率控制 (Adaptive Rate Control)

### 背景

当前 `max_qps` 是静态配置。遇到 429 时已能按服务端 `Retry-After` 或指数退避正确等待
（见 [advanced.md 自动重试](advanced.md#自动重试)），但请求速率本身不会下调——
退避只作用于失败的那条请求，其余并发请求仍以原速率涌向已限流的服务端。

### 目标

遇到 429 时自动降速，成功时逐步恢复，无需手动调优。

### 设计

#### 与现有参数的关系

- `max_qps` 参数保持不变，作为**硬上限**
- 新增 `adaptive=True` 启用自适应模式
- 启用后，实际 QPS 在 `[min_qps, max_qps]` 范围内动态调整

```python
# 现有用法（不变）
requester = ConcurrentRequester(concurrency_limit=10, max_qps=50)

# 启用自适应（新增）
requester = ConcurrentRequester(
    concurrency_limit=10,
    max_qps=50,        # 硬上限
    adaptive=True,     # 启用自适应
    min_qps=5,         # 可选，默认 1.0
)
```

#### 实现方案

扩展 `async_api/core.py` 的 `RateLimiter` 类。注意现有实现基于 aiolimiter 的
`AsyncLimiter` 且做了 lazy init（绑定当前 event loop），动态改速率需要重建 limiter：

```python
class RateLimiter:
    def __init__(
        self,
        max_qps: float | None = None,
        adaptive: bool = False,
        min_qps: float = 1.0,
        backoff_factor: float = 0.5,
        recovery_factor: float = 1.1,
        recovery_threshold: int = 10,
    ):
        self.max_qps = max_qps
        self.min_qps = min_qps
        self.adaptive = adaptive
        self.current_qps = max_qps  # 从最大值开始

        # 自适应参数
        self.backoff_factor = backoff_factor
        self.recovery_factor = recovery_factor
        self.recovery_threshold = recovery_threshold
        self._success_count = 0

    def on_rate_limit(self):
        """429 时调用：立即降速"""
        if self.adaptive and self.current_qps:
            self.current_qps = max(
                self.current_qps * self.backoff_factor,
                self.min_qps
            )
            self._success_count = 0

    def on_success(self):
        """成功时调用：逐步恢复"""
        if self.adaptive and self.current_qps:
            self._success_count += 1
            if self._success_count >= self.recovery_threshold:
                self.current_qps = min(
                    self.current_qps * self.recovery_factor,
                    self.max_qps
                )
                self._success_count = 0
```

#### 集成到错误处理

`_make_requests` 已把 429 归类为 `RetryableHTTPError`（携带 `status_code` 与
`retry_after`），在 `ConcurrentRequester._send_single_request` 的重试出口处挂钩即可：

```python
# _send_single_request 中
except RetryableHTTPError as e:
    if e.status_code == 429:
        self._rate_limiter.on_rate_limit()
    ...

# 成功时
self._rate_limiter.on_success()
```

#### 使用示例

```python
from flexllm import LLMClient

# 方式1：简单启用
client = LLMClient(model="gpt-4", max_qps=50, adaptive=True)

# 方式2：自定义参数
client = LLMClient(
    model="gpt-4",
    max_qps=100,
    adaptive=True,
    min_qps=10,
)

results = await client.chat_completions_batch(messages_list)
```

CLI 配置：

```yaml
# ~/.flexllm/config.yaml
batch:
  max_qps: 50
  adaptive: true
  min_qps: 5
```

### 可选扩展

- 进度条显示当前 QPS

### 已完成

- 解析 `Retry-After` 响应头精确等待 + 指数退避与抖动
