"""
成本追踪模块

提供批量处理过程中的实时成本追踪、预算控制和成本报告功能。

Example:
    # 仅追踪成本
    client = LLMClient(..., cost_tracker=True)
    results, cost_report = await client.chat_completions_batch(
        messages_list, return_cost_report=True
    )
    print(f"Total: ${cost_report.total_cost:.4f}")

    # 带预算控制
    client = LLMClient(
        ...,
        cost_tracker=CostTrackerConfig.with_budget(
            limit=5.0,
            warning=4.0,
            on_warning=lambda c, t: print(f"Warning: ${c:.2f}")
        )
    )
"""

from dataclasses import dataclass, field
from typing import Callable


class BudgetExceededError(Exception):
    """预算超限异常"""

    def __init__(self, current_cost: float, budget_limit: float, message: str = None):
        self.current_cost = current_cost
        self.budget_limit = budget_limit
        super().__init__(message or f"预算超限: ${current_cost:.4f} > ${budget_limit:.4f}")


@dataclass
class CostTrackerConfig:
    """成本追踪配置"""

    enabled: bool = False
    budget_limit: float | None = None  # 硬限制（美元），超过则停止
    budget_warning: float | None = None  # 软限制（美元），超过则警告
    on_budget_warning: Callable[[float, float], None] | None = None  # 警告回调(current, threshold)

    @classmethod
    def disabled(cls) -> "CostTrackerConfig":
        """禁用成本追踪"""
        return cls(enabled=False)

    @classmethod
    def tracking_only(cls) -> "CostTrackerConfig":
        """仅追踪成本，不设预算限制"""
        return cls(enabled=True)

    @classmethod
    def with_budget(
        cls,
        limit: float,
        warning: float | None = None,
        on_warning: Callable[[float, float], None] | None = None,
    ) -> "CostTrackerConfig":
        """
        带预算控制的成本追踪

        Args:
            limit: 硬限制（美元），超过则抛出 BudgetExceededError
            warning: 软限制（美元），超过则调用 on_warning
            on_warning: 警告回调函数，参数为 (当前成本, 警告阈值)
        """
        return cls(
            enabled=True,
            budget_limit=limit,
            budget_warning=warning,
            on_budget_warning=on_warning,
        )


@dataclass
class CostReport:
    """成本报告"""

    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    request_count: int = 0
    model: str = ""

    @property
    def avg_cost_per_request(self) -> float:
        """每请求平均成本"""
        return self.total_cost / self.request_count if self.request_count > 0 else 0.0

    @property
    def avg_input_tokens(self) -> float:
        """每请求平均输入 token"""
        return self.total_input_tokens / self.request_count if self.request_count > 0 else 0.0

    @property
    def avg_output_tokens(self) -> float:
        """每请求平均输出 token"""
        return self.total_output_tokens / self.request_count if self.request_count > 0 else 0.0

    @property
    def total_tokens(self) -> int:
        """总 token 数"""
        return self.total_input_tokens + self.total_output_tokens

    def __str__(self) -> str:
        return (
            f"CostReport(requests={self.request_count}, "
            f"cost=${self.total_cost:.4f}, "
            f"tokens={self.total_tokens:,} "
            f"[in:{self.total_input_tokens:,}/out:{self.total_output_tokens:,}])"
        )

    def summary(self) -> dict:
        """返回摘要字典"""
        return {
            "total_cost": self.total_cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "avg_cost_per_request": self.avg_cost_per_request,
            "avg_input_tokens": self.avg_input_tokens,
            "avg_output_tokens": self.avg_output_tokens,
            "model": self.model,
        }


class CostTracker:
    """
    成本追踪器

    记录 API 调用的成本，支持预算控制。
    """

    def __init__(self, config: CostTrackerConfig):
        self._config = config
        self._report = CostReport()
        self._warning_triggered = False

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def record(self, usage: dict | None, model: str) -> bool:
        """
        记录一次 API 调用的成本

        Args:
            usage: API 返回的 usage 字典，包含 prompt_tokens, completion_tokens 等
            model: 模型名称

        Returns:
            True 表示可以继续，False 表示应该停止（但不强制）

        Raises:
            BudgetExceededError: 当超过硬限制时
        """
        if not self._config.enabled or usage is None:
            return True

        # 提取 token 数量
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        # 计算成本（延迟导入避免循环引用）
        from . import estimate_cost

        cost = estimate_cost(input_tokens, output_tokens, model)

        # 更新报告
        self._report.total_input_tokens += input_tokens
        self._report.total_output_tokens += output_tokens
        self._report.total_cost += cost
        self._report.request_count += 1
        self._report.model = model

        # 检查硬限制
        if self._config.budget_limit is not None:
            if self._report.total_cost >= self._config.budget_limit:
                raise BudgetExceededError(
                    self._report.total_cost,
                    self._config.budget_limit,
                )

        # 检查软限制（只触发一次）
        if (
            self._config.budget_warning is not None
            and not self._warning_triggered
            and self._report.total_cost >= self._config.budget_warning
        ):
            self._warning_triggered = True
            if self._config.on_budget_warning:
                self._config.on_budget_warning(
                    self._report.total_cost,
                    self._config.budget_warning,
                )

        return True

    def get_report(self) -> CostReport:
        """获取当前成本报告"""
        return self._report

    def reset(self):
        """重置追踪器"""
        self._report = CostReport()
        self._warning_triggered = False
