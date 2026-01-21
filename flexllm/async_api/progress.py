import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from .interface import RequestResult


class ProgressBarStyle(Enum):
    SOLID = ("█", "─", "⚡")  # 实心样式
    BLANK = ("▉", " ", "⚡")
    GRADIENT = ("▰", "▱", "⚡")  # 渐变样式
    BLOCKS = ("▣", "▢", "⚡")  # 方块样式
    ARROW = ("━", "─", "⚡")  # 箭头样式
    DOTS = ("⣿", "⣀", "⚡")  # 点状样式
    PIPES = ("┃", "┆", "⚡")  # 管道样式
    STARS = ("★", "☆", "⚡")  # 星星样式


@dataclass
class ProgressBarConfig:
    bar_length: int = 30
    show_percentage: bool = True
    show_speed: bool = True
    show_counts: bool = True
    show_time_stats: bool = True
    show_cost: bool = False  # 是否显示成本
    style: ProgressBarStyle = ProgressBarStyle.BLANK
    use_colors: bool = True


class ProgressTracker:
    def __init__(
        self,
        total_requests: int,
        concurrency=1,
        config: ProgressBarConfig | None = None,
        model_name: str | None = None,
        input_price_per_1m: float | None = None,
        output_price_per_1m: float | None = None,
    ):
        self.console = Console()

        # 统计信息
        self.success_count = 0
        self.error_count = 0
        self.latencies: list[float] = []
        self.errors: dict[str, int] = defaultdict(int)  # 统计不同类型的错误

        self.total_requests = total_requests
        self.concurrency = concurrency
        self.config = config or ProgressBarConfig()
        self.completed_requests = 0
        self.success_count = 0
        self.error_count = 0
        self.start_time = time.time()
        self.latencies = []
        self.errors = {}
        self.last_speed_update = time.time()
        self.recent_latencies = []  # 用于计算实时速度

        # 成本追踪
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # 模型定价信息（用于双行显示）
        self.model_name: str | None = None
        self.input_price_per_1m: float | None = None  # $/1M tokens
        self.output_price_per_1m: float | None = None  # $/1M tokens

        # 双行显示控制
        self._first_render = True
        self._use_two_lines = False  # 是否使用双行显示

        # 如果提供了模型信息，直接启用双行显示
        if model_name is not None:
            self.set_model_pricing(model_name, input_price_per_1m, output_price_per_1m)

        # ANSI颜色代码
        self.colors = {
            "green": "\033[92m",
            "yellow": "\033[93m",
            "red": "\033[91m",
            "blue": "\033[94m",
            "purple": "\033[95m",
            "cyan": "\033[96m",
            "reset": "\033[0m",
        }

    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds > 3600:
            return f"{seconds / 3600:.1f}h"
        if seconds > 60:
            return f"{seconds / 60:.1f}m"
        return f"{seconds:.1f}s"

    @staticmethod
    def _format_speed(speed: float) -> str:
        """格式化速度显示"""
        # if speed >= 1:
        return f"{speed:.1f} req/s"
        # return f'{speed*1000:.0f} req/ms'

    @staticmethod
    def _format_cost(cost: float) -> str:
        """格式化成本显示"""
        if cost >= 1:
            return f"${cost:.2f}"
        elif cost >= 0.01:
            return f"${cost:.3f}"
        else:
            return f"${cost:.4f}"

    def set_model_pricing(
        self,
        model_name: str,
        input_price_per_1m: float | None,
        output_price_per_1m: float | None,
    ) -> None:
        """设置模型定价信息（用于双行显示）"""
        self.model_name = model_name
        self.input_price_per_1m = input_price_per_1m
        self.output_price_per_1m = output_price_per_1m
        # 启用双行显示（即使没有定价信息也显示模型名和 token 统计）
        self._use_two_lines = True

    def update_cost(self, input_tokens: int, output_tokens: int, cost: float) -> None:
        """更新成本信息并刷新进度条显示"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        # 刷新进度条以显示成本
        if self.config.show_cost:
            self._refresh_progress_bar()

    def _get_colored_text(self, text: str, color: str) -> str:
        """添加颜色到文本"""
        if self.config.use_colors:
            return f"{self.colors[color]}{text}{self.colors['reset']}"
        return text

    def _calculate_speed(self) -> float:
        """计算实际吞吐量（已完成请求数 / 已用时间）"""
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return 0
        return self.completed_requests / elapsed

    @staticmethod
    def _format_tokens(tokens: int) -> str:
        """格式化 token 数量显示"""
        if tokens >= 1_000_000:
            return f"{tokens / 1_000_000:.1f}M"
        elif tokens >= 1_000:
            return f"{tokens / 1_000:.1f}K"
        return str(tokens)

    def _build_cost_line(self) -> str:
        """构建成本信息行（双行显示的第一行）"""
        parts = []

        # 总成本
        parts.append(f"💰 {self._format_cost(self.total_cost)}")

        # 模型名称和定价
        if self.model_name:
            if self.input_price_per_1m is not None:
                price_info = f"{self.model_name}: ${self.input_price_per_1m:.2f}/${self.output_price_per_1m:.2f} per 1M"
            else:
                price_info = f"{self.model_name}"
            parts.append(price_info)

        # Token 统计
        if self.total_input_tokens > 0 or self.total_output_tokens > 0:
            token_info = f"{self._format_tokens(self.total_input_tokens)} in / {self._format_tokens(self.total_output_tokens)} out"
            parts.append(token_info)

        return " | ".join(parts)

    def _refresh_progress_bar(self) -> None:
        """刷新进度条显示"""
        total_time = time.time() - self.start_time
        progress = self.completed_requests / self.total_requests

        # 计算统计信息
        speed = self._calculate_speed()
        avg_latency = statistics.mean(self.latencies) if self.latencies else 0
        remaining_requests = self.total_requests - self.completed_requests
        estimated_remaining_time = (
            avg_latency * remaining_requests / self.concurrency if avg_latency > 0 else 0
        )

        # 创建进度条
        style = self.config.style.value
        filled_length = int(self.config.bar_length * progress)
        bar = style[0] * filled_length + style[1] * (self.config.bar_length - filled_length)

        # 构建输出组件
        components = []

        # 进度条和百分比
        progress_text = f"[{self._get_colored_text(bar, 'blue')}]"
        if self.config.show_percentage:
            progress_text += f" {self._get_colored_text(f'{progress * 100:.1f}%', 'green')}"
        components.append(progress_text)

        # 请求计数
        if self.config.show_counts:
            counts = f"({self.completed_requests}/{self.total_requests})"
            components.append(self._get_colored_text(counts, "yellow"))

        # 速度信息
        if self.config.show_speed:
            speed_text = f"{style[2]} {self._format_speed(speed)}"
            components.append(self._get_colored_text(speed_text, "cyan"))

        # 时间统计
        if self.config.show_time_stats:
            time_stats = (
                f"avg: {self._format_time(avg_latency)} "
                f"total: {self._format_time(total_time)} "
                f"eta: {self._format_time(estimated_remaining_time)}"
            )
            components.append(self._get_colored_text(time_stats, "purple"))

        # 单行模式下的成本显示（向后兼容）
        if self.config.show_cost and self.total_cost > 0 and not self._use_two_lines:
            cost_text = f"💰 {self._format_cost(self.total_cost)}"
            components.append(self._get_colored_text(cost_text, "green"))

        progress_line = " ".join(components)

        # 打印进度 - 修复Windows编码问题
        try:
            if self._use_two_lines and self.config.show_cost:
                cost_line = self._build_cost_line()
                if self._first_render:
                    # 首次渲染：打印两行
                    print(self._get_colored_text(cost_line, "green"))
                    print(progress_line, end="", flush=True)
                    self._first_render = False
                else:
                    # 后续刷新：上移光标，更新两行
                    # \033[A 上移一行, \033[K 清除到行尾
                    print(f"\r\033[A\033[K{self._get_colored_text(cost_line, 'green')}")
                    print(f"\033[K{progress_line}", end="", flush=True)
            else:
                print("\r" + progress_line, end="", flush=True)
        except UnicodeEncodeError:
            # Windows GBK编码兼容处理
            safe_components = []
            for comp in components:
                # 替换有问题的Unicode字符
                safe_comp = comp.replace("⚡", "*").replace("█", "#").replace("─", "-")
                safe_comp = safe_comp.replace("▉", "|").replace("▰", "=").replace("▱", "-")
                safe_comp = safe_comp.replace("▣", "[").replace("▢", "]").replace("━", "=")
                safe_comp = (
                    safe_comp.replace("┃", "|")
                    .replace("┆", ":")
                    .replace("★", "*")
                    .replace("☆", "+")
                )
                safe_comp = safe_comp.replace("⣿", "#").replace("⣀", ".").replace("💰", "$")
                safe_components.append(safe_comp)
            print("\r" + " ".join(safe_components), end="", flush=True)

    def update(self, result: "RequestResult") -> None:
        """
        更新进度和统计信息

        Args:
            result: 请求结果
        """
        self.completed_requests += 1
        self.latencies.append(result.latency)
        self.recent_latencies.append(result.latency)

        # 只保留最近的30个请求用于计算速度
        if len(self.recent_latencies) > 30:
            self.recent_latencies.pop(0)

        if result.status == "success":
            self.success_count += 1
        else:
            self.error_count += 1
            # 安全地获取错误类型，处理 result.data 为 None 的情况
            if result.data and isinstance(result.data, dict):
                error_type = result.data.get("error", "unknown")
            else:
                error_type = "unknown"
            self.errors[error_type] = self.errors.get(error_type, 0) + 1

        self._refresh_progress_bar()

    def summary(self, show_p999=False, print_to_console=True) -> str:
        """打印请求汇总信息"""
        total_time = time.time() - self.start_time
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        throughput = self.success_count / total_time if total_time > 0 else 0

        # 计算延迟分位数
        sorted_latencies = sorted(self.latencies)
        p50 = p95 = p99 = 0
        if sorted_latencies:
            p50 = sorted_latencies[int(len(sorted_latencies) * 0.5)]
            p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
            p995 = sorted_latencies[int(len(sorted_latencies) * 0.995)]
            p999 = sorted_latencies[int(len(sorted_latencies) * 0.999)]

        p99_str = f"> - P99 延迟: {p99:.2f} 秒"
        p999_str = f"""> - P99 延迟: {p99:.2f} 秒
> - P995 延迟: {p995:.2f} 秒
> - P999 延迟: {p999:.2f} 秒"""
        p99_or_p999_str = p999_str if show_p999 else p99_str

        summary = f"""
                                   请求统计

| 总体情况
|  - 总请求数: {self.total_requests}
|  - 成功请求数: {self.success_count}
|  - 失败请求数: {self.error_count}
|  - 成功率: {(self.success_count / self.total_requests * 100):.2f}%

| 性能指标
|  - 平均延迟: {avg_latency:.2f} 秒
|  - P50 延迟: {p50:.2f} 秒
|  - P95 延迟: {p95:.2f} 秒
|  - P99 延迟: {p99:.2f} 秒
|  - 吞吐量: {throughput:.2f} 请求/秒
|  - 总运行时间: {total_time:.2f} 秒

"""
        # 如果有成本信息，添加成本统计
        if self.total_cost > 0:
            avg_cost = self.total_cost / self.success_count if self.success_count > 0 else 0
            summary += f"""| 成本统计
|  - 总成本: ${self.total_cost:.4f}
|  - 平均成本/请求: ${avg_cost:.6f}
|  - 总输入 tokens: {self.total_input_tokens:,}
|  - 总输出 tokens: {self.total_output_tokens:,}

"""
        # 如果有错误，添加错误统计
        if self.errors:
            summary += (
                "| 错误分布                                                                   \n"
            )
            for error_type, count in self.errors.items():
                percentage = count / self.error_count * 100
                summary += (
                    f"|  - {error_type}: {count} ({percentage:.1f}%)                            \n"
                )

        summary += "-" * 76
        if print_to_console:
            print()  # 打印空行
            try:
                # 尝试使用Rich输出，如果失败则使用普通print
                self.console.print(summary)
            except UnicodeEncodeError:
                # 在Windows GBK环境下，如果出现编码错误，使用普通print
                print(summary)
        return summary


if __name__ == "__main__":
    from .interface import RequestResult

    config = ProgressBarConfig()
    tracker = ProgressTracker(100, 1, config)
    for i in range(100):
        time.sleep(0.1)
        tracker.update(
            result=RequestResult(
                request_id=i,
                data=None,
                status="success",
                latency=0.1,
            )
        )
