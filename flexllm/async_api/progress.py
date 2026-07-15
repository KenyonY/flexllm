import sys
import time
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
        config: ProgressBarConfig | None = None,
        model_name: str | None = None,
        input_price_per_1m: float | None = None,
        output_price_per_1m: float | None = None,
    ):
        self.console = Console(file=sys.stderr)

        self.total_requests = total_requests
        self.config = config or ProgressBarConfig()
        self.completed_requests = 0

        # 统计信息。latencies / queue_times 只收集成功请求，两者按下标一一对应：
        # 失败请求的 latency 是 timeout 全时长或报错耗时，与"服务多快"无关，
        # 混进来会让分位数被 timeout 常数支配（默认 timeout=60s 时 p95 直接等于 60）。
        self.success_count = 0
        self.error_count = 0
        self.latencies: list[float] = []
        self.queue_times: list[float] = []
        # running sum，让进度条每次刷新算 avg 是 O(1)（分位数才需要整表排序）
        self._latency_sum = 0.0
        self._queue_sum = 0.0
        self.errors: dict[str, int] = {}  # 统计不同类型的错误
        self.retry_count = 0  # fallback 重试次数
        self._seen_error_types: set[str] = set()  # 已打印过的错误类型
        # 计时一律用 perf_counter：time.time() 是墙钟，NTP 校时会跳
        self.start_time = time.perf_counter()

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

        # 节流控制：限制刷新频率，避免高并发时过多终端 I/O
        self._last_refresh_time = 0.0
        self._min_refresh_interval = 0.05  # 最小刷新间隔 50ms

        # TTY 检测：非 TTY 环境使用里程碑输出
        self._is_tty = sys.stdout.isatty()
        self._milestones = set(range(10, 101, 10))  # 10%, 20%, ..., 100%
        self._reported_milestones: set[int] = set()

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

    def increment_retry(self) -> None:
        """增加 fallback 重试计数并刷新进度条"""
        self.retry_count += 1
        self._refresh_progress_bar()

    def _get_colored_text(self, text: str, color: str) -> str:
        """添加颜色到文本"""
        if self.config.use_colors:
            return f"{self.colors[color]}{text}{self.colors['reset']}"
        return text

    def _calculate_speed(self) -> float:
        """计算实际吞吐量（已完成请求数 / 已用时间）"""
        elapsed = time.perf_counter() - self.start_time
        if elapsed <= 0:
            return 0
        return self.completed_requests / elapsed

    def _estimate_eta(self, speed: float) -> float:
        """剩余时间 = 剩余请求数 / 实测吞吐量

        不要用 avg_latency * remaining / concurrency：avg_latency 含排队时间，
        而排队时间本身就随并发上升，再除以 concurrency 只是让两个错误互相抵消。
        实测吞吐量已经包含了限流、重试、失败的全部影响，直接用即可。
        """
        remaining = self.total_requests - self.completed_requests
        return remaining / speed if speed > 0 else 0

    def _avg_service_time(self) -> float:
        """平均服务延迟（成功请求）= 平均端到端 - 平均排队"""
        if not self.success_count:
            return 0.0
        return (self._latency_sum - self._queue_sum) / self.success_count

    @staticmethod
    def _percentile(sorted_values: list[float], q: float) -> float:
        """取分位数（下标法）。空列表返回 0，下标越界钳到末位。"""
        if not sorted_values:
            return 0.0
        return sorted_values[min(int(len(sorted_values) * q), len(sorted_values) - 1)]

    @staticmethod
    def _format_tokens(tokens: int) -> str:
        """格式化 token 数量显示"""
        if tokens >= 1_000_000:
            return f"{tokens / 1_000_000:.1f}M"
        elif tokens >= 1_000:
            return f"{tokens / 1_000:.1f}K"
        return str(tokens)

    def _print_milestone(self, milestone: int) -> None:
        """输出里程碑进度（非 TTY 环境）

        格式: [10%] 100/1000, 2.3 req/s, elapsed: 43s, eta: 6m
        """
        elapsed = time.perf_counter() - self.start_time
        speed = self._calculate_speed()
        eta = self._estimate_eta(speed)

        # 构建输出
        parts = [
            f"[{milestone}%]",
            f"{self.completed_requests}/{self.total_requests},",
            f"{speed:.1f} req/s,",
        ]

        # eta 仅在未完成时显示，100% 时只显示 elapsed
        if milestone < 100:
            parts.append(f"elapsed: {self._format_time(elapsed)},")
            parts.append(f"eta: {self._format_time(eta)}")
        else:
            parts.append(f"elapsed: {self._format_time(elapsed)}")

        # 错误信息
        if self.error_count > 0:
            parts.append(f"(errors: {self.error_count})")

        print(" ".join(parts), flush=True, file=sys.stderr)

    def _build_cost_line(self) -> str:
        """构建成本信息行（双行显示的第一行）"""
        parts = []

        # 根据是否有定价信息决定显示内容
        has_pricing = self.input_price_per_1m is not None and self.output_price_per_1m is not None

        if has_pricing and self.total_cost > 0:
            # 有定价信息且有成本：显示成本
            parts.append(f"💰 {self._format_cost(self.total_cost)}")
        elif self.total_input_tokens > 0 or self.total_output_tokens > 0:
            # 无定价信息但有 token 数据：显示 token 统计（使用不同图标）
            parts.append(
                f"📊 {self._format_tokens(self.total_input_tokens + self.total_output_tokens)} tokens"
            )

        # 模型名称和定价
        if self.model_name:
            if has_pricing:
                price_info = f"{self.model_name}: ${self.input_price_per_1m:.2f}/${self.output_price_per_1m:.2f} per 1M"
            else:
                price_info = f"{self.model_name} (no pricing info)"
            parts.append(price_info)

        # Token 详细统计（仅在有定价信息时显示，避免重复）
        if has_pricing and (self.total_input_tokens > 0 or self.total_output_tokens > 0):
            token_info = f"{self._format_tokens(self.total_input_tokens)} in / {self._format_tokens(self.total_output_tokens)} out"
            parts.append(token_info)
        elif not has_pricing and (self.total_input_tokens > 0 or self.total_output_tokens > 0):
            # 无定价信息时显示详细的输入/输出分解
            token_info = f"{self._format_tokens(self.total_input_tokens)} in / {self._format_tokens(self.total_output_tokens)} out"
            parts.append(token_info)

        return " | ".join(parts)

    def _refresh_progress_bar(self, force: bool = False) -> None:
        """刷新进度条显示

        Args:
            force: 强制刷新，忽略节流限制
        """
        # 非 TTY 环境：使用里程碑输出
        if not self._is_tty:
            progress_pct = int(self.completed_requests / self.total_requests * 100)
            # 检查是否达到新的里程碑
            for milestone in self._milestones:
                if progress_pct >= milestone and milestone not in self._reported_milestones:
                    self._reported_milestones.add(milestone)
                    self._print_milestone(milestone)
            return

        now = time.perf_counter()
        # 节流：距离上次刷新不足间隔则跳过（除非强制刷新）
        if not force and self.completed_requests < self.total_requests:
            if now - self._last_refresh_time < self._min_refresh_interval:
                return

        self._last_refresh_time = now
        total_time = now - self.start_time
        progress = self.completed_requests / self.total_requests

        # 计算统计信息。avg 显示服务延迟而非端到端：排队时间由 max_qps/并发决定，
        # 属于用户自己的配置，混进 avg 会让"调低 max_qps"看起来像"服务变慢"。
        # 端到端的节奏由紧邻的 eta 表达。
        speed = self._calculate_speed()
        avg_latency = self._avg_service_time()
        estimated_remaining_time = self._estimate_eta(speed)

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
            # 显示 retry 和 error 计数
            if self.retry_count > 0 or self.error_count > 0:
                counts += f" ↻{self.retry_count}" if self.retry_count > 0 else ""
                counts += f" ✗{self.error_count}" if self.error_count > 0 else ""
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

        # 单行模式下的成本/token 显示（向后兼容）
        if self.config.show_cost and not self._use_two_lines:
            has_pricing = (
                self.input_price_per_1m is not None and self.output_price_per_1m is not None
            )
            if has_pricing and self.total_cost > 0:
                cost_text = f"💰 {self._format_cost(self.total_cost)}"
                components.append(self._get_colored_text(cost_text, "green"))
            elif self.total_input_tokens > 0 or self.total_output_tokens > 0:
                # 无定价信息时显示 token 数量
                token_text = (
                    f"📊 {self._format_tokens(self.total_input_tokens + self.total_output_tokens)}"
                )
                components.append(self._get_colored_text(token_text, "green"))

        progress_line = " ".join(components)

        # 打印进度 - 修复Windows编码问题
        try:
            if self._use_two_lines and self.config.show_cost:
                cost_line = self._build_cost_line()
                if self._first_render:
                    # 首次渲染：打印两行
                    print(self._get_colored_text(cost_line, "green"), file=sys.stderr)
                    print(progress_line, end="", flush=True, file=sys.stderr)
                    self._first_render = False
                else:
                    # 后续刷新：上移光标，更新两行
                    # \033[A 上移一行, \033[K 清除到行尾
                    print(
                        f"\r\033[A\033[K{self._get_colored_text(cost_line, 'green')}",
                        file=sys.stderr,
                    )
                    print(f"\033[K{progress_line}", end="", flush=True, file=sys.stderr)
            else:
                print("\r" + progress_line, end="", flush=True, file=sys.stderr)
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
            print("\r" + " ".join(safe_components), end="", flush=True, file=sys.stderr)

    def update(self, result: "RequestResult") -> None:
        """
        更新进度和统计信息

        Args:
            result: 请求结果
        """
        self.completed_requests += 1

        if result.status == "success":
            self.success_count += 1
            self.latencies.append(result.latency)
            self.queue_times.append(result.queue_time)
            self._latency_sum += result.latency
            self._queue_sum += result.queue_time
        else:
            self.error_count += 1
            # 安全地获取错误类型和详情，处理 result.data 为 None 的情况
            error_type = "unknown"
            error_detail = ""
            if result.data and isinstance(result.data, dict):
                error_type = result.data.get("error", "unknown")
                error_detail = result.data.get("detail", "")
            self.errors[error_type] = self.errors.get(error_type, 0) + 1

            # 首次出现的错误类型打印一次警告
            if error_type not in self._seen_error_types:
                self._seen_error_types.add(error_type)
                # 构建显示信息：错误类型 + 详情（不截断）
                display_error = f"{error_type}: {error_detail}" if error_detail else error_type
                # 清除当前行并打印警告，避免打乱进度条
                if self._use_two_lines:
                    # 双行模式：上移一行，清除两行，打印警告，重置首次渲染标志
                    print(
                        f"\r\033[A\033[K\033[K⚠️  新错误类型: {display_error}",
                        file=sys.stderr,
                    )
                    self._first_render = True
                else:
                    print(f"\r\033[K⚠️  新错误类型: {display_error}", file=sys.stderr)

        # 最后一个请求完成时强制刷新，确保显示 100%
        force = self.completed_requests >= self.total_requests
        self._refresh_progress_bar(force=force)

    # 排队占端到端低于此比例时，不单独列出排队/服务的拆分（三行退化为一行）
    _QUEUE_DISPLAY_THRESHOLD = 0.05

    def summary(self, show_p999=False, print_to_console=True) -> str:
        """打印请求汇总信息

        Args:
            show_p999: 额外显示 P995 / P999 尾部分位数
            print_to_console: 是否打印到 stderr

        分位数只统计成功请求（见 __init__ 中 latencies 的说明）。延迟按归因拆成
        "服务延迟"与"排队等待"：后者是 max_qps / 并发上限造成的客户端自身等待，
        调低 max_qps 会让它上升，但那不是服务变慢。仅在排队占比显著时才拆开显示。
        """
        total_time = time.perf_counter() - self.start_time
        throughput = self.success_count / total_time if total_time > 0 else 0
        success_rate = self.success_count / self.total_requests * 100 if self.total_requests else 0

        avg_e2e = self._latency_sum / self.success_count if self.success_count else 0
        avg_service = self._avg_service_time()
        queue_share = self._queue_sum / self._latency_sum if self._latency_sum > 0 else 0

        sorted_e2e = sorted(self.latencies)
        sorted_queue = sorted(self.queue_times)
        # 先逐条相减再排序：服务延迟是 per-request 的量，不能拿两条已排序的序列相减
        sorted_service = sorted(lat - q for lat, q in zip(self.latencies, self.queue_times))

        quantiles = [("P50", 0.5), ("P95", 0.95), ("P99", 0.99)]
        if show_p999:
            quantiles += [("P995", 0.995), ("P999", 0.999)]
        names = "/".join(name for name, _ in quantiles)

        def row(label: str, sorted_vals: list[float], note: str) -> str:
            vals = " / ".join(f"{self._percentile(sorted_vals, q):.2f}" for _, q in quantiles)
            return f"|  - {label} {names}: {vals} 秒（{note}）"

        if not self.success_count:
            # 不要在这里印 "延迟 0.00 秒"：扫一眼会读成"极快"，而真相是无数据
            perf_rows = ["|  - 无成功请求，无延迟数据"]
        elif queue_share >= self._QUEUE_DISPLAY_THRESHOLD:
            perf_rows = [
                row("服务延迟", sorted_service, f"平均 {avg_service:.2f}"),
                row(
                    "排队等待",
                    sorted_queue,
                    f"占端到端 {queue_share * 100:.0f}%，由并发/max_qps 决定",
                ),
                row("端到端", sorted_e2e, f"平均 {avg_e2e:.2f}"),
            ]
        else:
            perf_rows = [row("延迟", sorted_e2e, f"平均 {avg_e2e:.2f}")]
        perf_rows.append(f"|  - 吞吐量: {throughput:.2f} 请求/秒")
        perf_rows.append(f"|  - 总运行时间: {total_time:.2f} 秒")
        perf_section = "\n".join(perf_rows)

        summary = f"""
                                   请求统计

| 总体情况
|  - 总请求数: {self.total_requests}
|  - 成功请求数: {self.success_count}
|  - 失败请求数: {self.error_count}
|  - 成功率: {success_rate:.2f}%

| 性能指标（仅统计 {self.success_count} 个成功请求）
{perf_section}

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
            print(file=sys.stderr)  # 打印空行
            try:
                # 尝试使用Rich输出，如果失败则使用普通print
                self.console.print(summary)
            except UnicodeEncodeError:
                # 在Windows GBK环境下，如果出现编码错误，使用普通print
                print(summary, file=sys.stderr)
        return summary


if __name__ == "__main__":
    from .interface import RequestResult

    config = ProgressBarConfig()
    tracker = ProgressTracker(100, config)
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
