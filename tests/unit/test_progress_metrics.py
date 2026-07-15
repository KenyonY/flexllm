"""ProgressTracker 延迟指标口径测试

覆盖 issue #12 的两个正交失真：
1. 失败/超时请求混进分位数 —— 默认 timeout=60s 时 p95 直接等于 60，零信息量
2. 排队等待混进"延迟" —— 调低 max_qps 会让"延迟"暴涨，但服务并没变慢
"""

from flexllm.async_api.interface import RequestResult
from flexllm.async_api.progress import ProgressTracker


def ok(request_id: int, latency: float, queue_time: float = 0.0) -> RequestResult:
    return RequestResult(
        request_id=request_id, data={}, status="success", latency=latency, queue_time=queue_time
    )


def timeout(request_id: int, latency: float = 60.0) -> RequestResult:
    return RequestResult(
        request_id=request_id,
        data={"error": "Timeout error", "detail": ""},
        status="error",
        latency=latency,
    )


def perf_lines(tracker: ProgressTracker, **kwargs) -> list[str]:
    """摘出 summary 里的指标行，去掉 "|  - " 前缀"""
    section = tracker.summary(print_to_console=False, **kwargs)
    prefix = "|  - "
    return [line[len(prefix) :].strip() for line in section.splitlines() if line.startswith(prefix)]


class TestFailuresExcludedFromPercentiles:
    """分位数只统计成功请求"""

    def test_timeouts_do_not_dominate_percentiles(self):
        # 90 个成功(1s) + 10 个超时(60s)：p95 落在超时区，修复前会报 60s
        tracker = ProgressTracker(100)
        for i in range(90):
            tracker.update(ok(i, latency=1.0))
        for i in range(90, 100):
            tracker.update(timeout(i))

        assert tracker.latencies == [1.0] * 90, "失败请求不应进入 latencies"
        assert tracker.success_count == 90
        assert tracker.error_count == 10

        summary = tracker.summary(print_to_console=False)
        assert "60.00" not in summary, "timeout 常数不应出现在延迟分位数中"
        assert "仅统计 90 个成功请求" in summary

    def test_zero_success_reports_no_data_not_zero_latency(self):
        # 全部失败时不能印 "延迟 0.00 秒"——扫一眼会读成"极快"
        tracker = ProgressTracker(3)
        for i in range(3):
            tracker.update(timeout(i))

        lines = perf_lines(tracker)
        assert any("无成功请求" in line for line in lines)
        assert not any("延迟 P50" in line for line in lines)


class TestQueueTimeAttribution:
    """latency = queue_time + service_time，按归因拆分"""

    def test_service_latency_is_invariant_to_queueing(self):
        # 同样的服务延迟 0.5s，一组无排队、一组排队 9.5s：服务分位数应当一致
        no_queue = ProgressTracker(10)
        queued = ProgressTracker(10)
        for i in range(10):
            no_queue.update(ok(i, latency=0.5, queue_time=0.0))
            queued.update(ok(i, latency=10.0, queue_time=9.5))

        assert no_queue._avg_service_time() == queued._avg_service_time() == 0.5

    def test_split_shown_only_when_queue_significant(self):
        tracker = ProgressTracker(10)
        for i in range(10):
            tracker.update(ok(i, latency=10.0, queue_time=9.2))  # 排队占 92%

        lines = perf_lines(tracker)
        assert any(line.startswith("服务延迟 P50") for line in lines)
        assert any(line.startswith("排队等待 P50") for line in lines)
        assert any(line.startswith("端到端 P50") for line in lines)

    def test_negligible_queue_collapses_to_single_row(self):
        tracker = ProgressTracker(10)
        for i in range(10):
            tracker.update(ok(i, latency=1.0, queue_time=0.01))  # 排队占 1%

        lines = perf_lines(tracker)
        assert any(line.startswith("延迟 P50") for line in lines)
        assert not any("排队等待" in line for line in lines)

    def test_service_time_computed_per_request_before_sorting(self):
        # 排队与服务负相关：逐条相减再排序，不能拿两条已排序序列相减
        tracker = ProgressTracker(2)
        tracker.update(ok(0, latency=10.0, queue_time=9.0))  # service 1.0
        tracker.update(ok(1, latency=10.0, queue_time=1.0))  # service 9.0

        summary = tracker.summary(print_to_console=False)
        assert "服务延迟 P50/P95/P99: 9.00" in summary


class TestSummaryEdgeCases:
    def test_empty_tracker_does_not_raise(self):
        # 回归：p995/p999 曾未初始化 -> UnboundLocalError
        ProgressTracker(10).summary(show_p999=True, print_to_console=False)

    def test_zero_total_requests_does_not_raise(self):
        # 回归：成功率除以 total_requests -> ZeroDivisionError
        ProgressTracker(0).summary(print_to_console=False)

    def test_show_p999_renders_tail_quantiles(self):
        # 回归：show_p999 曾计算后从未被使用，参数完全无效
        tracker = ProgressTracker(10)
        for i in range(10):
            tracker.update(ok(i, latency=1.0))

        assert "P995" not in tracker.summary(print_to_console=False)
        assert "P995" in tracker.summary(show_p999=True, print_to_console=False)


class TestPercentile:
    def test_index_clamped_to_last_element(self):
        assert ProgressTracker._percentile([1.0, 2.0], 0.999) == 2.0

    def test_empty_returns_zero(self):
        assert ProgressTracker._percentile([], 0.95) == 0.0
