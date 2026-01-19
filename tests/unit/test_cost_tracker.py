"""Tests for cost_tracker module."""

import pytest

from flexllm import (
    BudgetExceededError,
    CostReport,
    CostTracker,
    CostTrackerConfig,
)


class TestCostTrackerConfig:
    """Test CostTrackerConfig class."""

    def test_disabled(self):
        """Test disabled config."""
        config = CostTrackerConfig.disabled()
        assert config.enabled is False
        assert config.budget_limit is None
        assert config.budget_warning is None

    def test_tracking_only(self):
        """Test tracking_only config."""
        config = CostTrackerConfig.tracking_only()
        assert config.enabled is True
        assert config.budget_limit is None
        assert config.budget_warning is None

    def test_with_budget(self):
        """Test with_budget config."""
        config = CostTrackerConfig.with_budget(limit=5.0, warning=4.0)
        assert config.enabled is True
        assert config.budget_limit == 5.0
        assert config.budget_warning == 4.0

    def test_with_budget_callback(self):
        """Test with_budget config with callback."""
        callback_calls = []

        def on_warning(current, threshold):
            callback_calls.append((current, threshold))

        config = CostTrackerConfig.with_budget(limit=5.0, warning=4.0, on_warning=on_warning)
        assert config.on_budget_warning is on_warning


class TestCostReport:
    """Test CostReport class."""

    def test_default_values(self):
        """Test default values."""
        report = CostReport()
        assert report.total_cost == 0.0
        assert report.total_input_tokens == 0
        assert report.total_output_tokens == 0
        assert report.request_count == 0
        assert report.model == ""

    def test_avg_cost_per_request(self):
        """Test avg_cost_per_request calculation."""
        report = CostReport(total_cost=1.0, request_count=10)
        assert report.avg_cost_per_request == 0.1

    def test_avg_cost_per_request_zero_requests(self):
        """Test avg_cost_per_request with zero requests."""
        report = CostReport(total_cost=0.0, request_count=0)
        assert report.avg_cost_per_request == 0.0

    def test_avg_input_tokens(self):
        """Test avg_input_tokens calculation."""
        report = CostReport(total_input_tokens=1000, request_count=10)
        assert report.avg_input_tokens == 100.0

    def test_avg_output_tokens(self):
        """Test avg_output_tokens calculation."""
        report = CostReport(total_output_tokens=500, request_count=10)
        assert report.avg_output_tokens == 50.0

    def test_total_tokens(self):
        """Test total_tokens calculation."""
        report = CostReport(total_input_tokens=1000, total_output_tokens=500)
        assert report.total_tokens == 1500

    def test_str(self):
        """Test string representation."""
        report = CostReport(
            total_cost=0.01,
            total_input_tokens=1000,
            total_output_tokens=500,
            request_count=5,
        )
        s = str(report)
        assert "0.01" in s
        assert "1,500" in s or "1500" in s
        assert "5" in s

    def test_summary(self):
        """Test summary dict."""
        report = CostReport(
            total_cost=0.01,
            total_input_tokens=1000,
            total_output_tokens=500,
            request_count=5,
            model="gpt-4o",
        )
        summary = report.summary()
        assert summary["total_cost"] == 0.01
        assert summary["total_input_tokens"] == 1000
        assert summary["total_output_tokens"] == 500
        assert summary["total_tokens"] == 1500
        assert summary["request_count"] == 5
        assert summary["model"] == "gpt-4o"


class TestCostTracker:
    """Test CostTracker class."""

    def test_disabled_tracker(self):
        """Test disabled tracker does nothing."""
        config = CostTrackerConfig.disabled()
        tracker = CostTracker(config)
        assert tracker.enabled is False

        # Record should return True and not track anything
        result = tracker.record({"prompt_tokens": 100, "completion_tokens": 50}, "gpt-4o")
        assert result is True
        report = tracker.get_report()
        assert report.request_count == 0

    def test_enabled_tracker(self):
        """Test enabled tracker records usage."""
        config = CostTrackerConfig.tracking_only()
        tracker = CostTracker(config)
        assert tracker.enabled is True

        # Record usage
        tracker.record({"prompt_tokens": 100, "completion_tokens": 50}, "gpt-4o")
        report = tracker.get_report()
        assert report.total_input_tokens == 100
        assert report.total_output_tokens == 50
        assert report.request_count == 1
        assert report.model == "gpt-4o"
        assert report.total_cost > 0

    def test_multiple_records(self):
        """Test multiple records accumulate."""
        config = CostTrackerConfig.tracking_only()
        tracker = CostTracker(config)

        tracker.record({"prompt_tokens": 100, "completion_tokens": 50}, "gpt-4o")
        tracker.record({"prompt_tokens": 200, "completion_tokens": 100}, "gpt-4o")

        report = tracker.get_report()
        assert report.total_input_tokens == 300
        assert report.total_output_tokens == 150
        assert report.request_count == 2

    def test_record_none_usage(self):
        """Test recording None usage."""
        config = CostTrackerConfig.tracking_only()
        tracker = CostTracker(config)

        result = tracker.record(None, "gpt-4o")
        assert result is True
        report = tracker.get_report()
        assert report.request_count == 0

    def test_budget_limit(self):
        """Test budget limit raises exception."""
        config = CostTrackerConfig.with_budget(limit=0.00001)  # Very small limit
        tracker = CostTracker(config)

        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.record({"prompt_tokens": 10000, "completion_tokens": 5000}, "gpt-4o")

        assert exc_info.value.budget_limit == 0.00001
        assert exc_info.value.current_cost > 0

    def test_budget_warning(self):
        """Test budget warning triggers callback."""
        warnings = []

        def on_warning(current, threshold):
            warnings.append((current, threshold))

        config = CostTrackerConfig.with_budget(
            limit=1.0,
            warning=0.00001,
            on_warning=on_warning,  # Very small warning threshold
        )
        tracker = CostTracker(config)

        tracker.record({"prompt_tokens": 1000, "completion_tokens": 500}, "gpt-4o")

        assert len(warnings) == 1
        assert warnings[0][1] == 0.00001

    def test_budget_warning_triggers_once(self):
        """Test budget warning only triggers once."""
        warnings = []

        def on_warning(current, threshold):
            warnings.append((current, threshold))

        config = CostTrackerConfig.with_budget(limit=1.0, warning=0.00001, on_warning=on_warning)
        tracker = CostTracker(config)

        tracker.record({"prompt_tokens": 100, "completion_tokens": 50}, "gpt-4o")
        tracker.record({"prompt_tokens": 100, "completion_tokens": 50}, "gpt-4o")
        tracker.record({"prompt_tokens": 100, "completion_tokens": 50}, "gpt-4o")

        # Warning should only trigger once
        assert len(warnings) == 1

    def test_reset(self):
        """Test reset clears the tracker."""
        config = CostTrackerConfig.tracking_only()
        tracker = CostTracker(config)

        tracker.record({"prompt_tokens": 100, "completion_tokens": 50}, "gpt-4o")
        assert tracker.get_report().request_count == 1

        tracker.reset()
        assert tracker.get_report().request_count == 0
        assert tracker.get_report().total_cost == 0.0


class TestBudgetExceededError:
    """Test BudgetExceededError class."""

    def test_error_message(self):
        """Test error message format."""
        error = BudgetExceededError(1.5, 1.0)
        assert "1.5" in str(error)
        assert "1.0" in str(error)
        assert error.current_cost == 1.5
        assert error.budget_limit == 1.0

    def test_custom_message(self):
        """Test custom error message."""
        error = BudgetExceededError(1.5, 1.0, message="Custom error")
        assert str(error) == "Custom error"
