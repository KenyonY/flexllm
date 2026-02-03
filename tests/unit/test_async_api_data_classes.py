"""
async_api 模块数据类测试

测试 interface.py 和 concurrent_executor.py 中的数据类：
- RequestResult
- TaskContext
- TaskItem
- ExecutionResult
- StreamingExecutionResult
"""

import heapq

import pytest

from flexllm.async_api.concurrent_executor import (
    ExecutionResult,
    StreamingExecutionResult,
    TaskContext,
    TaskItem,
)
from flexllm.async_api.interface import RequestResult


class TestRequestResult:
    """RequestResult 数据类测试"""

    def test_creation_with_required_fields(self):
        """测试必填字段创建"""
        result = RequestResult(
            request_id=1,
            data={"message": "hello"},
            status="success",
            latency=0.5,
        )

        assert result.request_id == 1
        assert result.data == {"message": "hello"}
        assert result.status == "success"
        assert result.latency == 0.5
        assert result.meta is None  # 默认值

    def test_creation_with_meta(self):
        """测试带 meta 字段创建"""
        result = RequestResult(
            request_id=2,
            data="response",
            status="error",
            latency=1.2,
            meta={"retry": 1, "source": "api"},
        )

        assert result.request_id == 2
        assert result.meta == {"retry": 1, "source": "api"}

    def test_data_can_be_any_type(self):
        """测试 data 字段可以是任意类型"""
        # dict
        r1 = RequestResult(request_id=1, data={"key": "value"}, status="success", latency=0.1)
        assert r1.data == {"key": "value"}

        # list
        r2 = RequestResult(request_id=2, data=[1, 2, 3], status="success", latency=0.1)
        assert r2.data == [1, 2, 3]

        # None
        r3 = RequestResult(request_id=3, data=None, status="error", latency=0.1)
        assert r3.data is None

        # string
        r4 = RequestResult(request_id=4, data="plain text", status="success", latency=0.1)
        assert r4.data == "plain text"


class TestTaskContext:
    """TaskContext 数据类测试"""

    def test_creation_with_required_fields(self):
        """测试必填字段创建"""
        ctx = TaskContext(task_id=1, data="task_data")

        assert ctx.task_id == 1
        assert ctx.data == "task_data"
        assert ctx.meta is None
        assert ctx.retry_count == 0
        assert ctx.executor_kwargs is None

    def test_creation_with_all_fields(self):
        """测试所有字段创建"""
        ctx = TaskContext(
            task_id=10,
            data={"input": "value"},
            meta={"source": "batch"},
            retry_count=2,
            executor_kwargs={"timeout": 30},
        )

        assert ctx.task_id == 10
        assert ctx.data == {"input": "value"}
        assert ctx.meta == {"source": "batch"}
        assert ctx.retry_count == 2
        assert ctx.executor_kwargs == {"timeout": 30}

    def test_retry_count_default_zero(self):
        """测试 retry_count 默认值为 0"""
        ctx = TaskContext(task_id=1, data="test")
        assert ctx.retry_count == 0


class TestTaskItem:
    """TaskItem 数据类测试"""

    def test_creation(self):
        """测试创建"""
        item = TaskItem(priority=1, task_id=100, data="task_data")

        assert item.priority == 1
        assert item.task_id == 100
        assert item.data == "task_data"
        assert item.meta == {}  # 默认值是 dict

    def test_creation_with_meta(self):
        """测试带 meta 创建"""
        item = TaskItem(priority=2, task_id=200, data="data", meta={"key": "value"})

        assert item.meta == {"key": "value"}

    def test_priority_comparison_lt(self):
        """测试优先级比较 __lt__"""
        high_priority = TaskItem(priority=1, task_id=1, data="high")
        low_priority = TaskItem(priority=10, task_id=2, data="low")

        assert high_priority < low_priority
        assert not low_priority < high_priority

    def test_priority_comparison_equal(self):
        """测试相同优先级比较"""
        item1 = TaskItem(priority=5, task_id=1, data="a")
        item2 = TaskItem(priority=5, task_id=2, data="b")

        # 相同优先级时 __lt__ 返回 False
        assert not item1 < item2
        assert not item2 < item1

    def test_heap_ordering(self):
        """测试在 heapq 中的优先级排序"""
        items = [
            TaskItem(priority=3, task_id=1, data="low"),
            TaskItem(priority=1, task_id=2, data="high"),
            TaskItem(priority=2, task_id=3, data="medium"),
        ]

        heap = []
        for item in items:
            heapq.heappush(heap, item)

        # 弹出顺序应该按优先级从小到大
        result = []
        while heap:
            result.append(heapq.heappop(heap))

        assert result[0].priority == 1
        assert result[1].priority == 2
        assert result[2].priority == 3

    def test_heap_with_many_items(self):
        """测试大量 TaskItem 的堆排序"""
        import random

        priorities = list(range(100))
        random.shuffle(priorities)

        items = [
            TaskItem(priority=p, task_id=i, data=f"data_{i}") for i, p in enumerate(priorities)
        ]

        heap = []
        for item in items:
            heapq.heappush(heap, item)

        # 验证弹出顺序
        prev_priority = -1
        while heap:
            item = heapq.heappop(heap)
            assert item.priority >= prev_priority
            prev_priority = item.priority


class TestExecutionResult:
    """ExecutionResult 数据类测试"""

    def test_success_result(self):
        """测试成功状态结果"""
        result = ExecutionResult(
            task_id=1,
            data={"output": "result"},
            status="success",
            latency=0.3,
        )

        assert result.task_id == 1
        assert result.data == {"output": "result"}
        assert result.status == "success"
        assert result.latency == 0.3
        assert result.error is None
        assert result.retry_count == 0

    def test_error_result(self):
        """测试错误状态结果"""
        error = ValueError("test error")
        result = ExecutionResult(
            task_id=2,
            data=None,
            status="error",
            latency=0.1,
            error=error,
            retry_count=3,
        )

        assert result.task_id == 2
        assert result.data is None
        assert result.status == "error"
        assert result.error is error
        assert result.retry_count == 3

    def test_with_meta(self):
        """测试带 meta 的结果"""
        result = ExecutionResult(
            task_id=3,
            data="output",
            status="success",
            meta={"source": "batch", "index": 5},
        )

        assert result.meta == {"source": "batch", "index": 5}

    def test_default_values(self):
        """测试默认值"""
        result = ExecutionResult(task_id=1, data="test", status="success")

        assert result.meta is None
        assert result.latency == 0.0
        assert result.error is None
        assert result.retry_count == 0


class TestStreamingExecutionResult:
    """StreamingExecutionResult 数据类测试"""

    def test_creation(self):
        """测试创建"""
        tasks = [
            ExecutionResult(task_id=1, data="r1", status="success"),
            ExecutionResult(task_id=2, data="r2", status="success"),
        ]

        result = StreamingExecutionResult(
            completed_tasks=tasks,
            progress=None,
            is_final=False,
        )

        assert len(result.completed_tasks) == 2
        assert result.progress is None
        assert result.is_final is False

    def test_final_result(self):
        """测试最终结果标记"""
        tasks = [ExecutionResult(task_id=1, data="done", status="success")]

        result = StreamingExecutionResult(
            completed_tasks=tasks,
            progress=None,
            is_final=True,
        )

        assert result.is_final is True

    def test_empty_completed_tasks(self):
        """测试空任务列表"""
        result = StreamingExecutionResult(
            completed_tasks=[],
            progress=None,
            is_final=False,
        )

        assert result.completed_tasks == []
