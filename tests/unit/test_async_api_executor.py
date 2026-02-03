"""
async_api 模块 ConcurrentExecutor 核心功能测试

测试 concurrent_executor.py 中的 ConcurrentExecutor 类：
- 批量执行
- 重试机制
- 优先级调度
- 流式执行
- 函数签名检测
- 自定义错误处理
"""

import asyncio

import pytest

from flexllm.async_api.concurrent_executor import (
    ConcurrentExecutor,
    ExecutionResult,
    TaskContext,
    TaskItem,
)

# ============== Mock 函数 ==============


async def mock_simple_task(data):
    """简单任务函数，只接收 data 参数"""
    await asyncio.sleep(0.001)
    return f"result:{data}"


async def mock_context_task(context: TaskContext):
    """上下文任务函数，接收 TaskContext 参数"""
    await asyncio.sleep(0.001)
    return f"task_{context.task_id}:{context.data}:retry_{context.retry_count}"


async def mock_kwargs_task(data, **kwargs):
    """带关键字参数的任务函数"""
    await asyncio.sleep(0.001)
    return f"result:{data}:{kwargs}"


async def mock_failing_task(data):
    """总是失败的任务函数"""
    raise ValueError(f"error:{data}")


async def mock_partial_failing_task(data):
    """部分失败的任务函数（偶数 ID 失败）"""
    await asyncio.sleep(0.001)
    if int(data) % 2 == 0:
        raise ValueError(f"even_error:{data}")
    return f"result:{data}"


# ============== 测试类 ==============


class TestExecutorInit:
    """ConcurrentExecutor 初始化测试"""

    def test_init_with_defaults(self):
        """测试默认参数初始化"""
        executor = ConcurrentExecutor(concurrency_limit=5)

        assert executor._concurrency_limit == 5
        assert executor.retry_times == 3
        assert executor.retry_delay == 0.3
        assert executor.error_handler is None

    def test_init_with_custom_params(self):
        """测试自定义参数初始化"""
        executor = ConcurrentExecutor(
            concurrency_limit=10,
            max_qps=100.0,
            retry_times=5,
            retry_delay=0.5,
        )

        assert executor._concurrency_limit == 10
        assert executor.retry_times == 5
        assert executor.retry_delay == 0.5
        assert executor._rate_limiter.max_qps == 100.0

    def test_init_with_error_handler(self):
        """测试带错误处理器初始化"""

        def handler(e, data, retry):
            return False

        executor = ConcurrentExecutor(concurrency_limit=5, error_handler=handler)

        assert executor.error_handler is handler


class TestExecuteBatch:
    """execute_batch 批量执行测试"""

    async def test_simple_func_batch(self):
        """测试简单函数批量执行"""
        executor = ConcurrentExecutor(concurrency_limit=5)

        results, progress = await executor.execute_batch(
            async_func=mock_simple_task,
            tasks_data=["a", "b", "c"],
            show_progress=False,
        )

        assert len(results) == 3
        assert all(r.status == "success" for r in results)
        assert results[0].data == "result:a"
        assert results[1].data == "result:b"
        assert results[2].data == "result:c"

    async def test_context_func_batch(self):
        """测试 TaskContext 函数批量执行"""
        executor = ConcurrentExecutor(concurrency_limit=5)

        results, _ = await executor.execute_batch(
            async_func=mock_context_task,
            tasks_data=["x", "y"],
            show_progress=False,
        )

        assert len(results) == 2
        assert results[0].data == "task_0:x:retry_0"
        assert results[1].data == "task_1:y:retry_0"

    async def test_kwargs_func_batch(self):
        """测试带 kwargs 的函数执行"""
        executor = ConcurrentExecutor(concurrency_limit=5)

        results, _ = await executor.execute_batch(
            async_func=mock_kwargs_task,
            tasks_data=["data1", "data2"],
            executor_kwargs={"user_id": 123, "mode": "fast"},
            show_progress=False,
        )

        assert len(results) == 2
        assert "user_id" in results[0].data
        assert "123" in results[0].data

    async def test_empty_tasks(self):
        """测试空任务列表处理"""
        executor = ConcurrentExecutor(concurrency_limit=5)

        results, _ = await executor.execute_batch(
            async_func=mock_simple_task,
            tasks_data=[],
            show_progress=False,
        )

        assert results == []

    async def test_single_task(self):
        """测试单任务执行"""
        executor = ConcurrentExecutor(concurrency_limit=5)

        results, _ = await executor.execute_batch(
            async_func=mock_simple_task,
            tasks_data=["single"],
            show_progress=False,
        )

        assert len(results) == 1
        assert results[0].data == "result:single"

    async def test_result_ordering(self):
        """测试结果按 task_id 排序"""
        executor = ConcurrentExecutor(concurrency_limit=10)

        # 创建大量任务以增加乱序可能性
        tasks_data = [str(i) for i in range(20)]

        results, _ = await executor.execute_batch(
            async_func=mock_simple_task,
            tasks_data=tasks_data,
            show_progress=False,
        )

        # 结果应该按 task_id 排序
        for i, result in enumerate(results):
            assert result.task_id == i


class TestConcurrencyControl:
    """并发控制测试"""

    async def test_concurrency_limit(self):
        """测试并发数限制"""
        concurrency_limit = 3
        executor = ConcurrentExecutor(concurrency_limit=concurrency_limit)

        concurrent_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        async def tracked_task(data):
            nonlocal concurrent_count, max_concurrent
            async with lock:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)

            await asyncio.sleep(0.05)

            async with lock:
                concurrent_count -= 1

            return data

        results, _ = await executor.execute_batch(
            async_func=tracked_task,
            tasks_data=list(range(10)),
            show_progress=False,
        )

        # 最大并发数应该不超过限制
        assert max_concurrent <= concurrency_limit


class TestRetryMechanism:
    """重试机制测试"""

    async def test_retry_on_error(self):
        """测试错误重试机制"""
        executor = ConcurrentExecutor(
            concurrency_limit=5,
            retry_times=2,
            retry_delay=0.01,
        )

        fail_counts = {}

        async def retry_task(data):
            fail_counts[data] = fail_counts.get(data, 0) + 1
            if fail_counts[data] <= 1:  # 第一次失败，第二次成功
                raise ValueError("temporary error")
            return f"success:{data}"

        results, _ = await executor.execute_batch(
            async_func=retry_task,
            tasks_data=["a", "b"],
            show_progress=False,
        )

        # 应该都成功（因为重试后成功了）
        assert all(r.status == "success" for r in results)
        assert all(r.retry_count == 1 for r in results)  # 重试了 1 次

    async def test_max_retries_exceeded(self):
        """测试超过最大重试次数"""
        executor = ConcurrentExecutor(
            concurrency_limit=5,
            retry_times=2,
            retry_delay=0.01,
        )

        results, _ = await executor.execute_batch(
            async_func=mock_failing_task,
            tasks_data=["x"],
            show_progress=False,
        )

        # 应该失败
        assert results[0].status == "error"
        assert results[0].error is not None
        assert results[0].retry_count == 2  # 重试了 2 次后放弃

    async def test_custom_error_handler(self):
        """测试自定义错误处理器"""
        stop_retry_errors = []

        def error_handler(e, data, retry_count):
            stop_retry_errors.append((str(e), data, retry_count))
            # 只重试一次就停止
            return retry_count < 1

        executor = ConcurrentExecutor(
            concurrency_limit=5,
            retry_times=5,  # 设置高重试次数
            retry_delay=0.01,
            error_handler=error_handler,
        )

        results, _ = await executor.execute_batch(
            async_func=mock_failing_task,
            tasks_data=["test"],
            show_progress=False,
        )

        # 错误处理器应该被调用
        assert len(stop_retry_errors) == 1
        # 由于处理器返回 False（retry_count=1 >= 1），只重试 0 次
        assert results[0].status == "error"


class TestPriorityBatch:
    """优先级批量执行测试"""

    async def test_priority_ordering(self):
        """测试优先级排序执行"""
        executor = ConcurrentExecutor(concurrency_limit=1)  # 串行执行以验证顺序

        execution_order = []

        async def ordered_task(data):
            execution_order.append(data)
            await asyncio.sleep(0.001)
            return data

        priority_tasks = [
            TaskItem(priority=3, task_id=0, data="low"),
            TaskItem(priority=1, task_id=1, data="high"),
            TaskItem(priority=2, task_id=2, data="medium"),
        ]

        results, _ = await executor.execute_priority_batch(
            async_func=ordered_task,
            priority_tasks=priority_tasks,
            show_progress=False,
        )

        # 执行顺序应该按优先级
        assert execution_order == ["high", "medium", "low"]

    async def test_priority_with_meta(self):
        """测试带 meta 的优先级任务"""
        executor = ConcurrentExecutor(concurrency_limit=5)

        priority_tasks = [
            TaskItem(priority=1, task_id=0, data="data", meta={"source": "test"}),
        ]

        results, _ = await executor.execute_priority_batch(
            async_func=mock_simple_task,
            priority_tasks=priority_tasks,
            show_progress=False,
        )

        assert results[0].meta == {"source": "test"}


class TestStreamingExecution:
    """流式执行测试"""

    async def test_aiter_execute_batch(self):
        """测试流式批量执行"""
        executor = ConcurrentExecutor(concurrency_limit=2)

        all_results = []
        batch_count = 0

        async for batch in executor.aiter_execute_batch(
            async_func=mock_simple_task,
            tasks_data=["a", "b", "c", "d"],
            show_progress=False,
            batch_size=2,
        ):
            all_results.extend(batch.completed_tasks)
            batch_count += 1

        assert len(all_results) == 4
        # 至少应该有一个 batch
        assert batch_count >= 1

    async def test_aiter_final_flag(self):
        """测试流式执行的 is_final 标记"""
        executor = ConcurrentExecutor(concurrency_limit=5)

        final_flags = []

        async for batch in executor.aiter_execute_batch(
            async_func=mock_simple_task,
            tasks_data=["a", "b"],
            show_progress=False,
        ):
            final_flags.append(batch.is_final)

        # 最后一个 batch 应该标记为 final
        assert final_flags[-1] is True


class TestSyncExecution:
    """同步版本执行测试"""

    def test_execute_batch_sync(self):
        """测试同步版本批量执行"""
        executor = ConcurrentExecutor(concurrency_limit=5)

        results, _ = executor.execute_batch_sync(
            async_func=mock_simple_task,
            tasks_data=["x", "y", "z"],
            show_progress=False,
        )

        assert len(results) == 3
        assert all(r.status == "success" for r in results)


class TestAdapterExecution:
    """适配器模式执行测试"""

    async def test_execute_with_adapter(self):
        """测试使用适配器的批量执行"""
        executor = ConcurrentExecutor(concurrency_limit=5)

        async def complex_func(item, user_id=None):
            return f"{item}:user_{user_id}"

        def adapter(data, context: TaskContext):
            # 返回 (args, kwargs)
            return data["item"], {"user_id": data["user_id"]}

        tasks_data = [
            {"item": "a", "user_id": 1},
            {"item": "b", "user_id": 2},
        ]

        results, _ = await executor.execute_batch_with_adapter(
            async_func=complex_func,
            tasks_data=tasks_data,
            task_adapter=adapter,
            show_progress=False,
        )

        assert results[0].data == "a:user_1"
        assert results[1].data == "b:user_2"


class TestFunctionSignatureInspection:
    """函数签名检测测试"""

    def test_inspect_simple_function(self):
        """测试检测简单函数签名"""
        executor = ConcurrentExecutor(concurrency_limit=5)

        sig_info = executor._inspect_function_signature(mock_simple_task)

        assert "data" in sig_info["param_names"]
        assert sig_info["has_context_param"] is False
        assert sig_info["param_count"] == 1

    def test_inspect_context_function(self):
        """测试检测 TaskContext 函数签名"""
        executor = ConcurrentExecutor(concurrency_limit=5)

        sig_info = executor._inspect_function_signature(mock_context_task)

        assert sig_info["has_context_param"] is True

    def test_inspect_kwargs_function(self):
        """测试检测带 **kwargs 的函数签名"""
        executor = ConcurrentExecutor(concurrency_limit=5)

        sig_info = executor._inspect_function_signature(mock_kwargs_task)

        assert sig_info["accepts_var_kwargs"] is True


class TestIntelligentFunctionCall:
    """智能函数调用测试"""

    async def test_call_simple_function(self):
        """测试智能调用简单函数"""
        executor = ConcurrentExecutor(concurrency_limit=5)

        ctx = TaskContext(task_id=1, data="test_data")
        result = await executor._call_function_intelligently(mock_simple_task, ctx)

        assert result == "result:test_data"

    async def test_call_context_function(self):
        """测试智能调用 TaskContext 函数"""
        executor = ConcurrentExecutor(concurrency_limit=5)

        ctx = TaskContext(task_id=5, data="ctx_data", retry_count=2)
        result = await executor._call_function_intelligently(mock_context_task, ctx)

        assert result == "task_5:ctx_data:retry_2"

    async def test_call_kwargs_function(self):
        """测试智能调用带 kwargs 的函数"""
        executor = ConcurrentExecutor(concurrency_limit=5)

        ctx = TaskContext(task_id=1, data="data")
        kwargs = {"mode": "test"}
        result = await executor._call_function_intelligently(mock_kwargs_task, ctx, kwargs)

        assert "mode" in result
        assert "test" in result


class TestLatencyTracking:
    """延迟统计测试"""

    async def test_latency_recorded(self):
        """测试延迟被正确记录"""
        executor = ConcurrentExecutor(concurrency_limit=5)

        async def slow_task(data):
            await asyncio.sleep(0.05)
            return data

        results, _ = await executor.execute_batch(
            async_func=slow_task,
            tasks_data=["a"],
            show_progress=False,
        )

        # 延迟应该至少 50ms
        assert results[0].latency >= 0.05

    async def test_error_latency_recorded(self):
        """测试错误时延迟也被记录"""
        executor = ConcurrentExecutor(
            concurrency_limit=5,
            retry_times=0,
        )

        async def slow_fail(data):
            await asyncio.sleep(0.02)
            raise ValueError("error")

        results, _ = await executor.execute_batch(
            async_func=slow_fail,
            tasks_data=["a"],
            show_progress=False,
        )

        # 即使失败，延迟也应该被记录
        assert results[0].latency >= 0.02


class TestContextExecution:
    """上下文模式执行测试"""

    async def test_execute_batch_with_context(self):
        """测试上下文模式批量执行"""
        executor = ConcurrentExecutor(concurrency_limit=5)

        async def ctx_func(context: TaskContext):
            return f"id:{context.task_id},data:{context.data},kwargs:{context.executor_kwargs}"

        results, _ = await executor.execute_batch_with_context(
            async_func=ctx_func,
            tasks_data=["x", "y"],
            executor_kwargs={"user": "test"},
            show_progress=False,
        )

        assert "id:0" in results[0].data
        assert "data:x" in results[0].data
        assert "user" in results[0].data


class TestFactoryExecution:
    """函数工厂模式执行测试"""

    async def test_execute_batch_with_factory(self):
        """测试函数工厂模式批量执行"""
        executor = ConcurrentExecutor(concurrency_limit=5)

        async def fast_processor(data):
            return f"fast:{data}"

        async def slow_processor(data):
            return f"slow:{data}"

        def factory(context: TaskContext):
            if context.task_id % 2 == 0:
                return fast_processor
            return slow_processor

        results, _ = await executor.execute_batch_with_factory(
            func_factory=factory,
            tasks_data=["a", "b", "c"],
            show_progress=False,
        )

        # task_id 为偶数使用 fast_processor
        assert results[0].data == "fast:a"  # task_id=0
        assert results[1].data == "slow:b"  # task_id=1
        assert results[2].data == "fast:c"  # task_id=2


class TestErrorHandling:
    """错误处理测试"""

    async def test_add_custom_error_handler(self):
        """测试添加自定义错误处理器"""
        executor = ConcurrentExecutor(concurrency_limit=5)

        handler_called = False

        def handler(e, data, retry):
            nonlocal handler_called
            handler_called = True
            return False

        executor.add_custom_error_handler(handler)
        assert executor.error_handler is handler

        await executor.execute_batch(
            async_func=mock_failing_task,
            tasks_data=["test"],
            show_progress=False,
        )

        assert handler_called

    async def test_partial_failures(self):
        """测试部分任务失败的情况"""
        executor = ConcurrentExecutor(
            concurrency_limit=5,
            retry_times=0,
        )

        results, _ = await executor.execute_batch(
            async_func=mock_partial_failing_task,
            tasks_data=["0", "1", "2", "3"],  # 0, 2 会失败
            show_progress=False,
        )

        success_count = sum(1 for r in results if r.status == "success")
        error_count = sum(1 for r in results if r.status == "error")

        assert success_count == 2  # 1, 3 成功
        assert error_count == 2  # 0, 2 失败
