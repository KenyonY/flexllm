"""
LLMClientPool 高并发测试

测试场景：
1. 基本正确性：高并发下所有任务都能正确完成
2. Fallback 高并发：一个 endpoint 高失败率，另一个正常
3. 单 endpoint 失败：无 fallback 时的错误处理
4. 慢节点负载均衡：快节点处理更多任务
5. 全部失败：所有 endpoint 都失败时的处理
6. 大量任务同时 fallback：100% 失败触发 fallback

运行方式:
    pytest tests/integration/test_pool_concurrency.py -v -s
"""

import asyncio

import pytest

from flexllm import LLMClientPool
from flexllm.mock import MockLLMServerGroup, MockServerConfig


def create_messages(n: int) -> list[list[dict]]:
    """创建 n 条测试消息"""
    return [[{"role": "user", "content": f"Test message {i}"}] for i in range(n)]


class TestPoolConcurrencyBasic:
    """基本高并发测试"""

    @pytest.mark.asyncio
    async def test_basic_correctness(self):
        """场景1: 基本正确性 - 2 endpoint, 100 并发, 1000 任务"""
        configs = [
            MockServerConfig(port=19001, delay_min=0.01, delay_max=0.01),
            MockServerConfig(port=19002, delay_min=0.01, delay_max=0.01),
        ]

        with MockLLMServerGroup(configs) as group:
            async with LLMClientPool(
                endpoints=group.endpoints,
                concurrency_limit=50,  # 每个 endpoint 50 并发，共 100
                fallback=True,
            ) as pool:
                messages_list = create_messages(1000)

                results, summary = await pool.chat_completions_batch(
                    messages_list,
                    show_progress=True,
                    return_summary=True,
                )

                # 验证：所有任务都有结果
                success_count = len([r for r in results if r is not None])
                assert success_count == 1000, f"Expected 1000 success, got {success_count}"
                assert summary["success"] == 1000
                assert summary["failed"] == 0


class TestPoolConcurrencyFallback:
    """Fallback 机制高并发测试"""

    @pytest.mark.asyncio
    async def test_fallback_high_concurrency(self):
        """场景2: Fallback 高并发 - A 失败率 80%, B 正常"""
        configs = [
            MockServerConfig(port=19003, delay_min=0.01, delay_max=0.01, error_rate=0.8),
            MockServerConfig(port=19004, delay_min=0.01, delay_max=0.01, error_rate=0),
        ]

        with MockLLMServerGroup(configs) as group:
            async with LLMClientPool(
                endpoints=group.endpoints,
                concurrency_limit=50,
                fallback=True,
            ) as pool:
                messages_list = create_messages(500)

                results, summary = await pool.chat_completions_batch(
                    messages_list,
                    show_progress=True,
                    return_summary=True,
                )

                # 验证：所有任务最终成功（通过 fallback 到 B）
                success_count = len([r for r in results if r is not None])
                assert success_count == 500, f"Expected 500 success, got {success_count}"
                assert summary["failed"] == 0

    @pytest.mark.asyncio
    async def test_single_endpoint_with_errors(self):
        """场景3: 单 endpoint 失败测试 - 失败率 30%，无 fallback

        注意：单 endpoint 禁用 fallback 时，会走 _batch_with_fallback 路径。
        此时内部 client 的重试机制仍然有效，所以实际成功率可能高于理论值。
        """
        configs = [
            MockServerConfig(port=19005, delay_min=0.01, delay_max=0.01, error_rate=0.3),
        ]

        with MockLLMServerGroup(configs) as group:
            async with LLMClientPool(
                endpoints=group.endpoints,
                concurrency_limit=50,
                fallback=False,  # 禁用 fallback
                retry_times=0,  # 禁用内部重试，观察真实失败率
            ) as pool:
                messages_list = create_messages(200)

                results, summary = await pool.chat_completions_batch(
                    messages_list,
                    show_progress=False,  # 单 endpoint 禁用进度条
                    return_summary=True,
                )

                # 验证：成功率约 70%（允许一定误差）
                success_count = len([r for r in results if r is not None])
                # 期望 70% 成功，允许 ±20% 的误差（随机性较大）
                assert 100 <= success_count <= 180, f"Expected ~140 success, got {success_count}"
                # 程序不应卡死，能正常返回
                assert len(results) == 200


class TestPoolConcurrencyLoadBalance:
    """负载均衡测试"""

    @pytest.mark.asyncio
    async def test_slow_node_load_balance(self):
        """场景4: 慢节点负载均衡 - A 延迟 0.01s, B 延迟 0.5s"""
        configs = [
            MockServerConfig(port=19006, delay_min=0.01, delay_max=0.01, model="fast-model"),
            MockServerConfig(port=19007, delay_min=0.5, delay_max=0.5, model="slow-model"),
        ]

        with MockLLMServerGroup(configs) as group:
            async with LLMClientPool(
                endpoints=group.endpoints,
                concurrency_limit=10,
                fallback=True,
            ) as pool:
                messages_list = create_messages(100)

                results, summary = await pool.chat_completions_batch(
                    messages_list,
                    show_progress=True,
                    return_summary=True,
                )

                # 验证：所有任务成功
                success_count = len([r for r in results if r is not None])
                assert success_count == 100

                # 验证：快节点处理的任务数应该更多
                # 由于是动态负载均衡，快节点应处理大部分任务
                # 这里无法直接统计，但可以通过执行时间间接验证
                # 如果负载均衡有效，总时间应该远小于 100 * 0.5s = 50s
                assert summary["elapsed"] < 30  # 应该在 30 秒内完成


class TestPoolConcurrencyEdgeCases:
    """边界情况测试"""

    @pytest.mark.asyncio
    async def test_all_endpoints_fail(self):
        """场景5: 全部失败 - 2 endpoint 都 100% 失败"""
        configs = [
            MockServerConfig(port=19008, delay_min=0.01, delay_max=0.01, error_rate=1.0),
            MockServerConfig(port=19009, delay_min=0.01, delay_max=0.01, error_rate=1.0),
        ]

        with MockLLMServerGroup(configs) as group:
            async with LLMClientPool(
                endpoints=group.endpoints,
                concurrency_limit=20,
                fallback=True,
            ) as pool:
                messages_list = create_messages(100)

                results, summary = await pool.chat_completions_batch(
                    messages_list,
                    show_progress=True,
                    return_summary=True,
                )

                # 验证：所有任务都失败（返回 None）
                fail_count = len([r for r in results if r is None])
                assert fail_count == 100, f"Expected 100 failures, got {fail_count}"
                assert summary["failed"] == 100
                # 程序不应卡死

    @pytest.mark.asyncio
    async def test_massive_fallback(self):
        """场景6: 大量任务同时 fallback - A 100% 失败, B 正常, 100 并发"""
        configs = [
            MockServerConfig(port=19010, delay_min=0.01, delay_max=0.01, error_rate=1.0),
            MockServerConfig(port=19011, delay_min=0.01, delay_max=0.01, error_rate=0),
        ]

        with MockLLMServerGroup(configs) as group:
            async with LLMClientPool(
                endpoints=group.endpoints,
                concurrency_limit=50,  # 总共 100 并发
                fallback=True,
            ) as pool:
                messages_list = create_messages(500)

                results, summary = await pool.chat_completions_batch(
                    messages_list,
                    show_progress=True,
                    return_summary=True,
                )

                # 验证：所有任务最终被 B 处理，无任务丢失
                success_count = len([r for r in results if r is not None])
                assert success_count == 500, f"Expected 500 success, got {success_count}"
                assert summary["failed"] == 0


class TestPoolConcurrencyStress:
    """压力测试（可选，耗时较长）"""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_stress_high_volume(self):
        """压力测试: 3 endpoint, 150 并发, 5000 任务"""
        configs = [
            MockServerConfig(port=19012, delay_min=0.01, delay_max=0.05),
            MockServerConfig(port=19013, delay_min=0.01, delay_max=0.05),
            MockServerConfig(port=19014, delay_min=0.01, delay_max=0.05),
        ]

        with MockLLMServerGroup(configs) as group:
            async with LLMClientPool(
                endpoints=group.endpoints,
                concurrency_limit=50,  # 每个 endpoint 50 并发，共 150
                fallback=True,
            ) as pool:
                messages_list = create_messages(5000)

                results, summary = await pool.chat_completions_batch(
                    messages_list,
                    show_progress=True,
                    return_summary=True,
                )

                # 验证：所有任务成功
                success_count = len([r for r in results if r is not None])
                assert success_count == 5000
                assert summary["failed"] == 0

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_stress_mixed_errors(self):
        """压力测试: 混合错误率"""
        configs = [
            MockServerConfig(port=19015, delay_min=0.01, delay_max=0.05, error_rate=0.1),
            MockServerConfig(port=19016, delay_min=0.01, delay_max=0.05, error_rate=0.2),
            MockServerConfig(port=19017, delay_min=0.01, delay_max=0.05, error_rate=0.05),
        ]

        with MockLLMServerGroup(configs) as group:
            async with LLMClientPool(
                endpoints=group.endpoints,
                concurrency_limit=30,
                fallback=True,
            ) as pool:
                messages_list = create_messages(2000)

                results, summary = await pool.chat_completions_batch(
                    messages_list,
                    show_progress=True,
                    return_summary=True,
                )

                # 验证：大部分任务成功（通过 fallback）
                success_count = len([r for r in results if r is not None])
                # 由于有 fallback，成功率应该很高
                assert success_count >= 1900, f"Expected >= 1900 success, got {success_count}"
