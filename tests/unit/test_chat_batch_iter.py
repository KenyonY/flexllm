"""chat_batch_iter 的契约，以及 0.17.8 起废弃的两个旧入口。

chat_batch_iter 与 chat_batch 共用 _iter_batch：每个 index 恰好产出一次，
收集后按 index 排好就等于 chat_batch。这里对着真实 mock 服务验证单 endpoint、
pool 分布式、pool fallback 三条路径。
"""

import asyncio
import json
import time
import warnings
from contextlib import aclosing

import pytest

from flexllm import LegacyResponseWarning, LLMClient, LLMClientPool
from flexllm.clients.base import ChatCompletionResult
from flexllm.clients.completion import think_tagged_text
from flexllm.pricing.cost_tracker import CostTrackerConfig
from tests.mock_server import MockLLMServer, MockLLMServerGroup, MockServerConfig


def _msgs(n):
    return [[{"role": "user", "content": f"q{i}"}] for i in range(n)]


async def _collect(agen):
    return [item async for item in agen]


def _assert_each_index_once(items, n):
    indices = [index for index, _ in items]
    assert sorted(indices) == list(range(n))
    assert all(isinstance(result, ChatCompletionResult) for _, result in items)


def _by_index(items):
    return [result.content for _, result in sorted(items, key=lambda item: item[0])]


@pytest.fixture
def qa_path(tmp_path):
    """确定性回复：q{i} → answer-{i}，内容可以逐条比对"""
    path = tmp_path / "qa.jsonl"
    path.write_text(
        "".join(json.dumps({"input": f"q{i}", "output": f"answer-{i}"}) + "\n" for i in range(30))
    )
    return str(path)


async def _served_requests(log_path, settle):
    """等在途请求在服务端跑完再数：mock 在响应延迟结束后才写日志"""
    await asyncio.sleep(settle)
    try:
        with open(log_path) as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return 0


class TestSingleEndpoint:
    async def test_collected_items_equal_chat_batch(self, qa_path):
        cfg = MockServerConfig(port=19601, delay_min=0.01, delay_max=0.05, qa_path=qa_path)
        with MockLLMServer(cfg) as server:
            async with LLMClient(base_url=server.url, model="mock-model", api_key="k") as client:
                items = await _collect(client.chat_batch_iter(_msgs(8), show_progress=False))
                batch = await client.chat_batch(_msgs(8), show_progress=False)
        _assert_each_index_once(items, 8)
        assert _by_index(items) == [r.content for r in batch] == [f"answer-{i}" for i in range(8)]

    async def test_yields_as_each_request_completes(self):
        """不能攒满一个并发窗口才交付：延迟各不相同时，交出时刻应当分散"""
        cfg = MockServerConfig(port=19606, delay_min=0.05, delay_max=1.0)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                base_url=server.url, model="mock-model", api_key="k", concurrency_limit=8
            ) as client:
                start = time.perf_counter()
                times = [
                    time.perf_counter() - start
                    async for _ in client.chat_batch_iter(_msgs(8), show_progress=False)
                ]
        assert max(times) - min(times) > 0.2

    async def test_resume_yields_restored_items_first(self, tmp_path):
        output = str(tmp_path / "out.jsonl")
        with MockLLMServer(MockServerConfig(port=19602, delay_min=0.01, delay_max=0.01)) as server:
            async with LLMClient(base_url=server.url, model="mock-model", api_key="k") as client:
                first = await client.chat_batch(_msgs(3), output_jsonl=output, show_progress=False)
                items = await _collect(
                    client.chat_batch_iter(_msgs(5), output_jsonl=output, show_progress=False)
                )
        _assert_each_index_once(items, 5)
        # 恢复项排在最前，内容与上一轮落盘的一致
        assert sorted(index for index, _ in items[:3]) == [0, 1, 2]
        restored = dict(items[:3])
        assert [restored[i].content for i in range(3)] == [r.content for r in first]
        with open(output) as f:
            assert len([line for line in f if line.strip()]) == 5

    async def test_failures_are_yielded_not_raised(self):
        cfg = MockServerConfig(port=19603, delay_min=0.01, delay_max=0.01, error_rate=1.0)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                base_url=server.url, model="mock-model", api_key="k", retry_delay=0.01
            ) as client:
                items = await _collect(client.chat_batch_iter(_msgs(3), show_progress=False))
        _assert_each_index_once(items, 3)
        assert all(not result.ok and result.content is None for _, result in items)

    async def test_early_break_cancels_pending_and_closes_checkpoint(self, tmp_path):
        output = str(tmp_path / "out.jsonl")
        log = str(tmp_path / "requests.jsonl")
        cfg = MockServerConfig(port=19604, delay_min=0.3, delay_max=0.3, log_path=log)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                base_url=server.url, model="mock-model", api_key="k", concurrency_limit=2
            ) as client:
                start = time.perf_counter()
                async with aclosing(
                    client.chat_batch_iter(_msgs(20), output_jsonl=output, show_progress=False)
                ) as items:
                    async for _ in items:
                        break
                elapsed = time.perf_counter() - start
                served = await _served_requests(log, settle=0.6)
        # 跑完整批要 ~3s；break 后应立即返回，且不再发新请求（窗口 2 + 至多一轮补位）
        assert elapsed < 1.5
        assert served <= 4
        with open(output) as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert 1 <= len(records) < 20

    @pytest.mark.parametrize("concurrency", [1, 3])
    async def test_budget_stop_delivers_every_sent_request(self, tmp_path, concurrency):
        """超预算：不再发新请求；已发出的（服务端已处理、已计费）全部交出，没发出的带 error"""
        log = str(tmp_path / "requests.jsonl")
        port = 19607 + concurrency
        cfg = MockServerConfig(port=port, delay_min=0.05, delay_max=0.05, log_path=log)
        with MockLLMServer(cfg) as server:
            async with LLMClient(
                base_url=server.url,
                model="gpt-4o",
                api_key="k",
                concurrency_limit=concurrency,
                cost_tracker=CostTrackerConfig.with_budget(limit=1e-12),
            ) as client:
                items = await _collect(client.chat_batch_iter(_msgs(12), show_progress=False))
                served = await _served_requests(log, settle=0.2)
        _assert_each_index_once(items, 12)
        ok = [result for _, result in items if result.ok]
        failed = [result for _, result in items if not result.ok]
        assert served < 12
        assert len(ok) == served
        assert failed and all("未发出请求" in str(result.error) for result in failed)


class TestPool:
    @staticmethod
    def _configs(base_port, **overrides):
        return [
            MockServerConfig(port=base_port, delay_min=0.01, delay_max=0.05, **overrides),
            MockServerConfig(port=base_port + 1, delay_min=0.01, delay_max=0.05),
        ]

    @staticmethod
    def _pool(group, **kwargs):
        return LLMClientPool(
            endpoints=[
                {"base_url": url, "model": "mock-model", "api_key": "k"} for url in group.urls
            ],
            **kwargs,
        )

    async def test_distributed_collected_items_equal_chat_batch(self, qa_path):
        configs = [
            MockServerConfig(port=port, delay_min=0.01, delay_max=0.05, qa_path=qa_path)
            for port in (19611, 19612)
        ]
        with MockLLMServerGroup(configs) as group:
            async with self._pool(group) as pool:
                items = await _collect(pool.chat_batch_iter(_msgs(12), show_progress=False))
                batch = await pool.chat_batch(_msgs(12), show_progress=False)
        _assert_each_index_once(items, 12)
        assert _by_index(items) == [r.content for r in batch] == [f"answer-{i}" for i in range(12)]

    async def test_distributed_same_base_url_all_failing_terminates(self):
        """同 base_url 不同 api_key 是两个 endpoint：全失败时要结束，而不是无限重排队"""
        cfg = MockServerConfig(port=19619, delay_min=0.01, delay_max=0.01, error_rate=1.0)
        with MockLLMServer(cfg) as server:
            endpoints = [
                {"base_url": server.url, "model": "mock-model", "api_key": key}
                for key in ("k1", "k2")
            ]
            async with LLMClientPool(endpoints=endpoints, retry_delay=0.01) as pool:
                items = await asyncio.wait_for(
                    _collect(pool.chat_batch_iter(_msgs(3), show_progress=False)), timeout=30
                )
        _assert_each_index_once(items, 3)
        assert all(not result.ok for _, result in items)

    async def test_distributed_resume_yields_restored_items(self, tmp_path):
        output = str(tmp_path / "out.jsonl")
        with MockLLMServerGroup(self._configs(19613)) as group:
            async with self._pool(group) as pool:
                await pool.chat_batch(_msgs(4), output_jsonl=output, show_progress=False)
                items = await _collect(
                    pool.chat_batch_iter(_msgs(6), output_jsonl=output, show_progress=False)
                )
        _assert_each_index_once(items, 6)
        assert sorted(index for index, _ in items[:4]) == [0, 1, 2, 3]

    async def test_distributed_early_break_stops_workers(self):
        configs = [
            MockServerConfig(port=19615, delay_min=0.3, delay_max=0.3),
            MockServerConfig(port=19616, delay_min=0.3, delay_max=0.3),
        ]
        with MockLLMServerGroup(configs) as group:
            async with self._pool(group, concurrency_limit=1) as pool:
                start = time.perf_counter()
                async with aclosing(pool.chat_batch_iter(_msgs(20), show_progress=False)) as items:
                    async for _ in items:
                        break
                elapsed = time.perf_counter() - start
        # 关闭时要等 worker 退出；worker 若没被取消，会一直跑到 20 条做完（~3s）
        assert elapsed < 1.5

    async def test_fallback_retries_failed_items_on_next_endpoint(self):
        configs = self._configs(19617, error_rate=1.0)
        with MockLLMServerGroup(configs) as group:
            async with self._pool(group, retry_delay=0.01) as pool:
                items = await _collect(
                    pool.chat_batch_iter(_msgs(4), distribute=False, show_progress=False)
                )
        _assert_each_index_once(items, 4)
        assert all(result.ok for _, result in items)


class TestDeprecatedEntryPoints:
    async def test_chat_completions_stream_warns_but_keeps_output(self):
        with MockLLMServer(MockServerConfig(port=19621, delay_min=0.01, delay_max=0.01)) as server:
            async with LLMClient(base_url=server.url, model="mock-model", api_key="k") as client:
                with pytest.warns(LegacyResponseWarning, match="chat_stream") as record:
                    chunks = await _collect(client.chat_completions_stream(_msgs(1)[0]))
        assert chunks and all(isinstance(chunk, str) for chunk in chunks)
        # 警告指向调用方，而不是 flexllm 内部
        assert record[0].filename == __file__

    async def test_iter_chat_completions_batch_warns_but_keeps_output(self):
        with MockLLMServer(MockServerConfig(port=19622, delay_min=0.01, delay_max=0.01)) as server:
            async with LLMClient(base_url=server.url, model="mock-model", api_key="k") as client:
                with pytest.warns(LegacyResponseWarning, match="chat_batch_iter") as record:
                    results = await _collect(
                        client.iter_chat_completions_batch(_msgs(3), show_progress=False)
                    )
        # 旧形状：带 original_idx/status，最后一条挂 summary
        assert sorted(r.original_idx for r in results) == [0, 1, 2]
        assert all(r.status == "success" and isinstance(r.content, str) for r in results)
        assert results[-1].summary["total"] == 3
        assert record[0].filename == __file__

    async def test_new_entry_points_do_not_warn(self):
        with MockLLMServer(MockServerConfig(port=19623, delay_min=0.01, delay_max=0.01)) as server:
            async with LLMClient(base_url=server.url, model="mock-model", api_key="k") as client:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", LegacyResponseWarning)
                    await _collect(client.chat_stream(_msgs(1)[0]))
                    await _collect(client.chat_batch_iter(_msgs(2), show_progress=False))


async def test_think_tagged_text_wraps_thinking_and_reports_result():
    result = ChatCompletionResult(content="answer")

    async def events():
        yield {"type": "thinking", "content": "hmm"}
        yield {"type": "content", "content": "answer"}
        yield {"type": "result", "result": result}

    received = []
    text = "".join(await _collect(think_tagged_text(events(), on_result=received.append)))
    assert text == "<think>\nhmm</think>answer"
    assert received == [result]
