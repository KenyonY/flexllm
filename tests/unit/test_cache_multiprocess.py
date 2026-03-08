"""多进程读写缓存测试

验证 flaxkv2 (LMDB) 在多进程并发场景下的正确性。
测试场景：
1. 多进程并发写入不同 key → 所有数据可读
2. 多进程并发读取 → 数据一致
3. 多进程同时读写 → 读到的值要么是旧值要么是新值，不出现损坏
4. 多进程写入相同 key → 最终值是其中一个进程写入的（不出现混合/损坏）
5. 多进程 batch 写入 + 单进程读取 → 数据完整
6. 跨进程可见性 → 进程 A 写入关闭后进程 B 立即可读
7. 大 value 多进程写入 → 不截断不损坏
"""

import json
import multiprocessing as mp
import shutil
import tempfile

import pytest

from flexllm.cache.response_cache import ResponseCache, ResponseCacheConfig


def _make_cache(cache_dir: str) -> ResponseCache:
    """在子进程中创建独立的 ResponseCache 实例"""
    config = ResponseCacheConfig(enabled=True, cache_dir=cache_dir, ttl=3600)
    return ResponseCache(config)


def _msg(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


# ========== 子进程 worker 函数（必须定义在模块顶层） ==========


def _worker_write_disjoint(cache_dir: str, worker_id: int, count: int):
    """每个进程写入不重叠的 key 范围"""
    cache = _make_cache(cache_dir)
    for i in range(count):
        key = f"w{worker_id}_q{i}"
        cache.set(_msg(key), {"worker": worker_id, "index": i, "content": f"a_{key}"}, model="m1")
    cache.close()


def _worker_read_to_queue(cache_dir: str, keys: list[str], queue: mp.Queue, reader_id: int):
    """读取指定 key，将结果通过 Queue 返回（避免 manager.dict 的 pickle 问题）"""
    cache = _make_cache(cache_dir)
    results = {}
    for key in keys:
        val = cache.get(_msg(key), model="m1")
        results[key] = val
    cache.close()
    queue.put((reader_id, json.dumps(results, ensure_ascii=False)))


def _worker_write_same_key(cache_dir: str, worker_id: int, iterations: int):
    """多个进程竞争写同一个 key"""
    cache = _make_cache(cache_dir)
    for i in range(iterations):
        cache.set(
            _msg("shared_key"),
            {"writer": worker_id, "iteration": i, "data": f"value_from_w{worker_id}_i{i}"},
            model="m1",
        )
    cache.close()


def _worker_concurrent_read_write(cache_dir: str, worker_id: int, count: int):
    """同时读写：写自己的 key，读其他进程的 key"""
    cache = _make_cache(cache_dir)
    for i in range(count):
        cache.set(
            _msg(f"rw_w{worker_id}_q{i}"),
            {"worker": worker_id, "index": i},
            model="m1",
        )
        # 尝试读其他进程的 key（可能还没写入）
        other_worker = (worker_id + 1) % 4
        cache.get(_msg(f"rw_w{other_worker}_q{i}"), model="m1")
    cache.close()


def _worker_batch_write(cache_dir: str, worker_id: int, count: int):
    """批量写入"""
    cache = _make_cache(cache_dir)
    msgs = [_msg(f"batch_w{worker_id}_q{i}") for i in range(count)]
    responses = [{"worker": worker_id, "index": i, "content": f"batch_a_{i}"} for i in range(count)]
    cache.set_batch(msgs, responses, model="m1")
    cache.close()


def _worker_large_write(cache_dir: str, worker_id: int):
    """写入大 value"""
    cache = _make_cache(cache_dir)
    large_content = "x" * 50_000  # 50KB
    for i in range(10):
        cache.set(
            _msg(f"large_w{worker_id}_q{i}"),
            {"content": large_content, "worker": worker_id, "index": i},
            model="m1",
        )
    cache.close()


class TestCacheMultiProcess:
    @pytest.fixture
    def cache_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def _run_workers(self, procs: list[mp.Process]):
        """启动并等待所有子进程，断言全部成功"""
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)
            assert p.exitcode == 0, f"{p.name} exitcode={p.exitcode}"

    def test_concurrent_write_disjoint_keys(self, cache_dir):
        """多进程写入不重叠的 key，全部数据应可读"""
        num_workers = 4
        count_per_worker = 50

        procs = [
            mp.Process(target=_worker_write_disjoint, args=(cache_dir, wid, count_per_worker))
            for wid in range(num_workers)
        ]
        self._run_workers(procs)

        # 主进程验证所有数据
        cache = _make_cache(cache_dir)
        for wid in range(num_workers):
            for i in range(count_per_worker):
                key = f"w{wid}_q{i}"
                val = cache.get(_msg(key), model="m1")
                assert val is not None, f"key={key} 丢失"
                assert val["worker"] == wid
                assert val["index"] == i
        cache.close()

    def test_concurrent_read_after_write(self, cache_dir):
        """先写入数据，多进程并发读取，验证每个读者都读到完全一致的数据"""
        total_keys = 100
        cache = _make_cache(cache_dir)
        keys = []
        for i in range(total_keys):
            key = f"read_q{i}"
            keys.append(key)
            cache.set(_msg(key), {"index": i, "content": f"read_a{i}"}, model="m1")
        cache.close()

        # 多进程并发读，通过 Queue 收集结果
        num_readers = 4
        queue = mp.Queue()
        procs = [
            mp.Process(target=_worker_read_to_queue, args=(cache_dir, keys, queue, rid))
            for rid in range(num_readers)
        ]
        self._run_workers(procs)

        # 验证每个读者的结果
        for _ in range(num_readers):
            reader_id, results_json = queue.get(timeout=5)
            results = json.loads(results_json)
            for i, key in enumerate(keys):
                val = results[key]
                assert val is not None, f"reader={reader_id}, key={key} 丢失"
                assert val["index"] == i
                assert val["content"] == f"read_a{i}"

    def test_concurrent_read_write(self, cache_dir):
        """多进程同时读写，不应崩溃或返回损坏数据"""
        num_workers = 4
        count_per_worker = 50

        procs = [
            mp.Process(
                target=_worker_concurrent_read_write, args=(cache_dir, wid, count_per_worker)
            )
            for wid in range(num_workers)
        ]
        self._run_workers(procs)

        # 验证每个进程自己写的数据都在
        cache = _make_cache(cache_dir)
        for wid in range(num_workers):
            for i in range(count_per_worker):
                val = cache.get(_msg(f"rw_w{wid}_q{i}"), model="m1")
                assert val is not None, f"worker={wid}, index={i} 丢失"
                assert val["worker"] == wid
                assert val["index"] == i
        cache.close()

    def test_concurrent_write_same_key(self, cache_dir):
        """多进程竞争写同一个 key，最终值必须是某个进程写入的完整值（非损坏）"""
        num_workers = 4
        iterations = 100

        procs = [
            mp.Process(target=_worker_write_same_key, args=(cache_dir, wid, iterations))
            for wid in range(num_workers)
        ]
        self._run_workers(procs)

        # 最终值应该是某个进程写入的完整值
        cache = _make_cache(cache_dir)
        val = cache.get(_msg("shared_key"), model="m1")
        assert val is not None, "shared_key 丢失"
        # 验证值结构完整（不是两个进程写入的混合）
        assert "writer" in val
        assert "iteration" in val
        assert "data" in val
        writer = val["writer"]
        iteration = val["iteration"]
        assert 0 <= writer < num_workers
        assert 0 <= iteration < iterations
        assert val["data"] == f"value_from_w{writer}_i{iteration}"
        cache.close()

    def test_concurrent_batch_write(self, cache_dir):
        """多进程 batch 写入，主进程全部读取验证"""
        num_workers = 4
        count_per_worker = 50

        procs = [
            mp.Process(target=_worker_batch_write, args=(cache_dir, wid, count_per_worker))
            for wid in range(num_workers)
        ]
        self._run_workers(procs)

        # 主进程验证
        cache = _make_cache(cache_dir)
        for wid in range(num_workers):
            msgs = [_msg(f"batch_w{wid}_q{i}") for i in range(count_per_worker)]
            cached, uncached = cache.get_batch(msgs, model="m1")
            assert uncached == [], f"worker={wid} 有 {len(uncached)} 个 key 丢失"
            for i, val in enumerate(cached):
                assert val["worker"] == wid
                assert val["index"] == i
        cache.close()

    def test_write_visibility_across_processes(self, cache_dir):
        """进程 A 写入后关闭，进程 B 能立即读到（验证持久化而非内存缓存）"""
        # 进程 A 写入
        p = mp.Process(target=_worker_write_disjoint, args=(cache_dir, 99, 10))
        p.start()
        p.join(timeout=10)
        assert p.exitcode == 0

        # 进程 B 读取
        queue = mp.Queue()
        keys = [f"w99_q{i}" for i in range(10)]
        p = mp.Process(target=_worker_read_to_queue, args=(cache_dir, keys, queue, 0))
        p.start()
        p.join(timeout=10)
        assert p.exitcode == 0

        _, results_json = queue.get(timeout=5)
        results = json.loads(results_json)
        for i in range(10):
            key = f"w99_q{i}"
            assert results[key] is not None, f"key={key} 未持久化"
            assert results[key]["index"] == i

    def test_large_value_multiprocess(self, cache_dir):
        """多进程写入大 value（模拟真实 LLM 响应），验证不截断不损坏"""
        procs = [mp.Process(target=_worker_large_write, args=(cache_dir, wid)) for wid in range(4)]
        self._run_workers(procs)

        cache = _make_cache(cache_dir)
        for wid in range(4):
            for i in range(10):
                val = cache.get(_msg(f"large_w{wid}_q{i}"), model="m1")
                assert val is not None
                assert len(val["content"]) == 50_000
                assert val["worker"] == wid
        cache.close()
