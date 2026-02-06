"""MemoryStore 单元测试"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

flaxkv2 = pytest.importorskip("flaxkv2")


@pytest.fixture
def memory_store(tmp_path):
    from flexllm.agent.memory import MemoryStore

    store = MemoryStore(db_path=str(tmp_path / "test_memory"))
    yield store
    store.close()


class TestMemoryStoreInit:
    def test_init_creates_db(self, memory_store):
        assert memory_store._db is not None

    def test_init_without_flaxkv2(self):
        with patch.dict("sys.modules", {"flaxkv2": None}):
            from importlib import reload

            from flexllm.agent import memory

            with pytest.raises(ImportError, match="flaxkv2"):
                reload(memory)
                memory.MemoryStore("/tmp/test_no_flaxkv2")


class TestSessionOperations:
    def test_save_and_load(self, memory_store):
        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ]
        memory_store.save("s1", messages)
        loaded = memory_store.load("s1")
        assert loaded == messages

    def test_load_nonexistent_returns_empty(self, memory_store):
        assert memory_store.load("nonexistent") == []

    def test_list_sessions(self, memory_store):
        memory_store.save("s1", [{"role": "user", "content": "a"}])
        memory_store.save("s2", [{"role": "user", "content": "b"}])
        sessions = memory_store.list_sessions()
        assert set(sessions) == {"s1", "s2"}

    def test_delete_session(self, memory_store):
        memory_store.save("s1", [{"role": "user", "content": "a"}])
        memory_store.delete_session("s1")
        assert memory_store.load("s1") == []
        assert "s1" not in memory_store.list_sessions()

    def test_delete_nonexistent_session(self, memory_store):
        # 不应抛异常
        memory_store.delete_session("nonexistent")


class TestFactOperations:
    def test_save_and_get_fact(self, memory_store):
        memory_store.save_fact("user.name", "张三")
        assert memory_store.get_fact("user.name") == "张三"

    def test_get_nonexistent_fact(self, memory_store):
        assert memory_store.get_fact("nonexistent") is None

    def test_get_facts_all(self, memory_store):
        memory_store.save_fact("user.name", "张三")
        memory_store.save_fact("user.lang", "中文")
        facts = memory_store.get_facts()
        assert facts == {"user.name": "张三", "user.lang": "中文"}

    def test_get_facts_with_prefix(self, memory_store):
        memory_store.save_fact("user.name", "张三")
        memory_store.save_fact("user.lang", "中文")
        memory_store.save_fact("system.theme", "dark")
        facts = memory_store.get_facts(prefix="user.")
        assert facts == {"user.name": "张三", "user.lang": "中文"}
        assert "system.theme" not in facts

    def test_delete_fact(self, memory_store):
        memory_store.save_fact("k1", "v1")
        memory_store.delete_fact("k1")
        assert memory_store.get_fact("k1") is None

    def test_delete_nonexistent_fact(self, memory_store):
        # 不应抛异常
        memory_store.delete_fact("nonexistent")


class TestContextManager:
    def test_context_manager(self, tmp_path):
        from flexllm.agent.memory import MemoryStore

        with MemoryStore(db_path=str(tmp_path / "ctx_test")) as store:
            store.save_fact("k", "v")
            assert store.get_fact("k") == "v"


class TestAgentClientMemoryIntegration:
    """AgentClient + MemoryStore 集成测试（mock MemoryStore，不依赖 flaxkv2）"""

    def _make_agent(self, memory=None, session_id=None):
        from flexllm.agent.client import AgentClient

        mock_client = MagicMock()
        return AgentClient(
            client=mock_client,
            memory=memory,
            session_id=session_id,
        )

    def test_init_loads_history_from_memory(self):
        memory = MagicMock()
        memory.load.return_value = [
            {"role": "user", "content": "之前的对话"},
            {"role": "assistant", "content": "之前的回复"},
        ]
        agent = self._make_agent(memory=memory, session_id="s1")
        memory.load.assert_called_once_with("s1")
        assert len(agent._history) == 2

    def test_init_without_memory(self):
        agent = self._make_agent()
        assert agent._history == []

    def test_init_with_memory_but_no_session_id(self):
        memory = MagicMock()
        agent = self._make_agent(memory=memory, session_id=None)
        memory.load.assert_not_called()
        assert agent._history == []

    @pytest.mark.asyncio
    async def test_chat_saves_to_memory(self):
        memory = MagicMock()
        memory.load.return_value = []

        agent = self._make_agent(memory=memory, session_id="s1")

        # Mock _run_loop 返回一个 AgentResult
        from flexllm.agent.types import AgentResult

        mock_result = AgentResult(content="回复内容", rounds=0, tool_calls=[])
        agent._run_loop = AsyncMock(return_value=mock_result)

        await agent.chat("你好")

        # 验证 memory.save 被调用
        memory.save.assert_called_once_with(
            "s1",
            [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "回复内容"},
            ],
        )

    @pytest.mark.asyncio
    async def test_chat_without_memory_does_not_save(self):
        agent = self._make_agent()

        from flexllm.agent.types import AgentResult

        mock_result = AgentResult(content="回复", rounds=0, tool_calls=[])
        agent._run_loop = AsyncMock(return_value=mock_result)

        await agent.chat("你好")
        # 没有 memory，不应该报错
