"""Test LLMClient core functionality"""

import pytest
from flexllm import LLMClient, ResponseCacheConfig


class TestLLMClientSync:
    """Test synchronous LLMClient methods"""

    def test_chat_completions_sync(self, siliconflow_config, simple_messages):
        """Test basic sync chat completion"""
        client = LLMClient(**siliconflow_config)
        result = client.chat_completions_sync(simple_messages)

        assert result is not None
        assert isinstance(result, str)
        assert "2" in result

    def test_chat_completions_sync_with_system(self, siliconflow_config):
        """Test sync chat with system message"""
        client = LLMClient(**siliconflow_config)
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is Python?"},
        ]
        result = client.chat_completions_sync(messages)

        assert result is not None
        assert len(result) > 0


class TestLLMClientAsync:
    """Test async LLMClient methods"""

    @pytest.mark.asyncio
    async def test_chat_completions(self, siliconflow_config, simple_messages):
        """Test basic async chat completion"""
        client = LLMClient(**siliconflow_config)
        result = await client.chat_completions(simple_messages)

        assert result is not None
        assert isinstance(result, str)
        assert "2" in result

    @pytest.mark.asyncio
    async def test_chat_completions_batch(self, siliconflow_config, batch_messages):
        """Test batch async chat completion"""
        client = LLMClient(**siliconflow_config)
        results = await client.chat_completions_batch(batch_messages)

        assert results is not None
        assert len(results) == 3
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_chat_completions_with_usage(self, siliconflow_config, simple_messages):
        """Test chat completion with usage info"""
        client = LLMClient(**siliconflow_config)
        result = await client.chat_completions(simple_messages, return_usage=True)

        assert result is not None
        assert hasattr(result, "content")
        assert hasattr(result, "usage")


class TestLLMClientBatchSync:
    """Test batch sync methods"""

    def test_chat_completions_batch_sync(self, siliconflow_config, batch_messages):
        """Test batch sync chat completion"""
        client = LLMClient(**siliconflow_config)
        results = client.chat_completions_batch_sync(batch_messages)

        assert results is not None
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)
