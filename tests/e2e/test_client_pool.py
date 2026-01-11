"""Test LLMClientPool functionality"""

import pytest
from flexllm import LLMClientPool


class TestClientPool:
    """Test client pool for load balancing"""

    @pytest.fixture
    def pool_config(self, siliconflow_config):
        """Create pool with single endpoint for testing"""
        return {
            "endpoints": [
                {
                    "base_url": siliconflow_config["base_url"],
                    "api_key": siliconflow_config["api_key"],
                    "model": siliconflow_config["model"],
                }
            ],
        }

    def test_pool_creation(self, pool_config):
        """Test pool creation"""
        pool = LLMClientPool(**pool_config)
        assert pool is not None

    def test_pool_sync_call(self, pool_config, simple_messages):
        """Test sync call through pool"""
        pool = LLMClientPool(**pool_config)
        result = pool.chat_completions_sync(simple_messages)

        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_pool_async_call(self, pool_config, simple_messages):
        """Test async call through pool"""
        pool = LLMClientPool(**pool_config)
        result = await pool.chat_completions(simple_messages)

        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_pool_batch_call(self, pool_config, batch_messages):
        """Test batch call through pool"""
        pool = LLMClientPool(**pool_config)
        results = await pool.chat_completions_batch(batch_messages)

        assert results is not None
        assert len(results) == 3
