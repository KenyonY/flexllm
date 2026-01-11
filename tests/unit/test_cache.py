"""Test response caching functionality"""

import pytest
import time
from flexllm import LLMClient, ResponseCacheConfig


class TestResponseCache:
    """Test caching functionality"""

    @pytest.mark.asyncio
    async def test_cache_hit(self, siliconflow_config):
        """Test that cache returns same result for same input"""
        import uuid
        # Use unique message to avoid cache from previous runs
        unique_msg = [{"role": "user", "content": f"What is 1+1? ID:{uuid.uuid4()}"}]

        client = LLMClient(
            **siliconflow_config,
            cache=ResponseCacheConfig(enabled=True, ttl=3600),
        )

        # First call - cache miss
        start1 = time.time()
        result1 = await client.chat_completions(unique_msg)
        time1 = time.time() - start1

        # Second call - should hit cache (much faster)
        start2 = time.time()
        result2 = await client.chat_completions(unique_msg)
        time2 = time.time() - start2

        assert result1 == result2
        # Cache hit should be faster (at least 10x for network call)
        assert time2 < time1 * 0.1 or time2 < 0.01  # Either 10x faster or < 10ms

    @pytest.mark.asyncio
    async def test_cache_disabled(self, siliconflow_config, simple_messages):
        """Test that disabled cache always makes API calls"""
        client = LLMClient(
            **siliconflow_config,
            cache=ResponseCacheConfig(enabled=False),
        )

        result1 = await client.chat_completions(simple_messages)
        result2 = await client.chat_completions(simple_messages)

        # Both should succeed (may or may not be equal due to model variance)
        assert result1 is not None
        assert result2 is not None


class TestCacheConfig:
    """Test cache configuration"""

    def test_cache_config_defaults(self):
        """Test default cache config"""
        config = ResponseCacheConfig()
        assert config.enabled == False

    def test_cache_config_enabled(self):
        """Test enabled cache config"""
        config = ResponseCacheConfig(enabled=True)
        assert config.enabled == True

    def test_cache_config_with_ttl(self):
        """Test cache config with TTL"""
        config = ResponseCacheConfig.with_ttl(3600)
        assert config.enabled == True
        assert config.ttl == 3600

    def test_cache_config_persistent(self):
        """Test persistent cache config"""
        config = ResponseCacheConfig.persistent()
        assert config.enabled == True
        assert config.ttl == 0  # No expiration
