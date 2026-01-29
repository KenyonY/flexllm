"""Unit tests for LLMClientPool"""

from unittest.mock import AsyncMock, patch

import pytest

from flexllm import LLMClientPool
from flexllm.clients.base import ChatCompletionResult


class TestClientPoolCreation:
    """Test pool creation and initialization"""

    def test_create_with_endpoints(self):
        """Test pool creation with endpoint configs"""
        pool = LLMClientPool(
            endpoints=[
                {"base_url": "http://api1.com/v1", "api_key": "key1", "model": "model1"},
                {"base_url": "http://api2.com/v1", "api_key": "key2", "model": "model2"},
            ]
        )
        assert len(pool._clients) == 2
        assert len(pool._endpoints) == 2

    def test_create_with_single_endpoint(self):
        """Test pool creation with single endpoint"""
        pool = LLMClientPool(
            endpoints=[
                {"base_url": "http://api1.com/v1", "api_key": "key1", "model": "model1"},
            ]
        )
        assert len(pool._clients) == 1

    def test_create_requires_endpoints_or_clients(self):
        """Test that creation requires endpoints or clients"""
        with pytest.raises(ValueError, match="必须提供 base_url.*或 endpoints"):
            LLMClientPool()

    def test_create_with_load_balance_strategy(self):
        """Test pool creation with different load balance strategies"""
        for strategy in ["round_robin", "weighted", "random", "fallback"]:
            pool = LLMClientPool(
                endpoints=[
                    {"base_url": "http://api1.com/v1", "model": "model1"},
                ],
                load_balance=strategy,
            )
            assert pool._load_balance == strategy

    def test_create_with_concurrency_limit(self):
        """Test pool creation with concurrency limit"""
        pool = LLMClientPool(
            endpoints=[
                {"base_url": "http://api1.com/v1", "model": "model1"},
            ],
            concurrency_limit=5,
        )
        # 现在 _clients 直接存储底层客户端，不再是 LLMClient 包装
        assert pool._clients[0]._concurrency_limit == 5


class TestClientPoolBatchParameters:
    """Test batch method parameters including track_cost"""

    @pytest.fixture
    def pool(self):
        """Create a test pool"""
        return LLMClientPool(
            endpoints=[
                {"base_url": "http://api1.com/v1", "api_key": "key1", "model": "test-model"},
            ]
        )

    def test_chat_completions_batch_sync_signature(self, pool):
        """Test that chat_completions_batch_sync has track_cost parameter"""
        import inspect

        sig = inspect.signature(pool.chat_completions_batch_sync)
        params = list(sig.parameters.keys())

        assert "track_cost" in params
        assert "show_progress" in params
        assert "return_summary" in params
        assert "output_jsonl" in params

    @pytest.mark.asyncio
    async def test_chat_completions_batch_signature(self, pool):
        """Test that chat_completions_batch has track_cost parameter"""
        import inspect

        sig = inspect.signature(pool.chat_completions_batch)
        params = list(sig.parameters.keys())

        assert "track_cost" in params
        assert "show_progress" in params
        assert "return_summary" in params


class TestClientPoolTrackCost:
    """Test track_cost functionality"""

    @pytest.mark.asyncio
    async def test_track_cost_enables_return_usage_in_fallback_mode(self):
        """Test that track_cost=True enables return_usage in fallback mode"""
        pool = LLMClientPool(
            endpoints=[
                {"base_url": "http://api1.com/v1", "model": "test-model"},
            ]
        )

        # Mock the client's chat_completions_batch method (used in _batch_with_fallback)
        mock_result = [
            ChatCompletionResult(
                content="Test",
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            )
        ]

        with patch.object(
            pool._clients[0], "chat_completions_batch", new_callable=AsyncMock
        ) as mock_batch:
            mock_batch.return_value = mock_result

            # Use distribute=False to use _batch_with_fallback
            await pool.chat_completions_batch(
                [[{"role": "user", "content": "test"}]],
                track_cost=True,
                show_progress=False,
                distribute=False,
            )

            # Verify return_usage or track_cost was passed as True
            call_kwargs = mock_batch.call_args[1]
            assert call_kwargs.get("return_usage") is True or call_kwargs.get("track_cost") is True


class TestClientPoolOutputJsonl:
    """Test output_jsonl functionality"""

    @pytest.mark.asyncio
    async def test_output_jsonl_extension_validation(self):
        """Test that output_jsonl must have .jsonl extension"""
        pool = LLMClientPool(
            endpoints=[
                {"base_url": "http://api1.com/v1", "model": "test-model"},
            ]
        )

        with pytest.raises(ValueError, match="必须使用 .jsonl 扩展名"):
            await pool.chat_completions_batch(
                [[{"role": "user", "content": "test"}]],
                output_jsonl="output.json",  # Wrong extension
            )


class TestClientPoolRepr:
    """Test string representation"""

    def test_repr(self):
        """Test pool repr"""
        pool = LLMClientPool(
            endpoints=[
                {"base_url": "http://api1.com/v1", "model": "model1"},
                {"base_url": "http://api2.com/v1", "model": "model2"},
            ],
            load_balance="round_robin",
            fallback=True,
        )

        repr_str = repr(pool)
        assert "LLMClientPool" in repr_str
        assert "endpoints=2" in repr_str
        assert "round_robin" in repr_str
        assert "fallback=True" in repr_str
