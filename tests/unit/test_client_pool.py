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
    """Test batch method parameters including track_cost and return_cost_report"""

    @pytest.fixture
    def pool(self):
        """Create a test pool"""
        return LLMClientPool(
            endpoints=[
                {"base_url": "http://api1.com/v1", "api_key": "key1", "model": "test-model"},
            ]
        )

    def test_chat_completions_batch_sync_signature(self, pool):
        """Test that chat_completions_batch_sync has track_cost and return_cost_report"""
        import inspect

        sig = inspect.signature(pool.chat_completions_batch_sync)
        params = list(sig.parameters.keys())

        assert "track_cost" in params
        assert "return_cost_report" in params
        assert "show_progress" in params
        assert "return_summary" in params
        assert "output_jsonl" in params

    @pytest.mark.asyncio
    async def test_chat_completions_batch_signature(self, pool):
        """Test that chat_completions_batch has track_cost and return_cost_report"""
        import inspect

        sig = inspect.signature(pool.chat_completions_batch)
        params = list(sig.parameters.keys())

        assert "track_cost" in params
        assert "return_cost_report" in params
        assert "show_progress" in params
        assert "return_summary" in params

    @pytest.mark.asyncio
    async def test_single_mode_track_cost_passthrough(self):
        """Test that track_cost is passed through in single endpoint mode"""
        pool = LLMClientPool(
            base_url="http://api1.com/v1",
            api_key="key1",
            model="test-model",
        )
        assert pool._mode == "single"

        mock_result = ["Test response"]
        with patch.object(
            pool._single_client, "chat_completions_batch", new_callable=AsyncMock
        ) as mock_batch:
            mock_batch.return_value = mock_result
            await pool.chat_completions_batch(
                [[{"role": "user", "content": "test"}]],
                track_cost=True,
                return_cost_report=True,
                show_progress=False,
            )
            call_kwargs = mock_batch.call_args[1]
            assert call_kwargs["track_cost"] is True
            assert call_kwargs["return_cost_report"] is True


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


class TestClientPoolDistributedAttributes:
    """验证 _batch_distributed 中属性直接在 base client 上可访问（回归测试）"""

    def test_response_cache_on_base_client(self):
        """base client 有 _response_cache 属性（修复前 client._client 导致获取不到）"""
        pool = LLMClientPool(
            endpoints=[
                {"base_url": "http://api1.com/v1", "model": "m1"},
            ]
        )
        client = pool._clients[0]
        # _response_cache 应直接在 base client 上（默认为 None）
        assert hasattr(client, "_response_cache")
        cache = getattr(client, "_response_cache", "MISSING")
        assert cache != "MISSING"

    def test_concurrency_limit_on_base_client(self):
        """base client 有 _concurrency_limit 属性"""
        pool = LLMClientPool(
            endpoints=[
                {"base_url": "http://api1.com/v1", "model": "m1", "api_key": "k1"},
            ],
            concurrency_limit=7,
        )
        client = pool._clients[0]
        assert hasattr(client, "_concurrency_limit")
        assert client._concurrency_limit == 7

    def test_concurrency_limit_not_on_nested_client(self):
        """确保不再需要 client._client 来获取 _concurrency_limit"""
        pool = LLMClientPool(
            endpoints=[
                {"base_url": "http://api1.com/v1", "model": "m1"},
            ],
            concurrency_limit=5,
        )
        client = pool._clients[0]
        # client._client 是 ConcurrentRequester，不应该用来获取 _concurrency_limit
        assert getattr(client, "_concurrency_limit", None) == 5


class TestInferProvider:
    """测试 _infer_provider 推断逻辑"""

    def test_openai_default(self):
        assert LLMClientPool._infer_provider("http://api.openai.com/v1", False) == "openai"

    def test_custom_url_defaults_to_openai(self):
        assert LLMClientPool._infer_provider("http://my-server.com/v1", False) == "openai"

    def test_gemini_url(self):
        assert (
            LLMClientPool._infer_provider("https://generativelanguage.googleapis.com/v1", False)
            == "gemini"
        )

    def test_vertex_ai_url(self):
        assert (
            LLMClientPool._infer_provider("https://us-central1-aiplatform.googleapis.com", False)
            == "gemini"
        )

    def test_vertex_ai_flag(self):
        """use_vertex_ai=True 时无论 URL 都返回 gemini"""
        assert LLMClientPool._infer_provider("http://anything.com", True) == "gemini"

    def test_claude_url(self):
        assert LLMClientPool._infer_provider("https://api.anthropic.com/v1", False) == "claude"

    def test_empty_url(self):
        assert LLMClientPool._infer_provider("", False) == "openai"

    def test_none_url(self):
        assert LLMClientPool._infer_provider(None, False) == "openai"


class TestClientPoolGetattr:
    """测试 __getattr__ 委托"""

    def test_single_mode_delegates(self):
        """单模式下委托到底层客户端"""
        pool = LLMClientPool(
            base_url="http://api1.com/v1",
            api_key="key1",
            model="test-model",
        )
        assert pool._mode == "single"
        # _single_client 的属性应可以通过 pool 访问
        assert hasattr(pool, "_concurrency_limit")

    def test_multi_mode_raises(self):
        """多模式下访问未定义属性抛 AttributeError"""
        pool = LLMClientPool(
            endpoints=[
                {"base_url": "http://api1.com/v1", "model": "m1"},
                {"base_url": "http://api2.com/v1", "model": "m2"},
            ]
        )
        with pytest.raises(AttributeError, match="仅单 endpoint 模式支持自动委托"):
            _ = pool.nonexistent_attribute


class TestChatCompletionsOrRaise:
    """测试 chat_completions_or_raise"""

    @pytest.mark.asyncio
    async def test_success_returns_content(self):
        """成功时返回内容"""
        pool = LLMClientPool(
            base_url="http://api1.com/v1",
            api_key="key1",
            model="test-model",
        )
        with patch.object(
            pool._single_client,
            "chat_completions",
            new_callable=AsyncMock,
            return_value="hello",
        ):
            result = await pool.chat_completions_or_raise([{"role": "user", "content": "test"}])
            assert result == "hello"

    @pytest.mark.asyncio
    async def test_failure_raises_runtime_error(self):
        """失败时抛出 RuntimeError"""
        from flexllm.async_api.interface import RequestResult

        pool = LLMClientPool(
            base_url="http://api1.com/v1",
            api_key="key1",
            model="test-model",
        )
        mock_result = RequestResult(
            request_id=0, status=500, data="Internal Server Error", latency=0.1
        )
        with patch.object(
            pool._single_client,
            "chat_completions",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            with pytest.raises(RuntimeError, match="LLM 请求失败"):
                await pool.chat_completions_or_raise([{"role": "user", "content": "test"}])


class TestClientPoolRepr:
    """Test string representation"""

    def test_repr(self):
        """Test pool repr"""
        pool = LLMClientPool(
            endpoints=[
                {"base_url": "http://api1.com/v1", "model": "model1"},
                {"base_url": "http://api2.com/v1", "model": "model2"},
            ],
            fallback=True,
        )

        repr_str = repr(pool)
        assert "LLMClientPool" in repr_str
        assert "endpoints=2" in repr_str
        assert "fallback=True" in repr_str
