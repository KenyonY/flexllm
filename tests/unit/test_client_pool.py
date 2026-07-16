"""Unit tests for LLMClientPool"""

from unittest.mock import AsyncMock, MagicMock, patch

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


class TestFromConfig:
    """测试 from_config 工厂方法"""

    def _mock_config(self, user_template=None):
        """构造 mock FlexLLMConfig"""
        config = MagicMock()
        config.config = {"default": "qwen-plus"}
        config.get_model_config.return_value = {
            "id": "qwen-plus",
            "name": "qwen-plus",
            "base_url": "https://api.example.com/v1",
            "api_key": "sk-test",
        }
        config.get_system.return_value = "你是一个有用的助手"
        config.get_user_template.return_value = user_template
        config.get_model_params.return_value = {"temperature": 0.7, "max_tokens": 1024}
        return config

    @patch("flexllm.cli.config.get_config")
    def test_from_config_default_model(self, mock_get_config):
        """无参数使用默认模型"""
        mock_get_config.return_value = self._mock_config()

        client = LLMClientPool.from_config()
        assert client._mode == "single"
        assert client._config_system == "你是一个有用的助手"
        assert client._config_params == {"temperature": 0.7, "max_tokens": 1024}

    @patch("flexllm.cli.config.get_config")
    def test_from_config_named_model(self, mock_get_config):
        """指定模型名称"""
        mock_get_config.return_value = self._mock_config()

        LLMClientPool.from_config(model="qwen-plus")
        mock_get_config.return_value.get_model_config.assert_called_with("qwen-plus")

    @patch("flexllm.cli.config.get_config")
    def test_from_config_overrides(self, mock_get_config):
        """overrides 覆盖配置"""
        mock_get_config.return_value = self._mock_config()

        client = LLMClientPool.from_config(model="qwen-plus", concurrency_limit=50)
        assert client._single_client._concurrency_limit == 50

    @patch("flexllm.cli.config.get_config")
    def test_from_config_not_found(self, mock_get_config):
        """模型未找到时抛出 ValueError"""
        config = MagicMock()
        config.get_model_config.return_value = None
        mock_get_config.return_value = config

        with pytest.raises(ValueError, match="未找到模型配置"):
            LLMClientPool.from_config(model="nonexistent")

    @patch("flexllm.cli.config.get_config")
    def test_from_config_no_system(self, mock_get_config):
        """配置中没有 system prompt 时 _config_system 为 None"""
        config = self._mock_config()
        config.get_system.return_value = None
        mock_get_config.return_value = config

        client = LLMClientPool.from_config()
        assert client._config_system is None

    @patch("flexllm.cli.config.get_config")
    def test_from_config_with_provider(self, mock_get_config):
        """配置中有 provider 字段"""
        config = self._mock_config()
        config.get_model_config.return_value = {
            "id": "claude-3",
            "name": "claude-3",
            "provider": "claude",
            "api_key": "sk-ant-test",
        }
        mock_get_config.return_value = config

        client = LLMClientPool.from_config(model="claude-3")
        assert client._provider == "claude"

    @pytest.mark.asyncio
    @patch("flexllm.cli.config.get_config")
    async def test_config_params_merged_in_chat_completions(self, mock_get_config):
        """chat_completions 中配置参数作为默认值合并"""
        mock_get_config.return_value = self._mock_config()

        client = LLMClientPool.from_config()

        with patch.object(
            client._single_client, "chat_completions", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = "response"
            await client.chat_completions(
                [{"role": "user", "content": "你好"}],
                temperature=0.9,  # 显式传入覆盖配置的 0.7
            )
            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["temperature"] == 0.9  # 用户显式传入的优先
            assert call_kwargs["max_tokens"] == 1024  # 配置中的默认值

    @pytest.mark.asyncio
    @patch("flexllm.cli.config.get_config")
    async def test_config_system_injected(self, mock_get_config):
        """messages 中没有 system 时自动注入配置的 system prompt"""
        mock_get_config.return_value = self._mock_config()

        client = LLMClientPool.from_config()

        with patch.object(
            client._single_client, "chat_completions", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = "response"
            await client.chat_completions([{"role": "user", "content": "你好"}])
            call_args = mock_chat.call_args
            messages = call_args[1]["messages"]
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "你是一个有用的助手"
            assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    @patch("flexllm.cli.config.get_config")
    async def test_config_system_not_injected_when_present(self, mock_get_config):
        """messages 中已有 system 时不注入"""
        mock_get_config.return_value = self._mock_config()

        client = LLMClientPool.from_config()

        with patch.object(
            client._single_client, "chat_completions", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = "response"
            await client.chat_completions(
                [
                    {"role": "system", "content": "自定义 system"},
                    {"role": "user", "content": "你好"},
                ]
            )
            call_args = mock_chat.call_args
            messages = call_args[1]["messages"]
            assert messages[0]["content"] == "自定义 system"
            assert len([m for m in messages if m["role"] == "system"]) == 1

    @pytest.mark.asyncio
    @patch("flexllm.cli.config.get_config")
    async def test_string_input_converted_to_messages(self, mock_get_config):
        """字符串输入自动转换为 messages"""
        mock_get_config.return_value = self._mock_config()

        client = LLMClientPool.from_config()

        with patch.object(
            client._single_client, "chat_completions", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = "response"
            await client.chat_completions("你好")
            messages = mock_chat.call_args[1]["messages"]
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "你是一个有用的助手"
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "你好"

    @pytest.mark.asyncio
    @patch("flexllm.cli.config.get_config")
    async def test_string_input_with_user_template(self, mock_get_config):
        """字符串输入 + user_template"""
        mock_get_config.return_value = self._mock_config(user_template="{content}/detail")

        client = LLMClientPool.from_config()

        with patch.object(
            client._single_client, "chat_completions", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = "response"
            await client.chat_completions("分析代码")
            messages = mock_chat.call_args[1]["messages"]
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "分析代码/detail"

    @pytest.mark.asyncio
    @patch("flexllm.cli.config.get_config")
    async def test_list_input_user_template_not_applied(self, mock_get_config):
        """list[dict] 输入时不应用 user_template"""
        mock_get_config.return_value = self._mock_config(user_template="{content}/detail")

        client = LLMClientPool.from_config()

        with patch.object(
            client._single_client, "chat_completions", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = "response"
            await client.chat_completions([{"role": "user", "content": "你好"}])
            messages = mock_chat.call_args[1]["messages"]
            # list[dict] 输入时 user_template 不生效，用户自己构建的 messages 应保持原样
            assert messages[1]["content"] == "你好"

    @pytest.mark.asyncio
    @patch("flexllm.cli.config.get_config")
    async def test_batch_string_input(self, mock_get_config):
        """batch 方法支持字符串列表"""
        mock_get_config.return_value = self._mock_config()

        client = LLMClientPool.from_config()

        with patch.object(
            client._single_client, "chat_completions_batch", new_callable=AsyncMock
        ) as mock_batch:
            mock_batch.return_value = ["r1", "r2"]
            await client.chat_completions_batch(["问题1", "问题2"], show_progress=False)
            messages_list = mock_batch.call_args[1]["messages_list"]
            assert len(messages_list) == 2
            assert messages_list[0][0]["role"] == "system"
            assert messages_list[0][1]["content"] == "问题1"
            assert messages_list[1][1]["content"] == "问题2"

    @patch("flexllm.cli.config.FlexLLMConfig")
    def test_from_config_with_custom_path(self, mock_cls):
        """指定配置文件路径"""
        mock_cls.return_value = self._mock_config()

        LLMClientPool.from_config("/path/to/config.yaml", model="qwen-plus")
        mock_cls.assert_called_once_with("/path/to/config.yaml")

    def test_default_config_attrs_on_normal_init(self):
        """正常 __init__ 创建的实例 _config_system 和 _config_params 为空"""
        pool = LLMClientPool(
            base_url="http://api.example.com/v1",
            api_key="key",
            model="test",
        )
        assert pool._config_system is None
        assert pool._config_params == {}
        assert pool._config_user_template is None


class TestCapacityAwareSelection:
    """测试容量感知选路（issue #13）"""

    def _make_pool(self, slow_limit=2, fast_limit=20):
        return LLMClientPool(
            endpoints=[
                {"base_url": "http://slow.com/v1", "model": "m", "concurrency_limit": slow_limit},
                {"base_url": "http://fast.com/v1", "model": "m", "concurrency_limit": fast_limit},
            ],
        )

    def test_router_capacity_comes_from_client(self):
        """router 的容量取自底层 client 的实际并发上限"""
        pool = self._make_pool()
        limits = {p.config.base_url: p.config.concurrency_limit for p in pool._router._providers}
        assert limits == {"http://slow.com/v1": 2, "http://fast.com/v1": 20}

    def test_router_capacity_default_from_shared_param(self):
        """endpoint 未指定 concurrency_limit 时容量取共享默认值"""
        pool = LLMClientPool(
            endpoints=[
                {"base_url": "http://api1.com/v1", "model": "m"},
                {"base_url": "http://api2.com/v1", "model": "m"},
            ],
            concurrency_limit=7,
        )
        assert all(p.config.concurrency_limit == 7 for p in pool._router._providers)

    async def test_routes_to_least_loaded_endpoint(self):
        """有在途请求的 endpoint 应被避开"""
        pool = self._make_pool(slow_limit=10, fast_limit=10)
        pool._router._providers[0].in_flight = 3

        pool._clients[0].chat_completions = AsyncMock(return_value="slow")
        pool._clients[1].chat_completions = AsyncMock(return_value="fast")

        result = await pool.chat_completions("hi")

        assert result == "fast"
        assert pool._clients[0].chat_completions.await_count == 0
        # release 后 in_flight 恢复到调用前
        assert pool._router._providers[1].in_flight == 0

    async def test_fanout_avoids_saturated_endpoint(self):
        """并发扇出时，慢 endpoint 饱和后流量应全部流向快 endpoint"""
        import asyncio

        pool = self._make_pool(slow_limit=2, fast_limit=20)

        async def respond(**kwargs):
            await asyncio.sleep(0.05)
            return "ok"

        pool._clients[0].chat_completions = AsyncMock(side_effect=respond)
        pool._clients[1].chat_completions = AsyncMock(side_effect=respond)

        results = await asyncio.gather(*[pool.chat_completions("hi") for _ in range(12)])

        assert results == ["ok"] * 12
        # slow limit=2 只拿 2 个，其余 10 个进 fast（盲轮询下会是 6/6）
        assert pool._clients[0].chat_completions.await_count == 2
        assert pool._clients[1].chat_completions.await_count == 10
        assert all(p.in_flight == 0 for p in pool._router._providers)

    async def test_fallback_excludes_tried_endpoint(self):
        """fallback 时不会重复尝试同一 endpoint，失败后 in-flight 归零"""
        pool = self._make_pool(slow_limit=10, fast_limit=10)

        pool._clients[0].chat_completions = AsyncMock(side_effect=RuntimeError("boom"))
        pool._clients[1].chat_completions = AsyncMock(return_value="ok")

        result = await pool.chat_completions("hi")

        assert result == "ok"
        assert pool._clients[0].chat_completions.await_count == 1
        assert pool._clients[1].chat_completions.await_count == 1
        assert all(p.in_flight == 0 for p in pool._router._providers)

    async def test_all_fail_raises_each_tried_once(self):
        """全部失败时抛出异常，且每个 endpoint 只被尝试一次"""
        pool = self._make_pool(slow_limit=10, fast_limit=10)

        pool._clients[0].chat_completions = AsyncMock(side_effect=RuntimeError("boom1"))
        pool._clients[1].chat_completions = AsyncMock(side_effect=RuntimeError("boom2"))

        with pytest.raises(RuntimeError):
            await pool.chat_completions("hi")

        assert pool._clients[0].chat_completions.await_count == 1
        assert pool._clients[1].chat_completions.await_count == 1
        assert all(p.in_flight == 0 for p in pool._router._providers)

    async def test_stream_counts_in_flight(self):
        """流式请求计入 in-flight，覆盖整个流的生命周期"""
        pool = self._make_pool(slow_limit=10, fast_limit=10)

        async def fake_stream(**kwargs):
            yield "a"
            yield "b"

        pool._clients[0].chat_completions_stream = fake_stream
        pool._clients[1].chat_completions_stream = fake_stream

        agen = pool.chat_completions_stream("hi")
        chunks = [await anext(agen)]
        # 流进行中：恰有一个 endpoint in_flight == 1
        assert sorted(p.in_flight for p in pool._router._providers) == [0, 1]

        async for chunk in agen:
            chunks.append(chunk)

        assert chunks == ["a", "b"]
        assert all(p.in_flight == 0 for p in pool._router._providers)
