"""Test that all modules can be imported correctly"""

import pytest


class TestImports:
    """Test module imports"""

    def test_import_main(self):
        """Test main package import"""
        import flexllm
        assert flexllm is not None

    def test_import_llm_client(self):
        """Test LLMClient import"""
        from flexllm import LLMClient
        assert LLMClient is not None

    def test_import_openai_client(self):
        """Test OpenAIClient import"""
        from flexllm import OpenAIClient
        assert OpenAIClient is not None

    def test_import_gemini_client(self):
        """Test GeminiClient import"""
        from flexllm import GeminiClient
        assert GeminiClient is not None

    def test_import_cache_config(self):
        """Test ResponseCacheConfig import"""
        from flexllm import ResponseCacheConfig
        assert ResponseCacheConfig is not None

    def test_import_client_pool(self):
        """Test LLMClientPool import"""
        from flexllm import LLMClientPool
        assert LLMClientPool is not None

    def test_import_provider_router(self):
        """Test ProviderRouter import"""
        from flexllm import ProviderRouter, ProviderConfig, create_router_from_urls
        assert ProviderRouter is not None
        assert ProviderConfig is not None
        assert create_router_from_urls is not None

    def test_import_base_client(self):
        """Test base client imports"""
        from flexllm import LLMClientBase, ChatCompletionResult, BatchResultItem
        assert LLMClientBase is not None
        assert ChatCompletionResult is not None
        assert BatchResultItem is not None

    def test_import_async_api(self):
        """Test async_api imports"""
        from flexllm.async_api import ConcurrentRequester
        assert ConcurrentRequester is not None

    def test_import_utils(self):
        """Test utils imports"""
        from flexllm.utils import async_retry
        assert async_retry is not None
