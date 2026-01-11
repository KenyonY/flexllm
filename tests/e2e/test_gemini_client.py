"""Test GeminiClient functionality"""

import pytest
from flexllm import LLMClient


class TestGeminiClient:
    """Test Gemini API through LLMClient"""

    def test_gemini_sync(self, gemini_config, simple_messages):
        """Test Gemini sync call"""
        client = LLMClient(**gemini_config)
        result = client.chat_completions_sync(simple_messages)

        assert result is not None
        assert isinstance(result, str)
        assert "2" in result

    @pytest.mark.asyncio
    async def test_gemini_async(self, gemini_config, simple_messages):
        """Test Gemini async call"""
        client = LLMClient(**gemini_config)
        result = await client.chat_completions(simple_messages)

        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_gemini_batch(self, gemini_config, batch_messages):
        """Test Gemini batch call"""
        client = LLMClient(**gemini_config)
        results = await client.chat_completions_batch(batch_messages)

        assert results is not None
        assert len(results) == 3
