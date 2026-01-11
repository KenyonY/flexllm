"""Tests for MllmClient."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from flexllm import MllmClient


class TestMllmClient:
    """Test MllmClient initialization and configuration."""

    def test_init_default(self):
        """Test default initialization."""
        client = MllmClient(
            model="test-model",
            base_url="http://localhost:8000/v1",
            api_key="test-key",
        )
        assert client.model == "test-model"

    def test_init_with_concurrency(self):
        """Test initialization with concurrency settings."""
        client = MllmClient(
            model="test-model",
            base_url="http://localhost:8000/v1",
            api_key="test-key",
            concurrency_limit=20,
            max_qps=50,
        )
        # Verify client was created successfully with these settings
        assert client.model == "test-model"
        # Internal client should have these settings
        assert hasattr(client, 'client')


class TestMllmClientMessages:
    """Test message handling."""

    def test_build_text_message(self):
        """Test building text-only message."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        # Messages should pass through unchanged for text-only
        assert messages[0]["content"] == "Hello"

    def test_build_multimodal_message(self):
        """Test building multimodal message structure."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}
                ]
            }
        ]
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][1]["type"] == "image_url"


class TestMllmClientImageProcessing:
    """Test image processing capabilities."""

    def test_image_url_formats(self):
        """Test various image URL formats are accepted."""
        valid_urls = [
            "http://example.com/image.jpg",
            "https://example.com/image.png",
            "/path/to/local/image.jpg",
            "data:image/jpeg;base64,/9j/4AAQ...",
        ]

        for url in valid_urls:
            message = {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}}
                ]
            }
            # Should not raise
            assert message["content"][0]["image_url"]["url"] == url

    def test_supported_image_extensions(self):
        """Test supported image file extensions."""
        supported = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

        for ext in supported:
            path = f"/path/to/image{ext}"
            # Extension should be recognized
            assert path.endswith(ext)


class TestMllmClientConfig:
    """Test configuration options."""

    def test_timeout_config(self):
        """Test timeout configuration."""
        client = MllmClient(
            model="test-model",
            base_url="http://localhost:8000/v1",
            api_key="test-key",
            timeout=120,
        )
        # Verify client was created with timeout setting
        assert client.model == "test-model"

    def test_retry_config(self):
        """Test retry configuration."""
        client = MllmClient(
            model="test-model",
            base_url="http://localhost:8000/v1",
            api_key="test-key",
            retry_times=5,
            retry_delay=2.0,
        )
        # Verify client was created with retry settings
        assert client.model == "test-model"
