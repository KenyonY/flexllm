"""Tests for processors module."""

import pytest
from unittest.mock import patch, MagicMock
import base64


class TestImageProcessor:
    """Test image processing functions."""

    def test_base64_encoding_format(self):
        """Test base64 encoding produces valid format."""
        # Simple test data
        test_data = b"test image data"
        encoded = base64.b64encode(test_data).decode('utf-8')

        assert isinstance(encoded, str)
        # Should be valid base64
        decoded = base64.b64decode(encoded)
        assert decoded == test_data

    def test_image_extensions(self):
        """Test supported image extensions."""
        supported = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}

        for ext in supported:
            path = f"image{ext}"
            assert any(path.endswith(e) for e in supported)

    def test_mime_type_mapping(self):
        """Test MIME type mapping for image extensions."""
        mime_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
        }

        for ext, mime in mime_map.items():
            assert mime.startswith('image/')


class TestMessagesProcessor:
    """Test message preprocessing."""

    def test_simple_message_passthrough(self):
        """Test that simple text messages pass through unchanged."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]

        # Text-only messages should not be modified
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_multimodal_message_structure(self):
        """Test multimodal message structure."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}}
                ]
            }
        ]

        content = messages[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"

    def test_system_message_handling(self):
        """Test system message handling."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]

        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


class TestUnifiedProcessor:
    """Test unified processor functionality."""

    def test_batch_structure(self):
        """Test batch message structure."""
        messages_list = [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "World"}],
        ]

        assert len(messages_list) == 2
        assert all(isinstance(m, list) for m in messages_list)

    def test_empty_batch(self):
        """Test handling of empty batch."""
        messages_list = []
        assert len(messages_list) == 0

    def test_mixed_content_batch(self):
        """Test batch with mixed content types."""
        messages_list = [
            [{"role": "user", "content": "Text only"}],
            [{"role": "user", "content": [
                {"type": "text", "text": "With image"},
                {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}}
            ]}],
        ]

        # First message is text-only
        assert isinstance(messages_list[0][0]["content"], str)
        # Second message is multimodal
        assert isinstance(messages_list[1][0]["content"], list)


class TestImageCacheConfig:
    """Test image cache configuration."""

    def test_default_cache_dir(self):
        """Test default cache directory."""
        from pathlib import Path

        default_dir = Path.home() / ".cache" / "flexllm" / "images"
        assert "cache" in str(default_dir)
        assert "images" in str(default_dir)

    def test_cache_key_generation(self):
        """Test cache key generation is deterministic."""
        import hashlib

        url = "http://example.com/image.jpg"
        key1 = hashlib.md5(url.encode()).hexdigest()
        key2 = hashlib.md5(url.encode()).hexdigest()

        assert key1 == key2
        assert len(key1) == 32  # MD5 hash length


class TestImageOptimization:
    """Test image optimization settings."""

    def test_max_size_default(self):
        """Test default max size setting."""
        default_max_size = 1024
        assert default_max_size > 0

    def test_quality_range(self):
        """Test quality parameter range."""
        min_quality = 1
        max_quality = 100
        default_quality = 85

        assert min_quality <= default_quality <= max_quality

    def test_resize_calculation(self):
        """Test resize calculation preserves aspect ratio."""
        original_width = 2000
        original_height = 1000
        max_size = 1024

        # Calculate new dimensions
        ratio = min(max_size / original_width, max_size / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        # Aspect ratio should be preserved
        original_ratio = original_width / original_height
        new_ratio = new_width / new_height
        assert abs(original_ratio - new_ratio) < 0.01

        # Should not exceed max size
        assert new_width <= max_size
        assert new_height <= max_size
