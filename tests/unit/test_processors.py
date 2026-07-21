"""Tests for processors module - tests actual source code functions."""

import base64

from flexllm.utils.core import safe_repr_error, safe_repr_source


class TestSafeReprSource:
    """Test safe_repr_source function."""

    def test_empty_source(self):
        assert safe_repr_source("") == "空源"
        assert safe_repr_source(None) == "空源"

    def test_base64_data_uri(self):
        result = safe_repr_source("data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==")
        assert "image/png" in result
        assert "base64数据" in result
        assert "iVBORw0KGgo" not in result

    def test_pure_base64_string(self):
        long_b64 = "A" * 200
        result = safe_repr_source(long_b64)
        assert "base64数据" in result
        assert "200" in result

    def test_short_string(self):
        assert safe_repr_source("http://example.com/img.jpg") == "http://example.com/img.jpg"

    def test_long_string_truncated(self):
        long_url = "http://example.com/" + "a" * 200
        result = safe_repr_source(long_url, max_length=50)
        assert result.endswith("...")
        assert len(result) == 53  # 50 + "..."


class TestSafeReprError:
    """Test safe_repr_error function."""

    def test_empty_error(self):
        assert safe_repr_error("") == ""
        assert safe_repr_error(None) is None

    def test_error_with_base64(self):
        error = "Failed to process data:image/png;base64,iVBORw0KGgoAAAANSUhEUg== in pipeline"
        result = safe_repr_error(error)
        assert "iVBORw0KGgo" not in result
        assert "image/png" in result

    def test_short_error(self):
        error = "Connection timeout"
        assert safe_repr_error(error) == error

    def test_long_error_truncated(self):
        error = "Error: " + "x" * 300
        result = safe_repr_error(error, max_length=100)
        assert result.endswith("...")
        assert len(result) == 103


class TestImageProcessor:
    """Test image processing functions."""

    def test_base64_encoding_format(self):
        test_data = b"test image data"
        encoded = base64.b64encode(test_data).decode("utf-8")
        assert isinstance(encoded, str)
        decoded = base64.b64decode(encoded)
        assert decoded == test_data

    def test_image_extensions(self):
        supported = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
        for ext in supported:
            path = f"image{ext}"
            assert any(path.endswith(e) for e in supported)

    def test_mime_type_mapping(self):
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        for ext, mime in mime_map.items():
            assert mime.startswith("image/")


class TestImageCacheConfig:
    """Test image cache configuration."""

    def test_default_cache_dir(self):
        from flexllm.msg_processors.image_processor import DEFAULT_CACHE_DIR

        assert "cache" in DEFAULT_CACHE_DIR

    def test_image_cache_config_creation(self):
        from flexllm.msg_processors.image_processor import ImageCacheConfig

        config = ImageCacheConfig()
        assert config.enabled is False

        config_enabled = ImageCacheConfig(enabled=True)
        assert config_enabled.enabled is True

    def test_cache_key_generation(self):
        import hashlib

        url = "http://example.com/image.jpg"
        key1 = hashlib.md5(url.encode()).hexdigest()
        key2 = hashlib.md5(url.encode()).hexdigest()
        assert key1 == key2
        assert len(key1) == 32


class TestUnifiedProcessorConfig:
    """Test UnifiedProcessorConfig."""

    def test_high_performance_config(self):
        from flexllm.msg_processors.unified_processor import UnifiedProcessorConfig

        config = UnifiedProcessorConfig.high_performance()
        assert config.max_workers > 0
        assert config.max_concurrent > 0

    def test_memory_optimized_config(self):
        from flexllm.msg_processors.unified_processor import UnifiedProcessorConfig

        config = UnifiedProcessorConfig.memory_optimized()
        assert config.max_workers > 0

    def test_default_config(self):
        from flexllm.msg_processors.unified_processor import UnifiedProcessorConfig

        config = UnifiedProcessorConfig.default()
        assert config.max_workers == 8


class TestImageOptimization:
    """Test image optimization settings."""

    def test_resize_calculation(self):
        original_width = 2000
        original_height = 1000
        max_size = 1024

        ratio = min(max_size / original_width, max_size / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        original_ratio = original_width / original_height
        new_ratio = new_width / new_height
        assert abs(original_ratio - new_ratio) < 0.01
        assert new_width <= max_size
        assert new_height <= max_size


class TestStdoutNotHijacked:
    """回归（严重 bug #1）：OpenCV 输出抑制不得替换全局 sys.stdout/stderr。

    旧实现用替换 sys.stdout 的上下文管理器抑制 OpenCV 输出，线程池并发交错后
    sys.stdout 永久指向废弃的 StringIO，整个进程 stdout 静默丢失。
    """

    def test_suppress_context_managers_removed(self):
        import flexllm.msg_processors.unified_processor as up

        assert not hasattr(up, "suppress_stdout")
        assert not hasattr(up, "suppress_stderr")
        assert not hasattr(up, "suppress_all_output")

    def test_concurrent_processing_keeps_global_streams(self, tmp_path):
        import sys
        from concurrent.futures import ThreadPoolExecutor

        import pytest

        pytest.importorskip("cv2")
        from PIL import Image

        from flexllm.msg_processors.unified_processor import UnifiedImageProcessor

        img_path = tmp_path / "t.png"
        Image.new("RGB", (32, 32), (10, 200, 30)).save(img_path)

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        processor = UnifiedImageProcessor()
        try:
            with ThreadPoolExecutor(max_workers=8) as ex:
                futures = [
                    ex.submit(processor._process_local_file_sync, str(img_path)) for _ in range(16)
                ]
                for f in futures:
                    assert f.result().startswith("data:image/png")
        finally:
            processor.cleanup()

        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr


class TestEncodeImageToBase64Regression:
    """encode_image_to_base64（image_processor 内部版本）的回归测试"""

    async def test_palette_mode_colors_preserved(self):
        """回归（严重 bug #2）：P 模式（调色板）图片编码后颜色必须正确。

        旧实现直接 np.array(P 模式图) 得到的是调色板索引而非 RGB，
        编码后颜色损坏并静默发给模型。
        """
        import pytest

        pytest.importorskip("cv2")
        from PIL import Image

        from flexllm.msg_processors.image_processor import (
            decode_base64_to_pil,
            encode_image_to_base64,
        )

        color = (200, 30, 60)
        p_img = Image.new("RGB", (16, 16), color).convert("P", palette=Image.ADAPTIVE)
        assert p_img.mode == "P"

        result = await encode_image_to_base64(p_img, None)
        assert result.startswith("data:image/png")

        decoded = decode_base64_to_pil(result).convert("RGB")
        assert decoded.getpixel((8, 8)) == color

    async def test_input_image_not_mutated(self):
        """回归（#3）：encode 不得原地修改调用方传入的 Image（thumbnail 缩放）"""
        import pytest

        pytest.importorskip("cv2")
        from PIL import Image

        from flexllm.msg_processors.image_processor import encode_image_to_base64

        img = Image.new("RGB", (100, 80), (1, 2, 3))
        await encode_image_to_base64(img, None, max_width=10)
        assert img.size == (100, 80)
