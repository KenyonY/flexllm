"""Tests for media (video/audio) preprocessing support."""

import base64

import pytest

from flexllm.msg_processors.image_processor import encode_media_to_base64
from flexllm.msg_processors.unified_processor import (
    _is_source_needs_conversion,
    process_content_recursive,
)


class TestIsSourceNeedsConversion:
    """Test _is_source_needs_conversion helper."""

    def test_absolute_path(self):
        assert _is_source_needs_conversion("/tmp/test.wav") is True

    def test_http_url(self):
        assert _is_source_needs_conversion("http://example.com/video.mp4") is True

    def test_https_url(self):
        assert _is_source_needs_conversion("https://example.com/audio.wav") is True

    def test_file_uri(self):
        assert _is_source_needs_conversion("file:///tmp/test.mp3") is True

    def test_base64_string(self):
        assert _is_source_needs_conversion("SGVsbG8gV29ybGQ=") is False

    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.wav"
        f.write_bytes(b"RIFF" + b"\x00" * 100)
        assert _is_source_needs_conversion(str(f)) is True


class TestEncodeMediaToBase64:
    """Test encode_media_to_base64 function."""

    async def test_local_file(self, tmp_path):
        """本地文件编码"""
        f = tmp_path / "test.mp4"
        content = b"fake video data"
        f.write_bytes(content)

        result = await encode_media_to_base64(str(f), return_with_mime=True)
        assert result.startswith("data:")
        assert ";base64," in result
        # 解码验证
        b64_part = result.split(";base64,", 1)[1]
        assert base64.b64decode(b64_part) == content

    async def test_local_file_no_mime(self, tmp_path):
        """本地文件编码，不带 MIME 前缀"""
        f = tmp_path / "test.wav"
        content = b"fake audio data"
        f.write_bytes(content)

        result = await encode_media_to_base64(str(f), return_with_mime=False)
        assert not result.startswith("data:")
        assert base64.b64decode(result) == content

    async def test_data_uri_passthrough(self):
        """data: URI 直接返回"""
        uri = "data:video/mp4;base64,AAAA"
        result = await encode_media_to_base64(uri, return_with_mime=True)
        assert result == uri

    async def test_data_uri_strip_mime(self):
        """data: URI 去除 MIME 前缀"""
        uri = "data:audio/wav;base64,AAAA"
        result = await encode_media_to_base64(uri, return_with_mime=False)
        assert result == "AAAA"

    async def test_file_uri(self, tmp_path):
        """file:// URI"""
        f = tmp_path / "test.ogg"
        content = b"fake ogg data"
        f.write_bytes(content)

        result = await encode_media_to_base64(f"file://{f}", return_with_mime=False)
        assert base64.b64decode(result) == content

    async def test_unsupported_source(self):
        """不支持的来源"""
        with pytest.raises(ValueError, match="Unsupported media source"):
            await encode_media_to_base64("not_a_file_or_url")


class TestProcessContentRecursiveVideoUrl:
    """Test process_content_recursive with video_url type."""

    async def test_video_url_local_file(self, tmp_path):
        """video_url 本地文件转换"""
        f = tmp_path / "test.mp4"
        f.write_bytes(b"fake video")

        import aiohttp

        content = {
            "type": "video_url",
            "video_url": {"url": str(f)},
        }
        async with aiohttp.ClientSession() as session:
            await process_content_recursive(content, session)

        url = content["video_url"]["url"]
        assert url.startswith("data:")
        assert ";base64," in url

    async def test_video_url_data_uri_skip(self):
        """video_url 已经是 data: URI 则跳过"""
        import aiohttp

        content = {
            "type": "video_url",
            "video_url": {"url": "data:video/mp4;base64,AAAA"},
        }
        async with aiohttp.ClientSession() as session:
            await process_content_recursive(content, session)

        assert content["video_url"]["url"] == "data:video/mp4;base64,AAAA"


class TestProcessContentRecursiveAudioUrl:
    """Test process_content_recursive with audio_url type."""

    async def test_audio_url_local_file(self, tmp_path):
        """audio_url 本地文件转换"""
        f = tmp_path / "test.wav"
        f.write_bytes(b"fake audio")

        import aiohttp

        content = {
            "type": "audio_url",
            "audio_url": {"url": str(f)},
        }
        async with aiohttp.ClientSession() as session:
            await process_content_recursive(content, session)

        url = content["audio_url"]["url"]
        assert url.startswith("data:")
        assert ";base64," in url


class TestProcessContentRecursiveInputAudio:
    """Test process_content_recursive with input_audio type."""

    async def test_input_audio_local_file(self, tmp_path):
        """input_audio 本地文件路径转换为纯 base64"""
        f = tmp_path / "test.wav"
        content_bytes = b"fake wav audio"
        f.write_bytes(content_bytes)

        import aiohttp

        content = {
            "type": "input_audio",
            "input_audio": {"data": str(f), "format": "wav"},
        }
        async with aiohttp.ClientSession() as session:
            await process_content_recursive(content, session)

        data = content["input_audio"]["data"]
        assert not data.startswith("data:")
        assert base64.b64decode(data) == content_bytes

    async def test_input_audio_already_base64_skip(self):
        """input_audio data 已经是 base64 则跳过"""
        import aiohttp

        content = {
            "type": "input_audio",
            "input_audio": {"data": "SGVsbG8gV29ybGQ=", "format": "wav"},
        }
        async with aiohttp.ClientSession() as session:
            await process_content_recursive(content, session)

        assert content["input_audio"]["data"] == "SGVsbG8gV29ybGQ="


class TestProcessContentRecursiveImageUrlRegression:
    """Regression test: image_url still works correctly."""

    async def test_image_url_data_uri_skip(self):
        """image_url 已经是 data: URI 则跳过"""
        import aiohttp

        content = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBOR"},
        }
        async with aiohttp.ClientSession() as session:
            await process_content_recursive(content, session)

        assert content["image_url"]["url"] == "data:image/png;base64,iVBOR"

    async def test_non_media_type_recursive(self):
        """非媒体 type 仍然递归处理子节点"""
        import aiohttp

        content = {
            "type": "text",
            "text": "hello",
        }
        async with aiohttp.ClientSession() as session:
            await process_content_recursive(content, session)

        assert content["text"] == "hello"


class TestClaudeClientMediaConversion:
    """Test Claude client format conversion for video/audio."""

    def test_convert_video_url_base64(self):
        from flexllm import ClaudeClient

        client = ClaudeClient(api_key="test-key")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": "data:video/mp4;base64,AAAA"},
                    }
                ],
            }
        ]
        body = client._build_request_body(messages, "claude-3-5-sonnet-20241022")
        msg_content = body["messages"][0]["content"]
        assert len(msg_content) == 1
        assert msg_content[0]["type"] == "document"
        assert msg_content[0]["source"]["type"] == "base64"
        assert msg_content[0]["source"]["media_type"] == "video/mp4"
        assert msg_content[0]["source"]["data"] == "AAAA"

    def test_convert_audio_url_base64(self):
        from flexllm import ClaudeClient

        client = ClaudeClient(api_key="test-key")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": "data:audio/wav;base64,BBBB"},
                    }
                ],
            }
        ]
        body = client._build_request_body(messages, "claude-3-5-sonnet-20241022")
        msg_content = body["messages"][0]["content"]
        assert len(msg_content) == 1
        assert msg_content[0]["type"] == "document"
        assert msg_content[0]["source"]["media_type"] == "audio/wav"
        assert msg_content[0]["source"]["data"] == "BBBB"

    def test_convert_input_audio(self):
        from flexllm import ClaudeClient

        client = ClaudeClient(api_key="test-key")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "CCCC", "format": "mp3"},
                    }
                ],
            }
        ]
        body = client._build_request_body(messages, "claude-3-5-sonnet-20241022")
        msg_content = body["messages"][0]["content"]
        assert len(msg_content) == 1
        assert msg_content[0]["type"] == "document"
        assert msg_content[0]["source"]["media_type"] == "audio/mp3"
        assert msg_content[0]["source"]["data"] == "CCCC"

    def test_image_url_still_works(self):
        """回归测试：image_url 仍然正确"""
        from flexllm import ClaudeClient

        client = ClaudeClient(api_key="test-key")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBOR"},
                    }
                ],
            }
        ]
        body = client._build_request_body(messages, "claude-3-5-sonnet-20241022")
        msg_content = body["messages"][0]["content"]
        assert len(msg_content) == 1
        assert msg_content[0]["type"] == "image"
        assert msg_content[0]["source"]["media_type"] == "image/png"


class TestGeminiClientMediaConversion:
    """Test Gemini client format conversion for video/audio."""

    def _get_client(self):
        from flexllm.clients.gemini import GeminiClient

        return GeminiClient(api_key="test-key", model="gemini-2.0-flash")

    def test_convert_video_url_base64(self):
        client = self._get_client()
        content = [
            {
                "type": "video_url",
                "video_url": {"url": "data:video/mp4;base64,AAAA"},
            }
        ]
        parts = client._convert_content_to_parts(content)
        assert len(parts) == 1
        assert parts[0]["inline_data"]["mime_type"] == "video/mp4"
        assert parts[0]["inline_data"]["data"] == "AAAA"

    def test_convert_audio_url_base64(self):
        client = self._get_client()
        content = [
            {
                "type": "audio_url",
                "audio_url": {"url": "data:audio/wav;base64,BBBB"},
            }
        ]
        parts = client._convert_content_to_parts(content)
        assert len(parts) == 1
        assert parts[0]["inline_data"]["mime_type"] == "audio/wav"
        assert parts[0]["inline_data"]["data"] == "BBBB"

    def test_convert_input_audio(self):
        client = self._get_client()
        content = [
            {
                "type": "input_audio",
                "input_audio": {"data": "CCCC", "format": "mp3"},
            }
        ]
        parts = client._convert_content_to_parts(content)
        assert len(parts) == 1
        assert parts[0]["inline_data"]["mime_type"] == "audio/mp3"
        assert parts[0]["inline_data"]["data"] == "CCCC"

    def test_image_url_still_works(self):
        """回归测试：image_url 仍然正确"""
        client = self._get_client()
        content = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/"},
            }
        ]
        parts = client._convert_content_to_parts(content)
        assert len(parts) == 1
        assert parts[0]["inline_data"]["mime_type"] == "image/jpeg"
        assert parts[0]["inline_data"]["data"] == "/9j/"
