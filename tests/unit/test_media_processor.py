"""Tests for media (video/audio) preprocessing support."""

import base64
import copy
import errno
import os

import pytest

from flexllm.msg_processors.image_processor import encode_media_to_base64
from flexllm.msg_processors.unified_processor import (
    _is_source_needs_conversion,
    process_content_recursive,
)

ENOENT = os.strerror(errno.ENOENT)


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

    def test_raw_base64_jpeg_not_treated_as_path(self):
        """回归：JPEG raw base64 以 /9j/ 开头，不能被误判为文件路径"""
        value = "/9j/" + "A" * 600
        assert _is_source_needs_conversion(value) is False

    def test_raw_base64_mp3_not_treated_as_path(self):
        """回归：MP3 raw base64 以 //uQ 开头，不能被误判为文件路径"""
        value = "//uQ" + "b" * 600 + "=="
        assert _is_source_needs_conversion(value) is False

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
    """Anthropic 协议没有音视频内容块：一律换成占位文本（不再转成必被拒的 document）"""

    @pytest.mark.parametrize(
        "part,placeholder",
        [
            ({"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}, "video"),
            ({"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,BBBB"}}, "audio"),
            ({"type": "input_audio", "input_audio": {"data": "CCCC", "format": "mp3"}}, "audio"),
        ],
    )
    def test_media_replaced_with_placeholder(self, part, placeholder):
        from flexllm import ClaudeClient
        from flexllm.clients.message_images import AUDIO_OMITTED_TEXT, VIDEO_OMITTED_TEXT

        text = {"video": VIDEO_OMITTED_TEXT, "audio": AUDIO_OMITTED_TEXT}[placeholder]
        messages = [{"role": "user", "content": [part]}]
        # video 开关取值不影响：协议本身不收
        for video in (True, False):
            body = ClaudeClient(api_key="k", video=video)._build_request_body(messages, "m")
            assert body["messages"][0]["content"] == [{"type": "text", "text": text}]

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


class TestNormalizeAudioFormat:
    """MIME 子类型 -> OpenAI input_audio.format 的规范化。"""

    @pytest.mark.parametrize(
        "subtype,expected",
        [
            ("x-wav", "wav"),
            ("wave", "wav"),
            ("vnd.wave", "wav"),
            ("X-WAV", "wav"),
            ("mpeg", "mp3"),
            ("x-m4a", "m4a"),
            ("wav", "wav"),
            ("mp3", "mp3"),
            ("flac", "flac"),
        ],
    )
    def test_normalize(self, subtype, expected):
        from flexllm.msg_processors.audio_processor import normalize_audio_format

        assert normalize_audio_format(subtype) == expected

    def test_unknown_subtype_passthrough(self):
        from flexllm.msg_processors.audio_processor import normalize_audio_format

        assert normalize_audio_format("weird-codec") == "weird-codec"


class TestOpenAIClientAudioConversion:
    """OpenAI 客户端 audio_url -> input_audio 转换。"""

    @staticmethod
    def _convert(url):
        from flexllm.clients.openai import OpenAIClient

        messages = [{"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": url}}]}]
        return OpenAIClient._convert_audio_url_to_input_audio(messages)[0]["content"][0]

    def test_x_wav_mime_normalized_to_wav(self):
        """mimetypes 在 Linux 上把 .wav 猜成 audio/x-wav，透传会被服务端以
        非法 format 拒绝（GLM 返回 error code 1214），必须规范化为 wav。"""
        part = self._convert("data:audio/x-wav;base64,AAAA")
        assert part["type"] == "input_audio"
        assert part["input_audio"] == {"data": "AAAA", "format": "wav"}

    def test_mpeg_mime_normalized_to_mp3(self):
        part = self._convert("data:audio/mpeg;base64,BBBB")
        assert part["input_audio"] == {"data": "BBBB", "format": "mp3"}

    def test_standard_wav_unchanged(self):
        part = self._convert("data:audio/wav;base64,CCCC")
        assert part["input_audio"] == {"data": "CCCC", "format": "wav"}

    def test_non_data_uri_kept_as_is(self):
        """非 data URI 无法拆出 base64，保持原样交给后端"""
        part = self._convert("https://example.com/a.wav")
        assert part["type"] == "audio_url"

    def test_non_list_content_untouched(self):
        from flexllm.clients.openai import OpenAIClient

        messages = [{"role": "user", "content": "纯文本"}]
        assert OpenAIClient._convert_audio_url_to_input_audio(messages) == messages


PH = "placeholder"


@pytest.fixture(scope="module")
def test_image(tmp_path_factory) -> str:
    """现场生成的真实 PNG（单测不能依赖 tests/e2e 下被 gitignore 的生成素材）"""
    import cv2
    import numpy as np

    path = tmp_path_factory.mktemp("media") / "test_image.png"
    img = np.zeros((32, 48, 3), np.uint8)
    img[:, :24] = (255, 0, 0)
    assert cv2.imwrite(str(path), img)
    return str(path)


def _missing_media_messages(missing) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"{missing}/a.png"}},
                {"type": "image_url", "image_url": {"url": f"file://{missing}/b.png"}},
                {"type": "audio_url", "audio_url": {"url": f"{missing}/c.wav"}},
                {"type": "video_url", "video_url": {"url": f"{missing}/d.mp4"}},
                {"type": "input_audio", "input_audio": {"data": f"{missing}/e.wav"}},
            ],
        }
    ]


class TestLocalMediaUnavailable:
    """本地媒体读不到：默认原样透传（后端可能读得到），placeholder 模式整块换成文字占位"""

    async def test_default_passes_local_paths_through(self, tmp_path):
        from flexllm.msg_processors.unified_processor import unified_messages_preprocess

        messages = _missing_media_messages(tmp_path / "gone")
        out = await unified_messages_preprocess(messages)
        assert out == messages

    async def test_placeholder_mode(self, tmp_path):
        from flexllm.msg_processors.unified_processor import unified_messages_preprocess

        missing = tmp_path / "gone"
        messages = _missing_media_messages(missing)
        snapshot = copy.deepcopy(messages)
        out = await unified_messages_preprocess(messages, missing_local_media=PH)

        assert messages == snapshot
        assert out[0]["content"] == [
            {"type": "text", "text": f"[{kind} unavailable: {missing}/{name}: {ENOENT}]"}
            for kind, name in [
                ("image", "a.png"),
                ("image", "b.png"),
                ("audio", "c.wav"),
                ("video", "d.mp4"),
                ("audio", "e.wav"),
            ]
        ]

    async def test_batch_preprocess_forwards_mode(self, tmp_path):
        from flexllm.msg_processors.unified_processor import unified_batch_messages_preprocess

        missing = tmp_path / "gone"
        out = await unified_batch_messages_preprocess(
            [_missing_media_messages(missing)], missing_local_media=PH
        )
        assert all(p["type"] == "text" for p in out[0][0]["content"])

    async def test_relative_path_reported_as_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        content = {"type": "video_url", "video_url": {"url": "d.mp4"}}
        await process_content_recursive(content, missing_local_media=PH)
        assert content == {
            "type": "text",
            "text": f"[video unavailable: {tmp_path}/d.mp4: {ENOENT}]",
        }

    async def test_undecodable_local_image(self, tmp_path):
        f = tmp_path / "broken.png"
        f.write_bytes(b"not an image")
        content = {"type": "image_url", "image_url": {"url": str(f)}}
        await process_content_recursive(content)
        assert content == {"type": "image_url", "image_url": {"url": str(f)}}
        await process_content_recursive(content, missing_local_media=PH)
        assert content == {"type": "text", "text": f"[image unavailable: {f}: cannot decode image]"}

    async def test_undecodable_local_audio_reason_is_stable(self, tmp_path):
        """非 OSError 的异常文本可能带内存地址，占位原因必须固定"""
        f = tmp_path / "bad.wav"
        f.write_bytes(b"not audio")
        content = {"type": "audio_url", "audio_url": {"url": str(f)}}
        await process_content_recursive(content, missing_local_media=PH, target_sample_rate=16000)
        assert content == {"type": "text", "text": f"[audio unavailable: {f}: cannot decode audio]"}

    @pytest.mark.parametrize("prefix", ["", "file://"])
    async def test_video_frames_path_missing_file(self, tmp_path, prefix):
        path = tmp_path / "d.mp4"
        part = {"type": "video_url", "video_url": {"url": f"{prefix}{path}"}}
        content = [dict(part)]
        await process_content_recursive(content, video_fps=1.0)
        assert content == [part]
        await process_content_recursive(content, missing_local_media=PH, video_fps=1.0)
        assert content == [{"type": "text", "text": f"[video unavailable: {path}: {ENOENT}]"}]

    @pytest.mark.parametrize("extra", [{"video_fps": 1.0}, {"target_sample_rate": 16000}])
    async def test_good_image_encodes_with_video_or_audio_kwargs(self, extra, test_image):
        """回归：视频/音频参数不能传进图片处理器（曾因 TypeError 让好图片编码失败）"""
        content = [{"type": "image_url", "image_url": {"url": test_image}}]
        await process_content_recursive(content, missing_local_media=PH, **extra)
        assert content[0]["image_url"]["url"].startswith("data:image/")

    async def test_unreachable_http_url_is_kept(self):
        url = "http://127.0.0.1:1/x.mp4"
        content = {"type": "video_url", "video_url": {"url": url}}
        await process_content_recursive(content, missing_local_media=PH)
        assert content == {"type": "video_url", "video_url": {"url": url}}

    @pytest.mark.parametrize("mode,expected_type", [(None, "video_url"), (PH, "text")])
    async def test_client_option(self, tmp_path, mode, expected_type):
        """流式与非流式共用 _preprocess_messages，模式由客户端构造参数决定"""
        from flexllm import OpenAIClient

        kw = {} if mode is None else {"missing_local_media": mode}
        client = OpenAIClient(base_url="http://x/v1", model="m", **kw)
        messages = [
            {
                "role": "user",
                "content": [{"type": "video_url", "video_url": {"url": str(tmp_path / "d.mp4")}}],
            }
        ]
        out = await client._preprocess_messages(messages, preprocess_msg=True)
        assert out[0]["content"][0]["type"] == expected_type
        batch = await client._preprocess_messages_batch([messages], preprocess_msg=True)
        assert batch[0][0]["content"][0]["type"] == expected_type

    def test_invalid_mode_rejected(self):
        from flexllm import LLMClient

        with pytest.raises(ValueError, match="missing_local_media"):
            LLMClient(base_url="http://x/v1", model="m", missing_local_media="drop")

    def test_config_entry(self, tmp_path):
        from flexllm import LLMClient
        from flexllm.cli.config import FlexLLMConfig

        path = tmp_path / "c.yaml"
        path.write_text(
            "default: m\nmodels:\n  - id: m\n    base_url: http://x/v1\n"
            "    missing_local_media: placeholder\n    temperature: 0.1\n"
        )
        assert FlexLLMConfig(path).get_model_params("m") == {"temperature": 0.1}
        client = LLMClient.from_config(str(path))
        assert client.client._missing_local_media == PH


class TestFileUriDecoding:
    """file:// 按标准百分号解码：Path.as_uri() 生成的带空格/中文路径要能读到"""

    @pytest.fixture
    def media(self, tmp_path):
        d = tmp_path / "目录 a"
        d.mkdir()
        f = d / "视频 1.mp4"
        f.write_bytes(b"fake video")
        return f

    def test_round_trips_path_as_uri(self, media):
        from flexllm.msg_processors.image_processor import file_uri_to_path

        assert media.as_uri() != f"file://{media}"  # 确实被编码过
        assert file_uri_to_path(media.as_uri()) == str(media)
        assert file_uri_to_path(f"file://localhost{media}") == str(media)

    def test_legacy_forms_still_work(self, tmp_path, monkeypatch):
        """直接拼原始路径的旧写法：原样空格、相对路径、文件名本身含 %xx"""
        from flexllm.msg_processors.image_processor import file_uri_to_path

        literal = tmp_path / "50%41off.png"
        literal.write_bytes(b"x")
        assert file_uri_to_path(f"file://{literal}") == str(literal)
        assert file_uri_to_path("file:///tmp/a b.png") == "/tmp/a b.png"
        monkeypatch.chdir(tmp_path)
        assert file_uri_to_path("file://imgs/a.png") == "imgs/a.png"

    async def test_encoded_uri_is_read(self, media):
        content = {"type": "video_url", "video_url": {"url": media.as_uri()}}
        await process_content_recursive(content, missing_local_media="placeholder")
        assert content["video_url"]["url"].startswith("data:video/mp4;base64,")

    async def test_encoded_uri_frames_and_placeholder_use_decoded_path(self, media):
        missing = media.with_name("没有 了.mp4")
        content = [{"type": "video_url", "video_url": {"url": missing.as_uri()}}]
        await process_content_recursive(content, missing_local_media="placeholder", video_fps=1.0)
        assert content == [{"type": "text", "text": f"[video unavailable: {missing}: {ENOENT}]"}]

    async def test_encoded_image_and_input_audio(self, tmp_path, test_image):
        img = tmp_path / "图 片 .png"
        img.write_bytes(open(test_image, "rb").read())
        wav = tmp_path / "声 音.wav"
        wav.write_bytes(b"fake wav")
        parts = [
            {"type": "image_url", "image_url": {"url": img.as_uri()}},
            {"type": "input_audio", "input_audio": {"data": wav.as_uri(), "format": "wav"}},
        ]
        await process_content_recursive(parts, missing_local_media="placeholder")
        assert parts[0]["image_url"]["url"].startswith("data:image/")
        assert base64.b64decode(parts[1]["input_audio"]["data"]) == b"fake wav"


class TestSkipUnsupportedParts:
    """模型不接收的块在预处理阶段就跳过：不读文件、不编码，发送前再替换为占位"""

    @pytest.fixture
    def files(self, tmp_path, test_image):
        paths = {}
        for name in ("v.mp4", "a.wav"):
            paths[name] = tmp_path / name
            paths[name].write_bytes(b"fake")
        paths["i.png"] = test_image
        return paths

    def _messages(self, files):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": str(files["v.mp4"])}},
                    {"type": "audio_url", "audio_url": {"url": str(files["a.wav"])}},
                    {"type": "image_url", "image_url": {"url": str(files["i.png"])}},
                ],
            }
        ]

    @staticmethod
    def _kinds(messages):
        return [p[p["type"]]["url"][:5] for p in messages[0]["content"]]

    @pytest.mark.parametrize(
        "make,expected",
        [
            (lambda: _openai_client(video=False), ["/", "data:", "data:"]),
            (lambda: _openai_client(vision=False), ["data:", "data:", "/"]),
            (lambda: _claude_client(), ["/", "/", "data:"]),
            (lambda: _openai_client(), ["data:", "data:", "data:"]),
        ],
    )
    async def test_unsupported_parts_are_not_encoded(self, files, make, expected):
        client = make()
        out = await client._preprocess_messages(self._messages(files), preprocess_msg=True)
        assert [k if k == "data:" else "/" for k in self._kinds(out)] == expected
        batch = await client._preprocess_messages_batch([self._messages(files)], True)
        assert batch[0] == out

    async def test_skipped_video_is_not_split_into_frames(self, files):
        content = [{"type": "video_url", "video_url": {"url": str(files["v.mp4"])}}]
        await process_content_recursive(
            content, skip_part_types=frozenset({"video_url"}), video_fps=1.0
        )
        assert content == [{"type": "video_url", "video_url": {"url": str(files["v.mp4"])}}]

    async def test_body_still_carries_placeholder(self, files):
        from flexllm.clients.message_images import VIDEO_OMITTED_TEXT

        client = _openai_client(video=False)
        out = await client._preprocess_messages(self._messages(files), preprocess_msg=True)
        body = client._build_request_body(out, "m")
        assert body["messages"][0]["content"][0] == {"type": "text", "text": VIDEO_OMITTED_TEXT}


def _openai_client(**kw):
    from flexllm import OpenAIClient

    return OpenAIClient(base_url="http://x/v1", model="m", **kw)


def _claude_client(**kw):
    from flexllm import ClaudeClient

    return ClaudeClient(api_key="k", model="claude-sonnet-5", **kw)


@pytest.mark.parametrize(
    "t,kind", [("video_url", "video"), ("image_url", "image"), ("input_audio", "audio")]
)
async def test_skipped_missing_file_keeps_unavailable_placeholder(tmp_path, t, kind):
    """跳过编码不能丢掉"文件缺失"的信息：placeholder 模式占位仍写明路径与原因"""
    missing = str(tmp_path / "gone")
    key = "data" if t == "input_audio" else "url"
    content = {"type": t, t: {key: missing}}
    await process_content_recursive(
        content, missing_local_media="placeholder", skip_part_types=frozenset({t})
    )
    assert content == {"type": "text", "text": f"[{kind} unavailable: {missing}: {ENOENT}]"}
    # passthrough 下原样保留，由发送前的 omit 统一替换
    content = {"type": t, t: {key: missing}}
    await process_content_recursive(content, skip_part_types=frozenset({t}))
    assert content == {"type": t, t: {key: missing}}
