"""Tests for audio endpoints: /audio/transcriptions and /audio/speech."""

import pytest

from flexllm.clients.audio import (
    TranscriptionResult,
    encode_multipart,
    parse_transcription,
    read_audio_source,
    segments_to_srt,
    segments_to_vtt,
)
from flexllm.clients.openai import OpenAIClient


def _client(**kwargs):
    return OpenAIClient(base_url="https://api.example.com/v1", api_key="k", **kwargs)


def _parse_multipart(body: bytes, content_type: str) -> dict:
    """把 multipart 体解析回 {字段名: [值...]}，文件字段值为 (filename, bytes)"""
    boundary = content_type.split("boundary=")[1]
    fields: dict[str, list] = {}
    for chunk in body.split(f"--{boundary}".encode()):
        if not chunk.strip() or chunk.strip() == b"--":
            continue
        head, _, payload = chunk.partition(b"\r\n\r\n")
        head_text = head.decode("utf-8", "replace")
        if 'name="' not in head_text:
            continue
        name = head_text.split('name="')[1].split('"')[0]
        payload = payload[:-2] if payload.endswith(b"\r\n") else payload
        if "filename=" in head_text:
            filename = head_text.split('filename="')[1].split('"')[0]
            fields.setdefault(name, []).append((filename, payload))
        else:
            fields.setdefault(name, []).append(payload.decode("utf-8"))
    return fields


class TestEncodeMultipart:
    def test_fields_and_file_roundtrip(self):
        body, content_type = encode_multipart(
            {"model": "glm-asr", "language": "zh"}, "a.wav", b"RIFFDATA"
        )
        assert content_type.startswith("multipart/form-data; boundary=")
        parsed = _parse_multipart(body, content_type)
        assert parsed["model"] == ["glm-asr"]
        assert parsed["language"] == ["zh"]
        assert parsed["file"] == [("a.wav", b"RIFFDATA")]

    def test_none_fields_skipped(self):
        body, ct = encode_multipart({"model": "m", "language": None}, "a.wav", b"x")
        assert "language" not in _parse_multipart(body, ct)

    def test_list_field_expands_to_repeated_parts(self):
        """timestamp_granularities[] 这类重复字段要展开成多个 part"""
        body, ct = encode_multipart(
            {"model": "m", "timestamp_granularities[]": ["segment", "word"]}, "a.wav", b"x"
        )
        assert _parse_multipart(body, ct)["timestamp_granularities[]"] == ["segment", "word"]

    def test_bool_serialized_lowercase(self):
        """stream=False 必须编码成 "false"，Python 的 "False" 服务端不认"""
        body, ct = encode_multipart({"stream": False}, "a.wav", b"x")
        assert _parse_multipart(body, ct)["stream"] == ["false"]

    def test_body_is_bytes_so_retry_can_resend(self):
        """重试要重发同一 body：bytes 可重发，aiohttp.FormData 不行"""
        body, _ = encode_multipart({"model": "m"}, "a.wav", b"x")
        assert isinstance(body, bytes)

    def test_filename_quotes_sanitized(self):
        """文件名里的引号/换行不能破坏 Content-Disposition 头

        未转义时 CRLF 会让攻击者在 header 段里插入额外的 header 行。
        """
        body, ct = encode_multipart({}, 'e"vil\r\nX: y.wav', b"x")
        header_block = body.split(b"\r\n\r\n")[0]
        # header 段只应有 boundary、Content-Disposition、Content-Type 三行
        assert len(header_block.split(b"\r\n")) == 3
        assert _parse_multipart(body, ct)["file"][0][0] == "e_vil__X: y.wav"

    def test_boundary_unique_per_call(self):
        _, ct1 = encode_multipart({}, "a.wav", b"x")
        _, ct2 = encode_multipart({}, "a.wav", b"x")
        assert ct1 != ct2


class TestReadAudioSource:
    def test_path(self, tmp_path):
        f = tmp_path / "a.wav"
        f.write_bytes(b"data")
        assert read_audio_source(str(f)) == (b"data", "a.wav")

    def test_bytes_requires_filename(self):
        with pytest.raises(ValueError, match="filename"):
            read_audio_source(b"data")

    def test_bytes_with_filename(self):
        assert read_audio_source(b"data", "x.mp3") == (b"data", "x.mp3")

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_audio_source(str(tmp_path / "nope.wav"))


class TestSubtitleRendering:
    SEGMENTS = [
        {"start": 2.0, "end": 6.1, "text": "第一句"},
        {"start": 6.1, "end": 3661.5, "text": " 第二句 "},
    ]

    def test_srt(self):
        srt = segments_to_srt(self.SEGMENTS)
        assert "1\n00:00:02,000 --> 00:00:06,100\n第一句" in srt
        assert "2\n00:00:06,100 --> 01:01:01,500\n第二句" in srt

    def test_vtt(self):
        vtt = segments_to_vtt(self.SEGMENTS)
        assert vtt.startswith("WEBVTT")
        assert "00:00:02.000 --> 00:00:06.100" in vtt

    def test_empty_segments(self):
        assert segments_to_srt([]) == ""
        assert segments_to_vtt([]).strip() == "WEBVTT"


class TestParseTranscription:
    def test_full_response(self):
        result = parse_transcription(
            {
                "text": "你好",
                "language": "zh",
                "duration": 1.5,
                "segments": [{"start": 0.0, "end": 1.5, "text": "你好"}],
            }
        )
        assert isinstance(result, TranscriptionResult)
        assert (result.text, result.language, result.duration) == ("你好", "zh", 1.5)
        assert len(result.segments) == 1

    def test_missing_segments_falls_back_to_single(self):
        """有的模型（如 glm-asr-2512）不返回 segments，字幕渲染仍需可用"""
        result = parse_transcription({"text": "你好", "duration": 2.0})
        assert result.segments == [{"id": 0, "start": 0.0, "end": 2.0, "text": "你好"}]
        assert "你好" in result.to_srt()

    def test_empty_response(self):
        result = parse_transcription({})
        assert result.text == ""
        assert result.segments == []


class TestTranscriptionRequestBuilding:
    def test_multipart_headers_drop_json_content_type(self):
        """必须换掉 application/json，否则 boundary 丢失、服务端解析不出字段"""
        headers = _client()._get_multipart_headers("multipart/form-data; boundary=abc")
        content_types = [v for k, v in headers.items() if k.lower() == "content-type"]
        assert content_types == ["multipart/form-data; boundary=abc"]
        assert "Authorization" in headers

    def test_params_carry_body_and_headers(self, tmp_path):
        f = tmp_path / "a.wav"
        f.write_bytes(b"RIFF")
        params = _client()._build_transcription_params(
            str(f), "glm-asr", None, "zh", None, None, None, None, {"stream": False}
        )
        assert set(params) == {"data", "headers"}
        parsed = _parse_multipart(params["data"], params["headers"]["Content-Type"])
        assert parsed["model"] == ["glm-asr"]
        assert parsed["language"] == ["zh"]
        assert parsed["stream"] == ["false"]

    def test_model_falls_back_to_client_default(self, tmp_path):
        f = tmp_path / "a.wav"
        f.write_bytes(b"RIFF")
        params = _client(model="glm-asr")._build_transcription_params(
            str(f), None, None, None, None, None, None, None, {}
        )
        parsed = _parse_multipart(params["data"], params["headers"]["Content-Type"])
        assert parsed["model"] == ["glm-asr"]

    def test_model_required(self, tmp_path):
        f = tmp_path / "a.wav"
        f.write_bytes(b"RIFF")
        with pytest.raises(ValueError, match="转录模型"):
            _client()._build_transcription_params(
                str(f), None, None, None, None, None, None, None, {}
            )

    def test_subtitle_response_format_rejected(self, tmp_path):
        """srt/vtt 不透传服务端：返回纯文本会被判为非 JSON 错误，本地渲染代替"""
        f = tmp_path / "a.wav"
        f.write_bytes(b"RIFF")
        with pytest.raises(ValueError, match="response_format"):
            _client()._build_transcription_params(
                str(f), "glm-asr", None, None, None, "srt", None, None, {}
            )

    def test_url(self):
        assert (
            _client()._get_transcription_url() == "https://api.example.com/v1/audio/transcriptions"
        )


class TestSpeechRequestBuilding:
    def test_body_and_bytes_response_type(self):
        params = _client()._build_speech_params("你好", "glm-tts", "tongtong", "wav", None, {})
        assert params["json"] == {
            "model": "glm-tts",
            "input": "你好",
            "voice": "tongtong",
            "response_format": "wav",
        }
        # 响应是音频字节，按 JSON 解析会被判为错误
        assert params["response_type"] == "bytes"

    def test_none_fields_dropped(self):
        params = _client()._build_speech_params("你好", "glm-tts", None, None, None, {})
        assert params["json"] == {"model": "glm-tts", "input": "你好"}

    def test_extra_passthrough(self):
        params = _client()._build_speech_params("你好", "glm-tts", None, "wav", 1.5, {"pitch": 2})
        assert params["json"]["speed"] == 1.5
        assert params["json"]["pitch"] == 2

    def test_model_required(self):
        with pytest.raises(ValueError, match="语音合成模型"):
            _client()._build_speech_params("你好", None, None, "wav", None, {})

    def test_url(self):
        assert _client()._get_speech_url() == "https://api.example.com/v1/audio/speech"


class TestResultFinalization:
    class _FakeResult:
        def __init__(self, status, data):
            self.status = status
            self.data = data

    def test_speech_writes_output_file(self, tmp_path):
        out = tmp_path / "nested" / "o.wav"
        result = OpenAIClient._finalize_speech(self._FakeResult("success", b"AUDIO"), out)
        assert result == out
        assert out.read_bytes() == b"AUDIO"

    def test_speech_returns_bytes_without_output(self):
        assert (
            OpenAIClient._finalize_speech(self._FakeResult("success", b"AUDIO"), None) == b"AUDIO"
        )

    def test_speech_error_returns_request_result(self, tmp_path):
        failed = self._FakeResult("error", {"error": "HTTP 400"})
        out = tmp_path / "o.wav"
        assert OpenAIClient._finalize_speech(failed, out) is failed
        assert not out.exists()

    def test_transcription_text_by_default(self):
        result = self._FakeResult("success", {"text": "你好"})
        assert OpenAIClient._finalize_transcription(result, False, False) == "你好"

    def test_transcription_details(self):
        result = self._FakeResult("success", {"text": "你好"})
        detailed = OpenAIClient._finalize_transcription(result, True, False)
        assert isinstance(detailed, TranscriptionResult)

    def test_transcription_raw_bypasses_parsing(self):
        result = self._FakeResult("error", {"error": "HTTP 400"})
        assert OpenAIClient._finalize_transcription(result, False, True) is result

    def test_transcription_error_returns_request_result(self):
        result = self._FakeResult("error", {"error": "HTTP 400"})
        assert OpenAIClient._finalize_transcription(result, False, False) is result


class TestBatchArgumentValidation:
    async def test_filenames_length_mismatch(self):
        with pytest.raises(ValueError, match="filenames 数量"):
            await _client().transcribe_batch([b"a", b"b"], model="m", filenames=["only.wav"])

    async def test_outputs_length_mismatch(self):
        with pytest.raises(ValueError, match="outputs 数量"):
            await _client().speech_batch(["a", "b"], model="m", outputs=["only.wav"])


class TestConcurrencyLimitValidation:
    """并发数 <= 0 会造出永远拿不到配额的信号量，请求静默挂死"""

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejected_at_construction(self, bad):
        from flexllm.async_api.core import ConcurrencyLimiter

        with pytest.raises(ValueError, match="并发数必须是 >= 1"):
            ConcurrencyLimiter(bad)

    def test_requester_rejects_zero(self):
        from flexllm.async_api.core import ConcurrentRequester

        with pytest.raises(ValueError, match="并发数必须是 >= 1"):
            ConcurrentRequester(concurrency_limit=0)

    def test_del_survives_half_constructed_object(self):
        """__init__ 抛异常后对象只构造了一半，__del__ 不能再抛 AttributeError 盖住真错误"""
        from flexllm.async_api.core import ConcurrentRequester

        obj = ConcurrentRequester.__new__(ConcurrentRequester)
        obj.__del__()  # 不应抛异常

    def test_valid_limit_accepted(self):
        from flexllm.async_api.core import ConcurrencyLimiter

        assert ConcurrencyLimiter(1).limit == 1


class TestPoolDelegation:
    def test_pool_exposes_audio_methods(self):
        from flexllm import LLMClient

        client = LLMClient(base_url="https://api.example.com/v1", api_key="k", model="m")
        for name in ("transcribe", "transcribe_batch", "speech", "speech_batch"):
            assert callable(getattr(client, name))
        assert client._audio_client() is client._single_client

    def test_non_openai_endpoint_rejected(self):
        from flexllm import ClaudeClient, LLMClient

        client = LLMClient(base_url="https://api.example.com/v1", api_key="k", model="m")
        client._single_client = ClaudeClient(api_key="k")
        with pytest.raises(NotImplementedError, match="语音端点"):
            client._audio_client()
