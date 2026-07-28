"""语音能力：转录 /audio/transcriptions 与合成 /audio/speech（OpenAI 兼容）

两个端点都复用 ConcurrentRequester —— 它把 request_params 原样透传给
aiohttp.ClientSession.request，因此并发、限流、重试、代理和进度条都是白拿的，
不需要另起一套引擎。

转录：请求体是 multipart/form-data。body 预先序列化成 bytes 而不是用
aiohttp.FormData —— FormData 是一次性的，重试时第二次发送会抛
"Form data has been processed already"，bytes 则可以重复发送。

字幕（srt/vtt）在本地由 segments 渲染，不透传给服务端：各家对 response_format 的
支持不一致（智谱 glm-asr 直接忽略它、永远返回 JSON），而 OpenAI 返回的纯文本响应
会被 ConcurrentRequester 判为非 JSON 错误。segments 是结构化的，本地渲染既跨
provider 一致，又不受服务端支持程度影响。

合成：请求体是 JSON，但响应是音频字节而非 JSON，所以走 response_type="bytes"。
"""

import asyncio
import logging
import mimetypes
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Union

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..async_api.interface import RequestResult

# 服务端 response_format 只透传这两个值，其余格式本地渲染（见模块 docstring）
_PASSTHROUGH_RESPONSE_FORMATS = ("json", "verbose_json")


@dataclass
class TranscriptionResult:
    """转录结果"""

    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[dict] = field(default_factory=list)
    raw: dict | None = None

    def to_srt(self) -> str:
        """渲染为 SRT 字幕"""
        return segments_to_srt(self.segments)

    def to_vtt(self) -> str:
        """渲染为 WebVTT 字幕"""
        return segments_to_vtt(self.segments)


def _format_timestamp(seconds: float, sep: str) -> str:
    """秒 -> HH:MM:SS<sep>mmm"""
    if seconds is None or seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    """segments -> SRT 字幕文本"""
    lines = []
    for i, seg in enumerate(segments or [], 1):
        start = _format_timestamp(seg.get("start", 0.0), ",")
        end = _format_timestamp(seg.get("end", 0.0), ",")
        lines.append(f"{i}\n{start} --> {end}\n{(seg.get('text') or '').strip()}\n")
    return "\n".join(lines)


def segments_to_vtt(segments: list[dict]) -> str:
    """segments -> WebVTT 字幕文本"""
    lines = ["WEBVTT\n"]
    for seg in segments or []:
        start = _format_timestamp(seg.get("start", 0.0), ".")
        end = _format_timestamp(seg.get("end", 0.0), ".")
        lines.append(f"{start} --> {end}\n{(seg.get('text') or '').strip()}\n")
    return "\n".join(lines)


def _sanitize_filename(name: str) -> str:
    """去掉会破坏 Content-Disposition 头的字符

    文件名来自用户输入，未转义的引号或换行会让攻击者伪造出额外的 header 行。
    """
    return name.replace("\\", "_").replace('"', "_").replace("\r", "_").replace("\n", "_")


def encode_multipart(
    fields: dict, filename: str, file_bytes: bytes, file_field: str = "file"
) -> tuple[bytes, str]:
    """把表单字段和文件编码成 multipart/form-data 请求体。

    Args:
        fields: 普通表单字段；值为 list/tuple 时展开成重复字段（如
                timestamp_granularities[]），值为 None 的字段跳过
        filename: 文件名，决定服务端识别的音频格式
        file_bytes: 文件内容
        file_field: 文件字段名

    Returns:
        (请求体 bytes, Content-Type 头的值)
    """
    boundary = uuid.uuid4().hex
    delimiter = f"--{boundary}\r\n".encode()
    parts = []

    for name, value in fields.items():
        if value is None:
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            if item is None:
                continue
            if isinstance(item, bool):
                item = "true" if item else "false"
            parts.append(delimiter)
            parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            parts.append(f"{item}\r\n".encode())

    safe_name = _sanitize_filename(filename)
    mime = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
    parts.append(delimiter)
    parts.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{safe_name}"\r\n'.encode()
    )
    parts.append(f"Content-Type: {mime}\r\n\r\n".encode())
    parts.append(file_bytes)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())

    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def read_audio_source(audio: Union[str, Path, bytes], filename: str | None = None):
    """把音频入参归一成 (bytes, filename)。

    Args:
        audio: 本地文件路径，或原始音频字节
        filename: audio 为 bytes 时的文件名，决定服务端识别的格式

    Returns:
        (音频字节, 文件名)
    """
    if isinstance(audio, (bytes, bytearray)):
        if not filename:
            raise ValueError("audio 为 bytes 时必须提供 filename，服务端依赖扩展名识别音频格式")
        return bytes(audio), filename

    path = Path(audio)
    if not path.is_file():
        raise FileNotFoundError(f"音频文件不存在: {path}")
    return path.read_bytes(), filename or path.name


def parse_transcription(data: dict) -> TranscriptionResult:
    """把服务端响应解析成 TranscriptionResult"""
    segments = data.get("segments") or []
    # 服务端未给 segments 时，用整段文本兜底成单条，让字幕渲染始终可用
    if not segments and data.get("text"):
        duration = data.get("duration")
        segments = [
            {"id": 0, "start": 0.0, "end": duration if duration else 0.0, "text": data["text"]}
        ]
    return TranscriptionResult(
        text=data.get("text", ""),
        language=data.get("language"),
        duration=data.get("duration"),
        segments=segments,
        raw=data,
    )


class AudioMixin:
    """给 OpenAI 兼容客户端提供 /audio/transcriptions 与 /audio/speech 能力。

    依赖宿主类的 _base_url / _api_key / _client / _model。
    """

    def _get_transcription_url(self) -> str:
        return f"{self._base_url}/audio/transcriptions"

    def _get_multipart_headers(self, content_type: str) -> dict:
        """转录用的请求头

        不能直接用 _get_headers()：那里带 Content-Type: application/json，
        会盖掉 multipart 的 boundary，服务端将解析不出任何字段。
        """
        headers = {k: v for k, v in self._get_headers().items() if k.lower() != "content-type"}
        headers["Content-Type"] = content_type
        return headers

    def _build_transcription_params(
        self,
        audio,
        model: str | None,
        filename: str | None,
        language: str | None,
        prompt: str | None,
        response_format: str | None,
        temperature: float | None,
        timestamp_granularities: list[str] | None,
        extra: dict,
    ) -> dict:
        """构造单个转录请求的 request_params（供 ConcurrentRequester 透传）"""
        effective_model = model or self._model
        if not effective_model:
            raise ValueError(
                "transcribe 需要指定转录模型（如 whisper-1 / glm-asr），"
                "客户端默认模型通常是对话模型，不能直接用于转录"
            )
        if response_format is not None and response_format not in _PASSTHROUGH_RESPONSE_FORMATS:
            raise ValueError(
                f"response_format 只支持 {_PASSTHROUGH_RESPONSE_FORMATS}；"
                "srt/vtt/text 由本地从 segments 渲染，用 TranscriptionResult.to_srt() 等"
            )

        file_bytes, name = read_audio_source(audio, filename)
        fields = {
            "model": effective_model,
            "language": language,
            "prompt": prompt,
            "response_format": response_format,
            "temperature": temperature,
            "timestamp_granularities[]": timestamp_granularities,
            **extra,
        }
        body, content_type = encode_multipart(fields, name, file_bytes)
        return {"data": body, "headers": self._get_multipart_headers(content_type)}

    async def transcribe(
        self,
        audio: Union[str, Path, bytes],
        model: str | None = None,
        filename: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: list[str] | None = None,
        return_details: bool = False,
        return_raw: bool = False,
        show_progress: bool = False,
        **extra,
    ) -> Union[str, TranscriptionResult, "RequestResult"]:
        """转录单个音频文件。

        Args:
            audio: 音频文件路径，或原始字节（此时需配合 filename）
            model: 转录模型（如 whisper-1、glm-asr）。省略时用客户端默认模型
            filename: audio 为 bytes 时的文件名，服务端据此识别格式
            language: 音频语言（ISO-639-1，如 zh、en），给出可提升准确率
            prompt: 引导用的提示词（如专有名词表）
            response_format: 仅 json / verbose_json；字幕格式在本地渲染
            temperature: 采样温度
            timestamp_granularities: 时间戳粒度，如 ["segment"]、["word"]
            return_details: 返回 TranscriptionResult（含 segments/language/duration）
            return_raw: 返回 RequestResult 原始响应
            show_progress: 显示进度条
            **extra: 透传给服务端的额外表单字段

        Returns:
            - return_raw=True: RequestResult
            - return_details=True: TranscriptionResult
            - 默认: 转录文本 str
            - 请求失败时: 返回 RequestResult（status="error"），不抛异常
        """
        params = self._build_transcription_params(
            audio,
            model,
            filename,
            language,
            prompt,
            response_format,
            temperature,
            timestamp_granularities,
            extra,
        )
        results, _ = await self._client.process_requests(
            request_params=[params],
            url=self._get_transcription_url(),
            method="POST",
            show_progress=show_progress,
        )
        return self._finalize_transcription(results[0], return_details, return_raw)

    @staticmethod
    def _finalize_transcription(result, return_details: bool, return_raw: bool):
        if return_raw:
            return result
        if result.status != "success":
            logger.warning("transcribe 请求失败: %s，返回 RequestResult", result.data)
            return result
        parsed = parse_transcription(result.data)
        return parsed if return_details else parsed.text

    def transcribe_sync(self, audio, **kwargs):
        """transcribe 的同步版本"""
        return asyncio.run(self.transcribe(audio, **kwargs))

    async def transcribe_batch(
        self,
        audios: list,
        model: str | None = None,
        filenames: list[str] | None = None,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: list[str] | None = None,
        return_details: bool = False,
        return_raw: bool = False,
        show_progress: bool = True,
        **extra,
    ) -> list:
        """并发转录多个音频。

        并发数与 QPS 由客户端的 concurrency_limit / max_qps 控制。
        返回值与入参一一对应，失败项为 RequestResult（与 chat_completions_batch 一致）。
        """
        if filenames is not None and len(filenames) != len(audios):
            raise ValueError(f"filenames 数量({len(filenames)})与 audios({len(audios)})不一致")

        request_params = [
            self._build_transcription_params(
                audio,
                model,
                filenames[i] if filenames else None,
                language,
                prompt,
                response_format,
                temperature,
                timestamp_granularities,
                extra,
            )
            for i, audio in enumerate(audios)
        ]
        results, _ = await self._client.process_requests(
            request_params=request_params,
            url=self._get_transcription_url(),
            method="POST",
            show_progress=show_progress,
        )
        return [self._finalize_transcription(r, return_details, return_raw) for r in results]

    def transcribe_batch_sync(self, audios: list, **kwargs) -> list:
        """transcribe_batch 的同步版本"""
        return asyncio.run(self.transcribe_batch(audios, **kwargs))

    # ========== 语音合成 /audio/speech ==========

    def _get_speech_url(self) -> str:
        return f"{self._base_url}/audio/speech"

    def _build_speech_params(
        self,
        text: str,
        model: str | None,
        voice: str | None,
        response_format: str | None,
        speed: float | None,
        extra: dict,
    ) -> dict:
        effective_model = model or self._model
        if not effective_model:
            raise ValueError(
                "speech 需要指定语音合成模型（如 glm-tts、tts-1），"
                "客户端默认模型通常是对话模型，不能直接用于合成"
            )
        body = {
            "model": effective_model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
            **extra,
        }
        body = {k: v for k, v in body.items() if v is not None}
        return {
            "json": body,
            "headers": self._get_headers(),
            # 响应是音频字节，不能按 JSON 解析
            "response_type": "bytes",
        }

    @staticmethod
    def _finalize_speech(result, output: Union[str, Path, None]):
        if result.status != "success":
            logger.warning("speech 请求失败: %s，返回 RequestResult", result.data)
            return result
        audio_bytes = result.data
        if output is None:
            return audio_bytes
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio_bytes)
        return path

    async def speech(
        self,
        text: str,
        model: str | None = None,
        voice: str | None = None,
        response_format: str | None = "wav",
        speed: float | None = None,
        output: Union[str, Path, None] = None,
        show_progress: bool = False,
        **extra,
    ) -> Union[bytes, Path, "RequestResult"]:
        """把文本合成为语音。

        Args:
            text: 要合成的文本（对应服务端的 input 字段）
            model: 合成模型（如 glm-tts、tts-1）。省略时用客户端默认模型
            voice: 音色（如 tongtong、alloy）
            response_format: 音频格式（wav/mp3 等），由服务端编码
            speed: 语速
            output: 给出则写入该路径并返回 Path，否则返回音频 bytes
            show_progress: 显示进度条
            **extra: 透传给服务端的额外字段

        Returns:
            - output 为 None: 音频 bytes
            - output 给出: 写入后的 Path
            - 请求失败时: RequestResult（status="error"），不抛异常
        """
        params = self._build_speech_params(text, model, voice, response_format, speed, extra)
        results, _ = await self._client.process_requests(
            request_params=[params],
            url=self._get_speech_url(),
            method="POST",
            show_progress=show_progress,
        )
        return self._finalize_speech(results[0], output)

    def speech_sync(self, text: str, **kwargs):
        """speech 的同步版本"""
        return asyncio.run(self.speech(text, **kwargs))

    async def speech_batch(
        self,
        texts: list[str],
        model: str | None = None,
        voice: str | None = None,
        response_format: str | None = "wav",
        speed: float | None = None,
        outputs: list | None = None,
        show_progress: bool = True,
        **extra,
    ) -> list:
        """并发合成多段文本。

        并发数与 QPS 由客户端的 concurrency_limit / max_qps 控制。
        返回值与入参一一对应，失败项为 RequestResult。
        """
        if outputs is not None and len(outputs) != len(texts):
            raise ValueError(f"outputs 数量({len(outputs)})与 texts({len(texts)})不一致")

        request_params = [
            self._build_speech_params(text, model, voice, response_format, speed, extra)
            for text in texts
        ]
        results, _ = await self._client.process_requests(
            request_params=request_params,
            url=self._get_speech_url(),
            method="POST",
            show_progress=show_progress,
        )
        return [
            self._finalize_speech(r, outputs[i] if outputs else None) for i, r in enumerate(results)
        ]

    def speech_batch_sync(self, texts: list[str], **kwargs) -> list:
        """speech_batch 的同步版本"""
        return asyncio.run(self.speech_batch(texts, **kwargs))
