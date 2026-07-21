"""音频预处理模块

支持采样率转换、声道转换、格式转换和时长截断。
依赖: soundfile + scipy（可选依赖组 flexllm[audio]）
"""

import io
import logging
from math import gcd

logger = logging.getLogger(__name__)

try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:
    sf = None
    HAS_SOUNDFILE = False

try:
    from scipy.signal import resample_poly as _resample_poly

    HAS_SCIPY = True
except ImportError:
    _resample_poly = None
    HAS_SCIPY = False

_FORMAT_TO_MIME = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
}


def preprocess_audio(
    audio_bytes: bytes,
    target_sample_rate: int | None = None,
    target_channels: int | None = None,
    target_audio_format: str | None = None,
    max_duration_seconds: float | None = None,
) -> tuple[bytes, str]:
    """预处理音频原始字节。

    Args:
        audio_bytes: 原始音频文件字节
        target_sample_rate: 目标采样率（如 16000）
        target_channels: 目标声道数（1=单声道, 2=立体声）
        target_audio_format: 目标格式（wav/flac/ogg）
        max_duration_seconds: 最大时长秒数，超出部分截断

    Returns:
        (处理后字节, MIME 类型)
    """
    if not HAS_SOUNDFILE:
        raise ImportError("音频预处理需要 soundfile。请运行: pip install flexllm[audio]")

    import numpy as np

    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    # 截断
    if max_duration_seconds is not None:
        max_samples = int(sr * max_duration_seconds)
        if data.shape[0] > max_samples:
            data = data[:max_samples]

    # 声道转换
    if target_channels is not None:
        cur_ch = 1 if data.ndim == 1 else data.shape[1]
        if target_channels == 1 and cur_ch > 1:
            data = data.mean(axis=1)
        elif target_channels > 1 and cur_ch == 1:
            vec = data if data.ndim == 1 else data.ravel()
            data = np.stack([vec] * target_channels, axis=1)

    # 重采样
    if target_sample_rate is not None and target_sample_rate != sr:
        if not HAS_SCIPY:
            raise ImportError("采样率转换需要 scipy。请运行: pip install flexllm[audio]")
        g = gcd(int(target_sample_rate), int(sr))
        up, down = int(target_sample_rate) // g, int(sr) // g
        if data.ndim == 1:
            data = _resample_poly(data, up, down).astype(np.float32)
        else:
            cols = [
                _resample_poly(data[:, c], up, down).astype(np.float32)
                for c in range(data.shape[1])
            ]
            data = np.stack(cols, axis=1)
        sr = target_sample_rate

    # 编码输出
    fmt = (target_audio_format or "wav").lower()
    if fmt not in _FORMAT_TO_MIME:
        logger.warning(f"不支持音频格式 '{fmt}'，回退到 wav")
        fmt = "wav"

    out = io.BytesIO()
    subtype = "VORBIS" if fmt == "ogg" else None
    sf.write(out, data, sr, format=fmt.upper(), subtype=subtype)
    out.seek(0)
    return out.read(), _FORMAT_TO_MIME[fmt]


def extract_audio_kwargs(kwargs: dict) -> dict:
    """从 kwargs 中提取音频预处理参数"""
    return {
        k: kwargs[k]
        for k in (
            "target_sample_rate",
            "target_channels",
            "target_audio_format",
            "max_duration_seconds",
        )
        if k in kwargs and kwargs[k] is not None
    }
