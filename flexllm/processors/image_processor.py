import numpy as np
import base64
import os
import hashlib
import json
import time
from copy import deepcopy
from io import BytesIO
from mimetypes import guess_type
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import aiohttp
import requests
from loguru import logger
from PIL import Image
from flexllm.utils.core import async_retry

# 兼容不同版本的PIL
try:
    LANCZOS = Image.LANCZOS
except AttributeError:
    # 在较旧版本的PIL中，LANCZOS可能被称为ANTIALIAS
    LANCZOS = Image.ANTIALIAS

# 默认的缓存目录
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/maque/image_cache")


def safe_repr_source(source: str, max_length: int = 100) -> str:
    """安全地表示图像源，避免输出大量base64字符串"""
    if not source:
        return "空源"
    
    # 检查是否是base64数据URI
    if source.startswith("data:image/") and ";base64," in source:
        parts = source.split(";base64,", 1)
        if len(parts) == 2:
            mime_type = parts[0].replace("data:", "")
            base64_data = parts[1]
            return f"[{mime_type} base64数据 长度:{len(base64_data)}]"
    
    # 检查是否是纯base64字符串（很长且只包含base64字符）
    if len(source) > 100 and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in source):
        return f"[base64数据 长度:{len(source)}]"
    
    # 普通字符串，截断显示
    if len(source) <= max_length:
        return source
    else:
        return source[:max_length] + "..."


def safe_repr_error(error_msg: str, max_length: int = 200) -> str:
    """安全地表示错误信息，避免输出大量base64字符串"""
    if not error_msg:
        return error_msg
    
    # 检查错误信息中是否包含data:image的base64数据
    if "data:image/" in error_msg and ";base64," in error_msg:
        import re
        # 使用正则表达式替换base64数据URI
        pattern = r'data:image/[^;]+;base64,[A-Za-z0-9+/]+=*'
        def replace_base64(match):
            full_uri = match.group(0)
            parts = full_uri.split(";base64,", 1)
            if len(parts) == 2:
                mime_type = parts[0].replace("data:", "")
                base64_data = parts[1]
                return f"[{mime_type} base64数据 长度:{len(base64_data)}]"
            return full_uri
        error_msg = re.sub(pattern, replace_base64, error_msg)
    
    # 截断过长的错误信息
    if len(error_msg) <= max_length:
        return error_msg
    else:
        return error_msg[:max_length] + "..."


@dataclass
class ImageCacheConfig:
    """图像缓存配置类，用于集中管理图像缓存的相关配置"""

    enabled: bool = False  # 是否启用缓存
    cache_dir: str = DEFAULT_CACHE_DIR  # 缓存目录路径
    force_refresh: bool = False  # 是否强制刷新缓存
    retry_failed: bool = False  # 是否重试已知失败的链接

    def __post_init__(self):
        """初始化后执行的操作，确保缓存目录存在"""
        if self.enabled:
            ensure_cache_dir(self.cache_dir)

    @classmethod
    def disabled(cls) -> "ImageCacheConfig":
        """快速创建一个禁用缓存的配置"""
        return cls(enabled=False)

    @classmethod
    def default(cls) -> "ImageCacheConfig":
        """创建默认配置（启用缓存但不强制刷新和重试失败）"""
        return cls(enabled=True)

    @classmethod
    def with_refresh(cls) -> "ImageCacheConfig":
        """创建启用缓存且强制刷新的配置"""
        return cls(enabled=True, force_refresh=True)

    @classmethod
    def with_retry(cls) -> "ImageCacheConfig":
        """创建启用缓存且重试失败链接的配置"""
        return cls(enabled=True, retry_failed=True)

    @classmethod
    def full_refresh(cls) -> "ImageCacheConfig":
        """创建启用缓存且同时强制刷新和重试失败的配置"""
        return cls(enabled=True, force_refresh=True, retry_failed=True)


def get_cache_path(url: str, cache_dir: str = DEFAULT_CACHE_DIR) -> Path:
    """获取图片的缓存路径"""
    # 使用URL的MD5作为文件名
    url_hash = hashlib.md5(url.encode()).hexdigest()
    # 获取URL中的文件扩展名
    ext = os.path.splitext(url)[-1].lower()
    if not ext or ext not in [".jpg", ".jpeg", ".png", ".webp", ".gif"]:
        ext = ".png"  # 默认使用.png 而不是 .jpg，因为 PNG 支持透明通道
    return Path(cache_dir) / f"{url_hash}{ext}"


def get_error_cache_path(url: str, cache_dir: str = DEFAULT_CACHE_DIR) -> Path:
    """获取图片请求错误的缓存文件路径"""
    # 使用URL的MD5作为文件名，但添加.error后缀
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return Path(cache_dir) / f"{url_hash}.error"


def ensure_cache_dir(cache_dir: str = DEFAULT_CACHE_DIR):
    """确保缓存目录存在"""
    os.makedirs(cache_dir, exist_ok=True)


def encode_base64_from_local_path(file_path, return_with_mime=True):
    """Encode a local file to a Base64 string, with optional MIME type prefix."""
    mime_type, _ = guess_type(file_path)
    mime_type = mime_type or "application/octet-stream"
    with open(file_path, "rb") as file:
        base64_data = base64.b64encode(file.read()).decode("utf-8")
        if return_with_mime:
            return f"data:{mime_type};base64,{base64_data}"
        return base64_data


async def encode_base64_from_url(
    url, session: aiohttp.ClientSession, return_with_mime=True
):
    """Fetch a file from a URL and encode it to a Base64 string, with optional MIME type prefix."""
    async with session.get(url) as response:
        response.raise_for_status()
        content = await response.read()
        mime_type = response.headers.get("Content-Type", "application/octet-stream")
        base64_data = base64.b64encode(content).decode("utf-8")
        if return_with_mime:
            return f"data:{mime_type};base64,{base64_data}"
        return base64_data


def encode_base64_from_pil(image: Image.Image, return_with_mime=True):
    """Encode a PIL image object to a Base64 string, with optional MIME type prefix."""
    buffer = BytesIO()
    image_format = image.format or "PNG"  # Default to PNG if format is unknown
    mime_type = f"image/{image_format.lower()}"
    image.save(buffer, format=image_format)
    buffer.seek(0)
    base64_data = base64.b64encode(buffer.read()).decode("utf-8")
    if return_with_mime:
        return f"data:{mime_type};base64,{base64_data}"
    return base64_data


async def encode_to_base64(
    file_source,
    session: aiohttp.ClientSession,
    return_with_mime: bool = True,
    return_pil: bool = False,
    cache_config: Optional[ImageCacheConfig] = None,
) -> Union[str, Tuple[str, Image.Image], Image.Image]:
    """A unified function to encode files to Base64 strings or return PIL Image objects.

    Args:
        file_source: File path, URL, or PIL Image object
        session: aiohttp ClientSession for async URL fetching
        return_with_mime: Whether to include MIME type prefix in base64 string
        return_pil: Whether to return PIL Image object (for image files)
        cache_config: Image cache configuration, if None or disabled, no caching will be used

    Returns:
        If return_pil is False: base64 string (with optional MIME prefix)
        If return_pil is True and input is image: (base64_string, PIL_Image) or just PIL_Image
        If return_pil is True and input is not image: base64 string
    """
    mime_type = None
    pil_image = None

    if isinstance(file_source, str):
        if file_source.startswith("file://"):
            file_path = file_source[7:]
            if not os.path.exists(file_path):
                raise ValueError("Local file not found.")
            mime_type, _ = guess_type(file_path)
            if return_pil and mime_type and mime_type.startswith("image"):
                pil_image = Image.open(file_path)
                if return_pil and not return_with_mime:
                    return pil_image
            with open(file_path, "rb") as file:
                content = file.read()

        elif os.path.exists(file_source):
            mime_type, _ = guess_type(file_source)
            if return_pil and mime_type and mime_type.startswith("image"):
                pil_image = Image.open(file_source)
                if return_pil and not return_with_mime:
                    return pil_image
            with open(file_source, "rb") as file:
                content = file.read()

        elif file_source.startswith("http"):
            # 对于URL，使用get_pil_image来获取图像，以利用缓存功能
            if return_pil or mime_type and mime_type.startswith("image"):
                # 获取PIL图像并利用缓存
                pil_image = await get_pil_image(
                    file_source, session, cache_config=cache_config
                )
                if return_pil and not return_with_mime:
                    return pil_image

                # 将PIL图像转换为字节内容
                buffer = BytesIO()
                image_format = pil_image.format or "PNG"
                mime_type = f"image/{image_format.lower()}"
                pil_image.save(buffer, format=image_format)
                content = buffer.getvalue()
            else:
                # 对于非图像文件，直接从URL获取内容
                async with session.get(file_source) as response:
                    response.raise_for_status()
                    content = await response.read()
                    mime_type = response.headers.get(
                        "Content-Type", "application/octet-stream"
                    )
        else:
            raise ValueError("Unsupported file source type.")

    elif isinstance(file_source, Image.Image):
        pil_image = file_source
        if return_pil and not return_with_mime:
            return pil_image

        buffer = BytesIO()
        image_format = file_source.format or "PNG"
        mime_type = f"image/{image_format.lower()}"
        file_source.save(buffer, format=image_format)
        content = buffer.getvalue()

    else:
        raise ValueError("Unsupported file source type.")

    base64_data = base64.b64encode(content).decode("utf-8")
    result = (
        f"data:{mime_type};base64,{base64_data}" if return_with_mime else base64_data
    )

    if return_pil and pil_image:
        return result, pil_image
    return result


async def encode_image_to_base64(
    image_source,
    session: aiohttp.ClientSession,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    max_pixels: Optional[int] = None,
    return_with_mime: bool = True,
    cache_config: Optional[ImageCacheConfig] = None,
) -> str:
    """Encode an image to base64 string with optional size constraints.

    Args:
        image_source: Can be a file path (str), URL (str), or PIL Image object
        session: aiohttp ClientSession for async URL fetching
        max_width: Optional maximum width for image resizing
        max_height: Optional maximum height for image resizing
        max_pixels: Optional maximum number of pixels (width * height)
        return_with_mime: Whether to include MIME type prefix in the result
        cache_config: Image cache configuration, if None or disabled, no caching will be used

    Returns:
        Base64 encoded string (with optional MIME prefix)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "图像处理功能需要安装 opencv-python。请运行: pip install flexllm[image]"
        )

    if isinstance(image_source, Image.Image):
        image = image_source
    else:
        # 使用更新后的get_pil_image函数，但不需要返回缓存路径
        image = await get_pil_image(
            image_source, session, cache_config=cache_config, return_cache_path=False
        )

    # Make a copy of the image to avoid modifying the original
    # image = image.copy()

    # Store original format
    original_format = image.format or "PNG"

    # Resize image based on provided constraints
    target_width, target_height = get_target_size(
        image, max_width, max_height, max_pixels
    )
    if target_width < image.width or target_height < image.height:
        image.thumbnail((target_width, target_height), LANCZOS)

    mime_type = f"image/{original_format.lower()}"

    # Convert processed image to base64
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # 直接编码为所需格式（可添加压缩参数）
    _, buffer = cv2.imencode(f".{original_format.lower()}", cv_img)
    base64_data = base64.b64encode(buffer).decode("utf-8")

    if return_with_mime:
        return f"data:{mime_type};base64,{base64_data}"
    return base64_data


def decode_base64_to_pil(base64_string):
    """将base64字符串解码为PIL Image对象"""
    try:
        # 如果base64字符串包含header (如 'data:image/jpeg;base64,')，去除它
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        # 解码base64为二进制数据
        image_data = base64.b64decode(base64_string)

        # 转换为PIL Image对象
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"无法将base64字符串解码为图像: {e!s}")


def decode_base64_to_file(base64_string, output_path, format="JPEG"):
    """将base64字符串解码并保存为图片文件"""
    try:
        # 获取PIL Image对象
        image = decode_base64_to_pil(base64_string)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # 保存图像
        image.save(output_path, format=format)
        return True
    except Exception as e:
        raise ValueError(f"无法将base64字符串保存为文件: {e!s}")


def decode_base64_to_bytes(base64_string):
    """将base64字符串解码为字节数据"""
    try:
        # 如果base64字符串包含header，去除它
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        # 解码为字节数据
        return base64.b64decode(base64_string)
    except Exception as e:
        raise ValueError(f"无法将base64字符串解码为字节数据: {e!s}")


@async_retry(retry_times=3, retry_delay=0.3)
async def _get_image_from_http(session, image_source):
    async with session.get(image_source) as response:
        response.raise_for_status()
        content = await response.read()
        image = Image.open(BytesIO(content))
        return image


async def get_pil_image(
    image_source,
    session: aiohttp.ClientSession = None,
    cache_config: Optional[ImageCacheConfig] = None,
    return_cache_path: bool = False,
):
    """从图像链接或本地路径获取PIL格式的图像。

    Args:
        image_source: 图像来源，可以是本地文件路径、URL或data URI
        session: 用于异步URL请求的aiohttp ClientSession，如果为None且需要时会创建临时会话
        cache_config: 图像缓存配置，如果为None则不使用缓存
        return_cache_path: 是否同时返回图像和缓存路径，如果为True则返回(image, cache_path)元组

    Returns:
        PIL.Image.Image 或 tuple[PIL.Image.Image, Path]:
            - 如果return_cache_path为False，返回加载的PIL图像对象
            - 如果return_cache_path为True，返回(image, cache_path)元组，对于非URL图像或未使用缓存时，cache_path为None

    Raises:
        ValueError: 当图像源无效或无法加载图像时
    """
    # 处理缓存配置
    if cache_config is None:
        cache_config = ImageCacheConfig.disabled()

    # 如果已经是PIL图像对象，直接返回
    if isinstance(image_source, Image.Image):
        return (image_source, None) if return_cache_path else image_source

    # 处理字符串类型的图像源（文件路径或URL）
    if isinstance(image_source, str):
        # 处理本地文件路径
        if image_source.startswith("file://"):
            file_path = image_source[7:]
            if not os.path.exists(file_path):
                raise ValueError(f"本地文件不存在: {file_path}")
            image = Image.open(file_path)
            return (image, Path(file_path)) if return_cache_path else image

        # 处理普通本地文件路径
        elif os.path.exists(image_source):
            image = Image.open(image_source)
            return (image, Path(image_source)) if return_cache_path else image

        # 处理data URI (base64编码的图像数据)
        elif image_source.startswith("data:image/"):
            try:
                # 解析data URI: data:image/jpeg;base64,xxxxx
                if ";base64," in image_source:
                    header, data = image_source.split(";base64,", 1)
                    import base64
                    image_data = base64.b64decode(data)
                    image = Image.open(BytesIO(image_data))
                    return (image, None) if return_cache_path else image
                else:
                    raise ValueError("不支持的data URI格式，仅支持base64编码")
            except Exception as e:
                raise ValueError(f"解析data URI失败: {str(e)}")

        # 处理URL
        elif image_source.startswith("http"):
            # 检查缓存
            cache_path = None
            error_cache_path = None

            if cache_config.enabled:
                cache_path = get_cache_path(image_source, cache_config.cache_dir)
                error_cache_path = get_error_cache_path(
                    image_source, cache_config.cache_dir
                )

                # 检查普通缓存
                if not cache_config.force_refresh and cache_path.exists():
                    image = Image.open(cache_path)
                    return (image, cache_path) if return_cache_path else image

                # 检查错误缓存
                if (
                    not cache_config.force_refresh
                    and not cache_config.retry_failed
                    and error_cache_path.exists()
                ):
                    try:
                        with open(error_cache_path, "r") as f:
                            error_data = json.load(f)
                        # 重新构造相同的错误
                        raise ValueError(f"缓存的错误: {error_data['message']}")
                    except (json.JSONDecodeError, KeyError):
                        # 如果错误缓存文件格式不正确，忽略它
                        pass  # 静默忽略错误缓存文件格式问题

            # 创建临时会话（如果未提供）
            close_session = False
            if session is None:
                session = aiohttp.ClientSession()
                close_session = True

            try:
                image = await _get_image_from_http(session, image_source)
                # 保存到缓存
                if cache_config.enabled:
                    try:
                        save_image_with_format(image, cache_path)
                        # 如果请求成功，删除可能存在的错误缓存
                        if error_cache_path and error_cache_path.exists():
                            error_cache_path.unlink()
                    except Exception as e:
                        pass  # 静默忽略缓存保存失败

                return (image, cache_path) if return_cache_path else image
            except Exception as e:
                # 缓存错误信息
                if cache_config.enabled and error_cache_path:
                    try:
                        with open(error_cache_path, "w") as f:
                            error_data = {
                                "url": image_source,
                                "timestamp": time.time(),
                                "message": str(e),
                                "type": type(e).__name__,
                            }
                            json.dump(error_data, f)
                    except Exception as cache_err:
                        pass  # 静默忽略缓存错误信息失败
                # 重新抛出原始异常
                raise
            finally:
                # 如果是临时创建的会话，确保关闭
                if close_session and session:
                    await session.close()
        else:
            raise ValueError(f"不支持的图像源类型: {safe_repr_source(str(image_source))}")
    else:
        raise ValueError(f"不支持的图像源类型: {type(image_source)}")


def get_pil_image_sync(
    image_source,
    cache_config: Optional[ImageCacheConfig] = None,
    return_cache_path: bool = False,
):
    """从图像链接或本地路径获取PIL格式的图像（同步版本）。

    Args:
        image_source: 图像来源，可以是本地文件路径、URL或data URI
        cache_config: 图像缓存配置，如果为None则不使用缓存
        return_cache_path: 是否同时返回图像和缓存路径，如果为True则返回(image, cache_path)元组

    Returns:
        PIL.Image.Image 或 tuple[PIL.Image.Image, Path]:
            - 如果return_cache_path为False，返回加载的PIL图像对象
            - 如果return_cache_path为True，返回(image, cache_path)元组，对于非URL图像或未使用缓存时，cache_path为None

    Raises:
        ValueError: 当图像源无效或无法加载图像时
    """
    # 处理缓存配置
    if cache_config is None:
        cache_config = ImageCacheConfig.disabled()

    # 如果已经是PIL图像对象，直接返回
    if isinstance(image_source, Image.Image):
        return (image_source, None) if return_cache_path else image_source

    # 处理字符串类型的图像源（文件路径或URL）
    if isinstance(image_source, str):
        # 处理本地文件路径
        if image_source.startswith("file://"):
            file_path = image_source[7:]
            if not os.path.exists(file_path):
                raise ValueError(f"本地文件不存在: {file_path}")
            image = Image.open(file_path)
            return (image, Path(file_path)) if return_cache_path else image

        # 处理普通本地文件路径
        elif os.path.exists(image_source):
            image = Image.open(image_source)
            return (image, Path(image_source)) if return_cache_path else image

        # 处理data URI (base64编码的图像数据)
        elif image_source.startswith("data:image/"):
            try:
                # 解析data URI: data:image/jpeg;base64,xxxxx
                if ";base64," in image_source:
                    header, data = image_source.split(";base64,", 1)
                    import base64
                    image_data = base64.b64decode(data)
                    image = Image.open(BytesIO(image_data))
                    return (image, None) if return_cache_path else image
                else:
                    raise ValueError("不支持的data URI格式，仅支持base64编码")
            except Exception as e:
                raise ValueError(f"解析data URI失败: {str(e)}")

        # 处理URL
        elif image_source.startswith("http"):
            # 检查缓存
            cache_path = None
            error_cache_path = None

            if cache_config.enabled:
                cache_path = get_cache_path(image_source, cache_config.cache_dir)
                error_cache_path = get_error_cache_path(
                    image_source, cache_config.cache_dir
                )

                # 检查普通缓存
                if not cache_config.force_refresh and cache_path.exists():
                    image = Image.open(cache_path)
                    return (image, cache_path) if return_cache_path else image

                # 检查错误缓存
                if (
                    not cache_config.force_refresh
                    and not cache_config.retry_failed
                    and error_cache_path.exists()
                ):
                    try:
                        with open(error_cache_path, "r") as f:
                            error_data = json.load(f)
                        # 重新构造相同的错误
                        raise ValueError(f"缓存的错误: {error_data['message']}")
                    except (json.JSONDecodeError, KeyError):
                        # 如果错误缓存文件格式不正确，忽略它
                        pass  # 静默忽略错误缓存文件格式问题

            try:
                response = requests.get(image_source)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))

                # 保存到缓存
                if cache_config.enabled:
                    try:
                        save_image_with_format(image, cache_path)
                        # 如果请求成功，删除可能存在的错误缓存
                        if error_cache_path and error_cache_path.exists():
                            error_cache_path.unlink()
                    except Exception as e:
                        pass  # 静默忽略缓存保存失败

                return (image, cache_path) if return_cache_path else image
            except Exception as e:
                # 缓存错误信息
                if cache_config.enabled and error_cache_path:
                    try:
                        with open(error_cache_path, "w") as f:
                            error_data = {
                                "url": image_source,
                                "timestamp": time.time(),
                                "message": str(e),
                                "type": type(e).__name__,
                            }
                            json.dump(error_data, f)
                    except Exception as cache_err:
                        pass  # 静默忽略缓存错误信息失败
                # 重新抛出原始异常
                raise
        else:
            raise ValueError(f"不支持的图像源类型: {safe_repr_source(str(image_source))}")
    else:
        raise ValueError(f"不支持的图像源类型: {type(image_source)}")


def save_image_with_format(image: Image.Image, path: Path):
    """保存图片，自动处理格式转换问题"""
    # 获取目标格式
    target_format = path.suffix[1:].upper()  # 去掉点号并转大写
    if target_format == "JPG":
        target_format = "JPEG"

    # 创建图片副本以避免修改原图
    image = image.copy()

    # 处理调色板模式（P模式）
    if image.mode == "P":
        if "transparency" in image.info:
            # 如果有透明通道，转换为 RGBA
            image = image.convert("RGBA")
        else:
            # 如果没有透明通道，转换为 RGB
            image = image.convert("RGB")

    # 如果是 JPEG 格式且图片有 alpha 通道，需要特殊处理
    if target_format == "JPEG" and image.mode in ("RGBA", "LA"):
        # 创建白色背景
        background = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "RGBA":
            background.paste(image, mask=image.split()[3])  # 使用alpha通道作为mask
        else:
            background.paste(image, mask=image.split()[1])  # LA模式，使用A通道作为mask
        image = background

    # 确保图片模式与目标格式兼容
    if target_format == "JPEG" and image.mode not in ("RGB", "CMYK", "L"):
        image = image.convert("RGB")

    # 保存图片
    try:
        if target_format == "JPEG":
            image.save(path, format=target_format, quality=95)
        else:
            image.save(path, format=target_format)
    except Exception as e:
        pass  # 静默忽略图片保存格式问题
        # 如果保存失败，尝试转换为 RGB 后保存
        if image.mode != "RGB":
            image = image.convert("RGB")
            image.save(path, format=target_format)


def get_target_size(image, max_width, max_height, max_pixels):
    width, height = image.width, image.height

    # 应用最大宽度/高度限制
    if max_width and width > max_width:
        height = int(height * max_width / width)
        width = max_width

    if max_height and height > max_height:
        width = int(width * max_height / height)
        height = max_height

    # 应用最大像素限制
    if max_pixels and (width * height > max_pixels):
        ratio = (max_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)

    return width, height
