import asyncio
import os
from typing import Any

import aiohttp
from PIL import Image

from .image_processor import (
    LANCZOS,
    ImageCacheConfig,
    decode_base64_to_bytes,
    decode_base64_to_file,
    decode_base64_to_pil,
    encode_base64_from_local_path,
    encode_base64_from_pil,
    encode_image_to_base64,
    encode_to_base64,
    get_pil_image,
    get_pil_image_sync,
)

# 导入消息处理功能
from .messages_processor import (
    batch_process_messages,
    messages_preprocess,
    process_content_recursive,
)


class ImageProcessor:
    """便捷的图像处理类，封装了image_processor.py中的异步方法，提供更简单的接口。

    这个类会自动管理aiohttp.ClientSession的生命周期，使得调用异步图像处理方法更加方便。

    # 异步使用
    async with ImageProcessor() as processor:
        # 从URL编码图像，自动处理session
        base64_data = await processor.encode_from_url("https://example.com/image.jpg")

        # 使用新的图像缩放选项
        resized_base64 = await processor.encode_image(
            "https://example.com/large_image.jpg",
            max_width=800,
            max_height=600,
            max_pixels=1000000
        )

    # 同步使用
    processor = ImageProcessor()
    try:
        # 从本地文件编码
        local_base64 = processor.encode_from_local_path("image.jpg")

        # 使用run_async调用异步方法
        url_base64 = run_async(processor.encode_from_url("https://example.com/image.jpg"))
    finally:
        # 记得关闭session
        run_async(processor.close())

    """

    def __init__(self, session: aiohttp.ClientSession | None = None):
        """初始化ImageProcessor

        Args:
            session: 可选的aiohttp.ClientSession实例。如果不提供，将在需要时自动创建。
        """
        self._session = session
        self._own_session = session is None
        self._session_initialized = False

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """确保session已初始化

        Returns:
            aiohttp.ClientSession实例
        """
        if self._session is None:
            self._session = aiohttp.ClientSession()
            self._own_session = True
            self._session_initialized = True
        return self._session

    async def close(self):
        """关闭session（如果是由本类创建的）"""
        if self._own_session and self._session_initialized and self._session is not None:
            await self._session.close()
            self._session = None
            self._session_initialized = False

    async def __aenter__(self):
        """支持异步上下文管理器"""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出异步上下文管理器时关闭session"""
        await self.close()

    # 同步方法（不需要session）
    def encode_from_local_path(self, file_path: str, return_with_mime: bool = True) -> str:
        """从本地文件路径编码图像为base64字符串

        Args:
            file_path: 本地图像文件路径
            return_with_mime: 是否在结果中包含MIME类型前缀

        Returns:
            base64编码的字符串
        """
        return encode_base64_from_local_path(file_path, return_with_mime)

    def encode_from_pil(self, image: Image.Image, return_with_mime: bool = True) -> str:
        """从PIL.Image对象编码为base64字符串

        Args:
            image: PIL.Image对象
            return_with_mime: 是否在结果中包含MIME类型前缀

        Returns:
            base64编码的字符串
        """
        return encode_base64_from_pil(image, return_with_mime)

    def encode_from_url_sync(self, url: str, cache_config: ImageCacheConfig | None = None) -> str:
        """从URL同步编码图像为base64字符串

        Args:
            url: 图像URL
            cache_config: 图像缓存配置，如果为None则不使用缓存

        Returns:
            base64编码的字符串
        """
        image = get_pil_image_sync(url, cache_config=cache_config)
        return encode_base64_from_pil(image)

    def decode_to_pil(self, base64_string: str) -> Image.Image:
        """将base64字符串解码为PIL.Image对象

        Args:
            base64_string: base64编码的图像字符串

        Returns:
            PIL.Image对象
        """
        return decode_base64_to_pil(base64_string)

    def decode_to_file(self, base64_string: str, output_path: str, format: str = "JPEG") -> None:
        """将base64字符串解码并保存为文件

        Args:
            base64_string: base64编码的图像字符串
            output_path: 输出文件路径
            format: 图像格式
        """
        decode_base64_to_file(base64_string, output_path, format)

    def decode_to_bytes(self, base64_string: str) -> bytes:
        """将base64字符串解码为字节数据

        Args:
            base64_string: base64编码的图像字符串

        Returns:
            字节数据
        """
        return decode_base64_to_bytes(base64_string)

    # 异步方法（需要session）
    async def encode_from_url(
        self,
        url: str,
        return_with_mime: bool = True,
        cache_config: ImageCacheConfig | None = None,
    ) -> str:
        """从URL异步编码图像为base64字符串

        Args:
            url: 图像URL
            return_with_mime: 是否在结果中包含MIME类型前缀
            cache_config: 图像缓存配置，如果为None则不使用缓存

        Returns:
            base64编码的字符串
        """
        session = await self._ensure_session()
        result = await encode_image_to_base64(
            url, session, return_with_mime=return_with_mime, cache_config=cache_config
        )
        return result

    async def get_pil_image(
        self,
        image_source: str,
        max_width: int | None = None,
        max_height: int | None = None,
        max_pixels: int | None = None,
        cache_config: ImageCacheConfig | None = None,
        return_cache_path: bool = False,
    ) -> Image.Image | tuple[Image.Image, str]:
        """获取图像，支持多种输入源和大小限制

        Args:
            image_source: 图像源，可以是文件路径、URL
            max_width: 可选的最大宽度
            max_height: 可选的最大高度
            max_pixels: 可选的最大像素数量
            cache_config: 图像缓存配置，如果为None则不使用缓存
            return_cache_path: 是否返回缓存路径
        Returns:
            PIL.Image对象，如果return_cache_path为True，则返回(image, cache_path)元组
        """
        session = await self._ensure_session()
        if return_cache_path:
            image, cache_path = await get_pil_image(
                image_source,
                session,
                cache_config=cache_config,
                return_cache_path=return_cache_path,
            )
        else:
            image = await get_pil_image(
                image_source,
                session,
                cache_config=cache_config,
                return_cache_path=return_cache_path,
            )

        # Resize image based on provided constraints
        if max_width and max_height:
            # Use thumbnail to maintain aspect ratio while fitting within max dimensions
            image.thumbnail((max_width, max_height), LANCZOS)
        elif max_width:
            # Use thumbnail with unlimited height
            image.thumbnail((max_width, float("inf")), LANCZOS)
        elif max_height:
            # Use thumbnail with unlimited width
            image.thumbnail((float("inf"), max_height), LANCZOS)

        # Handle max_pixels constraint (after other resizing to avoid unnecessary work)
        if max_pixels and (image.width * image.height > max_pixels):
            # Calculate the ratio needed to get to max_pixels
            ratio = (max_pixels / (image.width * image.height)) ** 0.5
            # Use thumbnail to maintain aspect ratio
            target_width = int(image.width * ratio)
            target_height = int(image.height * ratio)
            image.thumbnail((target_width, target_height), LANCZOS)

        if return_cache_path:
            return image, cache_path
        else:
            return image

    async def encode_image(
        self,
        image_source: str | Image.Image,
        max_width: int | None = None,
        max_height: int | None = None,
        max_pixels: int | None = None,
        return_with_mime: bool = True,
        cache_config: ImageCacheConfig | None = None,
    ) -> str:
        """编码图像为base64字符串，支持多种输入源和大小限制

        Args:
            image_source: 图像源，可以是文件路径、URL或PIL.Image对象
            max_width: 可选的最大宽度限制
            max_height: 可选的最大高度限制
            max_pixels: 可选的最大像素数限制
            return_with_mime: 是否在结果中包含MIME类型前缀
            cache_config: 图像缓存配置，如果为None则不使用缓存
        Returns:
            base64编码的字符串
        """
        session = await self._ensure_session()
        result = await encode_image_to_base64(
            image_source,
            session,
            max_width=max_width,
            max_height=max_height,
            max_pixels=max_pixels,
            return_with_mime=return_with_mime,
            cache_config=cache_config,
        )
        return result

    async def encode_to_base64(
        self,
        file_source: str | Image.Image,
        return_with_mime: bool = True,
        return_pil: bool = False,
        cache_config: ImageCacheConfig | None = None,
    ) -> str | tuple[str, Image.Image] | Image.Image:
        """通用的编码方法，支持多种输入源和返回格式

        Args:
            file_source: 文件源，可以是文件路径、URL或PIL.Image对象
            return_with_mime: 是否在结果中包含MIME类型前缀
            return_pil: 是否返回PIL.Image对象
            cache_config: 图像缓存配置，如果为None则不使用缓存

        Returns:
            根据参数返回base64字符串、(base64字符串, PIL.Image)元组或PIL.Image对象
        """
        session = await self._ensure_session()
        result = await encode_to_base64(
            file_source,
            session,
            return_with_mime=return_with_mime,
            return_pil=return_pil,
            cache_config=cache_config,
        )
        return result

    async def process_content(
        self,
        content: dict[str, Any],
        inplace: bool = True,
        cache_config: ImageCacheConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """递归处理内容中的图像URL

        Args:
            content: 包含可能的图像URL的字典
            inplace: 是否原地修改内容
            cache_config: 图像缓存配置，如果为None则不使用缓存
            **kwargs: 传递给encode_image_to_base64的额外参数

        Returns:
            处理后的内容
        """
        session = await self._ensure_session()
        result = await process_content_recursive(
            content, session, cache_config=cache_config, **kwargs
        )
        return result

    async def preprocess_messages(
        self,
        messages: list[dict[str, Any]],
        inplace: bool = False,
        cache_config: ImageCacheConfig | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """预处理消息列表中的图像URL

        Args:
            messages: 消息列表
            inplace: 是否原地修改内容
            cache_config: 图像缓存配置，如果为None则不使用缓存
            **kwargs: 传递给process_content_recursive的额外参数

        Returns:
            处理后的消息列表
        """
        session = await self._ensure_session()

        result = await messages_preprocess(
            messages, inplace=inplace, cache_config=cache_config, **kwargs
        )
        return result

    async def batch_process_messages(
        self,
        messages_list: list[list[dict[str, Any]]],
        max_concurrent: int = 5,
        inplace: bool = False,
        cache_config: ImageCacheConfig | None = None,
        **kwargs,
    ) -> list[list[dict[str, Any]]]:
        """批量处理多组消息列表

        Args:
            messages_list: 多组消息列表
            max_concurrent: 最大并发处理数
            inplace: 是否原地修改内容
            cache_config: 图像缓存配置，如果为None则不使用缓存
            **kwargs: 传递给messages_preprocess的额外参数

        Returns:
            处理后的多组消息列表
        """
        session = await self._ensure_session()

        result = await batch_process_messages(
            messages_list,
            max_concurrent=max_concurrent,
            inplace=inplace,
            cache_config=cache_config,
            **kwargs,
        )
        return result


# 便捷函数，用于在同步代码中使用异步方法
def run_async(coro):
    """运行异步协程并返回结果

    Args:
        coro: 异步协程

    Returns:
        协程的结果
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # 如果没有事件循环，创建一个新的
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


# 示例用法
if __name__ == "__main__":
    # 异步用法示例
    async def async_example():
        # 使用上下文管理器自动管理session生命周期
        async with ImageProcessor() as processor:
            # 从URL编码图像
            base64_data = await processor.encode_from_url("https://example.com/image.jpg")
            # print(f"Encoded image length: {len(base64_data)}")

            # 使用缓存配置
            cache_config = ImageCacheConfig.default()

            # 处理消息中的图像URL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here's an image"},
                        {"type": "image_url", "url": "https://example.com/image.jpg"},
                    ],
                }
            ]
            processed_messages = await processor.preprocess_messages(
                messages, cache_config=cache_config
            )
            print("Processed messages")

    # 同步用法示例
    def sync_example():
        # 创建处理器
        processor = ImageProcessor()

        try:
            # 从本地文件编码图像
            if os.path.exists("local_image.jpg"):
                base64_data = processor.encode_from_local_path("local_image.jpg")
                # print(f"Encoded local image length: {len(base64_data)}")

            # 创建缓存配置
            cache_config = ImageCacheConfig(enabled=True, retry_failed=True)

            # 从URL编码图像（使用run_async运行异步方法）
            base64_data = run_async(
                processor.encode_from_url(
                    "https://example.com/image.jpg", cache_config=cache_config
                )
            )
            # print(f"Encoded remote image length: {len(base64_data)}")
        finally:
            # 关闭处理器（关闭session）
            run_async(processor.close())

    # 运行示例
    # run_async(async_example())
    # sync_example()
