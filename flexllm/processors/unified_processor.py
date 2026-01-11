#!/usr/bin/env python3
"""
统一的图像处理器
合并本地文件处理和URL处理功能，提供高性能的批量消息预处理
"""

import asyncio
import aiohttp
import time
import os
import base64
import hashlib
import json
import sys
import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union, Callable
from dataclasses import dataclass
from functools import lru_cache
from copy import deepcopy
from collections import defaultdict
import gc
import contextlib

import cv2
import numpy as np
from PIL import Image
from loguru import logger

# 导入缓存配置
try:
    from .image_processor import (
        ImageCacheConfig,
        DEFAULT_CACHE_DIR,
        get_target_size,
        LANCZOS,
    )

    HAS_IMAGE_PROCESSOR = True
except ImportError:
    HAS_IMAGE_PROCESSOR = False
    DEFAULT_CACHE_DIR = "cache"

try:
    from tqdm.asyncio import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


@contextlib.contextmanager
def suppress_stdout():
    """上下文管理器，用于抑制stdout输出"""
    old_stdout = sys.stdout
    try:
        sys.stdout = io.StringIO()
        yield
    finally:
        sys.stdout = old_stdout


@contextlib.contextmanager
def suppress_stderr():
    """上下文管理器，用于抑制stderr输出"""
    old_stderr = sys.stderr
    try:
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stderr = old_stderr


@contextlib.contextmanager
def suppress_all_output():
    """上下文管理器，用于抑制所有输出"""
    with suppress_stdout(), suppress_stderr():
        yield


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
    if len(source) > 100 and all(
        c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
        for c in source
    ):
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
        pattern = r"data:image/[^;]+;base64,[A-Za-z0-9+/]+=*"

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
class UnifiedProcessorConfig:
    """统一处理器配置"""

    # 线程和并发配置
    max_workers: int = 8
    max_concurrent: int = 10
    enable_multithreading: bool = True

    # 图像质量配置
    jpeg_quality: int = 95
    png_compression: int = 1
    webp_quality: int = 90

    # 缓存配置
    memory_cache_size_mb: int = 500
    enable_disk_cache: bool = True
    disk_cache_dir: str = DEFAULT_CACHE_DIR
    force_refresh_disk_cache: bool = False
    retry_failed_disk_cache: bool = False

    # 性能配置
    prefetch_size: int = 50
    enable_simd: bool = True
    suppress_opencv_output: bool = True

    # 超时配置
    single_file_timeout: float = 10.0
    batch_timeout: float = 60.0
    network_timeout: float = 15.0

    @classmethod
    def default(cls) -> "UnifiedProcessorConfig":
        """默认配置"""
        return cls()

    @classmethod
    def high_performance(cls) -> "UnifiedProcessorConfig":
        """高性能配置"""
        return cls(
            max_workers=16,
            max_concurrent=32,
            jpeg_quality=95,
            png_compression=3,
            memory_cache_size_mb=1000,
            prefetch_size=100,
        )

    @classmethod
    def memory_optimized(cls) -> "UnifiedProcessorConfig":
        """内存优化配置"""
        return cls(
            max_workers=4,
            max_concurrent=6,
            jpeg_quality=80,
            png_compression=6,
            memory_cache_size_mb=200,
            prefetch_size=20,
        )

    @classmethod
    def from_image_cache_config(cls, cache_config: "ImageCacheConfig") -> "UnifiedProcessorConfig":
        """从旧版本ImageCacheConfig创建新配置"""
        return cls(
            enable_disk_cache=cache_config.enabled,
            disk_cache_dir=cache_config.cache_dir,
            force_refresh_disk_cache=cache_config.force_refresh,
            retry_failed_disk_cache=cache_config.retry_failed,
        )

    @classmethod
    def auto_detect(cls) -> "UnifiedProcessorConfig":
        """自适应配置，根据系统资源自动调整"""
        try:
            import psutil
            import os

            # 获取系统信息
            cpu_count = os.cpu_count() or 4
            memory_gb = psutil.virtual_memory().total / (1024**3)

            # 根据CPU核心数调整worker数量
            max_workers = max(4, min(cpu_count, 24))
            max_concurrent = max(6, min(cpu_count * 2, 40))

            # 根据内存大小调整缓存
            if memory_gb >= 16:
                # 16GB+: 高性能模式
                cache_size = 1000
                prefetch_size = 100
                jpeg_quality = 95
            elif memory_gb >= 8:
                # 8-16GB: 平衡模式
                cache_size = 500
                prefetch_size = 50
                jpeg_quality = 90
            else:
                # <8GB: 节省模式
                cache_size = 200
                prefetch_size = 20
                jpeg_quality = 80

            return cls(
                max_workers=max_workers,
                max_concurrent=max_concurrent,
                memory_cache_size_mb=cache_size,
                prefetch_size=prefetch_size,
                jpeg_quality=jpeg_quality,
                png_compression=3,
                enable_disk_cache=True,  # 默认启用磁盘缓存
            )

        except ImportError:
            # 如果没有psutil，回退到默认配置
            return cls.default()
        except Exception:
            # 其他异常，回退到默认配置
            return cls.default()


class UnifiedMemoryCache:
    """统一的线程安全内存缓存"""

    def __init__(self, max_size_mb: int = 500):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache = {}
        self.access_times = {}
        self.cache_sizes = {}
        self.current_size = 0
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0

    def _evict_lru(self):
        """清理LRU项目"""
        if not self.cache:
            return

        # 找到最少使用的项目
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])

        # 移除LRU项目
        if lru_key in self.cache:
            self.cache.pop(lru_key)
            self.access_times.pop(lru_key)
            size = self.cache_sizes.pop(lru_key, 0)
            self.current_size -= size

    def _generate_cache_key(
        self,
        source: str,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        max_pixels: Optional[int] = None,
        **kwargs,
    ) -> str:
        """生成缓存键"""
        try:
            key_parts = [source]

            # 添加处理参数
            if max_width is not None:
                key_parts.append(f"w:{max_width}")
            if max_height is not None:
                key_parts.append(f"h:{max_height}")
            if max_pixels is not None:
                key_parts.append(f"p:{max_pixels}")

            # 对于本地文件，添加修改时间
            if os.path.exists(source):
                try:
                    mtime = os.path.getmtime(source)
                    key_parts.append(f"mtime:{mtime}")
                except:
                    pass

            # 添加其他参数
            for key in sorted(kwargs.keys()):
                if kwargs[key] is not None:
                    key_parts.append(f"{key}:{kwargs[key]}")

            key_data = "|".join(key_parts)
            return hashlib.md5(key_data.encode()).hexdigest()
        except Exception:
            return hashlib.md5(source.encode()).hexdigest()

    def get(self, source: str, **kwargs) -> Optional[str]:
        """获取缓存数据"""
        cache_key = self._generate_cache_key(source, **kwargs)

        with self.lock:
            if cache_key in self.cache:
                self.access_times[cache_key] = time.time()
                self.hit_count += 1
                return self.cache[cache_key]
            else:
                self.miss_count += 1
                return None

    def put(self, source: str, data: str, **kwargs):
        """存储缓存数据"""
        cache_key = self._generate_cache_key(source, **kwargs)

        with self.lock:
            try:
                data_size = len(data.encode("utf-8"))

                # 如果数据太大，不缓存
                if data_size > self.max_size_bytes * 0.5:
                    return

                # 清理空间
                while (
                    self.current_size + data_size > self.max_size_bytes and self.cache
                ):
                    self._evict_lru()

                # 存储数据
                self.cache[cache_key] = data
                self.access_times[cache_key] = time.time()
                self.cache_sizes[cache_key] = data_size
                self.current_size += data_size
            except Exception:
                # 静默处理缓存错误
                pass

    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.cache_sizes.clear()
            self.current_size = 0
            gc.collect()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = (
                (self.hit_count / total_requests * 100) if total_requests > 0 else 0
            )

            return {
                "cached_items": len(self.cache),
                "current_size_mb": self.current_size / 1024 / 1024,
                "max_size_mb": self.max_size_bytes / 1024 / 1024,
                "usage_percent": (self.current_size / self.max_size_bytes * 100)
                if self.max_size_bytes > 0
                else 0,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate_percent": hit_rate,
                "total_requests": total_requests,
                "avg_item_size_kb": (self.current_size / 1024 / len(self.cache))
                if self.cache
                else 0,
                "cache_efficiency": "excellent"
                if hit_rate > 80
                else "good"
                if hit_rate > 60
                else "fair"
                if hit_rate > 40
                else "poor",
            }


class UnifiedImageProcessor:
    """统一的图像处理器，支持本地文件和URL"""

    def __init__(self, config: Optional[UnifiedProcessorConfig] = None):
        self.config = config or UnifiedProcessorConfig.default()
        self.memory_cache = UnifiedMemoryCache(self.config.memory_cache_size_mb)
        
        # 磁盘缓存配置
        self.disk_cache_config = None
        if self.config.enable_disk_cache and HAS_IMAGE_PROCESSOR:
            self.disk_cache_config = ImageCacheConfig(
                enabled=True,
                cache_dir=self.config.disk_cache_dir,
                force_refresh=self.config.force_refresh_disk_cache,
                retry_failed=self.config.retry_failed_disk_cache
            )

        # 线程池和锁
        self.executor = None
        self.processing_locks: Dict[str, asyncio.Lock] = {}
        self.lock = asyncio.Lock()
        self._executor_initialized = False
        self._init_lock = threading.Lock()

        # 性能统计
        self._total_processed = 0
        self._total_processing_time = 0.0
        self._start_time = time.time()

        # 初始化OpenCV优化
        self._init_opencv_optimizations()

    def _init_opencv_optimizations(self):
        """初始化OpenCV优化设置"""
        try:
            with (
                suppress_all_output()
                if self.config.suppress_opencv_output
                else contextlib.nullcontext()
            ):
                cv2.setUseOptimized(True)
                cv2.setNumThreads(self.config.max_workers)
                cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)

                if self.config.enable_simd and hasattr(cv2, "useOptimized"):
                    cv2.useOptimized()
        except Exception:
            pass

    def _get_executor(self) -> ThreadPoolExecutor:
        """获取线程池执行器（延迟初始化）"""
        if not self._executor_initialized:
            with self._init_lock:
                if not self._executor_initialized:
                    self.executor = ThreadPoolExecutor(
                        max_workers=self.config.max_workers,
                        thread_name_prefix="unified_processor",
                    )
                    self._executor_initialized = True
        return self.executor

    async def _get_processing_lock(self, cache_key: str) -> asyncio.Lock:
        """获取文件处理锁"""
        async with self.lock:
            if cache_key not in self.processing_locks:
                self.processing_locks[cache_key] = asyncio.Lock()
            return self.processing_locks[cache_key]

    def _detect_image_format(self, file_path: str) -> str:
        """检测图像格式"""
        try:
            ext = Path(file_path).suffix.lower()
            format_map = {
                ".jpg": "JPEG",
                ".jpeg": "JPEG",
                ".png": "PNG",
                ".webp": "WEBP",
                ".bmp": "BMP",
                ".tiff": "TIFF",
                ".tif": "TIFF",
            }
            return format_map.get(ext, "JPEG")
        except Exception:
            return "JPEG"

    def _get_encode_params(self, format_type: str) -> List[int]:
        """获取编码参数"""
        try:
            if format_type == "JPEG":
                return [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
            elif format_type == "PNG":
                return [cv2.IMWRITE_PNG_COMPRESSION, self.config.png_compression]
            elif format_type == "WEBP":
                return [cv2.IMWRITE_WEBP_QUALITY, self.config.webp_quality]
            else:
                return []
        except Exception:
            return []

    def _calculate_target_size(
        self,
        original_width: int,
        original_height: int,
        max_width: Optional[int],
        max_height: Optional[int],
        max_pixels: Optional[int],
    ) -> Tuple[int, int]:
        """计算目标尺寸"""
        try:
            width, height = original_width, original_height

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

            return max(1, width), max(1, height)
        except Exception:
            return original_width, original_height

    def _process_local_file_sync(
        self,
        file_path: str,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        max_pixels: Optional[int] = None,
        return_with_mime: bool = True,
    ) -> str:
        """同步处理本地文件"""
        try:
            with (
                suppress_all_output()
                if self.config.suppress_opencv_output
                else contextlib.nullcontext()
            ):
                # 使用OpenCV读取图像
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"无法读取图像文件: {file_path}")

                original_height, original_width = img.shape[:2]

                # 计算目标尺寸
                target_width, target_height = self._calculate_target_size(
                    original_width, original_height, max_width, max_height, max_pixels
                )

                # 如果需要调整大小
                if target_width != original_width or target_height != original_height:
                    img = cv2.resize(
                        img,
                        (target_width, target_height),
                        interpolation=cv2.INTER_LANCZOS4,
                    )

                # 检测原始格式并编码
                format_type = self._detect_image_format(file_path)
                encode_params = self._get_encode_params(format_type)

                ext = f".{format_type.lower()}"
                if format_type == "JPEG":
                    ext = ".jpg"

                success, buffer = cv2.imencode(ext, img, encode_params)
                if not success:
                    raise ValueError(f"图像编码失败: {file_path}")

            # 转换为base64
            base64_data = base64.b64encode(buffer.tobytes()).decode("utf-8")

            # 添加MIME类型前缀
            if return_with_mime:
                mime_type = f"image/{format_type.lower()}"
                result = f"data:{mime_type};base64,{base64_data}"
            else:
                result = base64_data

            return result

        except Exception as e:
            raise ValueError(f"处理本地文件失败: {file_path}, 错误: {str(e)}")

    async def _process_url_async(
        self,
        url: str,
        session: aiohttp.ClientSession,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        max_pixels: Optional[int] = None,
        return_with_mime: bool = True,
    ) -> str:
        """异步处理URL"""
        try:
            # 如果有image_processor，使用它处理URL（包含磁盘缓存）
            if HAS_IMAGE_PROCESSOR:
                from .image_processor import encode_image_to_base64

                return await encode_image_to_base64(
                    url, session, max_width, max_height, max_pixels, 
                    return_with_mime, cache_config=self.disk_cache_config
                )
            else:
                # 简单的URL处理实现
                timeout = aiohttp.ClientTimeout(total=self.config.network_timeout)
                async with session.get(url, timeout=timeout) as response:
                    if response.status == 200:
                        content = await response.read()
                        base64_data = base64.b64encode(content).decode("utf-8")

                        if return_with_mime:
                            content_type = response.headers.get(
                                "content-type", "image/jpeg"
                            )
                            return f"data:{content_type};base64,{base64_data}"
                        return base64_data
                    else:
                        raise ValueError(f"HTTP {response.status}")
        except Exception as e:
            raise ValueError(
                f"处理URL失败: {safe_repr_source(url)}, 错误: {safe_repr_error(str(e))}"
            )

    async def process_single_source(
        self,
        source: str,
        session: Optional[aiohttp.ClientSession] = None,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        max_pixels: Optional[int] = None,
        return_with_mime: bool = True,
    ) -> str:
        """处理单个图像源（本地文件或URL）"""

        # 1. 首先检查内存缓存
        cached_result = self.memory_cache.get(
            source,
            max_width=max_width,
            max_height=max_height,
            max_pixels=max_pixels,
            return_with_mime=return_with_mime,
        )
        if cached_result is not None:
            return cached_result

        # 获取处理锁
        cache_key = self.memory_cache._generate_cache_key(
            source,
            max_width=max_width,
            max_height=max_height,
            max_pixels=max_pixels,
            return_with_mime=return_with_mime,
        )
        file_lock = await self._get_processing_lock(cache_key)

        async with file_lock:
            # 再次检查内存缓存（双重检查锁定模式）
            cached_result = self.memory_cache.get(
                source,
                max_width=max_width,
                max_height=max_height,
                max_pixels=max_pixels,
                return_with_mime=return_with_mime,
            )
            if cached_result is not None:
                return cached_result

            # 开始性能计时
            start_time = time.time()

            try:
                # 判断是本地文件还是URL
                if os.path.exists(source) or source.startswith("file://"):
                    file_path = source[7:] if source.startswith("file://") else source

                    # 在线程池中处理本地文件
                    executor = self._get_executor()
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            executor,
                            self._process_local_file_sync,
                            file_path,
                            max_width,
                            max_height,
                            max_pixels,
                            return_with_mime,
                        ),
                        timeout=self.config.single_file_timeout,
                    )
                else:
                    # 处理URL（会自动使用磁盘缓存）
                    if session is None:
                        async with aiohttp.ClientSession() as temp_session:
                            result = await asyncio.wait_for(
                                self._process_url_async(
                                    source,
                                    temp_session,
                                    max_width,
                                    max_height,
                                    max_pixels,
                                    return_with_mime,
                                ),
                                timeout=self.config.network_timeout,
                            )
                    else:
                        result = await asyncio.wait_for(
                            self._process_url_async(
                                source,
                                session,
                                max_width,
                                max_height,
                                max_pixels,
                                return_with_mime,
                            ),
                            timeout=self.config.network_timeout,
                        )

                # 将结果缓存到内存
                self.memory_cache.put(
                    source,
                    result,
                    max_width=max_width,
                    max_height=max_height,
                    max_pixels=max_pixels,
                    return_with_mime=return_with_mime,
                )

                # 更新性能统计
                processing_time = time.time() - start_time
                self._total_processed += 1
                self._total_processing_time += processing_time

                return result

            except asyncio.TimeoutError:
                logger.warning(f"处理超时: {safe_repr_source(source)}")
                return ""
            except Exception as e:
                logger.error(
                    f"处理失败: {safe_repr_source(source)}, 错误: {safe_repr_error(str(e))}"
                )
                return ""

    async def process_batch(
        self,
        sources: List[str],
        session: Optional[aiohttp.ClientSession] = None,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        max_pixels: Optional[int] = None,
        return_with_mime: bool = True,
    ) -> List[str]:
        """批量处理图像源"""
        if not sources:
            return []

        # 去重并保持顺序映射
        unique_sources = []
        source_indices = {}
        for i, source in enumerate(sources):
            if source not in source_indices:
                source_indices[source] = []
                unique_sources.append(source)
            source_indices[source].append(i)

        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def process_single_with_semaphore(source: str) -> Tuple[str, str]:
            async with semaphore:
                result = await self.process_single_source(
                    source, session, max_width, max_height, max_pixels, return_with_mime
                )
                return source, result

        # 并发处理所有唯一源
        tasks = [process_single_with_semaphore(source) for source in unique_sources]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.batch_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("批处理超时")
            results = [(source, "") for source in unique_sources]

        # 构建结果映射
        result_mapping = {}
        for result in results:
            if isinstance(result, Exception):
                continue
            source, processed_result = result
            result_mapping[source] = processed_result

        # 根据原始顺序返回结果
        final_results = []
        for source in sources:
            final_results.append(result_mapping.get(source, ""))

        return final_results

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = {"memory_cache": self.memory_cache.get_stats()}
        
        # 如果启用了磁盘缓存，添加磁盘缓存统计
        if self.config.enable_disk_cache and self.disk_cache_config and self.disk_cache_config.enabled:
            disk_stats = self._get_disk_cache_stats()
            stats["disk_cache"] = disk_stats
            
        return stats
        
    def _get_disk_cache_stats(self) -> Dict[str, Any]:
        """获取磁盘缓存统计信息"""
        if not self.disk_cache_config or not self.disk_cache_config.enabled:
            return {"enabled": False}
            
        try:
            cache_dir = Path(self.disk_cache_config.cache_dir)
            if not cache_dir.exists():
                return {"enabled": True, "cached_files": 0, "total_size_mb": 0}
                
            # 统计缓存文件
            image_files = list(cache_dir.glob("*"))
            image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.gif']]
            error_files = list(cache_dir.glob("*.error"))
            
            # 计算总大小
            total_size = sum(f.stat().st_size for f in image_files if f.is_file())
            
            return {
                "enabled": True,
                "cache_dir": str(cache_dir),
                "cached_images": len(image_files),
                "error_cache_files": len(error_files),
                "total_files": len(image_files) + len(error_files),
                "total_size_mb": total_size / 1024 / 1024,
                "force_refresh": self.disk_cache_config.force_refresh,
                "retry_failed": self.disk_cache_config.retry_failed,
            }
        except Exception as e:
            return {"enabled": True, "error": str(e)}

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        uptime = time.time() - self._start_time
        avg_processing_time = (
            self._total_processing_time / self._total_processed
            if self._total_processed > 0
            else 0
        )
        throughput = self._total_processed / uptime if uptime > 0 else 0

        return {
            "total_processed": self._total_processed,
            "total_processing_time": self._total_processing_time,
            "uptime_seconds": uptime,
            "avg_processing_time": avg_processing_time,
            "throughput_per_second": throughput,
            "cache_stats": self.get_cache_stats(),
            "config": {
                "max_workers": self.config.max_workers,
                "max_concurrent": self.config.max_concurrent,
                "memory_cache_size_mb": self.config.memory_cache_size_mb,
                "jpeg_quality": self.config.jpeg_quality,
            },
        }

    def clear_cache(self, clear_disk_cache: bool = False):
        """清空缓存"""
        # 清空内存缓存
        self.memory_cache.clear()
        
        # 可选地清空磁盘缓存
        if clear_disk_cache and self.disk_cache_config and self.disk_cache_config.enabled:
            self._clear_disk_cache()
            
    def _clear_disk_cache(self):
        """清空磁盘缓存"""
        try:
            cache_dir = Path(self.disk_cache_config.cache_dir)
            if cache_dir.exists():
                # 删除所有缓存文件
                for cache_file in cache_dir.iterdir():
                    if cache_file.is_file():
                        cache_file.unlink()
                        
                logger.info(f"已清空磁盘缓存目录: {cache_dir}")
        except Exception as e:
            logger.warning(f"清空磁盘缓存失败: {e}")

    def cleanup(self):
        """清理资源"""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            self.clear_cache()
        except Exception:
            pass


# 全局处理器实例
_global_unified_processor = None
_unified_processor_lock = threading.Lock()


def get_global_unified_processor(
    config: Optional[UnifiedProcessorConfig] = None,
) -> UnifiedImageProcessor:
    """获取全局统一处理器实例（单例模式）"""
    global _global_unified_processor

    if _global_unified_processor is None:
        with _unified_processor_lock:
            if _global_unified_processor is None:
                _global_unified_processor = UnifiedImageProcessor(config)

    return _global_unified_processor


async def process_content_recursive(
    content: Any,
    session: Optional[aiohttp.ClientSession] = None,
    processor: Optional[UnifiedImageProcessor] = None,
    **kwargs,
):
    """递归处理内容中的图像URL"""
    if processor is None:
        processor = get_global_unified_processor()

    if isinstance(content, dict):
        for key, value in content.items():
            if key == "url" and isinstance(value, str):
                # 处理图像URL
                try:
                    base64_data = await processor.process_single_source(
                        value, session, **kwargs
                    )
                    if base64_data:
                        content[key] = base64_data
                except Exception as e:
                    logger.error(
                        f"处理URL失败 {safe_repr_source(value)}: {safe_repr_error(str(e))}"
                    )
            else:
                await process_content_recursive(value, session, processor, **kwargs)
    elif isinstance(content, list):
        for item in content:
            await process_content_recursive(item, session, processor, **kwargs)


async def unified_messages_preprocess(
    messages: List[Dict[str, Any]],
    inplace: bool = False,
    processor_config: Optional[UnifiedProcessorConfig] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    统一的单个消息列表预处理

    Args:
        messages: 单个消息列表
        inplace: 是否原地修改
        processor_config: 处理器配置
        **kwargs: 其他处理参数

    Returns:
        处理后的消息列表
    """
    # 创建或获取处理器
    if processor_config:
        processor = UnifiedImageProcessor(processor_config)
    else:
        processor = get_global_unified_processor()

    try:
        # 如果不是原地修改，创建副本
        if not inplace:
            messages = deepcopy(messages)

        # 使用HTTP会话处理所有图像
        async with aiohttp.ClientSession() as session:
            # 递归处理所有消息内容
            for message in messages:
                await process_content_recursive(message, session, processor, **kwargs)

        return messages

    except Exception as e:
        logger.error(f"消息预处理失败: {e}")
        return messages


async def unified_batch_messages_preprocess(
    messages_list: Union[List[List[Dict[str, Any]]], Any],
    max_concurrent: int = 10,
    inplace: bool = False,
    processor_config: Optional[UnifiedProcessorConfig] = None,
    as_iterator: bool = False,
    progress_callback: Optional[Callable] = None,
    show_progress: bool = False,
    progress_desc: str = "统一处理消息",
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    max_pixels: Optional[int] = None,
    **kwargs,
) -> Union[List[List[Dict[str, Any]]], Any]:
    """
    统一的批量消息预处理函数

    完全兼容messages_processor.py的API，支持本地文件和URL的高性能处理

    Args:
        messages_list: 消息列表的列表，可以是列表、迭代器或异步迭代器
        max_concurrent: 最大并发处理数
        inplace: 是否原地修改
        processor_config: 处理器配置
        as_iterator: 是否返回异步迭代器
        progress_callback: 进度回调函数
        show_progress: 是否显示进度条
        progress_desc: 进度描述
        max_width: 最大宽度
        max_height: 最大高度
        max_pixels: 最大像素数
        **kwargs: 其他处理参数

    Returns:
        处理后的消息列表或异步迭代器
    """

    # 创建或获取处理器
    if processor_config:
        processor = UnifiedImageProcessor(processor_config)
    else:
        processor = get_global_unified_processor()

    print(f"{processor.config=}")

    # 创建处理单个消息列表的函数
    async def process_single_batch(messages, semaphore, index=None):
        async with semaphore:
            try:
                processed_messages = await unified_messages_preprocess(
                    messages,
                    inplace=inplace,
                    processor_config=processor_config,
                    max_width=max_width,
                    max_height=max_height,
                    max_pixels=max_pixels,
                    **kwargs,
                )
            except Exception as e:
                logger.error(f"批处理错误 {index}: {e}")
                processed_messages = messages
            return processed_messages, index

    # 进度报告函数
    def report_progress(current: int, total: int, start_time: float = None):
        if progress_callback:
            try:
                # 计算时间信息
                elapsed_time = time.time() - start_time if start_time else 0

                # 创建扩展的进度信息
                progress_info = {
                    "current": current,
                    "total": total,
                    "percentage": (current / total * 100) if total > 0 else 0,
                    "elapsed_time": elapsed_time,
                    "estimated_total_time": (elapsed_time / current * total)
                    if current > 0
                    else 0,
                    "estimated_remaining_time": (
                        elapsed_time / current * (total - current)
                    )
                    if current > 0
                    else 0,
                    "rate": current / elapsed_time if elapsed_time > 0 else 0,
                }

                # 如果回调函数接受单个参数，传递扩展信息；否则保持兼容性
                import inspect

                sig = inspect.signature(progress_callback)
                if len(sig.parameters) == 1:
                    progress_callback(progress_info)
                else:
                    progress_callback(current, total)

            except Exception as e:
                logger.warning(f"进度回调函数执行失败: {e}")

    # 如果要求返回迭代器
    if as_iterator:

        async def process_iterator():
            semaphore = asyncio.Semaphore(max_concurrent)

            # 检查是否为异步迭代器
            is_async_iterator = hasattr(messages_list, "__aiter__")

            processed_count = 0
            total_count = None
            messages_to_process = messages_list

            # 如果可以获取总数，先计算总数
            if not is_async_iterator and hasattr(messages_list, "__len__"):
                total_count = len(messages_list)
            elif not is_async_iterator:
                # 对于迭代器，先转换为列表获取长度
                messages_list_converted = list(messages_list)
                total_count = len(messages_list_converted)
                messages_to_process = iter(messages_list_converted)

            # 创建进度条
            pbar = None
            start_time = time.time()
            if show_progress and TQDM_AVAILABLE and total_count:
                bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                pbar = tqdm(
                    total=total_count,
                    desc=progress_desc,
                    unit="批次",
                    bar_format=bar_format,
                    ncols=100,
                    miniters=1,
                )

            try:
                # 处理异步迭代器
                if is_async_iterator:
                    pending_tasks = []
                    task_index = 0
                    async for messages in messages_to_process:
                        # 如果已经达到最大并发数，等待一个任务完成
                        if len(pending_tasks) >= max_concurrent:
                            done, pending_tasks = await asyncio.wait(
                                pending_tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                            for task in done:
                                result, _ = await task
                                processed_count += 1
                                if pbar:
                                    pbar.update(1)
                                report_progress(
                                    processed_count,
                                    total_count or processed_count,
                                    start_time,
                                )
                                yield result

                        # 创建新任务
                        task = asyncio.create_task(
                            process_single_batch(messages, semaphore, task_index)
                        )
                        pending_tasks.append(task)
                        task_index += 1

                    # 等待所有剩余任务完成
                    if pending_tasks:
                        for task in asyncio.as_completed(pending_tasks):
                            result, _ = await task
                            processed_count += 1
                            if pbar:
                                pbar.update(1)
                            report_progress(
                                processed_count,
                                total_count or processed_count,
                                start_time,
                            )
                            yield result

                # 处理同步迭代器或列表
                else:
                    # 转换为列表以避免消耗迭代器
                    if not isinstance(messages_to_process, (list, tuple)):
                        messages_list_converted = list(messages_to_process)
                    else:
                        messages_list_converted = messages_to_process

                    if not total_count:
                        total_count = len(messages_list_converted)
                        if pbar:
                            pbar.total = total_count

                    # 分批处理
                    for i in range(0, len(messages_list_converted), max_concurrent):
                        batch = messages_list_converted[i : i + max_concurrent]
                        tasks = [
                            process_single_batch(messages, semaphore, i + j)
                            for j, messages in enumerate(batch)
                        ]
                        results = await asyncio.gather(*tasks)

                        for result, _ in results:
                            processed_count += 1
                            if pbar:
                                pbar.update(1)
                            report_progress(processed_count, total_count, start_time)
                            yield result

            finally:
                if pbar:
                    pbar.close()

        return process_iterator()

    # 原始实现，返回列表
    else:
        semaphore = asyncio.Semaphore(max_concurrent)

        # 检查是否为异步迭代器
        is_async_iterator = hasattr(messages_list, "__aiter__")

        # 转换为列表
        if is_async_iterator:
            messages_list_converted = []
            async for messages in messages_list:
                messages_list_converted.append(messages)
        elif not isinstance(messages_list, (list, tuple)):
            messages_list_converted = list(messages_list)
        else:
            messages_list_converted = messages_list

        if not messages_list_converted:
            return []

        total_count = len(messages_list_converted)
        processed_count = 0

        # 创建进度条
        pbar = None
        start_time = time.time()
        if show_progress and TQDM_AVAILABLE:
            bar_format = (
                "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
            pbar = tqdm(
                total=total_count,
                desc=progress_desc,
                unit=" items",
                bar_format=bar_format,
                ncols=100,
                miniters=1,
            )

        try:
            # 分批处理以实现进度更新
            results = []
            for i in range(0, len(messages_list_converted), max_concurrent):
                batch = messages_list_converted[i : i + max_concurrent]
                tasks = [
                    process_single_batch(messages, semaphore, i + j)
                    for j, messages in enumerate(batch)
                ]
                batch_results = await asyncio.gather(*tasks)

                for result, _ in batch_results:
                    results.append(result)
                    processed_count += 1
                    if pbar:
                        pbar.update(1)
                    report_progress(processed_count, total_count, start_time)

            return results

        finally:
            if pbar:
                pbar.close()


# 向后兼容的别名和完整的API兼容性
messages_preprocess = unified_messages_preprocess
batch_messages_preprocess = unified_batch_messages_preprocess
batch_process_messages = unified_batch_messages_preprocess

# 专用别名
optimized_batch_messages_preprocess = unified_batch_messages_preprocess
improved_batch_messages_preprocess = unified_batch_messages_preprocess
opencv_batch_messages_preprocess = unified_batch_messages_preprocess


# 便捷函数
async def unified_encode_image_to_base64(
    image_source: Union[str, List[str]],
    session: Optional[aiohttp.ClientSession] = None,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    max_pixels: Optional[int] = None,
    return_with_mime: bool = True,
    processor_config: Optional[UnifiedProcessorConfig] = None,
) -> Union[str, List[str]]:
    """
    统一的图像编码函数，支持本地文件和URL

    Args:
        image_source: 图像源，可以是单个路径/URL或列表
        session: HTTP会话（可选）
        max_width: 最大宽度
        max_height: 最大高度
        max_pixels: 最大像素数
        return_with_mime: 是否返回带MIME前缀的结果
        processor_config: 处理器配置

    Returns:
        Base64编码的图像数据
    """
    processor = (
        UnifiedImageProcessor(processor_config)
        if processor_config
        else get_global_unified_processor()
    )

    if isinstance(image_source, str):
        return await processor.process_single_source(
            image_source, session, max_width, max_height, max_pixels, return_with_mime
        )
    elif isinstance(image_source, list):
        return await processor.process_batch(
            image_source, session, max_width, max_height, max_pixels, return_with_mime
        )
    else:
        raise ValueError(f"不支持的图像源类型: {type(image_source)}")


# 向后兼容别名
encode_image_to_base64 = unified_encode_image_to_base64
safe_optimized_encode_image_to_base64 = unified_encode_image_to_base64


def cleanup_global_unified_processor():
    """清理全局统一处理器"""
    global _global_unified_processor
    if _global_unified_processor:
        _global_unified_processor.cleanup()
        _global_unified_processor = None


# 示例用法
if __name__ == "__main__":

    async def test_unified_processor():
        config = UnifiedProcessorConfig.high_performance()
        processor = UnifiedImageProcessor(config)

        # 测试本地文件
        # local_result = await processor.process_single_source(
        #     "test_image.jpg", max_width=800, max_height=600
        # )
        # print(f"本地文件处理完成，长度: {len(local_result)}")

        # 测试URL
        async with aiohttp.ClientSession() as session:
            url_result = await processor.process_single_source(
                "https://p2.itc.cn/q_70/images03/20230402/1853ae33e80b499ebc120426a80b19d3.jpeg",
                session,
                max_width=80,
                max_height=60,
            )
            # 安全打印，避免打印整个base64数据
            print(f"URL处理完成，长度: {len(url_result)}")
            if len(url_result) > 100:
                print(f"结果预览: {url_result[:100]}...")
            else:
                print(f"完整结果: {url_result}")

        # 获取统计信息
        stats = processor.get_cache_stats()
        print(f"缓存统计: {stats}")

        processor.cleanup()

    asyncio.run(test_unified_processor())
