"""图片磁盘缓存开关/路径接线（cache_image / cache_dir）

0.14 之前的 unified 重构（459dd6d）把 base 的 cache_image/cache_dir 与实际磁盘
缓存断开：缓存被写死为"总是开、路径固定 ~/.cache/flexllm/image_cache"，两个参数
被无视。本测试实证重新接线后：cache_image=False 不缓存、cache_image=True 缓存到
指定 cache_dir、缓存命中可避免重复下载。
"""

import os
import tempfile

from flexllm import LLMClient
from flexllm.msg_processors.image_processor import DEFAULT_CACHE_DIR

from .test_image_download_proxy import PngServer, _image_message


def _extract(processed: list[dict]) -> str:
    return processed[0]["content"][0]["image_url"]["url"]


def _client(**kw) -> LLMClient:
    return LLMClient(base_url="http://unused/v1", api_key="k", model="m", **kw)


class TestImageCacheWiring:
    async def test_default_does_not_cache(self):
        """cache_image 默认 False：图片被处理但不落磁盘缓存"""
        with tempfile.TemporaryDirectory() as d:
            async with PngServer() as img:
                c = _client(cache_image=False, cache_dir=d)
                out = await c._preprocess_messages(_image_message(img.url()), preprocess_msg=True)
            assert _extract(out).startswith("data:image/")
            assert os.listdir(d) == [], "cache_image=False 不应写缓存"

    async def test_cache_writes_to_custom_dir(self):
        """cache_image=True + cache_dir：缓存写到指定目录，证明两个参数都生效"""
        with tempfile.TemporaryDirectory() as d:
            async with PngServer() as img:
                c = _client(cache_image=True, cache_dir=d)
                out = await c._preprocess_messages(_image_message(img.url()), preprocess_msg=True)
            assert _extract(out).startswith("data:image/")
            assert len(os.listdir(d)) >= 1, "cache_image=True 应写缓存到 cache_dir"

    async def test_disk_cache_hit_avoids_redownload(self):
        """开缓存后，另一个共享同一 cache_dir 的 client 命中磁盘缓存，不再下载

        用两个独立 client（各自独立内存缓存）强制走磁盘缓存路径：第二个 client
        的内存缓存是空的，若不重复下载就只能是命中了磁盘缓存。
        """
        with tempfile.TemporaryDirectory() as d:
            async with PngServer() as img:
                url = img.url()
                await _client(cache_image=True, cache_dir=d)._preprocess_messages(
                    _image_message(url), preprocess_msg=True
                )
                assert img.hit_count == 1

                # 新 client、新内存缓存，同一磁盘缓存目录
                await _client(cache_image=True, cache_dir=d)._preprocess_messages(
                    _image_message(url), preprocess_msg=True
                )
                assert img.hit_count == 1, "第二次应命中磁盘缓存，不再下载"

    async def test_processor_reused_across_calls(self):
        """同一 client 跨调用复用同一处理器实例（避免每次新建、丢内存缓存）"""
        c = _client(cache_image=False)
        p1 = c._get_unified_processor()
        p2 = c._get_unified_processor()
        assert p1 is p2

    def test_default_cache_dir_relocated(self):
        """默认缓存路径已归拢到 ~/.flexllm/cache/image_cache"""
        assert DEFAULT_CACHE_DIR.endswith("/.flexllm/cache/image_cache")

    async def test_cache_image_false_default_dir_untouched(self):
        """cache_image=False 且不指定 cache_dir 时，默认缓存目录不应新增文件"""
        before = set(os.listdir(DEFAULT_CACHE_DIR)) if os.path.isdir(DEFAULT_CACHE_DIR) else set()
        async with PngServer() as img:
            c = _client(cache_image=False)
            await c._preprocess_messages(_image_message(img.url()), preprocess_msg=True)
        after = set(os.listdir(DEFAULT_CACHE_DIR)) if os.path.isdir(DEFAULT_CACHE_DIR) else set()
        assert after == before, "cache_image=False 不应往默认目录写缓存"
