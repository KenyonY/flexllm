# 统一的高性能图像处理和消息预处理模块

# 核心图像处理功能
from .image_processor import (
    ImageCacheConfig,
    decode_base64_to_bytes,
    decode_base64_to_file,
    decode_base64_to_pil,
    encode_base64_from_local_path,
    encode_base64_from_pil,
    encode_media_to_base64,
    encode_to_base64,
    get_pil_image,
    get_pil_image_sync,
)

# 统一高性能处理器（默认路径）
from .unified_processor import (
    UnifiedImageProcessor,
    UnifiedMemoryCache,
    UnifiedProcessorConfig,
    batch_process_messages,
    cleanup_global_unified_processor,
    get_global_unified_processor,
    process_content_recursive,
    unified_encode_image_to_base64,
)
from .unified_processor import batch_process_messages as batch_messages_preprocess
from .unified_processor import batch_process_messages as unified_batch_process_messages

# 公开的 encode_image_to_base64 统一为 unified 版本（session 可选）。
# image_processor.encode_image_to_base64（session 必填、支持磁盘缓存）仅供内部使用。
from .unified_processor import unified_encode_image_to_base64 as encode_image_to_base64
from .unified_processor import (
    unified_messages_preprocess as messages_preprocess,
)

__all__ = [
    # 图像缓存配置
    "ImageCacheConfig",
    # 核心图像处理
    "encode_image_to_base64",
    "encode_to_base64",
    "get_pil_image",
    "get_pil_image_sync",
    "decode_base64_to_pil",
    "decode_base64_to_file",
    "decode_base64_to_bytes",
    "encode_base64_from_local_path",
    "encode_base64_from_pil",
    "encode_media_to_base64",
    # 基础消息处理
    "process_content_recursive",
    "messages_preprocess",
    "batch_messages_preprocess",
    "batch_process_messages",
    # 统一高性能处理器（推荐）
    "UnifiedProcessorConfig",
    "UnifiedImageProcessor",
    "UnifiedMemoryCache",
    "unified_batch_process_messages",
    "unified_encode_image_to_base64",
    "get_global_unified_processor",
    "cleanup_global_unified_processor",
]
