# maque.mllm.processors - 统一的高性能图像处理和消息预处理模块

# 核心图像处理功能
from .image_processor import (
    ImageCacheConfig,
    encode_image_to_base64,
    encode_to_base64,
    get_pil_image,
    get_pil_image_sync,
    decode_base64_to_pil,
    decode_base64_to_file,
    decode_base64_to_bytes,
    encode_base64_from_local_path,
    encode_base64_from_pil,
)

# 基础消息处理功能
from .messages_processor import (
    process_content_recursive,
    messages_preprocess,
    batch_messages_preprocess,
    batch_process_messages,  # 别名
)

# 统一高性能处理器（推荐用于生产环境）
from .unified_processor import (
    UnifiedProcessorConfig,
    UnifiedImageProcessor,
    UnifiedMemoryCache,
    batch_process_messages as unified_batch_process_messages,
    unified_encode_image_to_base64,
    get_global_unified_processor,
    cleanup_global_unified_processor,
)

# 便捷的类接口
from .image_processor_helper import ImageProcessor

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
    # 便捷类接口
    "ImageProcessor",
]

# 版本和性能信息
__version__ = "3.0.0"  # 架构精简后的新版本
__description__ = "统一高性能图像处理和消息预处理模块 - 精简架构，专注于统一处理器"

# 性能建议
PERFORMANCE_RECOMMENDATIONS = {
    "unified_high_performance": {
        "function": "unified_batch_process_messages",
        "single_image_function": "unified_encode_image_to_base64",
        "config": "UnifiedProcessorConfig.high_performance()",
        "recommended_settings": {
            "max_concurrent": 20,
            "max_workers": 16,
            "memory_cache_size_mb": 1000,
        },
        "speedup": "50-200x",
        "description": "🚀 最高性能选择，统一处理本地文件和URL，支持自适应配置",
    },
    "unified_auto_detect": {
        "function": "unified_batch_process_messages",
        "single_image_function": "unified_encode_image_to_base64",
        "config": "UnifiedProcessorConfig.auto_detect()",
        "recommended_settings": "自动检测系统资源并调整",
        "speedup": "根据系统自动优化",
        "description": "🤖 智能配置选择，根据CPU和内存自动调整参数",
    },
    "unified_memory_optimized": {
        "function": "unified_batch_process_messages",
        "single_image_function": "unified_encode_image_to_base64",
        "config": "UnifiedProcessorConfig.memory_optimized()",
        "recommended_settings": {
            "max_concurrent": 6,
            "max_workers": 4,
            "memory_cache_size_mb": 200,
        },
        "speedup": "20-60x",
        "description": "💾 内存优化选择，适合资源受限环境",
    },
    "single_processing": {
        "function": "messages_preprocess",
        "description": "⚡ 推荐用于单个消息列表处理，简单快速",
    },
    "image_only": {
        "function": "encode_image_to_base64",
        "description": "🖼️ 仅处理单张图像时使用",
    },
}


def get_performance_recommendation(use_case: str = "unified_auto_detect") -> dict:
    """获取性能建议

    Args:
        use_case: 使用场景，可选：
            - 'unified_auto_detect': 自适应配置（推荐）
            - 'unified_high_performance': 最高性能配置
            - 'unified_memory_optimized': 内存优化配置
            - 'single_processing': 单个消息处理
            - 'image_only': 单张图像处理

    Returns:
        性能建议字典
    """
    return PERFORMANCE_RECOMMENDATIONS.get(
        use_case, PERFORMANCE_RECOMMENDATIONS["unified_auto_detect"]
    )


def print_performance_guide():
    """打印性能使用指南"""
    print("🚀 Sparrow VLLM Client 统一处理器使用指南")
    print("=" * 60)
    print()

    for use_case, config in PERFORMANCE_RECOMMENDATIONS.items():
        print(f"📊 {use_case.replace('_', ' ').title()}:")
        print(f"   {config['description']}")
        print(f"   函数: {config['function']}")
        if "speedup" in config:
            print(f"   性能提升: {config['speedup']}")
        if "config" in config:
            print(f"   配置: {config['config']}")
        print()

    print("💡 推荐: 优先使用 unified_auto_detect 获得最佳性能！")
    print()
    print("📝 批量消息处理示例:")
    print("```python")
    print(
        "from flexllm.processors import unified_batch_process_messages, UnifiedProcessorConfig"
    )
    print("")
    print("# 自适应配置（推荐）")
    print("config = UnifiedProcessorConfig.auto_detect()")
    print("processed = await unified_batch_process_messages(")
    print("    messages_list,")
    print("    processor_config=config,")
    print("    show_progress=True")
    print(")")
    print("```")
    print()
    print("🎯 精简架构特性:")
    print("   ✅ 统一处理本地文件和URL")
    print("   ✅ 自适应系统资源配置")
    print("   ✅ 智能缓存系统")
    print("   ✅ 性能监控统计")
