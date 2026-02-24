"""
端到端测试：验证 image_url / video_url / input_audio 预处理 + 模型调用

使用 OpenRouter 的 gemini-3-flash-preview 模型测试。
用法: python tests/e2e/media_test/test_media_e2e.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from flexllm import LLMClient
from flexllm.cli.utils import resolve_model_config
from flexllm.msg_processors import messages_preprocess

MEDIA_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "or-gemini-3-flash"


def get_client():
    model_id, base_url, api_key = resolve_model_config(MODEL_NAME)
    return LLMClient(model=model_id, base_url=base_url, api_key=api_key)


async def test_image():
    """测试 image_url 预处理 + 模型调用"""
    print("=" * 60)
    print("[TEST] image_url 预处理")
    print("=" * 60)

    image_path = os.path.join(MEDIA_DIR, "test_image.png")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_path}},
                {"type": "text", "text": "请用一句话描述这张图片中的内容。"},
            ],
        }
    ]

    print(f"  原始 image_url: {image_path}")
    processed = await messages_preprocess(messages)
    url = processed[0]["content"][0]["image_url"]["url"]
    print(f"  预处理后: data URI (长度={len(url)}, 前缀={url[:40]}...)")

    client = get_client()
    response = await client.chat_completions(processed)
    print(f"  模型回复: {response}")
    print()


async def test_video():
    """测试 video_url 预处理 + 模型调用"""
    print("=" * 60)
    print("[TEST] video_url 预处理")
    print("=" * 60)

    video_path = os.path.join(MEDIA_DIR, "test_video.mp4")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_path}},
                {"type": "text", "text": "请用一句话描述这个视频中发生了什么。"},
            ],
        }
    ]

    print(f"  原始 video_url: {video_path}")
    processed = await messages_preprocess(messages)
    url = processed[0]["content"][0]["video_url"]["url"]
    print(f"  预处理后: data URI (长度={len(url)}, 前缀={url[:40]}...)")

    client = get_client()
    response = await client.chat_completions(processed)
    print(f"  模型回复: {response}")
    print()


async def test_input_audio():
    """测试 input_audio 预处理 + 模型调用"""
    print("=" * 60)
    print("[TEST] input_audio 预处理")
    print("=" * 60)

    audio_path = os.path.join(MEDIA_DIR, "test_audio.wav")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_path, "format": "wav"},
                },
                {"type": "text", "text": "请描述你听到的音频内容。"},
            ],
        }
    ]

    print(f"  原始 input_audio.data: {audio_path}")
    processed = await messages_preprocess(messages)
    data = processed[0]["content"][0]["input_audio"]["data"]
    print(f"  预处理后: 纯 base64 (长度={len(data)}, 前20字符={data[:20]}...)")

    client = get_client()
    response = await client.chat_completions(processed)
    print(f"  模型回复: {response}")
    print()


async def main():
    print(f"\n使用模型: {MODEL_NAME}")
    print(f"媒体目录: {MEDIA_DIR}\n")

    results = {}
    for name, test_fn in [
        ("image_url", test_image),
        ("video_url", test_video),
        ("input_audio", test_input_audio),
    ]:
        try:
            await test_fn()
            results[name] = "PASS"
        except Exception as e:
            print(f"  [ERROR] {e}")
            results[name] = f"FAIL: {e}"
            print()

    print("=" * 60)
    print("测试结果汇总:")
    for name, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} {name}: {result}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
