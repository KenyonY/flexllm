"""生成测试用的图片、视频、音频文件"""

import wave

import numpy as np
from PIL import Image, ImageDraw


def generate_test_image(path="test_image.png"):
    """生成一张带文字的测试图片"""
    img = Image.new("RGB", (400, 300), color=(30, 100, 200))
    draw = ImageDraw.Draw(img)
    # 画几个几何图形
    draw.rectangle([50, 50, 200, 150], fill=(255, 200, 0), outline=(255, 255, 255), width=2)
    draw.ellipse([220, 80, 370, 230], fill=(200, 50, 50), outline=(255, 255, 255), width=2)
    draw.text((100, 260), "FlexLLM Test Image", fill=(255, 255, 255))
    img.save(path)
    print(f"[OK] 生成测试图片: {path} ({img.size})")


def generate_test_audio(path="test_audio.wav", duration=2, freq=440):
    """生成一段正弦波测试音频 (WAV)"""
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # 440Hz 正弦波 + 880Hz 泛音
    audio = 0.5 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2 * t)
    audio = (audio * 32767).astype(np.int16)

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    print(f"[OK] 生成测试音频: {path} ({duration}s, {freq}Hz, {sample_rate}Hz)")


def generate_test_video(path="test_video.mp4", duration=2, fps=10):
    """生成一段简单的测试视频 (需要 opencv)"""
    try:
        import cv2
    except ImportError:
        print("[SKIP] 生成视频需要 opencv-python，尝试用 ffmpeg...")
        generate_test_video_ffmpeg(path, duration, fps)
        return

    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    total_frames = duration * fps
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # 移动的圆
        cx = int(width * (i / total_frames))
        cy = height // 2
        color = (0, 255, 0)
        cv2.circle(frame, (cx, cy), 30, color, -1)
        cv2.putText(
            frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        out.write(frame)

    out.release()
    print(f"[OK] 生成测试视频: {path} ({duration}s, {fps}fps, {width}x{height})")


def generate_test_video_ffmpeg(path="test_video.mp4", duration=2, fps=10):
    """用 ffmpeg 生成测试视频"""
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration}:size=320x240:rate={fps}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[OK] 生成测试视频 (ffmpeg): {path}")
    else:
        print(f"[FAIL] ffmpeg 生成视频失败: {result.stderr[:200]}")


if __name__ == "__main__":
    import os

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    generate_test_image()
    generate_test_audio()
    generate_test_video()
