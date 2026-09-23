"""消息中图片块的发送前改写（纯函数，不修改调用方传入的 messages）。

统一消息格式允许 role=tool 的 content 是块列表（text + image_url），各协议在转换出口
按自身能力适配：
- omit_images: 模型不支持视觉时，把所有图片块替换为占位文本，避免带图历史让会话永久 400。
- move_tool_images_to_user: OpenAI Chat Completions 的 tool 消息只收文本，把图片挪到
  整串 tool 消息之后的一条 user 消息里。
"""

IMAGE_PART_TYPES = {"image_url", "image"}
VISION_OMITTED_TEXT = "[image omitted: model does not support vision]"
TOOL_IMAGE_PLACEHOLDER = "(see attached image)"
TOOL_IMAGES_HEADER = "Attached image(s) from tool result:"


def _is_image_part(part) -> bool:
    return isinstance(part, dict) and part.get("type") in IMAGE_PART_TYPES


def _is_text_part(part) -> bool:
    return isinstance(part, str) or (isinstance(part, dict) and part.get("type") == "text")


def has_non_text_parts(content) -> bool:
    """content 是块列表且含非文本块（图片/音视频等）"""
    return isinstance(content, list) and not all(_is_text_part(p) for p in content)


def _text_of(part) -> str:
    return part if isinstance(part, str) else part.get("text", "")


def omit_images(messages: list[dict]) -> list[dict]:
    """把所有消息中的图片块替换为占位文本；不含图片的消息原样复用。

    被改写的消息同时丢掉空文本块（Anthropic 拒绝空 text 块），占位保证内容非空。
    """
    result = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list) and any(_is_image_part(p) for p in content):
            content = [
                {"type": "text", "text": VISION_OMITTED_TEXT} if _is_image_part(p) else p
                for p in content
                if not (_is_text_part(p) and not _text_of(p))
            ]
            msg = {**msg, "content": content}
        result.append(msg)
    return result


def move_tool_images_to_user(messages: list[dict]) -> list[dict]:
    """OpenAI Chat Completions 适配：tool 消息只留文本，非文本块挪到整串 tool 消息之后。

    附件消息必须插在连续 tool 消息整串之后——assistant(tool_calls) 与其 tool 结果之间
    不能插入其他消息。一串里只插一条 user 消息，附件按 tool 结果顺序排列。
    """
    result = []
    pending: list = []  # 当前这串 tool 消息里挪出来的块

    def flush():
        if pending:
            result.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": TOOL_IMAGES_HEADER}, *pending],
                }
            )
            pending.clear()

    for msg in messages:
        if msg.get("role") != "tool":
            flush()
            result.append(msg)
            continue
        content = msg.get("content")
        if not has_non_text_parts(content):
            result.append(msg)
            continue
        text = "\n".join(_text_of(p) for p in content if _is_text_part(p))
        pending.extend(p for p in content if not _is_text_part(p))
        result.append({**msg, "content": text or TOOL_IMAGE_PLACEHOLDER})
    flush()
    return result
