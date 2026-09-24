"""tool 结果带图：统一格式 → 各协议转换出口的适配，以及 vision=False 降级。"""

import copy

import pytest

from flexllm import ClaudeClient, GeminiClient, LLMClient, OpenAIClient
from flexllm.cli.config import FlexLLMConfig, model_client_kwargs
from flexllm.clients.message_images import (
    TOOL_IMAGE_PLACEHOLDER,
    TOOL_IMAGES_HEADER,
    VISION_OMITTED_TEXT,
)


def _img(data: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def _assistant_calls(*ids) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": i, "type": "function", "function": {"name": "read", "arguments": "{}"}}
            for i in ids
        ],
    }


def _parallel_round() -> list[dict]:
    """一轮三个并行工具，只有第 1、3 个带图"""
    return [
        {"role": "user", "content": "看看这些文件"},
        _assistant_calls("c1", "c2", "c3"),
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": [{"type": "text", "text": "Read a.png"}, _img("AAA")],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "plain text"},
        {"role": "tool", "tool_call_id": "c3", "content": [_img("CCC")]},
    ]


def _openai(**kw):
    return OpenAIClient(base_url="http://x/v1", model="m", **kw)


def _claude(**kw):
    return ClaudeClient(api_key="k", model="m", **kw)


def _gemini(**kw):
    return GeminiClient(api_key="k", model="m", **kw)


class TestClaudeToolImages:
    def test_text_and_image_become_tool_result_blocks(self):
        body = _claude()._build_request_body(_parallel_round(), "m")
        tool_turns = body["messages"][2:]
        assert [t["content"][0]["tool_use_id"] for t in tool_turns] == ["c1", "c2", "c3"]
        assert tool_turns[0]["content"][0]["content"] == [
            {"type": "text", "text": "Read a.png"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "AAA"},
            },
        ]
        assert tool_turns[1]["content"][0]["content"] == "plain text"

    def test_image_only_gets_placeholder_text(self):
        body = _claude()._build_request_body(_parallel_round(), "m")
        blocks = body["messages"][4]["content"][0]["content"]
        assert blocks[0] == {"type": "text", "text": TOOL_IMAGE_PLACEHOLDER}
        assert blocks[1]["type"] == "image"

    def test_empty_text_with_image_gets_placeholder(self):
        """Anthropic 拒绝空 text 块，空文本等同于没有文字"""
        content = [{"type": "text", "text": ""}, _img("A")]
        msgs = [_assistant_calls("c1"), {"role": "tool", "tool_call_id": "c1", "content": content}]
        blocks = _claude()._build_request_body(msgs, "m")["messages"][1]["content"][0]["content"]
        assert [b["type"] for b in blocks] == ["text", "image"]
        assert blocks[0]["text"] == TOOL_IMAGE_PLACEHOLDER

    def test_text_only_list_passes_through_unchanged(self):
        """纯文本块列表（含 cache_control 等额外字段）保持原样透传"""
        content = [{"type": "text", "text": "r", "cache_control": {"type": "ephemeral"}}]
        msgs = [_assistant_calls("c1"), {"role": "tool", "tool_call_id": "c1", "content": content}]
        body = _claude()._build_request_body(msgs, "m")
        assert body["messages"][1]["content"][0]["content"] == content


class TestOpenAIToolImages:
    def test_images_move_after_the_whole_tool_run(self):
        msgs = _openai()._build_request_body(_parallel_round(), "m")["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "tool", "tool", "user"]
        assert [m["content"] for m in msgs[2:5]] == [
            "Read a.png",
            "plain text",
            TOOL_IMAGE_PLACEHOLDER,
        ]
        assert [m["tool_call_id"] for m in msgs[2:5]] == ["c1", "c2", "c3"]
        assert msgs[5]["content"] == [
            {"type": "text", "text": TOOL_IMAGES_HEADER},
            _img("AAA"),
            _img("CCC"),
        ]

    def test_each_tool_run_gets_its_own_attachment_message(self):
        msgs = [
            _assistant_calls("c1"),
            {"role": "tool", "tool_call_id": "c1", "content": [_img("A")]},
            _assistant_calls("c2"),
            {"role": "tool", "tool_call_id": "c2", "content": [_img("B")]},
        ]
        out = _openai()._build_request_body(msgs, "m")["messages"]
        assert [m["role"] for m in out] == [
            "assistant",
            "tool",
            "user",
            "assistant",
            "tool",
            "user",
        ]
        assert out[2]["content"][1] == _img("A")
        assert out[5]["content"][1] == _img("B")

    def test_text_only_tool_list_is_not_rewritten(self):
        content = [{"type": "text", "text": "r"}]
        msgs = [_assistant_calls("c1"), {"role": "tool", "tool_call_id": "c1", "content": content}]
        out = _openai()._build_request_body(msgs, "m")["messages"]
        assert out[1]["content"] == content
        assert len(out) == 2


class TestGeminiToolImages:
    def test_tool_image_rides_inside_function_response(self):
        """三条结果并进一条 user 消息；图片挂在各自 functionResponse.parts 上"""
        body = _gemini()._build_request_body(_parallel_round(), "m")
        responses = [p["functionResponse"] for p in body["contents"][-1]["parts"]]
        assert [r["id"] for r in responses] == ["c1", "c2", "c3"]
        assert [r["response"] for r in responses] == [
            {"result": "Read a.png"},
            {"result": "plain text"},
            {"result": ""},
        ]
        media = [[p["inline_data"]["data"] for p in r.get("parts", [])] for r in responses]
        assert media == [["AAA"], [], ["CCC"]]


ALL_CLIENTS = [_openai, _claude, _gemini]


@pytest.mark.parametrize("make", ALL_CLIENTS)
def test_stream_and_non_stream_convert_messages_identically(make):
    client = make()
    body = client._build_request_body(_parallel_round(), "m", stream=False)
    stream_body = client._build_request_body(_parallel_round(), "m", stream=True)
    key = "contents" if "contents" in body else "messages"
    assert body[key] == stream_body[key]


@pytest.mark.parametrize("make", ALL_CLIENTS)
def test_caller_messages_are_not_mutated(make):
    msgs = _parallel_round()
    snapshot = copy.deepcopy(msgs)
    make()._build_request_body(msgs, "m")
    make(vision=False)._build_request_body(msgs, "m")
    assert msgs == snapshot


class TestVisionDowngrade:
    def _history(self):
        return [{"role": "user", "content": [{"type": "text", "text": "看图"}, _img("U")]}] + (
            _parallel_round()[1:]
        )

    def _assert_no_images(self, body):
        text = repr(body)
        for data in ("U", "AAA", "CCC"):
            assert f"'{data}'" not in text
        assert VISION_OMITTED_TEXT in text

    def test_openai(self):
        msgs = _openai(vision=False)._build_request_body(self._history(), "m")["messages"]
        self._assert_no_images(msgs)
        # tool 消息保持字符串（部分兼容后端不收列表形式的 tool content）
        assert [m["content"] for m in msgs[2:5]] == [
            "Read a.png",
            "plain text",
            TOOL_IMAGE_PLACEHOLDER,
        ]
        assert msgs[5]["content"][1:] == [{"type": "text", "text": VISION_OMITTED_TEXT}] * 2

    def test_claude(self):
        self._assert_no_images(_claude(vision=False)._build_request_body(self._history(), "m"))

    def test_gemini(self):
        self._assert_no_images(_gemini(vision=False)._build_request_body(self._history(), "m"))

    def test_claude_empty_text_dropped_when_image_omitted(self):
        """占位替换后不能留下空 text 块（Anthropic 拒绝）"""
        content = [{"type": "text", "text": ""}, _img("A")]
        msgs = [_assistant_calls("c1"), {"role": "tool", "tool_call_id": "c1", "content": content}]
        body = _claude(vision=False)._build_request_body(msgs, "m")
        assert body["messages"][1]["content"][0]["content"] == [
            {"type": "text", "text": VISION_OMITTED_TEXT}
        ]

    def test_default_is_vision_enabled(self):
        assert _openai().vision is True
        assert LLMClient(base_url="http://x/v1", model="m").vision is True

    def test_pool_exposes_vision(self):
        assert LLMClient(base_url="http://x/v1", model="m", vision=False).vision is False
        pool = LLMClient(
            endpoints=[{"base_url": "http://a/v1"}, {"base_url": "http://b/v1"}],
            model="m",
            vision=False,
        )
        assert pool.vision is False


class TestVisionConfig:
    def test_model_client_kwargs_passes_vision(self):
        entry = {"id": "m", "base_url": "http://x/v1", "vision": False}
        assert model_client_kwargs(entry)["vision"] is False

    def test_vision_must_be_bool(self):
        with pytest.raises(ValueError, match="vision"):
            model_client_kwargs({"id": "m", "base_url": "http://x/v1", "vision": "no"})

    def test_vision_is_not_a_request_param(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(
            "default: m\nmodels:\n  - id: m\n    base_url: http://x/v1\n    vision: false\n"
            "    temperature: 0.1\n"
        )
        assert FlexLLMConfig(path).get_model_params("m") == {"temperature": 0.1}
        assert LLMClient.from_config(str(path)).vision is False


def test_string_tool_content_body_is_unchanged():
    """回归锁：纯字符串 tool content 的请求体保持改动前形状"""
    msgs = [
        {"role": "user", "content": "hi"},
        _assistant_calls("c1"),
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
    ]
    assert _openai()._build_request_body(msgs, "m")["messages"] == msgs
    claude_tool = _claude()._build_request_body(msgs, "m")["messages"][2]
    assert claude_tool == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "r1"}],
    }
