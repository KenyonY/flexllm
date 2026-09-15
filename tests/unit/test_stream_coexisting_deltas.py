"""同一帧 delta 里 reasoning / tool_calls / content 共存时，三者都不能被吞掉

vLLM 在"思考结束、正文开始"的那一帧会把思考的最后一个 token 和正文的第一个 token
放进同一个 delta：{"reasoning": "\\n", "content": "\\n\\n{\\""}。
流式循环曾经在 thinking 分支 continue，导致正文首 token 连同整帧一起丢失，
聚合出来的 JSON 缺头（truth `{"text": "abc"}` → got `text": "abc"}`）。
tool_calls 分支同理。
"""

import json

from flexllm.clients.openai import OpenAIClient

from .test_extra_passthrough import ScriptedServer, _sse

MESSAGES = [{"role": "user", "content": "hi"}]


def _chunk(delta: dict, finish_reason=None) -> dict:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


async def _collect(lines, **kwargs):
    async with ScriptedServer(stream_lines=lines) as server:
        client = OpenAIClient(base_url=server.base_url, api_key="k", model="m")
        chunks = [c async for c in client.chat_completions_stream(MESSAGES, **kwargs)]
        await client.aclose()
    return chunks


class TestReasoningAndContentSameFrame:
    async def test_content_survives_on_the_switchover_frame(self):
        """复现用例：正文首 token 与思考尾 token 同帧，聚合结果必须是完整 JSON"""
        lines = [
            _sse(_chunk({"reasoning": "let me think"})),
            _sse(_chunk({"reasoning": "\n", "content": '\n\n{"'})),
            _sse(_chunk({"content": 'text": "abc"}'})),
            "data: [DONE]\n\n",
        ]
        chunks = await _collect(lines, return_usage=True)

        content = "".join(c["content"] for c in chunks if c["type"] == "content")
        assert json.loads(content) == {"text": "abc"}

        thinking = "".join(c["content"] for c in chunks if c["type"] == "thinking")
        assert thinking == "let me think\n"

    async def test_plain_text_mode_closes_think_tag_on_the_same_frame(self):
        """return_usage=False 走 <think> 包裹路径，同帧切换也不能丢正文"""
        lines = [
            _sse(_chunk({"reasoning": "think"})),
            _sse(_chunk({"reasoning": "!", "content": "answer"})),
            "data: [DONE]\n\n",
        ]
        chunks = await _collect(lines)

        assert "".join(chunks) == "<think>\nthink!</think>answer"

    async def test_assistant_message_keeps_both_streams(self):
        lines = [
            _sse(_chunk({"reasoning": "r1"})),
            _sse(_chunk({"reasoning": "r2", "content": "c1"})),
            "data: [DONE]\n\n",
        ]
        chunks = await _collect(lines, return_usage=True)

        msgs = [c for c in chunks if c["type"] == "assistant_message"]
        assert len(msgs) == 1
        assert msgs[0]["message"]["reasoning_content"] == "r1r2"
        assert msgs[0]["message"]["content"] == "c1"


class TestToolCallAndContentSameFrame:
    async def test_content_survives_alongside_tool_call_delta(self):
        tc = [
            {
                "index": 0,
                "id": "t1",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }
        ]
        lines = [
            _sse(_chunk({"content": "before"})),
            _sse(_chunk({"tool_calls": tc, "content": "after"})),
            "data: [DONE]\n\n",
        ]
        chunks = await _collect(lines, return_usage=True)

        content = "".join(c["content"] for c in chunks if c["type"] == "content")
        assert content == "beforeafter"
        assert [c for c in chunks if c["type"] == "tool_call_delta"]

    async def test_tool_call_survives_alongside_reasoning(self):
        """思考尾帧直接带 tool_call 时，工具调用不能被 thinking 分支吞掉"""
        tc = [
            {
                "index": 0,
                "id": "t1",
                "type": "function",
                "function": {"name": "f", "arguments": '{"a":1}'},
            }
        ]
        lines = [
            _sse(_chunk({"reasoning": "r", "tool_calls": tc})),
            "data: [DONE]\n\n",
        ]
        chunks = await _collect(lines, return_usage=True)

        msgs = [c for c in chunks if c["type"] == "assistant_message"]
        assert len(msgs) == 1
        assert msgs[0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"a":1}'
