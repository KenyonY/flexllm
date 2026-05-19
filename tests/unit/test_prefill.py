"""Prefill 单元测试

只覆盖纯函数逻辑(不打 mock server):
1. OpenAIClient._build_request_body 检测末尾 assistant -> 自动设 continue_final_message + add_generation_prompt
2. LLMClientBase._trailing_assistant_prefix 工具方法
3. CLI utils.convert_to_messages 识别 jsonl 顶层 prefix 字段
"""

from flexllm.cli.utils import convert_to_messages, detect_input_format
from flexllm.clients import OpenAIClient
from flexllm.clients.base import LLMClientBase


class TestOpenAIBuildBody:
    def _client(self):
        return OpenAIClient(base_url="http://x/v1", api_key="k", model="m")

    def test_no_assistant_tail_no_prefill_flags(self):
        c = self._client()
        body = c._build_request_body([{"role": "user", "content": "hi"}], model="m", stream=False)
        assert "continue_final_message" not in body
        assert "add_generation_prompt" not in body
        c.close()

    def test_assistant_tail_auto_prefill(self):
        c = self._client()
        body = c._build_request_body(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Sure, "},
            ],
            model="m",
            stream=False,
        )
        assert body["continue_final_message"] is True
        assert body["add_generation_prompt"] is False
        # 原 messages 透传,不应被改写
        assert body["messages"][-1]["role"] == "assistant"
        assert body["messages"][-1]["content"] == "Sure, "
        c.close()

    def test_user_can_override_prefill_flags(self):
        """用户显式传入 continue_final_message 时,不被自动检测覆盖"""
        c = self._client()
        body = c._build_request_body(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Sure, "},
            ],
            model="m",
            stream=False,
            continue_final_message=False,
            add_generation_prompt=True,
        )
        assert body["continue_final_message"] is False
        assert body["add_generation_prompt"] is True
        c.close()

    def test_empty_messages_no_flags(self):
        c = self._client()
        body = c._build_request_body([], model="m", stream=False)
        assert "continue_final_message" not in body
        c.close()


class TestTrailingAssistantPrefix:
    def test_empty(self):
        assert LLMClientBase._trailing_assistant_prefix([]) is None

    def test_no_assistant_tail(self):
        assert LLMClientBase._trailing_assistant_prefix([{"role": "user", "content": "hi"}]) is None

    def test_assistant_tail_str(self):
        assert (
            LLMClientBase._trailing_assistant_prefix(
                [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "prefix-X"},
                ]
            )
            == "prefix-X"
        )

    def test_assistant_tail_non_str_content_returns_none(self):
        """assistant content 是 list (多模态等) 时不当作 prefix 返回"""
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]
        assert LLMClientBase._trailing_assistant_prefix(msgs) is None


class TestCliConvertToMessages:
    def test_simple_with_prefix_field(self):
        record = {"q": "what is 2+2?", "prefix": "Answer: not 4, but "}
        fmt, fields = detect_input_format(record)
        msgs, meta = convert_to_messages(record, fmt, fields)
        # 末尾应是 assistant message,content 等于 prefix
        assert msgs[-1] == {"role": "assistant", "content": "Answer: not 4, but "}
        # prefix 被消耗后不应进入 metadata
        assert "prefix" not in meta

    def test_simple_without_prefix(self):
        record = {"q": "hello"}
        fmt, fields = detect_input_format(record)
        msgs, _ = convert_to_messages(record, fmt, fields)
        assert msgs[-1]["role"] == "user"

    def test_openai_chat_existing_assistant_tail_not_duplicated(self):
        """openai_chat 已自带 assistant 结尾时,即使有 prefix 字段也不重复追加"""
        record = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Already prefilled"},
            ],
            "prefix": "Should-not-appear",
        }
        fmt, fields = detect_input_format(record)
        msgs, _ = convert_to_messages(record, fmt, fields)
        # 只有两条,且末尾 assistant 内容是原始的,不是 prefix 字段值
        assert len(msgs) == 2
        assert msgs[-1]["content"] == "Already prefilled"

    def test_openai_chat_user_tail_with_prefix_field(self):
        """openai_chat 末尾是 user 时, prefix 字段被识别并追加为 assistant message"""
        record = {
            "messages": [{"role": "user", "content": "hi"}],
            "prefix": "Sure! ",
        }
        fmt, fields = detect_input_format(record)
        msgs, _ = convert_to_messages(record, fmt, fields)
        assert len(msgs) == 2
        assert msgs[-1] == {"role": "assistant", "content": "Sure! "}

    def test_empty_prefix_ignored(self):
        record = {"q": "hi", "prefix": ""}
        fmt, fields = detect_input_format(record)
        msgs, _ = convert_to_messages(record, fmt, fields)
        assert msgs[-1]["role"] == "user"
