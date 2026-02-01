"""ServeServer 单元测试"""

import pytest

from flexllm.serve import ServeConfig, ServeServer


@pytest.fixture
def config():
    return ServeConfig(
        model="test-model",
        base_url="http://localhost:8001/v1",
        system_prompt="你是一个助手",
        user_template="[INST]{content}[/INST]",
        temperature=0.5,
        max_tokens=1024,
        thinking=True,
    )


@pytest.fixture
def server(config):
    return ServeServer(config)


@pytest.fixture
def config_minimal():
    return ServeConfig(
        model="test-model",
        base_url="http://localhost:8001/v1",
    )


@pytest.fixture
def server_minimal(config_minimal):
    return ServeServer(config_minimal)


class TestBuildMessages:
    def test_with_system_and_template(self, server):
        messages = server._build_messages("你好")
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "你是一个助手"}
        assert messages[1] == {"role": "user", "content": "[INST]你好[/INST]"}

    def test_without_system(self, server_minimal):
        messages = server_minimal._build_messages("你好")
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "你好"}

    def test_without_template(self):
        config = ServeConfig(model="m", base_url="http://x", system_prompt="sys")
        s = ServeServer(config)
        messages = s._build_messages("hello")
        assert len(messages) == 2
        assert messages[1]["content"] == "hello"

    def test_template_with_special_chars(self):
        config = ServeConfig(model="m", base_url="http://x", user_template="<s>{content}</s>")
        s = ServeServer(config)
        messages = s._build_messages("test input")
        assert messages[0]["content"] == "<s>test input</s>"


class TestGetKwargs:
    def test_server_defaults(self, server):
        kwargs = server._get_kwargs({})
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 1024
        assert kwargs["thinking"] is True

    def test_request_override(self, server):
        kwargs = server._get_kwargs({"temperature": 0.9, "max_tokens": 2048})
        assert kwargs["temperature"] == 0.9
        assert kwargs["max_tokens"] == 2048
        assert kwargs["thinking"] is True  # 不被请求覆盖

    def test_minimal_config(self, server_minimal):
        kwargs = server_minimal._get_kwargs({})
        assert kwargs == {}

    def test_partial_override(self, server):
        kwargs = server._get_kwargs({"temperature": 0.1})
        assert kwargs["temperature"] == 0.1
        assert kwargs["max_tokens"] == 1024


class TestParseResult:
    def test_normal_response(self):
        raw = {
            "choices": [{"message": {"content": "回答内容"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        result = ServeServer._parse_result(raw)
        assert result["content"] == "回答内容"
        assert result["thinking"] is None
        assert result["usage"]["total_tokens"] == 30

    def test_response_with_reasoning(self):
        raw = {
            "choices": [{"message": {"content": "答案", "reasoning": "思考过程"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        result = ServeServer._parse_result(raw)
        assert result["content"] == "答案"
        assert result["thinking"] == "思考过程"

    def test_response_with_think_tags(self):
        raw = {
            "choices": [{"message": {"content": "<think>思考中</think>最终答案"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        result = ServeServer._parse_result(raw)
        assert result["content"] == "最终答案"
        assert result["thinking"] == "思考中"

    def test_response_without_usage(self):
        raw = {
            "choices": [{"message": {"content": "内容"}}],
        }
        result = ServeServer._parse_result(raw)
        assert result["content"] == "内容"
        assert result["thinking"] is None
        assert result["usage"] is None

    def test_empty_thinking(self):
        raw = {
            "choices": [{"message": {"content": "纯内容", "reasoning": ""}}],
            "usage": {},
        }
        result = ServeServer._parse_result(raw)
        assert result["content"] == "纯内容"
        assert result["thinking"] is None  # 空字符串转为 None


class TestParseThinking:
    def test_true(self):
        from flexllm.__main__ import _parse_thinking

        assert _parse_thinking("true") is True
        assert _parse_thinking("True") is True

    def test_false(self):
        from flexllm.__main__ import _parse_thinking

        assert _parse_thinking("false") is False

    def test_levels(self):
        from flexllm.__main__ import _parse_thinking

        assert _parse_thinking("low") == "low"
        assert _parse_thinking("medium") == "medium"
        assert _parse_thinking("high") == "high"
        assert _parse_thinking("minimal") == "minimal"

    def test_integer(self):
        from flexllm.__main__ import _parse_thinking

        assert _parse_thinking("1024") == 1024

    def test_none(self):
        from flexllm.__main__ import _parse_thinking

        assert _parse_thinking(None) is None

    def test_invalid(self):
        from flexllm.__main__ import _parse_thinking

        with pytest.raises(SystemExit):
            _parse_thinking("invalid_value")
