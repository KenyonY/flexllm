"""测试 CLI 新功能: --schema, -x, -f，以及 CLI 命令回归测试"""

import json
import os
import tempfile

import pytest
import typer
from typer.testing import CliRunner

from flexllm.cli import config as config_module
from flexllm.cli.commands import _count_batch_output, register_commands
from flexllm.cli.utils import extract_code_block, parse_schema, read_file_contents
from flexllm.clients.base import ChatCompletionResult

# ========== parse_schema ==========


class TestParseSchema:
    def test_none(self):
        assert parse_schema(None) is None

    def test_json_shorthand(self):
        assert parse_schema("json") == {"type": "json_object"}
        assert parse_schema("JSON") == {"type": "json_object"}

    def test_json_schema_string(self):
        schema_str = '{"type": "object", "properties": {"name": {"type": "string"}}}'
        result = parse_schema(schema_str)
        assert result["type"] == "json_schema"
        assert result["json_schema"]["schema"]["type"] == "object"
        assert "name" in result["json_schema"]["schema"]["properties"]

    def test_passthrough_response_format(self):
        """已经是 response_format 格式的 JSON，直接返回"""
        rf = '{"type": "json_object"}'
        assert parse_schema(rf) == {"type": "json_object"}

    def test_passthrough_json_schema_format(self):
        rf = '{"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}'
        result = parse_schema(rf)
        assert result["type"] == "json_schema"
        assert result["json_schema"]["schema"]["type"] == "object"

    def test_file_reference(self):
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(schema, f)
            f.flush()
            path = f.name

        try:
            result = parse_schema(f"@{path}")
            assert result["type"] == "json_schema"
            assert result["json_schema"]["schema"] == schema
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        try:
            parse_schema("@nonexistent_file.json")
            assert False, "Should have raised Exit"
        except typer.Exit:
            pass

    def test_invalid_json(self):
        try:
            parse_schema("{invalid json}")
            assert False, "Should have raised Exit"
        except typer.Exit:
            pass


# ========== extract_code_block ==========


class TestExtractCodeBlock:
    def test_single_block(self):
        text = 'Here is code:\n```python\nprint("hello")\n```\nDone.'
        assert extract_code_block(text) == 'print("hello")'

    def test_block_without_language(self):
        text = "```\nfoo bar\n```"
        assert extract_code_block(text) == "foo bar"

    def test_multiple_blocks_returns_first(self):
        text = "```python\nfirst\n```\nsome text\n```js\nsecond\n```"
        assert extract_code_block(text) == "first"

    def test_no_code_block(self):
        text = "Just some plain text without any code blocks."
        assert extract_code_block(text) is None

    def test_multiline_block(self):
        text = "```python\ndef foo():\n    return 42\n```"
        assert extract_code_block(text) == "def foo():\n    return 42"

    def test_incomplete_fence(self):
        text = "```python\nincomplete code without closing fence"
        assert extract_code_block(text) is None


# ========== read_file_contents ==========


class TestReadFileContents:
    def test_single_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            path = f.name
        try:
            assert read_file_contents([path]) == "hello world"
        finally:
            os.unlink(path)

    def test_multiple_files(self):
        paths = []
        for content in ["file1 content", "file2 content"]:
            f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
            f.write(content)
            f.close()
            paths.append(f.name)
        try:
            result = read_file_contents(paths)
            assert "file1 content" in result
            assert "file2 content" in result
            assert "\n\n" in result
        finally:
            for p in paths:
                os.unlink(p)

    def test_file_not_found(self):
        try:
            read_file_contents(["nonexistent_file.txt"])
            assert False, "Should have raised Exit"
        except typer.Exit:
            pass


# ========== CLI 命令回归测试 ==========


def _make_app():
    app = typer.Typer()
    register_commands(app)
    return app


def _stub_config(monkeypatch, models=None, default=None, **extra):
    """注入全局配置 stub，并清掉可能干扰的环境变量"""
    for var in [
        "FLEXLLM_BASE_URL",
        "FLEXLLM_API_KEY",
        "FLEXLLM_MODEL",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
    ]:
        monkeypatch.delenv(var, raising=False)
    cfg = config_module.FlexLLMConfig.__new__(config_module.FlexLLMConfig)
    cfg.config = {"models": models or [], **extra}
    if default:
        cfg.config["default"] = default
    cfg._config_path = None
    cfg._loaded_path = None
    monkeypatch.setattr(config_module, "_config", cfg)
    return cfg


class _FakeRequestResultError:
    """模拟 chat_completions 失败时返回的 RequestResult(status='error')"""

    status = "error"
    data = {"error": "simulated failure"}


class _FakeLLMClient:
    """替身 LLMClient：记录调用参数，返回预设结果"""

    last_call = None
    result = "ok"

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def chat_completions(self, messages, return_usage=False, **kwargs):
        _FakeLLMClient.last_call = {
            "messages": messages,
            "return_usage": return_usage,
            "kwargs": kwargs,
        }
        return _FakeLLMClient.result


class TestChatSchemaPassthrough:
    """回归 bug#2：flexllm chat --schema 及配置文件模型级参数必须传给 single_chat"""

    def test_schema_and_model_params_passed(self, monkeypatch):
        _stub_config(
            monkeypatch,
            models=[
                {
                    "id": "m1",
                    "name": "m1",
                    "base_url": "http://localhost:9/v1",
                    "api_key": "EMPTY",
                    "top_p": 0.9,
                }
            ],
        )
        received = {}

        def fake_single_chat(
            message,
            model,
            base_url,
            api_key,
            system_prompt,
            model_params,
            stream,
            user_template=None,
            thinking=None,
            extract=False,
            output_format="text",
        ):
            received.update(model_params=model_params, thinking=thinking)

        monkeypatch.setattr("flexllm.cli.commands.single_chat", fake_single_chat)
        result = CliRunner().invoke(_make_app(), ["chat", "你好", "-m", "m1", "--schema", "json"])
        assert result.exit_code == 0, result.output
        assert received["model_params"]["response_format"] == {"type": "json_object"}
        assert received["model_params"]["top_p"] == 0.9  # 配置文件模型级参数传下去
        assert "temperature" in received["model_params"]
        assert "max_tokens" in received["model_params"]


class TestAskJsonUsage:
    """回归 bug#3：ask --format json 必须传 return_usage=True 并输出真实 usage/thinking"""

    def test_ask_format_json_returns_usage_and_thinking(self, monkeypatch):
        _stub_config(
            monkeypatch,
            models=[
                {"id": "m1", "name": "m1", "base_url": "http://localhost:9/v1", "api_key": "EMPTY"}
            ],
        )
        _FakeLLMClient.result = ChatCompletionResult(
            content="回答",
            usage={"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
            reasoning_content="思考过程",
        )
        monkeypatch.setattr("flexllm.LLMClient", _FakeLLMClient)
        result = CliRunner().invoke(_make_app(), ["ask", "你好", "-m", "m1", "--format", "json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output.strip().splitlines()[-1])
        assert payload["content"] == "回答"
        assert payload["thinking"] == "思考过程"
        assert payload["usage"]["total_tokens"] == 8
        assert _FakeLLMClient.last_call["return_usage"] is True


class TestBatchJsonCountFromFile:
    """回归 bug#4：batch --format json 统计以输出 JSONL 文件为准（断点续传场景）"""

    def test_count_success_from_output_file(self, tmp_path):
        out = tmp_path / "out.jsonl"
        lines = [
            {"index": 0, "status": "success", "output": "a"},
            {"index": 1, "status": "error", "output": None, "error": "boom"},
            {"index": 2, "status": "success", "output": "c"},
            # index 1 重试成功（未 compact 的重复行，success 优先）
            {"index": 1, "status": "success", "output": "b"},
            # 越界 index 不计入
            {"index": 99, "status": "success", "output": "x"},
        ]
        out.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
        assert _count_batch_output(str(out), total=4) == 3

    def test_missing_file_counts_zero(self, tmp_path):
        assert _count_batch_output(str(tmp_path / "nope.jsonl"), total=3) == 0


class TestBatchEffectiveModel:
    """回归 bug#5：batch 的 system/user_template 用 effective_model（含 batch.model）解析"""

    def test_batch_model_system_applied(self, monkeypatch, tmp_path):
        _stub_config(
            monkeypatch,
            models=[
                {
                    "id": "m1",
                    "name": "m1",
                    "base_url": "http://localhost:9/v1",
                    "api_key": "EMPTY",
                    "system": "BATCH_MODEL_SYSTEM",
                }
            ],
            batch={"model": "m1"},
        )
        input_file = tmp_path / "in.jsonl"
        input_file.write_text('{"q": "hello"}\n')
        out_file = tmp_path / "out.jsonl"
        result = CliRunner().invoke(
            _make_app(), ["batch", str(input_file), "-o", str(out_file), "--dry-run"]
        )
        assert result.exit_code == 10, result.output  # dry-run 退出码
        assert "BATCH_MODEL_SYSTEM" in result.output


class TestChatErrorHandling:
    """回归 bug#6：chat 失败时不把 RequestResult repr 当回复，且退出码非零"""

    def test_single_chat_error_result_exits_nonzero(self, monkeypatch, capsys):
        from flexllm.cli.chat_helpers import single_chat

        _FakeLLMClient.result = _FakeRequestResultError()
        monkeypatch.setattr("flexllm.LLMClient", _FakeLLMClient)

        with pytest.raises(typer.Exit) as exc_info:
            single_chat(
                "你好",
                "m1",
                "http://localhost:9/v1",
                "EMPTY",
                None,
                {"temperature": 0.7, "max_tokens": 128},
                stream=False,
                output_format="text",
            )
        assert exc_info.value.exit_code != 0
        captured = capsys.readouterr()
        assert "RequestResult" not in captured.out  # 不把 repr 当回复打印

    def test_single_chat_json_error_exits_nonzero(self, monkeypatch):
        from flexllm.cli.chat_helpers import single_chat

        _FakeLLMClient.result = _FakeRequestResultError()
        monkeypatch.setattr("flexllm.LLMClient", _FakeLLMClient)

        with pytest.raises(typer.Exit) as exc_info:
            single_chat(
                "你好",
                "m1",
                "http://localhost:9/v1",
                "EMPTY",
                None,
                {"temperature": 0.7, "max_tokens": 128},
                stream=False,
                output_format="json",
            )
        assert exc_info.value.exit_code != 0


class TestTestCommandExitCode:
    """回归 bug#7：flexllm test 失败时退出码非零"""

    def test_json_mode_connection_failure_nonzero_exit(self, monkeypatch):
        _stub_config(monkeypatch)
        # 连接不上的端口（RFC 5737 保留地址不可路由，用 localhost 关闭端口更快）
        result = CliRunner().invoke(
            _make_app(),
            [
                "test",
                "--json",
                "--base-url",
                "http://127.0.0.1:9",
                "--api-key",
                "EMPTY",
                "-m",
                "m1",
                "--timeout",
                "2",
            ],
        )
        assert result.exit_code != 0

    def test_claude_provider_reports_unsupported(self, monkeypatch):
        """回归 bug#16：claude/gemini 配置诚实报告不支持，而不是打出误导性连接失败"""
        _stub_config(
            monkeypatch,
            models=[{"id": "c1", "name": "c1", "provider": "claude", "api_key": "sk-ant-x"}],
        )
        result = CliRunner().invoke(_make_app(), ["test", "-m", "c1"])
        assert result.exit_code == 2  # INVALID_ARGS


class TestPoolModelGuard:
    """回归 bug#12：pool 型模型走 ask/chat 时明确报错，不发非法请求体"""

    def test_resolve_pool_model_errors(self, monkeypatch):
        from flexllm.cli.utils import resolve_model_config

        _stub_config(
            monkeypatch,
            models=[
                {
                    "id": "pool",
                    "name": "pool",
                    "endpoints": [{"base_url": "http://a/v1"}, {"base_url": "http://b/v1"}],
                }
            ],
        )
        with pytest.raises(typer.Exit) as exc_info:
            resolve_model_config("pool")
        assert exc_info.value.exit_code == 2
