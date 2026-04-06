"""测试 CLI 错误处理模块"""

import json

import pytest

from flexllm.cli.errors import (
    _ERROR_EXIT_MAP,
    ErrorType,
    ExitCode,
    cli_error,
    dry_run_output,
)


class TestExitCode:
    def test_values(self):
        assert ExitCode.SUCCESS == 0
        assert ExitCode.ERROR == 1
        assert ExitCode.USAGE == 2
        assert ExitCode.NOT_FOUND == 3
        assert ExitCode.AUTH == 4
        assert ExitCode.CONFLICT == 5
        assert ExitCode.NETWORK == 6
        assert ExitCode.DEPENDENCY == 7
        assert ExitCode.IO_ERROR == 8
        assert ExitCode.DRY_RUN == 10


class TestErrorType:
    def test_all_types_mapped(self):
        for et in ErrorType:
            assert et in _ERROR_EXIT_MAP, f"{et} 未映射到退出码"

    def test_exit_code_mapping(self):
        assert _ERROR_EXIT_MAP[ErrorType.INVALID_ARGS] == ExitCode.USAGE
        assert _ERROR_EXIT_MAP[ErrorType.NOT_FOUND] == ExitCode.NOT_FOUND
        assert _ERROR_EXIT_MAP[ErrorType.AUTH_FAILED] == ExitCode.AUTH
        assert _ERROR_EXIT_MAP[ErrorType.NETWORK_ERROR] == ExitCode.NETWORK
        assert _ERROR_EXIT_MAP[ErrorType.IO_ERROR] == ExitCode.IO_ERROR
        assert _ERROR_EXIT_MAP[ErrorType.GENERAL] == ExitCode.ERROR


class TestCliError:
    def test_tty_output(self, capsys, monkeypatch):
        monkeypatch.setattr(
            "sys.stderr",
            type(
                "FakeTTY",
                (),
                {
                    "isatty": lambda self: True,
                    "write": lambda self, s: capsys.readouterr,
                    "fileno": lambda self: 2,
                },
            )(),
        )
        # 无法简单 mock isatty，改用 capsys 捕获
        # 直接测试退出码
        from click.exceptions import Exit

        with pytest.raises(Exit) as exc_info:
            cli_error(ErrorType.INVALID_ARGS, "测试错误")
        assert exc_info.value.exit_code == ExitCode.USAGE

    def test_exit_code_not_found(self):
        from click.exceptions import Exit

        with pytest.raises(Exit) as exc_info:
            cli_error(ErrorType.NOT_FOUND, "资源未找到")
        assert exc_info.value.exit_code == ExitCode.NOT_FOUND

    def test_exit_code_auth(self):
        from click.exceptions import Exit

        with pytest.raises(Exit) as exc_info:
            cli_error(ErrorType.AUTH_FAILED, "认证失败")
        assert exc_info.value.exit_code == ExitCode.AUTH

    def test_exit_code_network(self):
        from click.exceptions import Exit

        with pytest.raises(Exit) as exc_info:
            cli_error(ErrorType.NETWORK_ERROR, "连接失败")
        assert exc_info.value.exit_code == ExitCode.NETWORK

    def test_non_tty_json_output(self, capsys, monkeypatch):
        """非 TTY 模式下输出 JSON 到 stderr"""
        import io

        fake_stderr = io.StringIO()
        fake_stderr.isatty = lambda: False
        monkeypatch.setattr("sys.stderr", fake_stderr)

        from click.exceptions import Exit

        with pytest.raises(Exit):
            cli_error(ErrorType.INVALID_ARGS, "测试错误", suggestion="修复建议", retryable=True)

        output = fake_stderr.getvalue()
        data = json.loads(output)
        assert data["error"] == "invalid_args"
        assert data["message"] == "测试错误"
        assert data["suggestion"] == "修复建议"
        assert data["retryable"] is True

    def test_non_tty_context_and_doc(self, monkeypatch):
        """context 和 doc 字段应作为一等字段输出到 JSON"""
        import io

        fake_stderr = io.StringIO()
        fake_stderr.isatty = lambda: False
        monkeypatch.setattr("sys.stderr", fake_stderr)

        from click.exceptions import Exit

        with pytest.raises(Exit):
            cli_error(
                ErrorType.INVALID_ARGS,
                "--format 参数值无效",
                context={
                    "arg": "--format",
                    "received": "xml",
                    "expected": ["json", "table", "csv"],
                },
                suggestion="使用 --format json",
                doc="flexllm ask --help",
            )

        data = json.loads(fake_stderr.getvalue())
        # Agent 可直接按字段读取，无需正则 message
        assert data["context"]["arg"] == "--format"
        assert data["context"]["received"] == "xml"
        assert data["context"]["expected"] == ["json", "table", "csv"]
        assert data["doc"] == "flexllm ask --help"
        # 字段顺序：error/message/retryable 在前，扩展字段在后
        assert list(data.keys())[:3] == ["error", "message", "retryable"]

    def test_non_tty_single_line_json(self, monkeypatch):
        """JSON 必须是单行输出，便于 Agent 逐行解析 stderr"""
        import io

        fake_stderr = io.StringIO()
        fake_stderr.isatty = lambda: False
        monkeypatch.setattr("sys.stderr", fake_stderr)

        from click.exceptions import Exit

        with pytest.raises(Exit):
            cli_error(
                ErrorType.IO_ERROR,
                "文件不存在",
                context={"path": "/tmp/missing.json", "exception_type": "FileNotFoundError"},
            )

        output = fake_stderr.getvalue().rstrip("\n")
        assert "\n" not in output, "JSON 输出不应包含换行"
        # 确保是合法 JSON
        assert json.loads(output)["context"]["path"] == "/tmp/missing.json"

    def test_non_tty_no_context_omits_field(self, monkeypatch):
        """不传 context/doc 时 JSON 中不应出现这些字段"""
        import io

        fake_stderr = io.StringIO()
        fake_stderr.isatty = lambda: False
        monkeypatch.setattr("sys.stderr", fake_stderr)

        from click.exceptions import Exit

        with pytest.raises(Exit):
            cli_error(ErrorType.GENERAL, "一般错误")

        data = json.loads(fake_stderr.getvalue())
        assert "context" not in data
        assert "doc" not in data
        assert "suggestion" not in data
        # 必选字段一直都在
        assert data["error"] == "general_error"
        assert data["retryable"] is False

    def test_tty_context_rendering(self, capsys, monkeypatch):
        """TTY 模式下 context 应缩进键值对显示"""

        class FakeTTY:
            def __init__(self, capsys):
                self._buf = []

            def isatty(self):
                return True

            def write(self, s):
                self._buf.append(s)
                return len(s)

            def flush(self):
                pass

            def fileno(self):
                return 2

        import sys as _sys

        fake = FakeTTY(capsys)
        monkeypatch.setattr(_sys, "stderr", fake)

        from click.exceptions import Exit

        with pytest.raises(Exit):
            cli_error(
                ErrorType.INVALID_ARGS,
                "参数错误",
                context={"arg": "--format", "expected": ["json", "csv"]},
                suggestion="改用 --format json",
                doc="flexllm ask --help",
            )

        output = "".join(fake._buf)
        assert "错误: 参数错误" in output
        assert "arg: --format" in output
        assert "expected: json, csv" in output  # 列表应被展平为逗号分隔
        assert "提示: 改用 --format json" in output
        assert "详见: flexllm ask --help" in output


class TestDryRunOutput:
    def test_json_output(self, capsys):
        from click.exceptions import Exit

        with pytest.raises(Exit) as exc_info:
            dry_run_output({"action": "test", "model": "gpt-4"})
        assert exc_info.value.exit_code == ExitCode.DRY_RUN

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["action"] == "test"
        assert data["model"] == "gpt-4"

    def test_chinese_chars(self, capsys):
        from click.exceptions import Exit

        with pytest.raises(Exit):
            dry_run_output({"action": "ask", "message": "你好世界"})

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["message"] == "你好世界"
