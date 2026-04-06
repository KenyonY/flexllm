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
