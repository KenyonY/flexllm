"""测试 CLI 新功能: --schema, -x, -f"""

import json
import os
import tempfile

import typer

from flexllm.cli.utils import extract_code_block, parse_schema, read_file_contents

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
