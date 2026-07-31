"""断点续传 _extract_save_input / _resume_from_jsonl 测试"""

import json

import pytest

from flexllm.clients.base import _extract_save_input, _resume_from_jsonl


class TestExtractSaveInput:
    """测试 _extract_save_input 三种模式"""

    def _make_messages(self):
        return [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
            {"role": "user", "content": "帮我查天气"},
        ]

    def test_save_input_true(self):
        """save_input=True 返回完整 messages"""
        msgs = self._make_messages()
        result = _extract_save_input(msgs, True)
        assert result is msgs

    def test_save_input_last(self):
        """save_input='last' 返回最后一个 user message 的 content"""
        msgs = self._make_messages()
        result = _extract_save_input(msgs, "last")
        assert result == "帮我查天气"

    def test_save_input_last_no_user(self):
        """save_input='last' 无 user message 时返回空字符串"""
        msgs = [{"role": "system", "content": "sys"}]
        result = _extract_save_input(msgs, "last")
        assert result == ""

    def test_save_input_false(self):
        """save_input=False 返回 None"""
        msgs = self._make_messages()
        result = _extract_save_input(msgs, False)
        assert result is None


class TestResumeFromJsonl:
    """测试 _resume_from_jsonl 断点续传"""

    def _make_messages_list(self, n=5):
        return [[{"role": "user", "content": f"msg-{i}"}] for i in range(n)]

    def test_file_not_exists(self):
        """文件不存在时返回空"""
        indices, records = _resume_from_jsonl("/tmp/nonexistent_file_xyz.jsonl", [], True)
        assert indices == set()
        assert records == []

    def test_normal_resume(self, tmp_path):
        """正常恢复已完成的记录"""
        messages_list = self._make_messages_list(5)
        output_file = tmp_path / "output.jsonl"

        # 写入 3 条已完成记录
        lines = []
        for i in [0, 2, 4]:
            record = {
                "index": i,
                "status": "success",
                "output": f"result-{i}",
                "input": messages_list[i],
            }
            lines.append(json.dumps(record, ensure_ascii=False))
        output_file.write_text("\n".join(lines) + "\n")

        indices, records = _resume_from_jsonl(str(output_file), messages_list, True)
        assert indices == {0, 2, 4}
        assert len(records) == 3

    def test_corrupted_line_skipped(self, tmp_path):
        """损坏行被跳过"""
        messages_list = self._make_messages_list(3)
        output_file = tmp_path / "output.jsonl"

        good_record = json.dumps(
            {
                "index": 0,
                "status": "success",
                "output": "ok",
                "input": messages_list[0],
            }
        )
        output_file.write_text(good_record + "\n" + "CORRUPTED LINE\n")

        indices, records = _resume_from_jsonl(str(output_file), messages_list, True)
        assert indices == {0}
        assert len(records) == 1

    def test_index_out_of_range_skipped(self, tmp_path):
        """index 超出范围的记录被跳过"""
        messages_list = self._make_messages_list(2)
        output_file = tmp_path / "output.jsonl"

        records_data = [
            {"index": 0, "status": "success", "output": "ok", "input": messages_list[0]},
            {"index": 999, "status": "success", "output": "bad"},  # 超出范围
        ]
        output_file.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records_data) + "\n"
        )

        indices, records = _resume_from_jsonl(str(output_file), messages_list, True)
        assert indices == {0}
        assert len(records) == 1

    def test_input_validation_failure(self, tmp_path):
        """input 校验失败时抛出 ValueError"""
        messages_list = self._make_messages_list(3)
        output_file = tmp_path / "output.jsonl"

        # 写入 input 不匹配的记录
        record = {
            "index": 0,
            "status": "success",
            "output": "ok",
            "input": [{"role": "user", "content": "WRONG CONTENT"}],
        }
        output_file.write_text(json.dumps(record) + "\n")

        with pytest.raises(ValueError, match="文件校验失败"):
            _resume_from_jsonl(str(output_file), messages_list, True)

    def test_middle_reorder_detected(self, tmp_path):
        """中间样本乱序（首尾不变）也要被检出——只校验首尾会让 checkpoint 静默错位"""
        messages_list = self._make_messages_list(4)
        output_file = tmp_path / "output.jsonl"

        # 文件按原顺序写入，本轮把中间两条互换后重跑
        records_data = [
            {"index": i, "status": "success", "output": f"r{i}", "input": messages_list[i]}
            for i in range(4)
        ]
        output_file.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records_data) + "\n"
        )
        swapped = [messages_list[0], messages_list[2], messages_list[1], messages_list[3]]

        with pytest.raises(ValueError, match="文件校验失败"):
            _resume_from_jsonl(str(output_file), swapped, True)

    def test_empty_file(self, tmp_path):
        """空文件返回空"""
        output_file = tmp_path / "output.jsonl"
        output_file.write_text("")

        indices, records = _resume_from_jsonl(str(output_file), self._make_messages_list(3), True)
        assert indices == set()
        assert records == []

    def test_error_status_skipped(self, tmp_path):
        """status != 'success' 的记录被跳过"""
        messages_list = self._make_messages_list(3)
        output_file = tmp_path / "output.jsonl"

        records_data = [
            {"index": 0, "status": "success", "output": "ok", "input": messages_list[0]},
            {"index": 1, "status": "error", "output": None, "input": messages_list[1]},
        ]
        output_file.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records_data) + "\n"
        )

        indices, records = _resume_from_jsonl(str(output_file), messages_list, True)
        assert indices == {0}
        assert len(records) == 1

    def test_resume_with_save_input_last(self, tmp_path):
        """save_input='last' 模式下恢复和校验"""
        messages_list = self._make_messages_list(3)
        output_file = tmp_path / "output.jsonl"

        # save_input='last' 保存最后一个 user content
        record = {
            "index": 0,
            "status": "success",
            "output": "ok",
            "input": "msg-0",  # 对应 messages_list[0] 的 user content
        }
        output_file.write_text(json.dumps(record) + "\n")

        indices, records = _resume_from_jsonl(str(output_file), messages_list, "last")
        assert indices == {0}

    def test_resume_without_input_field(self, tmp_path):
        """记录中无 input 字段时跳过校验"""
        messages_list = self._make_messages_list(3)
        output_file = tmp_path / "output.jsonl"

        record = {"index": 0, "status": "success", "output": "ok"}
        output_file.write_text(json.dumps(record) + "\n")

        indices, records = _resume_from_jsonl(str(output_file), messages_list, True)
        assert indices == {0}
