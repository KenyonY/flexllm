"""
batch_helpers - 批量处理的横切关注点组件

提取自 LLMClientBase 和 LLMClientPool 中重复的 JSONL 写入、断点续传、
文件 compact、进度追踪等逻辑，作为独立的可组合组件。
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def extract_save_input(messages: list[dict], save_input: bool | str):
    """根据 save_input 参数提取要保存的 input 内容

    Args:
        messages: 原始 messages 列表
        save_input: True=完整保存, "last"=仅保存最后一个 user message 的 content, False=不保存

    Returns:
        - save_input=True: 原始 messages
        - save_input="last": 最后一个 user message 的 content (str)
        - save_input=False: None
    """
    if save_input is True:
        return messages
    elif save_input == "last":
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""
    return None


def resume_from_jsonl(
    output_jsonl: str,
    messages_list: list[list[dict]],
    save_input: bool | str = True,
) -> tuple[set[int], list[dict]]:
    """从 JSONL 文件恢复已完成的索引和记录（断点续传公共逻辑）

    Args:
        output_jsonl: JSONL 文件路径
        messages_list: 原始消息列表（用于首尾校验）
        save_input: input 保存策略（与写入时一致）

    Returns:
        (completed_indices, records): 已完成的索引集合 和 有效记录列表
    """
    output_path = Path(output_jsonl)
    if not output_path.exists():
        return set(), []

    n = len(messages_list)
    records = []
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                if record.get("status") == "success":
                    idx = record.get("index")
                    if 0 <= idx < n:
                        records.append(record)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    # 首尾校验：只在记录包含 input 字段时校验
    file_valid = True
    if records:
        first, last = records[0], records[-1]
        if "input" in first:
            expected = extract_save_input(messages_list[first["index"]], save_input)
            if expected is not None and first["input"] != expected:
                file_valid = False
        if file_valid and len(records) > 1 and "input" in last:
            expected = extract_save_input(messages_list[last["index"]], save_input)
            if expected is not None and last["input"] != expected:
                file_valid = False

    if not file_valid:
        raise ValueError(
            f"文件校验失败: {output_jsonl} 中的 input 与当前 messages_list 不匹配。"
            f"请删除或重命名该文件后重试。"
        )

    completed_indices = {r["index"] for r in records}
    if completed_indices:
        logger.info(f"从文件恢复: 已完成 {len(completed_indices)}/{n}")

    return completed_indices, records


def compact_output_file(file_path: str):
    """去重输出文件，保留每个 index 的最新成功记录"""
    tmp_path = file_path + ".tmp"
    try:
        records = {}
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                idx = r.get("index")
                if idx is None:
                    continue
                # 成功记录优先，或者该 index 还没有记录
                if r.get("status") == "success" or idx not in records:
                    records[idx] = r

        # 先写入临时文件
        with open(tmp_path, "w", encoding="utf-8") as f:
            for r in sorted(records.values(), key=lambda x: x["index"]):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 原子替换（同一文件系统上 replace 是原子操作）
        os.replace(tmp_path, file_path)
    except Exception as e:
        logger.warning(f"Compact 输出文件失败: {e}")
        # 清理可能残留的临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def validate_batch_params(
    messages_list: list[list[dict]],
    metadata_list: list[dict] | None,
    output_jsonl: str | None,
):
    """校验 batch 参数（metadata_list 长度、output_jsonl 扩展名）"""
    if metadata_list is not None and len(metadata_list) != len(messages_list):
        raise ValueError(
            f"metadata_list 长度 ({len(metadata_list)}) 必须与 messages_list 长度 ({len(messages_list)}) 一致"
        )
    if output_jsonl and not output_jsonl.endswith(".jsonl"):
        raise ValueError(f"output_jsonl 必须使用 .jsonl 扩展名，当前: {output_jsonl}")


class JsonlWriter:
    """JSONL 文件写入器，负责缓冲写入、定时刷新和文件 compact

    Usage:
        writer = JsonlWriter("output.jsonl", messages_list, flush_interval=1.0)
        writer.write_result(idx, content, status="success", usage=usage)
        writer.close()  # 刷新并 compact
    """

    def __init__(
        self,
        output_jsonl: str | None,
        messages_list: list[list[dict]],
        save_input: bool | str = True,
        metadata_list: list[dict] | None = None,
        flush_interval: float = 1.0,
    ):
        self._messages_list = messages_list
        self._save_input = save_input
        self._metadata_list = metadata_list
        self._flush_interval = flush_interval
        self._output_jsonl = output_jsonl

        self._file_writer = None
        self._buffer: list[dict] = []
        self._last_flush_time = time.time()
        self._completed_indices: set[int] = set()

        if output_jsonl:
            self._completed_indices, _ = resume_from_jsonl(output_jsonl, messages_list, save_input)
            self._file_writer = open(Path(output_jsonl), "a", encoding="utf-8")

    @property
    def completed_indices(self) -> set[int]:
        return self._completed_indices

    def write_result(
        self,
        index: int,
        content: Any,
        status: str = "success",
        error: str = None,
        usage: dict = None,
    ):
        """写入一条结果记录（带缓冲）"""
        if self._file_writer is None:
            return

        record = {
            "index": index,
            "output": content,
            "status": status,
        }
        input_value = extract_save_input(self._messages_list[index], self._save_input)
        if input_value is not None:
            record["input"] = input_value
        if self._metadata_list is not None:
            record["metadata"] = self._metadata_list[index]
        if usage is not None:
            record["usage"] = usage
        if error:
            record["error"] = error

        self._buffer.append(record)

        # 基于时间刷新
        if time.time() - self._last_flush_time >= self._flush_interval:
            self.flush()

    def flush(self):
        """刷新缓冲区到文件"""
        if self._file_writer and self._buffer:
            for record in self._buffer:
                self._file_writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._file_writer.flush()
            self._buffer = []
            self._last_flush_time = time.time()

    def close(self):
        """刷新、关闭文件、compact"""
        self.flush()
        if self._file_writer:
            self._file_writer.close()
            self._file_writer = None
            if self._output_jsonl:
                compact_output_file(self._output_jsonl)
