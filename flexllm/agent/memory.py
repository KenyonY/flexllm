"""Agent 持久化记忆

支持对话历史持久化和事实/偏好存储，基于 flaxkv2。

Example:
    from flexllm.agent.memory import MemoryStore

    memory = MemoryStore()
    memory.save("session-1", [{"role": "user", "content": "你好"}])
    messages = memory.load("session-1")

    memory.save_fact("user.name", "张三")
    name = memory.get_fact("user.name")
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flaxkv2 import FlaxKV

logger = logging.getLogger(__name__)

# Key 前缀
_SESSION_PREFIX = "session:"
_FACT_PREFIX = "fact:"


class MemoryStore:
    """Agent 记忆存储，基于 flaxkv2"""

    def __init__(self, db_path: str = "~/.flexllm/agent_memory"):
        try:
            from flaxkv2 import FlaxKV
        except ImportError:
            raise ImportError("MemoryStore 需要 flaxkv2，请安装: pip install flexllm[memory]")

        db_path = os.path.expanduser(db_path)
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self._db: FlaxKV = FlaxKV("agent_memory", db_path)

    def save(self, session_id: str, messages: list[dict]):
        """保存对话历史"""
        self._db[f"{_SESSION_PREFIX}{session_id}"] = json.dumps(messages, ensure_ascii=False)

    def load(self, session_id: str) -> list[dict]:
        """加载对话历史，不存在返回空列表"""
        data = self._db.get(f"{_SESSION_PREFIX}{session_id}")
        if data is None:
            return []
        return json.loads(data)

    def list_sessions(self) -> list[str]:
        """列出所有 session ID"""
        return [k[len(_SESSION_PREFIX) :] for k in self._db.keys() if k.startswith(_SESSION_PREFIX)]

    def delete_session(self, session_id: str):
        """删除指定 session"""
        key = f"{_SESSION_PREFIX}{session_id}"
        if key in self._db:
            del self._db[key]

    def save_fact(self, key: str, value: str):
        """保存事实/偏好"""
        self._db[f"{_FACT_PREFIX}{key}"] = value

    def get_fact(self, key: str) -> str | None:
        """获取单个事实"""
        return self._db.get(f"{_FACT_PREFIX}{key}")

    def get_facts(self, prefix: str = "") -> dict[str, str]:
        """获取所有事实或按前缀过滤"""
        full_prefix = f"{_FACT_PREFIX}{prefix}"
        result = {}
        for k in self._db.keys():
            if k.startswith(full_prefix):
                fact_key = k[len(_FACT_PREFIX) :]
                val = self._db.get(k)
                if val is not None:
                    result[fact_key] = val
        return result

    def delete_fact(self, key: str):
        """删除事实"""
        full_key = f"{_FACT_PREFIX}{key}"
        if full_key in self._db:
            del self._db[full_key]

    def close(self):
        """关闭数据库"""
        if hasattr(self, "_db") and self._db is not None:
            self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
