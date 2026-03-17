"""TaskManager — 文件持久化的任务管理器

将任务存储为 JSON 文件，支持创建、查询、更新、删除和依赖管理。
"""

import json
from pathlib import Path


class TaskManager:
    """文件持久化的任务管理器

    Args:
        tasks_dir: 任务存储目录，相对于 cwd 或绝对路径
    """

    def __init__(self, tasks_dir: str | Path = ".tasks"):
        self.tasks_dir = Path(tasks_dir)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self._next_id = self._recover_next_id()

    def _recover_next_id(self) -> int:
        """从已有文件恢复下一个 ID"""
        max_id = 0
        for f in self.tasks_dir.glob("task_*.json"):
            try:
                tid = int(f.stem.split("_")[1])
                max_id = max(max_id, tid)
            except (IndexError, ValueError):
                continue
        return max_id + 1

    def _task_path(self, task_id: int) -> Path:
        return self.tasks_dir / f"task_{task_id}.json"

    def _load_task(self, task_id: int) -> dict | None:
        path = self._task_path(task_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_task(self, task: dict):
        path = self._task_path(task["id"])
        path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")

    def create(self, subject: str, description: str = "") -> str:
        """创建任务，返回 JSON 字符串"""
        task = {
            "id": self._next_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "blockedBy": [],
            "blocks": [],
            "owner": "",
        }
        self._save_task(task)
        self._next_id += 1
        return json.dumps(task, ensure_ascii=False)

    def get(self, task_id: int) -> str:
        """获取单个任务"""
        task = self._load_task(task_id)
        if not task:
            return json.dumps({"error": f"任务 #{task_id} 不存在"})
        return json.dumps(task, ensure_ascii=False)

    def update(
        self,
        task_id: int,
        status: str | None = None,
        add_blocked_by: list[int] | None = None,
        add_blocks: list[int] | None = None,
    ) -> str:
        """更新任务状态和依赖"""
        task = self._load_task(task_id)
        if not task:
            return json.dumps({"error": f"任务 #{task_id} 不存在"})

        if status:
            task["status"] = status

        if add_blocked_by:
            for bid in add_blocked_by:
                if bid not in task["blockedBy"]:
                    task["blockedBy"].append(bid)
                # 在被依赖的任务中添加反向引用
                blocker = self._load_task(bid)
                if blocker and task_id not in blocker["blocks"]:
                    blocker["blocks"].append(task_id)
                    self._save_task(blocker)

        if add_blocks:
            for bid in add_blocks:
                if bid not in task["blocks"]:
                    task["blocks"].append(bid)
                # 在依赖任务中添加反向引用
                blocked = self._load_task(bid)
                if blocked and task_id not in blocked["blockedBy"]:
                    blocked["blockedBy"].append(task_id)
                    self._save_task(blocked)

        # 完成时自动清理：从其他任务的 blockedBy 中移除自己
        if status == "completed":
            for bid in task["blocks"]:
                blocked = self._load_task(bid)
                if blocked and task_id in blocked["blockedBy"]:
                    blocked["blockedBy"].remove(task_id)
                    self._save_task(blocked)

        self._save_task(task)
        return json.dumps(task, ensure_ascii=False)

    def list_all(self) -> str:
        """列出所有任务，格式化输出"""
        tasks = []
        for f in sorted(self.tasks_dir.glob("task_*.json")):
            try:
                task = json.loads(f.read_text(encoding="utf-8"))
                tasks.append(task)
            except (json.JSONDecodeError, OSError):
                continue

        if not tasks:
            return "没有任务"

        lines = []
        completed = sum(1 for t in tasks if t["status"] == "completed")
        for t in tasks:
            icon = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(
                t["status"], "[?]"
            )
            line = f"{icon} #{t['id']}: {t['subject']}"
            if t.get("blockedBy"):
                line += f" (blocked by: {t['blockedBy']})"
            if t.get("owner"):
                line += f" (@{t['owner']})"
            lines.append(line)

        lines.append(f"\n({completed}/{len(tasks)} completed)")
        return "\n".join(lines)

    def delete(self, task_id: int) -> str:
        """删除任务"""
        path = self._task_path(task_id)
        if not path.exists():
            return json.dumps({"error": f"任务 #{task_id} 不存在"})

        task = self._load_task(task_id)

        # 清理依赖关系
        if task:
            for bid in task.get("blocks", []):
                blocked = self._load_task(bid)
                if blocked and task_id in blocked["blockedBy"]:
                    blocked["blockedBy"].remove(task_id)
                    self._save_task(blocked)
            for bid in task.get("blockedBy", []):
                blocker = self._load_task(bid)
                if blocker and task_id in blocker["blocks"]:
                    blocker["blocks"].remove(task_id)
                    self._save_task(blocker)

        path.unlink()
        return json.dumps({"deleted": task_id})
