"""TodoTracker — 轻量内存级进度追踪

在 Agent 运行期间追踪 todo 列表，定期提醒 LLM 更新进度。
"""


class TodoTracker:
    """内存级 todo 追踪器

    Args:
        nag_interval: 多少轮未更新 todo 后发送提醒
    """

    def __init__(self, nag_interval: int = 3):
        self.items: list[dict] = []  # {id, text, status}
        self.rounds_since_update: int = 0
        self.nag_interval = nag_interval

    def update(self, items: list[dict]) -> str:
        """更新 todo 列表

        Args:
            items: [{"id": 1, "text": "...", "status": "pending|in_progress|completed"}, ...]
        """
        validated = []
        for item in items:
            validated.append(
                {
                    "id": item.get("id", len(validated) + 1),
                    "text": item.get("text", ""),
                    "status": item.get("status", "pending"),
                }
            )
        self.items = validated
        self.rounds_since_update = 0
        return self.render()

    def render(self) -> str:
        """渲染 todo 列表"""
        if not self.items:
            return "(no todos)"

        icons = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
        lines = []
        for item in self.items:
            icon = icons.get(item["status"], "[?]")
            lines.append(f"{icon} #{item['id']}: {item['text']}")

        completed = sum(1 for i in self.items if i["status"] == "completed")
        lines.append(f"\n({completed}/{len(self.items)} completed)")
        return "\n".join(lines)

    def tick(self) -> str | None:
        """每轮调用，检查是否需要发送提醒"""
        if not self.items:
            return None

        self.rounds_since_update += 1
        if self.rounds_since_update >= self.nag_interval:
            pending = sum(1 for i in self.items if i["status"] != "completed")
            if pending > 0:
                return (
                    f"[reminder] 你有 {pending} 个未完成的 todo 项。"
                    f"请使用 todo 工具更新进度。\n\n当前状态:\n{self.render()}"
                )
        return None

    def notify_used(self):
        """重置计数器（todo 工具被调用时）"""
        self.rounds_since_update = 0
