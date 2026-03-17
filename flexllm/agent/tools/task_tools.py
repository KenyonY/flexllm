"""任务管理工具 — 注册到 ToolRegistry 的 4 个任务工具

通过 register_task_tools() 将工具注入到 ToolRegistry 实例。
使用闭包绑定 TaskManager 实例，每个 AgentClient 独立。
"""

from .base import ToolDef


def register_task_tools(registry, task_manager):
    """将 4 个任务管理工具注册到 registry

    Args:
        registry: ToolRegistry 实例
        task_manager: TaskManager 实例
    """
    tm = task_manager

    def task_create(subject: str, description: str = "") -> str:
        """创建新任务

        Args:
            subject: 任务标题
            description: 任务详细描述
        """
        return tm.create(subject, description)

    def task_update(
        task_id: int,
        status: str = None,
        addBlockedBy: list = None,
        addBlocks: list = None,
    ) -> str:
        """更新任务状态或依赖

        Args:
            task_id: 任务 ID
            status: 新状态 (pending/in_progress/completed)
            addBlockedBy: 添加阻塞当前任务的任务 ID 列表
            addBlocks: 添加被当前任务阻塞的任务 ID 列表
        """
        blocked_by = [int(x) for x in addBlockedBy] if addBlockedBy else None
        blocks = [int(x) for x in addBlocks] if addBlocks else None
        return tm.update(task_id, status=status, add_blocked_by=blocked_by, add_blocks=blocks)

    def task_list() -> str:
        """列出所有任务及其状态"""
        return tm.list_all()

    def task_get(task_id: int) -> str:
        """获取单个任务的详细信息

        Args:
            task_id: 任务 ID
        """
        return tm.get(task_id)

    tools = [
        ToolDef(
            name="task_create",
            description="创建新任务。用于将工作分解为可追踪的子任务。",
            parameters={
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "任务标题（简短、动词开头）"},
                    "description": {"type": "string", "description": "任务详细描述"},
                },
                "required": ["subject"],
            },
            executor=task_create,
            readonly=False,
        ),
        ToolDef(
            name="task_update",
            description=(
                "更新任务状态或依赖关系。完成任务时 status 设为 completed，开始时设为 in_progress。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer", "description": "任务 ID"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                        "description": "新状态",
                    },
                    "addBlockedBy": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "阻塞当前任务的任务 ID",
                    },
                    "addBlocks": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "被当前任务阻塞的任务 ID",
                    },
                },
                "required": ["task_id"],
            },
            executor=task_update,
            readonly=False,
        ),
        ToolDef(
            name="task_list",
            description="列出所有任务及其状态、依赖关系和完成进度。",
            parameters={"type": "object", "properties": {}},
            executor=task_list,
            readonly=True,
        ),
        ToolDef(
            name="task_get",
            description="获取单个任务的详细信息，包括描述和依赖。",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer", "description": "任务 ID"},
                },
                "required": ["task_id"],
            },
            executor=task_get,
            readonly=True,
        ),
    ]

    for tool_def in tools:
        registry.register(tool_def)


def register_todo_tool(registry, todo_tracker):
    """将 todo 工具注册到 registry

    Args:
        registry: ToolRegistry 实例
        todo_tracker: TodoTracker 实例
    """
    tracker = todo_tracker

    def todo(items: list) -> str:
        """更新待办事项列表，追踪当前任务的进度

        Args:
            items: 待办项列表，每项包含 id、text、status(pending/in_progress/completed)
        """
        return tracker.update(items)

    registry.register(
        ToolDef(
            name="todo",
            description=(
                "更新待办事项列表来追踪进度。每项包含 id(数字)、text(描述)、"
                "status(pending/in_progress/completed)。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "text": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                },
                            },
                            "required": ["id", "text", "status"],
                        },
                        "description": "待办项列表",
                    },
                },
                "required": ["items"],
            },
            executor=todo,
            readonly=False,
        )
    )
