"""AgentClient 数据类型定义"""

from dataclasses import dataclass, field


@dataclass
class ToolCallRecord:
    """单次工具调用记录"""

    name: str
    arguments: str  # JSON 字符串
    result: str


@dataclass
class AgentResult:
    """Agent 执行结果"""

    content: str | None  # LLM 最终回复内容
    rounds: int = 0  # tool-calling 轮数
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    usage: dict | None = None  # 累计 token 用量
    parsed: object = None  # structured output 解析结果（Pydantic model 实例）
