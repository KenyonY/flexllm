#! /usr/bin/env python3

"""
Chain of Thought client for orchestrating multiple LLM calls.
"""

import asyncio
import logging
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..async_api.progress import ProgressBarConfig, ProgressTracker
from .openai import OpenAIClient

# Rich库安全导入和使用


# 创建全局console实例
chain_console = Console(force_terminal=True, width=100, color_system="auto")


def safe_chain_print(*args, **kwargs):
    """安全的Rich打印函数，用于Chain of Thought模块"""
    try:
        chain_console.print(*args, **kwargs)
    except Exception:
        # 降级到普通print
        import re

        clean_args = []
        for arg in args:
            if isinstance(arg, str):
                clean_text = re.sub(r"\[/?[^\]]*\]", "", str(arg))
                clean_text = clean_text.encode("ascii", "ignore").decode("ascii")
                clean_args.append(clean_text)
            else:
                clean_args.append(str(arg))
        import builtins

        builtins.print(*clean_args, **kwargs)


class ChainProgressTracker:
    """链条批处理进度跟踪器，适配我们自己的 ProgressTracker"""

    def __init__(self, total_chains: int, show_progress: bool = True):
        self.total_chains = total_chains
        self.completed_chains = 0
        self.start_time = time.time()
        self.show_progress = show_progress

        # 创建内部进度跟踪器
        if self.show_progress:
            self.config = ProgressBarConfig(
                bar_length=40,
                show_percentage=True,
                show_speed=True,
                show_counts=True,
                show_time_stats=True,
                use_colors=True,
            )
            # 创建一个虚拟的结果类来适配接口
            from dataclasses import dataclass

            @dataclass
            class ChainResult:
                request_id: int
                data: Any = None
                status: str = "success"
                latency: float = 0.0

            self.ChainResult = ChainResult
            self.tracker = ProgressTracker(total_chains, config=self.config)
        else:
            self.tracker = None

    def update(self, chain_index: int, success: bool = True, execution_time: float = 0.0):
        """更新进度"""
        self.completed_chains += 1

        if self.tracker:
            result = self.ChainResult(
                request_id=chain_index,
                status="success" if success else "error",
                latency=execution_time,
            )
            self.tracker.update(result)

    def finish(self):
        """完成进度跟踪"""
        if self.tracker:
            # 确保最终进度条状态显示
            if self.completed_chains == self.total_chains:
                safe_chain_print()  # 换行，保留最终进度条

    def get_progress_info(self) -> dict:
        """获取进度信息"""
        elapsed_time = time.time() - self.start_time
        remaining_time = 0.0
        if self.completed_chains > 0:
            avg_time_per_chain = elapsed_time / self.completed_chains
            remaining_chains = self.total_chains - self.completed_chains
            remaining_time = avg_time_per_chain * remaining_chains

        return {
            "completed": self.completed_chains,
            "total": self.total_chains,
            "progress_percent": (
                (self.completed_chains / self.total_chains * 100) if self.total_chains > 0 else 0
            ),
            "elapsed_time": elapsed_time,
            "remaining_time": remaining_time,
            "rate": self.completed_chains / elapsed_time if elapsed_time > 0 else 0,
        }


# 为ChainOfThoughtClient创建专用logger
def setup_chain_logger():
    """设置ChainOfThoughtClient专用logger"""
    if not hasattr(setup_chain_logger, "_configured"):
        # 创建专用的logger
        chain_logger = logging.getLogger("maque.chain")
        chain_logger.setLevel(logging.INFO)

        # 创建自定义格式的handler
        if not chain_logger.handlers:  # 避免重复添加handler
            handler = logging.StreamHandler(sys.stderr)

            # 自定义格式器，模仿loguru的简洁格式
            class ChainFormatter(logging.Formatter):
                def format(self, record):
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    return f"{timestamp} | {record.getMessage()}"

            handler.setFormatter(ChainFormatter())
            chain_logger.addHandler(handler)

            # 防止消息传播到root logger（避免重复输出）
            chain_logger.propagate = False

        setup_chain_logger._configured = True
        setup_chain_logger._logger = chain_logger

    return setup_chain_logger._logger


# 创建专用logger实例
chain_logger = setup_chain_logger()


# 为了保持与loguru兼容的接口，创建一个包装类
class ChainLoggerWrapper:
    def __init__(self, logger):
        self._logger = logger

    def info(self, message):
        self._logger.info(message)

    def debug(self, message):
        self._logger.debug(message)

    def warning(self, message):
        self._logger.warning(message)

    def error(self, message):
        self._logger.error(message)

    def success(self, message):
        # 对于success级别，我们使用info
        self._logger.info(message)


# 使用包装后的logger
chain_logger = ChainLoggerWrapper(chain_logger)


class StepStatus(Enum):
    """步骤执行状态枚举"""

    RUNNING = "running"  # 正在执行
    COMPLETED = "completed"  # 执行完成
    FAILED = "failed"  # 执行失败
    TIMEOUT = "timeout"  # 执行超时


class ChainStatus(Enum):
    """链条执行状态枚举"""

    RUNNING = "running"  # 正在执行
    COMPLETED = "completed"  # 执行完成
    FAILED = "failed"  # 执行失败
    TIMEOUT = "timeout"  # 执行超时


@dataclass
class ExecutionConfig:
    """
    执行配置类。

    Attributes:
        step_timeout: 单个步骤的超时时间（秒），None表示无超时
        chain_timeout: 整个链条的超时时间（秒），None表示无超时
        max_retries: 单个步骤的最大重试次数
        retry_delay: 重试间隔时间（秒）
        enable_monitoring: 是否启用监控
        log_level: 日志级别 ("DEBUG", "INFO", "WARNING", "ERROR")
        enable_progress: 是否显示进度信息
    """

    step_timeout: float | None = None
    chain_timeout: float | None = None
    max_retries: int = 0
    retry_delay: float = 1.0
    enable_monitoring: bool = True
    log_level: str = "WARNING"
    enable_progress: bool = False


@dataclass
class StepExecutionInfo:
    """
    步骤执行信息。

    Attributes:
        step_name: 步骤名称
        status: 执行状态
        start_time: 开始时间
        end_time: 结束时间
        execution_time: 执行时间
        retry_count: 重试次数
        error: 错误信息
    """

    step_name: str
    status: StepStatus = StepStatus.RUNNING
    start_time: float | None = None
    end_time: float | None = None
    execution_time: float | None = None
    retry_count: int = 0
    error: str | None = None


@dataclass
class ChainExecutionInfo:
    """
    链条执行信息。

    Attributes:
        chain_id: 链条ID
        status: 执行状态
        start_time: 开始时间
        end_time: 结束时间
        total_execution_time: 总执行时间
        steps_info: 各步骤执行信息
        completed_steps: 已完成步骤数
        error: 错误信息
    """

    chain_id: str
    status: ChainStatus = ChainStatus.RUNNING
    start_time: float | None = None
    end_time: float | None = None
    total_execution_time: float | None = None
    steps_info: list[StepExecutionInfo] = field(default_factory=list)
    completed_steps: int = 0
    error: str | None = None


class ChainMonitor(ABC):
    """链条监控器抽象基类"""

    @abstractmethod
    async def on_chain_start(self, chain_info: ChainExecutionInfo) -> None:
        """链条开始执行时调用"""
        pass

    @abstractmethod
    async def on_chain_end(self, chain_info: ChainExecutionInfo) -> None:
        """链条执行结束时调用"""
        pass

    @abstractmethod
    async def on_step_start(
        self, step_info: StepExecutionInfo, chain_info: ChainExecutionInfo
    ) -> None:
        """步骤开始执行时调用"""
        pass

    @abstractmethod
    async def on_step_end(
        self, step_info: StepExecutionInfo, chain_info: ChainExecutionInfo
    ) -> None:
        """步骤执行结束时调用"""
        pass

    @abstractmethod
    async def on_error(self, error: Exception, chain_info: ChainExecutionInfo) -> None:
        """发生错误时调用"""
        pass

    @abstractmethod
    async def on_timeout(self, timeout_type: str, chain_info: ChainExecutionInfo) -> None:
        """超时时调用"""
        pass


class DefaultChainMonitor(ChainMonitor):
    """默认链条监控器实现"""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.log_levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        self.current_level = self.log_levels.get(config.log_level, 1)

    def _get_chain_prefix(self, chain_info: ChainExecutionInfo) -> str:
        """生成链条前缀"""
        chain_short_id = chain_info.chain_id.split("_")[-1][-4:]  # 取最后4位数字
        return f"[链条{chain_short_id}] "

    def _should_log(self, level: str) -> bool:
        return self.log_levels.get(level, 1) >= self.current_level

    def _log(self, level: str, message: str, chain_info: ChainExecutionInfo | None = None) -> None:
        if self._should_log(level):
            # 添加简化的链条ID前缀以区分不同链条的日志
            if chain_info:
                chain_prefix = self._get_chain_prefix(chain_info)
                formatted_message = f"{chain_prefix}{message}"
            else:
                formatted_message = message

            if level == "DEBUG":
                chain_logger.debug(formatted_message)
            elif level == "INFO":
                chain_logger.info(formatted_message)
            elif level == "WARNING":
                chain_logger.warning(formatted_message)
            elif level == "ERROR":
                chain_logger.error(formatted_message)
            else:
                chain_logger.info(formatted_message)

    async def on_chain_start(self, chain_info: ChainExecutionInfo) -> None:
        if self.config.enable_monitoring:
            self._log("INFO", "链条开始执行", chain_info)

    async def on_chain_end(self, chain_info: ChainExecutionInfo) -> None:
        if self.config.enable_monitoring:
            status_msg = f"链条执行结束 - 状态: {chain_info.status.value}"
            if chain_info.total_execution_time:
                status_msg += f", 总耗时: {chain_info.total_execution_time:.2f}秒"
            status_msg += f", 完成步骤: {chain_info.completed_steps}"
            self._log("INFO", status_msg, chain_info)

    async def on_step_start(
        self, step_info: StepExecutionInfo, chain_info: ChainExecutionInfo
    ) -> None:
        if self.config.enable_monitoring:
            progress = f"({chain_info.completed_steps + 1})"
            self._log("DEBUG", f"步骤 {step_info.step_name} 开始执行 {progress}", chain_info)

            if self.config.enable_progress:
                # 使用带链条ID的格式
                chain_prefix = self._get_chain_prefix(chain_info)
                chain_logger.info(
                    f"{chain_prefix}执行进度: 步骤 {chain_info.completed_steps + 1} - {step_info.step_name}"
                )

    async def on_step_end(
        self, step_info: StepExecutionInfo, chain_info: ChainExecutionInfo
    ) -> None:
        if self.config.enable_monitoring:
            status_msg = f"步骤 {step_info.step_name} 执行完成 - 状态: {step_info.status.value}"
            if step_info.execution_time:
                status_msg += f", 耗时: {step_info.execution_time:.2f}秒"
            if step_info.retry_count > 0:
                status_msg += f", 重试次数: {step_info.retry_count}"
            self._log("DEBUG", status_msg, chain_info)

    async def on_error(self, error: Exception, chain_info: ChainExecutionInfo) -> None:
        if self.config.enable_monitoring:
            self._log("ERROR", f"发生错误: {str(error)}", chain_info)

    async def on_timeout(self, timeout_type: str, chain_info: ChainExecutionInfo) -> None:
        if self.config.enable_monitoring:
            self._log("WARNING", f"{timeout_type}超时", chain_info)


class ExecutionController:
    """执行控制器"""

    def __init__(self, config: ExecutionConfig):
        self.config = config

    async def check_timeout(self, start_time: float, timeout: float | None) -> bool:
        """检查是否超时"""
        if timeout is None:
            return False
        return (time.time() - start_time) > timeout


@dataclass
class StepResult:
    """
    单个步骤的执行结果。

    Attributes:
        step_name: 步骤名称
        messages: 发送给LLM的消息列表
        response: LLM的响应内容
        model_params: 使用的模型参数
        execution_time: 执行时间（秒）
        status: 执行状态
        retry_count: 重试次数
        error: 错误信息（如果有）
    """

    step_name: str
    messages: list[dict[str, Any]]
    response: str
    model_params: dict[str, Any] = field(default_factory=dict)
    execution_time: float | None = None
    status: StepStatus = StepStatus.COMPLETED
    retry_count: int = 0
    error: str | None = None


@dataclass
class Context:
    """
    链条执行的上下文信息。

    Attributes:
        query: 初始用户查询（可选，用于通用场景）
        history: 所有步骤的执行历史
        custom_data: 自定义数据字典，用于存储任意额外信息
        execution_info: 链条执行信息（用于监控）
    """

    history: list[StepResult] = field(default_factory=list)
    query: str | None = None
    custom_data: dict[str, Any] = field(default_factory=dict)
    execution_info: ChainExecutionInfo | None = None

    def get_last_response(self) -> str | None:
        """获取最后一个步骤的响应。"""
        return self.history[-1].response if self.history else None

    def get_response_by_step(self, step_name: str) -> str | None:
        """根据步骤名称获取响应。"""
        for step_result in self.history:
            if step_result.step_name == step_name:
                return step_result.response
        return None

    def get_step_count(self) -> int:
        """获取已执行的步骤数量。"""
        return len(self.history)

    def add_custom_data(self, key: str, value: Any) -> None:
        """添加自定义数据。"""
        self.custom_data[key] = value

    def get_custom_data(self, key: str, default: Any = None) -> Any:
        """获取自定义数据。"""
        return self.custom_data.get(key, default)

    def get_execution_summary(self) -> dict[str, Any]:
        """获取执行摘要信息"""
        total_time = sum(s.execution_time or 0 for s in self.history)
        total_retries = sum(s.retry_count for s in self.history)
        failed_steps = [s.step_name for s in self.history if s.status == StepStatus.FAILED]

        return {
            "total_steps": len(self.history),
            "total_execution_time": total_time,
            "total_retries": total_retries,
            "failed_steps": failed_steps,
            "success_rate": (
                len([s for s in self.history if s.status == StepStatus.COMPLETED])
                / len(self.history)
                if self.history
                else 0
            ),
        }


@dataclass
class Step:
    """
    定义思想链中的一个步骤。

    Attributes:
        name: 步骤的唯一名称。
        prepare_messages_fn: 一个可调用对象，接收上下文（Context），返回用于LLM调用的消息列表（List[Dict]）。
        get_next_step_fn: 一个可调用对象，接收当前步骤的响应（str）和完整上下文（Context），返回下一个步骤的名称（str）或None表示结束。
        model_params: 调用LLM时使用的模型参数，例如 model, temperature等。
    """

    name: str
    prepare_messages_fn: Callable[[Context], list[dict[str, Any]]]
    get_next_step_fn: Callable[[str, Context], str | None]
    model_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class LinearStep:
    """
    定义线性链条中的一个步骤（简化版本）。

    Attributes:
        name: 步骤的唯一名称。
        prepare_messages_fn: 一个可调用对象，接收上下文（Context），返回用于LLM调用的消息列表（List[Dict]）。
        model_params: 调用LLM时使用的模型参数，例如 model, temperature等。
    """

    name: str
    prepare_messages_fn: Callable[[Context], list[dict[str, Any]]]
    model_params: dict[str, Any] = field(default_factory=dict)


class ChainOfThoughtClient:
    """
    一个客户端，用于执行由多个步骤组成的思想链（Chain of Thought）。
    它允许根据一个模型调用的结果动态决定下一个调用的模型和内容。
    """

    def __init__(
        self,
        openai_client: OpenAIClient,
        execution_config: ExecutionConfig | None = None,
    ):
        """
        初始化思想链客户端。

        Args:
            openai_client: 一个 OpenAIClient 实例，用于执行底层的LLM调用。
            execution_config: 执行配置，如果为None则使用默认配置。
        """
        self.openai_client = openai_client
        self.steps: dict[str, Step] = {}
        self.execution_config = execution_config or ExecutionConfig()
        self.monitor: ChainMonitor = DefaultChainMonitor(self.execution_config)
        self._chain_counter = 0

    def set_monitor(self, monitor: ChainMonitor) -> None:
        """设置自定义监控器"""
        self.monitor = monitor

    def add_step(self, step: Step):
        """
        向客户端注册一个步骤。

        Args:
            step: 一个 Step 实例。
        """
        if step.name in self.steps:
            raise ValueError(f"步骤 '{step.name}' 已存在。请确保每个步骤名称唯一。")
        self.steps[step.name] = step

    def add_steps(self, steps: list[Step]):
        """
        向客户端批量注册多个步骤。

        Args:
            steps: Step 实例的列表。
        """
        for step in steps:
            self.add_step(step)

    def create_linear_chain(self, linear_steps: list[LinearStep], chain_name: str = "linear_chain"):
        """
        创建一个线性的步骤链条，每个步骤按顺序执行。

        Args:
            linear_steps: LinearStep 实例的列表，按执行顺序排列。
            chain_name: 链条的名称前缀。
        """
        if not linear_steps:
            raise ValueError("线性链条至少需要一个步骤。")

        def create_next_step_fn(current_index: int, total_steps: int):
            """为线性链条创建next_step函数"""

            def next_step_fn(response: str, context: Context) -> str | None:
                if current_index < total_steps - 1:
                    return f"{chain_name}_{current_index + 1}"
                else:
                    return None  # 结束链条

            return next_step_fn

        # 转换LinearStep为Step并注册
        for i, linear_step in enumerate(linear_steps):
            step_name = f"{chain_name}_{i}"
            full_step = Step(
                name=step_name,
                prepare_messages_fn=linear_step.prepare_messages_fn,
                get_next_step_fn=create_next_step_fn(i, len(linear_steps)),
                model_params=linear_step.model_params,
            )
            self.add_step(full_step)

        return f"{chain_name}_0"  # 返回第一个步骤的名称

    def create_context(self, initial_data: dict[str, Any] | None = None) -> Context:
        """
        创建一个新的上下文对象。

        Args:
            initial_data: 初始数据字典，可以包含 'query' 和其他自定义字段

        Returns:
            新创建的Context对象
        """
        if initial_data is None:
            return Context()

        # 提取特殊字段
        query = initial_data.get("query")

        # 剩余字段作为custom_data
        custom_data = {k: v for k, v in initial_data.items() if k != "query"}

        return Context(query=query, custom_data=custom_data)

    def _generate_chain_id(self) -> str:
        """生成链条ID"""
        self._chain_counter += 1
        # 使用毫秒级时间戳确保ID唯一性
        timestamp_ms = int(time.time() * 1000)
        return f"chain_{self._chain_counter}_{timestamp_ms}"

    async def _execute_step_with_retry(
        self,
        step: Step,
        context: Context,
        controller: ExecutionController,
        step_info: StepExecutionInfo,
        chain_info: ChainExecutionInfo,
        show_step_details: bool = False,
    ) -> str | None:
        """执行单个步骤，包含重试逻辑"""
        last_error = None

        for attempt in range(self.execution_config.max_retries + 1):
            step_info.retry_count = attempt

            try:
                # 准备消息
                messages = step.prepare_messages_fn(context)

                # 显示步骤详细信息 - 输入
                if show_step_details:
                    chain_short_id = chain_info.chain_id.split("_")[-1][-4:]
                    chain_prefix = f"[链条{chain_short_id}] "

                    chain_logger.info(f"{chain_prefix}\n📝 步骤 '{step_info.step_name}' 输入消息:")
                    for i, msg in enumerate(messages):
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        chain_logger.info(
                            f"{chain_prefix}   {i + 1}. [{role}]: {content[:100]}{'...' if len(content) > 100 else ''}"
                        )
                    chain_logger.info(f"{chain_prefix}🔧 模型参数: {step.model_params}")
                    if attempt > 0:
                        chain_logger.warning(f"{chain_prefix}🔄 重试第 {attempt} 次")

                # 执行LLM调用
                start_time = time.time()

                # 创建超时任务
                llm_task = self.openai_client.chat_completions(
                    messages=messages,
                    preprocess_msg=True,
                    show_progress=False,  # LLM调用的进度条始终关闭
                    **step.model_params,
                )

                if self.execution_config.step_timeout:
                    response_content = await asyncio.wait_for(
                        llm_task, timeout=self.execution_config.step_timeout
                    )
                else:
                    response_content = await llm_task

                execution_time = time.time() - start_time
                step_info.execution_time = execution_time

                if response_content is None or not isinstance(response_content, str):
                    raise ValueError("LLM调用返回空响应")

                # 显示步骤详细信息 - 输出
                if show_step_details:
                    chain_short_id = chain_info.chain_id.split("_")[-1][-4:]
                    chain_prefix = f"[链条{chain_short_id}] "
                    chain_logger.success(f"{chain_prefix}✅ 步骤 '{step_info.step_name}' 输出响应:")
                    chain_logger.info(
                        f"{chain_prefix}   📄 响应内容: {response_content[:200]}{'...' if len(response_content) > 200 else ''}"
                    )
                    chain_logger.info(f"{chain_prefix}   ⏱️  执行时间: {execution_time:.3f}秒")
                    if attempt > 0:
                        chain_logger.success(f"{chain_prefix}   🔄 重试成功")

                step_info.status = StepStatus.COMPLETED
                return response_content

            except asyncio.TimeoutError:
                step_info.status = StepStatus.TIMEOUT
                step_info.error = f"步骤执行超时（{self.execution_config.step_timeout}秒）"
                if show_step_details:
                    chain_short_id = chain_info.chain_id.split("_")[-1][-4:]
                    chain_prefix = f"[链条{chain_short_id}] "
                    chain_logger.error(f"{chain_prefix}⏰ 步骤 '{step_info.step_name}' 执行超时")
                    chain_logger.warning(
                        f"{chain_prefix}   ⚠️  超时时间: {self.execution_config.step_timeout}秒"
                    )
                await self.monitor.on_timeout("step", chain_info)
                last_error = TimeoutError(step_info.error)

            except Exception as e:
                step_info.status = StepStatus.FAILED
                step_info.error = str(e)
                if show_step_details:
                    chain_short_id = chain_info.chain_id.split("_")[-1][-4:]
                    chain_prefix = f"[链条{chain_short_id}] "
                    chain_logger.error(f"{chain_prefix}❌ 步骤 '{step_info.step_name}' 执行失败")
                    chain_logger.error(f"{chain_prefix}   🐛 错误类型: {type(e).__name__}")
                    chain_logger.error(f"{chain_prefix}   📝 错误信息: {str(e)}")
                    if attempt < self.execution_config.max_retries:
                        chain_logger.warning(
                            f"{chain_prefix}   🔄 将在 {self.execution_config.retry_delay}秒后重试..."
                        )
                last_error = e
                await self.monitor.on_error(e, chain_info)

            # 如果不是最后一次尝试，等待重试间隔
            if attempt < self.execution_config.max_retries:
                await asyncio.sleep(self.execution_config.retry_delay)

        # 所有重试都失败了
        if last_error:
            raise last_error

        return None

    async def execute_chain(
        self,
        initial_step_name: str,
        initial_context: dict[str, Any] | Context | None = None,
        show_step_details: bool = False,
    ) -> Context:
        """
        异步执行一个完整的思想链。

        Args:
            initial_step_name: 起始步骤的名称。
            initial_context: 传递给第一个步骤的初始上下文，可以是字典或Context对象。
            show_step_details: 是否显示每个步骤的详细信息（输入消息、输出响应、执行时间等）。

        Returns:
            返回包含所有步骤历史记录的最终上下文。
        """
        if initial_step_name not in self.steps:
            raise ValueError(f"起始步骤 '{initial_step_name}' 未注册。")

        # 处理初始上下文
        if isinstance(initial_context, Context):
            context = initial_context
        elif isinstance(initial_context, dict):
            context = self.create_context(initial_context)
        else:
            context = Context()

        # 创建执行信息和控制器
        chain_id = self._generate_chain_id()
        chain_info = ChainExecutionInfo(
            chain_id=chain_id, status=ChainStatus.RUNNING, start_time=time.time()
        )
        context.execution_info = chain_info

        controller = ExecutionController(self.execution_config)

        try:
            await self.monitor.on_chain_start(chain_info)

            current_step_name: str | None = initial_step_name
            chain_start_time = time.time()

            while current_step_name:
                # 检查链条超时
                if self.execution_config.chain_timeout:
                    if await controller.check_timeout(
                        chain_start_time, self.execution_config.chain_timeout
                    ):
                        chain_info.status = ChainStatus.TIMEOUT
                        await self.monitor.on_timeout("chain", chain_info)
                        break

                if current_step_name not in self.steps:
                    raise ValueError(f"执行过程中发现未注册的步骤 '{current_step_name}'。")

                step = self.steps[current_step_name]

                # 创建步骤执行信息
                step_info = StepExecutionInfo(
                    step_name=current_step_name,
                    status=StepStatus.RUNNING,
                    start_time=time.time(),
                )

                chain_info.steps_info.append(step_info)
                await self.monitor.on_step_start(step_info, chain_info)

                try:
                    # 执行步骤（包含重试逻辑）
                    response_content = await self._execute_step_with_retry(
                        step,
                        context,
                        controller,
                        step_info,
                        chain_info,
                        show_step_details,
                    )

                    if response_content is None:
                        break  # 步骤执行失败或被取消

                    step_info.end_time = time.time()
                    step_info.execution_time = step_info.end_time - (step_info.start_time or 0)

                    # 记录步骤结果
                    step_result = StepResult(
                        step_name=current_step_name,
                        messages=step.prepare_messages_fn(context),
                        response=response_content,
                        model_params=step.model_params,
                        execution_time=step_info.execution_time,
                        status=step_info.status,
                        retry_count=step_info.retry_count,
                        error=step_info.error,
                    )
                    context.history.append(step_result)

                    chain_info.completed_steps += 1
                    await self.monitor.on_step_end(step_info, chain_info)

                    # 决定下一步
                    next_step_name = step.get_next_step_fn(response_content, context)
                    current_step_name = next_step_name

                except Exception as e:
                    step_info.status = StepStatus.FAILED
                    step_info.error = str(e)
                    step_info.end_time = time.time()

                    await self.monitor.on_step_end(step_info, chain_info)
                    await self.monitor.on_error(e, chain_info)

                    chain_info.status = ChainStatus.FAILED
                    chain_info.error = str(e)
                    break

            # 设置链条结束状态
            chain_info.end_time = time.time()
            chain_info.total_execution_time = chain_info.end_time - chain_info.start_time

            if chain_info.status == ChainStatus.RUNNING:
                chain_info.status = ChainStatus.COMPLETED

            await self.monitor.on_chain_end(chain_info)

        except Exception as e:
            chain_info.status = ChainStatus.FAILED
            chain_info.error = str(e)
            chain_info.end_time = time.time()
            if chain_info.start_time:
                chain_info.total_execution_time = chain_info.end_time - chain_info.start_time

            await self.monitor.on_error(e, chain_info)
            await self.monitor.on_chain_end(chain_info)
            raise

        return context

    async def execute_chains_batch(
        self,
        chain_requests: list[dict[str, Any]],
        show_step_details: bool = False,
        show_progress: bool = True,
    ) -> list[Context]:
        """
        并发执行多个思想链。

        Args:
            chain_requests: 一个请求列表，每个请求是一个字典，包含 'initial_step_name' 和 'initial_context'。
                例如: [{'initial_step_name': 'step1', 'initial_context': {'query': '你好'}}]
            show_step_details: 是否显示每个步骤的详细信息（输入消息、输出响应、执行时间等）。
            show_progress: 是否显示批处理进度条。

        Returns:
            一个结果列表，每个元素是对应调用链的最终上下文。
        """
        total_chains = len(chain_requests)
        if total_chains == 0:
            return []

        batch_start_time = time.time()
        completed_count = 0

        # 创建进度跟踪器
        progress_tracker = ChainProgressTracker(total_chains, show_progress)

        if show_progress and progress_tracker.show_progress:
            safe_chain_print(
                f"[bold green]🚀 开始执行 {total_chains} 个链条的批处理...[/bold green]"
            )
            safe_chain_print(f"[dim]{'=' * 80}[/dim]")

        # 包装任务以便跟踪进度
        async def wrapped_execute_chain(request_index: int, request: dict[str, Any]):
            try:
                result = await self.execute_chain(
                    initial_step_name=request["initial_step_name"],
                    initial_context=request.get("initial_context"),
                    show_step_details=show_step_details,
                )
                return request_index, result, None
            except Exception as e:
                # 创建错误上下文
                error_context = Context()
                error_context.execution_info = ChainExecutionInfo(
                    chain_id=self._generate_chain_id(),
                    status=ChainStatus.FAILED,
                    error=str(e),
                )
                return request_index, error_context, e

        # 创建任务列表
        tasks = []
        for i, request in enumerate(chain_requests):
            task = wrapped_execute_chain(i, request)
            tasks.append(task)

        # 执行任务并收集结果
        final_results = [None] * total_chains  # 预分配结果列表

        try:
            # 使用asyncio.as_completed来获得实时进度更新
            for future in asyncio.as_completed(tasks):
                request_index, result, error = await future
                final_results[request_index] = result
                completed_count += 1

                # 计算执行时间
                execution_time = 0.0
                success = False
                if result and result.execution_info:
                    execution_time = result.execution_info.total_execution_time or 0.0
                    success = result.execution_info.status.value == "completed"

                # 更新进度跟踪器
                progress_tracker.update(
                    chain_index=request_index,
                    success=success,
                    execution_time=execution_time,
                )

                # 如果启用了监控，记录批处理进度（只在不显示进度条时输出）
                if self.execution_config.enable_monitoring and not (
                    show_progress and progress_tracker.show_progress
                ):
                    progress_info = progress_tracker.get_progress_info()
                    chain_logger.info(
                        f"批处理进度: {completed_count}/{total_chains} ({progress_info['progress_percent']:.1f}%) - "
                        f"已用时: {progress_info['elapsed_time']:.2f}秒, "
                        f"预计剩余: {progress_info['remaining_time']:.2f}秒"
                    )

        finally:
            progress_tracker.finish()

        # 最终统计信息
        progress_info = progress_tracker.get_progress_info()
        successful_chains = sum(
            1
            for result in final_results
            if result
            and result.execution_info
            and result.execution_info.status == ChainStatus.COMPLETED
        )
        failed_chains = total_chains - successful_chains

        if show_progress and progress_tracker.show_progress:
            safe_chain_print(f"[dim]{'=' * 80}[/dim]")

            if chain_console:
                # 使用Rich Panel创建美观的结果显示
                try:
                    result_text = Text()
                    result_text.append("批处理执行完成!\n\n", style="bold green")
                    result_text.append(f"📊 总计: {total_chains} 个链条\n", style="cyan")
                    result_text.append(
                        f"⏱️  总耗时: {progress_info['elapsed_time']:.2f}秒\n",
                        style="cyan",
                    )
                    result_text.append(f"✅ 成功: {successful_chains} 个\n", style="green")
                    result_text.append(
                        f"❌ 失败: {failed_chains} 个\n",
                        style="red" if failed_chains > 0 else "dim",
                    )
                    result_text.append(
                        f"📈 成功率: {successful_chains / total_chains * 100:.1f}%\n",
                        style="yellow",
                    )
                    result_text.append(
                        f"⚡ 平均速率: {progress_info['rate']:.2f} 链/秒", style="blue"
                    )

                    panel = Panel(
                        result_text,
                        title="[bold blue]Chain of Thought 执行结果[/bold blue]",
                        border_style="green" if failed_chains == 0 else "yellow",
                    )
                    chain_console.print(panel)
                except Exception:
                    # fallback到简单输出
                    safe_chain_print("[bold green]批处理执行完成![/bold green]")
                    safe_chain_print(f"[cyan]📊 总计: {total_chains} 个链条[/cyan]")
                    safe_chain_print(
                        f"[cyan]⏱️  总耗时: {progress_info['elapsed_time']:.2f}秒[/cyan]"
                    )
                    safe_chain_print(f"[green]✅ 成功: {successful_chains} 个[/green]")
                    safe_chain_print(
                        f"[red]❌ 失败: {failed_chains} 个[/red]"
                        if failed_chains > 0
                        else f"[dim]❌ 失败: {failed_chains} 个[/dim]"
                    )
                    safe_chain_print(
                        f"[yellow]📈 成功率: {successful_chains / total_chains * 100:.1f}%[/yellow]"
                    )
                    safe_chain_print(f"[blue]⚡ 平均速率: {progress_info['rate']:.2f} 链/秒[/blue]")
            else:
                # 没有Rich库时的简单输出
                safe_chain_print("批处理执行完成!")
                safe_chain_print(f"总计: {total_chains} 个链条")
                safe_chain_print(f"总耗时: {progress_info['elapsed_time']:.2f}秒")
                safe_chain_print(f"成功: {successful_chains} 个")
                safe_chain_print(f"失败: {failed_chains} 个")
                safe_chain_print(f"成功率: {successful_chains / total_chains * 100:.1f}%")
                safe_chain_print(f"平均速率: {progress_info['rate']:.2f} 链/秒")

        if self.execution_config.enable_monitoring and not (
            show_progress and progress_tracker.show_progress
        ):
            chain_logger.info(
                f"批处理完成 - 总耗时: {progress_info['elapsed_time']:.2f}秒, "
                f"成功: {successful_chains}, 失败: {failed_chains}, "
                f"成功率: {successful_chains / total_chains * 100:.1f}%"
            )

        return final_results
