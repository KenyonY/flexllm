from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RequestResult:
    """请求结果的数据类"""
    request_id: int
    data: Any
    status: str
    latency: float
    meta: dict = None
