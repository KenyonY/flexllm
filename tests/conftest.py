"""Test configuration and fixtures"""

import os

import pytest

from tests.mock_server import MockLLMServer, MockLLMServerGroup, MockServerConfig


# Skip tests if API keys not configured
def get_api_key(env_var: str) -> str:
    """Get API key from environment variable"""
    key = os.environ.get(env_var)
    if not key:
        pytest.skip(f"{env_var} not set")
    return key


@pytest.fixture
def gemini_config():
    """Gemini API configuration"""
    return {
        "model": "gemini-3-flash-preview",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api_key": get_api_key("GEMINI_API_KEY"),
    }


@pytest.fixture
def siliconflow_config():
    """SiliconFlow API configuration"""
    return {
        "model": "deepseek-ai/DeepSeek-V3.2",
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key": get_api_key("SILICONFLOW_API_KEY"),
    }


@pytest.fixture
def simple_messages():
    """Simple test messages"""
    return [{"role": "user", "content": "1+1=? Answer with just the number."}]


@pytest.fixture
def batch_messages():
    """Batch test messages"""
    return [
        [{"role": "user", "content": "1+1=?"}],
        [{"role": "user", "content": "2+2=?"}],
        [{"role": "user", "content": "3+3=?"}],
    ]


# ============== Mock LLM Server Fixtures ==============


@pytest.fixture
def mock_llm_server():
    """单个 Mock LLM 服务器（快速响应，延迟 0.1s）"""
    with MockLLMServer(MockServerConfig(port=18001, delay_min=0.1, delay_max=0.1)) as server:
        yield server


@pytest.fixture
def mock_llm_server_slow():
    """单个 Mock LLM 服务器（慢响应，延迟 1-2s）"""
    with MockLLMServer(MockServerConfig(port=18002, delay_min=1.0, delay_max=2.0)) as server:
        yield server


@pytest.fixture
def mock_llm_servers():
    """两个 Mock LLM 服务器（快速响应）"""
    configs = [
        MockServerConfig(port=18001, delay_min=0.1, delay_max=0.1),
        MockServerConfig(port=18002, delay_min=0.1, delay_max=0.1),
    ]
    with MockLLMServerGroup(configs) as group:
        yield group


@pytest.fixture
def mock_llm_servers_slow():
    """两个 Mock LLM 服务器（随机延迟 5-10s，用于测试超时和 fallback）"""
    configs = [
        MockServerConfig(port=18003, delay_min=5.0, delay_max=10.0),
        MockServerConfig(port=18004, delay_min=5.0, delay_max=10.0),
    ]
    with MockLLMServerGroup(configs) as group:
        yield group
