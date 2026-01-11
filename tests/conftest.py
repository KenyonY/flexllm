"""Test configuration and fixtures"""

import pytest
import os

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
