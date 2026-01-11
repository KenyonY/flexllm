"""Tests for token_counter module."""

import pytest

from flexllm import (
    count_tokens,
    count_messages_tokens,
    estimate_cost,
    estimate_batch_cost,
    messages_hash,
    MODEL_PRICING,
)


class TestCountTokens:
    """Test token counting functions."""

    def test_count_tokens_simple(self):
        """Test counting tokens for simple text."""
        text = "Hello, world!"
        tokens = count_tokens(text, model="gpt-4")
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_count_tokens_empty(self):
        """Test counting tokens for empty text."""
        tokens = count_tokens("", model="gpt-4")
        assert tokens == 0 or tokens == 1  # May have minimal token

    def test_count_tokens_long_text(self):
        """Test counting tokens for longer text."""
        text = "This is a longer text. " * 100
        tokens = count_tokens(text, model="gpt-4")
        assert tokens > 100  # Should have many tokens

    def test_count_tokens_chinese(self):
        """Test counting tokens for Chinese text."""
        text = "你好，世界！"
        tokens = count_tokens(text, model="gpt-4")
        assert isinstance(tokens, int)
        assert tokens > 0


class TestCountMessagesTokens:
    """Test message token counting."""

    def test_count_messages_tokens_single(self):
        """Test counting tokens for single message list."""
        messages_list = [[{"role": "user", "content": "Hello!"}]]
        tokens = count_messages_tokens(messages_list, model="gpt-4")
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_count_messages_tokens_multiple(self):
        """Test counting tokens for multiple message lists."""
        messages_list = [
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ],
            [
                {"role": "user", "content": "What is Python?"},
            ],
        ]
        tokens = count_messages_tokens(messages_list, model="gpt-4")
        assert tokens > 10  # Should have reasonable token count

    def test_count_messages_tokens_empty(self):
        """Test counting tokens for empty messages."""
        messages_list = []
        tokens = count_messages_tokens(messages_list, model="gpt-4")
        assert tokens == 0


class TestEstimateCost:
    """Test cost estimation."""

    def test_estimate_cost_input_only(self):
        """Test cost estimation for input tokens only."""
        cost = estimate_cost(input_tokens=1000, output_tokens=0, model="gpt-4o")
        assert isinstance(cost, float)
        assert cost > 0

    def test_estimate_cost_with_output(self):
        """Test cost estimation with both input and output."""
        cost = estimate_cost(input_tokens=1000, output_tokens=500, model="gpt-4o")
        assert isinstance(cost, float)
        assert cost > 0

    def test_estimate_cost_output_adds_cost(self):
        """Test that output tokens add to cost."""
        input_only = estimate_cost(input_tokens=1000, output_tokens=0, model="gpt-4o")
        with_output = estimate_cost(input_tokens=1000, output_tokens=500, model="gpt-4o")
        assert with_output > input_only

    def test_estimate_cost_unknown_model(self):
        """Test cost estimation for unknown model uses default."""
        cost = estimate_cost(input_tokens=1000, output_tokens=0, model="unknown-model")
        # Should return some default cost
        assert isinstance(cost, float)
        assert cost >= 0


class TestEstimateBatchCost:
    """Test batch cost estimation."""

    def test_estimate_batch_cost(self):
        """Test batch cost estimation returns dict."""
        messages_list = [
            [{"role": "user", "content": "Hello!"}],
            [{"role": "user", "content": "World!"}],
        ]
        result = estimate_batch_cost(messages_list, model="gpt-4o")
        assert isinstance(result, dict)
        assert "estimated_cost_usd" in result
        assert "input_tokens" in result
        assert result["count"] == 2

    def test_estimate_batch_cost_empty(self):
        """Test batch cost estimation for empty list."""
        result = estimate_batch_cost([], model="gpt-4o")
        assert isinstance(result, dict)
        assert result["count"] == 0
        assert result["estimated_cost_usd"] == 0


class TestMessagesHash:
    """Test message hashing."""

    def test_messages_hash_deterministic(self):
        """Test that hash is deterministic."""
        messages = [{"role": "user", "content": "Hello!"}]
        hash1 = messages_hash(messages)
        hash2 = messages_hash(messages)
        assert hash1 == hash2

    def test_messages_hash_different_content(self):
        """Test that different content produces different hash."""
        messages1 = [{"role": "user", "content": "Hello!"}]
        messages2 = [{"role": "user", "content": "World!"}]
        hash1 = messages_hash(messages1)
        hash2 = messages_hash(messages2)
        assert hash1 != hash2

    def test_messages_hash_order_matters(self):
        """Test that message order affects hash."""
        messages1 = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"},
        ]
        messages2 = [
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Hello!"},
        ]
        hash1 = messages_hash(messages1)
        hash2 = messages_hash(messages2)
        assert hash1 != hash2


class TestModelPricing:
    """Test model pricing data."""

    def test_model_pricing_exists(self):
        """Test that MODEL_PRICING is defined."""
        assert isinstance(MODEL_PRICING, dict)
        assert len(MODEL_PRICING) > 0

    def test_model_pricing_structure(self):
        """Test MODEL_PRICING structure."""
        for model, pricing in MODEL_PRICING.items():
            assert isinstance(model, str)
            assert isinstance(pricing, dict)
            assert "input" in pricing
            assert "output" in pricing
            assert isinstance(pricing["input"], (int, float))
            assert isinstance(pricing["output"], (int, float))

    def test_common_models_present(self):
        """Test that common models are in pricing."""
        common_models = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
        for model in common_models:
            assert model in MODEL_PRICING or any(
                model in key for key in MODEL_PRICING.keys()
            )
