"""Tests for model-id -> provider inference (_detect_provider)."""

import pytest

from nodewatch.tracker import _detect_provider


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-opus-4-7", "anthropic"),
        ("claude-sonnet-4-6", "anthropic"),
        # Bedrock-served Claude carries a region routing prefix.
        ("us.anthropic.claude-sonnet-4-6-v1", "bedrock"),
        ("eu.anthropic.claude-opus-4-7", "bedrock"),
        ("ap.anthropic.claude-haiku-4-5", "bedrock"),
        ("gpt-5", "openai"),
        ("gpt-5.4-mini", "openai"),
        ("o3", "openai"),
        ("o4-mini", "openai"),
        ("gemini-2.5-pro", "google"),
        ("models/gemini-1.5-flash", "google"),
        ("mistral-large-latest", "mistral"),
        ("mixtral-8x7b", "mistral"),
        ("codestral-latest", "mistral"),
        ("deepseek-chat", "deepseek"),
        ("deepseek-reasoner", "deepseek"),
        ("command-r-plus", "cohere"),
        ("cohere.command-a", "cohere"),
        ("grok-4", "xai"),
        ("minimax-text-01", "minimax"),
        ("abab6.5s-chat", "minimax"),
        # Open-weight families default to local unless served via Groq.
        ("llama-3.3-70b-versatile", "local"),
        ("groq/llama-3.1-8b-instant", "groq"),
        ("qwen2.5-7b", "local"),
        ("totally-made-up-model", "unknown"),
        ("", "unknown"),
    ],
)
def test_detect_provider(model, expected):
    assert _detect_provider(model) == expected


def test_detect_provider_is_case_insensitive():
    assert _detect_provider("GEMINI-2.5-PRO") == "google"
    assert _detect_provider("Claude-Opus-4-7") == "anthropic"
