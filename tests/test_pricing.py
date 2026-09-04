"""Tests for cost computation and the unpriced-model warning."""

import logging

import pytest

import nodewatch.models as models
from nodewatch.models import PRICING_PER_MTOK, LLMCall, prices_for_model


def test_known_model_costs_more_than_zero():
    call = LLMCall("agent", "gemini-2.5-pro", "google", input_tokens=1000, output_tokens=500)
    assert call.cost_usd > 0


def test_cache_token_cost_math():
    # claude-sonnet-4-6 -> [input, output, cache_read, cache_creation] = [3.0, 15.0, 0.3, 3.75]
    call = LLMCall(
        "agent",
        "claude-sonnet-4-6",
        "anthropic",
        input_tokens=1000,
        output_tokens=200,
        cache_read_tokens=500,
        cache_creation_tokens=100,
    )
    expected = (1000 * 3.0 + 500 * 0.3 + 100 * 3.75 + 200 * 15.0) / 1_000_000
    assert call.cost_usd == expected


def test_unknown_model_costs_zero_and_warns_once(caplog):
    models._UNPRICED_WARNED.discard("brand-new-unpriced-model")
    call = LLMCall("agent", "brand-new-unpriced-model", "unknown", input_tokens=1000, output_tokens=200)

    with caplog.at_level(logging.WARNING, logger="nodewatch.models"):
        assert call.cost_usd == 0.0
        # A second access on the same model id must not emit a second warning.
        again = LLMCall("agent", "brand-new-unpriced-model", "unknown", input_tokens=5, output_tokens=5)
        assert again.cost_usd == 0.0

    warnings = [r for r in caplog.records if "no pricing for model" in r.getMessage()]
    assert len(warnings) == 1
    assert "brand-new-unpriced-model" in warnings[0].getMessage()


def test_zero_token_unpriced_call_does_not_warn(caplog):
    models._UNPRICED_WARNED.discard("another-unpriced-model")
    call = LLMCall("agent", "another-unpriced-model", "unknown")  # no tokens
    with caplog.at_level(logging.WARNING, logger="nodewatch.models"):
        assert call.cost_usd == 0.0
    assert not [r for r in caplog.records if "no pricing for model" in r.getMessage()]


# ── Longest-match lookup ────────────────────────────────────────────────────
#
# Matching is by substring so that decorated served-model ids resolve, which
# means a short key can be a substring of an unrelated model. Picking the first
# match made billing depend on key order in a JSON file users may replace.

def test_short_key_does_not_hijack_an_unrelated_model():
    """'o3' is a substring of many ids; it must not price them as o3."""
    assert "o3" in PRICING_PER_MTOK, "fixture assumes a short 'o3' key exists"
    o3_prices = PRICING_PER_MTOK["o3"]

    # A model that merely CONTAINS "o3" and is priced by a longer key of its own.
    hijackable = LLMCall("agent", "claude-sonnet-4-6-o3-preview", "anthropic", input_tokens=1_000_000)
    assert hijackable.cost_usd != o3_prices[0], "billed at o3 rates via a substring match"
    assert hijackable.cost_usd == PRICING_PER_MTOK["claude-sonnet-4-6"][0]


def test_longer_key_wins_over_its_own_prefix():
    """gpt-5 is a prefix of gpt-5.5; the more specific row must win."""
    assert prices_for_model("gpt-5.5") == PRICING_PER_MTOK["gpt-5.5"]
    assert prices_for_model("gpt-5") == PRICING_PER_MTOK["gpt-5"]
    assert PRICING_PER_MTOK["gpt-5.5"] != PRICING_PER_MTOK["gpt-5"], "fixture needs distinct prices"


def test_decorated_served_model_ids_still_resolve():
    """Vendor/region decoration the table does not repeat must still match."""
    assert prices_for_model("us.anthropic.claude-opus-4-8-v1:0") == PRICING_PER_MTOK["claude-opus-4-8"]
    assert prices_for_model("openai.gpt-5.5") == PRICING_PER_MTOK["gpt-5.5"]


@pytest.mark.parametrize("model", [None, "", "definitely-not-a-real-model"])
def test_unmatched_models_return_none(model):
    assert prices_for_model(model) is None
