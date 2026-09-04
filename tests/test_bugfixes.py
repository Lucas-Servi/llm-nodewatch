"""Tests verifying bug fixes: negative duration, cache cost, reporter header."""

import time

from nodewatch import LLMCall, NodeSpan, RunTrace
from nodewatch.reporter import trace_to_markdown


def test_duration_zero_when_end_time_unset():
    """NodeSpan.duration_ms should be 0 when end_time was never set (stays 0.0)."""
    span = NodeSpan(node_name="test", start_time=time.time(), end_time=0.0)
    assert span.duration_ms == 0.0


def test_duration_zero_when_end_before_start():
    """NodeSpan.duration_ms should be 0 when end_time < start_time."""
    span = NodeSpan(node_name="test", start_time=100.0, end_time=50.0)
    assert span.duration_ms == 0.0


def test_duration_positive_normal_case():
    span = NodeSpan(node_name="test", start_time=100.0, end_time=102.5)
    assert span.duration_ms == 2500.0


def test_cost_includes_cache_tokens():
    """Cache read tokens should be priced at 10% of input, creation at 125%."""
    call = LLMCall(
        node_name="test",
        model="claude-sonnet-4-6",
        provider="anthropic",
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=2000,
        cache_creation_tokens=400,
    )
    # input: 1000 * 3.0 = 3000
    # cache_read: 2000 * 3.0 * 0.1 = 600
    # cache_creation: 400 * 3.0 * 1.25 = 1500
    # output: 500 * 15.0 = 7500
    # total = 12600 / 1_000_000 = 0.0126
    expected = (1000 * 3.0 + 2000 * 3.0 * 0.1 + 400 * 3.0 * 1.25 + 500 * 15.0) / 1_000_000
    assert abs(call.cost_usd - expected) < 1e-10


def test_cost_includes_thinking_tokens():
    """Thinking tokens should be priced at output rate."""
    call = LLMCall(
        node_name="test",
        model="claude-opus-4-7",
        provider="anthropic",
        input_tokens=1000,
        output_tokens=200,
        thinking_tokens=5000,
    )
    # claude-opus-4-7 pricing: input 5.0, output 25.0; thinking billed at the output rate
    expected = (1000 * 5.0 + 200 * 25.0 + 5000 * 25.0) / 1_000_000
    assert abs(call.cost_usd - expected) < 1e-10


def test_reporter_header_says_model():
    """The markdown table header should say 'Model', not 'Type'."""
    trace = RunTrace(
        graph_name="test",
        query="hello",
        total_duration_ms=1000,
        node_spans=[
            NodeSpan(
                node_name="agent",
                start_time=100.0,
                end_time=101.0,
                llm_calls=[LLMCall(node_name="agent", model="claude-sonnet-4-6", provider="anthropic", input_tokens=100, output_tokens=50)],
            )
        ],
    )
    md = trace_to_markdown(trace, include_nodes=True)
    assert "| Node | Model |" in md
    assert "| Node | Type |" not in md
