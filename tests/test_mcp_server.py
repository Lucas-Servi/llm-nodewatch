"""Tests for the MCP server tools against a seeded SQLite database.

The MCP tools resolve storage from NODEWATCH_DB (or NODEWATCH_URL), so we point
them at a temp DB and call the tool functions directly.
"""

from __future__ import annotations

import pytest

from nodewatch import mcp_server
from nodewatch.models import LLMCall, NodeSpan, RunTrace, ToolCall
from nodewatch.storage.sqlite import SQLiteStorage


def _make_trace(run_id: str, graph: str, conversation_id: str = "", tokens: int = 1000) -> RunTrace:
    span = NodeSpan(node_name="agent", start_time=1.0, end_time=2.0)
    span.llm_calls.append(
        LLMCall("agent", "claude-sonnet-4-6", "anthropic", input_tokens=tokens, output_tokens=200)
    )
    # A realistic tool call with non-trivial input/output: these three fields are
    # what `get_run` must surface (a trace without them is a stopwatch), so the
    # fixture carries real-looking values rather than empty strings.
    span.tool_calls.append(ToolCall(
        "agent", "search", duration_ms=12.0, success=True,
        input='{"search_terms": "ecc, registered dimm", "catalog_ids": "MEM-1042"}',
        output_preview="7 of 30 catalog groups matched",
        output_size=2402,
    ))
    trace = RunTrace(
        run_id=run_id,
        graph_name=graph,
        query=f"query for {run_id}",
        total_duration_ms=1000.0,
        final_response="done",
        conversation_id=conversation_id,
    )
    trace.node_spans.append(span)
    return trace


@pytest.fixture
def seeded_db(tmp_db, monkeypatch):
    monkeypatch.delenv("NODEWATCH_URL", raising=False)
    monkeypatch.setenv("NODEWATCH_DB", tmp_db)
    storage = SQLiteStorage(tmp_db)
    storage.save(_make_trace("run_a", "v1", conversation_id="conv1", tokens=1000))
    storage.save(_make_trace("run_b", "v2", conversation_id="conv1", tokens=5000))
    storage.save(_make_trace("run_c", "v1", conversation_id="conv2", tokens=2000))
    storage.close()
    return tmp_db


def _call(tool):
    """MCPServer may wrap the function; fall back to the underlying callable."""
    return getattr(tool, "fn", tool)


def test_list_runs_returns_summaries(seeded_db):
    runs = _call(mcp_server.list_runs)(limit=10)
    assert len(runs) == 3
    assert {r["run_id"] for r in runs} == {"run_a", "run_b", "run_c"}
    assert all("total_tokens" in r and "cost_usd" in r for r in runs)


def test_list_runs_filters_by_graph(seeded_db):
    runs = _call(mcp_server.list_runs)(graph_name="v1", limit=10)
    assert {r["run_id"] for r in runs} == {"run_a", "run_c"}


def test_get_run_returns_detail(seeded_db):
    detail = _call(mcp_server.get_run)("run_b")
    assert detail["run_id"] == "run_b"
    assert detail["graph_name"] == "v2"
    assert detail["nodes"]
    assert detail["nodes"][0]["node_name"] == "agent"
    assert detail["nodes"][0]["llm_calls"][0]["provider"] == "anthropic"


def test_get_run_missing(seeded_db):
    assert "not found" in _call(mcp_server.get_run)("does-not-exist").lower()

def test_get_run_exposes_tool_input_and_output(seeded_db):
    """Tool args + answer must reach the MCP payload, not just name/duration.

    ToolCall and the storage layer always carried input/output_preview/output_size,
    but `_trace_to_detail` dropped all three, so `get_run` could only report that a
    tool "succeeded". A real truncation bug was invisible through this tool and had
    to be read off the raw server log instead.
    """
    tool = _call(mcp_server.get_run)("run_b")["nodes"][0]["tool_calls"][0]
    assert "MEM-1042" in tool["input"]
    assert tool["output_preview"] == "7 of 30 catalog groups matched"
    assert tool["output_size"] == 2402



def test_get_stats_aggregates(seeded_db):
    stats = _call(mcp_server.get_stats)(limit=100)
    assert stats["run_count"] == 3
    assert stats["avg_cost_usd"] > 0
    assert "models" in stats


def test_find_expensive_runs_by_tokens(seeded_db):
    runs = _call(mcp_server.find_expensive_runs)(top_n=1, metric="tokens")
    assert runs[0]["run_id"] == "run_b"  # 5000-token run is heaviest


def test_list_and_get_conversation(seeded_db):
    convs = _call(mcp_server.list_conversations)(limit=50)
    conv_ids = {c["conversation_id"] for c in convs}
    assert "conv1" in conv_ids and "conv2" in conv_ids

    conv1 = _call(mcp_server.get_conversation)("conv1")
    assert {r["run_id"] for r in conv1} == {"run_a", "run_b"}


def test_get_active_runs_empty(seeded_db):
    # All seeded runs default to "done"; none should be active.
    assert _call(mcp_server.get_active_runs)() == []
