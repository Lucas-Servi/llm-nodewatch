"""Tests for the storage backend."""

from datetime import datetime, timezone

from nodewatch import LLMCall, NodeSpan, RunTrace, ToolCall
from nodewatch.storage.sqlite import SQLiteStorage


def _make_sample_trace(run_id: str = "abc123", graph_name: str = "v2") -> RunTrace:
    return RunTrace(
        run_id=run_id,
        graph_name=graph_name,
        query="Summarize the latest quarterly report",
        timestamp=datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc),
        total_duration_ms=45200.0,
        node_spans=[
            NodeSpan(
                node_name="coordinator",
                node_type="agent",
                start_time=1000.0,
                end_time=1008.2,
                llm_calls=[
                    LLMCall(
                        node_name="coordinator",
                        model="claude-opus-4-7",
                        provider="anthropic",
                        input_tokens=12000,
                        output_tokens=800,
                        duration_ms=8200.0,
                        stop_reason="end_turn",
                    )
                ],
                iterations=1,
            ),
            NodeSpan(
                node_name="researcher",
                node_type="agent",
                start_time=1008.2,
                end_time=1030.3,
                llm_calls=[
                    LLMCall(
                        node_name="researcher",
                        model="claude-opus-4-7",
                        provider="anthropic",
                        input_tokens=18000,
                        output_tokens=2200,
                        duration_ms=22100.0,
                        stop_reason="tool_use",
                    )
                ],
                tool_calls=[
                    ToolCall(node_name="researcher", tool_name="search_documents", duration_ms=1200.0, success=True),
                    ToolCall(node_name="researcher", tool_name="search_database", duration_ms=800.0, success=True),
                ],
                iterations=3,
            ),
        ],
        final_response="Based on the analysis, revenue grew 12% quarter-over-quarter.",
        metadata={"experiment": "v1-vs-v2"},
    )


def test_save_and_load(tmp_db):
    storage = SQLiteStorage(tmp_db)
    trace = _make_sample_trace()

    storage.save(trace)
    loaded = storage.load("abc123")

    assert loaded is not None
    assert loaded.run_id == "abc123"
    assert loaded.graph_name == "v2"
    assert loaded.total_duration_ms == 45200.0
    assert len(loaded.node_spans) == 2
    assert loaded.node_spans[0].node_name == "coordinator"
    assert loaded.node_spans[0].llm_calls[0].input_tokens == 12000
    assert loaded.node_spans[1].iterations == 3
    assert len(loaded.node_spans[1].tool_calls) == 2

    storage.close()


def test_list_runs(tmp_db):
    storage = SQLiteStorage(tmp_db)
    storage.save(_make_sample_trace("run1", "v1"))
    storage.save(_make_sample_trace("run2", "v2"))
    storage.save(_make_sample_trace("run3", "v2"))

    all_runs = storage.list_runs()
    assert len(all_runs) == 3

    v2_runs = storage.list_runs(graph_name="v2")
    assert len(v2_runs) == 2

    storage.close()


def test_delete(tmp_db):
    storage = SQLiteStorage(tmp_db)
    storage.save(_make_sample_trace("del1"))

    assert storage.delete("del1") is True
    assert storage.load("del1") is None
    assert storage.delete("nonexistent") is False

    storage.close()


def test_load_nonexistent(tmp_db):
    storage = SQLiteStorage(tmp_db)
    assert storage.load("does_not_exist") is None
    storage.close()
