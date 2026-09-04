"""Tests for CLI run-id resolution: prefixes, ambiguity, and conversation-id hints."""

from __future__ import annotations

import pytest
import typer

from nodewatch import cli
from nodewatch.models import LLMCall, NodeSpan, RunTrace
from nodewatch.storage.sqlite import SQLiteStorage


def _make_trace(run_id: str, graph: str = "v2", conversation_id: str = "", tokens: int = 1000) -> RunTrace:
    span = NodeSpan(node_name="agent", start_time=1.0, end_time=2.0)
    span.llm_calls.append(
        LLMCall("agent", "claude-sonnet-4-6", "anthropic", input_tokens=tokens, output_tokens=200)
    )
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
def storage(tmp_db, monkeypatch):
    monkeypatch.delenv("NODEWATCH_URL", raising=False)
    store = SQLiteStorage(tmp_db)
    # ab12cd0/ab12cd1 share the "ab12cd" prefix; "593" is a conversation id, never a run id.
    store.save(_make_trace("ab12cd0000ff", conversation_id="593"))
    store.save(_make_trace("ab12cd1111ee", conversation_id="593"))
    store.save(_make_trace("ff99000000aa", conversation_id="592"))
    yield store
    store.close()


def test_full_run_id_resolves(storage):
    assert cli._resolve_trace(storage, "ff99000000aa").run_id == "ff99000000aa"


def test_unique_prefix_resolves(storage):
    """The prefix form the help text has always advertised."""
    assert cli._resolve_trace(storage, "ff9900").run_id == "ff99000000aa"


def test_ambiguous_prefix_exits_and_lists_candidates(storage, capsys):
    with pytest.raises(typer.Exit) as exc:
        cli._resolve_trace(storage, "ab12cd")
    assert exc.value.exit_code == 1
    out = capsys.readouterr().out
    assert "ambiguous" in out
    assert "ab12cd0000ff" in out and "ab12cd1111ee" in out


def test_conversation_id_gets_an_actionable_hint(storage, capsys):
    with pytest.raises(typer.Exit) as exc:
        cli._resolve_trace(storage, "593")
    assert exc.value.exit_code == 1
    out = capsys.readouterr().out
    assert "conversation ID" in out
    assert "ab12cd0000ff" in out and "ab12cd1111ee" in out
    assert "list-runs -c 593" in out


def test_conversation_id_in_metadata_only_is_still_detected(tmp_db, monkeypatch, capsys):
    """The conversation_id column is often empty, with the real id in metadata."""
    monkeypatch.delenv("NODEWATCH_URL", raising=False)
    store = SQLiteStorage(tmp_db)
    trace = _make_trace("aa11bb22cc33")
    trace.metadata = {"conversation_id": "777"}
    store.save(trace)

    with pytest.raises(typer.Exit):
        cli._resolve_trace(store, "777")
    out = capsys.readouterr().out
    assert "conversation ID" in out and "aa11bb22cc33" in out
    store.close()


def test_unknown_id_keeps_the_plain_not_found_message(storage, capsys):
    with pytest.raises(typer.Exit) as exc:
        cli._resolve_trace(storage, "zzzz")
    assert exc.value.exit_code == 1
    out = capsys.readouterr().out
    assert "not found" in out
    assert "conversation ID" not in out and "ambiguous" not in out


def test_exact_hit_does_not_scan(storage, monkeypatch):
    """An exact id must stay a single lookup — no paging through recent runs."""
    def _boom(*args, **kwargs):
        raise AssertionError("list_runs must not be called for an exact run id")

    monkeypatch.setattr(cli, "_scan_recent_runs", _boom)
    assert cli._resolve_trace(storage, "ff99000000aa").run_id == "ff99000000aa"


def test_scan_pages_until_a_short_batch(monkeypatch):
    """Paging must stop at a short page instead of walking the full page budget."""
    calls = []

    class _Paged:
        def list_runs(self, limit=200, offset=0):
            calls.append(offset)
            if offset == 0:
                return [_make_trace(f"r{i:011x}") for i in range(limit)]
            return [_make_trace("tail00000000")]

    traces = cli._scan_recent_runs(_Paged(), pages=5, page=200)
    assert len(traces) == 201
    assert calls == [0, 200]
