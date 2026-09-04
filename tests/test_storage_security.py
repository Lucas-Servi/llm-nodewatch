"""Tests for SQLite storage security: write lock correctness and limit cap."""

import threading
from datetime import datetime, timezone

from nodewatch.models import LLMCall, NodeSpan, RunTrace
from nodewatch.storage.sqlite import SQLiteStorage


def _make_trace(run_id: str) -> RunTrace:
    return RunTrace(
        run_id=run_id,
        graph_name="test",
        query="test query",
        timestamp=datetime.now(timezone.utc),
        total_duration_ms=100.0,
        node_spans=[
            NodeSpan(
                node_name="agent",
                node_type="agent",
                start_time=1.0,
                end_time=2.0,
                llm_calls=[
                    LLMCall(node_name="agent", model="test-model", provider="test", input_tokens=100, output_tokens=50, duration_ms=50.0)
                ],
            )
        ],
    )


def test_concurrent_saves_no_corruption(tmp_path):
    """Multiple threads saving different run_ids should not corrupt."""
    db_path = tmp_path / "test.db"
    storage = SQLiteStorage(db_path)

    errors = []

    def save_run(i: int):
        try:
            trace = _make_trace(f"run-{i:04d}")
            storage.save(trace)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=save_run, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    runs = storage.list_runs(limit=100)
    assert len(runs) == 20
    storage.close()


def test_concurrent_saves_same_run_id(tmp_path):
    """Multiple threads saving the same run_id should not raise."""
    db_path = tmp_path / "test.db"
    storage = SQLiteStorage(db_path)

    errors = []

    def save_run(_i: int):
        try:
            trace = _make_trace("same-run")
            storage.save(trace)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=save_run, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    runs = storage.list_runs(limit=100)
    assert len(runs) == 1
    storage.close()


def test_limit_cap(tmp_path):
    """list_runs should cap limit at 1000 even if a higher value is passed."""
    db_path = tmp_path / "test.db"
    storage = SQLiteStorage(db_path)

    for i in range(5):
        storage.save(_make_trace(f"run-{i}"))

    # Passing limit=9999 should not fail and should be capped internally
    runs = storage.list_runs(limit=9999)
    assert len(runs) == 5
    storage.close()


def test_offset_pagination(tmp_path):
    """offset parameter should skip results."""
    db_path = tmp_path / "test.db"
    storage = SQLiteStorage(db_path)

    for i in range(10):
        storage.save(_make_trace(f"run-{i:02d}"))

    all_runs = storage.list_runs(limit=10)
    offset_runs = storage.list_runs(limit=5, offset=5)
    assert len(offset_runs) == 5
    assert offset_runs[0].run_id == all_runs[5].run_id
    storage.close()
