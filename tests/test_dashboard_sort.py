"""Tests for the dashboard's sortable tables (default order, header clicks, indicators)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from textual.widgets import DataTable, TabbedContent

from nodewatch.dashboard import NodewatchDashboard, _conv_id_key
from nodewatch.models import LLMCall, NodeSpan, RunTrace

CONV_IDS = ["593", "1000", "87", "3c4242b5e052", "", "conv1"]


def test_conv_id_key_sorts_numerically():
    """Numeric ids must not sort as text: "1000" > "593" > "87"."""
    numeric = ["593", "1000", "87"]
    assert sorted(numeric, key=lambda v: _conv_id_key(v, False)) == ["87", "593", "1000"]
    assert sorted(numeric, key=lambda v: _conv_id_key(v, True), reverse=True) == ["1000", "593", "87"]


@pytest.mark.parametrize("reverse", [False, True])
def test_conv_id_key_keeps_non_numeric_last_in_both_directions(reverse):
    """Run-id fallbacks and empty ids sink to the bottom whichever way the sort runs."""
    ordered = sorted(CONV_IDS, key=lambda v: _conv_id_key(v, reverse), reverse=reverse)
    numeric_positions = [i for i, v in enumerate(ordered) if v.isdigit()]
    assert numeric_positions == [0, 1, 2], ordered


def _trace(run_id: str, conv_id: str, tokens: int, minutes_ago: int = 0) -> RunTrace:
    span = NodeSpan(node_name="agent", start_time=1.0, end_time=2.0)
    span.llm_calls.append(
        LLMCall("agent", "claude-sonnet-4-6", "anthropic", input_tokens=tokens, output_tokens=0)
    )
    trace = RunTrace(
        run_id=run_id,
        graph_name="v2",
        query=f"query {run_id}",
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
        total_duration_ms=1000.0,
        final_response="done",
        conversation_id=conv_id,
    )
    trace.node_spans.append(span)
    return trace


class _StubStorage:
    """Minimal storage stand-in: enough surface for the dashboard's startup path."""

    # Token counts are chosen so a *string* sort would disagree with a numeric one
    # ("9,000" > "90,000" lexicographically).
    TRACES = [
        _trace("r_a", "593", 9_000, minutes_ago=30),
        _trace("r_b", "1000", 90_000, minutes_ago=10),
        _trace("r_c", "87", 500, minutes_ago=20),
        _trace("r_d", "", 100, minutes_ago=0),
    ]

    def list_runs(self, **kwargs):
        return list(self.TRACES)

    def load(self, run_id):
        return None

    def get_status(self, run_id):
        return "done"

    def get_logs(self, position=-1):
        return {"lines": [], "position": 0}


def _labels(table: DataTable) -> list[str]:
    return [c.label.plain for c in table.ordered_columns]


def _row_keys(table: DataTable) -> list[str]:
    return [r.key.value for r in table.ordered_rows]


async def _boot(pilot) -> None:
    """Let the startup worker land and populate the tables."""
    await pilot.pause()
    await pilot.pause()


async def _show_tab(pilot, tab_id: str) -> None:
    """Activate a tab so its table is visible — pilot.click() can't hit a hidden widget."""
    pilot.app.query_one(TabbedContent).active = tab_id
    await pilot.pause()


@pytest.mark.asyncio
async def test_conversations_default_to_conv_id_descending():
    app = NodewatchDashboard(_StubStorage())
    async with app.run_test(size=(120, 30)) as pilot:
        await _boot(pilot)
        table = app.query_one("#convs-table", DataTable)
        # Highest conv id first; the run-id fallback for the blank conv id trails.
        assert _row_keys(table) == ["1000", "593", "87", "r_d"]
        assert _labels(table)[0] == "Conv ID ▼"


@pytest.mark.asyncio
async def test_clicking_conv_id_header_toggles_direction():
    app = NodewatchDashboard(_StubStorage())
    async with app.run_test(size=(120, 30)) as pilot:
        await _boot(pilot)
        await _show_tab(pilot, "tab-convs")
        table = app.query_one("#convs-table", DataTable)

        await pilot.click(table, offset=(2, 0))
        await pilot.pause()
        assert _row_keys(table) == ["87", "593", "1000", "r_d"]
        assert _labels(table)[0] == "Conv ID ▲"

        await pilot.click(table, offset=(2, 0))
        await pilot.pause()
        assert _row_keys(table) == ["1000", "593", "87", "r_d"]
        assert _labels(table)[0] == "Conv ID ▼"


@pytest.mark.asyncio
async def test_conversations_sort_by_tokens_numerically():
    """Cells are pre-formatted ("9,000" / "90,000"), so a text sort would fail this."""
    app = NodewatchDashboard(_StubStorage())
    async with app.run_test(size=(120, 30)) as pilot:
        await _boot(pilot)
        app._sort_by("convs-table", 2)  # Tokens
        await pilot.pause()
        table = app.query_one("#convs-table", DataTable)
        assert _row_keys(table) == ["1000", "593", "87", "r_d"]
        assert _labels(table)[2] == "Tokens ▼"

        app._sort_by("convs-table", 2)
        await pilot.pause()
        assert _row_keys(table) == ["r_d", "87", "593", "1000"]
        assert _labels(table)[2] == "Tokens ▲"


@pytest.mark.asyncio
async def test_runs_table_defaults_to_date_descending_and_sorts():
    app = NodewatchDashboard(_StubStorage())
    async with app.run_test(size=(120, 30)) as pilot:
        await _boot(pilot)
        table = app.query_one("#runs-table", DataTable)
        assert _row_keys(table) == ["r_d", "r_b", "r_c", "r_a"]
        assert _labels(table)[6] == "Date ▼"

        app._sort_by("runs-table", 3)  # Tokens
        await pilot.pause()
        assert _row_keys(table) == ["r_b", "r_a", "r_c", "r_d"]
        assert _labels(table)[3] == "Tokens ▼"


@pytest.mark.asyncio
async def test_sort_survives_a_refresh():
    """Re-populating from storage must not reset the user's chosen order."""
    app = NodewatchDashboard(_StubStorage())
    async with app.run_test(size=(120, 30)) as pilot:
        await _boot(pilot)
        app._sort_by("convs-table", 3)  # Cost, descending
        await pilot.pause()

        app._populate_conversations(_StubStorage.TRACES)
        await pilot.pause()
        table = app.query_one("#convs-table", DataTable)
        assert _labels(table)[3] == "Cost ▼"


@pytest.mark.asyncio
async def test_keyboard_sort_actions_target_the_focused_table():
    app = NodewatchDashboard(_StubStorage())
    async with app.run_test(size=(120, 30)) as pilot:
        await _boot(pilot)
        app.query_one("#convs-table", DataTable).focus()
        await pilot.pause()

        await pilot.press("s")  # advance to the next column
        await pilot.pause()
        table = app.query_one("#convs-table", DataTable)
        assert _labels(table)[1] == "Turns ▼"

        await pilot.press("S")  # reverse in place
        await pilot.pause()
        assert _labels(table)[1] == "Turns ▲"
