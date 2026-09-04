"""Interactive Textual TUI dashboard for nodewatch.

Launch with: nodewatch dashboard [--db PATH]
Requires: pip install "llm-nodewatch[client]"
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Footer, Header, Static, TabbedContent, TabPane, Tabs, Tree

from .models import RunTrace
from .stats import compute_conversation_stats, compute_summary


class InspectorTree(Tree):
    """Tree with arrow-key expand/collapse."""

    BINDINGS = [
        Binding("right", "expand_node", "Expand", show=False),
        Binding("left", "collapse_node", "Collapse", show=False),
    ]

    def action_expand_node(self) -> None:
        if self.cursor_node and not self.cursor_node.is_expanded:
            self.cursor_node.expand()

    def action_collapse_node(self) -> None:
        if self.cursor_node:
            if self.cursor_node.is_expanded:
                self.cursor_node.collapse()
            elif self.cursor_node.parent:
                self.action_cursor_parent()


def _short_model(model: str) -> str:
    import re
    if not model:
        return "-"
    m = re.search(r"(opus|sonnet|haiku)-(\d+-\d+)", model.lower())
    return f"{m.group(1)}-{m.group(2)}" if m else model


def _fmt_dur(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.1f}s"
    return f"{ms:.0f}ms"


def _conv_id_key(value: str, reverse: bool):
    """Sort key for a conversation id.

    Ids are numeric strings on the live server ("593"), but runs with an empty
    conversation_id are keyed by run_id instead (12 hex chars), so the column is mixed.
    Numeric ids sort numerically ("87" < "593" < "1000", not lexicographically) and
    non-numeric ones stay at the bottom in either direction.
    """
    junk = -1 if reverse else 1
    return (0, int(value), "") if value.isdigit() else (junk, 0, value)


# ── Table specs ───────────────────────────────────────────────────────────────────
# Sorting happens on the model objects, never on the rendered cells: those are
# pre-formatted strings ("1,234", "$0.42", "1.5s") and sorting them compares text.
# The sort key tuples are index-aligned with their column tuple.

_CONV_COLUMNS = ("Conv ID", "Turns", "Tokens", "Cost", "Avg Latency", "Graphs")
_CONV_SORT_KEYS = (
    lambda cs, rev: _conv_id_key(cs.conversation_id, rev),
    lambda cs, rev: cs.turn_count,
    lambda cs, rev: cs.total_tokens,
    lambda cs, rev: cs.total_cost,
    lambda cs, rev: cs.avg_latency_ms,
    lambda cs, rev: ", ".join(cs.graphs_used),
)

_RUNS_COLUMNS = ("Run ID", "Graph", "Query", "Tokens", "Cost", "Duration", "Date")
_RUNS_SORT_KEYS = (
    lambda t, rev: t.run_id,
    lambda t, rev: t.graph_name,
    lambda t, rev: t.query,
    lambda t, rev: t.total_tokens,
    lambda t, rev: t.total_cost_usd,
    lambda t, rev: t.total_duration_ms,
    lambda t, rev: t.timestamp.timestamp(),
)


def _conv_row(cs) -> tuple[str, ...]:
    return (
        cs.conversation_id[:14],
        str(cs.turn_count),
        f"{cs.total_tokens:,}",
        f"${cs.total_cost:.2f}",
        _fmt_dur(cs.avg_latency_ms),
        ", ".join(cs.graphs_used),
    )


def _run_row(t: RunTrace) -> tuple[str, ...]:
    return (
        t.run_id,
        t.graph_name,
        t.query[:40],
        f"{t.total_tokens:,}",
        f"${t.total_cost_usd:.3f}",
        _fmt_dur(t.total_duration_ms),
        t.timestamp.strftime("%Y-%m-%d %H:%M"),
    )


class LivePanel(Static):
    """Auto-refreshing panel showing active/recent run status."""

    def __init__(self, storage):
        super().__init__("Loading live data...")
        self._storage = storage
        self._fetching = False

    def on_mount(self) -> None:
        self._refresh_data()
        self.set_interval(10.0, self._refresh_data)

    def _refresh_data(self) -> None:
        if self._fetching:
            return
        self._fetching = True
        self.run_worker(self._fetch_live(), group="live")

    async def _fetch_live(self) -> None:
        try:
            try:
                recent = await asyncio.to_thread(self._storage.list_runs, limit=1)
            except Exception as e:
                self.update(f"Error fetching runs: {e}")
                return

            if not recent:
                self.update("No runs found.\n\nStart a graph execution to see live metrics here.")
                return

            run_id = recent[0].run_id
            try:
                status = await asyncio.to_thread(self._storage.get_status, run_id) or "done"
            except Exception:
                status = "done"

            try:
                trace = await asyncio.to_thread(self._storage.load, run_id)
            except Exception as e:
                self.update(f"Error loading trace: {e}")
                return

            if not trace:
                self.update("No runs found.")
                return

            is_running = status == "running"
            if is_running:
                self.update(self._render_trace(trace, "running"))
            else:
                _idle_msg = Text("○ NO ACTIVE CONVERSATION RUNNING", style="bold orange1")
                content = Text.assemble(
                    _idle_msg, "\n\nShowing most recent:\n\n",
                    self._render_trace(trace, "done"),
                )
                self.update(content)
        finally:
            self._fetching = False

    def _render_trace(self, trace: RunTrace, status: str) -> str:
        elapsed = trace.total_duration_ms / 1000
        nodes_done = sum(1 for s in trace.node_spans if s.end_time > 0)
        lines = [
            f"{'▶' if status == 'running' else '✓'} {trace.run_id}  |  {trace.graph_name}  |  {status.upper()}  |  {elapsed:.1f}s",
            f"  Tokens: {trace.total_tokens:,}  |  Cost: ${trace.total_cost_usd:.2f}  |  Nodes: {nodes_done}/{len(trace.node_spans)}",
        ]
        if trace.conversation_id:
            lines.append(f"  Conv: {trace.conversation_id}")
        if trace.query:
            query_display = trace.query if len(trace.query) <= 80 else trace.query[:77] + "..."
            lines.append(f"  Query: {query_display}")
        lines.append("")
        for span in trace.node_spans:
            if span.end_time > 0:
                st = "✓"
                dur = _fmt_dur(span.duration_ms)
            elif span.llm_calls or span.tool_calls:
                st = "▶"
                dur = _fmt_dur((time.time() - span.start_time) * 1000) if span.start_time > 0 else "..."
            else:
                st = "○"
                dur = "-"
            tokens = f"{span.total_tokens:,}" if span.total_tokens else "-"
            cost = f"${span.total_cost_usd:.2f}" if span.total_cost_usd > 0 else "-"
            lines.append(f"  {st} {span.node_name:<20} {tokens:>8}  {cost:>7}  {dur:>7}")
            for tc in span.tool_calls:
                tc_icon = "✓" if tc.success else "✗"
                tc_dur = _fmt_dur(tc.duration_ms) if tc.duration_ms > 0 else "-"
                lines.append(f"       {tc_icon} {tc.tool_name:<18}              {tc_dur:>7}")

        return "\n".join(lines)


class LogsPanel(Static):
    """Auto-refreshing panel that tails the server log."""

    def __init__(self, storage):
        super().__init__("Loading logs...", markup=False)
        self._storage = storage
        self._position: int = -1
        self._lines: list[str] = []
        self._fetching = False

    def on_mount(self) -> None:
        self._refresh_logs()
        self.set_interval(3.0, self._refresh_logs)

    def _refresh_logs(self) -> None:
        if self._fetching:
            return
        self._fetching = True
        self.run_worker(self._fetch_logs(), group="logs")

    async def _fetch_logs(self) -> None:
        try:
            result = await asyncio.to_thread(self._storage.get_logs, self._position)
            new_lines = result.get("lines", [])
            new_position = result.get("position", 0)
            error = result.get("error")

            if error and not self._lines:
                self.update(f"Error: {error}")
                return

            if new_lines:
                self._lines.extend(new_lines)
                self._lines = self._lines[-500:]
                self._position = new_position
                self.update("\n".join(self._lines))
            elif not self._lines:
                self.update("No log output yet.")
        finally:
            self._fetching = False


class NodewatchDashboard(App):
    """Interactive nodewatch dashboard."""

    TITLE = "nodewatch dashboard"
    CSS = """
    TabbedContent { height: 100%; }
    TabPane { padding: 1; }
    DataTable { height: 1fr; }
    #inspector-header { height: auto; max-height: 8; }
    #inspector-tree { height: 1fr; }
    #stats-content { height: 1fr; }
    #logs-content { height: 1fr; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("r", "refresh", "Refresh", priority=True),
        Binding("escape", "focus_tabs", "Tabs", show=False, priority=True),
        Binding("tab", "next_tab", "Next Tab", show=False, priority=True),
        Binding("shift+tab", "prev_tab", "Prev Tab", show=False, priority=True),
        Binding("s", "sort_next_column", "Sort col"),
        Binding("S", "sort_reverse", "Reverse", show=False),
        Binding("1", "tab_1", "Live", show=False, priority=True),
        Binding("2", "tab_2", "Runs", show=False, priority=True),
        Binding("3", "tab_3", "Conversations", show=False, priority=True),
        Binding("4", "tab_4", "Inspector", show=False, priority=True),
        Binding("5", "tab_5", "Stats", show=False, priority=True),
        Binding("6", "tab_6", "Logs", show=False, priority=True),
    ]

    def __init__(self, storage):
        super().__init__()
        self._storage = storage
        # Cached rows per table, so re-sorting never refetches: list_runs is an N+1 read
        # locally and a ~1.5s round-trip remotely. Cached per table, not globally, because
        # the runs table can be showing a conversation-filtered subset.
        self._conv_stats: list = []
        self._runs: list[RunTrace] = []
        # table id -> (column index, reverse). Conversations default to Conv ID descending
        # (newest first, ids grow over time); runs keep their historic Date-descending order.
        self._sort: dict[str, tuple[int, bool]] = {
            "convs-table": (0, True),
            "runs-table": (6, True),
        }

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent("Live", "Runs", "Conversations", "Inspector", "Stats", "Logs"):
            with TabPane("Live", id="tab-live"):
                yield VerticalScroll(LivePanel(self._storage))
            with TabPane("Runs", id="tab-runs"):
                yield VerticalScroll(DataTable(id="runs-table"))
            with TabPane("Conversations", id="tab-convs"):
                yield VerticalScroll(DataTable(id="convs-table"))
            with TabPane("Inspector", id="tab-inspector"):
                yield VerticalScroll(
                    Static("Select a run from the Runs tab (Enter) to inspect it.", id="inspector-header"),
                    InspectorTree("Nodes", id="inspector-tree"),
                )
            with TabPane("Stats", id="tab-stats"):
                yield VerticalScroll(Static("Loading...", id="stats-content"))
            with TabPane("Logs", id="tab-logs"):
                yield VerticalScroll(LogsPanel(self._storage))
        yield Footer()

    def on_mount(self) -> None:
        self._render_table("runs-table")
        self._render_table("convs-table")
        self.run_worker(self._initial_load(), exclusive=True, group="data")

    # ── Table rendering & sorting ─────────────────────────────────────────────

    def _render_table(self, table_id: str) -> None:
        """Sort the cached rows for a table and redraw it, header indicator included.

        The columns are rebuilt on every draw because Textual caches header labels:
        mutating ``Column.label`` in place leaves the old text on screen through
        ``refresh()`` and even ``clear()``. That also means column keys are regenerated
        each draw, which is why sort state is tracked by column *index*.
        """
        if table_id == "convs-table":
            columns, keys, items, row_of = _CONV_COLUMNS, _CONV_SORT_KEYS, self._conv_stats, _conv_row
            row_key = lambda cs: cs.conversation_id  # noqa: E731
        else:
            columns, keys, items, row_of = _RUNS_COLUMNS, _RUNS_SORT_KEYS, self._runs, _run_row
            row_key = lambda t: t.run_id  # noqa: E731

        index, reverse = self._sort[table_id]
        rows = sorted(items, key=lambda item: keys[index](item, reverse), reverse=reverse)

        table = self.query_one(f"#{table_id}", DataTable)
        table.clear(columns=True)
        arrow = " ▼" if reverse else " ▲"
        table.add_columns(*(c + arrow if i == index else c for i, c in enumerate(columns)))
        table.cursor_type = "row"
        for item in rows:
            table.add_row(*row_of(item), key=row_key(item))

    def _sort_by(self, table_id: str, index: int) -> None:
        """Sort a table by a column: same column toggles direction, a new one starts descending."""
        current, reverse = self._sort[table_id]
        self._sort[table_id] = (index, not reverse) if index == current else (index, True)
        self._render_table(table_id)

    def _focused_table_id(self) -> str | None:
        focused = self.focused
        table_id = getattr(focused, "id", None)
        return table_id if table_id in self._sort else None

    def action_sort_next_column(self) -> None:
        """Cycle the focused table's sort to the next column (keyboard equivalent of a header click)."""
        table_id = self._focused_table_id()
        if not table_id:
            return
        columns = _CONV_COLUMNS if table_id == "convs-table" else _RUNS_COLUMNS
        index, _ = self._sort[table_id]
        self._sort[table_id] = ((index + 1) % len(columns), True)
        self._render_table(table_id)

    def action_sort_reverse(self) -> None:
        """Reverse the focused table's current sort direction."""
        table_id = self._focused_table_id()
        if not table_id:
            return
        index, reverse = self._sort[table_id]
        self._sort[table_id] = (index, not reverse)
        self._render_table(table_id)

    async def _initial_load(self) -> None:
        """Fetch data in background on startup, populate all tabs."""
        try:
            traces = await asyncio.to_thread(self._storage.list_runs, limit=200)
        except Exception as e:
            self.notify(f"Connection error: {e}", severity="error", timeout=10)
            self.query_one("#stats-content", Static).update(f"Error fetching runs: {e}")
            return
        self._populate_all(traces)

    def _populate_all(self, traces: list[RunTrace]) -> None:
        """Populate all tabs from a list of traces. Runs on the main thread."""
        self._populate_runs(traces)
        self._populate_conversations(traces)
        self._populate_stats(traces)

    def _populate_runs(self, traces: list[RunTrace]) -> None:
        self._runs = list(traces)
        self._render_table("runs-table")

    def _populate_conversations(self, traces: list[RunTrace]) -> None:
        self._conv_stats = compute_conversation_stats(traces)
        self._render_table("convs-table")

    def _populate_stats(self, traces: list[RunTrace]) -> None:
        widget = self.query_one("#stats-content", Static)
        if not traces:
            widget.update("No runs found.")
            return

        stats = compute_summary(traces)
        lines = [
            f"{'═' * 60}",
            f"  Summary ({stats.run_count} runs)",
            f"{'═' * 60}",
            "",
            f"  Avg Cost:      ${stats.avg_cost:.4f}   (min: ${stats.min_cost:.4f}, max: ${stats.max_cost:.4f})",
            f"  Avg Tokens:    {stats.avg_tokens:,}    (min: {stats.min_tokens:,}, max: {stats.max_tokens:,})",
            f"  Avg Latency:   {_fmt_dur(stats.avg_latency_ms)}    (min: {_fmt_dur(stats.min_latency_ms)}, max: {_fmt_dur(stats.max_latency_ms)})",
            f"  Errors:        {stats.error_count}/{stats.run_count}",
            "",
            "  Model Breakdown:",
        ]
        for m in stats.models:
            lines.append(f"    {m.model:<30} {m.total_tokens:>10,} tokens  ${m.total_cost:.4f}")

        lines.extend([
            "",
            f"  Throughput:    {stats.throughput_tokens_per_s:.0f} tokens/s",
            f"  Cost/1k tok:   ${stats.cost_per_1k_tokens:.5f}",
            f"  Tool calls:    {stats.tool_calls_total} total ({stats.tool_calls_success} success)",
        ])

        pricing_path = os.getenv("NODEWATCH_PRICING", str(Path(__file__).parent / "data" / "pricing.json"))
        ppath = Path(pricing_path)
        if ppath.exists():
            data = json.loads(ppath.read_text())
            lines.extend(["", f"{'─' * 60}", "  Pricing ($/Mtok)", f"{'─' * 60}", ""])
            lines.append(f"  {'Model':<20} {'Input':>8} {'Output':>8} {'Cache R':>8} {'Cache C':>8}")
            for model, prices in data.items():
                if model.startswith("_"):
                    continue
                if len(prices) == 2:
                    inp, out = prices
                    cr, cc = inp * 0.1, inp * 1.25
                else:
                    inp, out, cr, cc = prices[:4]
                lines.append(f"  {model:<20} ${inp:>6.2f} ${out:>6.2f} ${cr:>6.3f} ${cc:>6.3f}")

        widget.update("\n".join(lines))

    # ── Event handlers ────────────────────────────────────────────────────────

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        table_id = event.data_table.id
        if table_id in self._sort:
            self._sort_by(table_id, event.column_index)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        table_id = event.data_table.id
        if table_id == "runs-table" and event.row_key:
            self._inspect_run(str(event.row_key.value))
        elif table_id == "convs-table" and event.row_key:
            conv_id = str(event.row_key.value)
            self._filter_runs_by_conversation(conv_id)

    def _inspect_run(self, run_id: str) -> None:
        header = self.query_one("#inspector-header", Static)
        header.update(f"Loading run {run_id}...")
        self.query_one(TabbedContent).active = "tab-inspector"
        self.run_worker(self._load_and_render_inspection(run_id), exclusive=True, group="inspect")

    async def _load_and_render_inspection(self, run_id: str) -> None:
        trace = await asyncio.to_thread(self._storage.load, run_id)
        if not trace:
            self.query_one("#inspector-header", Static).update(f"Run {run_id} not found.")
            return
        self._render_inspection(trace)

    def _render_inspection(self, trace: RunTrace) -> None:
        header = self.query_one("#inspector-header", Static)
        dur = _fmt_dur(trace.total_duration_ms)
        lines = [
            f"Run: {trace.run_id}  |  {trace.graph_name}  |  {trace.timestamp.strftime('%Y-%m-%d %H:%M')}",
            f"Query: {trace.query[:80]}",
            f"Duration: {dur}  |  Tokens: {trace.total_tokens:,}  |  Cost: ${trace.total_cost_usd:.3f}  |  LLMs: {trace.total_llm_calls}  |  Tools: {trace.total_tool_calls}",
        ]
        if trace.error:
            lines.append(f"Error: {trace.error}")

        total_in = trace.total_input_tokens
        total_out = trace.total_output_tokens
        if total_in + total_out > 0:
            bar_w = 30
            in_pct = total_in / (total_in + total_out)
            in_bar = int(in_pct * bar_w)
            out_bar = bar_w - in_bar
            lines.append(f"Tokens: {'█' * in_bar}{'░' * out_bar}  {total_in:,} in / {total_out:,} out ({in_pct*100:.0f}%/{(1-in_pct)*100:.0f}%)")

        lines.append("\n[dim]→ expand  ← collapse  space toggle  ↑↓ navigate[/dim]")
        header.update("\n".join(lines))

        tree = self.query_one("#inspector-tree", InspectorTree)
        tree.clear()
        tree.root.expand()

        for span in trace.node_spans:
            model = _short_model(span.llm_calls[0].model if span.llm_calls else "")
            inp = f"{span.total_input_tokens:,}" if span.total_input_tokens > 0 else "-"
            out = f"{span.total_output_tokens:,}" if span.total_output_tokens > 0 else "-"
            cost = f"${span.total_cost_usd:.3f}" if span.total_cost_usd > 0 else "-"
            dur_s = span.duration_ms
            dur_str = _fmt_dur(dur_s) if dur_s > 0 else "-"
            loops = str(span.iterations) if span.iterations > 0 else "-"

            node_label = f"{span.node_name:<20} {model:<12} {inp:>8} in  {out:>8} out  {cost:>7}  {dur_str:>7}  loops:{loops}"
            node_branch = tree.root.add(node_label, expand=True)

            for tc in span.tool_calls:
                tc_icon = "✓" if tc.success else "✗"
                tc_dur = _fmt_dur(tc.duration_ms) if tc.duration_ms > 0 else "-"
                out_sz = f"{tc.output_size:,}ch" if tc.output_size > 0 else "-"
                tc_label = f"{tc_icon} {tc.tool_name:<24} {tc_dur:>7}  out:{out_sz}"

                has_details = tc.input or tc.output_preview
                if has_details:
                    tc_branch = node_branch.add(tc_label)
                    if tc.input:
                        tc_branch.add_leaf(f"[input] {tc.input}")
                    if tc.output_preview:
                        tc_branch.add_leaf(f"[output] {tc.output_preview}")
                else:
                    node_branch.add_leaf(tc_label)

    def _filter_runs_by_conversation(self, conv_id: str) -> None:
        self.query_one(TabbedContent).active = "tab-runs"
        self.notify(f"Loading conversation {conv_id[:14]}...")
        self.run_worker(self._load_conversation_runs(conv_id), exclusive=True, group="filter")

    async def _load_conversation_runs(self, conv_id: str) -> None:
        traces = await asyncio.to_thread(self._storage.list_runs, conversation_id=conv_id, limit=100)
        self._populate_runs(traces)
        self.notify(f"Filtered to conversation: {conv_id[:14]}")

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_refresh(self) -> None:
        self.notify("Refreshing...")
        self.run_worker(self._background_refresh(), exclusive=True, group="data")

    async def _background_refresh(self) -> None:
        """Fetch fresh data in background, replace table contents when ready."""
        try:
            traces = await asyncio.to_thread(self._storage.list_runs, limit=200)
        except Exception as e:
            self.notify(f"Connection error: {e}", severity="error", timeout=10)
            return
        self._populate_all(traces)
        self.notify("Updated")

    def _focus_active_tab_content(self) -> None:
        """Move focus into the active tab's primary widget."""
        active = self.query_one(TabbedContent).active
        focus_map = {
            "tab-runs": "#runs-table",
            "tab-convs": "#convs-table",
            "tab-inspector": "#inspector-tree",
        }
        target = focus_map.get(active)
        if target:
            self.query_one(target).focus()

    def on_tabbed_content_tab_activated(self, _event: TabbedContent.TabActivated) -> None:
        self._focus_active_tab_content()

    _TAB_IDS = ["tab-live", "tab-runs", "tab-convs", "tab-inspector", "tab-stats", "tab-logs"]

    def action_next_tab(self) -> None:
        active = self.query_one(TabbedContent).active
        idx = self._TAB_IDS.index(active) if active in self._TAB_IDS else 0
        self.query_one(TabbedContent).active = self._TAB_IDS[(idx + 1) % len(self._TAB_IDS)]

    def action_prev_tab(self) -> None:
        active = self.query_one(TabbedContent).active
        idx = self._TAB_IDS.index(active) if active in self._TAB_IDS else 0
        self.query_one(TabbedContent).active = self._TAB_IDS[(idx - 1) % len(self._TAB_IDS)]

    def action_focus_tabs(self) -> None:
        self.query_one(Tabs).focus()

    def action_tab_1(self) -> None:
        self.query_one(TabbedContent).active = "tab-live"

    def action_tab_2(self) -> None:
        self.query_one(TabbedContent).active = "tab-runs"

    def action_tab_3(self) -> None:
        self.query_one(TabbedContent).active = "tab-convs"

    def action_tab_4(self) -> None:
        self.query_one(TabbedContent).active = "tab-inspector"

    def action_tab_5(self) -> None:
        self.query_one(TabbedContent).active = "tab-stats"

    def action_tab_6(self) -> None:
        self.query_one(TabbedContent).active = "tab-logs"


def run_dashboard(storage) -> None:
    """Entry point for the dashboard TUI."""
    app = NodewatchDashboard(storage)
    app.run()
