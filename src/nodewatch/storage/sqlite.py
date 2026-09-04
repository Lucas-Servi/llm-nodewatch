"""SQLite storage backend for RunTrace persistence."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from ..models import LLMCall, NodeSpan, RunTrace, ToolCall

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    graph_name TEXT NOT NULL,
    query TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    total_duration_ms REAL NOT NULL,
    final_response TEXT,
    error TEXT,
    metadata TEXT,
    conversation_id TEXT DEFAULT '',
    status TEXT DEFAULT 'done',
    last_heartbeat TEXT
);

CREATE TABLE IF NOT EXISTS node_spans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    node_name TEXT NOT NULL,
    node_type TEXT NOT NULL,
    start_time REAL,
    end_time REAL,
    iterations INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    node_name TEXT NOT NULL,
    model TEXT,
    provider TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    thinking_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_creation_tokens INTEGER DEFAULT 0,
    duration_ms REAL DEFAULT 0,
    stop_reason TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    node_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    duration_ms REAL DEFAULT 0,
    success INTEGER DEFAULT 1,
    error TEXT,
    input TEXT,
    output_preview TEXT,
    output_size INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_runs_graph ON runs(graph_name);
CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_node_spans_run ON node_spans(run_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_run ON tool_calls(run_id);
"""


class SQLiteStorage:
    """Zero-infrastructure SQLite storage for benchmark runs."""

    def __init__(self, db_path: str | Path = "nodewatch.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()
        self._write_lock = threading.Lock()

    def _migrate(self):
        """Add columns that may not exist in older databases."""
        c = self._conn.cursor()
        c.execute("PRAGMA table_info(runs)")
        cols = {row[1] for row in c.fetchall()}
        if "conversation_id" not in cols:
            c.execute("ALTER TABLE runs ADD COLUMN conversation_id TEXT DEFAULT ''")
        if "status" not in cols:
            c.execute("ALTER TABLE runs ADD COLUMN status TEXT DEFAULT 'done'")
        if "last_heartbeat" not in cols:
            c.execute("ALTER TABLE runs ADD COLUMN last_heartbeat TEXT")
        c.execute("CREATE INDEX IF NOT EXISTS idx_runs_conversation ON runs(conversation_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")

        c.execute("PRAGMA table_info(tool_calls)")
        tc_cols = {row[1] for row in c.fetchall()}
        if "input" not in tc_cols:
            c.execute("ALTER TABLE tool_calls ADD COLUMN input TEXT")
        if "output_preview" not in tc_cols:
            c.execute("ALTER TABLE tool_calls ADD COLUMN output_preview TEXT")
        if "output_size" not in tc_cols:
            c.execute("ALTER TABLE tool_calls ADD COLUMN output_size INTEGER DEFAULT 0")

    def save(self, trace: RunTrace, status: str = "done") -> None:
        with self._write_lock:
            now = datetime.now(timezone.utc).isoformat()
            c = self._conn.cursor()

            c.execute("DELETE FROM node_spans WHERE run_id = ?", (trace.run_id,))
            c.execute("DELETE FROM llm_calls WHERE run_id = ?", (trace.run_id,))
            c.execute("DELETE FROM tool_calls WHERE run_id = ?", (trace.run_id,))

            c.execute(
                "INSERT OR REPLACE INTO runs (run_id, graph_name, query, timestamp, total_duration_ms, final_response, error, metadata, conversation_id, status, last_heartbeat) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace.run_id,
                    trace.graph_name,
                    trace.query,
                    trace.timestamp.isoformat(),
                    trace.total_duration_ms,
                    trace.final_response,
                    trace.error,
                    json.dumps(trace.metadata),
                    trace.conversation_id,
                    status,
                    now,
                ),
            )

            for span in trace.node_spans:
                c.execute(
                    "INSERT INTO node_spans (run_id, node_name, node_type, start_time, end_time, iterations) VALUES (?, ?, ?, ?, ?, ?)",
                    (trace.run_id, span.node_name, span.node_type, span.start_time, span.end_time, span.iterations),
                )
                for llm in span.llm_calls:
                    c.execute(
                        "INSERT INTO llm_calls (run_id, node_name, model, provider, input_tokens, output_tokens, thinking_tokens, cache_read_tokens, cache_creation_tokens, duration_ms, stop_reason, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (trace.run_id, llm.node_name, llm.model, llm.provider, llm.input_tokens, llm.output_tokens, llm.thinking_tokens, llm.cache_read_tokens, llm.cache_creation_tokens, llm.duration_ms, llm.stop_reason, llm.error),
                    )
                for tc in span.tool_calls:
                    c.execute(
                        "INSERT INTO tool_calls (run_id, node_name, tool_name, duration_ms, success, error, input, output_preview, output_size) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (trace.run_id, tc.node_name, tc.tool_name, tc.duration_ms, int(tc.success), tc.error, tc.input, tc.output_preview, tc.output_size),
                    )

            self._conn.commit()

    def load(self, run_id: str) -> RunTrace | None:
        c = self._conn.cursor()
        c.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = c.fetchone()
        if not row:
            return None

        run_id, graph_name, query, timestamp_str, total_duration_ms, final_response, error, metadata_json = row[:8]
        conversation_id = row[8] if len(row) > 8 else ""

        # Load node spans
        c.execute("SELECT node_name, node_type, start_time, end_time, iterations FROM node_spans WHERE run_id = ?", (run_id,))
        span_rows = c.fetchall()

        # Load LLM calls
        c.execute("SELECT node_name, model, provider, input_tokens, output_tokens, thinking_tokens, cache_read_tokens, cache_creation_tokens, duration_ms, stop_reason, error FROM llm_calls WHERE run_id = ?", (run_id,))
        llm_rows = c.fetchall()

        # Load tool calls
        c.execute("SELECT node_name, tool_name, duration_ms, success, error, input, output_preview, output_size FROM tool_calls WHERE run_id = ?", (run_id,))
        tool_rows = c.fetchall()

        # Group by node
        llm_by_node: dict[str, list[LLMCall]] = {}
        for r in llm_rows:
            call = LLMCall(node_name=r[0], model=r[1], provider=r[2], input_tokens=r[3], output_tokens=r[4], thinking_tokens=r[5], cache_read_tokens=r[6], cache_creation_tokens=r[7], duration_ms=r[8], stop_reason=r[9], error=r[10])
            llm_by_node.setdefault(r[0], []).append(call)

        tool_by_node: dict[str, list[ToolCall]] = {}
        for r in tool_rows:
            call = ToolCall(node_name=r[0], tool_name=r[1], duration_ms=r[2], success=bool(r[3]), error=r[4], input=r[5], output_preview=r[6], output_size=r[7] or 0)
            tool_by_node.setdefault(r[0], []).append(call)

        spans = []
        for sr in span_rows:
            spans.append(NodeSpan(
                node_name=sr[0],
                node_type=sr[1],
                start_time=sr[2],
                end_time=sr[3],
                llm_calls=llm_by_node.get(sr[0], []),
                tool_calls=tool_by_node.get(sr[0], []),
                iterations=sr[4],
            ))

        return RunTrace(
            run_id=run_id,
            graph_name=graph_name,
            query=query,
            timestamp=datetime.fromisoformat(timestamp_str),
            total_duration_ms=total_duration_ms,
            node_spans=spans,
            final_response=final_response or "",
            error=error,
            metadata=json.loads(metadata_json) if metadata_json else {},
            conversation_id=conversation_id or "",
        )

    def list_runs(
        self,
        graph_name: str | None = None,
        since: str | None = None,
        conversation_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RunTrace]:
        limit = min(limit, 1000)
        # Safety: only fixed column names are interpolated; all values use ? params
        conditions = []
        params: list = []

        if graph_name:
            conditions.append("graph_name = ?")
            params.append(graph_name)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if conversation_id:
            conditions.append("conversation_id = ?")
            params.append(conversation_id)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT run_id FROM runs {where_clause} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.append(limit)
        params.append(offset)

        c = self._conn.cursor()
        c.execute(query, params)
        rows = c.fetchall()

        traces = []
        for (rid,) in rows:
            t = self.load(rid)
            if t:
                traces.append(t)
        return traces

    def list_active_runs(self, stale_timeout_s: float = 60.0) -> list[RunTrace]:
        """Return runs with status='running' that have a recent heartbeat."""
        c = self._conn.cursor()
        cutoff = datetime.now(timezone.utc).timestamp() - stale_timeout_s
        c.execute("SELECT run_id, last_heartbeat FROM runs WHERE status = 'running'")
        rows = c.fetchall()

        traces = []
        for run_id, hb in rows:
            if hb:
                try:
                    hb_ts = datetime.fromisoformat(hb).timestamp()
                    if hb_ts < cutoff:
                        with self._write_lock:
                            c.execute("UPDATE runs SET status = 'error' WHERE run_id = ?", (run_id,))
                            self._conn.commit()
                        continue
                except (ValueError, TypeError):
                    pass
            t = self.load(run_id)
            if t:
                traces.append(t)
        return traces

    def get_status(self, run_id: str) -> str | None:
        """Get the status of a run."""
        c = self._conn.cursor()
        c.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,))
        row = c.fetchone()
        return row[0] if row else None

    def get_logs(self, position: int = -1) -> dict:
        """Read log lines from NODEWATCH_LOG_PATH (local mode)."""
        import os as _os
        log_path = _os.getenv("NODEWATCH_LOG_PATH")
        if not log_path:
            return {"lines": [], "position": 0, "error": "NODEWATCH_LOG_PATH not set"}
        try:
            with open(log_path, "rb") as f:
                f.seek(0, 2)
                file_size = f.tell()
                if position < 0:
                    read_bytes = min(file_size, 32_768)
                    f.seek(file_size - read_bytes)
                    chunk = f.read(read_bytes).decode("utf-8", errors="replace")
                    lines = chunk.split("\n")[-51:-1] if "\n" in chunk else [chunk]
                    return {"lines": lines, "position": file_size}
                else:
                    pos = min(position, file_size)
                    f.seek(pos)
                    chunk = f.read(min(file_size - pos, 65_536)).decode("utf-8", errors="replace")
                    lines = chunk.split("\n")
                    if lines and lines[-1] == "":
                        lines = lines[:-1]
                    return {"lines": lines, "position": file_size}
        except (OSError, IOError) as e:
            return {"lines": [], "position": 0, "error": str(e)}

    def delete(self, run_id: str) -> bool:
        with self._write_lock:
            c = self._conn.cursor()
            c.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            self._conn.commit()
            return c.rowcount > 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        self._conn.close()
