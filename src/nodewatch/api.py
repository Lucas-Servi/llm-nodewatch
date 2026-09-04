"""FastAPI HTTP API for querying nodewatch traces.

Single-endpoint dispatcher: all operations go through POST / with a JSON body:
    {"method": "list_runs", "args": {"limit": 5}}

Run standalone:
    NODEWATCH_DB=/path/to/db uvicorn nodewatch.api:app --port 8052

Or mount in an existing app:
    from nodewatch.api import create_router
    app.include_router(create_router("/path/to/db"), prefix="/nodewatch")
"""

from __future__ import annotations

import inspect
import os
import secrets
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

from .models import RunTrace
from .reporter import trace_to_json, trace_to_markdown
from .stats import ABComparison, compute_ab_comparison, compute_summary
from .storage import SQLiteStorage


def _build_auth_dependency():
    """Return a FastAPI dependency that enforces Bearer token auth if NODEWATCH_API_TOKEN is set."""
    token = os.getenv("NODEWATCH_API_TOKEN")
    if not token:
        async def _noop():
            pass
        return _noop

    async def _check_token(request: Request):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            raise HTTPException(401, "Missing or invalid Authorization header")
        provided = auth[7:]
        if not secrets.compare_digest(provided, token):
            raise HTTPException(401, "Invalid token")

    return _check_token


class MethodCall(BaseModel):
    method: str
    args: dict = {}


def create_router(db_path: str | None = None) -> APIRouter:
    """Factory that returns a configured APIRouter bound to a storage instance."""
    _path = db_path or os.getenv("NODEWATCH_DB", "nodewatch.db")
    storage = SQLiteStorage(_path)
    auth_dep = _build_auth_dependency()
    router = APIRouter(tags=["nodewatch"], dependencies=[Depends(auth_dep)])

    # ── Handler functions ────────────────────────────────────────────────────

    def list_runs(graph_name: str | None = None, since: str | None = None, conversation_id: str | None = None, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        traces = storage.list_runs(graph_name=graph_name, since=since, conversation_id=conversation_id, limit=min(limit, 200), offset=offset)
        return [_summarize(t) for t in traces]

    def get_run(run_id: str) -> dict[str, Any]:
        trace = storage.load(run_id)
        if not trace:
            raise HTTPException(404, f"Run {run_id} not found")
        return trace_to_json(trace)

    def get_run_markdown(run_id: str) -> dict[str, Any]:
        trace = storage.load(run_id)
        if not trace:
            raise HTTPException(404, f"Run {run_id} not found")
        return {"markdown": trace_to_markdown(trace)}

    def get_run_nodes(run_id: str) -> list[dict[str, Any]]:
        trace = storage.load(run_id)
        if not trace:
            raise HTTPException(404, f"Run {run_id} not found")
        return [_node_summary(s) for s in trace.node_spans]

    def delete_run(run_id: str) -> dict[str, Any]:
        deleted = storage.delete(run_id)
        if not deleted:
            raise HTTPException(404, f"Run {run_id} not found")
        return {"deleted": run_id}

    def report(graph_name: str | None = None, since: str | None = None, limit: int = 50) -> dict[str, Any]:
        traces = storage.list_runs(graph_name=graph_name, since=since, limit=min(limit, 200))
        if not traces:
            return {"run_count": 0}
        stats = compute_summary(traces)
        return {
            "run_count": stats.run_count,
            "avg_cost": round(stats.avg_cost, 6),
            "min_cost": round(stats.min_cost, 6),
            "max_cost": round(stats.max_cost, 6),
            "avg_tokens": stats.avg_tokens,
            "min_tokens": stats.min_tokens,
            "max_tokens": stats.max_tokens,
            "avg_latency_ms": round(stats.avg_latency_ms, 1),
            "min_latency_ms": round(stats.min_latency_ms, 1),
            "max_latency_ms": round(stats.max_latency_ms, 1),
            "error_count": stats.error_count,
            "models": [
                {"model": m.model, "total_tokens": m.total_tokens, "total_cost": round(m.total_cost, 6)}
                for m in stats.models
            ],
            "throughput_tokens_per_s": round(stats.throughput_tokens_per_s, 1),
            "cost_per_1k_tokens": round(stats.cost_per_1k_tokens, 6),
            "tool_calls_total": stats.tool_calls_total,
            "tool_calls_success": stats.tool_calls_success,
        }

    def stats(graph_name: str | None = None, limit: int = 200) -> dict[str, Any]:
        traces = storage.list_runs(graph_name=graph_name, limit=min(limit, 200))
        graphs: dict[str, dict] = {}
        for t in traces:
            g = graphs.setdefault(
                t.graph_name, {"runs": 0, "total_tokens": 0, "total_cost_usd": 0.0}
            )
            g["runs"] += 1
            g["total_tokens"] += t.total_tokens
            g["total_cost_usd"] += t.total_cost_usd
        return {"total_runs": len(traces), "by_graph": graphs}

    def chart_cost_over_time(graph_name: str | None = None, since: str | None = None, limit: int = 50) -> dict[str, Any]:
        traces = storage.list_runs(graph_name=graph_name, since=since, limit=min(limit, 200))
        series: dict[str, list] = {}
        for t in traces:
            series.setdefault(t.graph_name, []).append({
                "timestamp": t.timestamp.isoformat(),
                "cost_usd": round(t.total_cost_usd, 6),
                "run_id": t.run_id,
            })
        return {"series": [{"graph_name": k, "points": v} for k, v in series.items()]}

    def chart_tokens_over_time(graph_name: str | None = None, since: str | None = None, limit: int = 50) -> dict[str, Any]:
        traces = storage.list_runs(graph_name=graph_name, since=since, limit=min(limit, 200))
        series: dict[str, list] = {}
        for t in traces:
            series.setdefault(t.graph_name, []).append({
                "timestamp": t.timestamp.isoformat(),
                "input_tokens": t.total_input_tokens,
                "output_tokens": t.total_output_tokens,
                "run_id": t.run_id,
            })
        return {"series": [{"graph_name": k, "points": v} for k, v in series.items()]}

    def chart_latency_over_time(graph_name: str | None = None, since: str | None = None, limit: int = 50) -> dict[str, Any]:
        traces = storage.list_runs(graph_name=graph_name, since=since, limit=min(limit, 200))
        series: dict[str, list] = {}
        for t in traces:
            series.setdefault(t.graph_name, []).append({
                "timestamp": t.timestamp.isoformat(),
                "duration_ms": round(t.total_duration_ms, 1),
                "run_id": t.run_id,
            })
        return {"series": [{"graph_name": k, "points": v} for k, v in series.items()]}

    def chart_node_breakdown(run_id: str) -> dict[str, Any]:
        trace = storage.load(run_id)
        if not trace:
            raise HTTPException(404, f"Run {run_id} not found")
        nodes = []
        for s in trace.node_spans:
            nodes.append({
                "node_name": s.node_name,
                "tokens": s.total_tokens,
                "input_tokens": s.total_input_tokens,
                "output_tokens": s.total_output_tokens,
                "cost_usd": round(s.total_cost_usd, 6),
                "duration_ms": round(s.duration_ms, 1),
                "tool_calls": len(s.tool_calls),
                "llm_calls": len(s.llm_calls),
                "filtered_count": s.filtered_count,
                "iterations": s.iterations,
            })
        return {"run_id": run_id, "graph_name": trace.graph_name, "nodes": nodes}

    def chart_node_comparison(graph_name: str | None = None, limit: int = 30) -> dict[str, Any]:
        traces = storage.list_runs(graph_name=graph_name, limit=min(limit, 100))
        node_agg: dict[str, dict] = {}
        for t in traces:
            for s in t.node_spans:
                agg = node_agg.setdefault(
                    s.node_name, {"tokens": [], "cost": [], "duration": [], "filtered": []}
                )
                agg["tokens"].append(s.total_tokens)
                agg["cost"].append(s.total_cost_usd)
                agg["duration"].append(s.duration_ms)
                agg["filtered"].append(s.filtered_count)
        nodes = sorted(node_agg.keys())
        return {
            "nodes": nodes,
            "metrics": {
                "avg_tokens": [round(sum(node_agg[n]["tokens"]) / len(node_agg[n]["tokens"])) for n in nodes],
                "avg_cost_usd": [round(sum(node_agg[n]["cost"]) / len(node_agg[n]["cost"]), 6) for n in nodes],
                "avg_duration_ms": [round(sum(node_agg[n]["duration"]) / len(node_agg[n]["duration"]), 1) for n in nodes],
                "avg_filtered": [round(sum(node_agg[n]["filtered"]) / len(node_agg[n]["filtered"]), 2) for n in nodes],
            },
            "sample_size": {n: len(node_agg[n]["tokens"]) for n in nodes},
        }

    def chart_tool_frequency(graph_name: str | None = None, since: str | None = None, limit: int = 50) -> dict[str, Any]:
        traces = storage.list_runs(graph_name=graph_name, since=since, limit=min(limit, 200))
        tool_agg: dict[str, dict] = {}
        for t in traces:
            for s in t.node_spans:
                for tc in s.tool_calls:
                    agg = tool_agg.setdefault(
                        tc.tool_name, {"count": 0, "durations": [], "successes": 0}
                    )
                    agg["count"] += 1
                    agg["durations"].append(tc.duration_ms)
                    if tc.success:
                        agg["successes"] += 1
        tools = sorted(tool_agg.items(), key=lambda x: x[1]["count"], reverse=True)
        return {
            "tools": [
                {
                    "tool_name": name,
                    "call_count": data["count"],
                    "avg_duration_ms": round(sum(data["durations"]) / len(data["durations"]), 1) if data["durations"] else 0,
                    "success_rate": round(data["successes"] / data["count"], 3) if data["count"] else 1.0,
                }
                for name, data in tools
            ]
        }

    def chart_model_usage(since: str | None = None, limit: int = 50) -> dict[str, Any]:
        traces = storage.list_runs(since=since, limit=min(limit, 200))
        model_agg: dict[str, dict] = {}
        for t in traces:
            for s in t.node_spans:
                for llm in s.llm_calls:
                    agg = model_agg.setdefault(
                        llm.model, {"tokens": 0, "cost": 0.0, "calls": 0}
                    )
                    agg["tokens"] += llm.total_tokens
                    agg["cost"] += llm.cost_usd
                    agg["calls"] += 1
        return {
            "models": [
                {
                    "model": m,
                    "total_tokens": d["tokens"],
                    "total_cost_usd": round(d["cost"], 6),
                    "call_count": d["calls"],
                }
                for m, d in sorted(model_agg.items(), key=lambda x: x[1]["cost"], reverse=True)
            ]
        }

    def chart_v1_vs_v2(since: str | None = None, limit: int = 50) -> dict[str, Any]:
        all_traces = storage.list_runs(since=since, limit=min(limit, 200))
        groups: dict[str, list] = {}
        for t in all_traces:
            groups.setdefault(t.graph_name, []).append(t)

        def _avg(traces: list, attr: str):
            if not traces:
                return 0
            return sum(getattr(t, attr) for t in traces) / len(traces)

        metrics = ["avg_cost_usd", "avg_tokens", "avg_duration_ms", "avg_tool_calls", "avg_llm_calls"]
        result: dict = {"metrics": metrics, "run_count": {}}
        for gname, gtraces in groups.items():
            result[gname] = [
                round(_avg(gtraces, "total_cost_usd"), 6),
                round(_avg(gtraces, "total_tokens")),
                round(_avg(gtraces, "total_duration_ms"), 1),
                round(_avg(gtraces, "total_tool_calls"), 1),
                round(_avg(gtraces, "total_llm_calls"), 1),
            ]
            result["run_count"][gname] = len(gtraces)
        return result

    def ab_compare(
        expected_a: str = "opus-4-8",
        expected_b: str = "opus-4-7",
        since: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        """A/B compare two served-model cohorts over the recent run history.

        Partitions runs by the model that actually served them (NOT graph_name): a run
        whose served opus models include `expected_a` goes to cohort A, `expected_b` to
        cohort B. Returns the ABComparison as a dict (verification + per-question deltas).
        """
        traces = storage.list_runs(since=since, limit=min(limit, 1000))
        a, b = [], []
        for t in traces:
            served = " ".join(
                c.model.lower() for s in t.node_spans for c in s.llm_calls if c.model
            )
            if expected_a in served:
                a.append(t)
            elif expected_b in served:
                b.append(t)
        comp = compute_ab_comparison(a, b, expected_a, expected_b)
        return _ab_to_dict(comp)

    def get_active_runs() -> list[dict[str, Any]]:
        """List runs currently in 'running' status."""
        traces = storage.list_active_runs(stale_timeout_s=120.0)
        return [_summarize_active(t, storage) for t in traces]

    def get_run_live(run_id: str) -> dict[str, Any]:
        """Get a running trace with status metadata."""
        trace = storage.load(run_id)
        if not trace:
            raise HTTPException(404, f"Run {run_id} not found")
        result = trace_to_json(trace)
        result["status"] = storage.get_status(run_id) or "unknown"
        return result

    # ── Logs ─────────────────────────────────────────────────────────────────

    def get_logs(position: int = -1) -> dict[str, Any]:
        """Read log lines from NODEWATCH_LOG_PATH.

        position=-1: return last ~50 lines and current file size.
        position>=0: return lines appended since that byte offset.
        """
        log_path = os.getenv("NODEWATCH_LOG_PATH")
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

    # ── Method registry ──────────────────────────────────────────────────────

    METHODS: dict[str, callable] = {
        "get_active_runs": get_active_runs,
        "get_run_live": get_run_live,
        "get_logs": get_logs,
        "list_runs": list_runs,
        "get_run": get_run,
        "get_run_markdown": get_run_markdown,
        "get_run_nodes": get_run_nodes,
        "delete_run": delete_run,
        "report": report,
        "stats": stats,
        "chart_cost_over_time": chart_cost_over_time,
        "chart_tokens_over_time": chart_tokens_over_time,
        "chart_latency_over_time": chart_latency_over_time,
        "chart_node_breakdown": chart_node_breakdown,
        "chart_node_comparison": chart_node_comparison,
        "chart_tool_frequency": chart_tool_frequency,
        "chart_model_usage": chart_model_usage,
        "chart_v1_vs_v2": chart_v1_vs_v2,
        "ab_compare": ab_compare,
    }

    # ── Dispatcher endpoint ──────────────────────────────────────────────────

    @router.post("/")
    def dispatch(call: MethodCall):
        handler = METHODS.get(call.method)
        if not handler:
            raise HTTPException(400, detail={
                "error": f"Unknown method: {call.method}",
                "available_methods": sorted(METHODS.keys()),
            })
        try:
            return {"result": handler(**call.args)}
        except TypeError as e:
            raise HTTPException(422, detail={
                "error": f"Invalid args for '{call.method}': {e}",
                "expected_args": _get_method_args(handler),
            })

    @router.get("/methods")
    def get_methods():
        """List all available methods and their expected arguments."""
        return {"methods": {name: _get_method_args(fn) for name, fn in METHODS.items()}}

    def _get_method_args(fn) -> dict:
        sig = inspect.signature(fn)
        args = {}
        for p in sig.parameters.values():
            ann = p.annotation
            type_str = ann.__name__ if hasattr(ann, "__name__") else str(ann) if ann != inspect.Parameter.empty else "any"
            default = p.default if p.default != inspect.Parameter.empty else None
            args[p.name] = {"type": type_str, "default": default}
        return args

    return router


def _summarize(trace: RunTrace) -> dict:
    return {
        "run_id": trace.run_id,
        "graph_name": trace.graph_name,
        "query": trace.query[:100],
        "timestamp": trace.timestamp.isoformat(),
        "duration_ms": round(trace.total_duration_ms, 1),
        "total_tokens": trace.total_tokens,
        "total_cost_usd": round(trace.total_cost_usd, 6),
        "total_filtered": trace.total_filtered,
        "nodes_visited": trace.nodes_visited,
        "conversation_id": trace.conversation_id,
        "error": trace.error,
    }


def _node_summary(span) -> dict:
    return {
        "node_name": span.node_name,
        "node_type": span.node_type,
        "duration_ms": round(span.duration_ms, 1),
        "total_tokens": span.total_tokens,
        "total_cost_usd": round(span.total_cost_usd, 6),
        "llm_calls": len(span.llm_calls),
        "tool_calls": len(span.tool_calls),
        "filtered_count": span.filtered_count,
        "iterations": span.iterations,
    }


def _summarize_active(trace, storage) -> dict:
    return {
        "run_id": trace.run_id,
        "graph_name": trace.graph_name,
        "timestamp": trace.timestamp.isoformat(),
        "duration_ms": round(trace.total_duration_ms, 1),
        "total_tokens": trace.total_tokens,
        "total_cost_usd": round(trace.total_cost_usd, 6),
        "total_filtered": trace.total_filtered,
        "nodes_visited": trace.nodes_visited,
        "status": storage.get_status(trace.run_id) or "running",
    }


def _ab_to_dict(comp: ABComparison) -> dict:
    """Serialize an ABComparison for the HTTP API."""
    return {
        "cohort_a": comp.cohort_a,
        "cohort_b": comp.cohort_b,
        "verified_ok": comp.verified_ok,
        "verification": [
            {
                "cohort": v.cohort,
                "expected_model": v.expected_model,
                "served_models": v.served_models,
                "ok": v.ok,
            }
            for v in comp.verification
        ],
        "per_question": [
            {
                "question_key": q.question_key,
                "matched_node_sig": q.matched_node_sig,
                "n_a": q.n_a,
                "n_b": q.n_b,
                "mean_duration_ms_a": round(q.mean_duration_ms_a, 1),
                "mean_duration_ms_b": round(q.mean_duration_ms_b, 1),
                "duration_delta_pct": round(q.duration_delta_pct, 1),
                "mean_tokens_a": round(q.mean_tokens_a),
                "mean_tokens_b": round(q.mean_tokens_b),
                "tokens_delta_pct": round(q.tokens_delta_pct, 1),
                "mean_filtered_per_call_a": round(q.mean_filtered_per_call_a, 3),
                "mean_filtered_per_call_b": round(q.mean_filtered_per_call_b, 3),
            }
            for q in comp.per_question
        ],
        "overall_duration_delta_pct": round(comp.overall_duration_delta_pct, 1),
        "overall_tokens_delta_pct": round(comp.overall_tokens_delta_pct, 1),
        "overall_filtered_per_call_a": round(comp.overall_filtered_per_call_a, 3),
        "overall_filtered_per_call_b": round(comp.overall_filtered_per_call_b, 3),
    }


def _create_standalone_app() -> FastAPI:
    """Create standalone app lazily (avoids creating db on import)."""
    from . import __version__

    _app = FastAPI(title="nodewatch", version=__version__)
    _app.include_router(create_router(), prefix="")
    return _app


def __getattr__(name: str):
    if name == "app":
        return _create_standalone_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
