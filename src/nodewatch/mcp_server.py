"""MCP server exposing nodewatch data to AI assistants.

Launch as stdio server:
    nodewatch mcp

Or run directly:
    python -m nodewatch.mcp_server

Configure in claude_desktop_config.json or .mcp.json:
    {
        "mcpServers": {
            "nodewatch": {
                "command": "nodewatch",
                "args": ["mcp"]
            }
        }
    }
"""

from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer

from .models import RunTrace
from .stats import compute_conversation_stats, compute_summary

mcp = MCPServer(
    "nodewatch",
    instructions="Query LLM agent execution traces: runs, nodes, tokens, costs, tool calls, and performance stats.",
)


def _get_storage():
    """Get a RemoteClient if NODEWATCH_URL is set, otherwise local SQLiteStorage."""
    from .client import RemoteClient, get_remote_url

    remote_url = get_remote_url()
    if remote_url:
        return RemoteClient(base_url=remote_url)

    from .storage.sqlite import SQLiteStorage
    db_path = os.getenv("NODEWATCH_DB", "nodewatch.db")
    return SQLiteStorage(db_path)


def _trace_to_summary(t: RunTrace) -> dict:
    return {
        "run_id": t.run_id,
        "graph_name": t.graph_name,
        "query": t.query[:100],
        "timestamp": t.timestamp.isoformat(),
        "duration_ms": t.total_duration_ms,
        "total_tokens": t.total_tokens,
        "cost_usd": t.total_cost_usd,
        "llm_calls": t.total_llm_calls,
        "tool_calls": t.total_tool_calls,
        "error": t.error,
        "conversation_id": t.conversation_id,
    }


def _trace_to_detail(t: RunTrace) -> dict:
    nodes = []
    for s in t.node_spans:
        node = {
            "node_name": s.node_name,
            "node_type": s.node_type,
            "duration_ms": s.duration_ms,
            "input_tokens": s.total_input_tokens,
            "output_tokens": s.total_output_tokens,
            "cost_usd": s.total_cost_usd,
            "iterations": s.iterations,
            "llm_calls": [
                {
                    "model": c.model,
                    "provider": c.provider,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "thinking_tokens": c.thinking_tokens,
                    "cache_read_tokens": c.cache_read_tokens,
                    "cache_creation_tokens": c.cache_creation_tokens,
                    "duration_ms": c.duration_ms,
                    "cost_usd": c.cost_usd,
                    "stop_reason": c.stop_reason,
                    "error": c.error,
                }
                for c in s.llm_calls
            ],
            # `input` / `output_preview` / `output_size` are the fields that make a
            # trace debuggable: which args produced which answer. The storage layer
            # and ToolCall always carried them; this serializer dropped them, so
            # get_run showed only that a tool "succeeded" — and conversation 607's
            # truncation bug was invisible here and had to be read off the raw
            # server log instead.
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "duration_ms": tc.duration_ms,
                    "success": tc.success,
                    "error": tc.error,
                    "input": tc.input,
                    "output_preview": tc.output_preview,
                    "output_size": tc.output_size,
                }
                for tc in s.tool_calls
            ],
        }
        nodes.append(node)

    return {
        "run_id": t.run_id,
        "graph_name": t.graph_name,
        "query": t.query,
        "timestamp": t.timestamp.isoformat(),
        "duration_ms": t.total_duration_ms,
        "total_tokens": t.total_tokens,
        "input_tokens": t.total_input_tokens,
        "output_tokens": t.total_output_tokens,
        "cost_usd": t.total_cost_usd,
        "llm_calls": t.total_llm_calls,
        "tool_calls": t.total_tool_calls,
        "error": t.error,
        "final_response": t.final_response[:500] if t.final_response else None,
        "conversation_id": t.conversation_id,
        "nodes": nodes,
    }


@mcp.tool()
def list_runs(
    graph_name: str | None = None,
    since: str | None = None,
    conversation_id: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """List recent agent execution runs.

    Args:
        graph_name: Filter by graph name (e.g. "v1", "v2")
        since: ISO datetime to filter runs after (e.g. "2025-01-01")
        conversation_id: Filter by conversation ID
        limit: Max results (default 20, max 100)
    """
    limit = min(limit, 100)
    storage = _get_storage()
    try:
        traces = storage.list_runs(
            graph_name=graph_name,
            since=since,
            conversation_id=conversation_id,
            limit=limit,
        )
        return [_trace_to_summary(t) for t in traces]
    finally:
        storage.close()


@mcp.tool()
def get_run(run_id: str) -> dict | str:
    """Get full details of a specific run including all node spans, LLM calls, and tool calls.

    Args:
        run_id: The run ID to inspect
    """
    storage = _get_storage()
    try:
        trace = storage.load(run_id)
        if not trace:
            return f"Run '{run_id}' not found."
        return _trace_to_detail(trace)
    finally:
        storage.close()


@mcp.tool()
def get_stats(
    graph_name: str | None = None,
    since: str | None = None,
    limit: int = 100,
) -> dict:
    """Get aggregate statistics across runs: avg cost, tokens, latency, model breakdown, tool usage.

    Args:
        graph_name: Filter by graph name
        since: ISO datetime to filter runs after
        limit: Max runs to analyze (default 100)
    """
    limit = min(limit, 500)
    storage = _get_storage()
    try:
        traces = storage.list_runs(graph_name=graph_name, since=since, limit=limit)
        stats = compute_summary(traces)
        return {
            "run_count": stats.run_count,
            "avg_cost_usd": stats.avg_cost,
            "min_cost_usd": stats.min_cost,
            "max_cost_usd": stats.max_cost,
            "avg_tokens": stats.avg_tokens,
            "min_tokens": stats.min_tokens,
            "max_tokens": stats.max_tokens,
            "avg_latency_ms": stats.avg_latency_ms,
            "min_latency_ms": stats.min_latency_ms,
            "max_latency_ms": stats.max_latency_ms,
            "error_count": stats.error_count,
            "throughput_tokens_per_s": stats.throughput_tokens_per_s,
            "cost_per_1k_tokens": stats.cost_per_1k_tokens,
            "tool_calls_total": stats.tool_calls_total,
            "tool_calls_success": stats.tool_calls_success,
            "cache_read_tokens": stats.cache_read_tokens,
            "cache_creation_tokens": stats.cache_creation_tokens,
            "cache_hit_rate_pct": stats.cache_hit_rate,
            "models": [
                {"model": m.model, "total_tokens": m.total_tokens, "total_cost_usd": m.total_cost}
                for m in stats.models
            ],
        }
    finally:
        storage.close()


@mcp.tool()
def list_conversations(
    limit: int = 50,
) -> list[dict]:
    """List conversations with aggregated metrics (turns, tokens, cost, latency).

    Args:
        limit: Max runs to scan for conversations (default 50)
    """
    limit = min(limit, 200)
    storage = _get_storage()
    try:
        traces = storage.list_runs(limit=limit)
        conv_stats = compute_conversation_stats(traces)
        return [
            {
                "conversation_id": cs.conversation_id,
                "turn_count": cs.turn_count,
                "total_tokens": cs.total_tokens,
                "total_cost_usd": cs.total_cost,
                "avg_latency_ms": cs.avg_latency_ms,
                "graphs_used": cs.graphs_used,
            }
            for cs in conv_stats
        ]
    finally:
        storage.close()


@mcp.tool()
def get_conversation(conversation_id: str) -> list[dict]:
    """Get all runs within a conversation, ordered by time.

    Args:
        conversation_id: The conversation ID to look up
    """
    storage = _get_storage()
    try:
        traces = storage.list_runs(conversation_id=conversation_id, limit=100)
        return [_trace_to_summary(t) for t in traces]
    finally:
        storage.close()


@mcp.tool()
def find_expensive_runs(
    top_n: int = 10,
    metric: str = "cost",
) -> list[dict]:
    """Find the most expensive or token-heavy runs.

    Args:
        top_n: Number of results (default 10)
        metric: Sort by "cost", "tokens", or "duration" (default "cost")
    """
    top_n = min(top_n, 50)
    storage = _get_storage()
    try:
        traces = storage.list_runs(limit=200)
        if metric == "tokens":
            traces.sort(key=lambda t: t.total_tokens, reverse=True)
        elif metric == "duration":
            traces.sort(key=lambda t: t.total_duration_ms, reverse=True)
        else:
            traces.sort(key=lambda t: t.total_cost_usd, reverse=True)
        return [_trace_to_summary(t) for t in traces[:top_n]]
    finally:
        storage.close()


@mcp.tool()
def get_active_runs() -> list[dict]:
    """Get currently running agent executions (if any)."""
    storage = _get_storage()
    try:
        traces = storage.list_active_runs(stale_timeout_s=120.0)
        return [_trace_to_summary(t) for t in traces]
    finally:
        storage.close()


@mcp.tool()
def compare_models(
    expected_a: str,
    expected_b: str,
    since: str | None = None,
    limit: int = 200,
) -> dict:
    """A/B compare two model cohorts among ALREADY-RECORDED runs (read-only, no model calls).

    Groups recent runs by the model that actually served them (from llm_calls, not graph_name),
    verifies each cohort ran its intended model, and reports per-question matched-node-path
    deltas for duration, tokens, and content-filter rate. Use this to analyze runs that already
    exist in storage; use run_model_ab to generate fresh runs by calling models directly.

    Args:
        expected_a: Served-model substring for cohort A (baseline), e.g. "opus-4-8"
        expected_b: Served-model substring for cohort B (new), e.g. "opus-4-7"
        since: ISO datetime to filter runs after (e.g. "2026-01-01")
        limit: Max runs to scan into the two cohorts (default 200)
    """
    from .abrun import comparison_to_dict
    from .stats import compute_ab_comparison

    storage = _get_storage()
    try:
        traces = storage.list_runs(since=since, limit=min(limit, 1000))
        a, b = [], []
        for t in traces:
            served = " ".join(c.model.lower() for s in t.node_spans for c in s.llm_calls if c.model)
            if expected_a in served:
                a.append(t)
            elif expected_b in served:
                b.append(t)
        comp = compute_ab_comparison(a, b, expected_a, expected_b)
        return comparison_to_dict(comp)
    finally:
        storage.close()


@mcp.tool()
def run_model_ab(
    prompts: list[str],
    models: list[str],
    provider: str = "anthropic",
    reps: int = 1,
    system: str | None = None,
    session: str | None = None,
    db: str | None = None,
    out_dir: str | None = None,
    pause_check: bool = False,
    confirm: bool = False,
) -> dict:
    """Run a live A/B model benchmark: send each prompt to each model and compare them.

    ⚠ This is the only tool here that CALLS THE MODELS (incurs real API cost) and WRITES new
    runs to storage. Every other tool is read-only. Each prompt is sent to every model in
    `models`; the runs are recorded and (for exactly 2 models) compared with the model isolated
    as the only variable. Requires credentials for `provider` in the environment
    (ANTHROPIC_API_KEY / OPENAI_API_KEY / AWS_*) and the matching langchain provider package.

    Args:
        prompts: The prompts to test (each becomes one question, asked to every model).
        models: Model ids to compare, e.g. ["claude-opus-4-8", "claude-opus-4-7"]. Exactly 2
            yields a pairwise comparison; more yields per-model summaries only.
        provider: "anthropic" (default), "openai", or "bedrock".
        reps: Repetitions per prompt per model (default 1).
        system: Optional system prompt applied to every call.
        session: Testing-session name (or folder path). When set, creates a self-contained session
            folder (a bare name → ./testing_sessions/<name>; NODEWATCH_SESSIONS_DIR overrides the
            base) and writes its config.json, runs.db, ab_<model>.json, and results.json all there.
            Easiest way to produce a reproducible, diffable test in one call. Overrides db/out_dir.
        db: SQLite path to record runs (defaults to the session DB, then NODEWATCH_DB / nodewatch.db).
        out_dir: Where to write ab_<model>.json + results.json (defaults to the session folder).
        pause_check: Require a confirmation before calling the models. When True, the first call
            returns {"status": "confirmation_required", ...} (and, in session mode, still writes
            config.json so it can be inspected) without spending anything; call again with
            confirm=True to run.
        confirm: Set True to proceed past a pause_check gate.

    Returns:
        A dict with `comparison` (the A/B verdict, or null for ≠2 models), `model_summaries`,
        `agent_reports` (per-model question-level detail), `report_paths` / `results_path` (files
        written), `session_dir` (when a session was used), and a one-line `summary`. When
        pause_check is set and confirm is False, returns {"status": "confirmation_required", ...}.
        On a missing credential/dependency, returns {"error": ...}.
    """
    import json as _json

    from .abrun import (
        CONFIG_FILENAME,
        parse_ab_config,
        preview_ab_config,
        resolve_session_dir,
        run_ab_config,
    )

    if not prompts or not models:
        return {"error": "provide at least one prompt and at least one model"}

    raw = {
        "transport": "model",
        "model": {"provider": provider, **({"system": system} if system else {})},
        "experiment": {"reps": reps, "settle_seconds": 0, "switch_mode": "per_request",
                       **({"pause_check": True} if pause_check else {})},
        "models": [{"id": m, "request_model": m, "expect": _short_model_id(m)} for m in models],
        "questions": [{"id": f"q{i+1}", "text": p} for i, p in enumerate(prompts)],
    }
    try:
        config = parse_ab_config(raw)
    except ValueError as e:
        return {"error": str(e)}

    # Session mode: a self-contained folder holding config + every artifact. Write the config
    # even on a pending confirmation so it can be inspected before the (paid) run.
    session_dir = None
    if session:
        session_dir = resolve_session_dir(session)
        try:
            session_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / CONFIG_FILENAME).write_text(_json.dumps(raw, indent=2))
        except OSError as e:
            return {"error": f"could not create session {session_dir}: {e}"}
        if db is None:
            db = str(session_dir / "runs.db")
        if out_dir is None:
            out_dir = str(session_dir)

    if config.pause_check and not confirm:
        return {
            "status": "confirmation_required",
            "message": config.pause_check,
            "preview": preview_ab_config(config),
            "session_dir": str(session_dir) if session_dir else None,
            "how_to_proceed": "call run_model_ab again with the same args plus confirm=true once the user approves",
        }

    from .storage.sqlite import SQLiteStorage
    db_path = db or os.getenv("NODEWATCH_DB", "nodewatch.db")
    storage = SQLiteStorage(db_path)
    try:
        result = run_ab_config(config, storage, out_dir=out_dir)
    except RuntimeError as e:
        return {"error": str(e)}
    finally:
        storage.close()

    if session_dir is not None:
        result["session_dir"] = str(session_dir)
    return result


def _short_model_id(model: str) -> str:
    """Verification substring for a model id (e.g. claude-opus-4-8 → opus-4-8)."""
    from .stats import _short_model

    short = _short_model(model)
    return short or model


@mcp.tool()
def init_ab_session(session: str, transport: str = "model", force: bool = False) -> dict:
    """Create an A/B testing-session folder with an editable config.json.

    A session is a self-contained folder: its config.json is the input, and the runs + per-agent
    reports + results.json are written back into it. Scaffold one here, edit it with
    write_ab_config (or fill it in yourself), then run it with run_ab_session.

    Args:
        session: Session name (bare name → ./testing_sessions/<name>; NODEWATCH_SESSIONS_DIR
            overrides the base) or an explicit folder path.
        transport: Template to write — "model" (call models directly) or "http" (an agent API).
        force: Overwrite an existing config.json.

    Returns:
        {"session_dir", "config_path", "config"} — or {"error": ...} if config.json exists.
    """
    from .abrun import init_session, resolve_session_dir

    if transport not in ("model", "http"):
        return {"error": "transport must be 'model' or 'http'"}
    session_dir = resolve_session_dir(session)
    try:
        config_path = init_session(session_dir, transport=transport, force=force)
    except FileExistsError:
        return {"error": f"config.json already exists in {session_dir} (pass force=true to overwrite)"}
    except (OSError, ValueError) as e:
        return {"error": str(e)}
    import json as _json
    return {
        "session_dir": str(session_dir),
        "config_path": str(config_path),
        "config": _json.loads(config_path.read_text()),
    }


@mcp.tool()
def read_ab_config(session: str) -> dict:
    """Read a session's config.json so you can edit and re-write it with write_ab_config.

    Args:
        session: Session name or folder path.

    Returns:
        {"session_dir", "config"} — or {"error": ...} if there is no config.json.
    """
    import json as _json

    from .abrun import CONFIG_FILENAME, resolve_session_dir

    session_dir = resolve_session_dir(session)
    config_path = session_dir / CONFIG_FILENAME
    if not config_path.exists():
        return {"error": f"no {CONFIG_FILENAME} in {session_dir}; create it with init_ab_session"}
    try:
        return {"session_dir": str(session_dir), "config": _json.loads(config_path.read_text())}
    except (OSError, ValueError) as e:
        return {"error": str(e)}


@mcp.tool()
def write_ab_config(session: str, config: dict) -> dict:
    """Write (create or overwrite) a session's config.json with the given config dict.

    This is how you EDIT the testing JSON: pass the full config you want. It is validated before
    writing (same schema as the config files), so an invalid config returns an error and nothing
    is written. Creates the session folder if needed.

    Args:
        session: Session name or folder path.
        config: The full A/B config (keys: transport, models, questions, and api/model + experiment
            as appropriate). See init_ab_session for a template to start from.

    Returns:
        {"session_dir", "config_path"} — or {"error": ...} if the config is invalid.
    """
    import json as _json

    from .abrun import CONFIG_FILENAME, parse_ab_config, resolve_session_dir

    try:
        parse_ab_config(config)            # validate; raises ValueError on a bad config
    except ValueError as e:
        return {"error": f"invalid config (not written): {e}"}
    session_dir = resolve_session_dir(session)
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
        config_path = session_dir / CONFIG_FILENAME
        config_path.write_text(_json.dumps(config, indent=2))
    except OSError as e:
        return {"error": str(e)}
    return {"session_dir": str(session_dir), "config_path": str(config_path)}


@mcp.tool()
def run_ab_session(session: str, confirm: bool = False) -> dict:
    """Run an existing testing session: load its config.json and execute the A/B benchmark.

    ⚠ For the "model" transport this CALLS THE MODELS (real API cost). Loads <session>/config.json,
    records runs to the session's runs.db (or the config's "db"), and writes ab_<model>.json +
    results.json into the session folder. Use this after init_ab_session / write_ab_config.

    Confirmation gate: if the config sets `experiment.pause_check`, this returns
    {"status": "confirmation_required", ...} WITHOUT running. Show the message + preview to the
    user, then call again with confirm=True to actually run. (No gate → runs immediately.)

    Note: an "http" + manual-switch session needs a human to reconfigure + restart the server
    between phases — that interactive flow is CLI-only (`nodewatch ab-run <session>`); over MCP it
    runs without pausing, so served-model verification will flag any phase that wasn't switched.

    Args:
        session: Session name or folder path.
        confirm: Set True to proceed past the config's pause_check gate.

    Returns:
        The run result dict (comparison, model_summaries, agent_reports, report_paths,
        results_path, session_dir, summary), a {"status": "confirmation_required", ...} dict when a
        gate is pending, or {"error": ...}.
    """
    from .abrun import (
        CONFIG_FILENAME,
        load_ab_config,
        preview_ab_config,
        resolve_session_dir,
        run_ab_config,
    )

    session_dir = resolve_session_dir(session)
    config_path = session_dir / CONFIG_FILENAME
    if not config_path.exists():
        return {"error": f"no {CONFIG_FILENAME} in {session_dir}; create it with init_ab_session"}
    try:
        cfg = load_ab_config(config_path)
    except (OSError, ValueError) as e:
        return {"error": str(e)}

    if cfg.pause_check and not confirm:
        return {
            "status": "confirmation_required",
            "message": cfg.pause_check,
            "preview": preview_ab_config(cfg),
            "session_dir": str(session_dir),
            "how_to_proceed": "call run_ab_session again with confirm=true once the user approves",
        }

    from .storage.sqlite import SQLiteStorage
    db_path = cfg.db or str(session_dir / "runs.db")
    storage = SQLiteStorage(db_path)
    try:
        result = run_ab_config(cfg, storage, out_dir=str(session_dir))
    except RuntimeError as e:
        return {"error": str(e)}
    finally:
        storage.close()
    result["session_dir"] = str(session_dir)
    return result


def run_mcp_server() -> None:
    """Entry point for the MCP server (stdio transport)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_mcp_server()
