# LangGraph Integration Guide

The Quick Start above shows the basics. This section walks through a production-grade integration — feature-flagged imports, per-request lifecycle, error recovery, and graceful degradation. Based on a real multi-agent system running in production.

## Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Your App   │────▶│ GraphTracker │────▶│  LangGraph  │
│             │     │  (callback)  │◀────│  Execution  │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    finalize()
                           │
                           ▼
                    ┌──────────────┐     ┌─────────────┐
                    │   RunTrace   │────▶│   SQLite    │
                    │  (dataclass) │     │  Storage    │
                    └──────────────┘     └─────────────┘
                                                │
                                                ▼
                                  CLI / Dashboard / MCP / API
```

The `GraphTracker` is a standard LangChain `BaseCallbackHandler`. It intercepts `on_chain_start/end`, `on_llm_start/end`, and `on_tool_start/end` events, correlates them by node, and assembles a `RunTrace` when you call `finalize()`.

## Minimal Integration (5 Lines)

If you just want tracing with no error handling:

```python
import nodewatch

storage = nodewatch.SQLiteStorage("runs.db")              # once at startup
tracker = nodewatch.GraphTracker("my-graph")              # once per invocation
result = await graph.ainvoke(state, config=tracker.config) 
trace = tracker.finalize(query="user input", final_response=result["output"])
storage.save(trace)
```

## Production Integration

For production systems that need graceful degradation, metadata, live monitoring, and error recovery:

### Step 1 — Feature-Flagged Import

Make nodewatch fully optional. Your app works without it installed:

```python
import os, logging

logger = logging.getLogger(__name__)

_NODEWATCH_AVAILABLE = False
_NODEWATCH_ENABLED = os.getenv("NODEWATCH_ENABLED", "1") == "1"

if _NODEWATCH_ENABLED:
    try:
        import nodewatch
        _NODEWATCH_AVAILABLE = True
    except ImportError:
        logger.debug("nodewatch not installed — tracing disabled")
```

Set `NODEWATCH_ENABLED=0` to disable at runtime without uninstalling.

### Step 2 — Initialize Storage Once

Create one `SQLiteStorage` per process lifetime (it manages a connection pool with WAL mode):

```python
class MyService:
    def __init__(self):
        self._nodewatch_storage = None
        if _NODEWATCH_AVAILABLE:
            try:
                db_path = os.getenv("NODEWATCH_DB", "./nodewatch.db")
                self._nodewatch_storage = nodewatch.SQLiteStorage(db_path)
            except Exception as exc:
                logger.warning("Nodewatch storage init failed: %r", exc)
```

### Step 3 — Create a Tracker Per Request

One `GraphTracker` per `graph.ainvoke()` call. Pass metadata to tag the trace, and enable `live=True` if you want the run to appear in the dashboard's Live tab while it is still executing:

```python
_tracker = None
if _NODEWATCH_AVAILABLE and self._nodewatch_storage is not None:
    try:
        _tracker = nodewatch.GraphTracker(
            "my-agent",                          # graph name (used for filtering)
            metadata={
                "user_id": user_id,
                "conversation_id": conversation_id,
                "any_key": "any_value",
            },
            storage=self._nodewatch_storage,      # enables live heartbeats
            live=True,                            # makes run visible in real-time
        )
    except Exception as exc:
        logger.warning("Tracker creation failed: %r", exc)
        _tracker = None
```

### Step 4 — Inject Tracker as a LangChain Callback

**Simple case** — use `tracker.config` directly:

```python
result = await graph.ainvoke(state, config=tracker.config)
```

**When you already have a config dict** (recursion_limit, configurable, etc.) — merge the callbacks:

```python
config = {
    "configurable": {"thread_id": conversation_id},
    "recursion_limit": 100,
}
if _tracker is not None:
    config["callbacks"] = [_tracker]

result = await graph.ainvoke(state, config=config)
```

### Step 5 — Finalize and Persist

After the graph completes, call `finalize()` exactly once to assemble the trace, then `save()` to persist:

```python
if _tracker is not None:
    trace = _tracker.finalize(query=user_prompt, final_response=response_text)
    self._nodewatch_storage.save(trace)
```

### Step 6 — Access Trace Metrics

The `RunTrace` exposes computed properties:

```python
if trace is not None:
    print(trace.run_id)              # unique UUID
    print(trace.total_tokens)        # input + output + thinking + cache
    print(trace.total_cost_usd)      # computed from model pricing
    print(trace.total_llm_calls)     # number of LLM invocations
    print(trace.total_tool_calls)    # number of tool executions
    print(trace.nodes_visited)       # list of node names in order
    print(trace.total_duration_ms)   # wall-clock time
```

### Step 7 — Handle Errors and Timeouts

**Critical**: always finalize even on exception or timeout. Partial traces are invaluable for debugging failures:

```python
try:
    result = await asyncio.wait_for(
        graph.ainvoke(state, config=config), timeout=600.0
    )
    response_text = result["messages"][-1].content

    if _tracker is not None:
        trace = _tracker.finalize(query=user_prompt, final_response=response_text)
        self._nodewatch_storage.save(trace)

except (TimeoutError, Exception) as exc:
    logger.exception("Graph failed: %r", exc)
    # Finalize with empty response — the trace still captures
    # which nodes ran, how many tokens were used, and where it stalled
    if _tracker is not None:
        try:
            trace = _tracker.finalize(query=user_prompt, final_response="")
            self._nodewatch_storage.save(trace)
        except Exception:
            pass
```

### Step 8 — Cleanup on Shutdown

Close the storage connection when your app shuts down:

```python
# FastAPI lifespan example
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    yield
    service.shutdown()

# Or in your class
def shutdown(self):
    if self._nodewatch_storage is not None:
        self._nodewatch_storage.close()
```

## Complete Example

A production-ready class combining all steps above:

```python
"""Production nodewatch integration for a LangGraph chatbot service."""
import os
import logging
import asyncio

logger = logging.getLogger(__name__)

# ─── Step 1: Feature-flagged import ─────────────────────────────────────
_NODEWATCH_AVAILABLE = False
_NODEWATCH_ENABLED = os.getenv("NODEWATCH_ENABLED", "1") == "1"
if _NODEWATCH_ENABLED:
    try:
        import nodewatch
        _NODEWATCH_AVAILABLE = True
    except ImportError:
        logger.debug("nodewatch not installed — tracing disabled")


class ChatbotService:
    def __init__(self):
        # ─── Step 2: Initialize storage once ────────────────────────────
        self._nodewatch_storage = None
        if _NODEWATCH_AVAILABLE:
            try:
                db_path = os.getenv("NODEWATCH_DB", "./nodewatch.db")
                self._nodewatch_storage = nodewatch.SQLiteStorage(db_path)
            except Exception as exc:
                logger.warning("Nodewatch storage init failed: %r", exc)

    async def handle_query(
        self, graph, user_prompt: str, user_id: str = "", conversation_id: str = ""
    ) -> dict:
        # ─── Step 3: Create tracker per request ─────────────────────────
        _tracker = None
        _trace = None
        if _NODEWATCH_AVAILABLE and self._nodewatch_storage is not None:
            try:
                _tracker = nodewatch.GraphTracker(
                    "my-agent",
                    metadata={
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                    },
                    storage=self._nodewatch_storage,
                    live=True,
                )
            except Exception as exc:
                logger.warning("Tracker creation failed: %r", exc)
                _tracker = None

        # ─── Step 4: Inject tracker as callback ─────────────────────────
        config = {
            "configurable": {"thread_id": conversation_id},
            "recursion_limit": 100,
        }
        if _tracker is not None:
            config["callbacks"] = [_tracker]

        # ─── Step 5 & 7: Invoke with error handling ─────────────────────
        try:
            result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": user_prompt}]},
                config=config,
            )
            response_text = result["messages"][-1].content

            # Finalize on success
            if _tracker is not None:
                _trace = _tracker.finalize(
                    query=user_prompt, final_response=response_text
                )
                self._nodewatch_storage.save(_trace)

        except Exception as exc:
            logger.exception("Graph execution failed: %r", exc)
            response_text = f"Error: {exc}"
            # Always finalize — partial traces help debug failures
            if _tracker is not None:
                try:
                    _trace = _tracker.finalize(query=user_prompt, final_response="")
                    self._nodewatch_storage.save(_trace)
                except Exception:
                    pass

        # ─── Step 6: Access trace metrics ───────────────────────────────
        trace_summary = None
        if _trace is not None:
            trace_summary = {
                "run_id": _trace.run_id,
                "total_tokens": _trace.total_tokens,
                "total_cost_usd": round(_trace.total_cost_usd, 6),
                "total_llm_calls": _trace.total_llm_calls,
                "total_tool_calls": _trace.total_tool_calls,
                "nodes_visited": _trace.nodes_visited,
                "duration_ms": round(_trace.total_duration_ms, 1),
            }

        return {"response": response_text, "trace": trace_summary}

    # ─── Step 8: Cleanup on shutdown ────────────────────────────────────
    def shutdown(self):
        if self._nodewatch_storage is not None:
            try:
                self._nodewatch_storage.close()
            except Exception as exc:
                logger.warning("Error closing storage: %r", exc)
```

Once traces are stored, use the CLI or dashboard to explore them:

```bash
nodewatch list-runs --last 5          # see recent runs
nodewatch inspect <run_id>            # per-node breakdown
nodewatch report --graph my-agent     # aggregate stats
nodewatch dashboard                   # interactive TUI
```

---

