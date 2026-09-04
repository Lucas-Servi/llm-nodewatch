# Environment variables

Every variable the library itself reads.

| Variable | Default | Description |
|----------|---------|-------------|
| `NODEWATCH_DB` | `nodewatch.db` | Path to the SQLite database (local mode) |
| `NODEWATCH_URL` | *(unset)* | Remote API base URL. Setting it switches the CLI and dashboard to [remote mode](remote.md) |
| `NODEWATCH_TOKEN` | *(unset)* | Pre-obtained bearer token for the remote API |
| `NODEWATCH_LOGIN_MODULE` | *(unset)* | Python module exposing a login function, used when no token is set |
| `NODEWATCH_LOGIN_FUNCTION` | `login` | Function name to call in that module. Must return a `str` |
| `NODEWATCH_API_TOKEN` | *(unset)* | **Server side.** If set, every [HTTP API](http-api.md) route requires this bearer token |
| `NODEWATCH_PRICING` | *(bundled `nodewatch/data/pricing.json`)* | Path to a custom [pricing](pricing.md) JSON |
| `NODEWATCH_SESSIONS_DIR` | `./testing_sessions` | Base directory for [A/B testing sessions](benchmarking.md) |
| `NODEWATCH_LOG_PATH` | *(unset)* | Log file tailed by the dashboard's Logs tab and `GET /logs` |
| `NODEWATCH_DEBUG` | *(unset)* | Set `1` to include raw remote response bodies in error messages |

## `NODEWATCH_ENABLED` is not one of them

You may see `NODEWATCH_ENABLED` referenced in integration examples. **The library does not
read it.** It is a caller-side convention: you check it yourself and skip creating a tracker,
which is what makes tracking genuinely zero-cost when disabled rather than merely quiet.

```python
NODEWATCH_ON = os.getenv("NODEWATCH_ENABLED", "1") != "0"

tracker = nodewatch.GraphTracker("my-graph") if NODEWATCH_ON else None
config = tracker.config if tracker else {}
```

See [`examples/langgraph_integration.py`](../examples/langgraph_integration.py) for the full
feature-flagged pattern.

## What gets tracked

For each graph execution:

- **Per node** — input/output/thinking/cache tokens, cost, latency, iteration (loop) count
- **LLM calls** — model, provider, token breakdown, stop reason, content-filter detection, errors
- **Tool calls** — tool name, duration, success/failure, error message
- **Overall** — total duration, node execution order, final response
