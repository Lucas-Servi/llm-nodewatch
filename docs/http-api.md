# HTTP API

Mount in an existing FastAPI app or run standalone:

```python
from nodewatch.api import create_router

app.include_router(create_router("/path/to/runs.db"), prefix="/nodewatch")
```

Standalone:

```bash
NODEWATCH_DB=./runs.db uvicorn nodewatch.api:app --port 8052
```

## Single-endpoint dispatcher

All operations go through `POST /` with a JSON body:

```json
{"method": "list_runs", "args": {"graph_name": "v2", "limit": 10}}
```

Response:

```json
{"result": [...]}
```

Use `GET /methods` to discover all available methods and their arguments.

## Available methods

| Method | Args | Description |
|--------|------|-------------|
| `list_runs` | `graph_name?`, `since?`, `limit?` | List stored runs |
| `get_run` | `run_id` | Full trace JSON |
| `get_run_markdown` | `run_id` | Markdown report |
| `get_run_nodes` | `run_id` | Per-node breakdown |
| `delete_run` | `run_id` | Delete a trace |
| `report` | `graph_name?`, `since?`, `limit?` | Aggregate stats summary |
| `stats` | `graph_name?`, `limit?` | Stats by graph |
| `chart_cost_over_time` | `graph_name?`, `since?`, `limit?` | Cost time series |
| `chart_tokens_over_time` | `graph_name?`, `since?`, `limit?` | Token time series |
| `chart_latency_over_time` | `graph_name?`, `since?`, `limit?` | Duration time series |
| `chart_node_breakdown` | `run_id` | Per-node bar chart data |
| `chart_node_comparison` | `graph_name?`, `limit?` | Avg metrics per node |
| `chart_tool_frequency` | `graph_name?`, `since?`, `limit?` | Tool usage histogram |
| `chart_model_usage` | `since?`, `limit?` | Token/cost by model |
| `chart_v1_vs_v2` | `since?`, `limit?` | Graph variant comparison |
| `get_active_runs` | *(none)* | Currently executing runs (live mode) |
| `get_run_live` | `run_id` | Full trace with status metadata |
| `get_logs` | `lines?` | Tail of the file named by `NODEWATCH_LOG_PATH`; backs the dashboard's Logs tab |
| `ab_compare` | `expected_a?`, `expected_b?`, `since?`, `limit?` | A/B model comparison (served-model cohorts) |

That is the complete registry — `GET /methods` returns the same list with argument names, so
it is authoritative if this table ever drifts.

## Authentication

Optional and off by default. Set `NODEWATCH_API_TOKEN` on the server and every route in the
router requires `Authorization: Bearer <token>`:

```bash
NODEWATCH_API_TOKEN=$(openssl rand -hex 32) NODEWATCH_DB=./runs.db \
  uvicorn nodewatch.api:app --port 8052
```

Clients pass it as `NODEWATCH_TOKEN` (see [remote mode](remote.md)). With no token set the
dependency is a no-op, so mounting the router in an app that is already authenticated adds
no second check.

⚠️ Remote mode is **read-only for recording**: traces are written by the process running your
graph, not over the API. An A/B run that needs to record must have local write access to the
database ([details](benchmarking.md)).

