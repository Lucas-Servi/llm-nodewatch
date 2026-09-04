# Internals

Maintainer notes: how the pieces fit together, and the decisions that look wrong until you
know why they were made. Read this before changing token accounting, the pricing lookup, or
anything in `tracker.py`. For using nodewatch, start at the [README](../README.md); for
contributing, [CONTRIBUTING.md](../CONTRIBUTING.md).

## Overview
LangGraph-specific benchmarking and observability tool. Instruments LangGraph agent graphs via LangChain callbacks to capture per-node token usage, latency, tool calls, errors, and cost — with zero modifications to graph source code.

## Architecture
```
GraphTracker (callback handler)
    ├── Intercepts: on_chain_start/end, on_llm_start/end, on_tool_start/end
    ├── Correlates events via run_id → parent_run_id + langgraph_node metadata
    └── Produces: RunTrace → NodeSpans → LLMCalls + ToolCalls

BenchmarkRunner
    ├── Orchestrates multi-query comparisons across graph variants
    └── Supports warmup runs and state builders

Storage (SQLite)
    └── Tables: runs, node_spans, llm_calls, tool_calls

CLI (Typer)
    └── Commands: list-runs, inspect, compare, ab-compare, ab-init, ab-run,
                  export, report, delete, dashboard, mcp, pricing show
```

## Key Files
- `src/nodewatch/tracker.py` — Core callback handler (node correlation, token extraction)
- `src/nodewatch/models.py` — Data models with computed cost/token properties
- `src/nodewatch/runner.py` — BenchmarkRunner for comparative runs
- `src/nodewatch/reporter.py` — Markdown/JSON report generation
- `src/nodewatch/storage/sqlite.py` — SQLite persistence
- `src/nodewatch/api.py` — FastAPI HTTP API (mountable router + standalone app)
- `src/nodewatch/cli.py` — Typer CLI
- `src/nodewatch/dashboard.py` — Textual TUI dashboard (`nodewatch dashboard`)
- `src/nodewatch/abrun.py` — JSON-config A/B runner (HTTP + direct-model transports)
- `src/nodewatch/inspector.py` — Static graph topology analysis

## Dashboard sorting (`dashboard.py`)

Both DataTables sort: **click a column header** to sort by it, click again to reverse
(`s` cycles columns / `S` reverses on the focused table). Defaults are **Conv ID
descending** for Conversations (ids grow over time, so newest first) and Date descending
for Runs. Two invariants a future edit will otherwise silently break:

- **Sort the model objects, never the rendered cells.** Cells are pre-formatted strings
  (`"1,234"`, `"$0.42"`, `_fmt_dur` emitting mixed `ms`/`s`), so `DataTable.sort()` on cell
  values compares text and is wrong. `_CONV_SORT_KEYS` / `_RUNS_SORT_KEYS` are index-aligned
  with `_CONV_COLUMNS` / `_RUNS_COLUMNS` and key off `ConversationStats` / `RunTrace` fields.
- **Sort state is keyed by column *index*, not `ColumnKey`.** Textual caches header labels:
  mutating `Column.label` in place leaves the old text on screen through `refresh()` and
  `clear()`, so `_render_table` rebuilds the columns (`clear(columns=True)` + `add_columns`)
  to repaint the ▼/▲ indicator — which regenerates the column keys every draw.
- Rows are cached **per table** (`_conv_stats`, `_runs`) so re-sorting never refetches:
  `list_runs` is an N+1 read locally and a ~1.5s round-trip remotely. Per-table because
  `_load_conversation_runs` leaves the Runs table holding a conversation-filtered subset.
- `compute_conversation_stats` (`stats.py`) still sorts by cost — deliberately unchanged,
  because the MCP `list_conversations` tool shares it. Ordering is a dashboard concern.

## Run-id arguments accept a prefix (`cli.py`)

`inspect` / `export` / `delete` / `compare` resolve run ids through `_resolve_trace`: exact
id first (one lookup, no scan), else a **unique run-id prefix**, else an actionable error —
an ambiguous prefix lists the candidates, and a **conversation id** (e.g. `nodewatch inspect
593`) is detected and answered with its run ids plus a `list-runs -c` hint. Resolution scans
recent runs **client-side** (`_scan_recent_runs`, paged 200×5) rather than server-side, so it
works against an already-deployed remote API; `storage.load` stays an exact lookup. A
server-side `resolve_run_id(prefix)` would save the scan but is pointless until the remote is
redeployed. Conversation matching uses `models.trace_matches_conversation` (column *or*
`metadata["conversation_id"]`, since the column is often empty) — shared with
`ABExperiment._matches_conv`.

## Token & Cost Accounting

- **Cache tokens**: `tracker.py` normalizes every usage source through `_usage_from_metadata`
  (LangChain `usage_metadata`) or `_usage_from_raw` (raw provider dict). LangChain nests cache
  counts under `usage_metadata["input_token_details"]` as `cache_read` / `cache_creation` (NOT
  the raw Anthropic `cache_*_input_tokens` keys), and its `input_tokens` is **inclusive** of
  cache. The rest of the codebase (cost in `models.py`, cache-hit rate in `stats.py`) assumes
  `input_tokens` is **exclusive** of cache, so `_usage_from_metadata` subtracts cache back out.
  When touching token extraction, keep `LLMCall.input_tokens` exclusive and cache tracked
  separately, or cost/stats will silently break.
- **Reasoning/thinking**: providers already count reasoning inside `output_tokens`, so
  `output_token_details.reasoning` is deliberately NOT mapped to `thinking_tokens` (which is
  billed separately at output rate) — mapping it would double-count.

## Filtered-message accounting

Per-node **content-filter** counts (a safety filter blocking the model output) are derived,
not stored — there is **no schema migration**:

- `tracker.on_llm_end` populates `LLMCall.stop_reason`. It first reads `gen.generation_info`,
  then **falls back to `gen.message.response_metadata`** trying `stopReason` (camelCase) →
  `stop_reason` → `finish_reason` → `model_stop_reason`. The fallback is load-bearing:
  `ChatBedrockConverse` exposes the stop reason ONLY at `response_metadata["stopReason"]`,
  never in `generation_info` — without it every Bedrock call recorded an empty `stop_reason`
  and content-filter events were invisible.
- `models.is_filtered_stop(stop_reason)` → `stop_reason ∈ {content_filtered, refusal}`
  (Bedrock uses `content_filtered`, the AWS-external/public Anthropic API uses `refusal`).
  Derived properties: `LLMCall.content_filtered`, `NodeSpan.filtered_count`,
  `RunTrace.total_filtered`.
- Surfaced everywhere: `inspect`/`compare`/`report` CLI (a `Filt` column), `trace_to_markdown`
  / `comparison_to_markdown` node tables, `trace_to_json` (`content_filtered` per call,
  `filtered_count` per node, `total_filtered` per run), and the API `_summarize`/`_node_summary`/
  `chart_node_breakdown`/`chart_node_comparison` (`avg_filtered`).
- **Caveat — counts only callback-visible filtered calls.** This reflects filtered LLM calls
  the nodewatch LangChain callback actually observes. If an agent's content-filter *fallback*
  path retries on a separate client without the tracker callbacks attached, those calls won't be
  counted, so `total_filtered` can under-count vs. scraping the upstream server log. The feature
  is forward-looking: historical rows with an empty `stop_reason` stay `filtered=0`.

## Tool-call success classification

`ToolCall.success` used to be a **constant `True`**: `tracker.on_tool_end` hardcoded it, and
`success=False` was set only in `on_tool_error`, which **never fires** for a large class of
tools — MCP and HTTP-backed tools typically return an error *payload* instead of raising. Result
on a real fleet: a self-QA smoke test recorded **every tool call as `success: true`, `error:
null` while several genuinely failed**. Same shape as "Filtered-message accounting" above:
derived at capture time, **no schema migration** (`ToolCall.success` / `.error` already existed).

- `models.classify_tool_error(output) -> str | None` is a pure function; `on_tool_end` calls it
  and sets `success` / `error` from the result. **Pass the FULL output string, not
  `output_preview`** — the preview is truncated to 128 chars, which often cuts before the marker,
  and the JSON shapes can only be parsed whole. `output_preview` / `output_size` behaviour is
  unchanged.
- Recognizers, all **anchored**: a leading `[... Error]` bracket tag (the
  `[Web Search Error]` / `[Web Read Error]` shape many tool wrappers emit), leading `Error:` /
  `Errors:` / `An error occurred` (the usual generic-catch wording) / `Traceback (most recent call
  last):`, and a whole-output JSON parse looking for a top-level `error` string, `errorCode`, or
  `result.errorCode`. That JSON branch covers the common `{"error": …}` shape, APIs returning
  `{"errorCode": 1, …}`, and typed `ErrorResult` models. **`errorCode: 0` means SUCCESS** in that
  convention and must not be flagged.
- ⚠️ **Never substring-match `"error"`.** Real tool *content* contains the word: any ECC hardware
  catalog is full of *"Error-Correcting …"* parts, and technical corpora routinely have hundreds
  of field values containing it. A naive `"error" in output` marks those honest results as
  failures — the same defect class, inverted. The `Errors?(?!-)` negative lookahead exists for
  exactly this. The negative test (an `Error-`-prefixed payload stays `success=True`) is the one
  that matters.
- ⚠️ **Metrics discontinuity at the deploy date.** Historical rows were captured with the
  hardcoded `True`, so tool-success rates step down the day this shipped. A trend view will read
  the fix as a regression — it isn't; the old 100% was the bug. Compare only within one side of
  the boundary.

## Tool-call I/O is the trace's whole point — don't drop it in the serializer

`ToolCall` and the SQLite layer have always carried `input`, `output_preview` and `output_size`, but
`mcp_server._trace_to_detail` serialized only `tool_name / duration_ms / success / error`. So an
assistant asking `get_run` through MCP could see *that* a tool ran and *that* it "succeeded", and
nothing about what it was asked or what it answered.

That is the difference between a trace and a stopwatch. The case that forced this: a search tool
that returned 1 of 30 available result groups while reporting success. Diagnosing it was
**impossible through nodewatch** — it took dumping the run from the remote API by hand and then
reading the agent's own server log. All three fields are now in the payload.

The stored widths were the second half of the same problem. `input` was cut at **256** chars and
`output_preview` at **128**, and in that one call the 256 truncated a 23-term search-terms argument
mid-list while the 128 cut off before the omission notice — the single line that explained the
failure. They are now `TOOL_INPUT_CHARS` / `TOOL_OUTPUT_PREVIEW_CHARS` (**2000 / 1000**),
overridable per session via `NODEWATCH_TOOL_INPUT_CHARS` / `NODEWATCH_TOOL_OUTPUT_CHARS`.

⚠️ `classify_tool_error` still reads the **full** output, never the preview — the error marker
routinely sits past any preview cut. `test_tracker_classifies_from_full_output_not_the_preview` pins
that, and it now sizes its payload off `TOOL_OUTPUT_PREVIEW_CHARS` rather than a literal `128`: the
first version asserted `len(output_preview) == 128`, which turned a deliberate cap change into a
test failure instead of testing the behaviour that matters.

## Pricing lookup is longest-match (`models.py`)

`prices_for_model()` matches a pricing key as a **substring** of the served model id, because
real ids carry vendor/region decoration the table doesn't repeat
(`us.anthropic.claude-opus-4-8-v1:0` must find the `claude-opus-4-8` row). **Longest match wins,
and that is what makes substring matching safe.** The previous `startswith(prefix) or prefix in
key` took the *first* dict hit, so billing depended on key order in `data/pricing.json` — a file
users are explicitly invited to replace via `NODEWATCH_PRICING`. Two keys make this concrete:
`o3` is a substring of a great many ids, and `gpt-5` is a prefix of `gpt-5.5`. Pinned in
`tests/test_pricing.py`.

## A/B model benchmarking

Compare the same query suite served by two model versions (e.g. Opus 4.8 vs 4.7), isolating
the model as the only variable. Core pieces (`stats.py` + `experiment.py`):

- **Comparator** — `compute_ab_comparison(traces_a, traces_b, expected_a, expected_b, …)`
  returns an `ABComparison`. Cohorts are identified by the model that **actually served** the
  run (derived from `llm_calls.model`), **NOT** `graph_name` — in a multi-model agent setup both
  versions often share one `graph_name` (e.g. `"v2"`), so grouping by graph_name would merge them.
  Served-model extraction is `stats._served_models` (no provider pre-filter — any vendor verifies
  by substring). Questions are paired by key precedence `metadata["ab_question_id"]` → normalized
  `query` text (the `conversation_id` *column* is deliberately not used — it is frequently empty).
  Only **matched node paths** are compared (`node_sig` de-dupes node order + strips `_tools`): a
  2-expert run vs a 4-expert run is not a fair pair. It never raises on a served-model mismatch —
  it sets `verified_ok=False` and the caller decides.
- **Phased runner** — `ABExperiment(storage, ExperimentSpec(...))` is **transport-agnostic**:
  the caller supplies `query_fn(question_text, conversation_id, **kwargs)` that issues ONE query
  over any transport (a live HTTP API, in-process graph, direct model call, …). The runner owns
  phase × question × rep iteration, conversation-id tagging, **resumability** (skips conv-ids that
  already resolve to a good run — `_resolve_run_by_conv` tolerates the empty column via a metadata
  fallback), reads runs back from storage, self-verifies the served model per phase, and (for
  exactly 2 phases) computes the comparison.
- **Surfaces**: CLI `nodewatch ab-compare --expected-a opus-4-8 --expected-b opus-4-7`; API
  method `ab_compare`; `RemoteClient.ab_compare`; `reporter.ab_comparison_to_markdown`.

### Config-driven A/B runs (`abrun.py` + `nodewatch ab-run`)

`abrun.py` turns a JSON config into a full A/B run on top of the runner above — no hand-written
`query_fn`. `nodewatch ab-run <session>` (or `--config <file>`) loads it, drives the phases, and
prints the verification + per-question deltas (`_render_ab`, shared with `ab-compare`).

- **Testing sessions** — a session is a self-contained folder (`abrun.resolve_session_dir`: a bare
  name → `./testing_sessions/<name>`, base overridable via `NODEWATCH_SESSIONS_DIR`; a path used
  as-is). `nodewatch ab-init <session>` scaffolds it with a template `config.json`
  (`abrun.init_session` / `default_config_template`; `--from` seeds from an existing file, `-t`
  picks the transport). `ab-run <session>` loads `<dir>/config.json`, records to `<dir>/runs.db`
  (unless the config's top-level `"db"` or `--db` overrides), and dumps `ab_<model>.json` +
  `results.json` into the folder. Legacy `ab-run --config <file>` (with `--db`/`--out-dir`) still
  works; the `db` resolution order is `--db` > config `db` > session `runs.db` > `DEFAULT_DB`.

- **Two transports** (`transport`): `"http"` POSTs each prompt to an agent API via stdlib
  `urllib` (field names config-driven; `${VAR}` in url/headers/body expanded from env at load);
  `"model"` calls the model directly via its LangChain client (`ChatAnthropic`/`ChatOpenAI`/
  `ChatBedrockConverse`, lazily imported — NOT `init_chat_model`, which needs the `langchain`
  meta-package) with a `GraphTracker` attached so the call records like any run. The model
  transport is credential-gated (`ANTHROPIC_API_KEY`/`OPENAI_API_KEY`/AWS creds + the provider
  extra `pip install "llm-nodewatch[ab-model]"`); a miss raises a clear `RuntimeError`.
- **Two switch modes** (`experiment.switch_mode`, http only): `"per_request"` injects the model
  into each request body (needs `api.model_field`); `"manual"` is for servers whose model is fixed
  at startup — the runner calls a `pause_hook` before each phase so an operator can reconfigure +
  restart, then self-verifies. The `"model"` transport always selects per request (rejects manual).
- `models[]` = one phase/cohort each (`id`, `request_model`, `expect`); `questions[]` run on every
  model. Exactly 2 models → pairwise comparison; >2 → per-model summaries, `comparison=None`.
- **Confirmation gate** (`experiment.pause_check`: `false`|`true`|`"message"`, parsed via
  `_parse_pause_check` → `ABRunConfig.pause_check`): a user action before the run spends anything.
  The runner itself never blocks — callers gate. CLI `ab-run` prompts `y/N` (skip with `--yes`);
  the MCP run tools return `{"status":"confirmation_required", "preview": preview_ab_config(cfg), …}`
  and re-run on `confirm=True`. `preview_ab_config` summarizes models × questions × reps + a
  cost note, with no side effects.
- Recording needs a **local** DB the target API writes to (remote mode is read-only). See
  `examples/ab_config.example.json` (HTTP) and `examples/ab_config.model.example.json` (direct).

### MCP tools for model testing (`mcp_server.py`)

⚠️ **The server is on `mcp>=2,<3`** (`mcp.server.mcpserver.MCPServer`). mcp 2.0 *deleted*
`mcp.server.fastmcp`, so the old `FastMCP` import was a hard `ModuleNotFoundError` in any
current environment — the migration is the import plus the constructor name; the 13
`@mcp.tool()` decorators and `mcp.run(transport="stdio")` are unchanged. Keep the `<3` bound:
the `mcp` extra used to say a bare `mcp>=1.0.0`, and an open-ended `mcp` pin is exactly what
silently resolved a broken dependency pair elsewhere in the workspace.

Tools added to the (otherwise read-only) MCP server so an assistant can scaffold, edit, and
run A/B tests end-to-end. All return plain dicts and never raise — failures come back as
`{"error": …}`. The session tools mirror the `ab-init`/`ab-run` CLI.

- **Session scaffold/edit** (filesystem only, no model calls):
  - `init_ab_session(session, transport="model", force=False)` — generate the session folder +
    template `config.json`; returns the config dict.
  - `read_ab_config(session)` / `write_ab_config(session, config)` — read and **edit** the testing
    JSON. `write_ab_config` validates via `parse_ab_config` before writing (invalid → error,
    nothing written).
- **Run** (⚠ calls the models / writes runs):
  - `run_ab_session(session, confirm=False)` — load `<session>/config.json`, run it (any
    transport), record to the session's `runs.db` (or the config's `db`), dump `ab_<model>.json` +
    `results.json` into the folder. If the config sets `pause_check`, returns
    `confirmation_required` until called with `confirm=True`. The http/`manual` interactive pause
    is CLI-only; over MCP it runs without pausing and relies on served-model verification.
  - `run_model_ab(prompts, models, provider="anthropic", reps=1, system=None, session=None,
    db=None, out_dir=None, pause_check=False, confirm=False)` — one-call test from two plain lists;
    builds an in-memory `transport:"model"` config and reuses `run_ab_config`. With `session`, it
    also creates the folder + `config.json` and dumps artifacts there. `pause_check=True` makes the
    first call return `confirmation_required` (config.json still written) until `confirm=True`.
    Credential-gated.
- **Analyze** (read-only): `compare_models(expected_a, expected_b, since=None, limit=200)` —
  A/B-compares already-recorded cohorts (partition by served model + `compute_ab_comparison`).

## Usage

```python
import nodewatch

# Instrument a graph run
tracker = nodewatch.GraphTracker("v2")
result = await graph.ainvoke(state, config=tracker.config)
trace = tracker.finalize(query="...", final_response="...")

# Full comparison
runner = nodewatch.BenchmarkRunner(storage=nodewatch.SQLiteStorage("bench.db"))
report = await runner.run_comparison(graphs, queries, state_builders)
print(nodewatch.comparison_to_markdown(report))
```

## HTTP API

Mount in any FastAPI app or run standalone:

Requires the `server` extra (`pip install 'llm-nodewatch[server]'`).

```python
# Mount in existing app
from nodewatch.api import create_router
app.include_router(create_router("/path/to/db"), prefix="/nodewatch")

# Or standalone
# NODEWATCH_DB=/path/to/db uvicorn nodewatch.api:app --port 8052
```

### There are only TWO routes — it is a dispatcher, not a REST API

⚠️ `create_router()` registers exactly **`POST /`** and **`GET /methods`**. There are no
`GET /runs`, `GET /runs/{id}`, `/charts/*` or `DELETE` routes; an earlier version of this file
listed ~14 of them and they never existed. Verify with
`python -c "from nodewatch.api import create_router; print([r.path for r in create_router().routes])"`
before documenting an endpoint.

Everything goes through the dispatcher:

```bash
curl -X POST localhost:8052/ -H 'Content-Type: application/json' \
  -d '{"method": "list_runs", "args": {"limit": 5}}'
```

`GET /methods` reflects the registry (19 methods) with each one's argument names, so it is the
authoritative list. The methods are: `get_active_runs`, `get_run_live`, `get_logs`, `list_runs`,
`get_run`, `get_run_markdown`, `get_run_nodes`, `delete_run`, `report`, `stats`,
`chart_cost_over_time`, `chart_tokens_over_time`, `chart_latency_over_time`,
`chart_node_breakdown`, `chart_node_comparison`, `chart_tool_frequency`, `chart_model_usage`,
`chart_v1_vs_v2`, `ab_compare`.

`nodewatch.api.app` (the standalone app) is built **lazily via a module-level `__getattr__`**, so
that importing the module doesn't create a database as a side effect. A consequence worth
knowing: each attribute access returns a *new* app instance, and the included router is nested
under a `_IncludedRouter`, so naively walking `app.routes` will not show the two real routes —
introspect `create_router()` instead.

## Testing
```bash
pip install -e ".[dev]"
pytest
```

⚠️ **Install editable from THIS directory before trusting a green run.** The dist name
`llm-nodewatch` can be bound by an editable `.pth` pointing at a *different* checkout of the
same package (this happened: `_editable_impl_llm_nodewatch.pth` pointed at the internal mirror,
so `pytest` here silently tested that tree instead). Confirm with
`python -c "import nodewatch; print(nodewatch.__file__)"` — it must resolve inside this repo.
`PYTHONPATH=src pytest` also works and doesn't disturb the ambient install.

⚠️ **`conftest.py`'s mock model is duck-typed, not a `BaseChatModel`**, so it never fires the
`on_llm_*` callbacks — graph fixtures using it produce node spans with **zero tokens**. That is
why `test_tracker.py`'s graph tests assert on spans and duration, not cost. To exercise token or
cost accounting end-to-end you need a real `BaseChatModel` subclass returning `usage_metadata`
(see `tests/test_tracker_async.py`, which drives the callbacks directly instead).

## Environment
- Python 3.11+ (tested on 3.14)
- Dependencies: langchain-core, langgraph, typer, rich
- Optional: `pip install -e ".[server]"` for the FastAPI HTTP API (fastapi, uvicorn).
  **There is no `api` extra** — the name is `server`, and a wrong extra name only emits a
  pip warning, so `.[api]` silently installs nothing.
- No external infrastructure required (SQLite only)
- `NODEWATCH_DB` — path to the SQLite database (default: `nodewatch.db`)
- `NODEWATCH_API_TOKEN` — if set, all API routes require Bearer token auth
- `NODEWATCH_PRICING` — path to custom pricing JSON (default: bundled `src/nodewatch/data/pricing.json`)
- `NODEWATCH_SESSIONS_DIR` — base dir for A/B testing sessions (default: `./testing_sessions`)
- `NODEWATCH_LOG_PATH` — log file tailed by the dashboard's Logs tab and `GET /logs`
- `NODEWATCH_URL` / `NODEWATCH_TOKEN` — remote-mode API base and bearer token
- `NODEWATCH_DEBUG` — set `1` to surface raw remote error bodies
- ⚠️ **`NODEWATCH_ENABLED` is NOT read by this library.** It is a caller-side convention
  (see `examples/langgraph_integration.py`, which does its own `os.getenv` guard). Do not
  document it as a nodewatch knob — users set it and are surprised tracking continues.

## Security

- SQLite writes are protected by a threading lock (concurrent-safe within a process)
- CLI output paths are sanitized: system dirs refused, overwrite requires `--force`
- Remote API errors are sanitized (no raw server text in exceptions)
- Optional Bearer token auth for the HTTP API via `NODEWATCH_API_TOKEN`
- Dynamic login module loading validates return type
