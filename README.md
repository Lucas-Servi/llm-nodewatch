<p align="center">
  <img src="https://raw.githubusercontent.com/Lucas-Servi/llm-nodewatch/main/assets/logo.png" alt="nodewatch" width="330">
</p>

<p align="center">
  <a href="https://github.com/Lucas-Servi/llm-nodewatch/actions/workflows/ci.yml"><img src="https://github.com/Lucas-Servi/llm-nodewatch/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/llm-nodewatch/"><img src="https://img.shields.io/pypi/v/llm-nodewatch" alt="PyPI"></a>
  <a href="https://pypi.org/project/llm-nodewatch/"><img src="https://img.shields.io/pypi/pyversions/llm-nodewatch" alt="Python"></a>
  <a href="https://github.com/Lucas-Servi/llm-nodewatch/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
</p>

**Which node in your LangGraph agent is burning the tokens?**

nodewatch answers that. It attaches as a LangChain callback, reconstructs per-node
token/cost/latency accounting from the callback stream, and writes it to one SQLite file —
no graph changes, no account, no collector, nothing leaves your machine.

```
## Run: a1b2c3d4e5f6 | rag-agent | 2026-05-14 15:15

**Duration**: 12.3s   **Tokens**: 38,400/6,800   **Cost**: $0.85

| Node        | Model      | Tokens (in/out) | Duration | Loops | Tools | Cost   |
|-------------|------------|-----------------|----------|-------|-------|--------|
| planner     | opus-4-6   | 8,200/1,400     | 3.2s     | 1     | 0     | $0.23  |
| retriever   | -          | 0/0             | 1.8s     | 1     | 3     | $0.00  |
| analyzer    | sonnet-4-6 | 22,100/3,800    | 5.1s     | 2     | 1     | $0.42  |
| summarizer  | sonnet-4-6 | 8,100/1,600     | 2.2s     | 1     | 0     | $0.20  |
```

`analyzer` is half your bill and it looped twice. That's the number a span tree makes you
compute yourself.

## Install

The distribution is **`llm-nodewatch`**; the import name and CLI are **`nodewatch`**.

```bash
pip install "llm-nodewatch[server]"   # tracker + SQLite + optional HTTP API
pip install "llm-nodewatch[client]"   # dashboard TUI + CLI, for a remote server
```

Both roles can live in one install, or on separate machines — the CLI switches to remote
mode automatically when `NODEWATCH_URL` is set.

## Quick start

```python
import nodewatch

# 1. Wrap your graph invocation with a tracker
tracker = nodewatch.GraphTracker("my-graph")
result = await graph.ainvoke(state, config=tracker.config)
trace = tracker.finalize(query="input query", final_response=result["output"])

# 2. Persist
storage = nodewatch.SQLiteStorage("runs.db")
storage.save(trace)

# 3. View the per-node breakdown
print(nodewatch.trace_to_markdown(trace))
```

That's the whole integration. `tracker.config` is just `{"callbacks": [tracker]}`, so your
graph is untouched. Then explore from the terminal:

```bash
nodewatch list-runs --limit 10
nodewatch inspect a1b2c3      # run-id prefixes work
nodewatch dashboard           # full-screen TUI
```

## Why not LangSmith / Langfuse / Phoenix?

Use those if you want a hosted UI, dataset management and team features. Use nodewatch when
you want per-node cost accounting with nothing to operate.

|                              | nodewatch            | LangSmith     | Langfuse            | OpenLLMetry     |
|------------------------------|----------------------|---------------|---------------------|-----------------|
| Infrastructure               | one SQLite file      | SaaS account  | Postgres+ClickHouse | OTLP collector  |
| Data leaves your machine     | never                | yes           | self-host option    | to a collector  |
| Unit of aggregation          | **LangGraph node**   | span tree     | span tree           | span tree       |
| Loop/iteration count/node    | **yes**              | derive it     | derive it           | derive it       |
| Built-in A/B experiment runner | **yes**            | no            | no                  | no              |
| Runtime dependencies         | **4**                | SDK + account | SDK + stack         | instrumentation |

Two things it gets right that are easy to get wrong, and worth checking in whatever you use:

- **Cache tokens.** LangChain reports cache counts nested in
  `usage_metadata["input_token_details"]` (as `cache_read`/`cache_creation`, *not* the raw
  provider key names), and its `input_tokens` is **inclusive** of them. nodewatch subtracts
  them back out so `input_tokens` stays exclusive and cache is priced at cache rates.
  Naively summing both double-counts your input.
- **Reasoning tokens.** Providers already count reasoning inside `output_tokens`, so
  `output_token_details.reasoning` is deliberately *not* added as a separate billable
  figure. Adding it inflates cost on every thinking-enabled call.

## What you get

- **Per node** — input/output/thinking/cache tokens, cost, latency, loop count, tool calls
- **Per LLM call** — model, provider, token breakdown, stop reason, content-filter detection
- **Per tool call** — name, duration, success/failure *derived from error payloads*, not just
  raised exceptions ([why that matters](https://github.com/Lucas-Servi/llm-nodewatch/blob/main/docs/README.md#things-that-are-easy-to-get-wrong))
- **A [dashboard](https://github.com/Lucas-Servi/llm-nodewatch/blob/main/docs/dashboard.md)** — a Textual TUI: runs, conversations, inspector, stats, live, logs
- **An [MCP server](https://github.com/Lucas-Servi/llm-nodewatch/blob/main/docs/mcp.md)** — 13 tools, so an AI assistant can query your traces directly
- **[A/B benchmarking](https://github.com/Lucas-Servi/llm-nodewatch/blob/main/docs/benchmarking.md)** — compare two model versions over one query suite,
  with cohorts identified by the model that *actually served* each run and only matched node
  paths compared
- **An optional [HTTP API](https://github.com/Lucas-Servi/llm-nodewatch/blob/main/docs/http-api.md)** — mount the router in an existing FastAPI app

## Documentation

| | |
|---|---|
| [Integration guide](https://github.com/Lucas-Servi/llm-nodewatch/blob/main/docs/integration.md) | Production wiring: feature flags, per-request lifecycle, error recovery, shutdown |
| [CLI reference](https://github.com/Lucas-Servi/llm-nodewatch/blob/main/docs/cli.md) | Every command and flag |
| [Dashboard](https://github.com/Lucas-Servi/llm-nodewatch/blob/main/docs/dashboard.md) | The TUI, with screenshots |
| [MCP server](https://github.com/Lucas-Servi/llm-nodewatch/blob/main/docs/mcp.md) | Tools and client configuration |
| [A/B benchmarking](https://github.com/Lucas-Servi/llm-nodewatch/blob/main/docs/benchmarking.md) | Comparing graph variants and model versions |
| [HTTP API](https://github.com/Lucas-Servi/llm-nodewatch/blob/main/docs/http-api.md) | Endpoints and mounting |
| [Remote mode](https://github.com/Lucas-Servi/llm-nodewatch/blob/main/docs/remote.md) | Driving a remote server from your laptop |
| [Pricing](https://github.com/Lucas-Servi/llm-nodewatch/blob/main/docs/pricing.md) | The bundled table and how to override it |
| [Environment variables](https://github.com/Lucas-Servi/llm-nodewatch/blob/main/docs/env.md) | Every variable the library reads |

## Requirements

Python 3.11+. Four runtime dependencies: `langchain-core`, `langgraph`, `typer`, `rich`.
Node identity comes from LangGraph's `langgraph_node` metadata, so **LangGraph is required** —
on plain LangChain runnables there are no node spans to attribute tokens to.

## Contributing

Issues and PRs welcome — see [CONTRIBUTING.md](https://github.com/Lucas-Servi/llm-nodewatch/blob/main/CONTRIBUTING.md). If you're touching token or
cost accounting, please read its accounting section first; the conventions there are load-bearing
and the tests encode them.

## License

MIT — see [LICENSE](https://github.com/Lucas-Servi/llm-nodewatch/blob/main/LICENSE). Built by [Lucas Servi](https://github.com/Lucas-Servi).
