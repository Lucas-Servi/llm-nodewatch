# CLI

The CLI operates in two modes:

| Mode | When | Data source |
|------|------|-------------|
| **Local** | `NODEWATCH_URL` is unset, or `--local` flag is passed | SQLite database on disk |
| **Remote** | `NODEWATCH_URL` is set in `.env` or environment | HTTP API on remote server |

## Run IDs

`inspect`, `export`, `delete` and `compare` accept a full run id **or a unique prefix**:

```bash
nodewatch inspect 3c4242b5e052   # full id
nodewatch inspect 3c4242         # unique prefix — same run
```

An ambiguous prefix lists the matching runs instead of guessing. Passing a **conversation
id** by mistake is recognised and answered with that conversation's run ids:

```console
$ nodewatch inspect 593
'593' is a conversation ID, not a run ID — 3 runs:
  3c4242b5e052, 9b2a33013f32, 07a9401c03d8
Try: nodewatch list-runs -c 593
```

## Local mode

Reads directly from a SQLite database file:

```bash
export NODEWATCH_DB=./runs.db
nodewatch list-runs --last 5
```

## Remote mode

Queries a remote nodewatch API server (see [remote mode](remote.md) for setup):

```bash
# Configured via .env (NODEWATCH_URL + auth)
nodewatch list-runs --last 5
```

## Switching between modes

```bash
# Force local mode even if NODEWATCH_URL is set
nodewatch --local list-runs --last 5
nodewatch -L report --last 10

# Remote is the default when NODEWATCH_URL is configured
nodewatch list-runs --last 5
```

## `nodewatch list-runs`

Display a table of stored runs with key metrics. Use filters to narrow results.

```bash
nodewatch list-runs                         # All recent runs (default: 20)
nodewatch list-runs --graph v2 --last 10    # Last 10 runs of graph "v2"
nodewatch list-runs --since 2026-05-01      # Runs after a specific date
```

Example output:

```
┌──────────────┬───────────┬──────────────────────────┬──────────┬─────────┬────────┬──────────────────┐
│ Run ID       │ Graph     │ Query                    │ Duration │  Tokens │   Cost │ Date             │
├──────────────┼───────────┼──────────────────────────┼──────────┼─────────┼────────┼──────────────────┤
│ a1b2c3d4e5f6 │ rag-agent │ What are the key findin  │    12.3s │  45,200 │ $0.850 │ 2026-05-14 15:15 │
│ 7c4e2af91b03 │ rag-agent │ Summarize the quarterly  │     8.1s │  32,400 │ $0.620 │ 2026-05-14 14:02 │
│ f9e8d7c6b5a4 │ qa-bot    │ How do I reset my passw  │     2.4s │   8,900 │ $0.120 │ 2026-05-14 12:45 │
└──────────────┴───────────┴──────────────────────────┴──────────┴─────────┴────────┴──────────────────┘
```

**Options:**
- `--graph, -g` — Filter by graph name (e.g., `v1`, `v2`)
- `--since, -s` — Show only runs after this date (ISO format: `2026-05-01`)
- `--limit, -n` — Maximum number of results (default: 20)
- `--db` — Database path (overrides `NODEWATCH_DB`)

---

## `nodewatch inspect`

Show a detailed per-node breakdown of a single run: model used, tokens in/out, duration, loop count, tool calls, and cost for every node.

```bash
nodewatch inspect a1b2c3d4e5f6
```

Example output:

```
## Run: a1b2c3d4e5f6 | rag-agent | 2026-05-14 15:15

**Query**: What are the key findings in the latest report?
**Duration**: 12.3s
**Tokens**: 38,400/6,800 (total: 45,200)
**Cost**: $0.85
**Tool calls**: 4
**LLM calls**: 5

### Node Breakdown

| Node                   | Model        | Tokens (in/out) | Duration | Loops | Tools | Cost   |
|------------------------|--------------|-----------------|----------|-------|-------|--------|
| planner                | opus-4-6     | 8,200/1,400     | 3.2s     | 1     | 0     | $0.23  |
| retriever              | -            | 0/0             | 1.8s     | 1     | 3     | $0.00  |
| analyzer               | sonnet-4-6   | 22,100/3,800    | 5.1s     | 2     | 1     | $0.42  |
| summarizer             | sonnet-4-6   | 8,100/1,600     | 2.2s     | 1     | 0     | $0.20  |
```

---

## `nodewatch compare`

Side-by-side comparison of two runs. Useful for A/B testing graph variants or measuring the effect of prompt changes.

```bash
nodewatch compare a1b2c3d4e5f6 7c4e2af91b03
```

Example output:

```
┌────────────────┬──────────────────────────┬──────────────────────────┬──────────────┐
│ Metric         │ rag-agent (a1b2c3d4e5f6) │ rag-agent (7c4e2af91b03) │           Δ  │
├────────────────┼──────────────────────────┼──────────────────────────┼──────────────┤
│ Duration       │                  12300ms │                   8100ms │    -4200ms   │
│ Input tokens   │                  38,400  │                  26,800  │   -11,600    │
│ Output tokens  │                   6,800  │                   5,600  │    -1,200    │
│ Total tokens   │                  45,200  │                  32,400  │   -12,800    │
│ Cost           │                  $0.8500 │                  $0.6200 │    -$0.2300  │
│ Tool calls     │                       4  │                       2  │         -2   │
│ LLM calls      │                       5  │                       3  │         -2   │
│ Nodes          │                       4  │                       3  │         -1   │
└────────────────┴──────────────────────────┴──────────────────────────┴──────────────┘
```

---

## `nodewatch report`

Generate an aggregate summary across recent runs with statistics, per-model breakdown, efficiency metrics, and ASCII bar charts.

```bash
nodewatch report --last 5                   # Summary of 5 most recent runs
nodewatch report --graph v2 --last 10       # Only v2 graph runs
nodewatch report --format json              # Raw JSON (no charts)
nodewatch report --last 5 --output report.md  # Save to file
```

Example output:

```
═══════════════════════════════════════════════════════════════
                       Summary (5 runs)
═══════════════════════════════════════════════════════════════

  Avg Cost:          $0.53    (min: $0.12, max: $1.02)
  Avg Tokens:       38,640    (min: 8,900, max: 52,900)
  Avg Latency:       8.7s     (min: 2.4s, max: 18.5s)
  Total Runs:   5
  Errors:       0/5

  Model Breakdown:
    opus-4-6   │    98,400 tokens │ $1.84
    sonnet-4-6 │    94,800 tokens │ $0.81

  Efficiency:
    Throughput:     4,440 tokens/s (avg)
    Cost/1k tok:    $0.0137
    Tool calls:     18 total (94% success)

───────────────────────────────────────────────────────────────

Cost per Run
────────────────────────────────────────────────
  a1b2c3 │ ████████████████████████ $0.85
  7c4e2a │ ██████████████████ $0.62
  f9e8d7 │ ███ $0.12
  b3d1e5 │ ██████████████████████████████ $1.02
  c8f2a9 │ ████████████████████████████ $0.95
────────────────────────────────────────────────

Tokens per Run
────────────────────────────────────────────────
  a1b2c3 │ █████████████████████████ 45,200
  7c4e2a │ ██████████████████ 32,400
  f9e8d7 │ █████ 8,900
  b3d1e5 │ ██████████████████████████████ 52,900
  c8f2a9 │ █████████████████████████████ 53,800
────────────────────────────────────────────────

Latency per Run
────────────────────────────────────────────────
  a1b2c3 │ ████████████████████ 12.3s
  7c4e2a │ █████████████ 8.1s
  f9e8d7 │ ████ 2.4s
  b3d1e5 │ ██████████████████████████████ 18.5s
  c8f2a9 │ █████████ 5.2s
────────────────────────────────────────────────
```

**Options:**
- `--graph, -g` — Filter by graph name
- `--last, -n` — Number of recent runs to include (default: 5)
- `--format, -f` — Output format: `markdown` (default) or `json`
- `--output, -o` — Save to file instead of printing

---

## `nodewatch export`

Export a single run's full trace data to JSON or Markdown for external processing or archival.

```bash
nodewatch export a1b2c3d4e5f6                   # JSON to stdout
nodewatch export a1b2c3d4e5f6 -f markdown        # Markdown to stdout
nodewatch export a1b2c3d4e5f6 -o trace.json      # Save to file
```

---

## `nodewatch delete`

Remove a stored run from the database. Prompts for confirmation unless `--force` is passed.

```bash
nodewatch delete a1b2c3d4e5f6              # Interactive confirmation
nodewatch delete a1b2c3d4e5f6 --force      # Skip confirmation
```

---

## Commands documented elsewhere

| Command | Where |
|---------|-------|
| `nodewatch dashboard` | [Dashboard](dashboard.md) |
| `nodewatch ab-compare` / `ab-init` / `ab-run` | [A/B benchmarking](benchmarking.md) |
| `nodewatch mcp` | [MCP server](mcp.md) |
| `nodewatch pricing show` | [Pricing](pricing.md) |

Run `nodewatch --help`, or `nodewatch <command> --help`, for the authoritative flag list.
