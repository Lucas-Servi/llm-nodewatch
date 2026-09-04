# MCP server

Expose your nodewatch data to an AI assistant via the [Model Context Protocol](https://modelcontextprotocol.io). The server runs locally over stdio and gives any MCP-compatible client full read access to your execution traces.

```bash
pip install "llm-nodewatch[mcp]"
```

## Launch

```bash
nodewatch mcp                          # Uses NODEWATCH_DB or default nodewatch.db
nodewatch mcp --db /path/to/runs.db    # Custom database path
```

## Configure a client

Most clients read a JSON config listing their MCP servers — commonly `.mcp.json` in the
project, or the client's own settings file. Add:

```json
{
  "mcpServers": {
    "nodewatch": {
      "command": "nodewatch",
      "args": ["mcp", "--db", "/path/to/nodewatch.db"]
    }
  }
}
```

## Available tools

| Tool | Description |
|------|-------------|
| `list_runs` | List recent runs with filters (graph, since, conversation, limit) |
| `get_run` | Full trace detail — nodes, LLM calls, tool calls, tokens, costs |
| `get_stats` | Aggregate stats: avg cost/tokens/latency, model breakdown, cache metrics |
| `list_conversations` | Conversations with total turns, tokens, cost, latency |
| `get_conversation` | All runs within a specific conversation |
| `find_expensive_runs` | Top N runs sorted by cost, tokens, or duration |
| `get_active_runs` | Currently executing runs (live mode) |
| `compare_models` | A/B-compare two model cohorts among already-recorded runs (read-only) |
| `init_ab_session` | Create a testing-session folder + editable `config.json` (filesystem only) |
| `read_ab_config` / `write_ab_config` | Read and **edit** a session's testing JSON (validated before writing) |
| `run_ab_session` | ⚠ Run an existing session's `config.json`; dump runs + reports into the folder |
| `run_model_ab` | ⚠ One-call live A/B test from two plain lists: call each model on each prompt, record + compare. **Incurs real API cost and writes runs.** Credential-gated. |

An assistant can drive the whole flow over MCP: `init_ab_session("opus48-vs-47")` → edit it with
`write_ab_config` → `run_ab_session("opus48-vs-47")`, or do it in one shot:
`run_model_ab(prompts=["What is the capital of France?", "Summarize relativity."], models=["claude-opus-4-8", "claude-opus-4-7"], session="opus48-vs-47")`
— each writes a self-contained session folder (`config.json`, `runs.db`, `ab_<model>.json`,
`results.json`). The run tools require provider credentials (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`
/ AWS) and `pip install "llm-nodewatch[ab-model]"`. If the config sets `pause_check` (or you pass
`pause_check=true` to `run_model_ab`), the run tool first returns `status: "confirmation_required"`
with a preview and runs only when called again with `confirm=true` — a user gate before any spend.

## Example queries an AI assistant can answer

- "What was the most expensive run this week?"
- "Show me the token breakdown for run abc123"
- "Compare average latency across my graphs"
- "Which conversation used the most tokens?"
- "Are there any active runs right now?"
- "A/B test opus-4-8 vs opus-4-7 on these three prompts."

---

