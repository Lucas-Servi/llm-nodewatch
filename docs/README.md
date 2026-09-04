# nodewatch documentation

Per-node token, cost and latency tracking for LangGraph agents.
Start with the [project README](../README.md) for the pitch and a quick start.

## Guides

| | |
|---|---|
| [Integration guide](integration.md) | Production wiring: feature-flagged imports, per-request lifecycle, error recovery, graceful shutdown |
| [CLI reference](cli.md) | Commands, flags, run-id prefixes, local vs remote mode |
| [Dashboard](dashboard.md) | The Textual TUI — six tabs, sorting, keyboard shortcuts |
| [MCP server](mcp.md) | Exposing traces to an AI assistant over stdio |
| [A/B benchmarking](benchmarking.md) | Comparing graph variants and model versions |
| [HTTP API](http-api.md) | The dispatcher endpoint, method registry, auth |
| [Remote mode](remote.md) | Driving a remote server from a laptop |
| [Pricing](pricing.md) | The bundled price table and how to override it |
| [Environment variables](env.md) | Every variable the library reads — and one it doesn't |

## Things that are easy to get wrong

Three behaviours are deliberate and load-bearing. If you are reading traces and something
looks off, check these first:

- **`input_tokens` is exclusive of cache.** LangChain reports it *inclusive*; nodewatch
  subtracts cache back out so cache can be priced at cache rates. See [pricing](pricing.md).
- **Reasoning tokens are already inside `output_tokens`.** They are not added again.
- **Tool success is derived from the output payload**, not only from raised exceptions — many
  tools return `{"error": ...}` and never raise. Recognizers are anchored, because real tool
  content contains the word "error".

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md).
