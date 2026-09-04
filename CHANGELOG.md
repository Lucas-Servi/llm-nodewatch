# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-09-03

First public release.

### Fixed
- **Cache-token accounting.** The tracker read cache usage from LangChain's `usage_metadata`
  using raw Anthropic key names (`cache_read_input_tokens` / `cache_creation_input_tokens`),
  which LangChain does not emit there — it nests them under `input_token_details`
  (`cache_read` / `cache_creation`). As a result every cache column was recorded as `0`,
  `cache_hit_rate` was always 0%, and because `usage_metadata.input_tokens` is *inclusive* of
  cache while cost/stats assume it is *exclusive*, cost was over-estimated for cache-heavy runs.
  Token extraction is now normalized through `_usage_from_metadata` / `_usage_from_raw`, so
  cache reads/creation are captured and billed at their discounted rates.
- **Pricing lookup could bill a model at an unrelated model's rates.** The lookup took the
  first key that matched as either a prefix *or* an unanchored substring, iterating in
  JSON-file order — so a short key such as `o3` matched any id containing those two
  characters, and `gpt-5` could win over `gpt-5.5`. Correctness depended on line order in a
  file users are explicitly invited to replace via `NODEWATCH_PRICING`. Lookup is now
  longest-match, exposed as `models.prices_for_model()`.
- **Tool-call success is now derived from the output payload.** `ToolCall.success` was
  effectively a constant `True`: it was set to `False` only in `on_tool_error`, which never
  fires for tools that return an error *payload* instead of raising (the norm for MCP- and
  HTTP-backed tools). Recognizers are anchored, so legitimate content containing the word
  "error" is not misread as a failure.
  ⚠️ This shifts historical metrics: rows captured before this change all read as successful,
  so tool-success rates step down at the upgrade boundary. The old 100% was the bug.
- **Content-filter detection on Bedrock.** `ChatBedrockConverse` exposes the stop reason only
  at `response_metadata["stopReason"]`, never in `generation_info`, so filtered calls were
  recorded with an empty `stop_reason` and were invisible.
- Warmup-run failures in `BenchmarkRunner` are logged instead of silently swallowed.

### Added
- **A/B benchmarking**: `nodewatch ab-compare`, `ab-init` and `ab-run`, plus the
  `ABExperiment` runner. Cohorts are identified by the model that *actually served* each run
  (derived from `llm_calls.model`, not `graph_name`, which multi-model setups share), only
  matched node paths are compared, and runs are resumable.
- **MCP server** (`nodewatch mcp`) — 13 tools for querying traces and driving A/B sessions
  from an AI assistant. Requires `mcp>=2,<3`.
- **Dashboard**: sortable Runs and Conversations tables (click a header, or `s`/`S`), and a
  Logs tab backed by `NODEWATCH_LOG_PATH`.
- Run-id arguments accept a **unique prefix**; a conversation id passed by mistake is
  recognised and answered with that conversation's run ids.
- Optional Bearer auth for the HTTP API via `NODEWATCH_API_TOKEN`, and a `get_logs` method.
- `nodewatch pricing show` to inspect the price table actually in effect.
- Coverage for the async callback wrappers (`aon_*`), which LangChain uses for `ainvoke` —
  the documented primary code path, previously untested.

### Changed
- `LLMCall.total_tokens` and `NodeSpan.total_input_tokens` now include cache read/creation
  tokens as input volume, keeping `total_input + total_output == total_tokens` coherent.
- Documentation split out of the README into [`docs/`](docs/README.md).

### Removed
- `nodewatch.config` (an unused TOML config loader) and `nodewatch.evaluators` (unused, and
  domain-specific to the codebase this was extracted from). Neither was exported or imported
  anywhere.

## [0.1.0] — 2026-06-09

- Initial internal release: LangGraph callback-based per-node token/cost/latency tracing,
  SQLite storage, Typer CLI, optional FastAPI HTTP API, and chart-ready data endpoints.
  Never published to PyPI.

[Unreleased]: https://github.com/Lucas-Servi/llm-nodewatch/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Lucas-Servi/llm-nodewatch/releases/tag/v0.2.0
[0.1.0]: https://github.com/Lucas-Servi/llm-nodewatch/releases/tag/v0.1.0
