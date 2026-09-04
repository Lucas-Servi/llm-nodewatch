# Benchmarking

Compare multiple graph variants across the same queries:

```python
runner = nodewatch.BenchmarkRunner(storage=storage)

report = await runner.run_comparison(
    graphs={"baseline": graph_a, "experimental": graph_b},
    queries=[nodewatch.Query(text="test query", tags=["demo"])],
    state_builders={
        "baseline": build_state_a,
        "experimental": build_state_b,
    },
)

print(nodewatch.comparison_to_markdown(report))
```

## A/B model benchmarking

When the graph is fixed and you only want to compare **models** (e.g. Opus 4.8 vs 4.7) on the
same prompts, with the model isolated as the only variable:

- **`nodewatch ab-compare`** — analyze runs **already in the database**, grouping them by the
  model that actually served each run (from `llm_calls.model`, not `graph_name`):

  ```bash
  nodewatch ab-compare --expected-a opus-4-8 --expected-b opus-4-7
  ```

  It verifies each cohort served its intended model, pairs questions (by `metadata.ab_question_id`
  or normalized query text), compares only **matched node paths**, and reports per-question
  duration / token / content-filter deltas.

- **`nodewatch ab-init` + `ab-run` (testing sessions)** — *generate* the runs from a JSON config,
  then compare. A **session** is a self-contained folder: its `config.json` is the input, and the
  run dumps everything back into it — so you only ever point at the folder.

  ```bash
  nodewatch ab-init opus48-vs-47               # → testing_sessions/opus48-vs-47/config.json
  #   ...edit config.json (models + prompts)...
  nodewatch ab-run opus48-vs-47                # runs it; dumps results into the folder
  ```

  After the run the session folder holds:

  ```
  testing_sessions/opus48-vs-47/
    config.json        # input you edited
    runs.db            # the recorded runs
    ab_opus-4-8.json   # per-agent: each question's time, tokens, nodes called, final answer …
    ab_opus-4-7.json   #   (diff the two files directly)
    results.json       # the full comparison + verification + summary
  ```

  A bare name lands under `./testing_sessions/` (override the base with `NODEWATCH_SESSIONS_DIR`);
  a path is used as-is, so a session can live anywhere. `ab-init -t http` scaffolds an HTTP config;
  `ab-init --from <file>` seeds it from an existing config.

  Set `experiment.pause_check` (to `true` or a custom message) in `config.json` to require a
  **confirmation before the run spends anything** — the CLI prompts `y/N` (skip with `--yes`).

  Two **transports** (set in `config.json`):

  - **`model`** — call the model **directly** via its provider client (Anthropic / OpenAI /
    Bedrock), no server required. The most self-contained option; needs credentials in the
    environment and `pip install "llm-nodewatch[ab-model]"`. ⚠ Incurs real API cost.
  - **`http`** — POST each prompt to your agent API (field names are config-driven; `${VAR}` in
    the URL/headers/body is expanded from the environment). Use `switch_mode: "per_request"` when
    the API selects the model from the request body, or `"manual"` when the model is fixed at
    server startup — the runner pauses before each phase so you can reconfigure + restart. Point
    `config.json`'s top-level `"db"` at the SQLite file your API writes to (the server records the
    runs); `--db` overrides it.

  Prefer ad-hoc files? `nodewatch ab-run --config <file> --db runs.db --out-dir reports/` still
  works without a session folder. See **`examples/ab_config.example.json`** (HTTP) and
  **`examples/ab_config.model.example.json`** (direct model) for the full annotated schema.

