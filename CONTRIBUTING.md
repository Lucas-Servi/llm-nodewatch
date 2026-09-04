# Contributing to llm-nodewatch

Thanks for your interest in improving llm-nodewatch! This guide covers how to set up a dev
environment, run the checks, and submit changes.

## Development setup

Requires Python 3.11+ (CI tests 3.11–3.13; the library is also used on 3.14).

```bash
git clone https://github.com/Lucas-Servi/llm-nodewatch.git
cd llm-nodewatch
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"   # includes fastapi, textual and mcp — everything the suite needs
```

## Running the checks

```bash
pytest                # full test suite
ruff check src tests  # lint — enforced by CI
```

Both must pass before a PR is merged.

`ruff format` is **not** enforced repo-wide: most of the tree predates it, so a blanket
reformat would bury real changes in noise. Please do run `ruff format` on files you touch.

## How it works

[`docs/internals.md`](docs/internals.md) covers the architecture and the non-obvious
decisions — node correlation, the token conventions below, the pricing lookup, and the
derived-at-capture-time signals. Worth reading before a non-trivial change.

## Code style

- Linting is handled by [ruff](https://docs.astral.sh/ruff/); config lives in
  `pyproject.toml` (`line-length` and rule selection). Don't hand-format files you touch —
  run `ruff format` on them.
- Keep public functions typed; the package ships `py.typed`.
- Match the surrounding code's naming and structure.

## Token & cost accounting (read before touching the tracker or pricing)

`tracker.py` normalizes provider usage into a single convention: `LLMCall.input_tokens` is
**exclusive** of cache tokens, with `cache_read_tokens` / `cache_creation_tokens` tracked
separately. LangChain's `usage_metadata` reports `input_tokens` **inclusive** of cache and nests
cache counts under `input_token_details` (`cache_read` / `cache_creation`). If you add a new
usage source, route it through `_usage_from_metadata` / `_usage_from_raw` so cost (`models.py`)
and cache-hit rate (`stats.py`) stay correct. Add a test under `tests/test_tracker.py`.

Two related invariants:

- **Reasoning tokens are not billed separately.** Providers already count reasoning inside
  `output_tokens`, so `output_token_details.reasoning` is deliberately not mapped to
  `thinking_tokens`. Mapping it double-bills every thinking-enabled call.
- **Pricing lookup is longest-match** (`models.prices_for_model`). Matching is by substring so
  decorated served-model ids resolve (`us.anthropic.claude-opus-4-8-v1:0` → `claude-opus-4-8`),
  and longest-match is what keeps that safe: `o3` is a substring of many ids and `gpt-5` is a
  prefix of `gpt-5.5`. Never reintroduce first-match-wins — it makes billing depend on key
  order in a JSON file users are invited to replace. Pinned in `tests/test_pricing.py`.

## Submitting changes

1. Branch off `main`.
2. Add tests for new behavior or bug fixes.
3. Ensure `pytest` and `ruff check src tests` pass.
4. Add a `## [Unreleased]` entry to `CHANGELOG.md`.
5. Open a PR with a clear description of the change and its motivation.

## Releasing

1. Bump `version` in `pyproject.toml`.
2. Move the `[Unreleased]` section in `CHANGELOG.md` to a new `## [X.Y.Z] — YYYY-MM-DD` heading.
3. Commit: `git commit -am "release: vX.Y.Z"`
4. Tag: `git tag vX.Y.Z`
5. Push: `git push origin main --tags`

The `release.yml` workflow builds the package, publishes to TestPyPI first, then to PyPI via
OIDC trusted publishing (no API token needed).

## Reporting security issues

Please do **not** open public issues for security vulnerabilities. See [SECURITY.md](SECURITY.md)
for the disclosure process.
