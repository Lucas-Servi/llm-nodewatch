# Pricing Configuration

Default pricing ships inside the package at `nodewatch/data/pricing.json` and covers
Anthropic, OpenAI, Google (Gemini), Mistral, DeepSeek, Llama-on-Groq, Cohere, xAI (Grok),
and MiniMax. Prices are **approximate public list prices** (per million tokens) and will drift
over time — treat them as estimates and override them whenever exact accounting matters.

Each entry maps a model prefix to `[input, output]` or `[input, output, cache_read, cache_creation]`:

```json
{
  "claude-opus-4-8": [5.0, 25.0, 0.5, 6.25],
  "gemini-2.5-pro": [1.25, 10.0],
  "gpt-5.4": [2.5, 15.0, 1.25, 2.5]
}
```

Models are prefix-matched against the full (lowercased) model ID (e.g., `"claude-opus-4-6"`
matches `us.anthropic.claude-opus-4-6-v1`). If cache prices are omitted, defaults are
`cache_read = input * 0.1`, `cache_creation = input * 1.25`.

A model with no matching entry is reported at **$0** and logs a one-time warning — add it to a
custom pricing file rather than trusting the zero. Inspect what's currently loaded with:

```bash
nodewatch pricing show
```

To use a custom pricing file (takes precedence over the bundled default):

```bash
export NODEWATCH_PRICING=/path/to/my-pricing.json
```

