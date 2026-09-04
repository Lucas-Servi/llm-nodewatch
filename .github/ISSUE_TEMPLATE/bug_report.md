---
name: Bug report
about: Something in nodewatch doesn't work as documented
title: ""
labels: bug
---

**What happened**

A clear description of the incorrect behavior.

**What you expected**

What the docs, or reasonable inference, led you to expect instead.

**Minimal repro**

```python
# The smallest GraphTracker / storage / CLI invocation that reproduces it.
```

**Environment**

- `nodewatch` version: `pip show llm-nodewatch`
- Python version:
- Extras installed (`server` / `client` / `mcp` / `ab-model`):
- Local or remote mode (`NODEWATCH_URL` set?):

**Trace, if relevant**

If the bug is about incorrect tokens/cost/success on a specific run, the output of
`nodewatch inspect <run_id>` or `nodewatch export <run_id>` (redact anything sensitive —
prompts and tool outputs are not scrubbed by nodewatch itself).
