from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_UNPRICED_WARNED: set[str] = set()


def pricing_source_path() -> Path:
    """Resolve the pricing file in effect: NODEWATCH_PRICING if set, else the bundled default."""
    custom_path = os.getenv("NODEWATCH_PRICING")
    if custom_path:
        return Path(custom_path)
    return Path(__file__).parent / "data" / "pricing.json"


def _load_pricing() -> dict[str, tuple[float, ...]]:
    """Load pricing from JSON. Uses NODEWATCH_PRICING env var if set, otherwise bundled default.

    Each entry is (input, output) or (input, output, cache_read, cache_creation).
    """
    with open(pricing_source_path()) as f:
        raw = json.load(f)

    return {k: tuple(v) for k, v in raw.items() if not k.startswith("_")}


PRICING_PER_MTOK: dict[str, tuple[float, float]] = _load_pricing()


def prices_for_model(model: str | None) -> tuple[float, ...] | None:
    """Look up a model's prices, matching the LONGEST key that applies.

    Matching is by substring, because served model ids carry vendor and region
    decoration the table does not repeat (``us.anthropic.claude-opus-4-8-v1:0``
    must find the ``claude-opus-4-8`` row).

    Longest-match is what makes that safe. Any shorter key that also appears in
    the id is a false positive waiting to happen: ``o3`` is a substring of a
    great many model names, and ``gpt-5`` is a prefix of ``gpt-5.5``. Picking
    the first key that matched made billing depend on dict iteration order,
    i.e. on line order in a JSON file that users are explicitly invited to
    replace.
    """
    if not model:
        return None
    key = model.lower()
    best: str | None = None
    for candidate in PRICING_PER_MTOK:
        if candidate in key and (best is None or len(candidate) > len(best)):
            best = candidate
    return PRICING_PER_MTOK[best] if best is not None else None


# Stop reasons that mean a safety filter blocked the model output. Bedrock uses
# "content_filtered", the AWS-external/public Anthropic API uses "refusal".
_FILTERED_STOP_REASONS = frozenset({"content_filtered", "refusal"})


def is_filtered_stop(stop_reason: str | None) -> bool:
    """True if a stop reason indicates the response was blocked by a content filter."""
    return (stop_reason or "").strip().lower() in _FILTERED_STOP_REASONS


# Tools that return an error *payload* instead of raising never trigger
# on_tool_error, so their failures used to be recorded as successes. These
# patterns are deliberately ANCHORED to the start of the output: a substring
# search for "error" matches legitimate tool *content* — any catalog of ECC
# hardware is full of parts named "Error-Correcting ...", and technical corpora
# generally contain the word — which would mark honest results as failures, the
# same defect class inverted.
_ERROR_PREFIX_RE = re.compile(
    r"""^\s*(?:
        \[[A-Za-z][A-Za-z ]{0,30}Error\]        # [Web Search Error] ...
        # "Error: ..." / "Error loading models: ...". The (?!-) is load-bearing:
        # without it this matches content beginning "Error-Correcting ..." (a
        # real product name), which would flag honest results as tool failures.
      | Errors?(?!-)\s*[:\s]
      | An\s+error\s+occurred\b                  # common generic-catch wording
      | Traceback\s+\(most\s+recent\s+call\s+last\):
    )""",
    re.VERBOSE | re.IGNORECASE,
)

_ERROR_PREVIEW_CHARS = 200


def _error_from_payload(output: str) -> str | None:
    """Extract an error message from a JSON error payload, if that's what this is.

    Handles the shapes MCP tools and HTTP-backed tools commonly return:
      * ``{"error": "..."}``             — MCP tools, typed ``ErrorResult`` models
      * ``{"errorCode": 1, ...}``        — APIs where 0 means SUCCESS
      * ``{"result": {"errorCode": 1}}`` — the same, wrapped
    """
    stripped = output.strip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        payload = json.loads(stripped)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None

    for candidate in (payload, payload.get("result")):
        if not isinstance(candidate, dict):
            continue
        err = candidate.get("error")
        if isinstance(err, str) and err.strip():
            return err.strip()[:_ERROR_PREVIEW_CHARS]
        code = candidate.get("errorCode")
        # errorCode 0 is the SUCCESS value in this convention.
        if isinstance(code, int) and not isinstance(code, bool) and code != 0:
            message = candidate.get("errorMessage") or candidate.get("message") or ""
            return (f"errorCode {code}: {message}".strip(": ").strip())[
                :_ERROR_PREVIEW_CHARS
            ]
    return None


def classify_tool_error(output: str | None) -> str | None:
    """Return an error message if a tool's *output* reports a failure, else None.

    Many tools return an error payload rather than raising, so ``on_tool_error``
    never fires for them and ``on_tool_end`` would otherwise record
    ``success=True``. This classifies the output at capture time — a derived
    signal, exactly like :func:`is_filtered_stop`, so no stored column or schema
    migration is needed.

    Pass the FULL output string, not a truncated preview: the JSON shapes can
    only be parsed whole.
    """
    if not output:
        return None
    text = str(output)
    match = _ERROR_PREFIX_RE.match(text)
    if match:
        return text.strip()[:_ERROR_PREVIEW_CHARS]
    return _error_from_payload(text)


@dataclass
class LLMCall:
    node_name: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    duration_ms: float = 0.0
    stop_reason: str = ""
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        # input_tokens is non-cached input only; cache reads/creation are input volume too.
        return self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens + self.output_tokens

    @property
    def content_filtered(self) -> bool:
        # Derived from stop_reason — no stored column needed (see is_filtered_stop).
        return is_filtered_stop(self.stop_reason)

    @property
    def cost_usd(self) -> float:
        prices = prices_for_model(self.model)
        if prices is not None:
            inp_price, out_price = prices[0], prices[1]
            cache_read_price = prices[2] if len(prices) > 2 else inp_price * 0.1
            cache_creation_price = prices[3] if len(prices) > 3 else inp_price * 1.25
            input_cost = self.input_tokens * inp_price
            cache_read_cost = self.cache_read_tokens * cache_read_price
            cache_creation_cost = self.cache_creation_tokens * cache_creation_price
            output_cost = self.output_tokens * out_price
            thinking_cost = self.thinking_tokens * out_price
            return (input_cost + cache_read_cost + cache_creation_cost + output_cost + thinking_cost) / 1_000_000
        if self.model and self.total_tokens > 0 and self.model not in _UNPRICED_WARNED:
            _UNPRICED_WARNED.add(self.model)
            logger.warning(
                "no pricing for model %r; cost reported as $0 — add it to pricing.json "
                "or point NODEWATCH_PRICING at a file that includes it",
                self.model,
            )
        return 0.0


@dataclass
class ToolCall:
    node_name: str
    tool_name: str
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None
    input: str | None = None
    output_preview: str | None = None
    output_size: int = 0


@dataclass
class NodeSpan:
    node_name: str
    node_type: str = "agent"
    start_time: float = 0.0
    end_time: float = 0.0
    llm_calls: list[LLMCall] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    iterations: int = 0

    @property
    def duration_ms(self) -> float:
        if self.end_time <= 0.0 or self.end_time < self.start_time:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    @property
    def total_input_tokens(self) -> int:
        # Full input volume incl. cached tokens, so total_input + total_output == total_tokens.
        return sum(c.input_tokens + c.cache_read_tokens + c.cache_creation_tokens for c in self.llm_calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.llm_calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.llm_calls)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.llm_calls)

    @property
    def filtered_count(self) -> int:
        # How many LLM calls in this node were blocked by a content filter.
        return sum(1 for c in self.llm_calls if c.content_filtered)


@dataclass
class RunTrace:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    graph_name: str = ""
    query: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_duration_ms: float = 0.0
    node_spans: list[NodeSpan] = field(default_factory=list)
    final_response: str = ""
    error: str | None = None
    metadata: dict = field(default_factory=dict)
    conversation_id: str = ""
    _tokens_override: int | None = field(default=None, repr=False)
    _cost_override: float | None = field(default=None, repr=False)

    @property
    def total_input_tokens(self) -> int:
        return sum(s.total_input_tokens for s in self.node_spans)

    @property
    def total_output_tokens(self) -> int:
        return sum(s.total_output_tokens for s in self.node_spans)

    @property
    def total_tokens(self) -> int:
        if self._tokens_override is not None:
            return self._tokens_override
        return sum(s.total_tokens for s in self.node_spans)

    @property
    def total_cost_usd(self) -> float:
        if self._cost_override is not None:
            return self._cost_override
        return sum(s.total_cost_usd for s in self.node_spans)

    @property
    def total_tool_calls(self) -> int:
        return sum(len(s.tool_calls) for s in self.node_spans)

    @property
    def total_llm_calls(self) -> int:
        return sum(len(s.llm_calls) for s in self.node_spans)

    @property
    def total_filtered(self) -> int:
        # Total content-filter events observed across all nodes in this run.
        return sum(s.filtered_count for s in self.node_spans)

    @property
    def nodes_visited(self) -> list[str]:
        return [s.node_name for s in self.node_spans]


def trace_matches_conversation(trace: RunTrace, conversation_id: str) -> bool:
    """True if a trace belongs to a conversation, tolerating the unreliable column.

    The ``runs.conversation_id`` column is frequently empty (the real id often lives in
    ``metadata["conversation_id"]``), so check both.
    """
    md = trace.metadata or {}
    return conversation_id in (trace.conversation_id, md.get("conversation_id"))
