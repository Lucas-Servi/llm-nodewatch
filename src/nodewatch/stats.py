"""Data extraction and aggregation across traces."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .models import RunTrace


def _short_model(model: str) -> str:
    import re
    if not model:
        return ""
    m = re.search(r"(opus|sonnet|haiku)-(\d+-\d+)", model.lower())
    return f"{m.group(1)}-{m.group(2)}" if m else model


def node_sig(nodes: list[str]) -> str:
    """Collapse a node-visit sequence into a stable path signature.

    De-dupes consecutive/repeated nodes, strips a trailing ``_tools`` suffix (so a
    ReAct ``foo``/``foo_tools`` loop reads as one node), and joins with ASCII " -> ".
    Two runs with the same signature took the same route — the basis for comparing
    like routing with like in :func:`compute_ab_comparison`.
    """
    seen: set[str] = set()
    out: list[str] = []
    for n in nodes or []:
        base = str(n)
        if base.endswith("_tools"):
            base = base[: -len("_tools")]
        if base not in seen:
            seen.add(base)
            out.append(base)
    return " -> ".join(out)


@dataclass
class ModelStats:
    model: str
    total_tokens: int = 0
    total_cost: float = 0.0


@dataclass
class SummaryStats:
    run_count: int = 0
    avg_cost: float = 0.0
    min_cost: float = 0.0
    max_cost: float = 0.0
    avg_tokens: int = 0
    min_tokens: int = 0
    max_tokens: int = 0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    error_count: int = 0
    models: list[ModelStats] = field(default_factory=list)
    throughput_tokens_per_s: float = 0.0
    cost_per_1k_tokens: float = 0.0
    tool_calls_total: int = 0
    tool_calls_success: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_hit_rate: float = 0.0


@dataclass
class ConversationStats:
    conversation_id: str
    turn_count: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    graphs_used: list[str] = field(default_factory=list)



def compute_summary(traces: list[RunTrace]) -> SummaryStats:
    """Compute aggregate statistics from a list of traces."""
    if not traces:
        return SummaryStats()

    n = len(traces)
    costs = [t.total_cost_usd for t in traces]
    tokens = [t.total_tokens for t in traces]
    latencies = [t.total_duration_ms for t in traces]

    # Per-model breakdown
    model_tokens: dict[str, int] = defaultdict(int)
    model_cost: dict[str, float] = defaultdict(float)
    for t in traces:
        for s in t.node_spans:
            for c in s.llm_calls:
                key = _short_model(c.model) or c.model or "unknown"
                model_tokens[key] += c.total_tokens
                model_cost[key] += c.cost_usd

    models = sorted(
        [ModelStats(model=k, total_tokens=model_tokens[k], total_cost=model_cost[k]) for k in model_tokens],
        key=lambda m: m.total_tokens,
        reverse=True,
    )

    # Tool calls
    all_tool_calls = [tc for t in traces for s in t.node_spans for tc in s.tool_calls]
    tool_total = len(all_tool_calls)
    tool_success = sum(1 for tc in all_tool_calls if tc.success)

    # Cache metrics
    all_llm_calls = [c for t in traces for s in t.node_spans for c in s.llm_calls]
    cache_read = sum(c.cache_read_tokens for c in all_llm_calls)
    cache_creation = sum(c.cache_creation_tokens for c in all_llm_calls)
    total_input = sum(c.input_tokens + c.cache_read_tokens + c.cache_creation_tokens for c in all_llm_calls)
    cache_hit_rate = (cache_read / total_input * 100) if total_input > 0 else 0.0

    # Efficiency
    total_tok = sum(tokens)
    total_cost = sum(costs)
    total_dur_s = sum(latencies) / 1000

    return SummaryStats(
        run_count=n,
        avg_cost=sum(costs) / n,
        min_cost=min(costs),
        max_cost=max(costs),
        avg_tokens=sum(tokens) // n,
        min_tokens=min(tokens),
        max_tokens=max(tokens),
        avg_latency_ms=sum(latencies) / n,
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        error_count=sum(1 for t in traces if t.error),
        models=models,
        throughput_tokens_per_s=total_tok / total_dur_s if total_dur_s > 0 else 0.0,
        cost_per_1k_tokens=total_cost / total_tok * 1000 if total_tok > 0 else 0.0,
        tool_calls_total=tool_total,
        tool_calls_success=tool_success,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
        cache_hit_rate=cache_hit_rate,
    )


def extract_chart_data(traces: list[RunTrace]) -> dict[str, list[tuple[str, float]]]:
    """Extract per-run data points for charting. Returns dict with keys: cost, tokens, latency."""
    labels = [f"{t.graph_name}#{i}" for i, t in enumerate(traces, 1)]
    return {
        "cost": [(lbl, t.total_cost_usd) for lbl, t in zip(labels, traces)],
        "tokens": [(lbl, float(t.total_tokens)) for lbl, t in zip(labels, traces)],
        "latency": [(lbl, t.total_duration_ms) for lbl, t in zip(labels, traces)],
    }


# ── A/B model comparison ───────────────────────────────────────────────────────
# Compare the same query suite served by two different models (e.g. Opus 4.8 vs 4.7),
# isolating the model as the only variable. Cohorts are identified by the model that
# ACTUALLY SERVED the run (derived from llm_calls.model), NOT by graph_name — in a
# multi-model agent setup both versions often share one graph_name (e.g. "v2"), so
# grouping by graph_name would merge the two cohorts. Questions are paired across cohorts
# and only MATCHED node paths are compared (a 2-expert run vs a 4-expert run is not a fair pair).


@dataclass
class CohortVerification:
    """Did a cohort actually run the model it was supposed to?"""
    cohort: str                 # display label, e.g. "opus-4-8"
    expected_model: str         # substring the caller expected, e.g. "opus-4-8"
    served_models: list[str]    # sorted set of short opus models actually observed
    ok: bool                    # any served model contains expected_model


@dataclass
class QuestionDelta:
    """Paired A→B deltas for one question, over the matched-node-path subset."""
    question_key: str
    matched_node_sig: str | None    # shared node signature, or None when none matched
    n_a: int
    n_b: int
    mean_duration_ms_a: float
    mean_duration_ms_b: float
    duration_delta_pct: float
    mean_tokens_a: float
    mean_tokens_b: float
    tokens_delta_pct: float
    mean_filtered_per_call_a: float
    mean_filtered_per_call_b: float


@dataclass
class ABComparison:
    cohort_a: str
    cohort_b: str
    verification: list[CohortVerification]      # one per cohort
    verified_ok: bool                           # all(v.ok) — caller decides whether to abort
    per_question: list[QuestionDelta]
    overall_duration_delta_pct: float
    overall_tokens_delta_pct: float
    overall_filtered_per_call_a: float
    overall_filtered_per_call_b: float


def _served_models(trace: RunTrace) -> set[str]:
    """Short model ids that served this run (the cohort-identity marker).

    Collects every distinct model across the run's llm_calls (normalized via
    ``_short_model``). No provider pre-filter — the comparator verifies cohorts by
    substring against the caller's ``expected`` id, so any vendor (opus, gpt, mistral, …)
    works. An A/B comparison isolates the served model, so this is what distinguishes
    cohort A from cohort B.
    """
    out: set[str] = set()
    for s in trace.node_spans:
        for c in s.llm_calls:
            if c.model:
                out.add(_short_model(c.model))
    return out


def _question_key(trace: RunTrace, field: str) -> str:
    """Cross-cohort pairing key: explicit metadata id, else normalized query text.

    The same question asked in both cohorts has identical query text, so the text is a
    reliable universal key. Callers who want deterministic ids (the phased runner) set
    ``metadata[field]``. The unreliable ``conversation_id`` column is deliberately NOT
    used (it is frequently empty; see the bridge notes).
    """
    md = trace.metadata or {}
    val = md.get(field)
    if val:
        return str(val)
    return " ".join((trace.query or "").split()).lower()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pct_delta(a: float, b: float) -> float:
    return (b - a) / a * 100 if a else 0.0


def _filtered_per_call(trace: RunTrace) -> float:
    return trace.total_filtered / trace.total_llm_calls if trace.total_llm_calls else 0.0


def compute_ab_comparison(
    traces_a: list[RunTrace],
    traces_b: list[RunTrace],
    expected_a: str,
    expected_b: str,
    cohort_a: str | None = None,
    cohort_b: str | None = None,
    question_key_field: str = "ab_question_id",
) -> ABComparison:
    """Compare two cohorts of runs (same questions, different served model).

    ``traces_a``/``traces_b`` are the runs served by model A / model B; ``expected_a``/
    ``expected_b`` are short-model substrings (e.g. ``"opus-4-8"``) used both to label
    the cohorts and to verify each actually ran the intended model. Returns an
    :class:`ABComparison`; it never raises on a model mismatch — it sets
    ``verified_ok=False`` and the caller decides whether to trust/abort.
    """
    cohort_a = cohort_a or expected_a
    cohort_b = cohort_b or expected_b

    # 1. Verify each cohort served its intended model.
    def _verify(traces: list[RunTrace], cohort: str, expected: str) -> CohortVerification:
        served: set[str] = set()
        for t in traces:
            served |= _served_models(t)
        ok = any(expected in m for m in served)
        return CohortVerification(cohort=cohort, expected_model=expected,
                                  served_models=sorted(served), ok=ok)

    verification = [
        _verify(traces_a, cohort_a, expected_a),
        _verify(traces_b, cohort_b, expected_b),
    ]
    verified_ok = all(v.ok for v in verification)

    # 2. Group by question key.
    def _by_q(traces: list[RunTrace]) -> dict[str, list[RunTrace]]:
        groups: dict[str, list[RunTrace]] = {}
        for t in traces:
            groups.setdefault(_question_key(t, question_key_field), []).append(t)
        return groups

    qa, qb = _by_q(traces_a), _by_q(traces_b)

    # 3. Per-question paired deltas over the matched node-path subset.
    per_question: list[QuestionDelta] = []
    for q in sorted(set(qa) & set(qb)):
        ra_all, rb_all = qa[q], qb[q]
        sigs_a = {node_sig(t.nodes_visited) for t in ra_all}
        sigs_b = {node_sig(t.nodes_visited) for t in rb_all}
        shared = sigs_a & sigs_b
        ra = [t for t in ra_all if node_sig(t.nodes_visited) in shared] or ra_all
        rb = [t for t in rb_all if node_sig(t.nodes_visited) in shared] or rb_all
        # Pick the most common shared signature for display (or None when no match).
        matched = None
        if shared:
            matched = max(shared, key=lambda sig: sum(
                1 for t in ra + rb if node_sig(t.nodes_visited) == sig))

        da, db = _mean([t.total_duration_ms for t in ra]), _mean([t.total_duration_ms for t in rb])
        ta, tb = _mean([t.total_tokens for t in ra]), _mean([t.total_tokens for t in rb])
        fa, fb = _mean([_filtered_per_call(t) for t in ra]), _mean([_filtered_per_call(t) for t in rb])
        per_question.append(QuestionDelta(
            question_key=q,
            matched_node_sig=matched,
            n_a=len(ra), n_b=len(rb),
            mean_duration_ms_a=da, mean_duration_ms_b=db, duration_delta_pct=_pct_delta(da, db),
            mean_tokens_a=ta, mean_tokens_b=tb, tokens_delta_pct=_pct_delta(ta, tb),
            mean_filtered_per_call_a=fa, mean_filtered_per_call_b=fb,
        ))

    # 4. Overall (cohort-wide, all runs).
    od_a, od_b = _mean([t.total_duration_ms for t in traces_a]), _mean([t.total_duration_ms for t in traces_b])
    ot_a, ot_b = _mean([t.total_tokens for t in traces_a]), _mean([t.total_tokens for t in traces_b])
    of_a, of_b = _mean([_filtered_per_call(t) for t in traces_a]), _mean([_filtered_per_call(t) for t in traces_b])

    return ABComparison(
        cohort_a=cohort_a, cohort_b=cohort_b,
        verification=verification, verified_ok=verified_ok,
        per_question=per_question,
        overall_duration_delta_pct=_pct_delta(od_a, od_b),
        overall_tokens_delta_pct=_pct_delta(ot_a, ot_b),
        overall_filtered_per_call_a=of_a,
        overall_filtered_per_call_b=of_b,
    )


def compute_conversation_stats(traces: list[RunTrace]) -> list[ConversationStats]:
    """Group traces by conversation_id and compute per-conversation aggregates."""
    groups: dict[str, list[RunTrace]] = {}
    for t in traces:
        key = t.conversation_id or t.run_id
        groups.setdefault(key, []).append(t)

    results = []
    for conv_id, conv_traces in groups.items():
        total_tokens = sum(t.total_tokens for t in conv_traces)
        total_cost = sum(t.total_cost_usd for t in conv_traces)
        avg_latency = sum(t.total_duration_ms for t in conv_traces) / len(conv_traces)
        graphs = sorted(set(t.graph_name for t in conv_traces))
        results.append(ConversationStats(
            conversation_id=conv_id,
            turn_count=len(conv_traces),
            total_tokens=total_tokens,
            total_cost=total_cost,
            avg_latency_ms=avg_latency,
            graphs_used=graphs,
        ))

    results.sort(key=lambda c: c.total_cost, reverse=True)
    return results


