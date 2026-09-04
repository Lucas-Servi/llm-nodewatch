"""Report generation: Markdown tables, JSON export."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .models import RunTrace
from .runner import ComparisonReport
from .stats import ABComparison


def _fmt_tokens(inp: int, out: int) -> str:
    return f"{inp:,}/{out:,}"


def _fmt_duration(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.1f}s"
    return f"{ms:.0f}ms"


def _fmt_cost(usd: float) -> str:
    if usd < 0.01:
        return f"${usd:.4f}"
    return f"${usd:.2f}"


def trace_to_markdown(trace: RunTrace, include_nodes: bool = True) -> str:
    """Generate a markdown summary for a single run."""
    lines = [
        f"## Run: {trace.run_id} | {trace.graph_name} | {trace.timestamp.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**Query**: {trace.query[:100]}{'...' if len(trace.query) > 100 else ''}",
        f"**Duration**: {_fmt_duration(trace.total_duration_ms)}",
        f"**Tokens**: {_fmt_tokens(trace.total_input_tokens, trace.total_output_tokens)} (total: {trace.total_tokens:,})",
        f"**Cost**: {_fmt_cost(trace.total_cost_usd)}",
        f"**Tool calls**: {trace.total_tool_calls}",
        f"**LLM calls**: {trace.total_llm_calls}",
        f"**Nodes visited**: {', '.join(trace.nodes_visited)}",
    ]

    if trace.error:
        lines.append(f"**Error**: {trace.error}")

    if trace.total_filtered:
        lines.append(f"**Content-filtered**: {trace.total_filtered}")

    if include_nodes and trace.node_spans:
        lines += ["", "### Node Breakdown", ""]
        lines.append("| Node | Model | Tokens (in/out) | Duration | Loops | Tools | Filt | Cost |")
        lines.append("|------|-------|-----------------|----------|-------|-------|------|------|")
        for span in trace.node_spans:
            model = span.llm_calls[0].model if span.llm_calls else "-"
            lines.append(
                f"| {span.node_name} | {model} | "
                f"{_fmt_tokens(span.total_input_tokens, span.total_output_tokens)} | "
                f"{_fmt_duration(span.duration_ms)} | {span.iterations} | "
                f"{len(span.tool_calls)} | {span.filtered_count} | {_fmt_cost(span.total_cost_usd)} |"
            )

    return "\n".join(lines)


def comparison_to_markdown(report: ComparisonReport) -> str:
    """Generate a full comparison report in Markdown."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [f"## Benchmark Comparison | {now}", ""]

    # Summary table
    lines.append("| Graph | Query | Duration | Tokens (in/out) | Tools | Cost | Error |")
    lines.append("|-------|-------|----------|-----------------|-------|------|-------|")

    for result in report.results:
        query_short = result.query.text[:40] + ("..." if len(result.query.text) > 40 else "")
        for graph_name, trace in result.traces.items():
            err = "Yes" if trace.error else "-"
            lines.append(
                f"| {graph_name} | {query_short} | "
                f"{_fmt_duration(trace.total_duration_ms)} | "
                f"{_fmt_tokens(trace.total_input_tokens, trace.total_output_tokens)} | "
                f"{trace.total_tool_calls} | {_fmt_cost(trace.total_cost_usd)} | {err} |"
            )

    # Per-graph node breakdowns
    for result in report.results:
        for graph_name, trace in result.traces.items():
            if trace.node_spans:
                lines += [
                    "",
                    f"### Node Breakdown: {graph_name} — \"{result.query.text[:50]}\"",
                    "",
                    "| Node | Model | Tokens (in/out) | Duration | Loops | Tools | Filt |",
                    "|------|-------|-----------------|----------|-------|-------|------|",
                ]
                for span in trace.node_spans:
                    model = span.llm_calls[0].model if span.llm_calls else "-"
                    lines.append(
                        f"| {span.node_name} | {model} | "
                        f"{_fmt_tokens(span.total_input_tokens, span.total_output_tokens)} | "
                        f"{_fmt_duration(span.duration_ms)} | {span.iterations} | "
                        f"{len(span.tool_calls)} | {span.filtered_count} |"
                    )

    return "\n".join(lines)


def trace_to_json(trace: RunTrace) -> dict:
    """Serialize a RunTrace to a JSON-friendly dict."""
    return {
        "run_id": trace.run_id,
        "graph_name": trace.graph_name,
        "query": trace.query,
        "conversation_id": trace.conversation_id,
        "timestamp": trace.timestamp.isoformat(),
        "total_duration_ms": trace.total_duration_ms,
        "total_tokens": trace.total_tokens,
        "total_input_tokens": trace.total_input_tokens,
        "total_output_tokens": trace.total_output_tokens,
        "total_cost_usd": trace.total_cost_usd,
        "total_tool_calls": trace.total_tool_calls,
        "total_llm_calls": trace.total_llm_calls,
        "total_filtered": trace.total_filtered,
        "error": trace.error,
        "metadata": trace.metadata,
        "node_spans": [
            {
                "node_name": s.node_name,
                "node_type": s.node_type,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "duration_ms": s.duration_ms,
                "iterations": s.iterations,
                "total_input_tokens": s.total_input_tokens,
                "total_output_tokens": s.total_output_tokens,
                "total_cost_usd": s.total_cost_usd,
                "filtered_count": s.filtered_count,
                "llm_calls": [
                    {
                        "model": c.model,
                        "provider": c.provider,
                        "input_tokens": c.input_tokens,
                        "output_tokens": c.output_tokens,
                        "thinking_tokens": c.thinking_tokens,
                        "cache_read_tokens": c.cache_read_tokens,
                        "duration_ms": c.duration_ms,
                        "stop_reason": c.stop_reason,
                        "content_filtered": c.content_filtered,
                        "error": c.error,
                    }
                    for c in s.llm_calls
                ],
                "tool_calls": [
                    {
                        "tool_name": t.tool_name,
                        "duration_ms": t.duration_ms,
                        "success": t.success,
                        "error": t.error,
                        "input": t.input,
                        "output_preview": t.output_preview,
                        "output_size": t.output_size,
                    }
                    for t in s.tool_calls
                ],
            }
            for s in trace.node_spans
        ],
    }


def traces_to_json(traces: list[RunTrace]) -> str:
    """Serialize multiple traces to JSON string."""
    return json.dumps([trace_to_json(t) for t in traces], indent=2)


def ab_comparison_to_markdown(comp: ABComparison) -> str:
    """Render an A/B model comparison as Markdown (verification + per-question deltas)."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [f"## A/B Model Comparison: {comp.cohort_a} vs {comp.cohort_b} | {now}", ""]

    # Served-model verification
    lines += ["### Served-model verification", ""]
    for v in comp.verification:
        status = "OK" if v.ok else "MISMATCH"
        served = ", ".join(v.served_models) or "(none)"
        lines.append(f"- **{v.cohort}**: expected `{v.expected_model}`, served `{served}` — **{status}**")
    if not comp.verified_ok:
        lines.append("")
        lines.append("> :warning: A cohort did not serve its intended model — deltas below are not trustworthy.")
    lines.append("")

    # Per-question deltas (matched node path)
    lines += [
        "### Per-question deltas (matched node path)",
        "",
        f"| Question | Path match | {comp.cohort_a} dur | {comp.cohort_b} dur | Δt% | "
        f"{comp.cohort_a} tok | {comp.cohort_b} tok | Δtok% | A filt/call | B filt/call |",
        "|----------|-----------|---------|---------|-----|---------|---------|-------|-------------|-------------|",
    ]
    for q in comp.per_question:
        match = "yes" if q.matched_node_sig else "NO (all reps)"
        lines.append(
            f"| {q.question_key[:40]} | {match} | "
            f"{_fmt_duration(q.mean_duration_ms_a)} | {_fmt_duration(q.mean_duration_ms_b)} | "
            f"{q.duration_delta_pct:+.0f}% | "
            f"{q.mean_tokens_a:,.0f} | {q.mean_tokens_b:,.0f} | {q.tokens_delta_pct:+.0f}% | "
            f"{q.mean_filtered_per_call_a:.2f} | {q.mean_filtered_per_call_b:.2f} |"
        )

    lines += [
        "",
        "### Overall",
        "",
        f"- **Duration**: {comp.overall_duration_delta_pct:+.0f}% ({comp.cohort_a} → {comp.cohort_b})",
        f"- **Tokens**: {comp.overall_tokens_delta_pct:+.0f}%",
        f"- **Filtered/call**: {comp.overall_filtered_per_call_a:.2f} ({comp.cohort_a}) "
        f"vs {comp.overall_filtered_per_call_b:.2f} ({comp.cohort_b})",
    ]
    return "\n".join(lines)
