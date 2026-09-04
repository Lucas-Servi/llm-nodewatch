"""ASCII chart rendering for terminal output."""

from __future__ import annotations

from .stats import SummaryStats


def render_bar_chart(title: str, data: list[tuple[str, float]], fmt_fn) -> str:
    """Render a horizontal bar chart. data: list of (label, value)."""
    if not data:
        return ""
    max_val = max(v for _, v in data)
    max_bar = 25
    label_w = max(len(lbl) for lbl, _ in data)
    chart_w = label_w + 4 + max_bar + 15
    lines = [f"  [bold]{title}[/bold]", f"  {'─' * chart_w}"]
    for label, val in data:
        bar_len = int((val / max_val) * max_bar) if max_val > 0 else 0
        bar = "█" * bar_len
        lines.append(f"  {label:>{label_w}} │ {bar:<{max_bar}} {fmt_fn(val)}")
    lines.append(f"  {'─' * chart_w}")
    return "\n".join(lines)


def render_summary(stats: SummaryStats) -> str:
    """Render aggregate statistics as formatted text."""

    def _fmt_lat(ms: float) -> str:
        return f"{ms / 1000:.1f}s" if ms >= 1000 else f"{ms:.0f}ms"

    def _fmt_cost(v: float) -> str:
        return f"${v:.4f}" if v < 0.01 else f"${v:.2f}"

    lines = [
        "═" * 63,
        f"{'Summary (' + str(stats.run_count) + ' runs)':^63}",
        "═" * 63,
        "",
        f"  Avg Cost:     {_fmt_cost(stats.avg_cost):>10}    (min: {_fmt_cost(stats.min_cost)}, max: {_fmt_cost(stats.max_cost)})",
        f"  Avg Tokens:   {stats.avg_tokens:>10,}    (min: {stats.min_tokens:,}, max: {stats.max_tokens:,})",
        f"  Avg Latency:  {_fmt_lat(stats.avg_latency_ms):>10}    (min: {_fmt_lat(stats.min_latency_ms)}, max: {_fmt_lat(stats.max_latency_ms)})",
        f"  Total Runs:   {stats.run_count}",
        f"  Errors:       {stats.error_count}/{stats.run_count}",
    ]

    if stats.models:
        lines.append("")
        lines.append("  Model Breakdown:")
        max_name = max(len(m.model) for m in stats.models)
        for m in stats.models:
            lines.append(
                f"    {m.model:<{max_name}} │ {m.total_tokens:>9,} tokens │ {_fmt_cost(m.total_cost)}"
            )

    lines.append("")
    lines.append("  Efficiency:")
    if stats.throughput_tokens_per_s > 0:
        lines.append(f"    Throughput:     {stats.throughput_tokens_per_s:,.0f} tokens/s (avg)")
    if stats.cost_per_1k_tokens > 0:
        lines.append(f"    Cost/1k tok:    ${stats.cost_per_1k_tokens:.4f}")
    if stats.tool_calls_total > 0:
        pct = stats.tool_calls_success / stats.tool_calls_total * 100
        lines.append(f"    Tool calls:     {stats.tool_calls_total} total ({pct:.0f}% success)")
    else:
        lines.append("    Tool calls:     0")

    if stats.cache_read_tokens > 0 or stats.cache_creation_tokens > 0:
        lines.append("")
        lines.append("  Cache:")
        lines.append(f"    Hit rate:       {stats.cache_hit_rate:.1f}% of input tokens from cache")
        lines.append(f"    Cache reads:    {stats.cache_read_tokens:,} tokens")
        lines.append(f"    Cache writes:   {stats.cache_creation_tokens:,} tokens")

    lines.append("")
    lines.append("─" * 63)
    return "\n".join(lines)


def fmt_cost(v: float) -> str:
    return f"${v:.4f}" if v < 0.01 else f"${v:.2f}"


def fmt_tokens(v: float) -> str:
    return f"{v:,.0f}"


def fmt_latency(v: float) -> str:
    return f"{v / 1000:.1f}s" if v >= 1000 else f"{v:.0f}ms"
