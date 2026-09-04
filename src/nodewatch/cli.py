"""Typer CLI for nodewatch."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .charts import fmt_cost, fmt_latency, fmt_tokens, render_bar_chart, render_summary
from .client import RemoteClient, get_remote_url
from .models import trace_matches_conversation
from .reporter import trace_to_json, trace_to_markdown, traces_to_json
from .stats import compute_ab_comparison, compute_summary, extract_chart_data
from .storage.sqlite import SQLiteStorage

app = typer.Typer(name="nodewatch", help="LangGraph agent observability: track tokens, cost, and latency per node.")
console = Console()

DEFAULT_DB = os.getenv("NODEWATCH_DB", "nodewatch.db")

_force_local: bool = False

_FORBIDDEN_PREFIXES = ("/etc", "/usr", "/bin", "/boot", "/sys", "/proc")


def _safe_write(output: str, content: str, force: bool = False) -> None:
    """Write content to a file with basic path safety checks."""
    path = Path(output).expanduser().resolve()
    resolved = str(path)
    for prefix in _FORBIDDEN_PREFIXES:
        if resolved.startswith(prefix):
            console.print(f"[red]Refusing to write to system path: {resolved}[/red]")
            raise typer.Exit(1)
    if path.exists() and not force:
        console.print(f"[red]File already exists: {path}[/red]")
        console.print("[dim]Use --force to overwrite.[/dim]")
        raise typer.Exit(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    console.print(f"[green]Written to {path}[/green]")


@app.callback()
def main(
    local: bool = typer.Option(False, "--local", "-L", help="Force local database mode (ignore NODEWATCH_URL)"),
):
    """LangGraph agent benchmarking and observability tool."""
    global _force_local
    _force_local = local


def _get_storage(db: str):
    """Return a RemoteClient if NODEWATCH_URL is set, otherwise local SQLiteStorage."""
    if not _force_local:
        remote_url = get_remote_url()
        if remote_url:
            try:
                return RemoteClient(base_url=remote_url)
            except Exception as e:
                console.print(f"[red]Cannot connect to remote API:[/red] {e}")
                console.print("[dim]Use --local (-L) to use a local database, or check your network/NODEWATCH_URL.[/dim]")
                raise typer.Exit(1)
    return SQLiteStorage(db)


# Prefix/conversation resolution scans recent runs client-side rather than server-side, so
# it works against an already-deployed remote API. The remote caps a page at 200 rows.
_SCAN_PAGE = 200
_SCAN_PAGES = 5


def _scan_recent_runs(storage, pages: int = _SCAN_PAGES, page: int = _SCAN_PAGE) -> list:
    """Page through recent runs, newest first, up to pages * page rows."""
    traces: list = []
    for n in range(pages):
        try:
            batch = storage.list_runs(limit=page, offset=n * page)
        except Exception:
            break
        if not batch:
            break
        traces.extend(batch)
        if len(batch) < page:
            break
    return traces


def _resolve_trace(storage, ident: str, pages: int = _SCAN_PAGES, page: int = _SCAN_PAGE):
    """Load a trace by full run id, or by a unique run-id prefix.

    On failure, print an actionable hint (ambiguous prefix, or "that's a conversation id")
    and exit 1 — never return None.
    """
    trace = storage.load(ident)
    if trace:
        return trace

    scanned = _scan_recent_runs(storage, pages=pages, page=page)
    prefix_matches = [t for t in scanned if t.run_id.startswith(ident)]
    if len(prefix_matches) == 1:
        return storage.load(prefix_matches[0].run_id) or prefix_matches[0]

    if len(prefix_matches) > 1:
        console.print(f"[red]Run id '{ident}' is ambiguous — {len(prefix_matches)} runs match:[/red]")
        for t in prefix_matches[:10]:
            console.print(
                f"  [cyan]{t.run_id}[/cyan]  {t.graph_name}  "
                f"{t.timestamp.strftime('%Y-%m-%d %H:%M')}  [dim]{t.query[:40]}[/dim]"
            )
        if len(prefix_matches) > 10:
            console.print(f"  [dim]… and {len(prefix_matches) - 10} more[/dim]")
        console.print("[dim]Use a longer prefix or the full run id.[/dim]")
        raise typer.Exit(1)

    conv_matches = [t for t in scanned if trace_matches_conversation(t, ident)]
    if conv_matches:
        run_ids = ", ".join(t.run_id for t in conv_matches[:10])
        console.print(f"[yellow]'{ident}' is a conversation ID, not a run ID[/yellow] — {len(conv_matches)} runs:")
        console.print(f"  [cyan]{run_ids}[/cyan]" + (" …" if len(conv_matches) > 10 else ""))
        console.print(f"[dim]Try: nodewatch list-runs -c {ident}[/dim]")
        console.print("[dim]     nodewatch inspect <run_id>   (a unique prefix works too)[/dim]")
        raise typer.Exit(1)

    console.print(f"[red]Run '{ident}' not found.[/red]")
    if len(scanned) >= pages * page:
        console.print(f"[dim]Searched the {len(scanned)} most recent runs.[/dim]")
    raise typer.Exit(1)


pricing_app = typer.Typer(name="pricing", help="Inspect the model pricing table in effect.")
app.add_typer(pricing_app)


@pricing_app.command("show")
def pricing_show():
    """Print the loaded model pricing table and the file it was loaded from."""
    from .models import PRICING_PER_MTOK, pricing_source_path

    source = pricing_source_path()
    console.print(f"[bold]Pricing source:[/bold] {source}")
    if os.getenv("NODEWATCH_PRICING"):
        console.print("[dim](from NODEWATCH_PRICING)[/dim]")
    else:
        console.print("[dim](bundled default — override with NODEWATCH_PRICING)[/dim]")

    table = Table(title="Model pricing (USD per million tokens)")
    table.add_column("Model prefix", style="cyan", no_wrap=True)
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")
    table.add_column("Cache read", justify="right")
    table.add_column("Cache create", justify="right")
    for model_prefix, prices in sorted(PRICING_PER_MTOK.items()):
        inp, out = prices[0], prices[1]
        cache_read = prices[2] if len(prices) > 2 else inp * 0.1
        cache_create = prices[3] if len(prices) > 3 else inp * 1.25
        table.add_row(
            model_prefix,
            f"{inp:g}",
            f"{out:g}",
            f"{cache_read:g}",
            f"{cache_create:g}",
        )
    console.print(table)
    console.print(f"[dim]{len(PRICING_PER_MTOK)} models priced. Unlisted models report $0.[/dim]")


@app.command()
def list_runs(
    graph: Optional[str] = typer.Option(None, "--graph", "-g", help="Show only this graph (e.g. v1, v2)"),
    conversation: Optional[str] = typer.Option(None, "--conversation", "-c", help="Show only this conversation/thread ID"),
    since: Optional[str] = typer.Option(None, "--since", "-s", help="Only runs after this date (e.g. 2026-05-14)"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max rows to display"),
    offset: int = typer.Option(0, "--offset", help="Skip this many runs (for pagination)"),
    db: str = typer.Option(DEFAULT_DB, "--db", help="Path to SQLite database (local mode)"),
):
    """Table of recent runs: ID, graph, query, tokens, cost, latency, date. Filterable by graph/conversation/date."""
    storage = _get_storage(db)
    traces = storage.list_runs(graph_name=graph, since=since, conversation_id=conversation, limit=limit, offset=offset)

    if not traces:
        console.print("[dim]No runs found.[/dim]")
        return

    table = Table(title="Benchmark Runs")
    # no_wrap: the run id must stay copyable — rich otherwise shrinks it to "3c4242b…",
    # leaving no full id on screen to paste into `inspect`/`export`/`delete`.
    table.add_column("Run ID", style="cyan", no_wrap=True)
    table.add_column("Graph", style="green")
    table.add_column("Conv", style="dim", max_width=12)
    table.add_column("Query", max_width=30)
    table.add_column("Duration", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Date")

    for t in traces:
        dur = f"{t.total_duration_ms / 1000:.1f}s" if t.total_duration_ms >= 1000 else f"{t.total_duration_ms:.0f}ms"
        conv = t.conversation_id[:12] if t.conversation_id else "-"
        table.add_row(
            t.run_id,
            t.graph_name,
            conv,
            t.query[:40],
            dur,
            f"{t.total_tokens:,}",
            f"${t.total_cost_usd:.3f}",
            t.timestamp.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)
    storage.close()


@app.command()
def inspect(
    run_id: str = typer.Argument(help="Run ID — full or unique prefix (e.g. da24c8)"),
    db: str = typer.Option(DEFAULT_DB, "--db", help="Path to SQLite database (local mode)"),
):
    """Detailed view of one run: summary metrics, per-node table (model, in/out tokens, cost, duration, tools), and token split."""
    storage = _get_storage(db)
    trace = _resolve_trace(storage, run_id)
    _render_inspect(trace)
    storage.close()


def _short_model_name(model: str) -> str:
    """Shorten model for display: 'us.anthropic.claude-opus-4-6-v1' → 'opus-4-6'."""
    import re
    if not model:
        return "-"
    m = re.search(r"(opus|sonnet|haiku)-(\d+-\d+)", model.lower())
    return f"{m.group(1)}-{m.group(2)}" if m else model


def _render_inspect(trace) -> None:
    dur = f"{trace.total_duration_ms / 1000:.1f}s" if trace.total_duration_ms >= 1000 else f"{trace.total_duration_ms:.0f}ms"

    # Header
    console.print()
    header = Text()
    header.append(f"  {trace.run_id}", style="cyan bold")
    header.append("  |  ", style="dim")
    header.append(f"{trace.graph_name}", style="green bold")
    header.append("  |  ", style="dim")
    header.append(f"{trace.timestamp.strftime('%Y-%m-%d %H:%M')}", style="dim")
    console.print(Panel(header, title="Run", border_style="blue"))

    # Query
    query_text = trace.query[:200] + ("..." if len(trace.query) > 200 else "")
    console.print(f"  [dim]Query:[/dim] {query_text}")
    console.print()

    # Summary metrics
    metrics_table = Table(box=None, pad_edge=False, show_edge=False, show_header=False)
    metrics_table.add_column("", width=14)
    metrics_table.add_column("", width=14)
    metrics_table.add_column("", width=14)
    metrics_table.add_column("", width=14)
    metrics_table.add_column("", width=14)
    metrics_table.add_row(
        f"[dim]Duration[/dim]\n[bold]{dur}[/bold]",
        f"[dim]Tokens[/dim]\n[bold]{trace.total_tokens:,}[/bold]",
        f"[dim]Cost[/dim]\n[bold]${trace.total_cost_usd:.2f}[/bold]",
        f"[dim]LLM calls[/dim]\n[bold]{trace.total_llm_calls}[/bold]",
        f"[dim]Tool calls[/dim]\n[bold]{trace.total_tool_calls}[/bold]",
    )
    console.print(metrics_table)
    console.print()

    if not trace.node_spans:
        console.print("[dim]  No node data available.[/dim]")
        return

    # Node breakdown table
    table = Table(title="Node Breakdown", box=None, pad_edge=False, show_edge=False)
    table.add_column("Node", style="bold", width=22)
    table.add_column("Model", style="dim", width=12)
    table.add_column("Input", justify="right", width=10)
    table.add_column("Output", justify="right", width=8)
    table.add_column("Cost", justify="right", width=7)
    table.add_column("Duration", justify="right", width=9)
    table.add_column("Loops", justify="right", width=5)
    table.add_column("Tools", justify="right", width=5)
    table.add_column("LLMs", justify="right", width=5)
    table.add_column("Filt", justify="right", width=5)

    for span in trace.node_spans:
        model = _short_model_name(span.llm_calls[0].model if span.llm_calls else "")
        inp = f"{span.total_input_tokens:,}" if span.total_input_tokens > 0 else "-"
        out = f"{span.total_output_tokens:,}" if span.total_output_tokens > 0 else "-"
        cost = f"${span.total_cost_usd:.2f}" if span.total_cost_usd > 0 else "-"
        dur_s = span.duration_ms
        dur_str = f"{dur_s/1000:.1f}s" if dur_s >= 1000 else f"{dur_s:.0f}ms" if dur_s > 0 else "-"
        loops = str(span.iterations) if span.iterations > 0 else "-"
        tools = str(len(span.tool_calls)) if span.tool_calls else "-"
        llms = str(len(span.llm_calls)) if span.llm_calls else "-"
        filt = f"[red]{span.filtered_count}[/red]" if span.filtered_count else "-"

        table.add_row(span.node_name, model, inp, out, cost, dur_str, loops, tools, llms, filt)

    console.print(table)
    console.print()

    # Token split bar (input vs output)
    total_in = trace.total_input_tokens
    total_out = trace.total_output_tokens
    if total_in + total_out > 0:
        bar_width = 40
        in_pct = total_in / (total_in + total_out)
        in_bar = int(in_pct * bar_width)
        out_bar = bar_width - in_bar
        console.print(f"  [dim]Token split:[/dim]  [blue]{'█' * in_bar}[/blue][yellow]{'█' * out_bar}[/yellow]  [blue]{total_in:,} in[/blue] / [yellow]{total_out:,} out[/yellow] ({in_pct*100:.0f}%/{(1-in_pct)*100:.0f}%)")

    # Error
    if trace.error:
        console.print(f"\n  [red bold]Error:[/red bold] {trace.error}")
    console.print()


@app.command()
def compare(
    run_a: str = typer.Argument(help="First run ID (baseline) — full or unique prefix"),
    run_b: str = typer.Argument(help="Second run ID (new) — full or unique prefix"),
    db: str = typer.Option(DEFAULT_DB, "--db", help="Path to SQLite database (local mode)"),
):
    """Quick A/B of two runs: shows absolute values + delta for duration, tokens, cost, tools, nodes."""
    storage = _get_storage(db)
    trace_a = _resolve_trace(storage, run_a)
    trace_b = _resolve_trace(storage, run_b)

    table = Table(title=f"Comparison: {trace_a.run_id} vs {trace_b.run_id}")
    table.add_column("Metric")
    table.add_column(f"{trace_a.graph_name} ({trace_a.run_id})", justify="right")
    table.add_column(f"{trace_b.graph_name} ({trace_b.run_id})", justify="right")
    table.add_column("Δ", justify="right")

    metrics = [
        ("Duration", trace_a.total_duration_ms, trace_b.total_duration_ms, "ms"),
        ("Input tokens", trace_a.total_input_tokens, trace_b.total_input_tokens, ""),
        ("Output tokens", trace_a.total_output_tokens, trace_b.total_output_tokens, ""),
        ("Total tokens", trace_a.total_tokens, trace_b.total_tokens, ""),
        ("Cost", trace_a.total_cost_usd, trace_b.total_cost_usd, "$"),
        ("Tool calls", trace_a.total_tool_calls, trace_b.total_tool_calls, ""),
        ("LLM calls", trace_a.total_llm_calls, trace_b.total_llm_calls, ""),
        ("Filtered", trace_a.total_filtered, trace_b.total_filtered, ""),
        ("Nodes", len(trace_a.node_spans), len(trace_b.node_spans), ""),
    ]

    for name, a, b, unit in metrics:
        diff = b - a
        sign = "+" if diff > 0 else ""
        if unit == "$":
            table.add_row(name, f"${a:.4f}", f"${b:.4f}", f"{sign}${diff:.4f}")
        elif unit == "ms":
            table.add_row(name, f"{a:.0f}ms", f"{b:.0f}ms", f"{sign}{diff:.0f}ms")
        else:
            table.add_row(name, f"{a:,.0f}", f"{b:,.0f}", f"{sign}{diff:,.0f}")

    console.print(table)
    storage.close()


def _ab_dict_from_local(storage, expected_a: str, expected_b: str, since, limit) -> dict:
    """Compute an A/B comparison locally and return it in the same dict shape the API emits."""
    traces = storage.list_runs(since=since, limit=min(limit, 1000))
    a, b = [], []
    for t in traces:
        served = " ".join(c.model.lower() for s in t.node_spans for c in s.llm_calls if c.model)
        if expected_a in served:
            a.append(t)
        elif expected_b in served:
            b.append(t)
    comp = compute_ab_comparison(a, b, expected_a, expected_b)
    return {
        "cohort_a": comp.cohort_a, "cohort_b": comp.cohort_b, "verified_ok": comp.verified_ok,
        "verification": [vars(v) for v in comp.verification],
        "per_question": [vars(q) for q in comp.per_question],
        "overall_duration_delta_pct": comp.overall_duration_delta_pct,
        "overall_tokens_delta_pct": comp.overall_tokens_delta_pct,
        "overall_filtered_per_call_a": comp.overall_filtered_per_call_a,
        "overall_filtered_per_call_b": comp.overall_filtered_per_call_b,
    }


@app.command(name="ab-compare")
def ab_compare(
    expected_a: str = typer.Option("opus-4-8", "--expected-a", "-a", help="Served-model substring for cohort A (baseline)"),
    expected_b: str = typer.Option("opus-4-7", "--expected-b", "-b", help="Served-model substring for cohort B (new)"),
    since: Optional[str] = typer.Option(None, "--since", "-s", help="Only runs after this date (e.g. 2026-06-01)"),
    last: int = typer.Option(200, "--last", "-n", help="Max runs to scan into the two cohorts"),
    db: str = typer.Option(DEFAULT_DB, "--db", help="Path to SQLite database (local mode)"),
):
    """A/B compare two served-model cohorts (e.g. opus-4-8 vs opus-4-7).

    Groups recent runs by the model that ACTUALLY SERVED them (from llm_calls, not
    graph_name), verifies each cohort ran its intended model, and reports per-question
    matched-node-path deltas for duration, tokens, and content-filter rate.
    """
    storage = _get_storage(db)
    if isinstance(storage, RemoteClient):
        data = storage.ab_compare(expected_a=expected_a, expected_b=expected_b, since=since, limit=last)
    else:
        data = _ab_dict_from_local(storage, expected_a, expected_b, since, last)
    storage.close()
    _render_ab(data)


def _render_ab(data: dict) -> None:
    """Render an A/B comparison dict (from ab_compare or ab-run) as Rich output."""
    ca, cb = data["cohort_a"], data["cohort_b"]
    console.print()
    console.print(f"[bold]A/B comparison:[/bold] [cyan]{ca}[/cyan] vs [magenta]{cb}[/magenta]")

    # Verification
    console.print("\n[bold]Served-model verification[/bold]")
    for v in data["verification"]:
        ok = v["ok"]
        mark = "[green]OK[/green]" if ok else "[red]MISMATCH[/red]"
        served = ", ".join(v["served_models"]) or "(none)"
        console.print(f"  {v['cohort']}: expected [bold]{v['expected_model']}[/bold], served {served} — {mark}")
    if not data["verified_ok"]:
        console.print("[red]  ⚠ A cohort did not serve its intended model — deltas below are NOT trustworthy.[/red]")

    # Per-question deltas
    pq = data["per_question"]
    if not pq:
        console.print("\n[dim]No shared questions across both cohorts (need the same query text or ab_question_id in both).[/dim]")
        return

    table = Table(title="Per-question deltas (matched node path)")
    table.add_column("Question", max_width=34)
    table.add_column("Match", justify="center")
    table.add_column(f"{ca} dur", justify="right")
    table.add_column(f"{cb} dur", justify="right")
    table.add_column("Δt%", justify="right")
    table.add_column(f"{ca} tok", justify="right")
    table.add_column(f"{cb} tok", justify="right")
    table.add_column("Δtok%", justify="right")
    table.add_column("A filt", justify="right")
    table.add_column("B filt", justify="right")
    for q in pq:
        match = "yes" if q["matched_node_sig"] else "[yellow]all[/yellow]"
        dt = q["duration_delta_pct"]
        tk = q["tokens_delta_pct"]
        table.add_row(
            q["question_key"][:34], match,
            f"{q['mean_duration_ms_a']/1000:.1f}s", f"{q['mean_duration_ms_b']/1000:.1f}s",
            f"{dt:+.0f}%", f"{q['mean_tokens_a']:,.0f}", f"{q['mean_tokens_b']:,.0f}",
            f"{tk:+.0f}%", f"{q['mean_filtered_per_call_a']:.2f}", f"{q['mean_filtered_per_call_b']:.2f}",
        )
    console.print()
    console.print(table)

    console.print(
        f"\n[bold]Overall:[/bold] duration {data['overall_duration_delta_pct']:+.0f}%, "
        f"tokens {data['overall_tokens_delta_pct']:+.0f}%, "
        f"filtered/call {data['overall_filtered_per_call_a']:.2f} ({ca}) "
        f"vs {data['overall_filtered_per_call_b']:.2f} ({cb})"
    )
    console.print()


def _render_model_summaries(summaries: list[dict]) -> None:
    """Render per-model aggregates (shown when there is no pairwise comparison)."""
    table = Table(title="Per-model summary")
    table.add_column("Model")
    table.add_column("Expected")
    table.add_column("Served")
    table.add_column("OK/runs", justify="right")
    table.add_column("Mean tok", justify="right")
    table.add_column("Mean dur", justify="right")
    for s in summaries:
        served = ", ".join(s["served_models"]) or "(none)"
        ok = any(s["expected"] in m for m in s["served_models"])
        served_disp = served if ok else f"[red]{served}[/red]"
        table.add_row(
            s["model"], s["expected"], served_disp,
            f"{s['n_ok']}/{s['n_runs']}",
            f"{s['mean_tokens']:,.0f}", f"{s['mean_duration_ms']/1000:.1f}s",
        )
    console.print()
    console.print(table)
    console.print()


@app.command(name="ab-init")
def ab_init(
    session: str = typer.Argument(help="Session name (under testing_sessions/) or a folder path"),
    transport: str = typer.Option("model", "--transport", "-t", help="Template transport: 'model' (direct) or 'http' (agent API)"),
    from_file: Optional[str] = typer.Option(None, "--from", help="Seed config.json from an existing config file instead of the template"),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config.json"),
):
    """Create a new A/B testing session folder with a config.json to edit.

    A session is a self-contained folder: its config.json is the input, and `nodewatch ab-run
    <session>` dumps the recorded DB, per-agent JSONs, and results.json back into it. A bare name
    lands under ./testing_sessions/<name> (override the base with NODEWATCH_SESSIONS_DIR); a path
    is used as-is.
    """
    from .abrun import init_session, resolve_session_dir

    if transport not in ("model", "http"):
        console.print("[red]--transport must be 'model' or 'http'.[/red]")
        raise typer.Exit(1)

    session_dir = resolve_session_dir(session)
    try:
        config_path = init_session(session_dir, transport=transport, from_file=from_file, force=force)
    except FileExistsError:
        console.print(f"[red]config.json already exists in {session_dir}[/red] — use --force to overwrite.")
        raise typer.Exit(1)
    except (OSError, ValueError) as e:
        console.print(f"[red]Could not create session:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[green]Created session:[/green] {session_dir}")
    console.print(f"[dim]Edit the config:[/dim] {config_path}")
    console.print(f"[dim]Then run it:[/dim] nodewatch ab-run {session}")


@app.command(name="ab-run")
def ab_run(
    session: Optional[str] = typer.Argument(None, help="Session name (under testing_sessions/) or folder; loads <session>/config.json and dumps results there"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Explicit config file (instead of a session folder)"),
    db: Optional[str] = typer.Option(None, "--db", help="SQLite database to record/read runs (overrides the session DB and the config 'db')"),
    out_dir: Optional[str] = typer.Option(None, "--out-dir", "-o", help="Where to write ab_<model>.json + results.json (defaults to the session folder)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the pause_check confirmation and manual-switch pauses (assume ready)"),
):
    """Run an A/B model benchmark from a session folder or a config file.

    Sends each prompt to the configured transport (an HTTP agent API, or a model called
    directly) across every model under test, records the runs, and prints the served-model
    verification + per-question deltas. In session mode (`ab-run <session>`) the config is loaded
    from <session>/config.json and the recorded DB, per-agent ab_<model>.json files, and
    results.json are all written into the session folder. See `nodewatch ab-init`.
    """
    from pathlib import Path as _Path

    from .abrun import CONFIG_FILENAME, load_ab_config, resolve_session_dir, run_ab_config

    # Resolve where the config comes from and where artifacts go.
    session_dir = None
    if session:
        session_dir = resolve_session_dir(session)
        config_path = session_dir / CONFIG_FILENAME
        if not config_path.exists():
            console.print(f"[red]No {CONFIG_FILENAME} in {session_dir}.[/red]")
            console.print(f"[dim]Create it first:[/dim] nodewatch ab-init {session}")
            raise typer.Exit(1)
        if out_dir is None:
            out_dir = str(session_dir)
    elif config:
        config_path = _Path(config)
    else:
        console.print("[red]Provide a session name/folder, or --config <file>.[/red]")
        console.print("[dim]e.g.[/dim] nodewatch ab-run my-test   [dim]or[/dim]   nodewatch ab-run -c config.json")
        raise typer.Exit(1)

    try:
        cfg = load_ab_config(config_path)
    except (OSError, ValueError) as e:
        console.print(f"[red]Could not load config:[/red] {e}")
        raise typer.Exit(1)

    # Confirmation gate (config `pause_check`) — a user action before anything runs.
    if cfg.pause_check and not yes:
        from .abrun import preview_ab_config
        pv = preview_ab_config(cfg)
        console.print(f"\n[yellow]{cfg.pause_check}[/yellow]")
        console.print(
            f"[dim]Will run {pv['total_runs']} request(s): {pv['n_models']} model(s) × "
            f"{pv['n_questions']} question(s) × {pv['reps']} rep(s) — {pv['note']}.[/dim]"
        )
        try:
            answer = input("Proceed? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            console.print("[red]Aborted.[/red] (pass --yes to skip this confirmation)")
            raise typer.Exit(1)

    # DB resolution: --db > config 'db' > session runs.db > DEFAULT_DB.
    if db:
        db_path = db
    elif cfg.db:
        db_path = cfg.db
    elif session_dir is not None:
        db_path = str(session_dir / "runs.db")
    else:
        db_path = DEFAULT_DB

    storage = SQLiteStorage(db_path)

    def pause_hook(phase) -> None:
        if yes:
            return
        console.print(
            f"\n[yellow]Manual switch:[/yellow] set the server to serve "
            f"[bold]{phase.expected_model}[/bold] for phase [bold]{phase.name}[/bold], "
            f"restart it, then press Enter to run this phase…"
        )
        try:
            input()
        except EOFError:
            pass

    try:
        result = run_ab_config(cfg, storage, pause_hook=pause_hook, out_dir=out_dir)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    finally:
        storage.close()

    console.print(f"\n[dim]{result['summary']}[/dim]")
    if result["comparison"] is not None:
        _render_ab(result["comparison"])
    else:
        console.print(
            "\n[dim]Pairwise comparison needs exactly 2 models; showing a per-model summary.[/dim]"
        )
        _render_model_summaries(result["model_summaries"])

    console.print(f"\n[green]Recorded runs to:[/green] {db_path}")
    for path in result.get("report_paths", []):
        console.print(f"[green]Wrote per-agent report:[/green] {path}")
    if result.get("results_path"):
        console.print(f"[green]Wrote results:[/green] {result['results_path']}")


@app.command()
def export(
    run_id: str = typer.Argument(help="Run ID to export — full or unique prefix"),
    format: str = typer.Option("json", "--format", "-f", help="json (machine-readable) or markdown (human-readable)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write to file (prints to stdout if omitted)"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing output file"),
    db: str = typer.Option(DEFAULT_DB, "--db", help="Path to SQLite database (local mode)"),
):
    """Export full trace data (all nodes, LLM calls, tool calls) to JSON or Markdown."""
    storage = _get_storage(db)
    trace = _resolve_trace(storage, run_id)

    if format == "json":
        content = json.dumps(trace_to_json(trace), indent=2)
    else:
        content = trace_to_markdown(trace)

    if output:
        _safe_write(output, content, force=force)
    else:
        print(content)

    storage.close()


@app.command()
def report(
    graph: Optional[str] = typer.Option(None, "--graph", "-g", help="Only include this graph (e.g. v2)"),
    conversation: Optional[str] = typer.Option(None, "--conversation", "-c", help="Only include this conversation ID"),
    last: int = typer.Option(5, "--last", "-n", help="Number of recent runs to summarize"),
    format: str = typer.Option("markdown", "--format", "-f", help="Output format: markdown (with charts) or json (raw data)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save report to file instead of printing"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing output file"),
    db: str = typer.Option(DEFAULT_DB, "--db", help="Path to SQLite database (local mode)"),
):
    """Summary report: avg cost/tokens/latency, model breakdown, cache stats, and per-run bar charts."""
    storage = _get_storage(db)
    is_remote = isinstance(storage, RemoteClient)
    traces = storage.list_runs(graph_name=graph, conversation_id=conversation, limit=last)

    if not traces:
        console.print("[dim]No runs found.[/dim]")
        raise typer.Exit(0)

    if format == "json":
        if is_remote:
            import json as _json
            content = _json.dumps([trace_to_json(t) for t in traces], indent=2)
        else:
            content = traces_to_json(traces)
        if output:
            _safe_write(output, content, force=force)
        else:
            print(content)
        storage.close()
        return

    # Summary stats — use pre-computed remote stats or compute locally
    if is_remote:
        from .stats import ModelStats, SummaryStats
        raw = storage.report(graph_name=graph, limit=last)
        stats = SummaryStats(
            run_count=raw.get("run_count", 0),
            avg_cost=raw.get("avg_cost", 0.0),
            min_cost=raw.get("min_cost", 0.0),
            max_cost=raw.get("max_cost", 0.0),
            avg_tokens=raw.get("avg_tokens", 0),
            min_tokens=raw.get("min_tokens", 0),
            max_tokens=raw.get("max_tokens", 0),
            avg_latency_ms=raw.get("avg_latency_ms", 0.0),
            min_latency_ms=raw.get("min_latency_ms", 0.0),
            max_latency_ms=raw.get("max_latency_ms", 0.0),
            error_count=raw.get("error_count", 0),
            models=[ModelStats(model=m["model"], total_tokens=m["total_tokens"], total_cost=m["total_cost"]) for m in raw.get("models", [])],
            throughput_tokens_per_s=raw.get("throughput_tokens_per_s", 0.0),
            cost_per_1k_tokens=raw.get("cost_per_1k_tokens", 0.0),
            tool_calls_total=raw.get("tool_calls_total", 0),
            tool_calls_success=raw.get("tool_calls_success", 0),
        )
    else:
        stats = compute_summary(traces)

    summary_text = render_summary(stats)
    console.print()
    console.print(summary_text)
    console.print()

    # Per-run table
    table = Table(title="Runs", box=None, pad_edge=False, show_edge=False)
    table.add_column("#", style="dim", justify="right", width=3)
    table.add_column("Run ID", style="cyan", width=8)
    table.add_column("Graph", style="green", width=6)
    table.add_column("Query", max_width=36)
    table.add_column("Tokens", justify="right", style="yellow")
    table.add_column("Cost", justify="right", style="bold")
    table.add_column("Latency", justify="right")
    table.add_column("Filt", justify="right")
    table.add_column("Date", style="dim")

    for i, t in enumerate(traces, 1):
        dur = f"{t.total_duration_ms / 1000:.1f}s" if t.total_duration_ms >= 1000 else f"{t.total_duration_ms:.0f}ms"
        query = t.query[:36] + ("..." if len(t.query) > 36 else "")
        filt = f"[red]{t.total_filtered}[/red]" if t.total_filtered else "-"
        table.add_row(
            str(i),
            t.run_id[:8],
            t.graph_name,
            query,
            f"{t.total_tokens:,}",
            f"${t.total_cost_usd:.2f}",
            dur,
            filt,
            t.timestamp.strftime("%m-%d %H:%M"),
        )

    console.print(table)
    console.print()

    # Bar charts
    chart_data = extract_chart_data(traces)
    cost_chart = render_bar_chart("Cost per Run", chart_data["cost"], fmt_cost)
    tokens_chart = render_bar_chart("Tokens per Run", chart_data["tokens"], fmt_tokens)
    latency_chart = render_bar_chart("Latency per Run", chart_data["latency"], fmt_latency)

    console.print(cost_chart)
    console.print()
    console.print(tokens_chart)
    console.print()
    console.print(latency_chart)
    console.print()

    if output:
        lines = [summary_text, "", str(table), "", cost_chart, "", tokens_chart, "", latency_chart]
        _safe_write(output, "\n".join(lines), force=force)

    storage.close()




@app.command()
def delete(
    run_id: str = typer.Argument(help="Run ID to remove — full or unique prefix"),
    db: str = typer.Option(DEFAULT_DB, "--db", help="Path to SQLite database (local mode)"),
    force: bool = typer.Option(False, "--force", "-f", help="Delete without asking for confirmation"),
):
    """Delete a run and all its node/LLM/tool data. Asks for confirmation unless --force."""
    storage = _get_storage(db)

    # Always resolve first: confirming (and deleting) against the resolved full run id is
    # what stops an abbreviated prefix from removing a run the user never saw.
    trace = _resolve_trace(storage, run_id)
    resolved_id = trace.run_id

    if not force:
        confirm = typer.confirm(f"Delete run {resolved_id} ({trace.graph_name}, {trace.query[:30]})?")
        if not confirm:
            raise typer.Exit(0)

    if storage.delete(resolved_id):
        console.print(f"[green]Deleted {resolved_id}[/green]")
    else:
        console.print(f"[red]Run '{resolved_id}' not found.[/red]")

    storage.close()




@app.command()
def dashboard(
    db: str = typer.Option(DEFAULT_DB, "--db", help="Path to SQLite database (local mode)"),
):
    """Interactive TUI dashboard: browse runs, conversations, inspect traces, view stats. Requires: pip install 'llm-nodewatch[client]'"""
    try:
        from .dashboard import run_dashboard
    except ImportError:
        console.print("[red]Textual is required for the dashboard.[/red]")
        console.print("[dim]Install with: pip install 'llm-nodewatch\\[client]'[/dim]")
        raise typer.Exit(1)
    storage = _get_storage(db)
    run_dashboard(storage)


@app.command(name="mcp")
def mcp_command(
    db: str = typer.Option(DEFAULT_DB, "--db", help="Path to SQLite database"),
):
    """Start a local MCP server (stdio) exposing nodewatch data to AI assistants."""
    os.environ.setdefault("NODEWATCH_DB", db)
    try:
        from .mcp_server import run_mcp_server
    except ImportError:
        console.print("[red]MCP SDK is required.[/red]")
        console.print("[dim]Install with: pip install 'mcp>=1.0'[/dim]")
        raise typer.Exit(1)
    run_mcp_server()


if __name__ == "__main__":
    app()
