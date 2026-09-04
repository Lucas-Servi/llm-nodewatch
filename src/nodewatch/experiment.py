"""Transport-agnostic A/B model benchmarking.

Runs the same question suite across two (or more) phases — each phase typically a
different served model (e.g. Opus 4.8 then 4.7) — and compares them with the model
isolated as the only variable. The caller supplies a ``query_fn`` that issues ONE
query over whatever transport (a live HTTP agent API, an in-process graph, a direct
model call, …) and returns once the run has been written to nodewatch storage; this
module owns the phase/rep iteration, conversation-id tagging, resumability, reading runs
back, served-model self-verification, and the matched-node-path comparison.

It deliberately imports no HTTP/transport specifics — those live in the caller. Example::

    import urllib.request, json
    from nodewatch import SQLiteStorage, ABExperiment, ExperimentSpec, Phase, ExperimentQuestion

    def query_fn(text, conversation_id, **kwargs):
        body = json.dumps({"conversation_id": conversation_id, "user_prompt": text,
                           **kwargs}).encode()
        req = urllib.request.Request("http://localhost:8000/v1/query", data=body,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=600).read()

    spec = ExperimentSpec(
        phases=[Phase("m48", "opus-4-8"), Phase("m47", "opus-4-7")],
        questions=[ExperimentQuestion("q1", "What is the capital of France?")],
        reps=3,
    )
    result = ABExperiment(SQLiteStorage("nodewatch.db"), spec).run(query_fn)
    print(result.comparison.verified_ok)

When the served model is fixed at server startup, the operator points each phase at the
right model (e.g. editing the server config + restarting) BEFORE running that phase — the
runner verifies the served model matched ``Phase.expected_model`` and surfaces any mismatch
through :attr:`ExperimentResult.comparison`. (For transports that select the model per
request, the caller just injects it into the request body.)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .models import RunTrace, trace_matches_conversation
from .stats import ABComparison, _served_models, compute_ab_comparison, node_sig
from .storage.base import StorageBackend

logger = logging.getLogger(__name__)

# query_fn(question_text, conversation_id, **question_kwargs) -> None
QueryFn = Callable[..., Any]


@dataclass
class Phase:
    """One arm of the experiment — typically one served model."""
    name: str               # short label, e.g. "m48"
    expected_model: str     # short-model substring expected to serve it, e.g. "opus-4-8"


@dataclass
class ExperimentQuestion:
    id: str                 # stable id, e.g. "q_capital"
    text: str               # the prompt
    kwargs: dict = field(default_factory=dict)   # passed through to query_fn (extra request fields)


@dataclass
class ExperimentSpec:
    phases: list[Phase]
    questions: list[ExperimentQuestion]
    reps: int = 1
    conv_id_template: str = "ab_{phase}_{qid}_r{rep}"
    settle_seconds: float = 3.0     # wait after a query before reading its run back


@dataclass
class PhaseRunRecord:
    phase: str
    question_id: str
    rep: int
    conversation_id: str
    run_id: str | None
    ok: bool
    served_models: list[str]
    node_sig: str
    total_tokens: int
    duration_ms: float
    filtered: int
    api_error: str | None = None


@dataclass
class ExperimentResult:
    spec: ExperimentSpec
    records: list[PhaseRunRecord]
    comparison: ABComparison | None = None
    # run_id -> RunTrace for every resolved run (so callers can re-render without re-querying).
    _traces_by_run: dict[str, RunTrace] = field(default_factory=dict, repr=False)

    def records_for_phase(self, phase_name: str) -> list[PhaseRunRecord]:
        return [r for r in self.records if r.phase == phase_name]

    def traces_for_phase(self, phase_name: str) -> list[RunTrace]:
        return [
            self._traces_by_run[r.run_id]
            for r in self.records
            if r.phase == phase_name and r.ok and r.run_id and r.run_id in self._traces_by_run
        ]


class ABExperiment:
    """Drives an A/B benchmark and assembles a comparison from nodewatch storage."""

    def __init__(self, storage: StorageBackend, spec: ExperimentSpec):
        self.storage = storage
        self.spec = spec

    # ── run-resolution / resumability ─────────────────────────────────────────

    def _resolve_run_by_conv(self, conversation_id: str) -> RunTrace | None:
        """Find the run for a conversation id, tolerating the unreliable column.

        The ``runs.conversation_id`` column is frequently empty (the real id often lives
        in ``metadata.conversation_id``), so try the scoped query first and fall back to
        scanning recent runs matched on either the column or the metadata key.
        """
        try:
            scoped = self.storage.list_runs(conversation_id=conversation_id, limit=5)
        except TypeError:
            scoped = self.storage.list_runs(limit=5)  # backend without the kwarg
        except Exception:
            scoped = []
        for t in scoped or []:
            if self._matches_conv(t, conversation_id):
                return t
        # Fallback: scan a window of recent runs and match on metadata.
        try:
            recent = self.storage.list_runs(limit=200)
        except Exception:
            recent = []
        for t in recent or []:
            if self._matches_conv(t, conversation_id):
                return t
        return None

    @staticmethod
    def _matches_conv(trace: RunTrace, conversation_id: str) -> bool:
        return trace_matches_conversation(trace, conversation_id)

    @staticmethod
    def _is_good(trace: RunTrace | None) -> bool:
        return bool(trace and trace.run_id and not trace.error and trace.total_tokens > 0)

    def _record_from_trace(self, phase: Phase, q: ExperimentQuestion, rep: int,
                           conv: str, trace: RunTrace | None,
                           api_error: str | None) -> PhaseRunRecord:
        if trace is None:
            return PhaseRunRecord(phase.name, q.id, rep, conv, None, False,
                                  [], "", 0, 0.0, 0, api_error)
        return PhaseRunRecord(
            phase=phase.name, question_id=q.id, rep=rep, conversation_id=conv,
            run_id=trace.run_id, ok=self._is_good(trace),
            served_models=sorted(_served_models(trace)),
            node_sig=node_sig(trace.nodes_visited),
            total_tokens=trace.total_tokens,
            duration_ms=trace.total_duration_ms,
            filtered=trace.total_filtered,
            api_error=api_error,
        )

    # ── execution ──────────────────────────────────────────────────────────────

    def run_phase(self, phase: Phase, query_fn: QueryFn) -> tuple[list[PhaseRunRecord], dict[str, RunTrace]]:
        """Run every question × rep for one phase. Resumable: skips conv-ids that already
        have a good run. Returns (records, traces_by_run_id)."""
        records: list[PhaseRunRecord] = []
        traces: dict[str, RunTrace] = {}
        for rep in range(1, self.spec.reps + 1):
            for q in self.spec.questions:
                conv = self.spec.conv_id_template.format(phase=phase.name, qid=q.id, rep=rep)

                existing = self._resolve_run_by_conv(conv)
                if self._is_good(existing):
                    logger.info("[ABExperiment] skip %s (already good)", conv)
                    traces[existing.run_id] = existing
                    records.append(self._record_from_trace(phase, q, rep, conv, existing, None))
                    continue

                logger.info("[ABExperiment] run %s", conv)
                api_error = None
                try:
                    query_fn(q.text, conv, **q.kwargs)
                except Exception as e:  # transport failed — record and move on
                    api_error = repr(e)
                    logger.warning("[ABExperiment] query_fn failed for %s: %s", conv, api_error)

                if self.spec.settle_seconds > 0:
                    time.sleep(self.spec.settle_seconds)

                trace = self._resolve_run_by_conv(conv)
                if trace is not None and trace.run_id:
                    traces[trace.run_id] = trace
                records.append(self._record_from_trace(phase, q, rep, conv, trace, api_error))
        return records, traces

    def run(self, query_fn: QueryFn) -> ExperimentResult:
        """Run all phases and (for exactly 2 phases) compute the A/B comparison."""
        all_records: list[PhaseRunRecord] = []
        all_traces: dict[str, RunTrace] = {}
        for phase in self.spec.phases:
            recs, traces = self.run_phase(phase, query_fn)
            all_records.extend(recs)
            all_traces.update(traces)

        result = ExperimentResult(spec=self.spec, records=all_records, _traces_by_run=all_traces)

        if len(self.spec.phases) == 2:
            pa, pb = self.spec.phases
            result.comparison = compute_ab_comparison(
                result.traces_for_phase(pa.name),
                result.traces_for_phase(pb.name),
                expected_a=pa.expected_model,
                expected_b=pb.expected_model,
                cohort_a=pa.name,
                cohort_b=pb.name,
            )
        return result
