"""Tests for the transport-agnostic A/B phased runner (experiment.ABExperiment)."""

from nodewatch.experiment import ABExperiment, ExperimentQuestion, ExperimentSpec, Phase
from nodewatch.models import LLMCall, NodeSpan, RunTrace
from nodewatch.storage.sqlite import SQLiteStorage

_PHASE_MODEL = {"m48": "us.anthropic.claude-opus-4-8", "m47": "us.anthropic.claude-opus-4-7"}


def _make_query_fn(storage, counter):
    """A fake transport that, per call, writes a synthetic run tagged with the conv id.

    The conversation_id is stored ONLY in metadata (the column is left empty) to mirror
    real-world data where the column is unreliable, exercising the runner's metadata fallback.
    """
    def query_fn(text, conversation_id, **kw):
        counter["n"] += 1
        phase = conversation_id.split("_")[1]
        model = _PHASE_MODEL[phase]
        span = NodeSpan(
            node_name="coordinator", start_time=0.0, end_time=1.0,
            llm_calls=[LLMCall(node_name="coordinator", model=model, provider="bedrock",
                               input_tokens=1000, output_tokens=200, stop_reason="end_turn")],
        )
        trace = RunTrace(
            graph_name="v2", query=text, total_duration_ms=1000.0, node_spans=[span],
            metadata={"conversation_id": conversation_id, "ab_question_id": kw.get("qid", "")},
        )
        storage.save(trace)
    return query_fn


def _spec():
    return ExperimentSpec(
        phases=[Phase("m48", "opus-4-8"), Phase("m47", "opus-4-7")],
        questions=[ExperimentQuestion("q1", "What is the capital of France?", {"qid": "q1"})],
        reps=1, settle_seconds=0.0,
    )


def test_ab_experiment_runs_tags_verifies_and_compares(tmp_db):
    storage = SQLiteStorage(tmp_db)
    counter = {"n": 0}
    exp = ABExperiment(storage, _spec())

    result = exp.run(_make_query_fn(storage, counter))

    assert counter["n"] == 2                       # 2 phases x 1 question x 1 rep
    assert all(r.ok for r in result.records)
    assert len(result.records_for_phase("m48")) == 1
    assert result.comparison is not None
    assert result.comparison.verified_ok is True
    storage.close()


def test_ab_experiment_is_resumable(tmp_db):
    """A second run() against an already-populated store issues ZERO new queries."""
    storage = SQLiteStorage(tmp_db)
    counter = {"n": 0}
    exp = ABExperiment(storage, _spec())
    query_fn = _make_query_fn(storage, counter)

    exp.run(query_fn)
    assert counter["n"] == 2

    counter["n"] = 0
    result2 = exp.run(query_fn)
    assert counter["n"] == 0                       # all convs already good → skipped
    assert result2.comparison.verified_ok is True
    storage.close()
