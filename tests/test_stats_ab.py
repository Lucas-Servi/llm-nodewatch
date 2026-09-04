"""Tests for the A/B model comparator (stats.compute_ab_comparison + node_sig)."""

from nodewatch.models import LLMCall, NodeSpan, RunTrace
from nodewatch.stats import compute_ab_comparison, node_sig


def _trace(model, query, dur_ms, in_tok, out_tok, nodes, filtered_node=None, qid=None):
    spans = []
    for n in nodes:
        calls = [LLMCall(
            node_name=n, model=model, provider="bedrock",
            input_tokens=in_tok, output_tokens=out_tok,
            stop_reason=("content_filtered" if n == filtered_node else "end_turn"),
        )]
        spans.append(NodeSpan(node_name=n, start_time=0.0, end_time=dur_ms / 1000.0, llm_calls=calls))
    md = {"ab_question_id": qid} if qid else {}
    return RunTrace(graph_name="v2", query=query, total_duration_ms=dur_ms, node_spans=spans, metadata=md)


def test_node_sig_dedup_and_tools_strip():
    assert node_sig(["coordinator", "researcher", "researcher_tools", "researcher", "compiler"]) == \
        "coordinator -> researcher -> compiler"
    assert node_sig([]) == ""


def test_compute_ab_comparison_basic():
    path = ["coordinator", "researcher", "compiler"]
    a = [_trace("us.anthropic.claude-opus-4-8", "Q1", 10000, 1000, 200, path, qid="q1")]
    b = [_trace("us.anthropic.claude-opus-4-7", "Q1", 12000, 1100, 250, path,
                filtered_node="researcher", qid="q1")]

    comp = compute_ab_comparison(a, b, "opus-4-8", "opus-4-7")

    assert comp.verified_ok is True
    assert len(comp.per_question) == 1
    qd = comp.per_question[0]
    assert qd.matched_node_sig == "coordinator -> researcher -> compiler"
    assert qd.duration_delta_pct == 20.0           # 10s -> 12s
    assert qd.tokens_delta_pct > 0                 # B has more tokens
    assert qd.mean_filtered_per_call_a == 0.0
    assert qd.mean_filtered_per_call_b == 1 / 3    # 1 filtered call of 3
    assert comp.overall_tokens_delta_pct > 0


def test_compute_ab_comparison_model_mismatch_sets_verified_false():
    # Cohort A was SUPPOSED to be 4-8 but actually ran 4-7.
    a = [_trace("us.anthropic.claude-opus-4-7", "Q1", 10000, 1000, 200, ["coordinator"], qid="q1")]
    b = [_trace("us.anthropic.claude-opus-4-7", "Q1", 10000, 1000, 200, ["coordinator"], qid="q1")]

    comp = compute_ab_comparison(a, b, "opus-4-8", "opus-4-7")

    assert comp.verification[0].ok is False        # A mismatch
    assert comp.verification[1].ok is True
    assert comp.verified_ok is False


def test_compute_ab_comparison_pairs_by_query_text_without_metadata():
    """When no ab_question_id is present, the normalized query text pairs the cohorts."""
    a = [_trace("us.anthropic.claude-opus-4-8", "What is entropy?", 5000, 500, 100, ["coordinator"])]
    b = [_trace("us.anthropic.claude-opus-4-7", "what is   ENTROPY?", 4000, 500, 100, ["coordinator"])]

    comp = compute_ab_comparison(a, b, "opus-4-8", "opus-4-7")
    assert len(comp.per_question) == 1             # paired despite case/whitespace differences


def test_compute_ab_comparison_non_opus_models_verify():
    """After dropping the opus-only pre-filter, arbitrary model ids must still verify.

    Deliberately non-opus: this guards that _served_models collects any vendor, so the
    comparator works for gpt/mistral/etc., not just opus-vs-opus.
    """
    path = ["agent"]
    a = [_trace("gpt-4o", "Q1", 5000, 500, 100, path, qid="q1")]
    b = [_trace("gpt-4o-mini", "Q1", 4000, 500, 100, path, qid="q1")]

    comp = compute_ab_comparison(a, b, "gpt-4o", "gpt-4o-mini")

    assert comp.verified_ok is True
    assert comp.verification[0].served_models == ["gpt-4o"]
    assert comp.verification[1].served_models == ["gpt-4o-mini"]
    assert len(comp.per_question) == 1


def test_compute_ab_comparison_unmatched_paths_fall_back_to_all_reps():
    """When the two cohorts took different node paths, matched_node_sig is None but the
    comparison still runs over all reps."""
    a = [_trace("us.anthropic.claude-opus-4-8", "Q", 10000, 1000, 200, ["coordinator", "researcher"], qid="q")]
    b = [_trace("us.anthropic.claude-opus-4-7", "Q", 12000, 1000, 200,
                ["coordinator", "researcher", "domains_expert", "compiler"], qid="q")]

    comp = compute_ab_comparison(a, b, "opus-4-8", "opus-4-7")
    assert len(comp.per_question) == 1
    assert comp.per_question[0].matched_node_sig is None
