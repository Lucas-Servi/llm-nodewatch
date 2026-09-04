"""Tests for classifying tool failures that arrive as error *payloads*.

``on_tool_end`` used to hardcode ``success=True``, and ``success=False`` was set
only in ``on_tool_error``. But most MCP / HTTP-backed tools return an error
payload instead of raising, so ``on_tool_error`` never fires for them. Result: a
graph run where several tools genuinely failed was recorded as all-successful
with ``error: null``, and the observability layer could not be used to check
whether a fix worked.

``classify_tool_error`` derives the failure at capture time (the
``is_filtered_stop`` precedent — no stored column, no schema migration).

The negative tests below are the important ones: the recognizers are ANCHORED
because real tool *content* contains the word "error". Hardware catalogs are the
easy example — every ECC part is literally named "Error-Correcting …" — but the
same trap shows up in any corpus with technical product or field names. A
substring search for "error" would mark those honest results as failures: the
same defect, inverted. That is what the ``Errors?(?!-)`` negative lookahead is
for.
"""

import uuid

import pytest

from nodewatch import GraphTracker, classify_tool_error

# ── Positives: real failure shapes seen in production ───────────────────────

@pytest.mark.parametrize(
    "output",
    [
        "[Web Search Error] upstream returned 503",
        "[Web Read Error] could not fetch url",
        "[API Search Error] rate limited",
        "[Attachment Error] file missing",
        "[PDF Parse Error] encrypted document",
    ],
)
def test_bracketed_error_markers_are_failures(output):
    assert classify_tool_error(output) is not None


@pytest.mark.parametrize(
    "output",
    [
        "Error: Layer 'foo' not found.",
        "Error - could not save memory",
        "An error occurred: KeyError('user_id')",
        "Traceback (most recent call last):\n  File ...",
    ],
)
def test_anchored_prose_errors_are_failures(output):
    assert classify_tool_error(output) is not None


def test_json_error_payload_is_a_failure():
    msg = classify_tool_error('{"error": "No valid record IDs provided."}')
    assert msg == "No valid record IDs provided."


def test_nonzero_error_code_is_a_failure():
    assert classify_tool_error('{"errorCode": 1, "message": "boom"}') is not None
    # The wrapped shape
    assert classify_tool_error('{"result": {"errorCode": 2, "data": null}}') is not None


def test_error_code_zero_is_success():
    """errorCode 0 is the SUCCESS value in this convention."""
    assert classify_tool_error('{"result": {"errorCode": 0, "data": [1, 2]}}') is None
    assert classify_tool_error('{"errorCode": 0}') is None


# ── Negatives: the reason the recognizers must be anchored ──────────────────

@pytest.mark.parametrize(
    "output",
    [
        # Legitimate catalog/field values that begin with "Error-".
        "Error-Correcting Memory Module (ECC), 32GB",
        "forward error-correcting channel decoder, rev B",
        '[{"sku": "MEM-1042", "count": 3, "name": "Error-Correcting Memory Module"}]',
        '{"fields": ["ecc"], "product": "error-correcting DIMM"}',
        # Prose that merely mentions errors.
        "Transcription error rate was 0.3% across the batch.",
        "No hits found for term 'ecc' in catalog: MEM-1042.",
    ],
)
def test_legitimate_content_mentioning_error_is_not_a_failure(output):
    assert classify_tool_error(output) is None, (
        "anchored matching is required: substring 'error' appears in real tool "
        "content"
    )


def test_empty_and_none_output_is_not_a_failure():
    assert classify_tool_error("") is None
    assert classify_tool_error(None) is None


def test_non_dict_json_is_not_a_failure():
    assert classify_tool_error("[1, 2, 3]") is None
    assert classify_tool_error('"just a string"') is None


def test_malformed_json_is_not_a_failure():
    """A truncated payload must not be guessed at."""
    assert classify_tool_error('{"error": "unterminated') is None


# ── Wiring: the tracker records it ──────────────────────────────────────────

def _emit_tool_call(tracker: GraphTracker, output: str, node: str = "agent"):
    run_id = uuid.uuid4()
    md = {"langgraph_node": node}
    tracker.on_tool_start({"name": "some_tool"}, "input", run_id=run_id, metadata=md)
    tracker.on_tool_end(output, run_id=run_id, metadata=md)
    return tracker._node_spans[node].tool_calls[-1]


def test_tracker_marks_payload_failure_as_unsuccessful():
    tracker = GraphTracker("test")
    call = _emit_tool_call(tracker, '{"error": "At least 2 records are required."}')
    assert call.success is False
    assert call.error == "At least 2 records are required."


def test_tracker_keeps_honest_results_successful():
    tracker = GraphTracker("test")
    call = _emit_tool_call(tracker, "Error-Correcting Memory Module (ECC), 32GB")
    assert call.success is True
    assert call.error is None


def test_tracker_classifies_from_full_output_not_the_preview():
    """output_preview is truncated — the marker can sit past its cut."""
    from nodewatch.tracker import TOOL_OUTPUT_PREVIEW_CHARS

    tracker = GraphTracker("test")
    # Sized off the CONFIGURED cap, not a literal: the caps were raised
    # (128 -> 1000) once a 128-char preview proved too short to diagnose a real
    # tool bug, and a hardcoded width turns that into a failure here instead of
    # testing the behaviour that matters.
    payload = '{"data": "' + "x" * (TOOL_OUTPUT_PREVIEW_CHARS + 200) + '", "error": "late marker"}'
    call = _emit_tool_call(tracker, payload)
    assert call.success is False
    assert call.error == "late marker"
    # The marker really does sit past the preview cut, so the classification
    # cannot have come from the preview.
    assert len(call.output_preview) == TOOL_OUTPUT_PREVIEW_CHARS
    assert "late marker" not in call.output_preview
    assert call.output_size == len(payload)


def test_tracker_still_records_raised_exceptions():
    """on_tool_error behaviour is untouched."""
    tracker = GraphTracker("test")
    run_id = uuid.uuid4()
    md = {"langgraph_node": "agent"}
    tracker.on_tool_start({"name": "t"}, "in", run_id=run_id, metadata=md)
    tracker.on_tool_error(ValueError("kaboom"), run_id=run_id, metadata=md)
    call = tracker._node_spans["agent"].tool_calls[-1]
    assert call.success is False
    assert "ValueError: kaboom" in call.error
