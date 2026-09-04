"""Tests for the async callback wrappers (``aon_*``).

LangChain prefers a handler's async methods when the graph is driven with
``ainvoke`` — which is the documented primary usage — so these wrappers, not the
sync ones, are what runs in production. They existed with no coverage at all.

Each wrapper is a thin delegation to its sync twin, so the property worth
pinning is *equivalence*: a run driven entirely through ``aon_*`` must produce
the same trace as the same events driven through ``on_*``. A wrapper that
silently dropped its arguments, swallowed an exception, or delegated to the
wrong method would still look fine in isolation and would only show up as
missing tokens in a trace.
"""

import uuid

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from nodewatch import GraphTracker

_USAGE = {
    "input_tokens": 1500,
    "output_tokens": 300,
    "input_token_details": {"cache_read": 400, "cache_creation": 100},
}


def _llm_result(text: str = "ok") -> LLMResult:
    msg = AIMessage(content=text)
    msg.usage_metadata = dict(_USAGE)
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


async def _drive_async(tracker: GraphTracker, node: str = "agent") -> None:
    """Drive one chain → llm → tool sequence entirely through the async wrappers."""
    md = {"langgraph_node": node}
    chain_id, llm_id, tool_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

    await tracker.aon_chain_start({"name": node}, {}, run_id=chain_id, metadata=md)
    await tracker.aon_llm_start(
        {}, [], run_id=llm_id, parent_run_id=chain_id, metadata=md,
        invocation_params={"model": "claude-sonnet-4-6"},
    )
    await tracker.aon_llm_end(_llm_result(), run_id=llm_id, parent_run_id=chain_id, metadata=md)
    await tracker.aon_tool_start(
        {"name": "search"}, "query", run_id=tool_id, parent_run_id=chain_id, metadata=md
    )
    await tracker.aon_tool_end("result rows", run_id=tool_id, parent_run_id=chain_id, metadata=md)
    await tracker.aon_chain_end({}, run_id=chain_id, metadata=md)


def _drive_sync(tracker: GraphTracker, node: str = "agent") -> None:
    """The same sequence through the sync callbacks."""
    md = {"langgraph_node": node}
    chain_id, llm_id, tool_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

    tracker.on_chain_start({"name": node}, {}, run_id=chain_id, metadata=md)
    tracker.on_llm_start(
        {}, [], run_id=llm_id, parent_run_id=chain_id, metadata=md,
        invocation_params={"model": "claude-sonnet-4-6"},
    )
    tracker.on_llm_end(_llm_result(), run_id=llm_id, parent_run_id=chain_id, metadata=md)
    tracker.on_tool_start(
        {"name": "search"}, "query", run_id=tool_id, parent_run_id=chain_id, metadata=md
    )
    tracker.on_tool_end("result rows", run_id=tool_id, parent_run_id=chain_id, metadata=md)
    tracker.on_chain_end({}, run_id=chain_id, metadata=md)


@pytest.mark.asyncio
async def test_async_wrappers_capture_the_same_trace_as_sync():
    """The whole point: aon_* must be equivalent to on_*, not merely not-crash."""
    async_tracker = GraphTracker("g")
    await _drive_async(async_tracker)
    async_trace = async_tracker.finalize(query="q", final_response="a")

    sync_tracker = GraphTracker("g")
    _drive_sync(sync_tracker)
    sync_trace = sync_tracker.finalize(query="q", final_response="a")

    assert [s.node_name for s in async_trace.node_spans] == [s.node_name for s in sync_trace.node_spans]
    assert async_trace.total_input_tokens == sync_trace.total_input_tokens
    assert async_trace.total_output_tokens == sync_trace.total_output_tokens
    assert async_trace.total_cost_usd == sync_trace.total_cost_usd
    assert len(async_trace.node_spans[0].tool_calls) == len(sync_trace.node_spans[0].tool_calls)


@pytest.mark.asyncio
async def test_async_wrappers_record_tokens_and_cache_detail():
    """Guards against a wrapper that delegates but loses its kwargs."""
    tracker = GraphTracker("g")
    await _drive_async(tracker)
    trace = tracker.finalize()

    assert len(trace.node_spans) == 1
    span = trace.node_spans[0]
    assert span.node_name == "agent"

    call = span.llm_calls[0]
    assert call.model == "claude-sonnet-4-6"
    # input_tokens is kept EXCLUSIVE of cache: 1500 - 400 - 100.
    assert call.input_tokens == 1000
    assert call.cache_read_tokens == 400
    assert call.cache_creation_tokens == 100
    assert call.output_tokens == 300
    assert call.cost_usd > 0


@pytest.mark.asyncio
async def test_async_tool_call_is_recorded_with_its_name():
    tracker = GraphTracker("g")
    await _drive_async(tracker)
    trace = tracker.finalize()

    tools = trace.node_spans[0].tool_calls
    assert [t.tool_name for t in tools] == ["search"]
    assert tools[0].success is True


@pytest.mark.asyncio
async def test_async_error_wrappers_record_the_failure():
    tracker = GraphTracker("g")
    md = {"langgraph_node": "agent"}

    chain_id, tool_id = uuid.uuid4(), uuid.uuid4()
    await tracker.aon_chain_start({"name": "agent"}, {}, run_id=chain_id, metadata=md)
    await tracker.aon_tool_start(
        {"name": "flaky"}, "in", run_id=tool_id, parent_run_id=chain_id, metadata=md
    )
    await tracker.aon_tool_error(ValueError("kaboom"), run_id=tool_id, parent_run_id=chain_id, metadata=md)

    call = tracker._node_spans["agent"].tool_calls[-1]
    assert call.success is False
    assert "kaboom" in call.error


@pytest.mark.asyncio
async def test_async_llm_error_wrapper_records_the_failure():
    tracker = GraphTracker("g")
    md = {"langgraph_node": "agent"}

    llm_id = uuid.uuid4()
    await tracker.aon_llm_start(
        {}, [], run_id=llm_id, metadata=md, invocation_params={"model": "claude-sonnet-4-6"}
    )
    await tracker.aon_llm_error(RuntimeError("upstream 503"), run_id=llm_id, metadata=md)

    call = tracker._node_spans["agent"].llm_calls[-1]
    assert call.error is not None
    assert "upstream 503" in call.error


@pytest.mark.asyncio
async def test_async_wrappers_are_used_by_a_real_ainvoke(simple_graph):
    """End-to-end: LangChain picks the async path on its own for ainvoke."""
    from langchain_core.messages import HumanMessage

    tracker = GraphTracker("async_e2e")
    await simple_graph.ainvoke(
        {"messages": [HumanMessage(content="hi")], "result": ""}, config=tracker.config
    )
    trace = tracker.finalize(query="hi")

    assert trace.node_spans, "ainvoke produced no node spans"
    assert trace.error is None
