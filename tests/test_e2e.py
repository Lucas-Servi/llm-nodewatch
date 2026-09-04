"""End-to-end smoke test: instrument a real (mock-LLM) LangGraph graph, persist
the trace to SQLite, and read it back through both the storage layer and the MCP tools.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from nodewatch import GraphTracker, mcp_server
from nodewatch.storage.sqlite import SQLiteStorage


@pytest.mark.asyncio
async def test_e2e_trace_roundtrip(multi_node_graph, tmp_db, monkeypatch):
    # 1. Instrument and run a multi-node graph.
    tracker = GraphTracker("e2e")
    result = await multi_node_graph.ainvoke(
        {"messages": [HumanMessage(content="hello")], "result": ""},
        config=tracker.config,
    )
    trace = tracker.finalize(query="hello", final_response=result.get("result", ""))

    assert trace.graph_name == "e2e"
    # The fixture's mock model doesn't emit LangChain LLM callbacks, so we assert on
    # node-span capture (the part the real graph execution drives) rather than llm_calls.
    assert {s.node_name for s in trace.node_spans} == {"planner", "researcher", "compiler"}

    # 2. Persist and reload via the storage layer.
    storage = SQLiteStorage(tmp_db)
    storage.save(trace)
    reloaded = storage.load(trace.run_id)
    storage.close()

    assert reloaded is not None
    assert reloaded.run_id == trace.run_id
    assert [s.node_name for s in reloaded.node_spans] == [s.node_name for s in trace.node_spans]
    assert reloaded.total_tokens == trace.total_tokens

    # 3. Read it back through the MCP tools (which resolve storage from NODEWATCH_DB).
    monkeypatch.delenv("NODEWATCH_URL", raising=False)
    monkeypatch.setenv("NODEWATCH_DB", tmp_db)

    list_runs = getattr(mcp_server.list_runs, "fn", mcp_server.list_runs)
    get_run = getattr(mcp_server.get_run, "fn", mcp_server.get_run)

    runs = list_runs(graph_name="e2e", limit=5)
    assert any(r["run_id"] == trace.run_id for r in runs)

    detail = get_run(trace.run_id)
    assert detail["graph_name"] == "e2e"
    assert {n["node_name"] for n in detail["nodes"]} == {"planner", "researcher", "compiler"}
