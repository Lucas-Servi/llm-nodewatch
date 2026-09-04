"""Tests for the BenchmarkRunner."""

import pytest
from langchain_core.messages import HumanMessage

from nodewatch import BenchmarkRunner, Query
from nodewatch.storage.sqlite import SQLiteStorage


@pytest.mark.asyncio
async def test_run_single(simple_graph, tmp_db):
    storage = SQLiteStorage(tmp_db)
    runner = BenchmarkRunner(storage=storage)

    def build_state(user_prompt: str = "", **kwargs):
        return {"messages": [HumanMessage(content=user_prompt)], "result": ""}

    trace = await runner.run_single(
        graph=simple_graph,
        graph_name="test",
        query=Query(text="Hello world"),
        state_builder=build_state,
    )

    assert trace.graph_name == "test"
    assert trace.query == "Hello world"
    assert trace.total_duration_ms > 0
    assert trace.error is None

    # Should be persisted
    loaded = storage.load(trace.run_id)
    assert loaded is not None

    storage.close()


@pytest.mark.asyncio
async def test_run_comparison(simple_graph, multi_node_graph, tmp_db):
    storage = SQLiteStorage(tmp_db)
    runner = BenchmarkRunner(storage=storage)

    def build_state(user_prompt: str = "", **kwargs):
        return {"messages": [HumanMessage(content=user_prompt)], "result": ""}

    report = await runner.run_comparison(
        graphs={"simple": simple_graph, "multi": multi_node_graph},
        queries=[Query(text="Test query")],
        state_builders={"simple": build_state, "multi": build_state},
    )

    assert len(report.results) == 1
    assert "simple" in report.results[0].traces
    assert "multi" in report.results[0].traces
    assert report.graph_names == ["simple", "multi"]

    storage.close()
