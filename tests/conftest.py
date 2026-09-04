"""Shared test fixtures: mock LangGraph graphs for testing the tracker."""

from __future__ import annotations

from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class SimpleState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    result: str


def _make_mock_model(responses: list[str] | None = None):
    """Create a mock LLM that returns preset responses and exposes usage_metadata."""
    responses = responses or ["Mock response from the agent."]
    call_count = {"n": 0}

    class MockModel:
        def invoke(self, messages, config=None):
            idx = min(call_count["n"], len(responses) - 1)
            call_count["n"] += 1
            msg = AIMessage(content=responses[idx])
            msg.usage_metadata = {
                "input_tokens": 1000 + idx * 100,
                "output_tokens": 200 + idx * 50,
                "total_tokens": 1200 + idx * 150,
            }
            msg.response_metadata = {"stop_reason": "end_turn"}
            return msg

        async def ainvoke(self, messages, config=None):
            return self.invoke(messages, config)

        def bind_tools(self, tools):
            return self

    return MockModel()


@pytest.fixture
def simple_graph():
    """A minimal 2-node graph: agent -> end."""

    mock_model = _make_mock_model(["The answer is 42."])

    def agent_node(state: SimpleState):
        response = mock_model.invoke(state["messages"])
        return {"messages": [response], "result": response.content}

    workflow = StateGraph(SimpleState)
    workflow.add_node("agent", agent_node)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    return workflow.compile()


@pytest.fixture
def multi_node_graph():
    """A 3-node graph: planner -> researcher -> compiler."""

    mock_model = _make_mock_model([
        "Plan: look up data",
        "Research findings: item X has property Y",
        "Final: Based on research, item X has property Y.",
    ])
    call_idx = {"n": 0}

    def planner(state: SimpleState):
        response = mock_model.invoke(state["messages"])
        call_idx["n"] += 1
        return {"messages": [response]}

    def researcher(state: SimpleState):
        response = mock_model.invoke(state["messages"])
        call_idx["n"] += 1
        return {"messages": [response]}

    def compiler(state: SimpleState):
        response = mock_model.invoke(state["messages"])
        call_idx["n"] += 1
        return {"messages": [response], "result": response.content}

    workflow = StateGraph(SimpleState)
    workflow.add_node("planner", planner)
    workflow.add_node("researcher", researcher)
    workflow.add_node("compiler", compiler)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "compiler")
    workflow.add_edge("compiler", END)

    return workflow.compile()


@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary database path."""
    return str(tmp_path / "test_bench.db")
