"""Benchmark orchestration: runs queries across graph variants and collects traces."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .models import RunTrace
from .storage.base import StorageBackend
from .tracker import GraphTracker

logger = logging.getLogger(__name__)


@dataclass
class Query:
    """A benchmark query to run against one or more graphs."""
    text: str
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Results from a single query across multiple graph variants."""
    query: Query
    traces: dict[str, RunTrace]  # graph_name -> trace


@dataclass
class ComparisonReport:
    """Complete benchmark report across all queries and graphs."""
    results: list[ComparisonResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def all_traces(self) -> list[RunTrace]:
        return [t for r in self.results for t in r.traces.values()]

    @property
    def graph_names(self) -> list[str]:
        names: list[str] = []
        for r in self.results:
            for g in r.traces:
                if g not in names:
                    names.append(g)
        return names


class BenchmarkRunner:
    """Orchestrates comparative benchmarks across LangGraph variants."""

    def __init__(self, storage: StorageBackend | None = None):
        self.storage = storage

    async def run_single(
        self,
        graph: Any,
        graph_name: str,
        query: Query,
        state_builder: Callable[..., dict[str, Any]],
        extra_config: dict | None = None,
    ) -> RunTrace:
        """Run a single query against a single graph, returning a full RunTrace."""
        tracker = GraphTracker(graph_name, metadata=query.metadata)

        config = tracker.config
        if extra_config:
            config.update(extra_config)

        initial_state = state_builder(user_prompt=query.text)

        try:
            result = await graph.ainvoke(initial_state, config=config)
        except Exception as e:
            trace = tracker.finalize(query=query.text)
            trace.error = f"{type(e).__name__}: {e}"
            if self.storage:
                self.storage.save(trace)
            return trace

        # Extract final response
        final_response = ""
        if isinstance(result, dict):
            final_response = result.get("final_response", "")
            if not final_response and "messages" in result:
                msgs = result["messages"]
                if msgs:
                    last = msgs[-1]
                    content = last.content if hasattr(last, "content") else ""
                    if isinstance(content, str):
                        final_response = content
                    elif isinstance(content, list):
                        final_response = "".join(
                            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                        )

        trace = tracker.finalize(query=query.text, final_response=final_response)

        if self.storage:
            self.storage.save(trace)

        return trace

    async def run_comparison(
        self,
        graphs: dict[str, Any],
        queries: list[Query],
        state_builders: dict[str, Callable[..., dict[str, Any]]],
        warmup: int = 0,
        extra_config: dict | None = None,
    ) -> ComparisonReport:
        """Run all queries against all graph variants sequentially."""
        # Warmup runs (discarded)
        if warmup > 0 and queries:
            for graph_name, graph in graphs.items():
                builder = state_builders[graph_name]
                for _ in range(warmup):
                    tracker = GraphTracker(graph_name)
                    try:
                        state = builder(user_prompt=queries[0].text)
                        await graph.ainvoke(state, config=tracker.config)
                    except Exception as e:
                        # Warmup results are discarded, so a failure here is not
                        # fatal — but swallowing it silently hides a graph that
                        # is broken for every subsequent measured run too.
                        logger.warning("warmup run for %r failed: %s", graph_name, e)

        results: list[ComparisonResult] = []

        for query in queries:
            traces: dict[str, RunTrace] = {}
            for graph_name, graph in graphs.items():
                builder = state_builders[graph_name]
                trace = await self.run_single(
                    graph=graph,
                    graph_name=graph_name,
                    query=query,
                    state_builder=builder,
                    extra_config=extra_config,
                )
                traces[graph_name] = trace

            results.append(ComparisonResult(query=query, traces=traces))

        return ComparisonReport(results=results)
