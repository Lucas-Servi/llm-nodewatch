"""Static graph topology analysis — extracts nodes, edges, and parallel structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodeInfo:
    name: str
    is_subgraph: bool = False


@dataclass
class EdgeInfo:
    source: str
    target: str
    conditional: bool = False


@dataclass
class GraphTopology:
    nodes: list[NodeInfo] = field(default_factory=list)
    edges: list[EdgeInfo] = field(default_factory=list)

    @property
    def node_names(self) -> list[str]:
        return [n.name for n in self.nodes]

    def get_parallel_lanes(self) -> list[list[str]]:
        """Find nodes that share the same source (fan-out pattern)."""
        from collections import defaultdict
        source_targets: dict[str, list[str]] = defaultdict(list)
        for e in self.edges:
            if e.conditional:
                source_targets[e.source].append(e.target)
        return [targets for targets in source_targets.values() if len(targets) > 1]

    def get_entry_nodes(self) -> list[str]:
        """Nodes with no incoming edges (except __start__)."""
        targets = {e.target for e in self.edges}
        sources = {e.source for e in self.edges}
        return [n.name for n in self.nodes if n.name not in targets and n.name in sources]

    def to_mermaid(self) -> str:
        lines = ["graph TD"]
        for e in self.edges:
            arrow = "-->" if not e.conditional else "-.->|conditional|"
            lines.append(f"    {e.source} {arrow} {e.target}")
        return "\n".join(lines)


def inspect_graph(compiled_graph: Any) -> GraphTopology:
    """Extract topology from a compiled LangGraph graph.

    Works with any CompiledStateGraph that exposes .get_graph().
    """
    try:
        graph = compiled_graph.get_graph()
    except AttributeError:
        return GraphTopology()

    nodes = []
    edges = []

    # Extract nodes
    for node_id, node_data in graph.nodes.items():
        if node_id in ("__start__", "__end__"):
            continue
        is_sub = hasattr(node_data, "graph") or "subgraph" in str(type(node_data)).lower()
        nodes.append(NodeInfo(name=node_id, is_subgraph=is_sub))

    # Extract edges
    for edge in graph.edges:
        src = edge.source if hasattr(edge, "source") else edge[0]
        tgt = edge.target if hasattr(edge, "target") else edge[1]
        conditional = hasattr(edge, "conditional") and edge.conditional
        if src == "__start__":
            src = "__start__"
        if tgt == "__end__":
            tgt = "__end__"
        edges.append(EdgeInfo(source=src, target=tgt, conditional=conditional))

    return GraphTopology(nodes=nodes, edges=edges)
