from __future__ import annotations
import networkx as nx
from vmd.graph.schema import ProceduralGraph, EdgeKind


def validate_graph(graph: ProceduralGraph) -> None:
    _check_step_dag(graph)
    _check_no_isolated_steps(graph)


def _check_step_dag(graph: ProceduralGraph) -> None:
    step_ids = set(graph.step_ids())
    G = nx.DiGraph()
    G.add_nodes_from(step_ids)
    for e in graph.edges:
        if e.kind == EdgeKind.after and e.src in step_ids and e.dst in step_ids:
            G.add_edge(e.src, e.dst)
    if not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))
        raise ValueError(f"Procedural graph has step cycles: {cycles}")


def _check_no_isolated_steps(graph: ProceduralGraph) -> None:
    if len(graph.steps) <= 1:
        return
    step_ids = set(graph.step_ids())
    connected = set()
    for e in graph.edges:
        if e.src in step_ids:
            connected.add(e.src)
        if e.dst in step_ids:
            connected.add(e.dst)
    isolated = step_ids - connected
    if isolated:
        raise ValueError(f"Steps have no edges (isolated): {isolated}")
