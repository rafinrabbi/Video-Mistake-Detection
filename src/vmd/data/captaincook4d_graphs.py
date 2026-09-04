"""Convert CaptainCook4D ground-truth task graphs to VMD `ProceduralGraph`.

CaptainCook4D ships a gold DAG per recipe (`task_graphs/<recipe>.json`):

    { "steps": {"0": "START", "1": "Mix-...", ..., "16": "END"},
      "edges": [[0, 3], [1, 2], [2, 16], ...] }     # [from, to] = precedence

We map each to `vmd.graph.schema.ProceduralGraph` (StepNode + after-Edge) and
cache it under `data/graphs/<recipe>.json` so `vmd eval` picks it up by stem,
identical to LLM-built graphs. These gold graphs also give Contribution A a
free validation target (LLM-built graph vs gold via graph-edit distance).

START/END sentinels are kept as steps so edge endpoints stay valid; they carry
``is_optional=True`` to flag them as non-executable anchors.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from vmd.graph.schema import Edge, EdgeKind, ProceduralGraph, StepNode
from vmd.graph.validator import validate_graph
from vmd.data.captaincook4d import recipe_key

_SENTINELS = {"START", "END"}


def cc4d_task_graph_to_procedural(path: Path, task_id: str) -> ProceduralGraph:
    raw = json.loads(Path(path).read_text())
    steps_map = raw["steps"]  # {idx(str): description}
    steps: List[StepNode] = []
    for idx, desc in steps_map.items():
        steps.append(
            StepNode(
                id=str(idx),
                description=desc,
                is_optional=desc.strip().upper() in _SENTINELS,
            )
        )
    edges = [
        Edge(src=str(a), dst=str(b), kind=EdgeKind.after)
        for a, b in raw.get("edges", [])
    ]
    graph = ProceduralGraph(
        task_id=task_id,
        task_name=task_id,
        steps=steps,
        objects=[],
        actions=[],
        states=[],
        edges=edges,
    )
    validate_graph(graph)
    return graph


def export_cc4d_graphs(annotations_root: Path, cache_dir: Path) -> List[str]:
    """Convert every task_graphs/*.json and write to cache_dir/<recipe>.json.

    Returns the list of task_ids written. ``<recipe>`` filenames already equal
    the recipe_key, so they line up with `CaptainCook4DDataset` clip.task_id.
    """
    annotations_root = Path(annotations_root)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for gpath in sorted((annotations_root / "task_graphs").glob("*.json")):
        task_id = recipe_key(gpath.stem)  # idempotent on already-normalised stems
        graph = cc4d_task_graph_to_procedural(gpath, task_id)
        (cache_dir / f"{task_id}.json").write_text(graph.model_dump_json(indent=2))
        written.append(task_id)
    return written
