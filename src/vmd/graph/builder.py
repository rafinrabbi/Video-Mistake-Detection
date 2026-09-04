from __future__ import annotations
import json
import hashlib
import re
from pathlib import Path
from typing import Optional

from vmd.graph.schema import ActionNode, Edge, EdgeKind, ProceduralGraph, StepNode
from vmd.graph.validator import validate_graph

_PROMPT_PATH = Path(__file__).parent / "prompts" / "build_graph.txt"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text()


def _cache_path(cache_dir: Path, task_id: str) -> Path:
    return cache_dir / f"{task_id}.json"


def build_graph(
    task_name: str,
    task_description: str,
    task_id: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    llm_model: str = "gpt-4o",
) -> ProceduralGraph:
    if task_id is None:
        task_id = hashlib.md5(task_name.encode()).hexdigest()[:8]

    if cache_dir is not None:
        cp = _cache_path(cache_dir, task_id)
        if cp.exists():
            return ProceduralGraph.model_validate_json(cp.read_text())

    if llm_model in {"local", "procedural_text", "rule"}:
        graph = build_graph_from_procedural_text(task_name, task_description, task_id)
    else:
        prompt = _render_prompt(task_name, task_description)
        raw = _call_llm(prompt, llm_model)
        graph = _parse_and_validate(raw, task_id, task_name)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp.write_text(graph.model_dump_json(indent=2))

    return graph


def build_graph_from_procedural_text(
    task_name: str,
    task_description: str,
    task_id: str,
) -> ProceduralGraph:
    """Build an order-only graph directly from numbered procedural text."""
    instructions = _extract_instructions(task_description)
    if not instructions:
        raise ValueError(f"No procedural instructions found for task {task_id!r}")

    steps = []
    actions = []
    edges = []
    for idx, instruction in enumerate(instructions, start=1):
        step_id = f"step_{idx:02d}"
        action_id = f"act_{idx:02d}"
        steps.append(
            StepNode(
                id=step_id,
                description=instruction,
                action_ids=[action_id],
            )
        )
        actions.append(
            ActionNode(
                id=action_id,
                verb=_guess_verb(instruction),
                target_object_ids=[],
            )
        )
        if idx > 1:
            edges.append(Edge(src=f"step_{idx - 1:02d}", dst=step_id, kind=EdgeKind.after))

    graph = ProceduralGraph(
        task_id=task_id,
        task_name=task_name,
        steps=steps,
        objects=[],
        actions=actions,
        states=[],
        edges=edges,
    )
    validate_graph(graph)
    return graph


def _call_llm(prompt: str, model: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content
    except ImportError:
        raise RuntimeError("openai package required for graph building. Install with: pip install openai")


def _render_prompt(task_name: str, task_description: str) -> str:
    return (
        _PROMPT_TEMPLATE
        .replace("{task_name}", task_name)
        .replace("{task_description}", task_description)
    )


def _extract_instructions(task_description: str) -> list[str]:
    matches = re.findall(
        r"(?:^|\n)\s*\d+[.)]\s+(.*?)(?=\n\s*\d+[.)]\s+|\Z)",
        task_description,
        flags=re.DOTALL,
    )
    instructions = matches if matches else task_description.splitlines()
    return [" ".join(item.split()) for item in instructions if item.strip()]


def _guess_verb(instruction: str) -> str:
    match = re.search(r"[A-Za-z]+", instruction)
    return match.group(0).lower() if match else "perform"


def _parse_and_validate(raw: str, task_id: str, task_name: str) -> ProceduralGraph:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\n---\n{raw[:500]}")

    # Inject task_id/task_name if missing
    data.setdefault("task_id", task_id)
    data.setdefault("task_name", task_name)

    graph = ProceduralGraph.model_validate(data)
    validate_graph(graph)
    return graph
