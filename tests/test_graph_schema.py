import pytest
from vmd.graph.schema import ProceduralGraph, StepNode, ObjectNode, ActionNode, StateNode, Edge, EdgeKind
from vmd.graph.builder import _render_prompt, build_graph_from_procedural_text
from vmd.graph.validator import validate_graph


def _minimal_graph() -> ProceduralGraph:
    return ProceduralGraph(
        task_id="test_task",
        task_name="Test Task",
        steps=[
            StepNode(id="step_01", description="Step 1", action_ids=["act_01"], required_object_ids=["obj_01"]),
            StepNode(id="step_02", description="Step 2", action_ids=["act_02"]),
        ],
        objects=[ObjectNode(id="obj_01", name="Widget")],
        actions=[
            ActionNode(id="act_01", verb="pick", target_object_ids=["obj_01"]),
            ActionNode(id="act_02", verb="place"),
        ],
        states=[],
        edges=[
            Edge(src="step_01", dst="step_02", kind=EdgeKind.after),
            Edge(src="step_01", dst="obj_01", kind=EdgeKind.requires),
        ],
    )


def test_minimal_graph_valid():
    g = _minimal_graph()
    validate_graph(g)


def test_step_ids():
    g = _minimal_graph()
    assert "step_01" in g.step_ids()
    assert "step_02" in g.step_ids()


def test_valid_next_steps():
    g = _minimal_graph()
    nexts = g.valid_next_steps("step_01")
    assert len(nexts) == 1
    assert nexts[0].id == "step_02"


def test_no_next_for_final_step():
    g = _minimal_graph()
    assert g.valid_next_steps("step_02") == []


def test_dangling_edge_raises():
    with pytest.raises(Exception):
        ProceduralGraph(
            task_id="bad",
            task_name="Bad",
            steps=[StepNode(id="step_01", description="x")],
            objects=[],
            actions=[],
            states=[],
            edges=[Edge(src="step_01", dst="NONEXISTENT", kind=EdgeKind.after)],
        )


def test_cycle_raises():
    from vmd.graph.validator import validate_graph
    g = ProceduralGraph(
        task_id="cycle",
        task_name="Cycle",
        steps=[
            StepNode(id="step_01", description="A"),
            StepNode(id="step_02", description="B"),
        ],
        objects=[],
        actions=[],
        states=[],
        edges=[
            Edge(src="step_01", dst="step_02", kind=EdgeKind.after),
            Edge(src="step_02", dst="step_01", kind=EdgeKind.after),
        ],
    )
    with pytest.raises(ValueError, match="cycle"):
        validate_graph(g)


def test_round_trip_json():
    g = _minimal_graph()
    restored = ProceduralGraph.model_validate_json(g.model_dump_json())
    assert restored.task_id == g.task_id
    assert len(restored.steps) == len(g.steps)


def test_build_graph_prompt_renders_literal_json_braces():
    prompt = _render_prompt("Task", "1. Do the thing")
    assert "TASK: Task" in prompt
    assert "DESCRIPTION: 1. Do the thing" in prompt
    assert '"task_id": "<snake_case_id>"' in prompt


def test_local_graph_builder_uses_procedural_order():
    g = build_graph_from_procedural_text(
        "Make tea",
        "1. Boil water.\n2. Add tea leaves.\n3. Pour into cup.",
        "make_tea",
    )
    validate_graph(g)
    assert [s.id for s in g.steps] == ["step_01", "step_02", "step_03"]
    assert [s.description for s in g.steps] == [
        "Boil water.",
        "Add tea leaves.",
        "Pour into cup.",
    ]
    assert [(e.src, e.dst, e.kind) for e in g.edges] == [
        ("step_01", "step_02", EdgeKind.after),
        ("step_02", "step_03", EdgeKind.after),
    ]
