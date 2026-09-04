from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, model_validator


class EdgeKind(str, Enum):
    after = "after"          # step B must come after step A
    requires = "requires"    # step requires object/state as precondition
    produces = "produces"    # step produces object/state as postcondition


class ObjectNode(BaseModel):
    id: str
    name: str
    expected_states: List[str] = []


class ActionNode(BaseModel):
    id: str
    verb: str
    target_object_ids: List[str] = []


class StepNode(BaseModel):
    id: str
    description: str
    action_ids: List[str] = []
    required_object_ids: List[str] = []
    produced_state_ids: List[str] = []
    valid_variations: List[str] = []   # acceptable alternative ways to do step
    is_optional: bool = False


class StateNode(BaseModel):
    id: str
    description: str
    object_id: str


class Edge(BaseModel):
    src: str    # node id
    dst: str    # node id
    kind: EdgeKind


class ProceduralGraph(BaseModel):
    task_id: str
    task_name: str
    steps: List[StepNode]
    objects: List[ObjectNode]
    actions: List[ActionNode]
    states: List[StateNode]
    edges: List[Edge]

    def step_ids(self) -> List[str]:
        return [s.id for s in self.steps]

    def object_ids(self) -> List[str]:
        return [o.id for o in self.objects]

    def valid_next_steps(self, current_step_id: str) -> List[StepNode]:
        after_edges = {e.dst for e in self.edges if e.kind == EdgeKind.after and e.src == current_step_id}
        return [s for s in self.steps if s.id in after_edges]

    @model_validator(mode="after")
    def _no_dangling_refs(self) -> "ProceduralGraph":
        obj_ids = self.object_ids()
        act_ids = {a.id for a in self.actions}
        state_ids = {s.id for s in self.states}
        all_ids = set(self.step_ids()) | set(obj_ids) | act_ids | state_ids
        for e in self.edges:
            assert e.src in all_ids, f"Edge src {e.src!r} not in graph"
            assert e.dst in all_ids, f"Edge dst {e.dst!r} not in graph"
        return self
