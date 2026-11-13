"""The DAG-constrained belief filter on a synthetic fork/join recipe.

Recipe:  a -> b -> {c, d} -> e -> f   (c and d may run in either order)
Graph list order is deliberately scrambled so the tests also pin that the
menu order is topological, not positional.
"""
import numpy as np
import pytest

from vmd.graph.schema import Edge, EdgeKind, ProceduralGraph, StepNode
from vmd.tracking.belief import (
    BeliefFilter, FilterConfig, build_step_space, one_hot,
)


def _graph():
    ids = ["e", "START", "c", "a", "f", "d", "b", "END"]
    steps = [StepNode(id=i, description=f"step {i}",
                      is_optional=i in ("START", "END")) for i in ids]
    edges = [("START", "a"), ("a", "b"), ("b", "c"), ("b", "d"),
             ("c", "e"), ("d", "e"), ("e", "f"), ("f", "END")]
    return ProceduralGraph(task_id="toy", task_name="toy", steps=steps,
                           objects=[], actions=[], states=[],
                           edges=[Edge(src=s, dst=d, kind=EdgeKind.after) for s, d in edges])


@pytest.fixture
def space():
    return build_step_space(_graph())


def _seq(space, names, reps=4):
    idx = {n: i for i, n in enumerate(space.node_ids)}
    return [idx[n] for n in names for _ in range(reps)]


def test_step_space_is_topological_and_drops_sentinels(space):
    assert space.node_ids[0] == "a" and space.node_ids[1] == "b"
    assert set(space.node_ids[2:4]) == {"c", "d"}
    assert space.node_ids[4:] == ["e", "f"]
    assert "START" not in space.node_ids and "END" not in space.node_ids
    assert space.roots == [0] and space.sinks == [5]
    e = space.node_ids.index("e")
    assert sorted(space.pred[e]) == [2, 3]


def test_perfect_emissions_track_truth_with_no_surprise(space):
    f = BeliefFilter(space, FilterConfig(p_stay=0.7))
    truth = _seq(space, ["a", "b", "c", "d", "e", "f"])
    states = f.run([one_hot(i, space.n, 0.95) for i in truth])
    acc = np.mean([s.argmax == t for s, t in zip(states, truth)])
    assert acc >= 0.9
    assert max(s.surprise for s in states) < 0.15


def test_either_branch_order_is_fine_under_dag_but_not_linear(space):
    truth = _seq(space, ["a", "b", "d", "c", "e", "f"])   # d before c
    ems = [one_hot(i, space.n, 0.95) for i in truth]
    dag = BeliefFilter(space, FilterConfig(p_stay=0.7, mode="dag")).run(ems)
    lin = BeliefFilter(space, FilterConfig(p_stay=0.7, mode="linear")).run(ems)
    assert max(s.surprise for s in dag) < 0.15
    assert max(s.illegal_occupancy for s in dag) < 0.05
    # menu order puts c before d; the linear chain sees d -> c as a violation
    assert max(s.surprise for s in lin) > 3 * max(s.surprise for s in dag)
    assert max(s.illegal_occupancy for s in lin) > 0.5


def test_swapped_steps_spike_surprise_at_the_violation(space):
    # e performed before c/d: arriving at e with unvisited predecessors.
    truth = _seq(space, ["a", "b", "e", "c", "d", "f"])
    ems = [one_hot(i, space.n, 0.95) for i in truth]
    states = BeliefFilter(space, FilterConfig(p_stay=0.7)).run(ems)
    sur = np.array([s.surprise for s in states])
    first_e = 8                       # windows 8..11 are e
    assert sur[first_e] == sur.max() and sur[first_e] > 0.4
    assert sur[:first_e].max() < 0.15
    assert states[first_e].violation_mass > 0.3
    occ = np.array([s.illegal_occupancy for s in states])
    # fires while e is being accepted (hysteresis = 2 windows), silent otherwise
    assert occ[first_e] > 0.3 and occ[first_e + 1] > 0.5
    assert occ[:first_e].max() < 0.05 and occ[first_e + 2:].max() < 0.05


def test_skipped_step_is_surprising_on_arrival(space):
    truth = _seq(space, ["a", "b", "c", "e", "f"])         # d skipped
    states = BeliefFilter(space, FilterConfig(p_stay=0.7)).run(
        [one_hot(i, space.n, 0.95) for i in truth])
    sur = np.array([s.surprise for s in states])
    occ = np.array([s.illegal_occupancy for s in states])
    assert sur[12] == sur.max() and sur[12] > 0.3 and sur[:12].max() < 0.15
    assert occ[12] > 0.3 and occ[:12].max() < 0.05


def test_dag_prior_beats_uniform_under_noise(space):
    rng = np.random.default_rng(0)
    truth = _seq(space, ["a", "b", "c", "d", "e", "f"], reps=8)
    ems = []
    for t in truth:
        if rng.random() < 0.45:               # 45% of windows are pure noise
            ems.append(one_hot(int(rng.integers(space.n)), space.n, 0.9))
        else:
            ems.append(one_hot(t, space.n, 0.9))
    acc = {}
    for mode in ("dag", "uniform"):
        st = BeliefFilter(space, FilterConfig(p_stay=0.85, mode=mode)).run(ems)
        acc[mode] = np.mean([s.argmax == t for s, t in zip(st, truth)])
    assert acc["dag"] > acc["uniform"]


def test_transit_state_absorbs_gaps(space):
    f = BeliefFilter(space, FilterConfig(p_stay=0.7))
    f.run([one_hot(0, space.n, 0.95)] * 4)
    s = f.update(one_hot(None, space.n, 0.95))
    assert s.argmax == space.n and s.surprise < 0.15


def test_rows_are_stochastic(space):
    f = BeliefFilter(space, FilterConfig())
    T, D = f.transition_matrices()
    assert np.allclose(T.sum(axis=1), 1.0)
    assert np.all(D <= T + 1e-12) and np.all(D >= 0)
