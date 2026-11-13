"""DAG-constrained belief over the current procedure step.

The previous trackers walked the recipe as a flat list: the per-window rule
raced to the final step, the rate-limited rule held each step for a fixed time,
and neither read the graph's edges except to decorate one prompt line. This
module treats the graph as what it is -- a partial order -- and tracks the
executor's position as a posterior over its executable nodes.

Per window the filter receives a likelihood over steps (from the VLM's step
token, or one-hot gold for diagnostics), predicts through a transition model
built from the DAG, and normalises. Two by-products matter downstream:

* the argmax, which becomes the "expected step" the next prompt is conditioned
  on (causal: belief at t-1 conditions window t);
* the structural surprise, the share of the observation's likelihood that can
  only be explained by transitions the graph forbids (skips, out-of-order
  arrivals). Order errors should light this up; technique errors should not.

Everything here is numpy and runs in microseconds, so the same filter is used
live in ``vmd.eval.online`` and replayed on saved records for analysis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence

import numpy as np

from vmd.graph.schema import EdgeKind, ProceduralGraph

FilterMode = Literal["dag", "linear", "uniform"]

# Menu labels shown to the model: one uppercase letter per executable step. No
# recipe in CaptainCook4D has more than 25 executable steps; the transit state
# uses "-" so it can never collide with a step letter.
MENU_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXY"
TRANSIT_LABEL = "-"


@dataclass
class StepSpace:
    """Executable nodes of one recipe in a fixed menu order.

    Menu order is a topological linear extension of the ``after`` edges, with
    ties broken by graph list order. Graph list order itself is NOT topological
    for any CaptainCook4D recipe, so callers must index through ``graph_pos``.
    """
    graph_id: str
    node_ids: List[str]            # StepNode.id per menu slot
    graph_pos: List[int]           # index into graph.steps per menu slot
    descriptions: List[str]
    succ: List[List[int]]          # DAG successors, menu indices
    pred: List[List[int]]          # DAG predecessors, menu indices

    @property
    def n(self) -> int:
        return len(self.node_ids)

    @property
    def roots(self) -> List[int]:
        return [i for i in range(self.n) if not self.pred[i]]

    @property
    def sinks(self) -> List[int]:
        return [i for i in range(self.n) if not self.succ[i]]

    @property
    def labels(self) -> List[str]:
        return list(MENU_LETTERS[: self.n])

    def menu_index_of_graph_pos(self, pos: int) -> Optional[int]:
        try:
            return self.graph_pos.index(pos)
        except ValueError:
            return None

    def is_edge(self, a: int, b: int) -> bool:
        return b in self.succ[a]


def build_step_space(graph: ProceduralGraph) -> StepSpace:
    """Executable steps in topological order with DAG adjacency.

    START/END sentinels (``is_optional``) are dropped; edges through them are
    collapsed so that a node whose only predecessor was START is a root.
    """
    n_all = len(graph.steps)
    exec_pos = [i for i, s in enumerate(graph.steps) if not s.is_optional] or list(range(n_all))
    exec_set = set(exec_pos)
    id_to_pos = {s.id: i for i, s in enumerate(graph.steps)}
    adj: Dict[int, set] = {i: set() for i in range(n_all)}
    for e in graph.edges:
        if e.kind != EdgeKind.after or e.src not in id_to_pos or e.dst not in id_to_pos:
            continue
        adj[id_to_pos[e.src]].add(id_to_pos[e.dst])
    # Collapse sentinel nodes: an edge a -> S -> b through a sentinel S becomes a -> b.
    for s in [i for i in range(n_all) if i not in exec_set]:
        preds_s = [a for a in range(n_all) if s in adj[a]]
        for a in preds_s:
            adj[a].discard(s)
            adj[a] |= {b for b in adj[s] if b in exec_set}
    exec_adj = {a: {b for b in adj[a] if b in exec_set and b != a} for a in exec_pos}
    # Kahn's algorithm, ties by graph list order (deterministic).
    indeg = {a: 0 for a in exec_pos}
    for a in exec_pos:
        for b in exec_adj[a]:
            indeg[b] += 1
    ready = sorted(a for a in exec_pos if indeg[a] == 0)
    order: List[int] = []
    while ready:
        a = ready.pop(0)
        order.append(a)
        for b in sorted(exec_adj[a]):
            indeg[b] -= 1
            if indeg[b] == 0:
                ready.append(b)
        ready.sort()
    if len(order) != len(exec_pos):          # cycle: fall back to list order
        order = list(exec_pos)
    menu_of_pos = {p: i for i, p in enumerate(order)}
    succ = [sorted(menu_of_pos[b] for b in exec_adj[p] if b in menu_of_pos) for p in order]
    pred: List[List[int]] = [[] for _ in order]
    for i, ss in enumerate(succ):
        for j in ss:
            pred[j].append(i)
    return StepSpace(
        graph_id=graph.task_id,
        node_ids=[graph.steps[p].id for p in order],
        graph_pos=list(order),
        descriptions=[graph.steps[p].description for p in order],
        succ=succ,
        pred=[sorted(p) for p in pred],
    )


@dataclass
class FilterConfig:
    """Transition and emission hyper-parameters. Tuned on validation only."""
    p_stay: float = 0.89        # self-loop; ~1 - stride/step_seconds
    # The leaving mass (1 - p_stay) is split into fractions so no setting of
    # p_stay can drive a transition negative.
    transit_frac: float = 0.3   # share of leaving mass into the between-steps state
    skip_frac: float = 0.1      # share of leaving mass onto DAG-forbidden nodes
    p_any: float = 0.005        # floor over every state (robustness)
    transit_stay: float = 0.5   # remain between steps
    emission_temp: float = 1.0  # likelihood ** temp
    emission_floor: float = 0.02  # mix with uniform so no state hits zero
    readiness: bool = True      # gate successors on visited predecessors
    visit_hysteresis: int = 2   # windows as argmax before a node counts as visited
    mode: FilterMode = "dag"

    @staticmethod
    def from_step_seconds(step_seconds: Optional[float], stride_sec: float = 4.0,
                          **overrides) -> "FilterConfig":
        cfg = FilterConfig(**overrides)
        if step_seconds and step_seconds > stride_sec:
            cfg.p_stay = float(np.clip(1.0 - stride_sec / step_seconds, 0.5, 0.98))
        return cfg


@dataclass
class BeliefState:
    belief: np.ndarray             # posterior over n steps + transit (last)
    argmax: int                    # menu index (n == transit)
    p_max: float
    entropy: float
    surprise: float                # -log(Z_dag / Z_full) >= 0
    violation_mass: float          # posterior mass that arrived via forbidden moves
    illegal_occupancy: float = 0.0  # posterior mass on steps whose prerequisites are undone
    illegal_hold: float = 0.0       # mass on the current step if it was entered out of order
    completed: List[int] = field(default_factory=list)   # visited menu indices
    ready_next: List[int] = field(default_factory=list)  # DAG-consistent next steps


class BeliefFilter:
    """Forward filter over ``StepSpace`` nodes plus one transit state."""

    def __init__(self, space: StepSpace, cfg: Optional[FilterConfig] = None):
        self.space = space
        self.cfg = cfg or FilterConfig()
        self.n = space.n
        self.transit = self.n
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        n = self.n
        b = np.full(n + 1, self.cfg.p_any)
        roots = self.space.roots or list(range(n))
        for r in roots:
            b[r] += 1.0 / len(roots)
        self.belief = b / b.sum()
        self.visited = np.zeros(n, dtype=bool)
        # Steps accepted as done while the DAG did not permit them: entered before
        # a prerequisite (early) or after one of their successors (late). Held
        # for the whole time the belief stays on that step, so the signal covers
        # the misordered step's span rather than only its first window.
        self.illegal_entry = np.zeros(n, dtype=bool)
        self._argmax_run: Optional[int] = None
        self._run_len = 0
        self.t = 0

    @property
    def argmax(self) -> int:
        return int(np.argmax(self.belief))

    def _ready(self, j: int) -> bool:
        """May step ``j`` legitimately start now? Mode-aware prerequisites."""
        if not self.cfg.readiness or self.cfg.mode == "uniform":
            return True
        if self.cfg.mode == "linear":
            return j == 0 or bool(self.visited[j - 1])
        return all(self.visited[p] for p in self.space.pred[j])

    # ------------------------------------------------------------- transition
    def transition_matrices(self):
        """(T_full, T_dag): row-stochastic full model and its DAG-consistent part.

        ``T_dag`` is ``T_full`` with the forbidden entries (skips, floor mass on
        non-successors) zeroed and NOT renormalised, so that the ratio of the
        two normalisers is a likelihood ratio in [0, 1].
        """
        n, c = self.n, self.cfg
        T = np.zeros((n + 1, n + 1))
        D = np.zeros((n + 1, n + 1))
        floor = c.p_any / (n + 1)
        for i in range(n):
            if c.mode == "uniform":
                allowed = [j for j in range(n) if j != i]
            elif c.mode == "linear":
                allowed = [i + 1] if i + 1 < n else []
            elif c.readiness:
                # A partial order allows any step whose prerequisites are all
                # done, not only i's own successors: after one branch of a fork
                # the sibling branch is legitimate next work.
                allowed = [j for j in range(n) if j != i and not self.visited[j] and self._ready(j)]
                if not allowed:                  # everything done: successors only
                    allowed = list(self.space.succ[i])
            else:
                allowed = list(self.space.succ[i])
            forbidden = [j for j in range(n) if j != i and j not in allowed]
            M = 1.0 - c.p_any
            row = np.full(n + 1, floor)
            stay = c.p_stay * M
            leave = M - stay
            transit = leave * c.transit_frac
            skip = leave * c.skip_frac if (forbidden and c.mode != "uniform") else 0.0
            advance = leave - transit - skip
            if not allowed:                   # sink or nothing ready: absorb
                stay += advance
                advance = 0.0
            row[i] += stay
            row[self.transit] += transit
            for j in allowed:
                row[j] += advance / len(allowed)
            for j in forbidden:
                row[j] += skip / len(forbidden)
            T[i] = row
            d = row.copy()
            for j in forbidden:
                d[j] = 0.0
            D[i] = d
        # transit row: stay, or move to any DAG-ready unvisited node
        frontier = [j for j in range(n) if not self.visited[j] and self._ready(j)] \
            if c.mode == "dag" else [j for j in range(n) if not self.visited[j]]
        row = np.full(n + 1, floor)
        M = 1.0 - c.p_any
        row[self.transit] += c.transit_stay * M
        if frontier:
            for j in frontier:
                row[j] += (1 - c.transit_stay) * M / len(frontier)
        else:
            row[self.transit] += (1 - c.transit_stay) * M
        T[self.transit] = row
        d = row.copy()
        for j in range(n):
            if j not in frontier:
                d[j] = 0.0
        D[self.transit] = d
        return T, D

    # ----------------------------------------------------------------- update
    def emission(self, step_probs: Sequence[float], p_transit: Optional[float] = None) -> np.ndarray:
        """Likelihood over n+1 states from a step distribution (length n or n+1)."""
        c = self.cfg
        p = np.asarray(step_probs, dtype=float)
        if p.shape[0] == self.n:
            pt = 0.0 if p_transit is None else float(p_transit)
            p = np.concatenate([p * (1.0 - pt), [pt]])
        if p.shape[0] != self.n + 1:
            raise ValueError(f"expected {self.n} or {self.n + 1} probabilities, got {p.shape[0]}")
        p = np.clip(p, 0.0, None)
        p = p / p.sum() if p.sum() > 0 else np.full(self.n + 1, 1.0 / (self.n + 1))
        e = p ** c.emission_temp
        e = (1.0 - c.emission_floor) * e / e.sum() + c.emission_floor / (self.n + 1)
        return e

    def update(self, step_probs: Sequence[float], p_transit: Optional[float] = None) -> BeliefState:
        e = self.emission(step_probs, p_transit)
        T, D = self.transition_matrices()
        pred_full = self.belief @ T
        pred_dag = self.belief @ D
        z_full = float(np.dot(pred_full, e))
        z_dag = float(np.dot(pred_dag, e))
        post = pred_full * e / max(z_full, 1e-300)
        surprise = -math.log(max(z_dag, 1e-300) / max(z_full, 1e-300))
        violation = float(np.dot(pred_full - pred_dag, e) / max(z_full, 1e-300))
        self.belief = post
        self.t += 1
        a = self.argmax
        # Mass sitting on a step the graph does not yet permit, judged before the
        # visited set is updated so a sustained illegal arrival fires only until
        # the step is accepted as done (hysteresis windows), then resets.
        illegal = float(sum(post[j] for j in range(self.n)
                            if not self.visited[j] and not self._ready(j)))
        # visited bookkeeping with hysteresis
        if a == self._argmax_run:
            self._run_len += 1
        else:
            self._argmax_run, self._run_len = a, 1
        if a < self.n and self._run_len >= self.cfg.visit_hysteresis and not self.visited[a]:
            early = not self._ready(a)
            late = any(self.visited[j] for j in self.space.succ[a]) if self.cfg.mode == "dag" \
                else (a + 1 < self.n and bool(self.visited[a + 1]) if self.cfg.mode == "linear" else False)
            self.illegal_entry[a] = early or late
            self.visited[a] = True
        hold = float(post[a]) if a < self.n and self.illegal_entry[a] else 0.0
        ent = float(-np.sum(post * np.log(np.clip(post, 1e-300, None))))
        ready_next = [j for j in (self.space.succ[a] if a < self.n else range(self.n))
                      if not self.visited[j] and self._ready(j)]
        return BeliefState(
            belief=post, argmax=a, p_max=float(post[a]), entropy=ent,
            surprise=max(0.0, surprise), violation_mass=max(0.0, violation),
            illegal_occupancy=illegal, illegal_hold=hold,
            completed=[int(i) for i in np.flatnonzero(self.visited)],
            ready_next=[int(j) for j in ready_next],
        )

    def run(self, emissions: Sequence[Sequence[float]]) -> List[BeliefState]:
        self.reset()
        return [self.update(e) for e in emissions]


def one_hot(index: Optional[int], n: int, confidence: float = 1.0) -> np.ndarray:
    """Emission for a known step (``None`` -> transit). ``confidence`` < 1 spreads
    the remainder uniformly, the knob used by the structural-oracle noise sweep."""
    e = np.full(n + 1, (1.0 - confidence) / (n + 1))
    e[n if index is None else index] += confidence
    return e / e.sum()
