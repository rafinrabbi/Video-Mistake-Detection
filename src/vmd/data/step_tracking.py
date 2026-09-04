"""Which procedure step is the person on?

The original system answered this with the window index, clamped to the last
step. Because a step spans many windows, that pinned 93% of training windows and
66% of test windows to the final step, so the "graph context" was a constant
string for most of every recording and the model had almost nothing positional
to learn from or to use. This module replaces that with two rules that actually
track progress, one for each setting.

* Training is teacher-forced on the benchmark's gold step segmentation, which
  ships with CaptainCook4D and had never been loaded.
* Evaluation cannot use gold steps without leaking, so it keeps the causal
  state machine -- the pointer advances on the model's own no-deviation
  judgements -- but rate-limits it to the pace a step actually takes, estimated
  per recipe from the TRAINING split only.
"""
from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional

from vmd.data.egoops import EgoOopsClip, StepSpan

# Fallback when a recipe has no usable training estimate (seconds per step).
DEFAULT_STEP_SEC = 35.0


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def gold_step_span(spans: List[StepSpan], t_sec: float) -> Optional[StepSpan]:
    """The gold step in progress at ``t_sec``, or None before the first one."""
    best, best_t = None, -1.0
    for sp in spans:
        if sp.start_sec < 0:
            continue
        if sp.start_sec <= t_sec and sp.start_sec >= best_t:
            best_t, best = sp.start_sec, sp
    return best


def graph_index_of_span(graph_steps, span: Optional[StepSpan]) -> Optional[int]:
    """Position of ``span`` in a recipe's graph, matched on description.

    Not on ``step_id``: the benchmark numbers steps GLOBALLY across all recipes
    (pinwheels uses 102--118) while each recipe's graph numbers them locally
    from zero. Only ``microwaveeggsandwich`` matches by id, and only by
    coincidence -- it is the first activity. Descriptions join 3157 of 3158
    training spans exactly.
    """
    if span is None:
        return None
    target = _norm(span.description)
    for i, st_ in enumerate(graph_steps):
        if _norm(st_.description) == target:
            return i
    for i, st_ in enumerate(graph_steps):        # fall back to the id
        if str(st_.id) == str(span.step_id):
            return i
    return None


def gold_step_id(spans: List[StepSpan], t_sec: float) -> Optional[int]:
    """``step_id`` of the gold step in progress at ``t_sec``, if any.

    Between two steps the last step that has started is still the reference: the
    person is finishing it or moving on, not yet inside the next one. Steps
    without bounds mark nothing on the timeline and are skipped.

    Returns the benchmark's ``step_id``, deliberately not a positional index:
    gold spans are ordered by TIME while a recipe's graph is in canonical order,
    and the two orderings differ. Indexing one by the other's position silently
    conditions on the wrong step.
    """
    best_id, best_t = None, -1.0
    for sp in spans:
        if sp.start_sec < 0:
            continue
        if sp.start_sec <= t_sec and sp.start_sec >= best_t:
            best_t, best_id = sp.start_sec, sp.step_id
    return best_id


def executable_indices(graph_steps) -> List[int]:
    """Indices of real steps, excluding the START/END sentinels.

    The sentinels are kept in the graph so edge endpoints resolve, but showing
    the model "START" as the step it should be performing is meaningless.
    """
    idx = [i for i, s in enumerate(graph_steps) if not getattr(s, "is_optional", False)]
    return idx or list(range(len(graph_steps)))


def graph_index_of_step_id(graph_steps, step_id: Optional[int]) -> Optional[int]:
    """Position in ``graph_steps`` of the step whose id matches ``step_id``."""
    if step_id is None:
        return None
    target = str(step_id)
    for i, s in enumerate(graph_steps):
        if str(s.id) == target:
            return i
    return None


def estimate_step_seconds(clips: List[EgoOopsClip]) -> Dict[str, float]:
    """Median seconds per step, per recipe, from gold spans.

    Must be built from the training split only: at test time this is a prior,
    not an observation.
    """
    per_task: Dict[str, List[float]] = {}
    for c in clips:
        for sp in c.step_spans:
            if sp.start_sec >= 0 and sp.end_sec > sp.start_sec:
                per_task.setdefault(c.task_id, []).append(sp.end_sec - sp.start_sec)
    return {k: float(st.median(v)) for k, v in per_task.items() if v}


def save_step_seconds(path: Path, table: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=2, sort_keys=True))


def load_step_seconds(path: Path) -> Dict[str, float]:
    return json.loads(path.read_text()) if Path(path).exists() else {}


class RateLimitedPointer:
    """Causal step pointer that advances at the pace a step actually takes.

    Same state-machine idea as before -- a deviation holds position, agreement
    moves on -- but a step is only allowed to be left once roughly the expected
    time for one step has elapsed at it. Without that limit the pointer advanced
    once per window and reached the end of the recipe after the first minute.
    """

    def __init__(self, n_steps: int, step_seconds: Optional[float] = None,
                 stride_sec: float = 4.0, order: Optional[List[int]] = None):
        # ``order`` is the list of graph positions to walk, normally the
        # executable steps only, so the pointer never rests on a sentinel.
        self.order = list(order) if order else list(range(max(1, n_steps)))
        self.n_steps = len(self.order)
        self.step_seconds = float(step_seconds or DEFAULT_STEP_SEC)
        self.stride_sec = stride_sec
        # windows to spend at a step before advancing is permitted
        self.min_windows = max(1, int(round(self.step_seconds / max(stride_sec, 1e-6))))
        self._pos = 0        # position within self.order
        self._held = 0

    def observe(self, deviation: bool) -> int:
        """Record one window's judgement; return the step index to condition on.

        A deviation holds position but does NOT discard the progress already
        made at this step. Resetting was the obvious reading of "a deviation
        holds position", and it is brittle here: precision is 0.407, so isolated
        false deviations are common, and one every few windows would stall the
        pointer permanently. Measured on the real prediction stream, holding
        leaves 16.8% of windows at the final step against 11.8% for resetting --
        both far below the 66% of the per-window rule -- and holding degrades
        gracefully when the model is noisy, which it is.
        """
        current = self.idx
        if not deviation:
            self._held += 1
            if self._held >= self.min_windows and self._pos < self.n_steps - 1:
                self._pos += 1
                self._held = 0
        return current

    @property
    def idx(self) -> int:
        """Index into the recipe graph's step list."""
        return self.order[min(self._pos, self.n_steps - 1)]
