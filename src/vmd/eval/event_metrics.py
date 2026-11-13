"""Event-matched detection, false-alarm rate, delay, and step-tracking accuracy.

The historical localization metric (EDS on the earliest gold event, early
alarms clamped to zero lag) rewards a detector for firing early and often: the
always-fire rule scores EDS 1.0. The metrics here match predicted alarm
EPISODES to every localizable gold event and count what does not match as a
false alarm, so a system can only score by being right about where mistakes
are.

Episodes come from the frozen decision layer (``gate_clip``): maximal runs of
fired windows of length >= ``min_consecutive``, consistent with
``detect_start_of_deviation``. Nothing in ``vmd.eval.decision_layer`` changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vmd.eval.decision_layer import gate_clip

Interval = Tuple[float, float]


@dataclass
class GoldEvent:
    start: float
    end: float
    mistake_class: str


def episodes_from_fired(fired: Sequence[bool], starts: Sequence[float],
                        ends: Sequence[float], min_consecutive: int = 1) -> List[Interval]:
    """Maximal runs of fired windows with at least ``min_consecutive`` windows."""
    out: List[Interval] = []
    i, n = 0, len(fired)
    while i < n:
        if fired[i]:
            j = i
            while j + 1 < n and fired[j + 1]:
                j += 1
            if j - i + 1 >= min_consecutive:
                out.append((float(starts[i]), float(ends[j])))
            i = j + 1
        else:
            i += 1
    return out


def episodes_for_clip(records: List[dict], tau: float, smooth_k: int, gate_mode: str,
                      min_consecutive: int) -> List[Interval]:
    g = gate_clip(records, tau, smooth_k, gate_mode)
    fired = [s > 0.0 for s in g["smoothed"]]
    return episodes_from_fired(fired, g["starts"], [float(r["end_sec"]) for r in records],
                               min_consecutive)


def iou(a: Interval, b: Interval) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _qualifies(ep: Interval, ev: GoldEvent, criterion: str, iou_thr: float, delta: float) -> bool:
    if criterion == "iou":
        return iou(ep, (ev.start, ev.end)) >= iou_thr
    if criterion == "onset":
        overlaps = min(ep[1], ev.end) > max(ep[0], ev.start)
        return overlaps and (ev.start - delta) <= ep[0] <= ev.end
    raise ValueError(criterion)


def match_events(episodes: List[Interval], events: List[GoldEvent], criterion: str = "onset",
                 iou_thr: float = 0.3, delta: float = 8.0) -> Tuple[List[Optional[int]], List[bool]]:
    """For each gold event the index of the earliest qualifying episode (or
    None); for each episode whether it matched at least one event.

    An episode may satisfy several overlapping gold events (43 pairs of test
    events overlap in time), and each gold event is credited once.
    """
    matched_ep = [False] * len(episodes)
    assignment: List[Optional[int]] = []
    for ev in events:
        hit = None
        for k, ep in enumerate(episodes):
            if _qualifies(ep, ev, criterion, iou_thr, delta):
                hit = k
                break
        assignment.append(hit)
        if hit is not None:
            matched_ep[hit] = True
    return assignment, matched_ep


@dataclass
class ClipEventStats:
    clip_id: str
    duration_min: float
    clean: bool                                  # recording has no localizable error
    n_events: int
    n_detected: int
    per_class_events: Dict[str, int]
    per_class_detected: Dict[str, int]
    n_episodes: int
    n_false_alarms: int
    delays: List[float] = field(default_factory=list)          # p_start - g_start, matched only
    norm_delays: List[float] = field(default_factory=list)     # delay / event duration


def clip_event_stats(clip_id: str, episodes: List[Interval], events: List[GoldEvent],
                     duration_sec: float, criterion: str = "onset", iou_thr: float = 0.3,
                     delta: float = 8.0) -> ClipEventStats:
    assignment, matched_ep = match_events(episodes, events, criterion, iou_thr, delta)
    per_ev: Dict[str, int] = {}; per_det: Dict[str, int] = {}
    delays, norm = [], []
    for ev, k in zip(events, assignment):
        per_ev[ev.mistake_class] = per_ev.get(ev.mistake_class, 0) + 1
        if k is not None:
            per_det[ev.mistake_class] = per_det.get(ev.mistake_class, 0) + 1
            d = episodes[k][0] - ev.start
            delays.append(d)
            norm.append(d / max(ev.end - ev.start, 1e-6))
    return ClipEventStats(
        clip_id=clip_id, duration_min=duration_sec / 60.0, clean=len(events) == 0,
        n_events=len(events), n_detected=sum(k is not None for k in assignment),
        per_class_events=per_ev, per_class_detected=per_det,
        n_episodes=len(episodes), n_false_alarms=sum(not m for m in matched_ep),
        delays=delays, norm_delays=norm,
    )


def aggregate_event_stats(stats: Sequence[ClipEventStats],
                          weights: Optional[Sequence[float]] = None) -> dict:
    w = np.ones(len(stats)) if weights is None else np.asarray(weights, dtype=float)
    n_ev = sum(wi * s.n_events for wi, s in zip(w, stats))
    n_det = sum(wi * s.n_detected for wi, s in zip(w, stats))
    n_ep = sum(wi * s.n_episodes for wi, s in zip(w, stats))
    n_fa = sum(wi * s.n_false_alarms for wi, s in zip(w, stats))
    minutes = sum(wi * s.duration_min for wi, s in zip(w, stats))
    clean_fa = sum(wi * s.n_false_alarms for wi, s in zip(w, stats) if s.clean)
    clean_min = sum(wi * s.duration_min for wi, s in zip(w, stats) if s.clean)
    delays = np.concatenate([np.repeat(s.delays, int(wi)) for wi, s in zip(w, stats) if s.delays and wi > 0]) \
        if any(s.delays and wi > 0 for wi, s in zip(w, stats)) else np.array([])
    norm = np.concatenate([np.repeat(s.norm_delays, int(wi)) for wi, s in zip(w, stats) if s.norm_delays and wi > 0]) \
        if any(s.norm_delays and wi > 0 for wi, s in zip(w, stats)) else np.array([])
    er = n_det / n_ev if n_ev else float("nan")
    ep = n_det / n_ep if n_ep else 0.0     # matched episodes ~ detected events (one per event)
    classes = sorted({c for s in stats for c in s.per_class_events})
    per_class = {}
    for c in classes:
        e = sum(wi * s.per_class_events.get(c, 0) for wi, s in zip(w, stats))
        d = sum(wi * s.per_class_detected.get(c, 0) for wi, s in zip(w, stats))
        per_class[c] = {"events": float(e), "recall": d / e if e else float("nan")}
    return {
        "event_recall": er,
        "event_precision": min(1.0, ep),
        "event_f1": 2 * er * min(1.0, ep) / (er + min(1.0, ep)) if (n_ev and n_ep and er + ep > 0) else 0.0,
        "false_alarms_per_min": n_fa / minutes if minutes else float("nan"),
        "clean_false_alarms_per_min": clean_fa / clean_min if clean_min else float("nan"),
        "median_delay_sec": float(np.median(delays)) if len(delays) else float("nan"),
        "delay_iqr_sec": [float(np.percentile(delays, 25)), float(np.percentile(delays, 75))] if len(delays) else None,
        "median_norm_delay": float(np.median(norm)) if len(norm) else float("nan"),
        "n_events": float(n_ev), "n_detected": float(n_det), "n_episodes": float(n_ep),
        "n_false_alarms": float(n_fa), "per_class": per_class,
    }


def froc_points(rows: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """(fa_per_min, event_recall) sorted by false-alarm rate, from per-tau aggregates."""
    fa = np.array([r["false_alarms_per_min"] for r in rows]); er = np.array([r["event_recall"] for r in rows])
    order = np.argsort(fa, kind="stable")
    return fa[order], er[order]


def recall_at_fa_rate(fa: np.ndarray, er: np.ndarray, target: float) -> float:
    """Best event recall achievable at <= ``target`` false alarms per minute.

    Takes the max over qualifying operating points (the curve need not be
    monotone), and 0 when no point qualifies.
    """
    ok = fa <= target
    return float(er[ok].max()) if ok.any() else 0.0


# ------------------------------------------------------------------ tracking

def tracking_accuracy(tracked: Sequence[Optional[int]], gold: Sequence[Optional[int]],
                      succ: Optional[List[List[int]]] = None) -> dict:
    """Window-level agreement between a tracked step index and the gold step.

    Windows without a gold step (gaps, before the first step) are excluded.
    ``succ`` (menu-index adjacency) enables the DAG-neighbourhood accuracy:
    tracked == gold or the two are joined by an edge in either direction.
    """
    pairs = [(t, g) for t, g in zip(tracked, gold) if g is not None]
    if not pairs:
        return {"n": 0, "step_accuracy": float("nan"), "neighbourhood_accuracy": float("nan")}
    exact = np.mean([t == g for t, g in pairs])
    if succ is not None:
        near = np.mean([t == g or (t is not None and t < len(succ) and
                                   (g in succ[t] or t in succ[g])) for t, g in pairs])
    else:
        near = float("nan")
    return {"n": len(pairs), "step_accuracy": float(exact), "neighbourhood_accuracy": float(near)}


def dedup(seq: Sequence[Optional[int]]) -> List[int]:
    """Collapse consecutive repeats and drop None -- the segment sequence."""
    out: List[int] = []
    for x in seq:
        if x is None:
            continue
        if not out or out[-1] != x:
            out.append(int(x))
    return out
