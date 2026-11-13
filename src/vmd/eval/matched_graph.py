"""Matched topology controls for replay of graph-independent step emissions.

The historical replay retained the original DAG roots in every mode. Here
the start prior and adjacency both follow the requested control topology.
All modes use the same emissions, durations, filter settings and update time.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from vmd.eval.fusion import apply_fusion, fit_fusion
from vmd.tracking.belief import BeliefFilter, FilterConfig


def control_space(space, mode):
    if mode == "dag":
        return space
    if mode == "linear":
        succ = [[i + 1] if i + 1 < space.n else [] for i in range(space.n)]
        pred = [[i - 1] if i else [] for i in range(space.n)]
    elif mode == "uniform":
        succ = [[j for j in range(space.n) if j != i] for i in range(space.n)]
        pred = [[] for _ in range(space.n)]
    else:
        raise ValueError(f"Unknown topology: {mode}")
    return replace(space, succ=succ, pred=pred)


def replay_clip(step_rows, content_rows, space, step_seconds, mode):
    """Join exact windows, update once per observation, return fresh records.

    Recognition-only outputs intentionally have no boolean score. Validity of
    the content score and of the recognition distribution is checked separately.
    Historical content scalars cannot be repaired by this replay.
    """
    if not step_rows or len(step_rows) != len(content_rows):
        raise ValueError("Both sources must cover exactly the same nonempty windows")
    spac = control_space(space, mode)
    filt = BeliefFilter(spac, FilterConfig.from_step_seconds(step_seconds, 4.0, mode=mode))
    out = []
    for s, c in zip(step_rows, content_rows):
        key = ("clip_id", "start_sec", "end_sec", "true_label", "true_class", "task_id")
        if any(s[k] != c[k] for k in key):
            raise ValueError("Recognition and content windows or labels do not align")
        p = np.asarray(s["step_probs"], dtype=float)
        if (p.shape != (space.n + 1,) or not np.isfinite(p).all()
                or (p < 0).any() or p.sum() <= 0):
            raise ValueError("Invalid step emission")
        score = c.get("confidence")
        if (score is None or not np.isfinite(score) or not 0 <= score <= 1
                or c.get("score_status", "ok") != "ok" or not c.get("parse_ok", True)):
            raise ValueError("Invalid source content score")
        prior = filt.argmax
        state = filt.update(p)
        r = dict(c)
        r.update(topology=mode, step_prior_idx=prior, step_posterior_idx=state.argmax,
                 emission_argmax=int(np.argmax(p)),
                 gold_step_idx_strict=s["gold_step_idx_strict"],
                 illegal_hold=state.illegal_hold, structural_surprise=state.surprise,
                 belief_entropy=state.entropy, violation_mass=state.violation_mass,
                 source_content_score_version=c.get("score_version", "legacy_unspecified"))
        out.append(r)
    return out


def recording_folds(recording_ids, n_folds=5, seed=20261020):
    ids = sorted(recording_ids)
    if len(ids) < n_folds:
        raise ValueError("Too few recordings for cross-validation")
    shuffled = np.random.default_rng(seed).permutation(ids)
    return {str(cid): int(i % n_folds) for i, cid in enumerate(shuffled)}


def crossfit_fusion(records, names, folds):
    """Every validation recording is scored by weights fitted without it."""
    out = []
    params = {}
    for fold in sorted(set(folds.values())):
        train = [r for r in records if folds[r["clip_id"]] != fold]
        held = [r for r in records if folds[r["clip_id"]] == fold]
        if {int(r["true_label"]) for r in train} != {0, 1}:
            raise ValueError("A fusion training fold lacks a class")
        model = fit_fusion(train, names)
        params[str(fold)] = model
        for r in apply_fusion(held, model):
            r["crossfit_fold"] = fold
            out.append(r)
    out.sort(key=lambda r: (r["clip_id"], r["start_sec"]))
    return out, params


def weighted_ap_preparation(y, scores):
    """Sort once, preserving grouped ties for recording-bootstrap AP."""
    y, scores = np.asarray(y, float), np.asarray(scores, float)
    order = np.argsort(-scores, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(scores[order]) != 0), len(y) - 1]
    return order, ends, y[order]


def weighted_ap(prepared, weights):
    order, ends, y = prepared
    w = np.asarray(weights)[order]
    tp = np.cumsum(y * w)[ends]
    mass = np.cumsum(w)[ends]
    if tp[-1] <= 0:
        return None
    precision = np.divide(tp, mass, out=np.zeros_like(tp), where=mass > 0)
    return float(np.sum(np.diff(np.r_[0, tp]) * precision) / tp[-1])
