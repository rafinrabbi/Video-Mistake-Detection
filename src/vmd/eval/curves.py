"""Threshold-free detection curves with recording-level bootstrap weights.

Everything takes an optional per-window weight vector so a bootstrap draw over
recordings (weight = how many times the recording was drawn) reuses the same
code as the point estimate. Ties in the score share one cutoff, matching
``sklearn.metrics`` on unweighted data (pinned in tests/test_curves.py).

Two kinds of curve are exposed and must not be confused in figures:

* ``pr_curve`` ranks the RAW per-window score -- the classifier before the
  decision layer;
* ``decision_layer_sweep`` runs the actual gate + median filter at each tau of
  the validation grid, giving the operating points the pipeline can reach.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vmd.eval.decision_layer import gate_clip

# Same tau grid as scripts/cc4d_decision_sweep.py.
TAU_GRID = tuple(round(0.30 + 0.025 * i, 3) for i in range(25))
RECALL_GRID = np.linspace(0.0, 1.0, 201)


def _weights(y, w):
    return np.ones(len(y)) if w is None else np.asarray(w, dtype=float)


def _cut_counts(y, s, w):
    """Positive/negative weight per distinct score, scores ascending."""
    y = np.asarray(y, dtype=float); s = np.asarray(s, dtype=float); w = _weights(y, w)
    order = np.argsort(s, kind="stable")
    cuts = np.r_[0, np.flatnonzero(np.diff(s[order])) + 1]
    pos = np.add.reduceat((w * y)[order], cuts)
    neg = np.add.reduceat((w * (1 - y))[order], cuts)
    return s[order][cuts], pos, neg


def pr_curve(y, s, w=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(precision, recall, threshold) for every cutoff ``score >= threshold``,
    thresholds descending, so recall is non-decreasing along the arrays."""
    thr, pos, neg = _cut_counts(y, s, w)
    tp = np.cumsum(pos[::-1]); fp = np.cumsum(neg[::-1])
    p_total = pos.sum()
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = tp / p_total if p_total > 0 else np.zeros_like(tp)
    return precision, recall, thr[::-1]


def average_precision(y, s, w=None) -> float:
    """Step-wise AP: sum over cutoffs of precision x recall increment."""
    thr, pos, neg = _cut_counts(y, s, w)
    p_total = pos.sum()
    if p_total <= 0:
        return float("nan")
    tp = np.cumsum(pos[::-1]); tot = np.cumsum((pos + neg)[::-1])
    prec = np.divide(tp, tot, out=np.zeros_like(tp), where=tot > 0)
    return float(np.sum(pos[::-1] * prec) / p_total)


def roc_auc(y, s, w=None) -> float:
    thr, pos, neg = _cut_counts(y, s, w)
    p, q = pos.sum(), neg.sum()
    if p <= 0 or q <= 0:
        return float("nan")
    return float(np.sum(pos * (np.cumsum(neg) - neg / 2)) / (p * q))


def interpolated_precision(precision, recall, grid=RECALL_GRID) -> np.ndarray:
    """Max precision at recall >= r for each r in ``grid`` (0 where unreachable)."""
    precision = np.asarray(precision); recall = np.asarray(recall)
    out = np.zeros(len(grid))
    if len(recall) == 0:
        return out
    # suffix max over recall-ascending arrays
    order = np.argsort(recall, kind="stable")
    r_sorted = recall[order]; p_sorted = precision[order]
    suffix = np.maximum.accumulate(p_sorted[::-1])[::-1]
    idx = np.searchsorted(r_sorted, grid, side="left")
    ok = idx < len(r_sorted)
    out[ok] = suffix[idx[ok]]
    return out


def precision_at_recall(y, s, r_target: float, w=None) -> float:
    p, r, _ = pr_curve(y, s, w)
    ok = r >= r_target
    return float(p[ok].max()) if ok.any() else 0.0


def recall_at_fpr(y, s, fpr_target: float, w=None) -> Tuple[float, float]:
    """(TPR, threshold) at the largest threshold whose FPR <= target."""
    thr, pos, neg = _cut_counts(y, s, w)
    p, q = pos.sum(), neg.sum()
    tp = np.cumsum(pos[::-1]); fp = np.cumsum(neg[::-1]); t = thr[::-1]
    fpr = fp / q if q > 0 else np.zeros_like(fp)
    tpr = tp / p if p > 0 else np.zeros_like(tp)
    ok = np.flatnonzero(fpr <= fpr_target)
    if len(ok) == 0:
        return 0.0, float("inf")
    k = ok[-1]
    return float(tpr[k]), float(t[k])


def decision_layer_sweep(preds_by_clip: Dict[str, List[dict]], smooth_k: int,
                         gate_mode: str, taus: Sequence[float] = TAU_GRID,
                         clip_weights: Optional[Dict[str, float]] = None) -> List[dict]:
    """Pooled precision/recall/firing rate of the frozen pipeline at each tau."""
    rows = []
    for tau in taus:
        tp = fp = fn = fired = n = 0.0
        for cid, recs in preds_by_clip.items():
            w = 1.0 if clip_weights is None else clip_weights.get(cid, 0.0)
            if w == 0.0:
                continue
            g = gate_clip(recs, tau, smooth_k, gate_mode)
            y = np.array([int(r["true_label"]) for r in recs]); p = np.array(g["pred_bin"])
            tp += w * np.sum(y * p); fp += w * np.sum((1 - y) * p); fn += w * np.sum(y * (1 - p))
            fired += w * p.sum(); n += w * len(recs)
        rows.append({"tau": float(tau),
                     "precision": tp / (tp + fp) if tp + fp else 0.0,
                     "recall": tp / (tp + fn) if tp + fn else 0.0,
                     "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                     "firing_rate": fired / n if n else 0.0})
    return rows


def bootstrap_multiplicities(n_clips: int, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """(n_boot, n_clips) draw counts."""
    draws = np.random.default_rng(seed).integers(0, n_clips, (n_boot, n_clips))
    return np.array([np.bincount(d, minlength=n_clips) for d in draws])
