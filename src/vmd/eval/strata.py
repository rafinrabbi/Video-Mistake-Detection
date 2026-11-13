"""Per-recipe and per-error-type breakdowns with honest uncertainty.

Recipe identity is not stored in the saved records; it is recovered from the
dataset loader (``clip.task_id``) rather than parsed from the id, so the same
mapping the graphs use is applied. With 3--6 test recordings per recipe the
within-recipe interval is wide by construction and is shown as such; the
question "where does the graph help" is answered at three levels:

1. within-recipe paired bootstrap over that recipe's recordings;
2. macro over recipes (each recipe one unit) beside the pooled number;
3. sign statistics over the 24 paired per-recipe differences.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

from vmd.eval.curves import average_precision, roc_auc


def recipe_of_clips(raw_dir, split: str) -> Dict[str, str]:
    from vmd.data.captaincook4d import CaptainCook4DDataset
    ds = CaptainCook4DDataset(raw_dir, split)
    return {c.clip_id: c.task_id for c in ds.clips}


def group_metric(y: np.ndarray, s: np.ndarray, groups: np.ndarray, fn: Callable,
                 w: np.ndarray = None) -> Dict[str, float]:
    out = {}
    for g in np.unique(groups):
        m = groups == g
        out[str(g)] = fn(y[m], s[m], None if w is None else w[m])
    return out


def macro_mean(values: Dict[str, float]) -> float:
    v = np.array([x for x in values.values() if not np.isnan(x)])
    return float(v.mean()) if len(v) else float("nan")


def sign_summary(deltas: Sequence[float]) -> dict:
    d = np.array([x for x in deltas if not np.isnan(x)])
    if len(d) == 0:
        return {"n": 0}
    wins = int((d > 0).sum()); losses = int((d < 0).sum())
    # exact two-sided sign test
    from math import comb
    n = wins + losses
    k = min(wins, losses)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2 if n else float("nan")
    return {"n": int(len(d)), "wins": wins, "losses": losses, "median": float(np.median(d)),
            "mean": float(d.mean()), "sign_test_p": float(min(1.0, p))}


def one_vs_normal_auc(y_cls: np.ndarray, s: np.ndarray, cls: str, w: np.ndarray = None) -> float:
    """AUC of class ``cls`` windows against no_mistake windows only."""
    m = (y_cls == cls) | (y_cls == "no_mistake")
    if m.sum() == 0:
        return float("nan")
    return roc_auc((y_cls[m] == cls).astype(float), s[m], None if w is None else w[m])


__all__ = ["recipe_of_clips", "group_metric", "macro_mean", "sign_summary",
           "one_vs_normal_auc", "average_precision", "roc_auc"]
