"""Fuse the VLM's content score with the tracker's structural signals.

The decision layer is frozen and reads only ``confidence``/``deviation``, so
fusion produces a DERIVED prediction file in the same record format whose
``confidence`` is the fused probability and ``deviation`` its 0.5 decision.
Weights are fitted on validation records (logistic regression on a handful of
features) and applied unchanged to test. ``structural_only`` keeps just the
graph signal, to show what the graph detects on its own.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

FEATURES = ("content_logit", "illegal_hold", "structural_surprise", "belief_entropy")


def _logit(p: float, eps: float = 1e-4) -> float:
    p = min(max(p, eps), 1 - eps)
    return float(np.log(p / (1 - p)))


def features_of(record: dict, names: Sequence[str] = FEATURES) -> List[float]:
    out = []
    for n in names:
        if n == "content_logit":
            out.append(_logit(float(record["confidence"])))
        else:
            out.append(float(record.get(n, 0.0)))
    return out


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 500, lr: float = 0.1) -> np.ndarray:
    """Plain gradient-descent logistic regression with an intercept (no sklearn
    dependency at fusion time; features are standardised inside)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]]
        w -= lr * g
    return np.r_[w, mu, sd]


def predict_logistic(params: np.ndarray, X: np.ndarray) -> np.ndarray:
    k = X.shape[1]
    w, mu, sd = params[: k + 1], params[k + 1: 2 * k + 1], params[2 * k + 1:]
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    return 1 / (1 + np.exp(-Z @ w))


def fit_fusion(val_records: Sequence[dict], names: Sequence[str] = FEATURES) -> dict:
    X = np.array([features_of(r, names) for r in val_records])
    y = np.array([int(r["true_label"]) for r in val_records], float)
    params = fit_logistic(X, y)
    return {"features": list(names), "params": params.tolist(), "n_val": len(y),
            "fitted_on": "validation records only"}


def apply_fusion(records: Sequence[dict], fusion: dict) -> List[dict]:
    names = fusion["features"]
    X = np.array([features_of(r, names) for r in records])
    p = predict_logistic(np.array(fusion["params"]), X)
    out = []
    for r, pi in zip(records, p):
        q = copy.copy(r)
        q["content_confidence"] = r["confidence"]
        q["confidence"] = float(pi)
        q["deviation"] = bool(pi >= 0.5)
        out.append(q)
    return out


def structural_only(records: Sequence[dict], signal: str = "illegal_hold") -> List[dict]:
    """Derived arm whose score is the graph signal alone (typed class kept)."""
    out = []
    for r in records:
        q = copy.copy(r)
        v = float(r.get(signal, 0.0))
        q["content_confidence"] = r["confidence"]
        q["confidence"] = min(1.0, max(0.0, v))
        q["deviation"] = bool(v >= 0.5)
        out.append(q)
    return out


def write_derived(records: Sequence[dict], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(list(records), indent=1))
