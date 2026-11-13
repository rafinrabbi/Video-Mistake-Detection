from __future__ import annotations
import numpy as np
from typing import Dict, List, Sequence
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score
)

from vmd.data.mistake_classes import MISTAKE_CLASSES as _CANON_CLASSES


MistakeClass = str   # "no_mistake" or one of the mistake types

MISTAKE_CLASSES: list[str] = list(_CANON_CLASSES)

# ---------- Binary + multiclass classification ----------

def binary_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: List[str]) -> float:
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def per_class_f1(
    y_true: Sequence[str], y_pred: Sequence[str], labels: List[str]
) -> Dict[str, float]:
    """Per-class F1, one entry per label in `labels` (order preserved).

    Uses ``average=None`` with an explicit ``labels`` list so a class the model
    never predicts (or that never appears in y_true) still yields 0.0 rather
    than being silently dropped — honest for the 8-class ablation table.
    """
    scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {c: float(s) for c, s in zip(labels, scores)}


def per_class_support(y_true: Sequence[str], labels: List[str]) -> Dict[str, int]:
    """Count of ground-truth windows per class (context for low-support classes)."""
    counts = {c: 0 for c in labels}
    for t in y_true:
        if t in counts:
            counts[t] += 1
    return counts


def auc_roc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


# ---------- Edit distance ----------

def edit_distance(seq_pred: List[str], seq_true: List[str]) -> int:
    m, n = len(seq_pred), len(seq_true)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if seq_pred[i-1] == seq_true[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def normalised_edit_distance(seq_pred: List[str], seq_true: List[str]) -> float:
    denom = max(len(seq_pred), len(seq_true), 1)
    return edit_distance(seq_pred, seq_true) / denom


# ---------- Temporal detection ----------

def start_time_mae(
    pred_starts: Sequence[float],
    true_starts: Sequence[float],
) -> float:
    errors = [abs(p - t) for p, t in zip(pred_starts, true_starts)]
    return float(np.mean(errors)) if errors else float("nan")


def early_detection_score(
    pred_start: float,
    true_start: float,
    event_end: float,
) -> float:
    """EDS ∈ [0,1]: 1 = detect exactly at start, 0 = detect at end or later."""
    duration = max(event_end - true_start, 1e-6)
    lag = max(pred_start - true_start, 0.0)
    return max(0.0, 1.0 - lag / duration)


# ---------- Calibration ----------

def expected_calibration_error(
    confidences: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> float:
    confs = np.array(confidences)
    correct = np.array(labels)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confs)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (confs >= lo) & (confs < hi)
        if i == n_bins - 1:
            mask |= confs == 1.0
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confs[mask].mean()
        ece += mask.sum() / n * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(
    confidences: Sequence[float],
    labels: Sequence[int],
) -> float:
    confs = np.array(confidences)
    correct = np.array(labels)
    return float(np.mean((confs - correct) ** 2))


# ---------- Aggregate report ----------

def compute_all(
    y_true_binary: List[int],
    y_pred_binary: List[int],
    y_true_class: List[str],
    y_pred_class: List[str],
    y_score: List[float],
    confidences: List[float],
    mistake_classes: List[str],
) -> dict:
    # The raw score has positive-class polarity, not confidence in the selected
    # decision. Calibration therefore compares scores with mistake labels and
    # must not depend on gating or smoothing. Historical decision_v2 artifacts
    # used decision correctness; corrected analyses are versioned separately.
    return {
        "binary_f1": binary_f1(y_true_binary, y_pred_binary),
        "macro_f1": macro_f1(y_true_class, y_pred_class, mistake_classes),
        "per_class_f1": per_class_f1(y_true_class, y_pred_class, mistake_classes),
        "support": per_class_support(y_true_class, mistake_classes),
        "auc_roc": auc_roc(y_true_binary, y_score),
        "ece": expected_calibration_error(confidences, y_true_binary),
        "brier": brier_score(confidences, y_true_binary),
        "precision": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
    }
