"""Weighted PR/AP/AUC agree with scikit-learn; decision-layer sweep is exact."""
import numpy as np
import pytest
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)

from vmd.eval import curves

Y = np.array([0, 1, 1, 0, 1, 0, 1, 0])
S = np.array([0.1, 0.5, 0.5, 0.2, 1.0, 1.0, 0.7, 0.3])


@pytest.mark.parametrize("w", [None, [1, 2, 0, 1, 3, 1, 1, 2]])
def test_ap_and_auc_match_sklearn(w):
    ww = None if w is None else np.array(w, dtype=float)
    assert curves.average_precision(Y, S, ww) == pytest.approx(
        average_precision_score(Y, S, sample_weight=ww))
    assert curves.roc_auc(Y, S, ww) == pytest.approx(roc_auc_score(Y, S, sample_weight=ww))


def test_pr_curve_points_match_sklearn():
    p, r, thr = curves.pr_curve(Y, S)
    sp, sr, sthr = precision_recall_curve(Y, S)
    # sklearn appends the (1, 0) point and orders thresholds ascending
    assert set(np.round(thr, 6)) == set(np.round(sthr, 6))
    for t in thr:
        k = np.flatnonzero(np.isclose(sthr, t))[0]
        j = np.flatnonzero(np.isclose(thr, t))[0]
        assert p[j] == pytest.approx(sp[k]) and r[j] == pytest.approx(sr[k])


def test_interpolated_precision_is_monotone_and_reaches_zero():
    p, r, _ = curves.pr_curve(Y, S)
    ip = curves.interpolated_precision(p, r, np.linspace(0, 1, 11))
    assert np.all(np.diff(ip) <= 1e-12)
    assert ip[0] == pytest.approx(p.max()) and ip[-1] == pytest.approx(p[r >= 1].max())


def test_precision_at_recall_and_recall_at_fpr():
    assert curves.precision_at_recall(Y, S, 1.0) == pytest.approx(4 / 5)   # cutoff 0.5: tp 4, fp 1
    tpr, thr = curves.recall_at_fpr(Y, S, 0.25)
    assert thr == pytest.approx(0.5) and tpr == pytest.approx(1.0)
    tpr0, thr0 = curves.recall_at_fpr(Y, S, 0.0)
    assert tpr0 == 0.0 and thr0 == float("inf")   # every cutoff admits the tied 1.0 negative


def test_constant_score_ap_is_prevalence():
    y = np.array([1, 0, 0, 0, 1])
    assert curves.average_precision(y, np.ones(5)) == pytest.approx(0.4)
    assert curves.roc_auc(y, np.ones(5)) == pytest.approx(0.5)


def test_decision_layer_sweep_matches_gate_clip():
    recs = [{"clip_id": "c", "start_sec": 4.0 * i, "end_sec": 4.0 * i + 4, "true_label": y,
             "true_class": "order_error" if y else "no_mistake", "deviation": True,
             "confidence": s, "mistake_class": "order_error"} for i, (y, s) in enumerate(zip(Y, S))]
    rows = curves.decision_layer_sweep({"c": recs}, smooth_k=1, gate_mode="conf_only", taus=[0.5])
    assert rows[0]["recall"] == pytest.approx(1.0) and rows[0]["precision"] == pytest.approx(4 / 5)
    assert rows[0]["firing_rate"] == pytest.approx(5 / 8)
