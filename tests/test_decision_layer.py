"""Regression net for the decision layer (gate → smooth → metrics).

Hand-computed fixture pinning the exact semantics the historical eval used:
the AND-gate, the UNsmoothed typed-class stream, post-smoothing binary
decisions, persistence localization, and tolerance of legacy record keys.
"""
import math

import pytest

from vmd.eval.decision_layer import (
    evaluate_records, gate_clip, load_preds, localization_metrics,
)


def _rec(start, dev, conf, cls, true_label, true_cls, **extra):
    r = {
        "clip_id": "unused", "start_sec": float(start), "end_sec": float(start) + 4.0,
        "deviation": dev, "confidence": conf, "mistake_class": cls,
        "true_label": true_label, "true_class": true_cls,
    }
    r.update(extra)
    return r


# Clip A: gate + smoothing + persistence behaviour.
#   gated (tau .5, and):  [0.9, 0, 0, 0.7, 0.6]
#   median k=3 (edge pad): [0.9, 0, 0, 0.6, 0.6] -> pred_bin [1,0,0,1,1]
CLIP_A = [
    _rec(0,  True,  0.9, "order_error",     1, "order_error"),
    _rec(4,  True,  0.4, "order_error",     1, "order_error"),      # below tau
    _rec(8,  False, 0.8, "technique_error", 0, "no_mistake"),       # AND blocks
    _rec(12, True,  0.7, "timing_error",    0, "no_mistake"),
    # legacy extra keys must be tolerated
    _rec(16, True,  0.6, "order_error",     1, "order_error",
         features=[0.5, 1.0, 0.2, 0.2, 0.0], tau_adaptive=0.55, tau_fixed=0.5),
]

# Clip B: isolated fired window — smoothing zeroes the binary stream but the
# typed stream keeps the class (the un-smoothed asymmetry).
CLIP_B = [
    _rec(0, False, 0.1, "no_mistake",        0, "no_mistake"),
    _rec(4, True,  0.9, "measurement_error", 0, "no_mistake"),
    _rec(8, False, 0.2, "no_mistake",        0, "no_mistake"),
]

PREDS = {"clip_a": CLIP_A, "clip_b": CLIP_B}
GOLD = {"clip_a": (0.0, 16.0), "clip_b": None}


def test_gate_clip_and_mode():
    g = gate_clip(CLIP_A, tau=0.5, smooth_k=3, gate_mode="and")
    assert g["gated"] == [0.9, 0.0, 0.0, 0.7, 0.6]
    assert g["smoothed"] == [0.9, 0.0, 0.0, 0.6, 0.6]
    assert g["pred_bin"] == [1, 0, 0, 1, 1]
    assert g["pred_cls"] == [
        "order_error", "no_mistake", "no_mistake", "timing_error", "order_error",
    ]


def test_gate_clip_conf_only_mode():
    g = gate_clip(CLIP_A, tau=0.5, smooth_k=3, gate_mode="conf_only")
    # window 2 (deviation=False, conf .8) now fires
    assert g["gated"] == [0.9, 0.0, 0.8, 0.7, 0.6]
    assert g["pred_cls"][2] == "technique_error"


def test_typed_stream_is_not_smoothed():
    g = gate_clip(CLIP_B, tau=0.5, smooth_k=3, gate_mode="and")
    assert g["pred_bin"] == [0, 0, 0]          # smoothing kills the isolated fire
    assert g["pred_cls"][1] == "measurement_error"  # typed keeps it


def test_evaluate_records_binary_and_localization():
    m = evaluate_records(PREDS, GOLD, tau=0.5, smooth_k=3, min_consecutive=2)
    # y_true [1,1,0,0,1, 0,0,0]; pred [1,0,0,1,1, 0,0,0] -> TP2 FP1 FN1
    assert m["binary_f1"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    # clip_a: first >=2-consecutive fired run is windows 3,4 -> start 12.0
    # eds = 1 - (12-0)/16 = 0.25 ; clip_b has no gold event
    assert m["n_loc_events"] == 1
    assert m["n_loc_detected"] == 1
    assert m["eds"] == pytest.approx(0.25)
    assert m["start_mae"] == pytest.approx(12.0)
    assert m["n_windows"] == 8
    assert m["n_clips"] == 2


def test_min_consecutive_persistence():
    m = evaluate_records(PREDS, GOLD, tau=0.5, smooth_k=3, min_consecutive=3)
    # longest fired run in clip_a's smoothed stream is 2 -> no detection
    assert m["n_loc_detected"] == 0
    assert m["eds"] == 0.0
    assert math.isnan(m["start_mae"])


def test_extra_metrics_present():
    m = evaluate_records(PREDS, GOLD, tau=0.5, smooth_k=3)
    assert 0.0 <= m["auc_raw"] <= 1.0
    assert "macro_f1_excl_missing_step" in m
    assert set(m["per_class_f1"]) == set(m["support"])


def test_localization_metrics_empty():
    m = localization_metrics([])
    assert m["n_loc_events"] == 0 and math.isnan(m["eds"])


def test_load_preds_groups_in_order(tmp_path):
    import json
    records = [dict(r, clip_id=cid) for cid, rows in PREDS.items() for r in rows]
    p = tmp_path / "preds.json"
    p.write_text(json.dumps(records))
    by_clip = load_preds(p)
    assert list(by_clip) == ["clip_a", "clip_b"]
    assert [r["start_sec"] for r in by_clip["clip_a"]] == [0, 4, 8, 12, 16]
