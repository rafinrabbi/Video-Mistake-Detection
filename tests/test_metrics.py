import pytest
from vmd.eval.metrics import (
    binary_f1, macro_f1, per_class_f1, per_class_support, compute_all,
    edit_distance, normalised_edit_distance,
    expected_calibration_error, brier_score, early_detection_score, auc_roc,
    MISTAKE_CLASSES,
)


def test_binary_f1_perfect():
    assert binary_f1([1, 0, 1], [1, 0, 1]) == pytest.approx(1.0)


def test_binary_f1_all_wrong():
    assert binary_f1([1, 1, 1], [0, 0, 0]) == pytest.approx(0.0)


def test_macro_f1_perfect():
    # Pass only the active labels to get correct 1.0; full MISTAKE_CLASSES with missing classes gives <1.0
    labels = ["no_mistake", "order_error"]
    active = ["no_mistake", "order_error"]
    assert macro_f1(labels, labels, active) == pytest.approx(1.0)


def test_per_class_f1_keys_and_never_predicted_zero():
    # A class that never appears in preds must yield 0.0, not be dropped.
    y_true = ["no_mistake", "order_error", "timing_error"]
    y_pred = ["no_mistake", "order_error", "no_mistake"]
    pcf = per_class_f1(y_true, y_pred, MISTAKE_CLASSES)
    assert set(pcf.keys()) == set(MISTAKE_CLASSES)
    assert pcf["order_error"] == pytest.approx(1.0)
    assert pcf["timing_error"] == pytest.approx(0.0)        # present in truth, never predicted
    assert pcf["temperature_error"] == pytest.approx(0.0)   # absent everywhere


def test_per_class_support_counts():
    y_true = ["no_mistake", "no_mistake", "order_error"]
    sup = per_class_support(y_true, MISTAKE_CLASSES)
    assert sup["no_mistake"] == 2
    assert sup["order_error"] == 1
    assert sup["timing_error"] == 0


def test_compute_all_exposes_per_class_and_support():
    n = 6
    y_true_bin = [0, 1, 0, 1, 0, 1]
    y_pred_bin = [0, 1, 0, 1, 0, 1]
    y_true_cls = ["no_mistake", "order_error"] * 3
    y_pred_cls = ["no_mistake", "order_error"] * 3
    m = compute_all(y_true_bin, y_pred_bin, y_true_cls, y_pred_cls,
                    [0.0, 0.9] * 3, [0.8] * n, MISTAKE_CLASSES)
    assert set(m["per_class_f1"].keys()) == set(MISTAKE_CLASSES)
    assert set(m["support"].keys()) == set(MISTAKE_CLASSES)
    assert m["binary_f1"] == pytest.approx(1.0)


def test_ece_brier_score_correctness_not_label():
    # Confident + CORRECT predictions (incl. correct negatives) must be well
    # calibrated. Two correct negatives at conf 0.9: correctness=1, so Brier is
    # (0.9-1)^2=0.01 (low). The old label-as-correctness bug gave (0.9-0)^2=0.81.
    y_true_bin = [0, 0, 1, 1]
    y_pred_bin = [0, 0, 1, 1]
    y_cls = ["no_mistake", "no_mistake", "order_error", "order_error"]
    conf = [0.9, 0.9, 0.9, 0.9]
    m = compute_all(y_true_bin, y_pred_bin, y_cls, y_cls,
                    [0.0, 0.0, 0.9, 0.9], conf, MISTAKE_CLASSES)
    assert m["brier"] == pytest.approx(0.01, abs=1e-6)
    assert m["ece"] < 0.15


def test_edit_distance_identical():
    assert edit_distance(["a", "b", "c"], ["a", "b", "c"]) == 0


def test_edit_distance_insert():
    assert edit_distance(["a", "c"], ["a", "b", "c"]) == 1


def test_edit_distance_delete():
    assert edit_distance(["a", "b", "c"], ["a", "c"]) == 1


def test_normalised_ed():
    ned = normalised_edit_distance(["a"], ["a", "b", "c"])
    assert 0 <= ned <= 1


def test_ece_perfect_calibration():
    confs = [0.9, 0.9, 0.1, 0.1]
    correct = [1, 1, 0, 0]
    ece = expected_calibration_error(confs, correct)
    assert ece < 0.1


def test_brier_perfect():
    assert brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)


def test_brier_worst():
    assert brier_score([0.0, 1.0], [1, 0]) == pytest.approx(1.0)


def test_early_detection_score_at_start():
    assert early_detection_score(5.0, 5.0, 15.0) == pytest.approx(1.0)


def test_early_detection_score_at_end():
    assert early_detection_score(15.0, 5.0, 15.0) == pytest.approx(0.0)


def test_early_detection_score_midpoint():
    eds = early_detection_score(10.0, 5.0, 15.0)
    assert pytest.approx(0.5) == eds


def test_auc_roc_perfect():
    assert auc_roc([0, 1], [0.0, 1.0]) == pytest.approx(1.0)
