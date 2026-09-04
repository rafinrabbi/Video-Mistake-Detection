"""Calibrate the token-probability confidence on VAL, evaluate on TEST. CPU-only.

Fits temperature scaling (single T on the logit) and Platt scaling (a,b) against
the binary mistake label, picks the better by val NLL, and reports label-ECE and
label-Brier on test, raw vs calibrated. Both maps are monotone, so calibration
cannot change any thresholded operating point — it is a reporting-layer fix,
fitted on val only.

  python scripts/cc4d_calibrate.py \
      --val_preds outputs/full_eval_8class/lora_val_preds.json \
      --test_preds outputs/full_eval_8class/lora_test_preds.json \
      --arm lora --out_dir outputs/decision_v2
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from vmd.eval.metrics import brier_score, expected_calibration_error

EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _nll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _load(path):
    records = json.loads(Path(path).read_text())
    conf = np.array([float(r["confidence"]) for r in records])
    y = np.array([int(r["true_label"]) for r in records])
    return conf, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_preds", required=True)
    ap.add_argument("--test_preds", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out_dir", default=str(ROOT / "outputs/decision_v2"))
    args = ap.parse_args()

    conf_val, y_val = _load(args.val_preds)
    conf_test, y_test = _load(args.test_preds)
    z_val, z_test = _logit(conf_val), _logit(conf_test)

    # Temperature scaling: p = sigmoid(z / T), T > 0 by 1-D grid + refinement.
    def temp_nll(T):
        return _nll(_sigmoid(z_val / T), y_val)
    grid = np.exp(np.linspace(np.log(0.05), np.log(50.0), 400))
    T = float(grid[int(np.argmin([temp_nll(t) for t in grid]))])

    # Platt scaling: p = sigmoid(a*z + b), 1-feature logistic regression.
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(z_val.reshape(-1, 1), y_val)
    a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])

    cands = {
        "temperature": (lambda z: _sigmoid(z / T), {"T": T}),
        "platt": (lambda z: _sigmoid(a * z + b), {"a": a, "b": b}),
    }
    val_nlls = {name: _nll(fn(z_val), y_val) for name, (fn, _) in cands.items()}
    chosen = min(val_nlls, key=val_nlls.get)
    fn, fit_params = cands[chosen]

    p_test_cal = fn(z_test)
    out = {
        "arm": args.arm,
        "method": chosen,
        "params": fit_params,
        "val_nll": {"raw": _nll(conf_val, y_val), **val_nlls},
        "test": {
            "ece_label_raw": expected_calibration_error(conf_test, y_test),
            "ece_label_calibrated": expected_calibration_error(p_test_cal, y_test),
            "brier_label_raw": brier_score(conf_test, y_test),
            "brier_label_calibrated": brier_score(p_test_cal, y_test),
            "nll_raw": _nll(conf_test, y_test),
            "nll_calibrated": _nll(p_test_cal, y_test),
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"calibration_{args.arm}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
