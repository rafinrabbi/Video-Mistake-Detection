"""Strictly causal (trailing) re-gate: does the centred median filter matter?

The per-window model is causal -- frames, step pointer and judgement history all
come from the past. The decision layer's median filter, however, is CENTRED
(``decision.postprocess.median_smooth`` pads by k//2 on both sides), so with
k=5 a window's binary decision reads two future windows: 8 s of lookahead added
after the model has spoken.

This script measures what that lookahead is worth by replaying the SAME saved
predictions through a trailing filter (window i sees i-k+1..i only), leaving
``eval/decision_layer.py`` and every saved row untouched.

    python scripts/cc4d_causal_smoothing.py \
        --preds outputs/decision_v2/lora_test_preds.json --split test \
        --params outputs/decision_v2/decision_params_lora.json \
        --out outputs/decision_v2/causal_smoothing.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vmd.decision.postprocess import detect_start_of_deviation  # noqa: E402
from vmd.eval.decision_layer import (  # noqa: E402
    FIRED_EPS, gate_clip, gold_events, load_preds, localization_metrics,
)
from vmd.eval.metrics import auc_roc, compute_all, macro_f1  # noqa: E402
from vmd.data.mistake_classes import MISTAKE_CLASSES  # noqa: E402


def trailing_median(scores, k):
    """Median over the k most recent windows, inclusive of the current one.

    The head is padded with the first value, mirroring the edge padding the
    centred filter uses, so the two differ only in which side they look at.
    """
    arr = np.asarray(scores, dtype=float)
    padded = np.pad(arr, (k - 1, 0), mode="edge")
    return [float(np.median(padded[i:i + k])) for i in range(len(arr))]


def evaluate(preds_by_clip, gold, tau, smooth_k, min_consecutive, gate_mode,
             causal):
    yt_b, yp_b, yt_c, yp_c, ys, conf = [], [], [], [], [], []
    loc = []
    for clip_id, records in preds_by_clip.items():
        g = gate_clip(records, tau, smooth_k, gate_mode)
        smoothed = (trailing_median(g["gated"], smooth_k) if causal
                    else g["smoothed"])
        yt_b.extend(int(r["true_label"]) for r in records)
        yp_b.extend(int(s > 0.0) for s in smoothed)
        yt_c.extend(r["true_class"] for r in records)
        yp_c.extend(g["pred_cls"])
        ys.extend(g["gated"])
        conf.extend(float(r["confidence"]) for r in records)
        event = gold.get(clip_id)
        if event is not None and g["starts"]:
            gs, ge = event
            ps = detect_start_of_deviation(smoothed, g["starts"], FIRED_EPS,
                                           min_consecutive)
            loc.append((gs, ps, ge))
    out = compute_all(yt_b, yp_b, yt_c, yp_c, ys, conf, list(MISTAKE_CLASSES))
    out.update(localization_metrics(loc))
    out["auc_raw"] = auc_roc(yt_b, conf)
    out["macro_f1_excl_missing_step"] = macro_f1(
        yt_c, yp_c, [c for c in MISTAKE_CLASSES if c != "missing_step"])
    out["n_windows"] = len(yt_b)
    out["n_clips"] = len(preds_by_clip)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True,
                    help="one or more saved *_preds.json (label=path form ok)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--raw_dir", default="data/raw/captaincook4d")
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    p = json.loads(Path(a.params).read_text())
    tau, k = float(p["tau"]), int(p["smooth_k"])
    m, mode = int(p["min_consecutive"]), p.get("gate_mode", "and")
    gold = gold_events(Path(a.raw_dir), a.split)

    result = {"params": {"tau": tau, "smooth_k": k, "min_consecutive": m,
                         "gate_mode": mode},
              "note": ("centred = the reported decision layer (looks k//2 "
                       "windows ahead); trailing = strictly causal, same k."),
              "arms": {}}
    for spec in a.preds:
        label, _, path = spec.partition("=")
        if not path:
            label, path = Path(spec).stem, spec
        by_clip = load_preds(Path(path))
        cen = evaluate(by_clip, gold, tau, k, m, mode, causal=False)
        cau = evaluate(by_clip, gold, tau, k, m, mode, causal=True)
        keys = ["binary_f1", "macro_f1", "precision", "recall", "eds",
                "start_mae", "n_loc_detected"]
        result["arms"][label] = {
            "centred": {kk: cen[kk] for kk in keys},
            "trailing": {kk: cau[kk] for kk in keys},
            "delta": {kk: cau[kk] - cen[kk] for kk in keys},
        }
        print(f"{label}: F1 {cen['binary_f1']:.3f} (centred) -> "
              f"{cau['binary_f1']:.3f} (trailing), "
              f"EDS {cen['eds']:.3f} -> {cau['eds']:.3f}", flush=True)

    Path(a.out).write_text(json.dumps(result, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
