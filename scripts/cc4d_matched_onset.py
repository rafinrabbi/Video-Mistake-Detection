#!/usr/bin/env python
"""Earliness with detection controlled.

The early-detection score rewards firing: a detector that fires on every window
scores 1.0. So an EDS gap between two arms is confounded with their firing
rates. This computes onset error only on events BOTH arms detected, and
bootstraps the median paired difference. CPU-only; writes matched_onset.json.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
from vmd.eval.decision_layer import load_preds, gold_events, gate_clip, FIRED_EPS
from vmd.decision.postprocess import detect_start_of_deviation
D = ROOT / "outputs/decision_v2"
ARMS = {"R4": ROOT / "outputs/full_eval_8class/lora_test_preds.json", "R6": D / "lora_nogr_trained_test_preds.json",
        "FIXED": D / "lora_fixedptr_test_preds.json", "ALWAYS": D / "alwaysfire_test_preds.json"}
def main():
    P = json.loads((D / "decision_params_lora.json").read_text()); gold = gold_events(ROOT / "data/raw/captaincook4d", "test")
    def onsets(path):
        out = {}
        for cid, recs in load_preds(path).items():
            ev = gold.get(cid)
            if ev is None or not recs: continue
            g = gate_clip(recs, P["tau"], P["smooth_k"], P["gate_mode"])
            ps = detect_start_of_deviation(g["smoothed"], g["starts"], FIRED_EPS, P["min_consecutive"])
            if ps is not None: out[cid] = ps - ev[0]
        return out
    O = {k: onsets(v) for k, v in ARMS.items() if v.exists()}
    rng = np.random.default_rng(0); res = {}
    for a, b in (("R4", "R6"), ("R4", "ALWAYS"), ("FIXED", "R6"), ("R6", "ALWAYS")):
        if a not in O or b not in O: continue
        common = sorted(set(O[a]) & set(O[b])); d = np.array([O[a][c] - O[b][c] for c in common])
        boots = [float(np.median(rng.choice(d, len(d)))) for _ in range(1000)]
        res[f"{a}_vs_{b}"] = {"n": len(common), "median_diff_s": float(np.median(d)),
                              "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
                              "a_earlier": int((d < 0).sum()), "a_later": int((d > 0).sum())}
    (D / "matched_onset.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=1))
if __name__ == "__main__": main()
