#!/usr/bin/env python
"""Figure: the graph buys earliness. Per-event onset error for the three
conditioning variants under R4's frozen decision layer. Same events are found;
the graph-conditioned model finds them sooner.

    python paper/figures/make_onset.py
"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from vmd.eval.decision_layer import load_preds, gold_events, gate_clip, FIRED_EPS
from vmd.decision.postprocess import detect_start_of_deviation

ARMS = [("outputs/full_eval_8class/lora_test_preds.json", "row_lora_tuned_test.json", "graph-conditioned (R4)", "#0072B2"),
        ("outputs/decision_v2/lora_nograph_test_preds.json", "row_lora_nograph_test.json", "graph off at inference (R5)", "#56B4E9"),
        ("outputs/decision_v2/lora_nogr_trained_test_preds.json", "row_lora_nogr_trained_test.json", "graph-free train + test (R6)", "#8CC5E3")]

def onset_errors(preds_path, gold, params):
    by = load_preds(ROOT / preds_path); errs = []
    for cid, recs in by.items():
        ev = gold.get(cid)
        if ev is None or not recs: continue
        g = gate_clip(recs, params["tau"], params["smooth_k"], params["gate_mode"])
        ps = detect_start_of_deviation(g["smoothed"], g["starts"], FIRED_EPS, params["min_consecutive"])
        if ps is not None: errs.append(ps - ev[0])
    return np.array(errs)

def main():
    params = json.loads((ROOT / "outputs/decision_v2/decision_params_lora.json").read_text())
    gold = gold_events(ROOT / "data/raw/captaincook4d", "test")
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    for i, (path, row, name, col) in enumerate(ARMS):
        e = onset_errors(path, gold, params); y = np.full_like(e, i, dtype=float)
        eds = json.loads((ROOT / "outputs/decision_v2" / row).read_text())["metrics"]["eds"]
        late = int((e > 60).sum())
        rng = np.random.default_rng(0); y = y + rng.uniform(-0.18, 0.18, len(e))
        ax.scatter(e, y, s=14, color=col, alpha=0.75, lw=0, zorder=3)
        med = np.median(e)
        ax.plot([med, med], [i - 0.32, i + 0.32], color="#222", lw=1.6, zorder=4)
        ax.text(-640, i - 0.36, f"EDS {eds:.3f}   median {med:+.0f} s   {late} events later than 60 s", ha="left", fontsize=7)
    ax.axvline(0, color="#222", ls=":", lw=0.9); ax.set_xlim(-650, 650)
    ax.set_yticks(range(len(ARMS))); ax.set_yticklabels([n for _, _, n, _ in ARMS], fontsize=8)
    ax.set_xlabel("predicted onset minus gold onset (s); negative = detected early", fontsize=8)
    ax.tick_params(labelsize=7.5); ax.invert_yaxis(); ax.grid(axis="x", color="#EBEBEB", lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); out = ROOT / "paper/figures/onset.pdf"; fig.savefig(out, bbox_inches="tight"); print("wrote", out)

if __name__ == "__main__": main()
