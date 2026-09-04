#!/usr/bin/env python
"""Figure: where the decision signal comes from. Distribution of the
decision-token confidence on mistake vs non-mistake test windows, against the
verbalised confidence field, which is a single constant.

    python paper/figures/make_scores.py
"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
def main():
    r = json.loads((ROOT / "outputs/full_eval_8class/lora_test_preds.json").read_text())
    pos = np.array([x["confidence"] for x in r if x["true_label"] == 1])
    neg = np.array([x["confidence"] for x in r if x["true_label"] == 0])
    q = json.loads((ROOT / "outputs/decision_v2/qualitative_preds.json").read_text())
    verb = sorted({x.get("verbalized_confidence") for x in q if x.get("verbalized_confidence") is not None})
    tau = json.loads((ROOT / "outputs/decision_v2/decision_params_lora.json").read_text())["tau"]
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.hist(neg, bins, density=True, color="#999999", alpha=0.75, label=f"no-mistake windows (n={len(neg):,})")
    ax.hist(pos, bins, density=True, color="#0072B2", alpha=0.65, label=f"mistake windows (n={len(pos):,})")
    ax.axvline(tau, color="#222", ls="--", lw=1.0, label=rf"selected $\tau={tau}$")
    for v in verb: ax.axvline(v, color="#D55E00", lw=2.2, label=f"verbalised confidence: constant {v}")
    ax.set_xlabel("decision-token confidence $s_i$", fontsize=8); ax.set_ylabel("density", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="upper center"); ax.tick_params(labelsize=7.5)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); out = ROOT / "paper/figures/scores.pdf"; fig.savefig(out, bbox_inches="tight"); print("wrote", out)
if __name__ == "__main__": main()
