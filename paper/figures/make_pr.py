#!/usr/bin/env python
"""Figure: precision-recall from the raw, ungated per-window scores, with each
system's validation-selected operating point marked. Shows the trade-off the
text describes: the baseline buys F1 with recall, ours with precision.

    python paper/figures/make_pr.py
"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
SYS = [("outputs/full_eval_8class/lora_test_preds.json", "row_lora_tuned_test.json", "graph-conditioned (R4)", "#0072B2"),
       ("outputs/decision_v2/lora_nogr_trained_test_preds.json", "row_lora_nogr_trained_test.json", "graph-free (R6)", "#8CC5E3"),
       ("outputs/decision_v2/mistsense_test_preds.json", "row_mistsense_tuned_test.json", "MistSense, matched", "#009E73"),
       ("outputs/full_eval_8class/zs_test_preds.json", "row_zs_tuned_test.json", "zero-shot", "#999999")]
def main():
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    for preds, row, name, col in SYS:
        r = json.loads((ROOT / preds).read_text())
        y = np.array([x["true_label"] for x in r]); s = np.array([x["confidence"] for x in r])
        p, rc, _ = precision_recall_curve(y, s)
        m = json.loads((ROOT / "outputs/decision_v2" / row).read_text())["metrics"]
        ax.plot(rc, p, color=col, lw=1.3, label=name)
        ax.scatter([m["recall"]], [m["precision"]], color=col, s=42, zorder=5, edgecolor="#222", lw=0.7)
    ax.axhline(y.mean(), color="#222", ls=":", lw=0.8); ax.text(0.02, y.mean() + 0.012, "positive rate", ha="left", fontsize=6.5)
    ax.set_xlabel("recall", fontsize=8); ax.set_ylabel("precision", fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right"); ax.tick_params(labelsize=7.5)
    ax.grid(color="#EBEBEB", lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); out = ROOT / "paper/figures/pr.pdf"; fig.savefig(out, bbox_inches="tight"); print("wrote", out)
if __name__ == "__main__": main()
