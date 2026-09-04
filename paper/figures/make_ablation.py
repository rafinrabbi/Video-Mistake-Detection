#!/usr/bin/env python
"""Figure 3: where the effect comes from.

Two panels sharing one story. Left, binary F1 with clip-level bootstrap
intervals: every graph variant sits inside every other one's interval, and the
whole cluster sits far above zero-shot -- the distance is fine-tuning, not the
graph. Right, the same rows on early detection, where the graph variants
separate and the intervals of the paired differences exclude zero.

Putting the two side by side is the point: the paper's claim is not that the
graph does nothing, but that it moves one axis and not the other.

    python paper/figures/make_ablation.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs/decision_v2"

# Okabe-Ito; also separable in greyscale by bar order and the reference rules.
C_MAIN, C_ABL, C_BASE, C_ZS = "#0072B2", "#56B4E9", "#009E73", "#999999"

ROWS = [
    ("lora_tuned",        "graph-conditioned (R4)",              C_MAIN),
    ("lora_nograph",      "graph off at inference (R5)",         C_ABL),
    ("lora_nogr_trained", "graph-free train + test (R6)",        C_ABL),
    ("lora_fixedptr",     "corrected step pointer (R11)",        C_ABL),
    ("mistsense_tuned",   "MistSense, protocol-matched",         C_BASE),
    ("zs_tuned",          "zero-shot",                           C_ZS),
    ("always",            "fire on every window (trivial)",      "#222222"),
]


def load(label):
    if label == "always":
        a = json.loads((D / "row_trivial_baselines.json").read_text())["always_fire"]
        pt = {"binary_f1": a["binary_f1"], "eds": a["eds"]}
        return pt, {k: [v, v] for k, v in pt.items()}     # deterministic: no interval
    d = json.loads((D / f"bootstrap_{label}.json").read_text())
    return d["point"], d["ci95"]


def panel(ax, metric, title, xlabel, rows=None, xmax=1.0):
    rows = rows or ROWS
    ys = range(len(rows))
    for y, (lab, name, colour) in zip(ys, rows):
        pt, ci = load(lab)
        v = pt[metric]
        lo, hi = ci[metric]
        ax.barh(y, v, height=0.6, color=colour, zorder=2)
        ax.plot([lo, hi], [y, y], color="#222", lw=1.1, zorder=3)
        for b in (lo, hi):
            ax.plot([b, b], [y - 0.13, y + 0.13], color="#222", lw=1.1, zorder=3)
        # label past the upper CI bound, never on top of the whisker
        ax.text(hi + 0.022 * xmax, y, f"{v:.3f}", va="center", fontsize=7.5, zorder=4)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([n for _, n, _ in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(title, fontsize=8, loc="left", pad=4)
    ax.tick_params(labelsize=7.5)
    ax.grid(axis="x", color="#E8E8E8", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    panel(axes[0], "binary_f1", "(a) detection: every arm within noise of trivial",
          "binary F1", xmax=0.78)
    # Only the three conditioning variants. MistSense (0.991) and zero-shot
    # (0.985) top this metric because they fire on almost every window, so
    # plotting them beside the ablation would invite exactly the misreading the
    # text warns against: EDS is only interpretable alongside precision.
    panel(axes[1], "eds", "(b) earliness, uncontrolled for firing rate",
          "early detection score", rows=ROWS[:4] + ROWS[-1:], xmax=1.0)
    axes[0].set_xlim(0, 0.78)
    axes[1].set_xlim(0, 1.0)
    axes[1].set_yticklabels([])
    fig.tight_layout()
    out = ROOT / "paper/figures/ablation.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
