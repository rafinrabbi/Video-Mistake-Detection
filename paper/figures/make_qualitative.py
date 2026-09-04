#!/usr/bin/env python
"""Figure 2: how one recording is judged over time.

Two panels, chosen to show both sides of the operating point:
  (a) 1_49  -- a recording dense with real errors; the system tracks them.
  (b) 1_25  -- an error-free recording the system fires on repeatedly. This is
      the precision problem of Section 6.7 made concrete, and it is included
      deliberately rather than showing only the favourable panel.

The walked step pointer is not stored in the predictions, so it is reconstructed
here with the same rule the online evaluator applies: advance by one on every
no-deviation judgement, clamped at the final step. That makes the stall visible
-- the pointer stops climbing exactly while the model believes a deviation is in
progress.

    python paper/figures/make_qualitative.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vmd.data.captaincook4d import CaptainCook4DDataset       # noqa: E402
from vmd.graph.schema import ProceduralGraph                   # noqa: E402

TAU, N_STEPS = 0.45, 14
# Colourblind-safe (Okabe-Ito) and distinguishable in greyscale by linestyle.
C_SCORE, C_PTR, C_GOLD, C_TAU = "#0072B2", "#D55E00", "#999999", "#333333"


def pointer_track(records):
    """Replay the online step pointer from the deviation flags."""
    idx, out = 0, []
    for r in records:
        out.append(idx)
        if not r["deviation"] and idx < N_STEPS - 1:
            idx += 1
    return out


def main():
    preds = json.loads((ROOT / "outputs/decision_v2/qualitative_preds.json").read_text())
    ds = CaptainCook4DDataset(str(ROOT / "data/raw/captaincook4d"), "test")
    gold = {c.clip_id: [(a.start_sec, a.end_sec) for a in c.annotations if a.start_sec >= 0]
            for c in ds.clips}

    panels = [("1_49", "(a) recording with six ordering and preparation errors"),
              ("1_25", "(b) error-free recording: every firing is a false alarm")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.1, 4.3), sharex=False)

    for ax, (cid, title) in zip(axes, panels):
        rec = sorted([r for r in preds if r["clip_id"] == cid], key=lambda r: r["start_sec"])
        t = [r["start_sec"] for r in rec]
        conf = [r["confidence"] for r in rec]

        for k, (s, e) in enumerate(gold.get(cid, [])):
            ax.axvspan(s, e, color=C_GOLD, alpha=0.28, lw=0,
                       label="gold error span" if k == 0 else None)

        ax.plot(t, conf, color=C_SCORE, lw=1.1, label="decision-token score")
        ax.axhline(TAU, color=C_TAU, ls="--", lw=0.9, label=rf"$\tau={TAU}$")

        fired = [r["deviation"] and r["confidence"] >= TAU for r in rec]
        if any(fired):
            onset = t[fired.index(True)]
            ax.axvline(onset, color=C_SCORE, ls=":", lw=1.3,
                       label="predicted onset")
        if gold.get(cid):
            ax.axvline(min(s for s, _ in gold[cid]), color="#B00000", ls="-.", lw=1.1,
                       label="gold onset")

        ax.set_ylim(0, 1.02)
        ax.set_ylabel("score", fontsize=8)
        ax.set_title(title, fontsize=8.5, loc="left", pad=3)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="y", color="#E5E5E5", lw=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        ax2 = ax.twinx()
        ax2.step(t, pointer_track(rec), where="post", color=C_PTR, lw=1.0,
                 alpha=0.85, label=r"walked step pointer $\sigma_i$")
        ax2.set_ylim(0, N_STEPS)
        ax2.set_ylabel(r"step $\sigma_i$", fontsize=8, color=C_PTR)
        ax2.tick_params(labelsize=7.5, colors=C_PTR)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_color(C_PTR)

    axes[-1].set_xlabel("time in recording (s)", fontsize=8)

    handles = [
        Line2D([], [], color=C_SCORE, lw=1.1, label="decision-token score"),
        Line2D([], [], color=C_PTR, lw=1.0, label=r"walked step pointer $\sigma_i$"),
        Patch(facecolor=C_GOLD, alpha=0.28, label="gold error span"),
        Line2D([], [], color=C_TAU, ls="--", lw=0.9, label=rf"threshold $\tau={TAU}$"),
        Line2D([], [], color=C_SCORE, ls=":", lw=1.3, label="predicted onset"),
        Line2D([], [], color="#B00000", ls="-.", lw=1.1, label="gold onset"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.045))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = ROOT / "paper/figures/qualitative.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
