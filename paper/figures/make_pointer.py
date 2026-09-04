#!/usr/bin/env python
"""Figure: the step pointer saturates, and the fix does not.

(a) One recording. Step index the model is conditioned on, over time, under the
    original per-window rule (races to the top), the rate-limited rule
    (tracks), and the gold segmentation (truth). Same deviation flags drive both
    causal rules, so the only difference is the rule.
(b) Every test recording. ECDF of the fraction of windows conditioned on the
    final step, old rule vs new. The old rule's mass sits near 1.0.

    python paper/figures/make_pointer.py
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from vmd.data.captaincook4d import CaptainCook4DDataset
from vmd.data.step_tracking import (RateLimitedPointer, executable_indices,
                                    gold_step_span, graph_index_of_span, load_step_seconds)
from vmd.graph.schema import ProceduralGraph

C_OLD, C_NEW, C_GOLD = "#D55E00", "#0072B2", "#999999"

def main():
    ds = CaptainCook4DDataset(str(ROOT / "data/raw/captaincook4d"), "test")
    clips = {c.clip_id: c for c in ds.clips}
    graphs = {p.stem: ProceduralGraph.model_validate_json(p.read_text())
              for p in (ROOT / "data/graphs").glob("*.json")}
    sec = load_step_seconds(ROOT / "data/step_seconds.json")
    rec = defaultdict(list)
    for r in json.loads((ROOT / "outputs/full_eval_8class/lora_test_preds.json").read_text()):
        rec[r["clip_id"]].append(r)

    def trajectories(cid):
        rs = sorted(rec[cid], key=lambda r: r["start_sec"]); c = clips[cid]; g = graphs[c.task_id]
        ex = executable_indices(g.steps); n = len(g.steps)
        old, new, gold, t = [], [], [], []
        idx = 0
        ptr = RateLimitedPointer(n, sec.get(c.task_id), 4.0, order=ex)
        for r in rs:
            t.append(r["start_sec"]); old.append(idx); new.append(ptr.idx)
            gi = graph_index_of_span(g.steps, gold_step_span(c.step_spans, r["start_sec"] + 2.0))
            gold.append(gi if gi is not None else np.nan)
            if not r["deviation"] and idx < n - 1: idx += 1
            ptr.observe(r["deviation"])
        return np.array(t), np.array(old), np.array(new), np.array(gold, float), n, ex

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1.5, 1]})
    # (a) a recording of median length, so the panel is typical rather than the
    # worst case; (b) carries the whole distribution.
    lens = sorted((len(rec[k]), k) for k in rec if clips[k].step_spans)
    cid = lens[len(lens) // 2][1]
    t, old, new, gold, n, ex = trajectories(cid)
    a.step(t, gold, where="post", color=C_GOLD, lw=2.2, label="gold segmentation")
    a.step(t, old, where="post", color=C_OLD, lw=1.2, label="original rule (per window)")
    a.step(t, new, where="post", color=C_NEW, lw=1.2, label="rate-limited rule")
    a.axhline(ex[-1], color="#222", ls=":", lw=0.8)
    a.text(t[-1], ex[-1] + 0.25, "final step", ha="right", fontsize=7, color="#222")
    a.set_xlabel("time in recording (s)", fontsize=8); a.set_ylabel("graph step index", fontsize=8)
    a.set_title(f"(a) recording {cid}: which step the model is told it is on", fontsize=8, loc="left")
    a.legend(fontsize=7, frameon=False, loc="lower right"); a.tick_params(labelsize=7.5)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    # (b) ECDF across clips
    fo, fn = [], []
    for k in rec:
        if not clips[k].step_spans: continue
        _, o, nw, _, n, ex = trajectories(k)
        fo.append(np.mean(o == n - 1)); fn.append(np.mean(nw == ex[-1]))
    for vals, col, lab in ((fo, C_OLD, "original rule"), (fn, C_NEW, "rate-limited rule")):
        x = np.sort(vals); y = np.arange(1, len(x) + 1) / len(x)
        b.step(x, y, where="post", color=col, lw=1.4, label=f"{lab} (median {np.median(vals):.0%})")
    b.set_xlabel("fraction of windows conditioned on the final step", fontsize=8)
    b.set_ylabel("fraction of recordings", fontsize=8); b.set_xlim(0, 1); b.set_ylim(0, 1)
    b.set_title("(b) all 109 test recordings", fontsize=8, loc="left")
    b.legend(fontsize=7, frameon=False, loc="lower right"); b.tick_params(labelsize=7.5)
    b.grid(color="#EBEBEB", lw=0.5); b.set_axisbelow(True)
    for sp in ("top", "right"): b.spines[sp].set_visible(False)
    fig.tight_layout(); out = ROOT / "paper/figures/pointer.pdf"; fig.savefig(out, bbox_inches="tight"); print("wrote", out)

if __name__ == "__main__": main()
