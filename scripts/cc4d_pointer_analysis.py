#!/usr/bin/env python
"""Measure how much positional information the walked step pointer actually carries.

Motivated by the R5 result: stripping the graph at inference changed no detection
metric. One candidate explanation is that the graph context is near-constant for
most windows anyway -- the online pointer advances once per NO-DEVIATION WINDOW,
but a real recipe step spans many windows, so the pointer races to the final step
and clamps there. This script quantifies that. CPU-only.

    python scripts/cc4d_pointer_analysis.py --preds outputs/full_eval_8class/lora_test_preds.json
"""
import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vmd.data.captaincook4d import CaptainCook4DDataset   # noqa: E402
from vmd.graph.schema import ProceduralGraph              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(ROOT / "outputs/full_eval_8class/lora_test_preds.json"))
    ap.add_argument("--raw_dir", default=str(ROOT / "data/raw/captaincook4d"))
    ap.add_argument("--graphs_dir", default=str(ROOT / "data/graphs"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(ROOT / "outputs/decision_v2/pointer_saturation.json"))
    args = ap.parse_args()

    ds = CaptainCook4DDataset(args.raw_dir, args.split)
    task = {c.clip_id: c.task_id for c in ds.clips}
    graphs = {p.stem: ProceduralGraph.model_validate_json(p.read_text())
              for p in Path(args.graphs_dir).glob("*.json")}

    by_clip = defaultdict(list)
    for r in json.loads(Path(args.preds).read_text()):
        by_clip[r["clip_id"]].append(r)

    sat_frac, reach_at, dev_rate = [], [], []
    total_w = sat_w = 0
    for cid, rs in by_clip.items():
        rs.sort(key=lambda x: x["start_sec"])
        n_steps = len(graphs[task[cid]].steps)
        idx, at_last, first = 0, 0, None
        for i, r in enumerate(rs):
            if idx == n_steps - 1:
                at_last += 1
                if first is None:
                    first = i
            if not r["deviation"] and idx < n_steps - 1:
                idx += 1
        sat_frac.append(at_last / len(rs))
        dev_rate.append(sum(r["deviation"] for r in rs) / len(rs))
        total_w += len(rs)
        sat_w += at_last
        if first is not None:
            reach_at.append(first / len(rs))

    # ---- training-side step assignment -----------------------------------
    # Training does NOT use the advancing pointer: lora_train.py assigns
    # step = graph.steps[min(window_index, n_steps-1)], i.e. the offline rule.
    # Quantify the positional signal the model was actually given.
    tr = CaptainCook4DDataset(args.raw_dir, "train")
    tr_tot = tr_sat = 0
    tr_fracs = []
    for c in tr.clips:
        g = graphs.get(c.task_id)
        if g is None:
            continue
        n = len(g.steps)
        w = max(0, int((c.duration_sec - 4.0) // 4.0) + 1)
        if w == 0:
            continue
        at_last = max(0, w - (n - 1))
        tr_tot += w
        tr_sat += at_last
        tr_fracs.append(at_last / w)

    out = {
        "n_clips": len(by_clip),
        "n_windows": total_w,
        "windows_at_final_step": sat_w,
        "frac_windows_at_final_step": sat_w / total_w,
        "per_clip_saturated_frac_median": st.median(sat_frac),
        "per_clip_saturated_frac_mean": st.mean(sat_frac),
        "n_clips_majority_saturated": sum(1 for f in sat_frac if f > 0.5),
        "n_clips_reaching_final_step": len(reach_at),
        "reach_final_step_at_frac_median": st.median(reach_at) if reach_at else None,
        "deviation_rate_median": st.median(dev_rate),
        "train_n_windows": tr_tot,
        "train_windows_at_final_step": tr_sat,
        "train_frac_windows_at_final_step": tr_sat / tr_tot if tr_tot else None,
        "train_per_clip_saturated_frac_median": st.median(tr_fracs) if tr_fracs else None,
        "train_note": "Training assigns step = min(window_index, n_steps-1), so "
                      "positional variation exists only in the first n_steps windows "
                      "of a recording. This is a different rule from the one used at "
                      "evaluation -- a train/test mismatch in the conditioning.",
        "note": "The pointer advances once per no-deviation WINDOW, while a recipe "
                "step spans many windows; it therefore reaches the last step early "
                "and clamps, leaving the graph context constant thereafter.",
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
