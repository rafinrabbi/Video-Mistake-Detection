#!/usr/bin/env python
"""Step-tracking accuracy from saved GraphTrack records, with filter re-runs. CPU.

Uses the per-window ``step_probs`` stored by the belief tracker to (a) score
the tracked step actually shown to the model against the gold step, (b) replay
the filter in dag/linear/uniform modes and as raw argmax of the emission, and
(c) sweep filter hyper-parameters. Everything is a replay: no GPU.

    python scripts/cc4d_tracking_eval.py --preds outputs/review_v4/gt_test_preds.json \
        --split test --out outputs/review_v4/tracking_gt_test.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vmd.data.captaincook4d import CaptainCook4DDataset            # noqa: E402
from vmd.data.step_tracking import load_step_seconds                # noqa: E402
from vmd.eval.decision_layer import load_preds                      # noqa: E402
from vmd.eval.event_metrics import dedup, tracking_accuracy         # noqa: E402
from vmd.eval.metrics import normalised_edit_distance               # noqa: E402
from vmd.graph.schema import ProceduralGraph                        # noqa: E402
from vmd.tracking.belief import BeliefFilter, FilterConfig, build_step_space  # noqa: E402


def replay(by_clip, clips, graphs, step_sec, mode, overrides):
    tracked, gold, edit = [], [], []
    for cid, recs in by_clip.items():
        clip = clips[cid]; space = build_step_space(graphs[clip.task_id])
        recs = sorted(recs, key=lambda r: r["start_sec"])
        if mode == "emission_argmax":
            tr = [int(np.argmax(r["step_probs"])) for r in recs]
        else:
            cfg = FilterConfig.from_step_seconds(step_sec.get(clip.task_id), 4.0, mode=mode, **overrides)
            f = BeliefFilter(space, cfg)
            tr = []
            for r in recs:                      # causal: state before this window's update
                tr.append(f.argmax)
                f.update(r["step_probs"])
        g = [None if r["gold_step_idx"] < 0 else r["gold_step_idx"] for r in recs]
        tracked.extend(tr); gold.extend(g)
        edit.append(normalised_edit_distance([str(x) for x in dedup(tr)], [str(x) for x in dedup(g)]))
    acc = tracking_accuracy(tracked, gold)
    acc["segment_edit_score"] = float(1 - np.mean(edit))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--raw_dir", default=str(ROOT / "data/raw/captaincook4d"))
    ap.add_argument("--graphs_dir", default=str(ROOT / "data/graphs"))
    ap.add_argument("--step_seconds", default=str(ROOT / "data/step_seconds.json"))
    ap.add_argument("--recipe_holdout", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweep", action="store_true", help="grid over p_stay/skip_frac/emission_temp (val only)")
    args = ap.parse_args()

    ds = CaptainCook4DDataset(args.raw_dir, args.split,
                              recipe_holdout=Path(args.recipe_holdout) if args.recipe_holdout else None)
    clips = {c.clip_id: c for c in ds.clips}
    graphs = {p.stem: ProceduralGraph.model_validate_json(p.read_text()) for p in Path(args.graphs_dir).glob("*.json")}
    step_sec = load_step_seconds(Path(args.step_seconds))
    by = load_preds(Path(args.preds))
    recs = [r for c in by.values() for r in c]
    out = {"preds": args.preds, "n_windows": len(recs)}
    # what the model was actually told
    shown = [None if r["tracked_step_idx"] < 0 else r["tracked_step_idx"] for r in recs]
    gold = [None if r["gold_step_idx"] < 0 else r["gold_step_idx"] for r in recs]
    out["as_run"] = tracking_accuracy(shown, gold)
    out["as_run"]["transit_fraction"] = float(np.mean([r["tracked_step_idx"] >= len(r["step_probs"]) - 1 for r in recs]))
    for mode in ("emission_argmax", "dag", "linear", "uniform"):
        out[mode] = replay(by, clips, graphs, step_sec, mode, {})
        print(f"[{mode}] step acc {out[mode]['step_accuracy']:.3f} edit {out[mode]['segment_edit_score']:.3f}", flush=True)
    if args.sweep:
        grid = []
        for p_stay in (None, 0.7, 0.8, 0.9, 0.95):
            for skip in (0.05, 0.1, 0.2):
                for temp in (0.5, 1.0, 2.0):
                    ov = {"skip_frac": skip, "emission_temp": temp}
                    if p_stay is not None:
                        ov["p_stay"] = p_stay
                    a = replay(by, clips, graphs, step_sec, "dag", ov)
                    grid.append({"overrides": ov, **a})
        grid.sort(key=lambda g: -g["step_accuracy"])
        out["sweep"] = grid
        print("best:", grid[0])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
