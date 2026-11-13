#!/usr/bin/env python
"""Build the unseen-recipe split by a stated rule. CPU only.

The benchmark's ``recipes_data_split_*.json`` files are empty, so cross-recipe
evaluation needs a split of our own. Rule (printed and stored with the file):

  * hold out ``--k`` recipes (default 4);
  * every held-out recipe has >= ``--min_order`` recordings containing a
    localizable order error (the graph's home turf must be testable);
  * no held-out recipe has repeated step text (keeps menu letters unambiguous);
  * spread over step count: rank eligible recipes by executable-step count and
    take one from each of ``k`` equal-size bands, choosing within a band the
    recipe with the most error recordings (ties: alphabetical).

Also writes a training-only step-seconds table for the remaining recipes.

    python scripts/cc4d_make_recipe_holdout.py --out data/splits/recipe_holdout_v1.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vmd.data.captaincook4d import CaptainCook4DDataset            # noqa: E402
from vmd.data.step_tracking import estimate_step_seconds, save_step_seconds, _norm  # noqa: E402
from vmd.graph.schema import ProceduralGraph                        # noqa: E402
from vmd.tracking.belief import build_step_space                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(ROOT / "data/raw/captaincook4d"))
    ap.add_argument("--graphs_dir", default=str(ROOT / "data/graphs"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min_order", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "data/splits/recipe_holdout_v1.json"))
    ap.add_argument("--step_seconds_out", default=str(ROOT / "data/step_seconds_holdout_v1.json"))
    args = ap.parse_args()

    graphs = {p.stem: ProceduralGraph.model_validate_json(p.read_text()) for p in Path(args.graphs_dir).glob("*.json")}
    clips = []
    for split in ("train", "val", "test"):
        clips.extend(CaptainCook4DDataset(args.raw_dir, split).clips)
    stats = {}
    for rec in sorted({c.task_id for c in clips}):
        if rec not in graphs:
            continue
        g = graphs[rec]; sp = build_step_space(g)
        dup = any(v > 1 for v in Counter(_norm(d) for d in sp.descriptions).values())
        rc = [c for c in clips if c.task_id == rec]
        n_order = sum(1 for c in rc if any(a.mistake_class == "order_error" and a.start_sec >= 0 for a in c.annotations))
        n_err = sum(1 for c in rc if any(a.start_sec >= 0 for a in c.annotations))
        stats[rec] = {"n_steps": sp.n, "n_recordings": len(rc), "n_error_recordings": n_err,
                      "n_order_recordings": n_order, "duplicate_step_text": dup}
    eligible = sorted([r for r, s in stats.items() if s["n_order_recordings"] >= args.min_order and not s["duplicate_step_text"]],
                      key=lambda r: (stats[r]["n_steps"], r))
    bands = [eligible[i * len(eligible) // args.k:(i + 1) * len(eligible) // args.k] for i in range(args.k)]
    chosen = [max(b, key=lambda r: (stats[r]["n_error_recordings"], -ord(r[0]))) for b in bands if b]
    held = sorted(chosen)
    out = {"schema": "recipe_holdout_v1", "k": args.k, "min_order_recordings": args.min_order,
           "rule": "eligible = >=min_order order-error recordings and no duplicate step text; ranked by step count, "
                   "split into k bands, most error recordings per band",
           "heldout_recipes": held, "eligible": eligible, "stats": stats,
           "n_heldout_recordings": sum(stats[r]["n_recordings"] for r in held)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("held out:", {r: stats[r] for r in held})
    print("held-out recordings:", out["n_heldout_recordings"])
    # step seconds from the holdout TRAIN role only
    train_clips = CaptainCook4DDataset(args.raw_dir, "train", recipe_holdout=Path(args.out)).clips
    save_step_seconds(Path(args.step_seconds_out), estimate_step_seconds(train_clips))
    print("wrote", args.out, "and", args.step_seconds_out)


if __name__ == "__main__":
    main()
