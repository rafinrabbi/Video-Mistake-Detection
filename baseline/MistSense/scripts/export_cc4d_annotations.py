"""Export CaptainCook4D into EgoOops-format annotation JSONs for MistSense.

MistSense's extract_features.py / train.py consume per-split JSON lists with
the EgoOops schema (clip_id, task_*, duration_sec, mistakes[{start_sec,
end_sec, mistake_class, description}]). CaptainCook4DDataset already produces
that content via EgoOopsClip, so we just serialise it. Reusing the schema lets
the whole MistSense pipeline run on CC4D unchanged.

Only LOCALIZABLE mistakes (start>=0) are written: MistSense supervises a
per-frame binary label from time spans, so start=-1 segments cannot mark
frames anyway (they are still counted/logged for transparency).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vmd.data.captaincook4d import CaptainCook4DDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(REPO_ROOT / "data/raw/captaincook4d"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--refine_prep", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    totals = {}
    for split in ["train", "val", "test"]:
        ds = CaptainCook4DDataset(args.raw_dir, split, refine_prep=args.refine_prep)
        entries = []
        dropped_nonloc = 0
        for c in ds.clips:
            mistakes = []
            for a in c.annotations:
                if a.start_sec < 0 or a.end_sec < 0:
                    dropped_nonloc += 1
                    continue
                mistakes.append({
                    "start_sec": a.start_sec,
                    "end_sec": a.end_sec,
                    "mistake_class": a.mistake_class,
                    "description": a.description,
                })
            entries.append({
                "clip_id": c.clip_id,
                "task_id": c.task_id,
                "task_name": c.task_name,
                "task_description": c.task_description,
                "duration_sec": c.duration_sec,
                "mistakes": mistakes,
            })
        (out_dir / f"{split}.json").write_text(json.dumps(entries, indent=2))
        n_mistake = sum(1 for e in entries if e["mistakes"])
        totals[split] = (len(entries), n_mistake, dropped_nonloc)
        print(f"{split}: {len(entries)} clips, {n_mistake} w/ localizable mistake, "
              f"{dropped_nonloc} nonlocalizable dropped")
    print(f"wrote EgoOops-format annotations -> {out_dir}")


if __name__ == "__main__":
    main()
