"""Re-tile MistSense per-frame probabilities onto the VLM 4s window grid. CPU.

Reads the per-clip per-frame probs dumped by baseline/MistSense/scripts/
eval_online.py --dump_probs (2 fps features) and produces a preds JSON in our
per-window record format, scored on the IDENTICAL windows and ground truth as
the VLM arms (windows + labels joined from our own preds file). Window score =
max frame prob inside the window (a window is a mistake if any moment is).

MistSense emits no typed class and no deviation flag: records carry
deviation=True (gate on confidence only) and mistake_class="no_mistake"
(binary-only baseline — typed metrics are not meaningful for this row).

  python scripts/cc4d_mistsense_retile.py \
      --probs_dir baseline/MistSense/runs/probs_test \
      --our_preds outputs/full_eval_8class/lora_test_preds.json \
      --out outputs/decision_v2/mistsense_test_preds.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs_dir", required=True)
    ap.add_argument("--our_preds", required=True,
                    help="a VLM preds JSON for the same split; supplies the "
                         "window grid and ground truth")
    ap.add_argument("--fps", type=float, default=2.0,
                    help="MistSense feature rate (frames per second)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    our = json.loads(Path(args.our_preds).read_text())
    probs_dir = Path(args.probs_dir)

    cache = {}
    out_records = []
    n_missing_clip = 0
    for r in our:
        cid = r["clip_id"]
        if cid not in cache:
            p = probs_dir / cid / "probs.npy"
            cache[cid] = np.load(p) if p.exists() else None
            if cache[cid] is None:
                n_missing_clip += 1
        probs = cache[cid]
        if probs is None:
            continue
        lo = int(round(r["start_sec"] * args.fps))
        hi = int(round(r["end_sec"] * args.fps))
        seg = probs[lo:min(hi, len(probs))]
        score = float(seg.max()) if len(seg) else 0.0
        out_records.append({
            "clip_id": cid,
            "start_sec": r["start_sec"],
            "end_sec": r["end_sec"],
            "true_label": r["true_label"],
            "true_class": r["true_class"],
            "deviation": True,
            "confidence": score,
            "mistake_class": "no_mistake",
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_records, indent=2))
    n_clips = len({r['clip_id'] for r in out_records})
    print(f"retiled {len(out_records)} windows over {n_clips} clips "
          f"(missing prob dumps: {n_missing_clip}) -> {out}")
    if n_missing_clip:
        print("WARNING: some clips lacked prob dumps; coverage differs from "
              "the VLM rows — do not compare until resolved")


if __name__ == "__main__":
    main()
