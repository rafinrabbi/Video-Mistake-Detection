#!/usr/bin/env python
"""Join step-recogniser records with a content detector's records. CPU only.

The recogniser pass writes, per window, the step distribution and the belief
filter's structural signals but no content score. This script attaches those
fields to a saved content arm (same window grid) so the fusion / structural
derived arms and all review_v4 analyses run unchanged.

    python scripts/cc4d_join_stepreco.py \
        --stepreco outputs/review_v4/stepreco_test_preds.json \
        --content outputs/full_eval_8class/lora_test_preds.json \
        --out outputs/review_v4/gt2_test_preds.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

STEP_FIELDS = ("task_id", "window_idx", "gold_step_idx", "gold_step_idx_strict", "tracked_step_idx",
               "tracked_p", "step_pred_idx", "step_probs", "belief_argmax", "belief_entropy",
               "structural_surprise", "illegal_hold", "violation_mass")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stepreco", required=True)
    ap.add_argument("--content", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    step = {(r["clip_id"], round(float(r["start_sec"]), 3)): r for r in json.loads(Path(args.stepreco).read_text())}
    content = json.loads(Path(args.content).read_text())
    out, missing = [], 0
    for r in content:
        key = (r["clip_id"], round(float(r["start_sec"]), 3))
        s = step.get(key)
        if s is None:
            missing += 1
            continue
        assert s["true_label"] == r["true_label"] and s["true_class"] == r["true_class"], key
        q = dict(r)
        for k in STEP_FIELDS:
            if k in s:
                q[k] = s[k]
        q["step_source"] = args.stepreco
        q["content_source"] = args.content
        out.append(q)
    if missing:
        print(f"warning: {missing} content windows had no recogniser record and were dropped")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}: {len(out)} windows")


if __name__ == "__main__":
    main()
