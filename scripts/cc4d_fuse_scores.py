#!/usr/bin/env python
"""Fit score fusion on GraphTrack VAL records, apply to TEST, write derived arms. CPU.

    python scripts/cc4d_fuse_scores.py \
        --val_preds outputs/review_v4/gt_val_preds.json \
        --test_preds outputs/review_v4/gt_test_preds.json \
        --out_dir outputs/review_v4

Writes <out_dir>/gt_fused_{val,test}_preds.json, gt_structural_{val,test}_preds.json
and fusion_params.json. The frozen decision layer is then tuned on the fused
VAL file with scripts/cc4d_decision_sweep.py exactly as for every other arm.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vmd.eval.fusion import (FEATURES, apply_fusion, fit_fusion, structural_only,  # noqa: E402
                             write_derived)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_preds", required=True)
    ap.add_argument("--test_preds", required=True)
    ap.add_argument("--out_dir", default=str(ROOT / "outputs/review_v4"))
    ap.add_argument("--features", default=",".join(FEATURES))
    ap.add_argument("--structural_signal", default="illegal_hold")
    ap.add_argument("--prefix", default="gt", help="output name prefix: <prefix>_fused_*, <prefix>_structural_*")
    args = ap.parse_args()
    names = args.features.split(",")
    val = json.loads(Path(args.val_preds).read_text())
    test = json.loads(Path(args.test_preds).read_text())
    fusion = fit_fusion(val, names)
    out = Path(args.out_dir)
    pre = args.prefix
    (out / f"fusion_params_{pre}.json").write_text(json.dumps(fusion, indent=1))
    write_derived(apply_fusion(val, fusion), out / f"{pre}_fused_val_preds.json")
    write_derived(apply_fusion(test, fusion), out / f"{pre}_fused_test_preds.json")
    write_derived(structural_only(val, args.structural_signal), out / f"{pre}_structural_val_preds.json")
    write_derived(structural_only(test, args.structural_signal), out / f"{pre}_structural_test_preds.json")
    k = len(names)
    print("fusion weights (standardised):", dict(zip(["bias"] + names, [round(x, 3) for x in fusion["params"][: k + 1]])))
    print("wrote derived arms to", out)


if __name__ == "__main__":
    main()
