"""Re-run the decision layer over a saved *_preds.json — no GPU needed.

Produces one ablation-table "row JSON" from per-window predictions, applying a
(tau, smooth_k, min_consecutive, gate_mode) decision config. With
--verify_against it instead checks that the re-gated metrics reproduce a
historical eval JSON's ``results.online.fixed`` block exactly (1e-9), which is
the regression guard for any surgery on the decision path.

  python scripts/cc4d_decision_eval.py \
      --preds outputs/full_eval_8class/lora_test_preds.json --split test \
      --verify_against outputs/full_eval_8class/lora_test_eval.json

  python scripts/cc4d_decision_eval.py \
      --preds outputs/full_eval_8class/lora_test_preds.json --split test \
      --tau 0.6 --gate_mode conf_only --out outputs/decision_v2/row.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vmd.eval.decision_layer import evaluate_records, gold_events, load_preds

# Keys checked against the historical eval block (everything it reported).
VERIFY_KEYS = [
    "binary_f1", "macro_f1", "auc_roc", "ece", "brier", "precision", "recall",
    "eds", "start_mae", "n_loc_events", "n_loc_detected",
]


def _close(a, b, tol=1e-9) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= tol
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--raw_dir", default=str(ROOT / "data/raw/captaincook4d"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--smooth_k", type=int, default=3)
    ap.add_argument("--min_consecutive", type=int, default=2)
    ap.add_argument("--gate_mode", default="and", choices=["and", "conf_only"])
    ap.add_argument("--out", default="")
    ap.add_argument("--verify_against", default="",
                    help="historical eval JSON; compare against results.online.fixed")
    args = ap.parse_args()

    preds = load_preds(Path(args.preds))
    gold = gold_events(Path(args.raw_dir), args.split)
    metrics = evaluate_records(
        preds, gold,
        tau=args.tau, smooth_k=args.smooth_k,
        min_consecutive=args.min_consecutive, gate_mode=args.gate_mode,
    )

    if args.verify_against:
        ref = json.loads(Path(args.verify_against).read_text())
        ref_fixed = ref["results"]["online"]["fixed"]
        failures = []
        for k in VERIFY_KEYS:
            if not _close(metrics[k], ref_fixed[k]):
                failures.append(f"{k}: got {metrics[k]!r}, expected {ref_fixed[k]!r}")
        for cls, ref_v in ref_fixed["per_class_f1"].items():
            if not _close(metrics["per_class_f1"][cls], ref_v):
                failures.append(f"per_class_f1[{cls}]: got "
                                f"{metrics['per_class_f1'][cls]!r}, expected {ref_v!r}")
        for cls, ref_v in ref_fixed["support"].items():
            if metrics["support"][cls] != ref_v:
                failures.append(f"support[{cls}]: got "
                                f"{metrics['support'][cls]!r}, expected {ref_v!r}")
        if failures:
            print(f"VERIFY FAILED against {args.verify_against}:")
            for f in failures:
                print(f"  {f}")
            sys.exit(1)
        print(f"VERIFY OK: {len(VERIFY_KEYS)} metrics + per-class F1 + support "
              f"reproduce {args.verify_against} exactly "
              f"({metrics['n_windows']} windows, {metrics['n_clips']} clips)")

    row = {
        "config": {
            "preds": args.preds, "split": args.split, "tau": args.tau,
            "smooth_k": args.smooth_k, "min_consecutive": args.min_consecutive,
            "gate_mode": args.gate_mode,
        },
        "metrics": metrics,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(row, indent=2))
        print(f"Wrote {out}")
    else:
        print(json.dumps({k: v for k, v in metrics.items()
                          if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
