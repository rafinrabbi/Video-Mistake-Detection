"""Tune the decision layer on VAL predictions, freeze, apply to TEST. CPU-only.

Grid: tau x smooth_k x gate_mode selected by val binary-F1; min_consecutive
(localization-only — it cannot change binary F1) selected by val EDS at the
chosen operating point. The frozen config is applied once to the test preds.
Honest protocol: selection happens exclusively on val; test is touched once.

  python scripts/cc4d_decision_sweep.py \
      --val_preds outputs/full_eval_8class/lora_val_preds.json \
      --test_preds outputs/full_eval_8class/lora_test_preds.json \
      --arm lora --out_dir outputs/decision_v2
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from vmd.eval.decision_layer import evaluate_records, gold_events, load_preds

SMOOTH_KS = [1, 3, 5]
MIN_CONSECS = [1, 2, 3]
GATE_MODES = ["and", "conf_only"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_preds", required=True)
    ap.add_argument("--test_preds", required=True)
    ap.add_argument("--raw_dir", default=str(ROOT / "data/raw/captaincook4d"))
    ap.add_argument("--arm", required=True, help="name for output files, e.g. 'lora'")
    ap.add_argument("--out_dir", default=str(ROOT / "outputs/decision_v2"))
    ap.add_argument("--tau_min", type=float, default=0.30)
    ap.add_argument("--tau_max", type=float, default=0.90)
    ap.add_argument("--tau_step", type=float, default=0.025)
    args = ap.parse_args()
    taus = [round(t, 4) for t in
            np.arange(args.tau_min, args.tau_max + 1e-9, args.tau_step)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val = load_preds(Path(args.val_preds))
    test = load_preds(Path(args.test_preds))
    gold_val = gold_events(Path(args.raw_dir), "val")
    gold_test = gold_events(Path(args.raw_dir), "test")

    # --- sweep on val (min_consecutive fixed: it cannot affect binary F1) ---
    sweep_rows = []
    best = None
    for gate in GATE_MODES:
        for k in SMOOTH_KS:
            for tau in taus:
                m = evaluate_records(val, gold_val, tau=tau, smooth_k=k,
                                     min_consecutive=2, gate_mode=gate)
                row = {"tau": tau, "smooth_k": k, "gate_mode": gate,
                       "binary_f1": m["binary_f1"], "macro_f1": m["macro_f1"],
                       "precision": m["precision"], "recall": m["recall"],
                       "eds": m["eds"]}
                sweep_rows.append(row)
                if best is None or row["binary_f1"] > best["binary_f1"]:
                    best = row

    # --- pick min_consecutive by val EDS at the chosen operating point ---
    mc_scores = {}
    for mc in MIN_CONSECS:
        m = evaluate_records(val, gold_val, tau=best["tau"],
                             smooth_k=best["smooth_k"], min_consecutive=mc,
                             gate_mode=best["gate_mode"])
        mc_scores[mc] = m["eds"]
    best_mc = max(mc_scores, key=mc_scores.get)

    params = {
        "tau": best["tau"], "smooth_k": best["smooth_k"],
        "min_consecutive": best_mc, "gate_mode": best["gate_mode"],
        "selected_on": "val", "val_binary_f1": best["binary_f1"],
        "val_eds": mc_scores[best_mc],
    }
    params_path = out_dir / f"decision_params_{args.arm}.json"
    params_path.write_text(json.dumps(params, indent=2))
    (out_dir / f"sweep_{args.arm}_val.json").write_text(json.dumps(sweep_rows, indent=2))

    # --- default and tuned config, both applied once to test ---
    m_test_default = evaluate_records(test, gold_test, tau=0.5, smooth_k=3,
                                      min_consecutive=2, gate_mode="and")
    m_test_tuned = evaluate_records(test, gold_test, tau=params["tau"],
                                    smooth_k=params["smooth_k"],
                                    min_consecutive=params["min_consecutive"],
                                    gate_mode=params["gate_mode"])
    m_val_default = evaluate_records(val, gold_val, tau=0.5, smooth_k=3,
                                     min_consecutive=2, gate_mode="and")

    for name, cfg, m in [
        (f"{args.arm}_default", {"tau": 0.5, "smooth_k": 3, "min_consecutive": 2,
                                 "gate_mode": "and"}, m_test_default),
        (f"{args.arm}_tuned", params, m_test_tuned),
    ]:
        (out_dir / f"row_{name}_test.json").write_text(json.dumps(
            {"config": cfg, "metrics": m}, indent=2))

    print(f"[{args.arm}] best val config: {best} + min_consecutive={best_mc}")
    print(f"[{args.arm}] val   F1 default 0.5/3/and : {m_val_default['binary_f1']:.4f}")
    print(f"[{args.arm}] val   F1 tuned            : {best['binary_f1']:.4f}")
    print(f"[{args.arm}] test  F1 default 0.5/3/and : {m_test_default['binary_f1']:.4f}")
    print(f"[{args.arm}] test  F1 tuned  (frozen)   : {m_test_tuned['binary_f1']:.4f}")
    print(f"[{args.arm}] val->test gap (tuned)      : "
          f"{best['binary_f1'] - m_test_tuned['binary_f1']:+.4f}")
    for key in ("precision", "recall", "eds", "start_mae", "macro_f1", "auc_raw"):
        print(f"    test tuned {key}: {m_test_tuned[key]:.4f} "
              f"(default {m_test_default[key]:.4f})")
    print(f"wrote {params_path}")


if __name__ == "__main__":
    main()
