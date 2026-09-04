"""Clip-level bootstrap CIs + paired significance for decision-layer rows. CPU.

Single mode: 95% CIs (percentile, clip-level resampling — windows within a clip
are correlated, so clips are the exchangeable unit) for binary F1, macro F1,
raw AUC, EDS.

Paired mode (--preds_b): both systems evaluated on the SAME resampled clips per
replicate -> CI on the F1 difference (primary significance evidence), plus
McNemar on per-window decisions joined on (clip_id, start_sec) (secondary — the
window-independence assumption is violated by within-clip correlation).

  python scripts/cc4d_bootstrap_compare.py \
      --preds outputs/full_eval_8class/lora_test_preds.json \
      --params outputs/decision_v2/decision_params_lora.json --split test

  python scripts/cc4d_bootstrap_compare.py \
      --preds outputs/full_eval_8class/lora_test_preds.json \
      --params outputs/decision_v2/decision_params_lora.json \
      --preds_b outputs/full_eval_8class/zs_test_preds.json \
      --params_b outputs/decision_v2/decision_params_zs.json \
      --split test --label lora_vs_zs
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from vmd.data.mistake_classes import MISTAKE_CLASSES
from vmd.decision.postprocess import detect_start_of_deviation
from vmd.eval.decision_layer import FIRED_EPS, gate_clip, gold_events, load_preds
from vmd.eval.metrics import early_detection_score

N_BOOT = 1000
SEED = 0


def _default_params():
    return {"tau": 0.5, "smooth_k": 3, "min_consecutive": 2, "gate_mode": "and"}


def per_clip_stats(preds_by_clip, gold, params):
    """Precompute per-clip sufficient statistics so each bootstrap replicate is
    a cheap aggregation instead of a full re-evaluation."""
    classes = list(MISTAKE_CLASSES)
    stats = []
    for clip_id, records in preds_by_clip.items():
        g = gate_clip(records, params["tau"], params["smooth_k"], params["gate_mode"])
        y_true = np.array([int(r["true_label"]) for r in records])
        y_pred = np.array(g["pred_bin"])
        conf = np.array([float(r["confidence"]) for r in records])
        s = {
            "clip_id": clip_id,
            "tp": int(((y_pred == 1) & (y_true == 1)).sum()),
            "fp": int(((y_pred == 1) & (y_true == 0)).sum()),
            "fn": int(((y_pred == 0) & (y_true == 1)).sum()),
            "y_true": y_true, "y_pred": y_pred, "conf": conf,
            "cls_counts": {},
            "eds": None,
        }
        for c in classes:
            t = np.array([r["true_class"] == c for r in records])
            p = np.array([pc == c for pc in g["pred_cls"]])
            s["cls_counts"][c] = (int((t & p).sum()), int((~t & p).sum()),
                                  int((t & ~p).sum()))
        event = gold.get(clip_id)
        if event is not None and g["starts"]:
            g_start, g_end = event
            ps = detect_start_of_deviation(
                g["smoothed"], g["starts"], FIRED_EPS, params["min_consecutive"])
            s["eds"] = (early_detection_score(ps, g_start, g_end)
                        if ps is not None else 0.0)
        stats.append(s)
    return stats


def aggregate(stats_subset):
    tp = sum(s["tp"] for s in stats_subset)
    fp = sum(s["fp"] for s in stats_subset)
    fn = sum(s["fn"] for s in stats_subset)
    binary_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    f1s = []
    for c in MISTAKE_CLASSES:
        ctp = sum(s["cls_counts"][c][0] for s in stats_subset)
        cfp = sum(s["cls_counts"][c][1] for s in stats_subset)
        cfn = sum(s["cls_counts"][c][2] for s in stats_subset)
        f1s.append(2 * ctp / (2 * ctp + cfp + cfn) if (2 * ctp + cfp + cfn) else 0.0)
    macro_f1 = float(np.mean(f1s))

    y_true = np.concatenate([s["y_true"] for s in stats_subset])
    conf = np.concatenate([s["conf"] for s in stats_subset])
    if y_true.min() == y_true.max():
        auc = float("nan")
    else:
        order = np.argsort(conf, kind="mergesort")
        ranks = np.empty(len(conf))
        ranks[order] = np.arange(1, len(conf) + 1)
        # average ties
        sorted_conf = conf[order]
        i = 0
        while i < len(conf):
            j = i
            while j + 1 < len(conf) and sorted_conf[j + 1] == sorted_conf[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos
        auc = float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2)
                    / (n_pos * n_neg))

    eds_vals = [s["eds"] for s in stats_subset if s["eds"] is not None]
    eds = float(np.mean(eds_vals)) if eds_vals else float("nan")
    return {"binary_f1": binary_f1, "macro_f1": macro_f1,
            "auc_raw": auc, "eds": eds}


def ci(values, lo=2.5, hi=97.5):
    arr = np.array([v for v in values if not np.isnan(v)])
    return [float(np.percentile(arr, lo)), float(np.percentile(arr, hi))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--params", default="")
    ap.add_argument("--preds_b", default="")
    ap.add_argument("--params_b", default="")
    ap.add_argument("--raw_dir", default=str(ROOT / "data/raw/captaincook4d"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--label", default="")
    ap.add_argument("--out_dir", default=str(ROOT / "outputs/decision_v2"))
    args = ap.parse_args()

    def read_params(p):
        return json.loads(Path(p).read_text()) if p else _default_params()

    gold = gold_events(Path(args.raw_dir), args.split)
    preds_a = load_preds(Path(args.preds))
    stats_a = per_clip_stats(preds_a, gold, read_params(args.params))
    clip_ids = [s["clip_id"] for s in stats_a]
    idx_a = {s["clip_id"]: s for s in stats_a}

    rng = np.random.default_rng(SEED)
    resamples = [rng.choice(clip_ids, size=len(clip_ids), replace=True)
                 for _ in range(N_BOOT)]

    point_a = aggregate(stats_a)
    boots_a = [aggregate([idx_a[c] for c in rs]) for rs in resamples]
    result = {
        "label": args.label or Path(args.preds).stem,
        "n_clips": len(clip_ids), "n_boot": N_BOOT, "seed": SEED,
        "point": point_a,
        "ci95": {k: ci([b[k] for b in boots_a]) for k in point_a},
    }

    if args.preds_b:
        preds_b = load_preds(Path(args.preds_b))
        stats_b = per_clip_stats(preds_b, gold, read_params(args.params_b))
        idx_b = {s["clip_id"]: s for s in stats_b}
        assert set(idx_b) == set(clip_ids), "paired compare needs identical clips"
        point_b = aggregate(stats_b)
        boots_b = [aggregate([idx_b[c] for c in rs]) for rs in resamples]
        deltas = {k: [ba[k] - bb[k] for ba, bb in zip(boots_a, boots_b)]
                  for k in point_a}
        result["b"] = {"point": point_b,
                       "ci95": {k: ci([b[k] for b in boots_b]) for k in point_b}}
        result["delta_a_minus_b"] = {
            k: {"point": point_a[k] - point_b[k], "ci95": ci(deltas[k])}
            for k in point_a
        }

        # McNemar on per-window binary decisions (secondary; windows correlated)
        joined = 0, 0, 0
        b_cnt = c_cnt = n_join = 0
        for cid in clip_ids:
            sa, sb = idx_a[cid], idx_b[cid]
            assert len(sa["y_true"]) == len(sb["y_true"])
            corr_a = sa["y_pred"] == sa["y_true"]
            corr_b = sb["y_pred"] == sb["y_true"]
            b_cnt += int((~corr_a & corr_b).sum())   # a wrong, b right
            c_cnt += int((corr_a & ~corr_b).sum())   # a right, b wrong
            n_join += len(corr_a)
        from scipy.stats import binomtest
        p = (binomtest(min(b_cnt, c_cnt), b_cnt + c_cnt, 0.5).pvalue
             if (b_cnt + c_cnt) else float("nan"))
        result["mcnemar"] = {"n_windows": n_join, "a_wrong_b_right": b_cnt,
                             "a_right_b_wrong": c_cnt, "p_two_sided": float(p),
                             "caveat": "windows within a clip are correlated; "
                                       "treat as secondary to the paired bootstrap"}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.label or Path(args.preds).stem
    path = out_dir / f"bootstrap_{name}.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("label", "point", "ci95", "delta_a_minus_b",
                               "mcnemar")}, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
