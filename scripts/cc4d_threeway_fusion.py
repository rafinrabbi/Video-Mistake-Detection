"""Three-way fusion (pre-registration amendment 2, S10). CPU only.

Takes a finished matched run (R4 content; structural features per topology,
the five validation folds) and joins MistSense's score on the same windows.
The same L2 logistic fusion is refitted with one extra feature. Output layout
matches the matched runs so cc4d_confirm_aggregate.py applies unchanged:

    mistsense := MistSense only           vlm := VLM only
    content   := MistSense + VLM          dag / linear / uniform := + structure

    python scripts/cc4d_threeway_fusion.py --run outputs/confirm_v1/matched_r4_s42 \
        --out_dir outputs/confirm_v1/threeway_s42
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vmd.data.captaincook4d import CaptainCook4DDataset  # noqa: E402
from vmd.eval.fusion import _logit, apply_fusion, fit_fusion  # noqa: E402
from vmd.eval.matched_graph import crossfit_fusion, weighted_ap, weighted_ap_preparation  # noqa: E402
from vmd.eval.provenance import manifest  # noqa: E402

_spec = importlib.util.spec_from_file_location("mge", ROOT / "scripts/cc4d_matched_graph_experiment.py")
mge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mge)

STRUCT = ("illegal_hold", "structural_surprise", "belief_entropy")
ARMS = {
    "mistsense": ("mistsense_logit",),
    "vlm": ("content_logit",),
    "content": ("content_logit", "mistsense_logit"),
    "dag": ("content_logit", "mistsense_logit") + STRUCT,
    "linear": ("content_logit", "mistsense_logit") + STRUCT,
    "uniform": ("content_logit", "mistsense_logit") + STRUCT,
}
TOPOLOGY = {"mistsense": "uniform", "vlm": "uniform", "content": "uniform",
            "dag": "dag", "linear": "linear", "uniform": "uniform"}
CONTRASTS = (("dag", "content"), ("dag", "linear"), ("dag", "uniform"),
             ("dag", "mistsense"), ("content", "mistsense"), ("dag", "vlm"))


def restore(rows, mistsense):
    """Undo the matched run's fusion (raw VLM score back into ``confidence``)
    and attach MistSense's score for the same window."""
    out = []
    for r in rows:
        q = {k: v for k, v in r.items() if k not in ("crossfit_fold", "deviation")}
        q["confidence"] = float(r["content_confidence"])
        del q["content_confidence"]
        m = mistsense[(r["clip_id"], round(float(r["start_sec"]), 3))]
        if int(m["true_label"]) != int(r["true_label"]) or abs(float(m["end_sec"]) - float(r["end_sec"])) > 1e-6:
            raise ValueError(f"MistSense window does not match: {r['clip_id']} {r['start_sec']}")
        p = float(m["confidence"])
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid MistSense score")
        q["mistsense_confidence"] = p
        q["mistsense_logit"] = _logit(p)
        out.append(q)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="finished matched run (R4 content)")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--mistsense_val", type=Path, default=ROOT / "outputs/decision_v2/mistsense_val_preds.json")
    ap.add_argument("--mistsense_test", type=Path, default=ROOT / "outputs/decision_v2/mistsense_test_preds.json")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    out = args.out_dir
    raw = ROOT / "data/raw/captaincook4d"

    run_files = sorted(args.run.glob("*.json"))
    config = dict(schema="threeway_fusion_v1", source_run=str(args.run), arms={k: list(v) for k, v in ARMS.items()},
                  topology_source=TOPOLOGY, fusion="same L2 logistic; folds from the source run; final fit all VAL",
                  thresholds="0..1 by .01 on out-of-fold VAL scores", seed=args.seed, n_boot=args.n_boot,
                  status="secondary (amendment 2 / S10); no multiplicity claim")
    mge.save(out / "protocol_manifest.json",
             manifest(config, run_files + [args.mistsense_val, args.mistsense_test, Path(__file__)]))

    folds = json.loads((args.run / "validation_folds.json").read_text())
    ms = {}
    for split, path in (("val", args.mistsense_val), ("test", args.mistsense_test)):
        ms[split] = {(r["clip_id"], round(float(r["start_sec"]), 3)): r for r in json.loads(path.read_text())}
    clips = {split: {c.clip_id: c for c in CaptainCook4DDataset(raw, split).clips} for split in ("val", "test")}

    cache = {}
    def records(topology, split):
        if (topology, split) not in cache:
            name = f"{topology}_oof_val_preds.json" if split == "val" else f"{topology}_test_preds.json"
            rows = restore(json.loads((args.run / name).read_text()), ms[split])
            rows.sort(key=lambda r: (r["clip_id"], r["start_sec"]))
            cache[(topology, split)] = rows
        return cache[(topology, split)]

    test_arms, evaluations, models, selected = {}, {}, {}, {}
    for arm, names in ARMS.items():
        val, test = records(TOPOLOGY[arm], "val"), records(TOPOLOGY[arm], "test")
        oof, fold_params = crossfit_fusion(val, names, folds)
        points, sweep = mge.select_operating_points(oof, clips["val"])
        model = fit_fusion(val, names)
        final = apply_fusion(test, model)
        for r in final:
            r.update(score_version="threeway_fusion_v1", score_status="ok", parse_ok=True)
        models[arm] = {"final_fit": model, "crossfit": fold_params}
        selected[arm] = points
        test_arms[arm] = final
        evaluations[arm] = {key: mge.evaluate(final, clips["test"], p["tau"]) for key, p in points.items()}
        mge.save(out / f"{arm}_test_preds.json", final)
        mge.save(out / f"{arm}_oof_val_preds.json", oof)
        mge.save(out / f"{arm}_val_sweep.json", sweep)
        mge.save(out / f"{arm}_causal_test.json", evaluations[arm])
        m = evaluations[arm]["window_f1"]["metrics"]
        print(f"{arm}: TEST AP={m['average_precision']:.4f}; AUC={m['auc_raw']:.4f}; "
              f"window F1={m['binary_f1']:.4f}", flush=True)
    mge.save(out / "fusion_models.json", models)
    mge.save(out / "selected_operating_points.json", selected)
    shutil.copy(args.run / "strict_tracking.json", out / "strict_tracking.json")

    types = {}
    for arm, rows in test_arms.items():
        types[arm] = {}
        for typ in sorted({r["true_class"] for r in rows} - {"no_mistake"}):
            rr = [r for r in rows if r["true_class"] in (typ, "no_mistake")]
            y = [r["true_class"] == typ for r in rr]
            types[arm][typ] = dict(n_positive=sum(y), auc=float(roc_auc_score(y, [r["confidence"] for r in rr])))
    mge.save(out / "per_error_type.json", types)

    # Per-seed paired recording bootstrap on AP (the cross-seed version lives in the aggregate).
    ids = sorted({r["clip_id"] for r in test_arms["dag"]})
    idx = {cid: i for i, cid in enumerate(ids)}
    y = np.array([r["true_label"] for r in test_arms["dag"]])
    wclip = np.array([idx[r["clip_id"]] for r in test_arms["dag"]])
    prepared = {arm: weighted_ap_preparation(y, [r["confidence"] for r in rows]) for arm, rows in test_arms.items()}
    rng = np.random.default_rng(args.seed)
    draws = {c: [] for c in CONTRASTS}
    for _ in range(args.n_boot):
        w = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)))[wclip]
        aps = {arm: weighted_ap(prepared[arm], w) for arm in ARMS}
        for a, b in CONTRASTS:
            draws[(a, b)].append(aps[a] - aps[b])
    ones = np.ones(len(y))
    boot = {f"{a}_minus_{b}": dict(point=weighted_ap(prepared[a], ones) - weighted_ap(prepared[b], ones),
                                   ci95=np.quantile(draws[(a, b)], [.025, .975]).tolist())
            for a, b in CONTRASTS}
    mge.save(out / "paired_bootstrap.json", dict(n_boot=args.n_boot, seed=args.seed, unit="recording", contrasts=boot))

    summary = dict(protocol=config, arms={arm: {point: ev["metrics"] for point, ev in arms.items()}
                                          for arm, arms in evaluations.items()}, bootstrap=boot)
    mge.save(out / "summary.json", summary)
    lines = [f"# Three-way fusion ({args.run.name})", "", "| Arm | AP | AUC | F1 (OOF-selected) |", "|---|---:|---:|---:|"]
    for arm, ev in evaluations.items():
        m = ev["window_f1"]["metrics"]
        lines.append(f"| {arm} | {m['average_precision']:.4f} | {m['auc_raw']:.4f} | {m['binary_f1']:.4f} |")
    lines += ["", "## Paired AP differences (recording bootstrap)", ""]
    lines += [f"- {k}: {v['point']:+.4f} {['%+.4f' % x for x in v['ci95']]}" for k, v in boot.items()]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(boot, indent=1), flush=True)


if __name__ == "__main__":
    main()
