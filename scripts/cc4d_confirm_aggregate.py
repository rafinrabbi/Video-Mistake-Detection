"""Pre-registered aggregation across recogniser seeds.

Each --run is an output directory of scripts/cc4d_matched_graph_experiment.py
for one recogniser seed, all sharing one content detector and test split.

    python scripts/cc4d_confirm_aggregate.py --label r4 \
        --run outputs/confirm_v1/matched_r4_s42 --run outputs/confirm_v1/matched_r4_s1 \
        --run outputs/confirm_v1/matched_r4_s2 --out outputs/confirm_v1/aggregate_r4.json

One recording bootstrap draw is shared by every seed and arm. The estimate is
the seed mean of the paired difference; Holm is applied over the primary pair
(dag-linear, dag-uniform) only when --primary is given.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vmd.data.captaincook4d import CaptainCook4DDataset
from vmd.eval.matched_graph import weighted_ap, weighted_ap_preparation

_spec = importlib.util.spec_from_file_location("mge", ROOT / "scripts/cc4d_matched_graph_experiment.py")
mge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mge)

DEFAULT_ARMS = "content,dag,linear,uniform"
DEFAULT_CONTRASTS = "dag-linear,dag-uniform,dag-content"
TRACKED = ("dag", "linear", "uniform")
TAUS = np.linspace(0, 1, 101)
FA_GRID = np.linspace(0, 1, 101)


def froc_pauc(recall, fa):
    """Mean over FA in [0, 1] of the best recall reachable at <= that FA rate."""
    order = np.argsort(fa)
    fa, best = np.asarray(fa)[order], np.maximum.accumulate(np.asarray(recall)[order])
    idx = np.searchsorted(fa, FA_GRID, side="right") - 1
    return float(np.mean(np.where(idx >= 0, best[np.clip(idx, 0, None)], 0.0)))


def holm(pvalues, alpha=0.05):
    order = sorted(pvalues, key=pvalues.get)
    out, failed = {}, False
    for rank, key in enumerate(order):
        threshold = alpha / (len(order) - rank)
        failed = failed or pvalues[key] > threshold
        out[key] = dict(p_one_sided=pvalues[key], holm_threshold=threshold, confirmed=not failed)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, action="append", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--recipe_holdout", type=Path, default=None)
    ap.add_argument("--primary", action="store_true", help="apply Holm to dag-linear and dag-uniform")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261020)
    # Amendment 2 / S10 runs add arms; the defaults reproduce the registered analysis.
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--contrasts", default=DEFAULT_CONTRASTS, help="comma list of a-b pairs")
    args = ap.parse_args()
    ARMS = tuple(args.arms.split(","))
    CONTRASTS = tuple(tuple(c.split("-")) for c in args.contrasts.split(","))

    clips = {c.clip_id: c for c in CaptainCook4DDataset(
        ROOT / "data/raw/captaincook4d", "test", recipe_holdout=args.recipe_holdout).clips}
    runs = {}
    for run in args.run:
        runs[run.name] = {arm: json.loads((run / f"{arm}_test_preds.json").read_text()) for arm in ARMS}
    ids = sorted({r["clip_id"] for rows in next(iter(runs.values())).values() for r in rows})
    cid_idx = {cid: i for i, cid in enumerate(ids)}
    duration = np.array([clips[cid].duration_sec for cid in ids])

    prepared, window_clip, froc = {}, {}, {}
    point = defaultdict(dict)
    for name, arms in runs.items():
        for arm, rows in arms.items():
            if sorted({r["clip_id"] for r in rows}) != ids:
                raise ValueError(f"{name}/{arm}: recording set differs across runs")
            y = [r["true_label"] for r in rows]
            prepared[name, arm] = weighted_ap_preparation(y, [r["confidence"] for r in rows])
            window_clip[name, arm] = np.array([cid_idx[r["clip_id"]] for r in rows])
            point[name][f"ap_{arm}"] = weighted_ap(prepared[name, arm], np.ones(len(y)))
            # Per-clip event counts at every threshold, for FROC and its bootstrap.
            vec = np.zeros((len(TAUS), 3, len(ids)))
            for t, tau in enumerate(TAUS):
                pc = mge.evaluate(rows, clips, tau)["per_clip"]
                for j, key in enumerate(("n_events", "n_detected", "n_false_alarms")):
                    vec[t, j] = [pc[cid][key] for cid in ids]
            froc[name, arm] = vec
            ones = np.ones(len(ids))
            point[name][f"froc_pauc_{arm}"] = froc_pauc(
                vec[:, 1] @ ones / (vec[:, 0] @ ones), vec[:, 2] @ ones / (duration.sum() / 60))
        types = json.loads((args.run[list(runs).index(name)] / "per_error_type.json").read_text())
        for arm in ARMS:
            point[name][f"order_auc_{arm}"] = types[arm]["order_error"]["auc"]
        tracking = json.loads((args.run[list(runs).index(name)] / "strict_tracking.json").read_text())
        for arm in TRACKED:
            if arm in ARMS:
                point[name][f"tracking_posterior_{arm}"] = tracking[arm]["step_posterior_idx"]
        point[name]["tracking_emission_argmax"] = tracking["dag"]["emission_argmax"]

    rng = np.random.default_rng(args.seed)
    draws = defaultdict(list)
    for _ in range(args.n_boot):
        counts = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)))
        for a, b in CONTRASTS:
            ap_d, fr_d = [], []
            for name in runs:
                vals = {}
                for arm in (a, b):
                    vals[arm] = weighted_ap(prepared[name, arm], counts[window_clip[name, arm]])
                    v = froc[name, arm]
                    vals["fr_" + arm] = froc_pauc(v[:, 1] @ counts / (v[:, 0] @ counts),
                                                  v[:, 2] @ counts / (counts @ duration / 60))
                ap_d.append(vals[a] - vals[b])
                fr_d.append(vals["fr_" + a] - vals["fr_" + b])
            draws[f"ap_{a}_minus_{b}"].append(np.mean(ap_d))
            draws[f"froc_pauc_{a}_minus_{b}"].append(np.mean(fr_d))

    contrasts = {}
    for key, samples in draws.items():
        prefix = "froc_pauc" if key.startswith("froc") else "ap"
        a, b = key[len(prefix) + 1:].split("_minus_")
        est = float(np.mean([point[n][f"{prefix}_{a}"] - point[n][f"{prefix}_{b}"] for n in runs]))
        s = np.asarray(samples)
        contrasts[key] = dict(estimate_seed_mean=est, ci95=np.quantile(s, [.025, .975]).tolist(),
                              p_one_sided=float(np.mean(s <= 0)),
                              per_seed={n: point[n][f"{prefix}_{a}"] - point[n][f"{prefix}_{b}"] for n in runs})
    result = dict(label=args.label, runs=[str(r) for r in args.run], n_recordings=len(ids),
                  n_boot=args.n_boot, seed=args.seed, per_seed=point, contrasts=contrasts,
                  seed_mean={k: float(np.mean([point[n][k] for n in runs])) for k in point[next(iter(runs))]})
    if args.primary:
        result["primary_holm"] = holm({k: contrasts[k]["p_one_sided"]
                                       for k in ("ap_dag_minus_linear", "ap_dag_minus_uniform")})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: {kk: v[kk] for kk in ("estimate_seed_mean", "ci95", "p_one_sided")}
                      for k, v in contrasts.items()}, indent=1))
    if args.primary:
        print(json.dumps(result["primary_holm"], indent=1))


if __name__ == "__main__":
    main()
