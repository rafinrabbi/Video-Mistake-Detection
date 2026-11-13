#!/usr/bin/env python
"""E0: what the graph alone can detect, given the true step sequence. CPU only.

Feeds GOLD step emissions (one-hot on the annotated step, transit in gaps)
through the DAG-constrained belief filter and asks whether its structural
signals separate order-error windows from normal ones. No model is involved,
so this is the upper bound of the structural term and, via the noise sweep,
the step-recognition accuracy the term needs to stay useful.

    python scripts/cc4d_structural_oracle.py --split test \
        --preds outputs/full_eval_8class/lora_test_preds.json \
        --out outputs/review_v4/structural_oracle.json

``--preds`` only supplies the window grid and labels, so the oracle rows line
up window-for-window with every saved arm.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vmd.data.captaincook4d import CaptainCook4DDataset            # noqa: E402
from vmd.data.step_tracking import (gold_span_index_at, load_step_seconds,  # noqa: E402
                                    match_spans_to_graph)
from vmd.eval.decision_layer import load_preds                      # noqa: E402
from vmd.graph.schema import ProceduralGraph                        # noqa: E402
from vmd.tracking.belief import BeliefFilter, FilterConfig, build_step_space, one_hot  # noqa: E402
from vmd.eval.matched_graph import control_space  # noqa: E402

SIGNALS = ("surprise", "illegal_occupancy", "illegal_hold", "violation_mass")


def rank_auc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    if y.min() == y.max():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    ss = s[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    npos = y.sum(); nneg = len(y) - npos
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def gold_menu_sequence(clip, graph, space):
    """Menu index of the gold step at each window midpoint (None in gaps)."""
    m = match_spans_to_graph(clip.step_spans, graph.steps, order=space.graph_pos)
    def at(t):
        i = gold_span_index_at(clip.step_spans, t, strict=True)
        if i is None or i not in m:
            return None
        return space.menu_index_of_graph_pos(m[i])
    return at


def run_arm(by_clip, clips, graphs, step_sec, acc, mode, rng, hysteresis=2, matched=False):
    """One pass over the split; returns flat per-window arrays + per-recording flags."""
    ys, cls, sig = [], [], {k: [] for k in SIGNALS}
    rec_rows = []
    for cid, recs in by_clip.items():
        clip = clips[cid]; graph = graphs[clip.task_id]
        space = build_step_space(graph)
        # matched: start prior and adjacency follow the control topology too
        # (the default keeps DAG roots in every mode, as in review_v4).
        fspace = control_space(space, mode) if matched else space
        cfg = FilterConfig.from_step_seconds(step_sec.get(clip.task_id), 4.0,
                                             mode=mode, visit_hysteresis=hysteresis)
        filt = BeliefFilter(fspace, cfg)
        gold_at = gold_menu_sequence(clip, graph, space)
        peak = {k: 0.0 for k in SIGNALS}
        for r in sorted(recs, key=lambda r: r["start_sec"]):
            g = gold_at((r["start_sec"] + r["end_sec"]) / 2.0)
            if acc < 1.0 and rng.random() > acc:          # recognition error
                wrong = [j for j in range(space.n) if j != g]
                g = int(rng.choice(wrong))
            st = filt.update(one_hot(g, space.n, 0.95))
            ys.append(int(r["true_label"])); cls.append(r["true_class"])
            for k in SIGNALS:
                v = getattr(st, k); sig[k].append(v); peak[k] = max(peak[k], v)
        rec_rows.append({"clip_id": cid, "has_error": any(a.start_sec >= 0 for a in clip.annotations),
                         "has_order": any(a.mistake_class == "order_error" and a.start_sec >= 0
                                          for a in clip.annotations), **{f"peak_{k}": peak[k] for k in SIGNALS}})
    ys = np.array(ys); cls = np.array(cls)
    out = {}
    normal = cls == "no_mistake"
    for k in SIGNALS:
        s = np.array(sig[k])
        out[k] = {
            "auc_all_vs_normal": rank_auc(ys, s),
            "auc_order_vs_normal": rank_auc((cls == "order_error")[normal | (cls == "order_error")],
                                            s[normal | (cls == "order_error")]),
            "auc_nonorder_vs_normal": rank_auc(((~normal) & (cls != "order_error"))[normal | ((~normal) & (cls != "order_error"))],
                                               s[normal | ((~normal) & (cls != "order_error"))]),
            "mean_on_order": float(s[cls == "order_error"].mean()),
            "mean_on_normal": float(s[normal].mean()),
            "frac_normal_windows_above_0.5": float((s[normal] > 0.5).mean()),
        }
        # recording level: peak signal separates order-error recordings from clean ones
        pk = np.array([r[f"peak_{k}"] for r in rec_rows])
        clean = np.array([not r["has_error"] for r in rec_rows])
        order = np.array([r["has_order"] for r in rec_rows])
        out[k]["recording_auc_order_vs_clean"] = rank_auc(order[clean | order], pk[clean | order])
    out["n_windows"] = int(len(ys)); out["n_recordings"] = len(rec_rows)
    return out, sig, rec_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--preds", default=str(ROOT / "outputs/full_eval_8class/lora_test_preds.json"))
    ap.add_argument("--raw_dir", default=str(ROOT / "data/raw/captaincook4d"))
    ap.add_argument("--graphs_dir", default=str(ROOT / "data/graphs"))
    ap.add_argument("--step_seconds", default=str(ROOT / "data/step_seconds.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs/review_v4/structural_oracle.json"))
    ap.add_argument("--save_windows", default=str(ROOT / "outputs/review_v4/structural_oracle_windows.json"),
                    help="per-window gold-emission signals (acc=1, dag) for fusion/plots")
    ap.add_argument("--accuracies", default="1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3")
    ap.add_argument("--matched", action="store_true",
                    help="matched control topologies and a noise sweep for every mode")
    args = ap.parse_args()

    ds = CaptainCook4DDataset(args.raw_dir, args.split)
    clips = {c.clip_id: c for c in ds.clips}
    graphs = {p.stem: ProceduralGraph.model_validate_json(p.read_text())
              for p in Path(args.graphs_dir).glob("*.json")}
    step_sec = load_step_seconds(Path(args.step_seconds))
    by_clip = load_preds(Path(args.preds))

    result = {"schema": "structural_oracle_v1", "split": args.split, "preds_grid": args.preds,
              "emission": "one-hot(0.95) on gold step at window midpoint; transit in gaps",
              "modes": {}, "noise_sweep": {}}
    windows_dump = None
    for mode in ("dag", "linear", "uniform"):
        res, sig, rec_rows = run_arm(by_clip, clips, graphs, step_sec, 1.0, mode, np.random.default_rng(0),
                                     matched=args.matched)
        result["modes"][mode] = res
        if mode == "dag":
            windows_dump = {"signals": sig, "recordings": rec_rows}
        print(f"[{mode}] order-vs-normal AUC: " +
              ", ".join(f"{k}={res[k]['auc_order_vs_normal']:.3f}" for k in SIGNALS), flush=True)
    sweep_modes = ("dag", "linear", "uniform") if args.matched else ("dag",)
    for mode, acc in [(m, float(a)) for m in sweep_modes for a in args.accuracies.split(",")]:
        runs = []
        for seed in range(3):
            res, _, _ = run_arm(by_clip, clips, graphs, step_sec, acc, mode, np.random.default_rng(seed),
                                matched=args.matched)
            runs.append(res)
        key = f"{mode}/{acc:.2f}" if args.matched else f"{acc:.2f}"
        result["noise_sweep"][key] = {
            k: {m: float(np.mean([r[k][m] for r in runs]))
                for m in ("auc_order_vs_normal", "auc_all_vs_normal", "frac_normal_windows_above_0.5",
                          "recording_auc_order_vs_clean")}
            for k in SIGNALS}
        print(f"[{key}] " + ", ".join(
            f"{k}: order AUC {result['noise_sweep'][key][k]['auc_order_vs_normal']:.3f}"
            for k in SIGNALS), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    if windows_dump is not None and args.save_windows:
        Path(args.save_windows).write_text(json.dumps(windows_dump))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
