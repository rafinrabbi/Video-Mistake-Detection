"""Exploratory matched-topology CPU experiment on cached CC4D predictions.

Defaults freeze all filter settings, use five recording folds on cached VAL,
and assess all TEST arms without any test-based parameter selection.
Historical content scores are shared, not silently upgraded to boolean_pair_v1.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vmd.data.captaincook4d import CaptainCook4DDataset
from vmd.data.step_tracking import load_step_seconds
from vmd.eval.causal import evaluate_causal, validate_records
from vmd.eval.event_metrics import GoldEvent
from vmd.eval.fusion import FEATURES, apply_fusion, fit_fusion
from vmd.eval.matched_graph import (crossfit_fusion, recording_folds, replay_clip,
                                   weighted_ap, weighted_ap_preparation)
from vmd.eval.provenance import manifest
from vmd.graph.schema import ProceduralGraph
from vmd.tracking.belief import build_step_space


def save(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)


def grouped(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["clip_id"]].append(r)
    return {cid: sorted(rr, key=lambda r: r["start_sec"]) for cid, rr in sorted(by.items())}


def evaluate(rows, clips, tau):
    by = grouped(rows)
    events = {cid: [GoldEvent(a.start_sec, a.end_sec, a.mistake_class)
                    for a in clips[cid].annotations if a.start_sec >= 0 and a.end_sec > a.start_sec]
              for cid in by}
    durations = {cid: clips[cid].duration_sec for cid in by}
    # Keep the derivation explicit and the same across all arms.
    by = {cid: [dict(r, score_version="matched_fusion_legacy_content_v1",
                     score_status="ok", parse_ok=True) for r in rr] for cid, rr in by.items()}
    return evaluate_causal(by, events, durations, tau=float(tau), smooth_k=1,
                           min_consecutive=1, gate_mode="conf_only", clock="window_end")


def select_operating_points(rows, clips):
    grid = []
    for tau in np.linspace(0, 1, 101):
        metrics = evaluate(rows, clips, tau)["metrics"]
        grid.append({"tau": float(tau), **metrics})
    best = max(grid, key=lambda m: (m["binary_f1"], -m["false_alarms_per_min"], m["tau"]))
    points = {"window_f1": best}
    for budget in (.1, .2):
        eligible = [m for m in grid if m["false_alarms_per_min"] <= budget]
        points[f"fa_budget_{budget}"] = max(
            eligible, key=lambda m: (m["event_recall"], m["event_precision"],
                                      -m["false_alarms_per_min"], m["tau"]))
    return points, grid


def strict_tracking(rows):
    labeled = [r for r in rows if r["gold_step_idx_strict"] >= 0]
    return {"n_strict_labeled": len(labeled), **{
        field: sum(r[field] == r["gold_step_idx_strict"] for r in labeled) / len(labeled)
        for field in ("emission_argmax", "step_prior_idx", "step_posterior_idx")}}


def paired_bootstrap(arms, evaluations, clips, n_boot, seed):
    names = list(arms)
    ids = sorted(grouped(arms[names[0]]))
    cid_idx = {cid: i for i, cid in enumerate(ids)}
    rows0 = arms[names[0]]
    window_clip = np.array([cid_idx[r["clip_id"]] for r in rows0])
    y = np.array([r["true_label"] for r in rows0])
    prepared = {name: weighted_ap_preparation(y, [r["confidence"] for r in rows])
                for name, rows in arms.items()}
    ap_samples = {name: [] for name in names}
    er_samples = {name: [] for name in names}
    fa_samples = {name: [] for name in names}
    duration = np.array([clips[cid].duration_sec for cid in ids])
    event_vectors = {}
    for name in names:
        pc = evaluations[name]["fa_budget_0.2"]["per_clip"]
        event_vectors[name] = {key: np.array([pc[cid][key] for cid in ids])
                               for key in ("n_events", "n_detected", "n_false_alarms")}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        counts = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)))
        w = counts[window_clip]
        for name in names:
            ap_samples[name].append(weighted_ap(prepared[name], w))
            v = event_vectors[name]
            er_samples[name].append(float(counts @ v["n_detected"] / (counts @ v["n_events"])))
            fa_samples[name].append(float(counts @ v["n_false_alarms"] / (counts @ duration / 60)))
    comparisons = {}
    for other in ("linear", "uniform", "content"):
        key = f"dag_minus_{other}"
        comparisons[key] = {}
        for metric, samples in [("average_precision", ap_samples),
                                ("event_recall_at_val_fa_budget_0.2", er_samples),
                                ("false_alarms_per_min_at_val_fa_budget_0.2", fa_samples)]:
            delta = np.asarray(samples["dag"]) - np.asarray(samples[other])
            if metric == "average_precision":
                point = weighted_ap(prepared["dag"], np.ones(len(y))) - weighted_ap(prepared[other], np.ones(len(y)))
            else:
                field = "event_recall" if metric.startswith("event") else "false_alarms_per_min"
                point = (evaluations["dag"]["fa_budget_0.2"]["metrics"][field]
                         - evaluations[other]["fa_budget_0.2"]["metrics"][field])
            comparisons[key][metric] = {"point": point,
                                        "ci95": np.quantile(delta, [.025, .975]).tolist()}
    return {"n_boot": n_boot, "seed": seed, "unit": "recording", "comparisons": comparisons}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source_dir", type=Path, default=ROOT / "outputs/review_v4")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261020)
    # Explicit sources override <source_dir>/{stepreco,gt2}_<split>_preds.json.
    # Content files without task_id (raw detector passes) take it from the clip.
    for split in ("val", "test"):
        ap.add_argument(f"--stepreco_{split}", type=Path, default=None)
        ap.add_argument(f"--content_{split}", type=Path, default=None)
    ap.add_argument("--recipe_holdout", type=Path, default=None)
    ap.add_argument("--step_seconds", type=Path, default=ROOT / "data/step_seconds.json")
    ap.add_argument("--status", default="exploratory: TEST already examined; shared historical content scores")
    ap.add_argument("--title", default="Matched graph replay — 20 October 2026")
    args = ap.parse_args()
    step_src = {sp: getattr(args, f"stepreco_{sp}") or args.source_dir / f"stepreco_{sp}_preds.json"
                for sp in ("val", "test")}
    content_src = {sp: getattr(args, f"content_{sp}") or args.source_dir / f"gt2_{sp}_preds.json"
                   for sp in ("val", "test")}
    args.out_dir.mkdir(parents=True, exist_ok=False)
    out = args.out_dir
    raw = ROOT / "data/raw/captaincook4d"
    graph_paths = sorted((ROOT / "data/graphs").glob("*.json"))
    graphs = {p.stem: ProceduralGraph.model_validate_json(p.read_text()) for p in graph_paths}
    step_seconds_path = args.step_seconds
    step_seconds = load_step_seconds(step_seconds_path)
    source_files = [step_src[sp] for sp in ("val", "test")] + [content_src[sp] for sp in ("val", "test")]
    if args.recipe_holdout:
        source_files.append(args.recipe_holdout)
    config = dict(seed=args.seed, n_boot=args.n_boot, filter_settings="unchanged defaults, recipe TRAIN duration",
                  modes=["dag", "linear", "uniform"], features=list(FEATURES),
                  fusion="same L2 logistic; 5 recording folds for VAL threshold scores; final fit all cached VAL",
                  thresholds="0..1 by .01; maximize OOF window F1 or OOF event recall at FA/min <= 0.1,0.2",
                  decisions="causal_v1; k=1; persistence=1; confidence only; window-end clock",
                  validation_coverage="all recordings present in the supplied VAL sources (audited below)",
                  sources={sp: dict(stepreco=str(step_src[sp]), content=str(content_src[sp])) for sp in step_src},
                  recipe_holdout=str(args.recipe_holdout) if args.recipe_holdout else None,
                  primary_comparisons=["dag_minus_linear", "dag_minus_uniform"],
                  status=args.status,
                  timing="same-window posterior for every mode; report prior separately",
                  topology="linear uses chain roots and edges; uniform uses all-node roots and no prerequisites",
                  no_graph_definition="uniform temporal filter; keeps visitation/transit bookkeeping, removes procedure edges")
    ann_files = [p for p in (raw / "annotations").rglob("*") if p.is_file() and p.suffix in (".json", ".csv")]
    sources = source_files + graph_paths + [step_seconds_path, Path(__file__)] + ann_files
    sources += list((ROOT / "src/vmd").rglob("*.py"))
    sources += [p.with_suffix(p.suffix + ".manifest.json") for p in source_files
                if p.with_suffix(p.suffix + ".manifest.json").exists()]
    save(out / "protocol_manifest.json", manifest(config, sources))
    print("Protocol frozen before replay; sources hashed", flush=True)

    clips_by_split = {}
    replays = {}
    audit = {}
    for split in ("val", "test"):
        clips = {c.clip_id: c for c in CaptainCook4DDataset(raw, split, recipe_holdout=args.recipe_holdout).clips}
        clips_by_split[split] = clips
        steps = grouped(json.loads(step_src[split].read_text()))
        content = grouped([r if "task_id" in r else dict(r, task_id=clips[r["clip_id"]].task_id)
                           for r in json.loads(content_src[split].read_text())])
        if set(steps) != set(content) or set(steps) - set(clips):
            raise ValueError("Recording membership mismatch")
        if split == "test" and set(steps) != set(clips):
            raise ValueError("Incomplete TEST coverage")
        modes = {mode: [] for mode in ("dag", "linear", "uniform")}
        for cid in steps:
            c = clips[cid]
            rr = content[cid]
            validate_records(rr)
            expected = max(1, int(np.floor((c.duration_sec - 4.) / 4.)) + 1)
            if len(rr) != expected:
                raise ValueError(f"Incomplete window grid: {cid}: {len(rr)} != {expected}")
            for i, r in enumerate(rr):
                mid = (r["start_sec"] + r["end_sec"]) / 2
                if (r["start_sec"] != i * 4 or abs(r["end_sec"] - min(i * 4 + 4, c.duration_sec)) > 1e-6
                        or int(r["true_label"]) != int(c.is_mistake(mid))
                        or r["true_class"] != c.mistake_class_at(mid)):
                    raise ValueError(f"Grid/label mismatch: {cid}, {i}")
            space = build_step_space(graphs[c.task_id])
            for mode in modes:
                modes[mode].extend(replay_clip(steps[cid], rr, space, step_seconds.get(c.task_id), mode))
        replays[split] = modes
        audit[split] = dict(n_recordings=len(steps), n_split_recordings=len(clips),
                            n_windows=sum(map(len, content.values())), recording_ids=sorted(steps),
                            source_score_versions=sorted({r.get("score_version", "legacy_unspecified")
                                                          for rows in content.values() for r in rows}))
        print(f"{split}: {audit[split]['n_recordings']} recordings replayed in all 3 modes", flush=True)
    save(out / "coverage_audit.json", audit)
    folds = recording_folds(audit["val"]["recording_ids"], seed=args.seed)
    save(out / "validation_folds.json", folds)

    test_arms, evaluations, models, selected, tracking = {}, {}, {}, {}, {}
    for name in ("content", "dag", "linear", "uniform"):
        mode = "uniform" if name == "content" else name
        features = ("content_logit",) if name == "content" else FEATURES
        val = replays["val"][mode]
        test = replays["test"][mode]
        oof, fold_params = crossfit_fusion(val, features, folds)
        points, sweep = select_operating_points(oof, clips_by_split["val"])
        model = fit_fusion(val, features)
        final = apply_fusion(test, model)
        for r in final:
            r.update(score_version="matched_fusion_legacy_content_v1", score_status="ok", parse_ok=True)
        models[name] = {"final_fit": model, "crossfit": fold_params}
        selected[name] = points
        test_arms[name] = final
        evaluations[name] = {key: evaluate(final, clips_by_split["test"], p["tau"])
                             for key, p in points.items()}
        tracking[name] = strict_tracking(test) if name != "content" else None
        save(out / f"{name}_test_preds.json", final)
        save(out / f"{name}_oof_val_preds.json", oof)
        save(out / f"{name}_val_sweep.json", sweep)
        save(out / f"{name}_causal_test.json", evaluations[name])
        m = evaluations[name]["window_f1"]["metrics"]
        print(f"{name}: TEST AP={m['average_precision']:.4f}; window F1={m['binary_f1']:.4f}", flush=True)
    save(out / "fusion_models.json", models)
    save(out / "selected_operating_points.json", selected)
    save(out / "strict_tracking.json", tracking)
    boot = paired_bootstrap(test_arms, evaluations, clips_by_split["test"], args.n_boot, args.seed)
    save(out / "paired_bootstrap.json", boot)

    # Recipe and mistake-type diagnostics use the same frozen test probabilities.
    recipe = {}
    for name, rows in test_arms.items():
        by_task = defaultdict(list)
        for r in rows:
            by_task[r["task_id"]].append(r)
        recipe[name] = {task: weighted_ap(weighted_ap_preparation(
            [r["true_label"] for r in rr], [r["confidence"] for r in rr]), np.ones(len(rr)))
            for task, rr in sorted(by_task.items())}
    save(out / "per_recipe_ap.json", recipe)
    types = {}
    for name, rows in test_arms.items():
        types[name] = {}
        for typ in sorted({r["true_class"] for r in rows} - {"no_mistake"}):
            rr = [r for r in rows if r["true_class"] in (typ, "no_mistake")]
            y = [r["true_class"] == typ for r in rr]
            types[name][typ] = dict(n_positive=sum(y), auc=float(roc_auc_score(y, [r["confidence"] for r in rr])))
    save(out / "per_error_type.json", types)
    with (out / "pr_curves.csv").open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arm", "precision", "recall", "threshold"])
        for name, rows in test_arms.items():
            precision, recall, tau = precision_recall_curve([r["true_label"] for r in rows],
                                                           [r["confidence"] for r in rows])
            for i in range(len(precision)):
                writer.writerow([name, precision[i], recall[i], tau[i] if i < len(tau) else ""])
    summary = dict(protocol=config, coverage=audit, strict_tracking=tracking,
                   arms={name: {point: ev["metrics"] for point, ev in arms.items()}
                         for name, arms in evaluations.items()}, bootstrap=boot)
    save(out / "summary.json", summary)
    lines = [f"# {args.title}", "", f"Status: {args.status}.",
             f"Validation: {audit['val']['n_recordings']} recordings, five recording folds. "
             f"Test: {audit['test']['n_recordings']} recordings. All graph filters fixed.",
             "Each arm uses the same fusion fitting procedure; thresholds selected on out-of-fold validation scores.",
             "Window-end timing excludes inference/transport. No smoothing or persistence beyond one window.", "",
             "| Arm | AP | AUC | F1 at OOF-selected threshold | Event recall at VAL FA budget .2 | Actual TEST FA/min |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, ev in evaluations.items():
        m, e = ev["window_f1"]["metrics"], ev["fa_budget_0.2"]["metrics"]
        lines.append(f"| {name} | {m['average_precision']:.4f} | {m['auc_raw']:.4f} | {m['binary_f1']:.4f} | {e['event_recall']:.4f} | {e['false_alarms_per_min']:.4f} |")
    lines += ["", "Validation FA budgets do not guarantee matched realized TEST false-alarm rates.", "",
              "## Paired differences (DAG minus control)", ""]
    for key, v in boot["comparisons"].items():
        d = v["average_precision"]
        lines.append(f"- {key}: AP {d['point']:+.4f}, recording-bootstrap 95% CI {d['ci95']}.")
    lines += ["", "## Limits", "",
              "- Historical scalar scores are shared across arms; corrected boolean scores require fresh inference.",
              "- Dedicated recognizer was trained with additional step supervision; the uniform arm controls that access.",
              "- Linear is one deterministic topological extension. Uniform retains generic temporal visitation bookkeeping.",
              "- Same-dimensional feature menus can have constant features when prerequisites are absent.",
              "- One training seed, partial validation coverage, and previously used test set; no confirmatory novelty claim.",
              "- Completion is still approximated by repeated step recognition, rather than explicit completion uncertainty."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(boot["comparisons"], indent=2), flush=True)
    print(f"Completed: {out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
