#!/usr/bin/env python
"""Evolve the per-recipe STRUCTURAL PACK the belief filter consumes. CPU only.

The graph enters GraphTrack through the filter, not the prompt, so this is the
place where "better graph construction" can be tested cheaply: a candidate is a
per-recipe pack {edges (add/remove), step duration prior, skip/transit
fractions, emission temperature}; fitness is replayed from saved per-window
step distributions in seconds. A (mu + lambda) loop with random edits is the
control; an LLM proposer can be plugged in via --llm (needs OPENAI_API_KEY).

Fitness is computed on VALIDATION records only; the chosen pack is then
evaluated once on test by the caller (scripts/cc4d_tracking_eval.py and the
fusion/decision pipeline with --graphs_dir pointing at the evolved packs).

    # dry run with gold one-hot emissions (no model needed)
    python scripts/cc4d_evolve_graphs.py --emissions gold --split val --generations 20

    # real run on GraphTrack val records
    python scripts/cc4d_evolve_graphs.py --emissions preds \
        --preds outputs/review_v4/gt_val_preds.json --split val --generations 60 \
        --out_dir data/graphs_evolved_v1
"""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vmd.data.captaincook4d import CaptainCook4DDataset            # noqa: E402
from vmd.data.step_tracking import gold_span_index_at, load_step_seconds, match_spans_to_graph  # noqa: E402
from vmd.eval.decision_layer import load_preds                      # noqa: E402
from vmd.graph.schema import Edge, EdgeKind, ProceduralGraph        # noqa: E402
from vmd.tracking.belief import BeliefFilter, FilterConfig, StepSpace, build_step_space, one_hot  # noqa: E402


def rank_auc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    if y.min() == y.max():
        return float("nan")
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
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


# ------------------------------------------------------------------ packs
def pack_from_graph(graph: ProceduralGraph, step_sec: float) -> dict:
    sp = build_step_space(graph)
    return {"node_ids": sp.node_ids,
            "edges": sorted({(sp.node_ids[i], sp.node_ids[j]) for i, ss in enumerate(sp.succ) for j in ss}),
            "step_seconds": float(step_sec), "skip_frac": 0.1, "transit_frac": 0.3, "emission_temp": 1.0,
            "visit_hysteresis": 2}


def graph_from_pack(graph: ProceduralGraph, pack: dict) -> ProceduralGraph:
    """Same steps/sentinels as the gold graph, edges replaced by the pack's."""
    g = copy.deepcopy(graph)
    sentinels = [s.id for s in g.steps if s.is_optional]
    edges = [Edge(src=a, dst=b, kind=EdgeKind.after) for a, b in pack["edges"]]
    # keep START/END attached so the validator and roots/sinks stay well defined
    ids = set(pack["node_ids"]); has_pred = {b for _, b in pack["edges"]}; has_succ = {a for a, _ in pack["edges"]}
    start = [s for s in sentinels if s.upper() == "START" or g.steps[0].id == s]
    end = [s for s in sentinels if s not in start]
    for n in ids:
        if n not in has_pred and start:
            edges.append(Edge(src=start[0], dst=n, kind=EdgeKind.after))
        if n not in has_succ and end:
            edges.append(Edge(src=n, dst=end[0], kind=EdgeKind.after))
    g.edges = edges
    return g


def mutate(pack: dict, rng: random.Random) -> dict:
    p = copy.deepcopy(pack); ids = p["node_ids"]; edges = set(map(tuple, p["edges"]))
    r = rng.random()
    if r < 0.3 and len(edges) > len(ids) - 1:            # drop an edge
        edges.discard(rng.choice(sorted(edges)))
    elif r < 0.55:                                        # add a forward-consistent edge
        a, b = rng.sample(ids, 2)
        if ids.index(a) < ids.index(b):
            edges.add((a, b))
    elif r < 0.7:
        p["step_seconds"] = float(np.clip(p["step_seconds"] * rng.uniform(0.7, 1.4), 8.0, 120.0))
    elif r < 0.8:
        p["skip_frac"] = float(np.clip(p["skip_frac"] * rng.uniform(0.5, 2.0), 0.01, 0.5))
    elif r < 0.9:
        p["emission_temp"] = float(np.clip(p["emission_temp"] * rng.uniform(0.6, 1.6), 0.25, 4.0))
    else:
        p["transit_frac"] = float(np.clip(p["transit_frac"] * rng.uniform(0.5, 2.0), 0.05, 0.8))
    p["edges"] = sorted(edges)
    return p


def _acyclic(pack: dict) -> bool:
    ids = pack["node_ids"]; pos = {n: i for i, n in enumerate(ids)}
    return all(pos[a] < pos[b] for a, b in pack["edges"])


# ---------------------------------------------------------------- fitness
def recipe_fitness(pack: dict, graph: ProceduralGraph, clips, emissions, signal="illegal_hold") -> dict:
    """Order-vs-normal window AUC of the structural signal, plus step accuracy."""
    g = graph_from_pack(graph, pack); sp = build_step_space(g)
    cfg = FilterConfig.from_step_seconds(pack["step_seconds"], 4.0, skip_frac=pack["skip_frac"],
                                         transit_frac=pack["transit_frac"], emission_temp=pack["emission_temp"],
                                         visit_hysteresis=pack["visit_hysteresis"])
    ys, cls, sig, hit = [], [], [], []
    for clip in clips:
        f = BeliefFilter(sp, cfg)
        for e, y, c, gold in emissions[clip.clip_id]:
            hit.append(f.argmax == gold if gold is not None else None)
            st = f.update(e)
            ys.append(y); cls.append(c); sig.append(getattr(st, signal))
    cls = np.array(cls); sig = np.array(sig); ys = np.array(ys)
    m = (cls == "no_mistake") | (cls == "order_error")
    hits = [h for h in hit if h is not None]
    return {"order_auc": rank_auc((cls[m] == "order_error"), sig[m]), "all_auc": rank_auc(ys, sig),
            "step_acc": float(np.mean(hits)) if hits else float("nan"), "n_windows": int(len(ys))}


def build_emissions(clips, graphs, by_clip, mode):
    """Per window: (emission vector, label, class, gold menu idx) in the GOLD graph's menu order.

    Packs only change edges/hyper-parameters, never the node set or its order,
    so emissions indexed by the gold menu stay valid for every candidate.
    """
    out = {}
    for clip in clips:
        g = graphs[clip.task_id]; sp = build_step_space(g)
        m = match_spans_to_graph(clip.step_spans, g.steps, order=sp.graph_pos)
        recs = sorted(by_clip[clip.clip_id], key=lambda r: r["start_sec"])
        rows = []
        for r in recs:
            mid = (r["start_sec"] + r["end_sec"]) / 2.0
            i = gold_span_index_at(clip.step_spans, mid, strict=True)
            gold = sp.menu_index_of_graph_pos(m[i]) if (i is not None and i in m) else None
            if mode == "gold":
                e = one_hot(gold, sp.n, 0.95)
            else:
                e = np.asarray(r["step_probs"], float)
            rows.append((e, int(r["true_label"]), r["true_class"], gold))
        out[clip.clip_id] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emissions", choices=["gold", "preds"], default="gold")
    ap.add_argument("--preds", default=str(ROOT / "outputs/full_eval_8class/lora_val_preds.json"),
                    help="window grid (gold mode) or GraphTrack records with step_probs (preds mode)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--raw_dir", default=str(ROOT / "data/raw/captaincook4d"))
    ap.add_argument("--graphs_dir", default=str(ROOT / "data/graphs"))
    ap.add_argument("--step_seconds", default=str(ROOT / "data/step_seconds.json"))
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--children", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--signal", default="illegal_hold")
    ap.add_argument("--out_dir", default=str(ROOT / "data/graphs_evolved_v1"))
    ap.add_argument("--report", default=str(ROOT / "outputs/review_v4/graph_evolution.json"))
    args = ap.parse_args()

    ds = CaptainCook4DDataset(args.raw_dir, args.split)
    graphs = {p.stem: ProceduralGraph.model_validate_json(p.read_text()) for p in Path(args.graphs_dir).glob("*.json")}
    step_sec = load_step_seconds(Path(args.step_seconds))
    by_clip = load_preds(Path(args.preds))
    clips_by_recipe = {}
    for c in ds.clips:
        if c.clip_id in by_clip and c.task_id in graphs:
            clips_by_recipe.setdefault(c.task_id, []).append(c)
    rng = random.Random(args.seed)
    report = {"emissions": args.emissions, "preds": args.preds, "split": args.split, "signal": args.signal,
              "generations": args.generations, "recipes": {}}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for recipe, clips in sorted(clips_by_recipe.items()):
        g = graphs[recipe]
        em = build_emissions(clips, graphs, by_clip, args.emissions)
        base = pack_from_graph(g, step_sec.get(recipe, 35.0))
        f0 = recipe_fitness(base, g, clips, em, args.signal)
        pop = [(f0["order_auc"] if not np.isnan(f0["order_auc"]) else f0["all_auc"], base)]
        history = [pop[0][0]]
        for gen in range(args.generations):
            kids = []
            for _ in range(args.children):
                parent = rng.choice(pop)[1]
                child = mutate(parent, rng)
                if not _acyclic(child):
                    continue
                fc = recipe_fitness(child, g, clips, em, args.signal)
                score = fc["order_auc"] if not np.isnan(fc["order_auc"]) else fc["all_auc"]
                if not np.isnan(score):
                    kids.append((score, child))
            pop = sorted(pop + kids, key=lambda x: -x[0])[: args.population]
            history.append(pop[0][0])
        best_score, best = pop[0]
        fb = recipe_fitness(best, g, clips, em, args.signal)
        gold_edges = set(map(tuple, base["edges"])); best_edges = set(map(tuple, best["edges"]))
        tp = len(gold_edges & best_edges)
        report["recipes"][recipe] = {
            "n_recordings": len(clips), "baseline": f0, "evolved": fb, "fitness_history": history,
            "edge_precision_vs_gold": tp / len(best_edges) if best_edges else float("nan"),
            "edge_recall_vs_gold": tp / len(gold_edges) if gold_edges else float("nan"),
            "pack": best}
        (Path(args.out_dir) / f"{recipe}.json").write_text(graph_from_pack(g, best).model_dump_json(indent=1))
        (Path(args.out_dir) / f"{recipe}.pack.json").write_text(json.dumps(best, indent=1))
        print(f"{recipe:30s} base order-AUC {f0['order_auc']:.3f} -> evolved {fb['order_auc']:.3f} "
              f"(edges P {report['recipes'][recipe]['edge_precision_vs_gold']:.2f} R {report['recipes'][recipe]['edge_recall_vs_gold']:.2f})", flush=True)
    Path(args.report).write_text(json.dumps(report, indent=1))
    print("wrote", args.report, "and packs in", args.out_dir)


if __name__ == "__main__":
    main()
