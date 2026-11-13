#!/usr/bin/env python
"""How many annotated order errors does the gold partial order flag? CPU only.

For every localizable order_error annotation, take the gold steps whose spans
overlap it. A step is *early* if one of its DAG prerequisites had not started
before it started, and *late* if one of its DAG successors started before it.
An order-error span is DAG-detectable if any overlapping step is early or late;
otherwise it is consistent with the partial order, and no prerequisite check
over the gold graph can flag it. The same audit is run against a linear chain
(one topological extension) for comparison.

    python scripts/cc4d_order_audit.py --split test --out outputs/confirm_v1/order_audit_test.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vmd.data.captaincook4d import CaptainCook4DDataset  # noqa: E402
from vmd.data.step_tracking import match_spans_to_graph  # noqa: E402
from vmd.eval.matched_graph import control_space  # noqa: E402
from vmd.graph.schema import ProceduralGraph  # noqa: E402
from vmd.tracking.belief import build_step_space  # noqa: E402


def flags(clip, graph, space):
    """Menu index -> (start, early, late) for the first occurrence of each gold step."""
    m = match_spans_to_graph(clip.step_spans, graph.steps, order=space.graph_pos)
    first = {}
    for i, sp in enumerate(clip.step_spans):
        if sp.start_sec < 0 or i not in m:
            continue
        node = space.menu_index_of_graph_pos(m[i])
        if node is not None and (node not in first or sp.start_sec < first[node][0]):
            first[node] = (sp.start_sec, sp.end_sec)
    out = {}
    for node, (start, _) in first.items():
        early = any(p not in first or first[p][0] > start for p in space.pred[node])
        late = any(s in first and first[s][0] < start for s in space.succ[node])
        out[node] = (early, late)
    return out, m


def audit(clips, graphs, mode):
    rows = []
    for clip in clips:
        graph = graphs[clip.task_id]
        space = control_space(build_step_space(graph), mode)
        f, m = flags(clip, graph, space)
        for a in clip.annotations:
            if a.mistake_class != "order_error" or a.start_sec < 0 or a.end_sec <= a.start_sec:
                continue
            nodes = {space.menu_index_of_graph_pos(m[i]) for i, sp in enumerate(clip.step_spans)
                     if i in m and sp.start_sec >= 0
                     and min(sp.end_sec, a.end_sec) > max(sp.start_sec, a.start_sec)}
            nodes.discard(None)
            early = any(f.get(n, (False, False))[0] for n in nodes)
            late = any(f.get(n, (False, False))[1] for n in nodes)
            rows.append(dict(clip_id=clip.clip_id, task_id=clip.task_id, start=a.start_sec,
                             end=a.end_sec, n_steps=len(nodes), early=early, late=late))
    # Step-level view: flagged gold steps that overlap no order-error span are
    # legitimate orderings the topology calls violations (false flags).
    steps_flagged = steps_total = false_flags = 0
    for clip in clips:
        graph = graphs[clip.task_id]
        space = control_space(build_step_space(graph), mode)
        f, m = flags(clip, graph, space)
        order_spans = [(a.start_sec, a.end_sec) for a in clip.annotations
                       if a.mistake_class == "order_error" and a.start_sec >= 0 and a.end_sec > a.start_sec]
        for i, sp in enumerate(clip.step_spans):
            if sp.start_sec < 0 or i not in m:
                continue
            node = space.menu_index_of_graph_pos(m[i])
            if node is None or node not in f:
                continue
            steps_total += 1
            if any(f[node]):
                steps_flagged += 1
                if not any(min(sp.end_sec, e) > max(sp.start_sec, s0) for s0, e in order_spans):
                    false_flags += 1
    n = len(rows)
    det = sum(r["early"] or r["late"] for r in rows)
    return dict(n_order_spans=n, violating=det, consistent=n - det,
                early=sum(r["early"] for r in rows), late=sum(r["late"] for r in rows),
                both=sum(r["early"] and r["late"] for r in rows),
                no_overlapping_step=sum(r["n_steps"] == 0 for r in rows),
                gold_steps=steps_total, gold_steps_flagged=steps_flagged,
                flagged_steps_outside_order_errors=false_flags, spans=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    clips = CaptainCook4DDataset(ROOT / "data/raw/captaincook4d", args.split).clips
    graphs = {p.stem: ProceduralGraph.model_validate_json(p.read_text())
              for p in (ROOT / "data/graphs").glob("*.json")}
    result = {"split": args.split, "definition": __doc__.strip().split("\n\n")[1]}
    for mode in ("dag", "linear"):
        result[mode] = audit(clips, graphs, mode)
        r = result[mode]
        print(f"{mode}: {r['violating']}/{r['n_order_spans']} order spans violate "
              f"(early {r['early']}, late {r['late']}, both {r['both']}; "
              f"no overlapping step {r['no_overlapping_step']}); steps flagged "
              f"{r['gold_steps_flagged']}/{r['gold_steps']}, outside order errors "
              f"{r['flagged_steps_outside_order_errors']}")
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
