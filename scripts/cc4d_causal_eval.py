"""Replay saved predictions through causal_v1 without altering historical files.

Old scores require --allow_legacy_scores: replay repairs decision/event semantics,
not the unavailable true/false logits. No tuning or test threshold selection occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vmd.data.captaincook4d import CaptainCook4DDataset
from vmd.eval.causal import evaluate_causal
from vmd.eval.decision_layer import load_preds
from vmd.eval.event_metrics import GoldEvent
from vmd.eval.provenance import file_hash, manifest
from vmd.vlm.boolean_scoring import SCORE_VERSION


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raw_dir", type=Path, default=ROOT / "data/raw/captaincook4d")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--params", type=Path, help="Existing decision settings; no tuning performed")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--smooth_k", type=int, default=1)
    ap.add_argument("--min_consecutive", type=int, default=1)
    ap.add_argument("--clock", choices=["window_end", "serial_inference"], default="window_end")
    ap.add_argument("--allow_legacy_scores", action="store_true")
    ap.add_argument("--allow_subset", action="store_true")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit("Output exists; choose a new versioned path")
    by_clip = load_preds(args.preds)
    versions = {r.get("score_version", "legacy_unspecified") for rows in by_clip.values() for r in rows}
    if versions != {SCORE_VERSION} and not args.allow_legacy_scores:
        raise SystemExit("Legacy/unknown scores require --allow_legacy_scores; they cannot be repaired from scalars")
    ds = CaptainCook4DDataset(args.raw_dir, args.split)
    clips = {c.clip_id: c for c in ds.clips}
    if set(by_clip) - set(clips):
        raise SystemExit("Predictions contain recordings outside the requested split")
    if not args.allow_subset and set(by_clip) != set(clips):
        raise SystemExit("Incomplete split coverage; pass --allow_subset only for declared diagnostics")
    # Reconcile each saved target with the loader rather than mixing label versions.
    for cid, rows in by_clip.items():
        for r in rows:
            mid = (r["start_sec"] + r["end_sec"]) / 2
            if (int(r["true_label"]) != int(clips[cid].is_mistake(mid)) or
                    r["true_class"] != clips[cid].mistake_class_at(mid)):
                raise SystemExit(f"Saved labels differ from the current label policy: {cid} at {mid}")
    gold = {cid: [GoldEvent(a.start_sec, a.end_sec, a.mistake_class)
                  for a in clips[cid].annotations if a.start_sec >= 0 and a.end_sec > a.start_sec]
            for cid in by_clip}
    durations = {cid: clips[cid].duration_sec for cid in by_clip}
    config = dict(tau=args.tau, smooth_k=args.smooth_k, min_consecutive=args.min_consecutive,
                  gate_mode="conf_only", clock=args.clock)
    if args.params:
        params = json.loads(args.params.read_text())
        params = params.get("config", params)
        config.update({k: params[k] for k in ("tau", "smooth_k", "min_consecutive", "gate_mode")})
    result = evaluate_causal(by_clip, gold, durations, **config)
    result["localization"] = {
        str(thr): evaluate_causal(by_clip, gold, durations, criterion="iou", iou_threshold=thr,
                                  **config)["metrics"] for thr in (0.1, 0.25, 0.5)}
    ann = args.raw_dir / "annotations"
    files = [args.preds, Path(__file__)] + list((ROOT / "src/vmd").rglob("*.py"))
    for folder, pattern in (("annotation_json", "*.json"), ("annotation_csv", "*.csv"),
                            ("data_splits", "*.json"), ("metadata", "*.csv")):
        files += list((ann / folder).glob(pattern))
    if args.params:
        files.append(args.params)
    result["manifest"] = manifest({
        "split": args.split, "recording_ids": sorted(by_clip),
        "n_split_recordings": len(clips), "subset": set(by_clip) != set(clips),
        "label_policy": "historical_other_as_normal_midpoint_v1",
        "operating_point_origin": "provided_config_unverified" if args.params else "explicit_pilot_defaults",
        "score_recomputed": False, "legacy_scores": versions != {SCORE_VERSION}, **config,
    }, files)
    source_manifest = args.preds.with_suffix(args.preds.suffix + ".manifest.json")
    result["collection_manifest_sha256"] = file_hash(source_manifest) if source_manifest.exists() else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result["metrics"], indent=2, allow_nan=False))
    print(f"Saved {args.out}; score version: {result['score_version']}")


if __name__ == "__main__":
    main()
