"""Decision layer: gate → smooth → metrics, replayable from saved predictions.

Single source of truth for how per-window VLM predictions become detection
decisions and metrics. Two consumers:

* ``vmd.eval.online`` applies it to freshly collected windows;
* the re-gating tools (``scripts/cc4d_decision_eval.py``, the val sweep, the
  bootstrap) apply it to a saved ``*_preds.json``, reproducing a GPU run's
  numbers exactly without touching the model.

Reproduction contract (must match the historical eval to float precision):
  1. gate:   flag = deviation AND confidence >= tau        (gate_mode="and")
             flag = confidence >= tau                      (gate_mode="conf_only")
  2. score:  gated = confidence if flag else 0.0
  3. typed:  pred_class = mistake_class if flag else "no_mistake"  — UNsmoothed
  4. binary: median_smooth(gated, k) → pred_bin = int(smoothed > 0.0)
  5. loc:    detect_start_of_deviation(smoothed, starts, 1e-9, min_consecutive)
  6. ECE/Brier take the RAW confidence stream; "auc_roc" takes the gated score
     stream (legacy, τ-dependent); "auc_raw" takes raw confidences (τ-free).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vmd.data import get_dataset
from vmd.data.mistake_classes import MISTAKE_CLASSES
from vmd.decision.postprocess import median_smooth, detect_start_of_deviation
from vmd.eval.metrics import (
    auc_roc, compute_all, early_detection_score, macro_f1, start_time_mae,
)

# Smoothed score arrays are already gated (0.0 when not flagged, confidence when
# flagged), so any positive value marks a fired window for localization.
FIRED_EPS = 1e-9

GATE_MODES = ("and", "conf_only")


def load_preds(path: Path) -> Dict[str, List[dict]]:
    """Group saved per-window records by clip, preserving file order.

    Records were appended clip-by-clip in ascending start_sec, so file order
    within a clip is temporal order — required for smoothing/localization.
    Extra legacy keys (``features``, ``tau_adaptive``, ``tau_fixed``) are
    tolerated and ignored.
    """
    records = json.loads(Path(path).read_text())
    by_clip: Dict[str, List[dict]] = {}
    for r in records:
        by_clip.setdefault(r["clip_id"], []).append(r)
    return by_clip


def gold_events(raw_dir: Path, split: str) -> Dict[str, Optional[Tuple[float, float]]]:
    """Earliest LOCALIZABLE gold mistake per clip as (start, end), or None.

    start < 0 (nonlocalizable, e.g. Missing-Step) marks no window and is
    skipped, mirroring the collection-time ``_gold_event``.
    """
    ds = get_dataset("captaincook4d", raw_dir, split)
    events: Dict[str, Optional[Tuple[float, float]]] = {}
    for clip in ds.clips:
        loc = [(a.start_sec, a.end_sec) for a in clip.annotations if a.start_sec >= 0]
        events[clip.clip_id] = min(loc, key=lambda se: se[0]) if loc else None
    return events


def gate_clip(
    records: List[dict],
    tau: float,
    smooth_k: int,
    gate_mode: str = "and",
) -> dict:
    """Run steps 1–4 of the contract on one clip's windows (file order)."""
    if gate_mode not in GATE_MODES:
        raise ValueError(f"gate_mode {gate_mode!r} not in {GATE_MODES}")
    gated: List[float] = []
    pred_cls: List[str] = []
    for r in records:
        conf = float(r["confidence"])
        if gate_mode == "and":
            flag = bool(r["deviation"]) and conf >= tau
        else:
            flag = conf >= tau
        gated.append(conf if flag else 0.0)
        pred_cls.append(r["mistake_class"] if flag else "no_mistake")
    smoothed = median_smooth(gated, smooth_k)
    return {
        "gated": gated,
        "smoothed": smoothed,
        "pred_bin": [int(s > 0.0) for s in smoothed],
        "pred_cls": pred_cls,
        "starts": [float(r["start_sec"]) for r in records],
    }


def localization_metrics(events: List[tuple]) -> dict:
    """EDS + start-time MAE over (gold_start, pred_start, gold_end) triples.

    A clip whose mistake was never fired contributes eds=0 (missed detection)
    and is excluded from MAE (no predicted start to compare). ``n_loc_detected``
    is surfaced so the MAE denominator is auditable.
    """
    if not events:
        return {"eds": float("nan"), "start_mae": float("nan"),
                "n_loc_events": 0, "n_loc_detected": 0}
    eds_vals = []
    pred_starts, gold_starts = [], []
    for g_start, p_start, g_end in events:
        if p_start is None:
            eds_vals.append(0.0)
            continue
        eds_vals.append(early_detection_score(p_start, g_start, g_end))
        pred_starts.append(p_start)
        gold_starts.append(g_start)
    mae = start_time_mae(pred_starts, gold_starts) if pred_starts else float("nan")
    return {
        "eds": float(sum(eds_vals) / len(eds_vals)),
        "start_mae": float(mae),
        "n_loc_events": len(events),
        "n_loc_detected": len(pred_starts),
    }


def evaluate_records(
    preds_by_clip: Dict[str, List[dict]],
    gold: Dict[str, Optional[Tuple[float, float]]],
    tau: float = 0.5,
    smooth_k: int = 3,
    min_consecutive: int = 2,
    gate_mode: str = "and",
) -> dict:
    """Full decision-layer evaluation over saved per-window records."""
    y_true_bin: List[int] = []
    y_pred_bin: List[int] = []
    y_true_cls: List[str] = []
    y_pred_cls: List[str] = []
    y_score: List[float] = []
    confidences: List[float] = []
    loc_events: List[tuple] = []

    for clip_id, records in preds_by_clip.items():
        g = gate_clip(records, tau, smooth_k, gate_mode)
        y_true_bin.extend(int(r["true_label"]) for r in records)
        y_pred_bin.extend(g["pred_bin"])
        y_true_cls.extend(r["true_class"] for r in records)
        y_pred_cls.extend(g["pred_cls"])
        y_score.extend(g["gated"])
        confidences.extend(float(r["confidence"]) for r in records)

        event = gold.get(clip_id)
        if event is not None and g["starts"]:
            g_start, g_end = event
            ps = detect_start_of_deviation(
                g["smoothed"], g["starts"], FIRED_EPS, min_consecutive
            )
            loc_events.append((g_start, ps, g_end))

    out = compute_all(
        y_true_bin, y_pred_bin, y_true_cls, y_pred_cls,
        y_score, confidences, list(MISTAKE_CLASSES),
    )
    out.update(localization_metrics(loc_events))
    out["auc_raw"] = auc_roc(y_true_bin, confidences)
    out["macro_f1_excl_missing_step"] = macro_f1(
        y_true_cls, y_pred_cls,
        [c for c in MISTAKE_CLASSES if c != "missing_step"],
    )
    out["n_windows"] = len(y_true_bin)
    out["n_clips"] = len(preds_by_clip)
    return out
