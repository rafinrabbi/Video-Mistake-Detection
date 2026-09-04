from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

from vmd.data.egoops import EgoOopsClip
from vmd.data.sampler import online_clips
from vmd.data.step_tracking import (RateLimitedPointer, executable_indices,
                                    load_step_seconds)
from vmd.graph.schema import ProceduralGraph
from vmd.vlm.qwen25vl import Qwen25VL, VLMPrediction
from vmd.eval.decision_layer import evaluate_records
from tqdm.auto import tqdm


def run_online_eval(
    clips: List[EgoOopsClip],
    graphs: Dict[str, ProceduralGraph],
    vlm: Qwen25VL,
    clip_len_sec: float = 4.0,
    stride_sec: float = 4.0,
    fps: float = 1.0,
    smooth_k: int = 3,
    min_consecutive: int = 2,
    fixed_tau: float = 0.5,
    gate_mode: str = "and",
    save_preds_path: Optional[Path] = None,
    log_text: bool = False,
    resume: bool = False,
    step_seconds_path: Optional[Path] = None,
) -> dict:
    """Causal online evaluation: collect per-window VLM predictions, then run
    the shared decision layer (gate → smooth → metrics) over them.

    Collection and decision are strictly separated: the loop here only produces
    raw per-window records; every decision (thresholding, smoothing,
    localization, metrics) happens in ``vmd.eval.decision_layer`` — the same
    code path used when re-gating a saved ``*_preds.json``, so saved
    predictions replay to identical numbers without the model.
    """
    preds_by_clip: Dict[str, List[dict]] = {}
    gold: Dict[str, Optional[tuple]] = {}

    # Coverage accounting — surfaced so reviewers see nothing was silently capped.
    n_clips_total = len(clips)
    n_graph_missing = 0
    n_dropped_frames = 0

    # Per-recipe seconds-per-step, estimated on the TRAINING split only. Used to
    # rate-limit the step pointer; without it the pointer advances once per
    # window and saturates after the first sixth of a recording.
    step_seconds = load_step_seconds(step_seconds_path) if step_seconds_path else {}

    jsonl_path = None
    done_clips: set = set()
    if save_preds_path is not None:
        save_preds_path = Path(save_preds_path)
        save_preds_path.parent.mkdir(parents=True, exist_ok=True)
        # Crash-resilient sidecar: one record per line, flushed after each clip.
        # A clip's records are appended in a single write once the clip finishes,
        # so every clip in this file is COMPLETE -- which is what makes clip-level
        # resume safe. Per-clip state (step pointer, judgement history) is
        # initialised inside the clip loop and never crosses clips, so skipping a
        # finished clip reproduces the unskipped run exactly.
        jsonl_path = save_preds_path.with_suffix(save_preds_path.suffix + "l")
        if resume and jsonl_path.exists():
            for line in jsonl_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # Only the final line can be torn (killed mid-write); drop it.
                    continue
                preds_by_clip.setdefault(rec["clip_id"], []).append(rec)
            done_clips = set(preds_by_clip)
            if done_clips:
                print(f"[resume] {len(done_clips)} clips already complete in "
                      f"{jsonl_path.name} — skipping them", flush=True)
        else:
            jsonl_path.write_text("")

    for clip in tqdm(clips, desc="online clips"):
        graph = graphs.get(clip.task_id)
        if graph is None:
            n_graph_missing += 1
            continue

        # Earliest localizable gold event, for start-of-deviation scoring.
        # Computed before the resume skip: a skipped clip still needs its gold
        # event, or localization metrics would silently lose it.
        loc = [(a.start_sec, a.end_sec) for a in clip.annotations if a.start_sec >= 0]
        gold[clip.clip_id] = min(loc, key=lambda se: se[0]) if loc else None

        if clip.clip_id in done_clips:
            continue

        records: List[dict] = []
        history: List[VLMPrediction] = []
        pointer = RateLimitedPointer(
            n_steps=len(graph.steps),
            step_seconds=step_seconds.get(clip.task_id),
            stride_sec=stride_sec,
            order=executable_indices(graph.steps),
        )

        total_windows = _num_windows(clip.duration_sec, clip_len_sec, stride_sec)
        windows = online_clips(
            clip.video_path, clip.clip_id, clip.task_id,
            clip.duration_sec, clip.annotations,
            clip_len_sec, stride_sec, fps, load_frames=True,
        )

        for window in tqdm(windows, total=total_windows, desc=f"online {clip.clip_id}", leave=False):
            if window.frames is None:
                n_dropped_frames += 1
                continue

            step = graph.steps[min(pointer.idx, len(graph.steps) - 1)]
            next_steps = graph.valid_next_steps(step.id)
            ctx = {
                "task_name": clip.task_name,
                "current_step_id": step.id,
                "current_step_desc": step.description,
                "valid_next_steps": str([s.description for s in next_steps]),
                "required_objects": str(step.required_object_ids),
                "expected_states": str(step.produced_state_ids),
                "valid_variations": str(step.valid_variations),
                # Position-free view of the same plan, used only by the
                # "staticdump" prompt variant (the walked-pointer control).
                "all_steps": "\n".join(
                    f"  {i + 1}. {st.description}"
                    for i, st in enumerate(graph.steps) if not st.is_optional
                ),
            }

            pred = vlm.predict(window.frames, ctx, _summarise_history(history[-3:]))
            history.append(pred)
            record = {
                "clip_id": clip.clip_id,
                "start_sec": window.start_sec,
                "end_sec": window.end_sec,
                "true_label": window.true_label,
                "true_class": window.true_class,
                "deviation": pred.deviation,
                "confidence": pred.confidence,
                "mistake_class": pred.mistake_class,
            }
            if log_text:
                # Off by default: these fields multiply the size of a full-split
                # preds file several-fold and no metric reads them. Turn on for
                # qualitative runs, and for any study of the verbalized
                # confidence, which is otherwise unrecoverable after the run --
                # `confidence` above has already been overwritten with the
                # decision-token probability.
                record.update({
                    "observed_action": pred.observed_action,
                    "observed_objects": pred.observed_objects,
                    "step_id": pred.step_id,
                    "rationale": pred.rationale,
                    "expected_vs_observed": pred.expected_vs_observed,
                    "uncertainty_reason": pred.uncertainty_reason,
                    "verbalized_confidence": pred.verbalized_confidence,
                })
            records.append(record)

            # Graph state machine, rate-limited to the pace a step actually
            # takes (see vmd.data.step_tracking).
            pointer.observe(pred.deviation)

        preds_by_clip[clip.clip_id] = records
        if jsonl_path is not None:
            with jsonl_path.open("a") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

    if save_preds_path is not None:
        all_records = [r for recs in preds_by_clip.values() for r in recs]
        save_preds_path.write_text(json.dumps(all_records, indent=2))

    metrics = evaluate_records(
        preds_by_clip, gold,
        tau=fixed_tau, smooth_k=smooth_k,
        min_consecutive=min_consecutive, gate_mode=gate_mode,
    )
    coverage = {
        "n_clips_total": n_clips_total,
        "n_clips_evaluated": n_clips_total - n_graph_missing,
        "n_graph_missing": n_graph_missing,
        "n_windows": metrics["n_windows"],
        "n_dropped_frames": n_dropped_frames,
        "n_localizable_events": metrics["n_loc_events"],
    }
    return {"metrics": metrics, "coverage": coverage}


def _summarise_history(history: List[VLMPrediction]) -> str:
    if not history:
        return "No prior clips."
    lines = []
    for i, p in enumerate(history):
        lines.append(f"Clip -{len(history)-i}: {p.observed_action} | deviation={p.deviation} | class={p.mistake_class}")
    return "\n".join(lines)


def _num_windows(duration_sec: float, clip_len_sec: float, stride_sec: float) -> int:
    if duration_sec < clip_len_sec:
        return 0
    return int((duration_sec - clip_len_sec) // stride_sec) + 1
