from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from vmd.data.egoops import EgoOopsClip
from vmd.data.sampler import online_clips, _load_frames
from vmd.data.step_tracking import (RateLimitedPointer, executable_indices,
                                    gold_span_index_at, load_step_seconds,
                                    match_spans_to_graph)
from vmd.graph.schema import ProceduralGraph
from vmd.tracking.belief import (MENU_LETTERS, TRANSIT_LABEL, BeliefFilter, FilterConfig,
                                 StepSpace, build_step_space, one_hot)
from vmd.vlm.qwen25vl import Qwen25VL, VLMPrediction
from vmd.eval.decision_layer import evaluate_records
from tqdm.auto import tqdm

TRACKERS = ("pointer", "belief")


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
    tracker: str = "pointer",
    oracle_step: bool = False,
    filter_overrides: Optional[dict] = None,
    evaluation_protocol: str = "legacy_v1",
    context_sec: float = 0.0,
    context_fps: float = 1.0,
) -> dict:
    """Collect causal VLM predictions and select an explicit evaluation protocol.

    Collection and decision are strictly separated: the loop here only produces
    raw per-window records; every decision (thresholding, smoothing,
    localization, metrics) happens in a replayable evaluator. ``legacy_v1``
    retains the historical API default; the collection CLI explicitly defaults
    to ``causal_v1`` (trailing filtering, one-to-one matching, emission clocks).

    ``tracker="pointer"`` is the legacy rate-limited pointer driving the
    ``graph``/``nograph``/``staticdump`` prompts; ``tracker="belief"`` is the
    DAG-constrained belief filter driving the ``belief`` prompts and reading the
    model's step letter back as an emission. ``oracle_step`` replaces the
    tracked step by the gold step of the window (a diagnostic, never a
    reportable system). Every record carries the tracked and gold step so all
    tracking analysis is a CPU replay.
    """
    if evaluation_protocol not in ("legacy_v1", "causal_v1"):
        raise ValueError("Unknown evaluation protocol")
    if tracker not in TRACKERS:
        raise ValueError(f"tracker {tracker!r} not in {TRACKERS}")
    from vmd.vlm.qwen25vl import STEP_ONLY_VARIANTS
    step_only = getattr(vlm, "prompt_variant", "graph") in STEP_ONLY_VARIANTS
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
    prompt_variant = getattr(vlm, "prompt_variant", "graph")

    jsonl_path = None
    done_clips: set = set()
    if save_preds_path is not None:
        save_preds_path = Path(save_preds_path)
        save_preds_path.parent.mkdir(parents=True, exist_ok=True)
        # Crash-resilient sidecar: one record per line, flushed after each clip.
        # A clip's records are appended in a single write once the clip finishes,
        # so every clip in this file is COMPLETE -- which is what makes clip-level
        # resume safe. Per-clip state (step pointer / belief filter, judgement
        # history) is initialised inside the clip loop and never crosses clips,
        # so skipping a finished clip reproduces the unskipped run exactly.
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
            expected_score = getattr(vlm, "score_version", None)
            if expected_score and any(r.get("score_version") != expected_score
                                      for rows in preds_by_clip.values() for r in rows):
                raise ValueError("Cannot resume predictions with different or missing score versions")
            # A write can be interrupted after several complete lines of a clip.
            # Resume only complete, ordered window grids; recompute partial clips.
            clip_lookup = {c.clip_id: c for c in clips}
            if set(preds_by_clip) - set(clip_lookup):
                raise ValueError("Resume sidecar contains recordings outside this run")
            preds_by_clip = {
                cid: rows for cid, rows in preds_by_clip.items()
                if len(rows) == _num_windows(clip_lookup[cid].duration_sec, clip_len_sec, stride_sec)
                and all(abs(r["start_sec"] - i * stride_sec) < 1e-6 and
                        abs(r["end_sec"] - (i * stride_sec + clip_len_sec)) < 1e-6
                        for i, r in enumerate(rows))
            }
            # Remove torn/partial data before appending, so a second resume is safe.
            jsonl_path.write_text("".join(json.dumps(r) + "\n"
                                        for rows in preds_by_clip.values() for r in rows))
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
        space = build_step_space(graph)
        span_to_pos = match_spans_to_graph(clip.step_spans, graph.steps, order=space.graph_pos)

        def gold_menu_index(t_sec: float, strict: bool = False) -> Optional[int]:
            i = gold_span_index_at(clip.step_spans, t_sec, strict=strict)
            if i is None or i not in span_to_pos:
                return None
            return space.menu_index_of_graph_pos(span_to_pos[i])

        pointer = RateLimitedPointer(
            n_steps=len(graph.steps),
            step_seconds=step_seconds.get(clip.task_id),
            stride_sec=stride_sec,
            order=executable_indices(graph.steps),
        )
        filt = None
        if tracker == "belief":
            cfg = FilterConfig.from_step_seconds(step_seconds.get(clip.task_id), stride_sec,
                                                 **(filter_overrides or {}))
            filt = BeliefFilter(space, cfg)
        state = None

        total_windows = _num_windows(clip.duration_sec, clip_len_sec, stride_sec)
        windows = online_clips(
            clip.video_path, clip.clip_id, clip.task_id,
            clip.duration_sec, clip.annotations,
            clip_len_sec, stride_sec, fps, load_frames=True,
        )

        for w_idx, window in enumerate(tqdm(windows, total=total_windows,
                                            desc=f"online {clip.clip_id}", leave=False)):
            if window.frames is None:
                n_dropped_frames += 1
                continue

            mid = (window.start_sec + window.end_sec) / 2.0
            gold_idx = gold_menu_index(mid)                    # gaps -> previous step (historical)
            gold_strict = gold_menu_index(mid, strict=True)    # gaps -> None (transit)

            # --- which step is the model told it is on -------------------------
            if tracker == "belief":
                tracked_idx = filt.argmax if state is None else state.argmax
                tracked_p = float(filt.belief[tracked_idx])
                completed = [] if state is None else state.completed
                ready_next = space.roots if state is None else state.ready_next
            else:
                tracked_idx = space.menu_index_of_graph_pos(pointer.idx)
                tracked_p = None
                completed, ready_next = [], []
            if oracle_step:
                tracked_idx = gold_idx if gold_idx is not None else space.n   # transit when no gold step
                tracked_p = 1.0

            ctx = _build_ctx(clip, graph, space, tracked_idx, tracked_p, completed, ready_next)
            ctx["context_sec"] = int(context_sec) if context_sec else int(clip_len_sec)
            # Longer causal look-back for the step recogniser: frames from the
            # last ``context_sec`` seconds ending at this window's end (clipped at
            # the recording start), sampled at ``context_fps``. Never the future.
            frames = window.frames
            if context_sec and context_sec > clip_len_sec:
                frames = _load_frames(clip.video_path, max(0.0, window.end_sec - context_sec),
                                      window.end_sec, context_fps)

            inference_start = time.perf_counter()
            if tracker == "belief":
                pred, probs, emitted = vlm.predict_scored(
                    frames, ctx, _summarise_history(history[-3:]), n_steps=space.n)
                if probs is None:
                    # unparseable step: uninformative emission, the prior carries the state
                    probs = one_hot(None, space.n, 0.0)
                state = filt.update(probs)
            else:
                pred = vlm.predict(frames, ctx, _summarise_history(history[-3:]))
                probs, emitted = None, None
            inference_sec = time.perf_counter() - inference_start
            history.append(pred)
            if step_only:
                # The recogniser emits no deviation field; keep the record loadable
                # by the frozen decision layer and mark the content score absent.
                pred.deviation = False
                pred.confidence = 0.0
                pred.score_status = "step_only"

            record = {
                "clip_id": clip.clip_id,
                "start_sec": window.start_sec,
                "end_sec": window.end_sec,
                "true_label": window.true_label,
                "true_class": window.true_class,
                "deviation": pred.deviation,
                "confidence": pred.confidence,
                "score_version": getattr(pred, "score_version", "legacy_unspecified"),
                "score_status": getattr(pred, "score_status", "ok"),
                "parse_ok": getattr(pred, "parse_ok", True),
                "inference_sec": inference_sec,
                "mistake_class": pred.mistake_class,
                # GraphTrack provenance and tracking fields (optional for old readers)
                "task_id": clip.task_id,
                "window_idx": w_idx,
                "gold_step_idx": -1 if gold_idx is None else int(gold_idx),
                "gold_step_idx_strict": -1 if gold_strict is None else int(gold_strict),
                "tracked_step_idx": -1 if tracked_idx is None else int(tracked_idx),
                "prompt_variant": prompt_variant,
                "tracker": tracker,
                "oracle_step": bool(oracle_step),
            }
            if tracker == "belief":
                record.update({
                    "tracked_p": tracked_p,
                    "step_pred_idx": -1 if emitted is None else int(emitted),
                    "step_probs": [round(float(x), 5) for x in probs],
                    "belief_argmax": int(state.argmax),
                    "belief_entropy": round(state.entropy, 5),
                    "structural_surprise": round(state.surprise, 6),
                    "illegal_hold": round(state.illegal_hold, 6),
                    "violation_mass": round(state.violation_mass, 6),
                })
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

            # Legacy graph state machine, rate-limited to the pace a step actually
            # takes (see vmd.data.step_tracking). Advanced in both modes so the
            # pointer replay stays comparable across arms.
            pointer.observe(pred.deviation)

        preds_by_clip[clip.clip_id] = records
        if jsonl_path is not None:
            with jsonl_path.open("a") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

    if save_preds_path is not None:
        all_records = [r for recs in preds_by_clip.values() for r in recs]
        save_preds_path.write_text(json.dumps(all_records, indent=2))

    causal_result = None
    if evaluation_protocol == "causal_v1":
        from vmd.eval.causal import evaluate_causal
        from vmd.eval.event_metrics import GoldEvent
        selected = {c.clip_id: c for c in clips if c.clip_id in preds_by_clip}
        all_events = {cid: [GoldEvent(a.start_sec, a.end_sec, a.mistake_class)
                            for a in c.annotations if a.start_sec >= 0 and a.end_sec > a.start_sec]
                      for cid, c in selected.items()}
        causal_result = evaluate_causal(
            preds_by_clip, all_events, {cid: c.duration_sec for cid, c in selected.items()},
            tau=fixed_tau, smooth_k=smooth_k, min_consecutive=min_consecutive, gate_mode=gate_mode)
        metrics = causal_result["metrics"]
    else:
        metrics = evaluate_records(
            preds_by_clip, gold, tau=fixed_tau, smooth_k=smooth_k,
            min_consecutive=min_consecutive, gate_mode=gate_mode,
        )
    coverage = {
        "n_clips_total": n_clips_total,
        "n_clips_evaluated": len(preds_by_clip),
        "n_graph_missing": n_graph_missing,
        "n_windows": metrics["n_windows"],
        "n_dropped_frames": n_dropped_frames,
        "n_localizable_events": metrics.get("n_events", metrics.get("n_loc_events", 0)),
    }
    result = {"metrics": metrics, "coverage": coverage, "protocol": evaluation_protocol}
    if causal_result:
        result["causal_evaluation"] = causal_result
    return result


def _letter(idx: Optional[int], n: int) -> str:
    if idx is None or idx < 0:
        return "?"
    return TRANSIT_LABEL if idx >= n else MENU_LETTERS[idx]


def _build_ctx(clip: EgoOopsClip, graph: ProceduralGraph, space: StepSpace,
               tracked_idx: Optional[int], tracked_p: Optional[float],
               completed: List[int], ready_next: List[int]) -> dict:
    """Prompt context for every template family from one tracked menu index.

    Legacy keys (``current_step_*``, ``valid_next_steps``...) are filled from the
    same step so ``reason.txt`` can be driven by either tracker; belief keys
    are ignored by the legacy templates.
    """
    from vmd.vlm.step_scoring import build_menu
    n = space.n
    if tracked_idx is None or tracked_idx < 0 or tracked_idx >= n:
        step = None
        desc = "between steps / no recipe step in progress"
    else:
        step = graph.steps[space.graph_pos[tracked_idx]]
        desc = step.description
    next_steps = graph.valid_next_steps(step.id) if step is not None else \
        [graph.steps[space.graph_pos[r]] for r in space.roots]
    return {
        "task_name": clip.task_name,
        "current_step_id": step.id if step is not None else "none",
        "current_step_desc": desc,
        "valid_next_steps": str([s.description for s in next_steps]),
        "required_objects": str(step.required_object_ids if step is not None else []),
        "expected_states": str(step.produced_state_ids if step is not None else []),
        "valid_variations": str(step.valid_variations if step is not None else []),
        # Position-free view of the same plan, used only by the "staticdump"
        # prompt variant (the walked-pointer control).
        "all_steps": "\n".join(
            f"  {i + 1}. {st.description}"
            for i, st in enumerate(graph.steps) if not st.is_optional
        ),
        # GraphTrack ("belief") keys
        "step_menu": build_menu(space.descriptions),
        "tracked_letter": _letter(tracked_idx, n),
        "tracked_desc": desc,
        "tracked_p": "?" if tracked_p is None else f"{tracked_p:.2f}",
        "completed_letters": ", ".join(MENU_LETTERS[i] for i in completed) if completed else "none",
        "next_letters": ", ".join(MENU_LETTERS[i] for i in ready_next) if ready_next else "none",
    }


def _summarise_history(history: List[VLMPrediction]) -> str:
    if not history:
        return "No prior clips."
    lines = []
    for i, p in enumerate(history):
        step = getattr(p, "step_letter", None)
        head = f"step={step}" if step else (p.observed_action or "")
        lines.append(f"Clip -{len(history)-i}: {head} | deviation={p.deviation} | class={p.mistake_class}")
    return "\n".join(lines)


def _num_windows(duration_sec: float, clip_len_sec: float, stride_sec: float) -> int:
    if duration_sec < clip_len_sec:
        return 0
    return int((duration_sec - clip_len_sec) // stride_sec) + 1
