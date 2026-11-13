"""Versioned causal evaluation. Historical decision/event modules stay frozen.

Window labels refer to evidence intervals; alarm delays refer to emission times.
One-to-one maximum-cardinality matching is used for event precision and recall.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from vmd.eval.curves import average_precision, roc_auc
from vmd.eval.event_metrics import GoldEvent, iou

PROTOCOL = "causal_v1"


@dataclass(frozen=True)
class Alarm:
    evidence_start: float
    evidence_end: float
    emitted_at: float
    cleared_at: float


def validate_records(records):
    previous_start = previous_end = -math.inf
    for r in records:
        start, end = float(r["start_sec"]), float(r["end_sec"])
        if not (math.isfinite(start) and math.isfinite(end) and
                0 <= start < end and start > previous_start and end > previous_end):
            raise ValueError("Evidence windows must have finite, strictly ordered bounds")
        p = r.get("confidence")
        if (p is None or not math.isfinite(float(p)) or not 0 <= float(p) <= 1
                or r.get("score_status", "ok") != "ok"):
            raise ValueError("Invalid score: retain the record for diagnosis; do not silently drop it")
        if r["true_label"] not in (0, 1):
            raise ValueError("Expected binary true_label")
        previous_start, previous_end = start, end


def causal_decisions(records, tau=0.5, smooth_k=1, min_consecutive=1,
                     gate_mode="conf_only", clock="window_end"):
    """Trailing median then threshold then persistence, with no backdating.

    Startup uses the available prefix (no padding). Gaps reset smoothing and
    persistence. `serial_inference` adds measured inference time and queueing;
    it excludes decoding/transport and is not end-to-end deployment latency.
    """
    if not 0 <= tau <= 1 or smooth_k < 1 or min_consecutive < 1:
        raise ValueError("Invalid decision configuration")
    if gate_mode not in ("conf_only", "and") or clock not in ("window_end", "serial_inference"):
        raise ValueError("Unknown gate or clock")
    validate_records(records)
    history, decisions, available = [], [], []
    run, previous_end, finish = 0, None, 0.0
    for r in records:
        end = float(r["end_sec"])
        if previous_end is not None and float(r["start_sec"]) > previous_end + 1e-6:
            history, run = [], 0
        p = float(r["confidence"])
        if gate_mode == "and" and not r["deviation"]:
            p = 0.0
        history.append(p)
        fired = float(np.median(history[-smooth_k:])) >= tau
        if gate_mode == "and" and p == 0 and tau == 0:
            fired = False
        run = run + 1 if fired else 0
        decisions.append(run >= min_consecutive)
        if clock == "serial_inference":
            latency = r.get("inference_sec")
            if latency is None or not math.isfinite(float(latency)) or latency < 0:
                raise ValueError("serial_inference requires measured nonnegative inference_sec")
            finish = max(end, finish) + float(latency)
        else:
            finish = end
        available.append(finish)
        previous_end = end
    return decisions, available


def alarm_episodes(records, duration_sec, **config):
    decisions, available = causal_decisions(records, **config)
    alarms = []
    i = 0
    while i < len(records):
        if not decisions[i]:
            i += 1
            continue
        j = i
        while (j + 1 < len(records) and decisions[j + 1] and
               records[j + 1]["start_sec"] <= records[j]["end_sec"] + 1e-6):
            j += 1
        # Localized evidence begins at the first actually emitted decision,
        # never at the pre-persistence candidate window.
        clear = (available[j + 1] if j + 1 < len(records) and
                 records[j + 1]["start_sec"] <= records[j]["end_sec"] + 1e-6
                 else max(available[j], float(records[j]["end_sec"])))
        if j == len(records) - 1:
            clear = max(available[j], duration_sec)
        alarms.append(Alarm(float(records[i]["start_sec"]), float(records[j]["end_sec"]),
                            available[i], clear))
        i = j + 1
    return alarms, decisions


def match_one_to_one(alarms, events, criterion="online", iou_threshold=0.25):
    """Maximum number of disjoint event/episode matches, deterministic ties.

    Events are visited by (end, start, original index); candidates by descending
    IoU then emission time for localization, emission time for online alarms.
    Augmenting paths recover matches a simple greedy assignment would miss.
    Overlapping gold annotations remain separate events; one alarm earns one TP.
    """
    if criterion not in ("online", "iou") or not 0 < iou_threshold <= 1:
        raise ValueError("Invalid matching criterion")
    candidates = []
    for ev in events:
        if not (math.isfinite(ev.start) and math.isfinite(ev.end) and 0 <= ev.start < ev.end):
            raise ValueError("Gold events must have localizable, positive-duration bounds")
        eligible = []
        for k, alarm in enumerate(alarms):
            overlap = min(alarm.evidence_end, ev.end) > max(alarm.evidence_start, ev.start)
            quality = iou((alarm.evidence_start, alarm.evidence_end), (ev.start, ev.end))
            ok = (overlap and ev.start <= alarm.emitted_at <= ev.end) if criterion == "online" else quality >= iou_threshold
            if ok:
                eligible.append(((-quality if criterion == "iou" else 0), alarm.emitted_at, k))
        candidates.append([x[2] for x in sorted(eligible)])
    owner = {}

    def assign(event, seen):
        for episode in candidates[event]:
            if episode in seen:
                continue
            seen.add(episode)
            if episode not in owner or assign(owner[episode], seen):
                owner[episode] = event
                return True
        return False

    for j in sorted(range(len(events)), key=lambda j: (events[j].end, events[j].start, j)):
        assign(j, set())
    assignment = [None] * len(events)
    for episode, event in owner.items():
        assignment[event] = episode
    return assignment


def _union_length(intervals):
    total, right = 0.0, -math.inf
    for a, b in sorted(intervals):
        total += max(0.0, b - max(a, right))
        right = max(right, b)
    return total


def evaluate_causal(preds_by_clip, events_by_clip, durations, *, tau=0.5,
                    smooth_k=1, min_consecutive=1, gate_mode="conf_only",
                    clock="window_end", criterion="online", iou_threshold=0.25):
    if not preds_by_clip or set(preds_by_clip) != set(events_by_clip) or set(preds_by_clip) != set(durations):
        raise ValueError("Predictions, gold events and durations must cover exactly the same recordings")
    config = dict(tau=tau, smooth_k=smooth_k, min_consecutive=min_consecutive,
                  gate_mode=gate_mode, clock=clock)
    y, p, scores, per_clip, delays = [], [], [], {}, []
    n_events = n_tp = n_alarms = 0
    clean_seconds = clean_alarm_seconds = clean_clips = clean_alarm_clips = 0
    duration_total = 0.0
    score_versions = set()
    parse_failures = 0
    rounded_event_ends = []
    for cid, records in preds_by_clip.items():
        duration = float(durations[cid])
        if not records or not math.isfinite(duration) or duration <= 0 or records[-1]["end_sec"] > duration + 1e-6:
            raise ValueError(f"Empty predictions or invalid duration for {cid}")
        alarms, decisions = alarm_episodes(records, duration, **config)
        events = events_by_clip[cid]
        # CC4D metadata rounds duration to centiseconds; annotations retain
        # sub-millisecond values. Permit only this rounding discrepancy.
        if any(ev.end > duration + 0.01 for ev in events):
            raise ValueError(f"Gold event exceeds recording duration: {cid}")
        rounded_event_ends.extend(dict(clip_id=cid, original_end=ev.end, clipped_end=duration)
                                  for ev in events if ev.end > duration)
        events = [GoldEvent(ev.start, min(ev.end, duration), ev.mistake_class) for ev in events]
        matches = match_one_to_one(alarms, events, criterion, iou_threshold)
        clip_delays = [alarms[k].emitted_at - ev.start for ev, k in zip(events, matches) if k is not None]
        tp = len(clip_delays)
        n_tp += tp
        n_events += len(events)
        n_alarms += len(alarms)
        delays.extend(clip_delays)
        duration_total += duration
        if not events:
            clean_clips += 1
            clean_alarm_clips += bool(alarms)
            clean_seconds += duration
            clean_alarm_seconds += _union_length([
                (min(duration, a.emitted_at), min(duration, a.cleared_at)) for a in alarms])
        y.extend(int(r["true_label"]) for r in records)
        scores.extend(float(r["confidence"]) for r in records)
        p.extend(decisions)
        score_versions.update(r.get("score_version", "legacy_unspecified") for r in records)
        parse_failures += sum(not r.get("parse_ok", True) for r in records)
        per_clip[cid] = dict(n_events=len(events), n_detected=tp, n_alarms=len(alarms),
                             n_false_alarms=len(alarms) - tp, matches=matches,
                             delays_sec=clip_delays, alarms=[asdict(a) for a in alarms])
    if len(score_versions) != 1:
        raise ValueError("Mixed score versions must not be pooled")
    yy, pp = np.asarray(y, bool), np.asarray(p, bool)
    tp = int((yy & pp).sum())
    fp = int((~yy & pp).sum())
    fn = int((yy & ~pp).sum())
    metrics = dict(
        n_windows=len(y), n_clips=len(per_clip), n_events=n_events, n_detected=n_tp,
        n_alarms=n_alarms, n_false_alarms=n_alarms - n_tp,
        binary_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else None,
        average_precision=average_precision(y, scores) if yy.any() else None,
        auc_raw=roc_auc(y, scores) if yy.any() and (~yy).any() else None,
        event_precision=n_tp / n_alarms if n_alarms else 0.0,
        event_recall=n_tp / n_events if n_events else None,
        event_f1=2 * n_tp / (n_events + n_alarms) if n_events + n_alarms else 0.0,
        false_alarms_per_min=(n_alarms - n_tp) / (duration_total / 60),
        median_delay_sec=float(np.median(delays)) if delays else None,
        missed_event_fraction=1 - n_tp / n_events if n_events else None,
        clean_time_alarm_fraction=clean_alarm_seconds / clean_seconds if clean_seconds else None,
        clean_recording_alarm_fraction=clean_alarm_clips / clean_clips if clean_clips else None,
        n_parse_failures=parse_failures,
    )
    if criterion == "online":
        for limit in (4, 8, 16):
            metrics[f"event_recall_within_{limit}s"] = sum(0 <= d <= limit for d in delays) / n_events if n_events else None
    return dict(protocol=PROTOCOL, config={**config, "criterion": criterion, "iou_threshold": iou_threshold},
                score_version=next(iter(score_versions)), metrics=metrics, per_clip=per_clip,
                rounded_event_ends=rounded_event_ends,
                clean_definition="No localizable gold events; nonlocalizable omissions are outside this protocol.",
                clock_note=("Window-end availability; excludes computation and transport (latency lower bound)."
                            if clock == "window_end" else
                            "Serial inference availability including queueing; excludes decoding and transport."))
