import pytest

from vmd.eval.causal import (Alarm, alarm_episodes, causal_decisions,
                             evaluate_causal, match_one_to_one)
from vmd.eval.event_metrics import GoldEvent
from vmd.eval.provenance import guard_collection, manifest


def records(values):
    return [dict(start_sec=i * 4., end_sec=(i + 1) * 4., confidence=p,
                 true_label=int(i >= 2), true_class="order_error" if i >= 2 else "no_mistake",
                 deviation=p >= .5, inference_sec=1.) for i, p in enumerate(values)]


def test_causal_prefix_invariance_and_persistence_clock():
    rows = records([.1, .9, .9, .9, .1])
    for n in range(1, len(rows) + 1):
        full, _ = causal_decisions(rows, smooth_k=3, min_consecutive=2)
        prefix, _ = causal_decisions(rows[:n], smooth_k=3, min_consecutive=2)
        assert prefix == full[:n]
    alarms, _ = alarm_episodes(rows, 20, smooth_k=1, min_consecutive=2)
    assert alarms[0].emitted_at == 12  # second positive window ends here
    assert alarms[0].evidence_start == 8  # no backdating to window starting at 4
    assert alarms[0].cleared_at == 20


def test_serial_availability_includes_backlog():
    rows = records([.9] * 3)
    for r in rows:
        r["inference_sec"] = 6
    _, available = causal_decisions(rows, clock="serial_inference")
    assert available == [10, 16, 22]


def test_gap_breaks_persistence():
    rows = records([.9, .9])
    rows[1].update(start_sec=12., end_sec=16.)
    decisions, _ = causal_decisions(rows, min_consecutive=2)
    assert decisions == [False, False]


def test_overlapping_events_cannot_reuse_an_alarm():
    alarms = [Alarm(16, 28, 20, 32), Alarm(64, 72, 68, 76), Alarm(100, 108, 104, 112)]
    gold = [GoldEvent(20, 40, "order_error"), GoldEvent(60, 70, "technique_error"),
            GoldEvent(65, 90, "order_error")]
    matches = match_one_to_one(alarms, gold)
    assert sum(k is not None for k in matches) == 2
    assert len({k for k in matches if k is not None}) == 2


def test_maximum_matching_recovers_greedy_miss():
    # Event 0 can take either episode; event 1 can only take episode 0.
    alarms = [Alarm(0, 10, 4, 10), Alarm(0, 5, 2, 5)]
    events = [GoldEvent(0, 10, "x"), GoldEvent(5, 15, "x")]
    assert sum(x is not None for x in match_one_to_one(alarms, events, "iou")) == 2


def test_event_precision_has_same_true_positive_denominator_as_recall():
    rows = records([.1, .9, .1, .9, .1, .9, .1])
    events = [GoldEvent(4, 10, "x"), GoldEvent(12, 18, "x"), GoldEvent(13, 19, "y")]
    result = evaluate_causal({"a": rows}, {"a": events}, {"a": 28})["metrics"]
    assert result["n_detected"] == 2 and result["n_false_alarms"] == 1
    assert result["event_precision"] == pytest.approx(2 / 3)
    assert result["event_recall"] == pytest.approx(2 / 3)
    assert result["event_f1"] == pytest.approx(2 / 3)


def test_early_or_late_alarm_does_not_count_as_online_detection():
    gold = [GoldEvent(10, 20, "x")]
    assert match_one_to_one([Alarm(8, 24, 9, 24)], gold) == [None]
    assert match_one_to_one([Alarm(8, 24, 24, 28)], gold) == [None]
    # Retrospective localization is a separate question.
    assert match_one_to_one([Alarm(8, 24, 24, 28)], gold, "iou") == [0]


def test_always_fire_is_exposed_by_clean_time_burden():
    rows = records([1.] * 30)
    for r in rows:
        r["true_label"] = 0
    m = evaluate_causal({"a": rows}, {"a": []}, {"a": 120})["metrics"]
    assert m["n_alarms"] == 1 and m["event_precision"] == 0
    assert m["clean_time_alarm_fraction"] == pytest.approx(116 / 120)
    assert m["clean_recording_alarm_fraction"] == 1
    assert m["event_recall"] is None


@pytest.mark.parametrize("value", [None, float("nan"), -1, 1.1])
def test_invalid_scores_cannot_silently_enter_metrics(value):
    rows = records([.5])
    rows[0]["confidence"] = value
    with pytest.raises(ValueError, match="Invalid score"):
        causal_decisions(rows)


def test_manifest_prevents_mixed_resume_and_overwrite(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("v1")
    output = tmp_path / "predictions.json"
    first = manifest({"score_version": "v1"}, [source])
    guard_collection(output, first)
    output.with_suffix(".jsonl").write_text("")
    guard_collection(output, first, resume=True)
    with pytest.raises(ValueError, match="Output exists"):
        guard_collection(output, first)
    source.write_text("v2")
    with pytest.raises(ValueError, match="manifest differs"):
        guard_collection(output, manifest({"score_version": "v1"}, [source]), resume=True)


def test_duration_rounding_is_audited_but_large_overruns_rejected():
    result = evaluate_causal({"a": records([.9] * 3)}, {"a": [GoldEvent(4, 12.004, "x")]}, {"a": 12})
    assert result["rounded_event_ends"] == [dict(clip_id="a", original_end=12.004, clipped_end=12)]
    with pytest.raises(ValueError, match="exceeds recording duration"):
        evaluate_causal({"a": records([.9] * 3)}, {"a": [GoldEvent(4, 13, "x")]}, {"a": 12})
