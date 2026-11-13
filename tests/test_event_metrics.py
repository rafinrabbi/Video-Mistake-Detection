"""Hand-computed fixtures for event matching, false alarms, delay, tracking."""
import numpy as np
import pytest

from vmd.eval import event_metrics as em

G = [em.GoldEvent(20.0, 40.0, "order_error"), em.GoldEvent(60.0, 70.0, "technique_error"),
     em.GoldEvent(65.0, 90.0, "order_error")]          # overlaps the previous one


def test_episodes_from_fired_respects_min_consecutive():
    fired = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1]
    starts = [4.0 * i for i in range(10)]; ends = [s + 4 for s in starts]
    assert em.episodes_from_fired(fired, starts, ends, 1) == [(4.0, 12.0), (16.0, 20.0), (28.0, 40.0)]
    assert em.episodes_from_fired(fired, starts, ends, 2) == [(4.0, 12.0), (28.0, 40.0)]


def test_onset_matching_credits_each_event_once_and_flags_early_alarms():
    eps = [(0.0, 8.0),        # early false alarm: starts 20 s before the event
           (16.0, 28.0),      # within 8 s before onset and overlapping -> matches event 0
           (64.0, 72.0)]      # overlaps events 1 and 2, starts within both windows
    assign, matched = em.match_events(eps, G, "onset", delta=8.0)
    assert assign == [1, 2, 2] and matched == [False, True, True]
    st = em.clip_event_stats("c", eps, G, duration_sec=120.0)
    assert st.n_events == 3 and st.n_detected == 3 and st.n_false_alarms == 1
    assert st.delays == pytest.approx([-4.0, 4.0, -1.0])
    assert st.per_class_detected == {"order_error": 2, "technique_error": 1}


def test_iou_matching():
    eps = [(20.0, 30.0)]
    a03, _ = em.match_events(eps, G[:1], "iou", iou_thr=0.3)
    a06, _ = em.match_events(eps, G[:1], "iou", iou_thr=0.6)
    assert a03 == [0] and a06 == [None]


def test_always_fire_scores_zero_recall_under_onset_and_one_fa_per_recording():
    eps = [(0.0, 120.0)]
    st = em.clip_event_stats("c", eps, G, 120.0)
    assert st.n_detected == 0 and st.n_false_alarms == 1
    agg = em.aggregate_event_stats([st])
    assert agg["event_recall"] == 0.0 and agg["false_alarms_per_min"] == pytest.approx(0.5)


def test_aggregate_weights_and_clean_fa_rate():
    a = em.clip_event_stats("a", [(16.0, 28.0)], G[:1], 60.0)
    b = em.clip_event_stats("b", [(0.0, 8.0), (30.0, 40.0)], [], 120.0)   # clean recording
    agg = em.aggregate_event_stats([a, b], weights=[1, 2])
    assert agg["event_recall"] == 1.0
    assert agg["false_alarms_per_min"] == pytest.approx(4 / 5)
    assert agg["clean_false_alarms_per_min"] == pytest.approx(4 / 4)
    assert agg["median_delay_sec"] == pytest.approx(-4.0)


def test_froc_and_recall_at_fa_rate():
    rows = [{"false_alarms_per_min": 2.0, "event_recall": 0.9},
            {"false_alarms_per_min": 0.5, "event_recall": 0.4},
            {"false_alarms_per_min": 1.0, "event_recall": 0.6}]
    fa, er = em.froc_points(rows)
    assert list(fa) == [0.5, 1.0, 2.0]
    assert em.recall_at_fa_rate(fa, er, 1.0) == 0.6 and em.recall_at_fa_rate(fa, er, 0.1) == 0.0


def test_tracking_accuracy_with_neighbourhood():
    succ = [[1], [2], []]
    r = em.tracking_accuracy([0, 1, 2, 2, 0], [0, 2, 2, None, 2], succ)
    assert r["n"] == 4 and r["step_accuracy"] == pytest.approx(0.5)
    assert r["neighbourhood_accuracy"] == pytest.approx(0.75)
    assert em.dedup([0, 0, None, 1, 1, 0]) == [0, 1, 0]
