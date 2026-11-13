"""Span-to-graph matching with repeated step text, and exact pointer replay."""
from vmd.data.egoops import StepSpan
from vmd.data.step_tracking import (
    RateLimitedPointer, gold_span_index_at, graph_index_of_span,
    match_spans_to_graph, replay_pointer,
)
from vmd.graph.schema import StepNode


def _steps(descs):
    return [StepNode(id=str(i), description=d) for i, d in enumerate(descs)]


def test_repeated_descriptions_resolve_in_time_order():
    steps = _steps(["Roll", "Slice", "Slice", "Serve"])
    spans = [StepSpan(step_id=900, start_sec=0, end_sec=10, description="Roll"),
             StepSpan(step_id=901, start_sec=30, end_sec=40, description="Slice"),
             StepSpan(step_id=901, start_sec=10, end_sec=20, description="Slice"),
             StepSpan(step_id=902, start_sec=40, end_sec=50, description="Serve"),
             StepSpan(step_id=903, start_sec=-1, end_sec=-1, description="Slice")]
    m = match_spans_to_graph(spans, steps)
    assert m == {0: 0, 2: 1, 1: 2, 3: 3}           # earlier "Slice" -> first node
    # the legacy matcher collapses both onto the first node
    assert graph_index_of_span(steps, spans[1]) == 1


def test_topological_order_controls_duplicate_assignment():
    steps = _steps(["Slice", "Roll", "Slice"])
    spans = [StepSpan(step_id=1, start_sec=0, end_sec=5, description="Roll"),
             StepSpan(step_id=2, start_sec=5, end_sec=9, description="Slice"),
             StepSpan(step_id=2, start_sec=9, end_sec=12, description="Slice")]
    m = match_spans_to_graph(spans, steps, order=[1, 2, 0])
    assert m == {0: 1, 1: 2, 2: 0}


def test_strict_gold_index_returns_none_in_gaps():
    spans = [StepSpan(step_id=1, start_sec=0, end_sec=10, description="a"),
             StepSpan(step_id=2, start_sec=20, end_sec=30, description="b")]
    assert gold_span_index_at(spans, 15.0) == 0
    assert gold_span_index_at(spans, 15.0, strict=True) is None
    assert gold_span_index_at(spans, 25.0, strict=True) == 1


def test_replay_matches_live_pointers():
    devs = [False, True, False, False, True, False, False, False]
    assert replay_pointer(devs, 3, "per_window") == [0, 1, 1, 2, 2, 2, 2, 2]
    ptr = RateLimitedPointer(4, step_seconds=8.0, stride_sec=4.0, order=[1, 2, 3])
    live = []
    for d in devs:
        live.append(ptr.idx)
        ptr.observe(d)
    assert replay_pointer(devs, 4, "rate_limited", 8.0, 4.0, order=[1, 2, 3]) == live
