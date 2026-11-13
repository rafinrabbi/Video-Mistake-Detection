"""Training-time tracked-state corruption: target stays gold, context varies."""
import random

import pytest

from vmd.graph.schema import Edge, EdgeKind, ProceduralGraph, StepNode
from vmd.tracking.belief import MENU_LETTERS, build_step_space
from vmd.train.lora_train import belief_context


def _space():
    ids = ["START", "a", "b", "c", "d", "e", "END"]
    steps = [StepNode(id=i, description=f"do {i}", is_optional=i in ("START", "END")) for i in ids]
    edges = [("START", "a"), ("a", "b"), ("b", "c"), ("b", "d"), ("c", "e"), ("d", "e"), ("e", "END")]
    return build_step_space(ProceduralGraph(task_id="t", task_name="t", steps=steps, objects=[], actions=[],
                                            states=[], edges=[Edge(src=s, dst=d, kind=EdgeKind.after) for s, d in edges]))


def test_gold_context_is_faithful():
    sp = _space()
    ctx = belief_context(sp, gold_idx=2, completed=[0, 1], rng=random.Random(0), corrupt_rate=0.0, prev_gold=1)
    assert ctx["tracked_letter"] == "C" and ctx["tracked_p"] == "0.90"
    assert ctx["completed_letters"] in ("A, B", "A, B, C", "A", "B")   # 20% completed-set jitter is allowed
    assert "C" not in ctx["next_letters"]


def test_corruption_rate_is_realised_and_transit_handled():
    sp = _space()
    rng = random.Random(1)
    shown = [belief_context(sp, 2, [0, 1], rng, 0.5, 1)["tracked_letter"] for _ in range(2000)]
    wrong = sum(l != "C" for l in shown) / len(shown)
    assert 0.42 < wrong < 0.58
    assert all(l in set(MENU_LETTERS[:5]) | {"-"} for l in shown)
    ctx = belief_context(sp, None, [0, 1, 2], rng, 0.0, 2)
    assert ctx["tracked_letter"] == "-" and "between steps" in ctx["tracked_desc"]


def test_next_letters_respect_dag_readiness():
    sp = _space()
    ctx = belief_context(sp, 2, [0, 1], random.Random(3), 0.0, 1)
    # completed {a,b} (possibly jittered); e needs both c and d, so e is never "next"
    assert "E" not in ctx["next_letters"]


def test_step_only_variant_builds_menu_only_context_and_letter_target(monkeypatch):
    """The recogniser sees the menu and a look-back note, never a tracked state; its
    target is the gold letter alone, and all windows are kept (no mistake balancing)."""
    import json
    from types import SimpleNamespace
    from vmd.train import lora_train as lt
    from vmd.data.egoops import EgoOopsClip, StepSpan

    from vmd.graph.schema import Edge, EdgeKind, ProceduralGraph, StepNode
    steps = [StepNode(id=i, description=f"do {i}", is_optional=i in ("START", "END")) for i in ("START", "a", "b", "END")]
    graph = ProceduralGraph(task_id="t", task_name="T", steps=steps, objects=[], actions=[], states=[],
                            edges=[Edge(src=s, dst=d, kind=EdgeKind.after) for s, d in (("START", "a"), ("a", "b"), ("b", "END"))])
    clip = EgoOopsClip(clip_id="c", task_id="t", task_name="T", task_description="", video_path=None, duration_sec=16.0,
                       annotations=[], step_spans=[StepSpan(1, 0.0, 8.0, "do a"), StepSpan(2, 8.0, 16.0, "do b")])

    def fake_windows(video_path, clip_id, task_id, duration, anns, clip_len, stride, fps, load_frames=False):
        for i in range(int(duration // stride)):
            yield SimpleNamespace(start_sec=i * stride, end_sec=i * stride + clip_len, true_label=0, true_class="no_mistake")
    monkeypatch.setattr(lt, "online_clips", fake_windows)
    ds = lt.MistakeDataset(SimpleNamespace(clips=[clip]), {"t": graph}, stride_sec=4.0, prompt_variant="stepreco",
                           neg_per_pos=0, max_windows=0, context_sec=16, context_fps=0.5)
    assert len(ds.samples) == 4
    letters = sorted(json.loads(s["target"])["step"] for s in ds.samples)
    assert letters == ["A", "A", "B", "B"]
    ctx = ds.samples[0]["ctx"]
    assert set(ctx) == {"task_name", "step_menu", "context_sec"} and ctx["context_sec"] == "16"
