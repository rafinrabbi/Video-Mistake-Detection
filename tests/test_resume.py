"""Resume must reproduce an uninterrupted run exactly.

The GPU on this host is shared and gets rebooted, so a 7-24 h inference pass is
routinely interrupted. Resume is only useful if restarting is indistinguishable
from never having stopped -- otherwise it silently changes the reported numbers.
These tests pin that equivalence, and pin the two ways it could break: a torn
final line in the sidecar, and gold events going missing for skipped clips.
"""
import json
import math
from types import SimpleNamespace

import pytest

from pathlib import Path as _Path

from vmd.data.egoops import EgoOopsClip, MistakeAnnotation
from vmd.eval import online as online_mod
from vmd.graph.schema import ProceduralGraph, StepNode


class StubVLM:
    """Stand-in whose output depends only on (clip, window), never on call order.

    A stub keyed on a global call counter would produce DIFFERENT predictions on
    resume (the counter restarts), so the metrics would differ for a reason that
    has nothing to do with the resume logic -- masking the very bug under test.
    The window carries its clip id and index, so key on those.
    """

    def __init__(self):
        self.seen = []

    def predict(self, frames, ctx, history):
        cid, i = frames            # injected by the fake sampler below
        self.seen.append(cid)
        n = int(cid[1:]) * 7 + i   # deterministic, order-independent
        return SimpleNamespace(
            observed_action="a", observed_objects=[], step_id="s0",
            deviation=(n % 2 == 0), confidence=0.1 * (n % 10),
            mistake_class="order_error" if n % 2 == 0 else "no_mistake",
            uncertainty_reason="none", rationale="r", expected_vs_observed="d",
            verbalized_confidence=0.9,
        )


def _clip(cid, n_windows=3):
    return EgoOopsClip(
        clip_id=cid, task_id="t", task_name="t", task_description="",
        video_path=_Path(f"/nonexistent/{cid}.mp4"),
        duration_sec=4.0 * n_windows,
        annotations=[MistakeAnnotation(start_sec=4.0, end_sec=8.0,
                                       mistake_class="order_error")],
    )


@pytest.fixture
def patched(monkeypatch):
    """Bypass video decoding: yield synthetic windows with fixed labels."""
    def fake_online_clips(video_path, clip_id, task_id, duration, anns,
                          clip_len, stride, fps, load_frames=True):
        n = int(duration // stride)
        for i in range(n):
            start = i * stride
            yield SimpleNamespace(
                start_sec=start, end_sec=start + clip_len, frames=(clip_id, i),
                true_label=1 if start == 4.0 else 0,
                true_class="order_error" if start == 4.0 else "no_mistake",
            )
    monkeypatch.setattr(online_mod, "online_clips", fake_online_clips)
    monkeypatch.setattr(online_mod, "_num_windows", lambda d, c, s: int(d // s))
    return None


GRAPHS = {"t": ProceduralGraph(
    task_id="t", task_name="t",
    steps=[StepNode(id=f"s{i}", description=f"step {i}") for i in range(4)],
    objects=[], actions=[], states=[], edges=[],
)}


def _same(a, b):
    """Deep equality that treats NaN as equal to itself.

    start_mae is legitimately NaN when no clip fired, and NaN != NaN would make
    an otherwise-identical run compare unequal.
    """
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and \
            all(_same(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def _run(tmp_path, clips, resume, vlm=None):
    vlm = vlm or StubVLM()
    out = online_mod.run_online_eval(
        clips, GRAPHS, vlm,
        save_preds_path=tmp_path / "preds.json", resume=resume,
    )
    return out, vlm


def test_resume_reproduces_uninterrupted_run(patched, tmp_path):
    clips = [_clip(f"c{i}") for i in range(4)]

    full, _ = _run(tmp_path / "a", clips, resume=False)

    # Simulate an interruption: run 2 clips, then resume the remaining 2.
    part_dir = tmp_path / "b"
    _run(part_dir, clips[:2], resume=False)
    resumed, vlm2 = _run(part_dir, clips, resume=True)

    assert set(vlm2.seen) == {"c2", "c3"}, \
        "resume must not re-infer clips already in the sidecar"
    assert _same(resumed["metrics"], full["metrics"])
    assert _same(resumed["coverage"], full["coverage"])


def test_resume_tolerates_a_torn_final_line(patched, tmp_path):
    """A process killed mid-write leaves a partial JSON line; drop only that."""
    clips = [_clip(f"c{i}") for i in range(3)]
    full, _ = _run(tmp_path / "a", clips, resume=False)

    d = tmp_path / "b"
    _run(d, clips[:2], resume=False)
    sidecar = d / "preds.jsonl"
    sidecar.write_text(sidecar.read_text() + '{"clip_id": "c2", "start_sec": 0.0')

    resumed, vlm = _run(d, clips, resume=True)
    assert "c2" in vlm.seen, "a clip with a torn record must be recomputed"
    assert _same(resumed["metrics"], full["metrics"])


def test_skipped_clips_keep_their_gold_events(patched, tmp_path):
    """Localization denominators must not shrink just because a clip was skipped."""
    clips = [_clip(f"c{i}") for i in range(3)]
    full, _ = _run(tmp_path / "a", clips, resume=False)

    d = tmp_path / "b"
    _run(d, clips, resume=False)          # every clip already done
    resumed, vlm = _run(d, clips, resume=True)

    assert vlm.seen == [], "nothing left to infer"
    assert resumed["metrics"]["n_loc_events"] == full["metrics"]["n_loc_events"]
    assert _same(resumed["metrics"], full["metrics"])


def test_resume_recomputes_partial_clip_with_valid_lines(patched, tmp_path):
    clips = [_clip("c0"), _clip("c1")]
    full, _ = _run(tmp_path / "full", clips, resume=False)
    directory = tmp_path / "partial"
    _run(directory, clips, resume=False)
    sidecar = directory / "preds.jsonl"
    lines = sidecar.read_text().splitlines()
    sidecar.write_text("\n".join(lines[:4]) + '\n{"clip_id":')
    resumed, vlm = _run(directory, clips, resume=True)
    assert set(vlm.seen) == {"c1"}
    assert _same(resumed["metrics"], full["metrics"])
    second, vlm2 = _run(directory, clips, resume=True)
    assert vlm2.seen == [] and _same(second["metrics"], full["metrics"])
    assert len(sidecar.read_text().splitlines()) == 6


def test_resume_rejects_legacy_scores_for_corrected_model(patched, tmp_path):
    clips = [_clip("c0")]
    _run(tmp_path, clips, resume=False)
    vlm = StubVLM()
    vlm.score_version = "boolean_pair_v1"
    with pytest.raises(ValueError, match="score versions"):
        _run(tmp_path, clips, resume=True, vlm=vlm)


def test_collection_causal_protocol_replays_identically(patched, tmp_path):
    from vmd.eval.causal import evaluate_causal
    from vmd.eval.decision_layer import load_preds
    from vmd.eval.event_metrics import GoldEvent
    clips = [_clip("c0"), _clip("c1")]
    result = online_mod.run_online_eval(
        clips, GRAPHS, StubVLM(), evaluation_protocol="causal_v1",
        save_preds_path=tmp_path / "preds.json")
    replay = evaluate_causal(load_preds(tmp_path / "preds.json"),
                             {c.clip_id: [GoldEvent(4, 8, "order_error")] for c in clips},
                             {c.clip_id: c.duration_sec for c in clips},
                             smooth_k=3, min_consecutive=2, gate_mode="and")
    assert _same(result["metrics"], replay["metrics"])


class StubScoredVLM(StubVLM):
    """Belief-tracker stand-in: also returns a step distribution per window."""
    prompt_variant = "belief"

    def predict_scored(self, frames, ctx, history, n_steps=0, history_len=3):
        import numpy as np
        cid, i = frames
        pred = self.predict(frames, ctx, history)
        pred.step_letter = "A"
        probs = np.full(n_steps + 1, 0.02)
        probs[(int(cid[1:]) + i) % n_steps] += 1.0
        return pred, probs / probs.sum(), (int(cid[1:]) + i) % n_steps


def test_belief_tracker_resume_and_record_schema(patched, tmp_path):
    clips = [_clip(f"c{i}") for i in range(3)]
    full = online_mod.run_online_eval(clips, GRAPHS, StubScoredVLM(), tracker="belief",
                                      save_preds_path=tmp_path / "a" / "preds.json")
    online_mod.run_online_eval(clips[:1], GRAPHS, StubScoredVLM(), tracker="belief",
                               save_preds_path=tmp_path / "b" / "preds.json")
    resumed = online_mod.run_online_eval(clips, GRAPHS, StubScoredVLM(), tracker="belief",
                                         save_preds_path=tmp_path / "b" / "preds.json", resume=True)
    assert _same(resumed["metrics"], full["metrics"])
    recs = json.loads((tmp_path / "a" / "preds.json").read_text())
    r = recs[0]
    for key in ("task_id", "gold_step_idx", "gold_step_idx_strict", "tracked_step_idx", "step_probs",
                "structural_surprise", "illegal_hold", "belief_entropy", "tracker", "prompt_variant"):
        assert key in r, key
    assert r["tracker"] == "belief" and len(r["step_probs"]) == 5   # 4 steps + transit
    # the frozen decision layer still reads these records
    from vmd.eval.decision_layer import evaluate_records, load_preds
    by = load_preds(tmp_path / "a" / "preds.json")
    assert evaluate_records(by, {c.clip_id: (4.0, 8.0) for c in clips})["n_windows"] == len(recs)
