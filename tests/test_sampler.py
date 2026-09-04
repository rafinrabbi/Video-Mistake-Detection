from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from vmd.data.sampler import online_clips, offline_clips
from vmd.vlm.qwen25vl import _build_prompt


def _dummy_annotations():
    from vmd.data.egoops import MistakeAnnotation
    return [MistakeAnnotation(start_sec=5.0, end_sec=10.0, mistake_class="order_error")]


def _make_frames(n=4):
    return np.zeros((n, 224, 224, 3), dtype=np.uint8)


def test_online_clips_causal_order():
    with patch("vmd.data.sampler._DECORD_AVAILABLE", False):
        clips = list(online_clips(
            video_path=Path("dummy.mp4"),
            clip_id="c1",
            task_id="t1",
            duration_sec=20.0,
            annotations=_dummy_annotations(),
            clip_len_sec=4.0,
            stride_sec=2.0,
            fps=1.0,
            load_frames=False,
        ))
    assert len(clips) > 0
    starts = [c.start_sec for c in clips]
    assert starts == sorted(starts), "Clips not in causal order"


def test_online_clips_no_future_access():
    with patch("vmd.data.sampler._DECORD_AVAILABLE", False):
        clips = list(online_clips(
            video_path=Path("dummy.mp4"),
            clip_id="c1",
            task_id="t1",
            duration_sec=20.0,
            annotations=[],
            clip_len_sec=4.0,
            stride_sec=2.0,
            fps=1.0,
            load_frames=False,
        ))
    for c in clips:
        assert c.end_sec <= 20.0


def test_online_clips_labels():
    with patch("vmd.data.sampler._DECORD_AVAILABLE", False):
        clips = list(online_clips(
            video_path=Path("dummy.mp4"),
            clip_id="c1",
            task_id="t1",
            duration_sec=20.0,
            annotations=_dummy_annotations(),
            clip_len_sec=4.0,
            stride_sec=2.0,
            fps=1.0,
            load_frames=False,
        ))
    mistake_clips = [c for c in clips if c.true_label == 1]
    assert len(mistake_clips) > 0


def test_offline_clips_count_matches_online():
    with patch("vmd.data.sampler._DECORD_AVAILABLE", False):
        on = list(online_clips(Path("d.mp4"), "c", "t", 20.0, [], 4.0, 2.0, 1.0, False))
        off = offline_clips(Path("d.mp4"), "c", "t", 20.0, [], 4.0, 2.0, 1.0, False)
    assert len(on) == len(off)


def test_vlm_prompt_renders_literal_json_braces():
    prompt = _build_prompt(
        {
            "task_name": "Task",
            "current_step_id": "step_01",
            "current_step_desc": "Do the thing",
        },
        "No prior clips.",
        3,
    )
    assert "TASK: Task" in prompt
    assert "Current expected step: step_01" in prompt
    assert '"observed_action": "..."' in prompt


def test_nograph_prompt_strips_graph_context():
    ctx = {
        "task_name": "Task",
        "current_step_id": "step_01",
        "current_step_desc": "Do the thing",
    }
    prompt = _build_prompt(ctx, "No prior clips.", 3, variant="nograph")
    # graph block gone, everything else intact — same JSON schema
    assert "PROCEDURAL GRAPH CONTEXT" not in prompt
    assert "step_01" not in prompt
    assert "Do the thing" not in prompt
    assert "TASK: Task" in prompt
    assert "RECENT HISTORY" in prompt
    assert '"observed_action": "..."' in prompt
    assert '"mistake_class"' in prompt
