"""Tests for the CaptainCook4D loader, class remap, and graph converter.

The remap tests run unconditionally (pure logic). The loader/graph tests need
the cloned annotations under data/raw/captaincook4d/annotations and skip
otherwise, keeping `pytest tests/` runnable with no data per CLAUDE.md.
"""
from pathlib import Path

import pytest

from vmd.data.mistake_classes import (
    CC4D_TAG_TO_CLASS,
    MISTAKE_CLASSES,
    remap_tag,
    step_label,
)

ANN_ROOT = Path("data/raw/captaincook4d/annotations")
_HAS_DATA = (ANN_ROOT / "annotation_json" / "error_annotations.json").exists()
_needs_data = pytest.mark.skipif(not _HAS_DATA, reason="CaptainCook4D annotations not present")


# --- remap (pure logic) -----------------------------------------------------

def test_remap_targets_are_valid_vmd_classes():
    for tag in CC4D_TAG_TO_CLASS:
        cls = remap_tag(tag)
        assert cls is None or cls in MISTAKE_CLASSES


def test_only_other_is_dropped():
    dropped = [t for t, c in CC4D_TAG_TO_CLASS.items() if c is None]
    assert dropped == ["Other"]


def test_unknown_tag_raises():
    with pytest.raises(KeyError):
        remap_tag("Nonexistent Error")


def test_step_label_priority_and_empty():
    assert step_label([]) == "no_mistake"
    # missing_step outranks technique_error when both present
    errs = [{"tag": "Technique Error"}, {"tag": "Missing Step"}]
    assert step_label(errs) == "missing_step"
    # only-Other collapses to no_mistake (sole dropped tag)
    assert step_label([{"tag": "Other"}]) == "no_mistake"
    # Timing + Temperature now first-class, not folded into anything
    assert step_label([{"tag": "Timing Error"}]) == "timing_error"
    assert step_label([{"tag": "Temperature Error"}]) == "temperature_error"


def test_find_video_prefix_ambiguity(tmp_path):
    # ids are prefix-ambiguous: 12_2 must not resolve to 12_26's file
    from vmd.data.captaincook4d import _find_video

    (tmp_path / "12_26_360p.mp4").touch()
    assert _find_video(tmp_path, "12_2") == tmp_path / "12_2.mp4"  # fallback, no match
    (tmp_path / "12_2_360p.mp4").touch()
    assert _find_video(tmp_path, "12_2") == tmp_path / "12_2_360p.mp4"
    (tmp_path / "12_2.mp4").touch()
    assert _find_video(tmp_path, "12_2") == tmp_path / "12_2.mp4"  # exact stem wins


# --- loader (needs annotations) ---------------------------------------------

@_needs_data
def test_loader_split_sizes_and_interface():
    from vmd.data import get_dataset
    from vmd.data.egoops import EgoOopsClip

    sizes = {}
    for sp in ("train", "val", "test"):
        ds = get_dataset("captaincook4d", "data/raw/captaincook4d", sp)
        sizes[sp] = len(ds)
        assert all(isinstance(c, EgoOopsClip) for c in ds.clips)
        for c in ds.clips:
            for a in c.annotations:
                assert a.mistake_class in MISTAKE_CLASSES
    assert sum(sizes.values()) == 384  # combined split total


@_needs_data
def test_loader_task_id_matches_graph_stem():
    from vmd.data.captaincook4d import CaptainCook4DDataset

    ds = CaptainCook4DDataset("data/raw/captaincook4d", "test")
    graph_dir = ANN_ROOT / "task_graphs"
    for c in ds.clips:
        assert (graph_dir / f"{c.task_id}.json").exists(), c.task_id


# --- graph converter (needs annotations) ------------------------------------

@_needs_data
def test_graph_conversion_validates(tmp_path):
    from vmd.data.captaincook4d_graphs import export_cc4d_graphs
    from vmd.graph.schema import ProceduralGraph

    written = export_cc4d_graphs(ANN_ROOT, tmp_path)
    assert len(written) == 24
    for task_id in written:
        g = ProceduralGraph.model_validate_json((tmp_path / f"{task_id}.json").read_text())
        assert len(g.steps) >= 2
